from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
    LanguageElectronicReplacement,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
)


class RobustCCDNormalizer(nn.Module):
    """Remove global CCD dark level and gain before the canonical MoE readout."""

    def __init__(self, settings: Any) -> None:
        super().__init__()
        self.active_size = int(settings.active_size)
        self.background_quantile = float(
            settings.language_optical_background_quantile
        )
        self.relative_clip = float(settings.language_optical_normalization_clip)
        self.log_compression = float(settings.language_optical_log_compression)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        expected = (self.active_size, self.active_size)
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != expected:
            raise ValueError(
                f"MoE4 CCD intensity must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(intensity.shape)}"
            )
        value = intensity.float().clamp_min(0.0)
        if not torch.isfinite(value).all():
            raise RuntimeError("CCD intensity contains NaN or Inf")
        background = torch.quantile(
            value.flatten(1), self.background_quantile, dim=1
        ).detach()[:, None, None]
        value = (value - background).clamp_min(0.0)
        frame_mean = value.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        relative = (value / frame_mean).clamp_max(self.relative_clip)
        return torch.log1p(self.log_compression * relative)


class MoE4LanguageTwoBlockOpticalPath(nn.Module):
    """Two-block MoE4 path: experts in Block 1, global phase in Block 2."""

    def __init__(self, width: int, settings: Any) -> None:
        super().__init__()
        optical_settings = copy.copy(settings)
        # The optical detector always maps 478 CCD pixels to 224 token rows.
        # The outer electronic retrieval head remains mean/max 384 -> 64.
        optical_settings.detector_output_size = int(settings.input_adapter_dim)
        self.core = HomogeneousMoEOpticalCore(
            int(width), settings.max_language_tokens, optical_settings
        )
        self.width = int(width)
        self.active_size = int(settings.active_size)
        self.input_rms = float(settings.language_optical_input_rms)
        self.target_mean = float(settings.language_optical_ccd_target_mean)
        self.input_shift_pixels = int(settings.language_optical_max_shift_pixels)
        self.global_shift_pixels = int(settings.language_optical_phase_shift_pixels)
        self.ccd_shift_pixels = int(settings.language_optical_ccd_shift_pixels)
        self.gain_min = float(settings.language_optical_gain_min)
        self.gain_max = float(settings.language_optical_gain_max)
        self.offset_fraction = float(settings.language_optical_offset_fraction)
        self.read_noise_fraction = float(
            settings.language_optical_read_noise_fraction
        )
        self.ccd_normalizer = RobustCCDNormalizer(settings)
        self.current_operating_loss: torch.Tensor | None = None
        self.last_global_input_amplitude: torch.Tensor | None = None
        self.last_raw_ccd: torch.Tensor | None = None
        self.last_normalized_ccd: torch.Tensor | None = None

    @staticmethod
    def _translate_zero(value: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
        shifted = torch.roll(value, (shift_y, shift_x), dims=(-2, -1))
        if shift_y > 0:
            shifted[..., :shift_y, :] = 0
        elif shift_y < 0:
            shifted[..., shift_y:, :] = 0
        if shift_x > 0:
            shifted[..., :, :shift_x] = 0
        elif shift_x < 0:
            shifted[..., :, shift_x:] = 0
        return shifted

    def _random_shift(self, value: torch.Tensor, maximum: int) -> torch.Tensor:
        if not self.training or maximum <= 0:
            return value
        shift_y = int(torch.randint(-maximum, maximum + 1, ()).item())
        shift_x = int(torch.randint(-maximum, maximum + 1, ()).item())
        return self._translate_zero(value, shift_y, shift_x)

    def _perturb_ccd(self, intensity: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return intensity
        intensity = self._random_shift(intensity, self.ccd_shift_pixels)
        batch = intensity.shape[0]
        gain = torch.empty(batch, 1, 1, device=intensity.device).uniform_(
            self.gain_min, self.gain_max
        )
        reference = intensity.mean(dim=(-2, -1), keepdim=True).detach()
        offset = torch.empty_like(gain).uniform_(0.0, self.offset_fraction) * reference
        noise = torch.randn_like(intensity) * self.read_noise_fraction * reference
        return (gain * intensity + offset + noise).clamp_min(0.0)

    def _readout_delta(
        self,
        detector_intensity: torch.Tensor,
        lengths: list[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        normalized_intensity = self.ccd_normalizer(detector_intensity)
        readout, _ = self.core.readout.forward_intensity(normalized_intensity)
        self.core.current_detector_readout = readout
        self.last_normalized_ccd = normalized_intensity.detach()
        packed = torch.cat(
            [readout[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        return self.core.output_adapter(packed).to(dtype)

    def run_expert_block(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        """Run the router and 2x2 expert plane assigned to Language Block 1."""
        lengths = [int((~row).sum()) for row in padding_mask]
        groups = [latent[index, :length] for index, length in enumerate(lengths)]
        input_fields = self.core.encode_groups(groups)
        rms = input_fields.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        input_fields = input_fields * (self.input_rms / rms)
        field, routing = self.core.begin(input_fields)
        field = self._random_shift(field, self.input_shift_pixels)
        field = self.core.run_stage(0, field, routing)
        return field, lengths

    def run_global_block(
        self,
        field: torch.Tensor,
        lengths: list[int],
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the global phase and final CCD assigned to Language Block 2."""
        field = self._random_shift(field, self.global_shift_pixels)
        active = self.core.geometry.active_aperture
        self.last_global_input_amplitude = field[
            :, active.y0 : active.y1, active.x0 : active.x1
        ].abs().detach()
        detector_field = self.core.propagator(self.core.global_phase(field))
        raw_ccd = detector_field[
            :, active.y0 : active.y1, active.x0 : active.x1
        ].abs().square().float()
        clean_mean = raw_ccd.mean(dim=(-2, -1)).clamp_min(1.0e-8)
        target = raw_ccd.new_tensor(self.target_mean).clamp_min(1.0e-8)
        self.current_operating_loss = F.smooth_l1_loss(
            clean_mean.log(), target.log().expand_as(clean_mean)
        )
        self.last_raw_ccd = raw_ccd.detach()
        packed_delta = self._readout_delta(
            self._perturb_ccd(raw_ccd), lengths, dtype
        )
        delta = torch.zeros(
            padding_mask.shape[0], padding_mask.shape[1], self.width,
            device=field.device, dtype=dtype,
        )
        delta[~padding_mask] = packed_delta
        return delta


    def forward(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        field, lengths = self.run_expert_block(latent, padding_mask)
        return self.run_global_block(
            field, lengths, padding_mask, latent.dtype
        )

    def decode_measured_ccd(
        self,
        detector_intensity: torch.Tensor,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lengths = [int((~row).sum()) for row in padding_mask]
        packed = self._readout_delta(detector_intensity, lengths, dtype)
        delta = torch.zeros(
            len(lengths), padding_mask.shape[1], self.width,
            device=detector_intensity.device, dtype=dtype,
        )
        delta[~padding_mask] = packed
        return delta

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.set_phase_dropout_active(active)


class LanguageTwoBlockOpticalCore(ElectronicSequenceCore):
    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(
            hidden_size,
            max_tokens,
            settings,
            settings.electronic_language_token_mixer_type,
            settings.electronic_language_token_mixer_kernel_size,
        )
        if len(self.blocks) != 2:
            raise ValueError("Two-slot Language optics requires two electronic mixer blocks")
        # DeepStackMultimodalReplacement uses this length to occupy consecutive
        # Qwen language slots.  These are placement markers only; the physical
        # phases live in optical_branch.core.
        self.expert_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.optical_branch = MoE4LanguageTwoBlockOpticalPath(self.width, settings)
        self.optical_fusion_logit = nn.Parameter(
            torch.logit(torch.tensor(float(settings.optical_fusion_initial)))
        )
        self.last_block2_input_groups: list[torch.Tensor] = []
        self.last_electronic_block2_groups: list[torch.Tensor] = []
        self._stage1_field: torch.Tensor | None = None
        self._stage1_latent: torch.Tensor | None = None
        self._stage1_lengths: list[int] = []
        self._stage1_padding_mask: torch.Tensor | None = None

    @property
    def optical_fusion(self) -> torch.Tensor:
        return torch.sigmoid(self.optical_fusion_logit)

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        optical = self.optical_branch.core.parameter_breakdown()
        report["implementation"] = "moe4_expert_block1_global_block2_residual"
        report["optical_parameters"] = optical["total_parameters"]
        report["optical_phase_parameters"] = optical["optical_phase_parameters"]
        report["router_parameters"] = optical["router_parameters"]
        report["optical_fusion_parameters"] = self.optical_fusion_logit.numel()
        return report

    def _pad_groups(
        self, groups: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not groups or any(group.ndim != 2 for group in groups):
            raise ValueError("Language core expects a non-empty list of [T,D] groups")
        lengths = [len(group) for group in groups]
        if any(length <= 0 or length > self.max_tokens for length in lengths):
            raise RuntimeError(f"Invalid Language token lengths {lengths}")
        max_length = max(lengths)
        padded = groups[0].new_zeros(len(groups), max_length, self.hidden_size)
        padding_mask = torch.ones(
            len(groups), max_length, dtype=torch.bool, device=groups[0].device
        )
        for index, group in enumerate(groups):
            padded[index, : len(group)] = group
            padding_mask[index, : len(group)] = False
        return padded, padding_mask, lengths

    @staticmethod
    def _pack(
        value: torch.Tensor, lengths: list[int]
    ) -> torch.Tensor:
        return torch.cat(
            [value[index, :length] for index, length in enumerate(lengths)], dim=0
        )

    def forward_stage_groups(
        self,
        stage: int,
        groups: list[torch.Tensor],
        *,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not causal:
            raise RuntimeError("Language optical core expects causal token mixing")
        padded, padding_mask, lengths = self._pad_groups(groups)
        gate = torch.sigmoid(self.residual_logit)
        if stage == 0:
            input_latent = self.input_norm(self.input_adapter(padded.float()))
            optical_field, optical_lengths = self.optical_branch.run_expert_block(
                input_latent, padding_mask
            )
            block1_latent = self.blocks[0](
                input_latent, padding_mask=padding_mask, causal=True
            )
            self._stage1_field = optical_field
            self._stage1_latent = block1_latent
            self._stage1_lengths = optical_lengths
            self._stage1_padding_mask = padding_mask
            stage1_output = padded.float() + gate * self.output_adapter(block1_latent)
            stage1_output = stage1_output.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            ).to(groups[0].dtype)
            return self._pack(stage1_output, lengths), block1_latent
        if stage != 1:
            raise RuntimeError("Language MoE4 replacement has exactly stages 0 and 1")
        if (
            self._stage1_field is None
            or self._stage1_latent is None
            or self._stage1_padding_mask is None
            or lengths != self._stage1_lengths
        ):
            raise RuntimeError("Language Block 1 expert stage must run before Block 2 global")
        if tuple(padding_mask.shape) != tuple(self._stage1_padding_mask.shape):
            raise RuntimeError("Language token layout changed between optical blocks")
        block2_input = self._stage1_latent
        electronic = self.blocks[1](
            block2_input, padding_mask=padding_mask, causal=True
        )
        optical_delta = self.optical_branch.run_global_block(
            self._stage1_field,
            self._stage1_lengths,
            padding_mask,
            block2_input.dtype,
        )
        latent = self.output_norm(
            electronic + self.optical_fusion * optical_delta
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output = padded.float() + gate * self.output_adapter(latent)
        output = output.to(groups[0].dtype)
        self.last_latent_groups = [
            latent[index, :length] for index, length in enumerate(lengths)
        ]
        self.last_block2_input_groups = [
            block2_input[index, :length].detach() for index, length in enumerate(lengths)
        ]
        self.last_electronic_block2_groups = [
            electronic[index, :length].detach() for index, length in enumerate(lengths)
        ]
        self.last_routing = self.optical_branch.core.last_routing
        return self._pack(output, lengths), latent

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        *,
        causal: bool,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience path used by unit tests; production uses two Qwen slots."""
        packed, _ = self.forward_stage_groups(0, groups, causal=causal)
        lengths = [len(group) for group in groups]
        stage1_groups = list(packed.split(lengths))
        return self.forward_stage_groups(1, stage1_groups, causal=causal)

    def detector_features_from_cached(
        self, electronic: torch.Tensor, ccd: torch.Tensor
    ) -> torch.Tensor:
        if electronic.ndim != 2 or electronic.shape[-1] != self.width:
            raise ValueError("Cached electronic block-2 output must be [L,width]")
        length = electronic.shape[0]
        mask = torch.ones(
            1, self.max_tokens, dtype=torch.bool, device=ccd.device
        )
        mask[:, :length] = False
        delta = self.optical_branch.decode_measured_ccd(
            ccd.unsqueeze(0), mask, electronic.dtype
        )[0, :length]
        latent = self.output_norm(
            electronic.to(delta.device).float() + self.optical_fusion * delta.float()
        )
        return torch.cat((latent.mean(dim=0), latent.amax(dim=0)), dim=0)


class LanguageTwoBlockOpticalReplacement(LanguageElectronicReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(hidden_size, settings)
        self.core = LanguageTwoBlockOpticalCore(
            hidden_size, settings.max_language_tokens, settings
        )

    def forward_stage(
        self,
        stage: int,
        hidden_states: torch.Tensor,
        optical_input: torch.Tensor | None = None,
        residual_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        branch = hidden_states if optical_input is None else optical_input
        if self.valid_mask is None or self.valid_mask.shape != branch.shape[:2]:
            raise RuntimeError("Call prepare_student_batch before language replacement")
        mask = self.valid_mask.to(branch.device)
        groups = [branch[index, mask[index]] for index in range(branch.shape[0])]
        packed, _ = self.core.forward_stage_groups(stage, groups, causal=True)
        output = torch.zeros_like(hidden_states)
        output[mask] = packed
        return output

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.optical_branch.set_phase_dropout_active(active)
