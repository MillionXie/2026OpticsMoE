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


def _translate_with_fill(
    value: torch.Tensor,
    shift_y: int,
    shift_x: int,
    *,
    fill_value: float | complex,
) -> torch.Tensor:
    """Translate the last two dimensions without circular wraparound."""

    shift_y = int(shift_y)
    shift_x = int(shift_x)
    shifted = torch.roll(value, (shift_y, shift_x), dims=(-2, -1))
    height, width = value.shape[-2:]
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return torch.full_like(value, fill_value)
    if shift_y > 0:
        shifted[..., :shift_y, :] = fill_value
    elif shift_y < 0:
        shifted[..., shift_y:, :] = fill_value
    if shift_x > 0:
        shifted[..., :, :shift_x] = fill_value
    elif shift_x < 0:
        shifted[..., :, shift_x:] = fill_value
    return shifted


def _sample_integer_shift(maximum: int, *, training: bool) -> tuple[int, int]:
    """Draw one batch-shared integer translation inside the configured bound."""

    maximum = int(maximum)
    if not training or maximum <= 0:
        return 0, 0
    return (
        int(torch.randint(-maximum, maximum + 1, ()).item()),
        int(torch.randint(-maximum, maximum + 1, ()).item()),
    )


def _shift_full_detector_then_crop(
    full_intensity: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> torch.Tensor:
    """Move the 518-plane detector response before selecting the 478 ROI."""

    shifted = _translate_with_fill(
        full_intensity,
        shift_y,
        shift_x,
        fill_value=0.0,
    )
    return shifted[..., int(y0) : int(y1), int(x0) : int(x1)]


def _initial_bounded_fusion_logit(initial: float, minimum: float) -> torch.Tensor:
    """Map an actual initial fusion fraction into the bounded raw gate."""

    normalized = (float(initial) - float(minimum)) / (1.0 - float(minimum))
    if not 0.0 < normalized < 1.0:
        raise ValueError("initial optical fusion must lie strictly above its floor")
    return torch.logit(torch.tensor(normalized))


def _bounded_fusion(raw_logit: torch.Tensor, minimum: float) -> torch.Tensor:
    return float(minimum) + (1.0 - float(minimum)) * torch.sigmoid(raw_logit)


class RobustCCDNormalizer(nn.Module):
    """Normalize measured intensity scale without inventing a background frame."""

    def __init__(self, settings: Any) -> None:
        super().__init__()
        self.active_size = int(settings.active_size)
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
        frame_mean = value.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        relative = (value / frame_mean).clamp_max(self.relative_clip)
        return torch.log1p(self.log_compression * relative)


class MoE4LanguageTwoBlockOpticalPath(nn.Module):
    """Two-block MoE4 path shared by Vision and Language."""

    def __init__(
        self, width: int, settings: Any, *, max_tokens: int | None = None
    ) -> None:
        super().__init__()
        optical_settings = copy.copy(settings)
        # The optical detector always maps 478 CCD pixels to 224 token rows.
        # The outer electronic retrieval head remains mean/max 384 -> 64.
        optical_settings.detector_output_size = int(settings.input_adapter_dim)
        self.core = HomogeneousMoEOpticalCore(
            int(width),
            int(settings.max_language_tokens if max_tokens is None else max_tokens),
            optical_settings,
        )
        self.width = int(width)
        self.active_size = int(settings.active_size)
        self.input_rms = float(settings.language_optical_input_rms)
        self.target_mean = float(settings.language_optical_ccd_target_mean)
        self.input_shift_pixels = int(settings.language_optical_max_shift_pixels)
        self.phase_shift_pixels = int(settings.language_optical_phase_shift_pixels)
        self.ccd_shift_pixels = int(settings.language_optical_ccd_shift_pixels)
        self.gain_min = float(settings.language_optical_gain_min)
        self.gain_max = float(settings.language_optical_gain_max)
        self.offset_fraction = float(settings.language_optical_offset_fraction)
        self.read_noise_fraction = float(
            settings.language_optical_read_noise_fraction
        )
        self.ccd_normalizer = RobustCCDNormalizer(settings)
        # The two optical blocks use the same readout architecture but own
        # independent weights.  Hardware fine-tuning Block 2 must not silently
        # change the simulated Block-1 expert readout.
        self.expert_readout = copy.deepcopy(self.core.readout)
        self.expert_output_adapter = copy.deepcopy(self.core.output_adapter)
        self.current_operating_loss: torch.Tensor | None = None
        self.current_expert_operating_loss: torch.Tensor | None = None
        self.current_global_operating_loss: torch.Tensor | None = None
        self.last_global_input_amplitude: torch.Tensor | None = None
        self.last_expert_input_amplitude: torch.Tensor | None = None
        self.last_raw_expert_ccd: torch.Tensor | None = None
        self.last_raw_ccd: torch.Tensor | None = None
        self.last_normalized_expert_ccd: torch.Tensor | None = None
        self.last_normalized_ccd: torch.Tensor | None = None
        self.measured_expert_ccd: torch.Tensor | None = None
        self.measured_global_ccd: torch.Tensor | None = None
        self.last_sampled_shifts: dict[
            str, dict[str, tuple[int, int]]
        ] = {}

    @staticmethod
    def _translate_zero(value: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
        return _translate_with_fill(
            value, shift_y, shift_x, fill_value=0.0
        )

    def _draw_stage_shifts(self, stage: str) -> dict[str, tuple[int, int]]:
        """Independently sample input, phase-map and detector-ROI offsets."""

        if stage not in {"expert", "global"}:
            raise ValueError(f"Unknown optical stage {stage!r}")
        shifts = {
            "input": _sample_integer_shift(
                self.input_shift_pixels, training=self.training
            ),
            "phase": _sample_integer_shift(
                self.phase_shift_pixels, training=self.training
            ),
            "ccd": _sample_integer_shift(
                self.ccd_shift_pixels, training=self.training
            ),
        }
        self.last_sampled_shifts[stage] = shifts
        return shifts

    def _expert_phase_modulation(self, field: torch.Tensor) -> torch.Tensor:
        """Build the full 518 expert modulation map with identity-filled gaps."""

        layer = self.core.expert_layers[0]
        modulation = layer._stacked_modulation(field.shape[0])
        # ExpertPhasePlane returns [E,H,W] when dropout is inactive, and
        # [1|B,E,H,W] when dropout is active. Normalize both paths explicitly;
        # eval/export must not rely on dropout adding a batch dimension.
        if modulation.ndim == 3:
            modulation = modulation.unsqueeze(0)
        elif modulation.ndim != 4:
            raise RuntimeError(
                "Expert phase modulation must be [E,H,W] or [1|B,E,H,W], "
                f"got {tuple(modulation.shape)}"
            )
        if modulation.shape[0] == 1 and field.shape[0] != 1:
            modulation = modulation.expand(field.shape[0], -1, -1, -1)
        elif modulation.shape[0] != field.shape[0]:
            raise RuntimeError(
                "Expert phase modulation batch does not match optical field: "
                f"{modulation.shape[0]} != {field.shape[0]}"
            )
        canvas = torch.ones_like(field, dtype=torch.complex64)
        for index, aperture in enumerate(self.core.geometry.expert_apertures):
            canvas[
                :,
                aperture.y0 : aperture.y1,
                aperture.x0 : aperture.x1,
            ] = modulation[:, index]
        return canvas

    def _global_phase_modulation(self, field: torch.Tensor) -> torch.Tensor:
        """Build the full 518 global modulation map with identity outside 478."""

        identity = torch.ones_like(field, dtype=torch.complex64)
        return self.core.global_phase(identity)

    def _simulate_detector_roi(
        self,
        field: torch.Tensor,
        modulation: torch.Tensor,
        shifts: dict[str, tuple[int, int]],
    ) -> torch.Tensor:
        """Shift the phase map, propagate, then shift the full CCD before ROI crop."""

        phase_y, phase_x = shifts["phase"]
        shifted_modulation = _translate_with_fill(
            modulation,
            phase_y,
            phase_x,
            fill_value=1.0 + 0.0j,
        )
        detector_field = self.core.propagator(
            field.to(torch.complex64) * shifted_modulation
        )
        full_intensity = detector_field.abs().square().float()
        ccd_y, ccd_x = shifts["ccd"]
        active = self.core.geometry.active_aperture
        return _shift_full_detector_then_crop(
            full_intensity,
            shift_y=ccd_y,
            shift_x=ccd_x,
            y0=active.y0,
            y1=active.y1,
            x0=active.x0,
            x1=active.x1,
        )

    def _perturb_ccd(self, intensity: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return intensity
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
        *,
        final: bool,
    ) -> torch.Tensor:
        normalized_intensity = self.ccd_normalizer(detector_intensity)
        if final:
            readout, _ = self.core.readout.forward_intensity(normalized_intensity)
            self.core.current_detector_readout = readout
            self.last_normalized_ccd = normalized_intensity.detach()
        else:
            readout, _ = self.expert_readout.forward_intensity(normalized_intensity)
            self.last_normalized_expert_ccd = normalized_intensity.detach()
        packed = torch.cat(
            [readout[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        adapter = self.core.output_adapter if final else self.expert_output_adapter
        return adapter(packed).to(dtype)

    def _operating_loss(self, raw_ccd: torch.Tensor) -> torch.Tensor:
        clean_mean = raw_ccd.mean(dim=(-2, -1)).clamp_min(1.0e-8)
        target = raw_ccd.new_tensor(self.target_mean).clamp_min(1.0e-8)
        return F.smooth_l1_loss(
            clean_mean.log(), target.log().expand_as(clean_mean)
        )

    def _encode_input_fields(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[int]]:
        lengths = [int((~row).sum()) for row in padding_mask]
        groups = [latent[index, :length] for index, length in enumerate(lengths)]
        input_fields = self.core.encode_groups(groups)
        rms = input_fields.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return input_fields * (self.input_rms / rms), lengths

    def _scatter_delta(
        self,
        packed: torch.Tensor,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        delta = torch.zeros(
            padding_mask.shape[0], padding_mask.shape[1], self.width,
            device=packed.device, dtype=dtype,
        )
        delta[~padding_mask] = packed
        return delta

    def run_expert_block(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[int]]:
        """Expert optics -> CCD -> a readout architecturally matching global."""
        input_fields, lengths = self._encode_input_fields(latent, padding_mask)
        field, routing = self.core.begin(input_fields)
        shifts = self._draw_stage_shifts("expert")
        input_y, input_x = shifts["input"]
        field = _translate_with_fill(
            field, input_y, input_x, fill_value=0.0
        )
        active = self.core.geometry.active_aperture
        self.last_expert_input_amplitude = field[
            :, active.y0 : active.y1, active.x0 : active.x1
        ].abs().detach()
        if self.measured_expert_ccd is None:
            raw_ccd = self._simulate_detector_roi(
                field,
                self._expert_phase_modulation(field),
                shifts,
            )
        else:
            raw_ccd = self.measured_expert_ccd.to(field.device).float()
            if tuple(raw_ccd.shape) != (len(field), self.active_size, self.active_size):
                raise RuntimeError(
                    "Measured expert CCD must match [batch,478,478], got "
                    f"{tuple(raw_ccd.shape)}"
                )
        self.current_expert_operating_loss = self._operating_loss(raw_ccd)
        self.current_operating_loss = self.current_expert_operating_loss
        self.last_raw_expert_ccd = raw_ccd.detach()
        measured = self.measured_expert_ccd is not None
        packed = self._readout_delta(
            raw_ccd if measured else self._perturb_ccd(raw_ccd),
            lengths,
            latent.dtype,
            final=False,
        )
        return self._scatter_delta(packed, padding_mask, latent.dtype), routing, lengths

    def encode_global_input(
        self,
        fused_latent: torch.Tensor,
        padding_mask: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Electronically reload the Block-1 fused result for global optics."""
        input_fields, _ = self._encode_input_fields(fused_latent, padding_mask)
        return self.core.fanout(input_fields, routing)

    def run_global_block(
        self,
        field: torch.Tensor,
        lengths: list[int],
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the global phase and final CCD assigned to Language Block 2."""
        shifts = self._draw_stage_shifts("global")
        input_y, input_x = shifts["input"]
        field = _translate_with_fill(
            field, input_y, input_x, fill_value=0.0
        )
        active = self.core.geometry.active_aperture
        self.last_global_input_amplitude = field[
            :, active.y0 : active.y1, active.x0 : active.x1
        ].abs().detach()
        if self.measured_global_ccd is None:
            raw_ccd = self._simulate_detector_roi(
                field,
                self._global_phase_modulation(field),
                shifts,
            )
        else:
            raw_ccd = self.measured_global_ccd.to(field.device).float()
            if tuple(raw_ccd.shape) != (len(field), self.active_size, self.active_size):
                raise RuntimeError(
                    "Measured global CCD must match [batch,478,478], got "
                    f"{tuple(raw_ccd.shape)}"
                )
        self.current_global_operating_loss = self._operating_loss(raw_ccd)
        expert_loss = self.current_expert_operating_loss
        self.current_operating_loss = (
            self.current_global_operating_loss
            if expert_loss is None
            else 0.5 * (expert_loss + self.current_global_operating_loss)
        )
        self.last_raw_ccd = raw_ccd.detach()
        measured = self.measured_global_ccd is not None
        packed_delta = self._readout_delta(
            raw_ccd if measured else self._perturb_ccd(raw_ccd),
            lengths,
            dtype,
            final=True,
        )
        return self._scatter_delta(packed_delta, padding_mask, dtype)

    def decode_measured_ccd(
        self,
        detector_intensity: torch.Tensor,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lengths = [int((~row).sum()) for row in padding_mask]
        packed = self._readout_delta(
            detector_intensity, lengths, dtype, final=True
        )
        return self._scatter_delta(packed, padding_mask, dtype)

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.set_phase_dropout_active(active)

    def set_measured_ccd(
        self,
        *,
        expert: torch.Tensor | None = None,
        global_: torch.Tensor | None = None,
    ) -> None:
        self.measured_expert_ccd = expert
        self.measured_global_ccd = global_

    def clear_measured_ccd(self) -> None:
        self.measured_expert_ccd = None
        self.measured_global_ccd = None


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
        self.minimum_optical_fusion = float(settings.optical_fusion_minimum)
        initial_fusion = _initial_bounded_fusion_logit(
            settings.optical_fusion_initial,
            self.minimum_optical_fusion,
        )
        self.block1_optical_fusion_logit = nn.Parameter(initial_fusion.clone())
        self.block2_optical_fusion_logit = nn.Parameter(initial_fusion.clone())
        self.last_block2_input_groups: list[torch.Tensor] = []
        self.last_electronic_block2_groups: list[torch.Tensor] = []
        self._stage1_global_input: torch.Tensor | None = None
        self._stage1_latent: torch.Tensor | None = None
        self._stage1_lengths: list[int] = []
        self._stage1_padding_mask: torch.Tensor | None = None

    @property
    def block1_optical_fusion(self) -> torch.Tensor:
        return _bounded_fusion(
            self.block1_optical_fusion_logit, self.minimum_optical_fusion
        )

    @property
    def block2_optical_fusion(self) -> torch.Tensor:
        return _bounded_fusion(
            self.block2_optical_fusion_logit, self.minimum_optical_fusion
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        optical = self.optical_branch.core.parameter_breakdown()
        report["implementation"] = "moe4_expert_block1_global_block2_dual_fusion"
        expert_readout_parameters = sum(
            parameter.numel()
            for module in (
                self.optical_branch.expert_readout,
                self.optical_branch.expert_output_adapter,
            )
            for parameter in module.parameters()
        )
        report["optical_parameters"] = (
            optical["total_parameters"] + expert_readout_parameters
        )
        report["expert_readout_parameters"] = expert_readout_parameters
        report["optical_phase_parameters"] = optical["optical_phase_parameters"]
        report["router_parameters"] = optical["router_parameters"]
        report["optical_fusion_parameters"] = (
            self.block1_optical_fusion_logit.numel()
            + self.block2_optical_fusion_logit.numel()
        )
        report["minimum_optical_fusion"] = self.minimum_optical_fusion
        report["fusion_parameterization"] = (
            "minimum + (1-minimum)*sigmoid(raw_gate)"
        )
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
            expert_delta, routing, optical_lengths = self.optical_branch.run_expert_block(
                input_latent, padding_mask
            )
            block1_latent = self.blocks[0](
                input_latent, padding_mask=padding_mask, causal=True
            )
            fused_block1 = (
                block1_latent + self.block1_optical_fusion * expert_delta
            ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            self._stage1_global_input = self.optical_branch.encode_global_input(
                fused_block1, padding_mask, routing
            )
            self._stage1_latent = fused_block1
            self._stage1_lengths = optical_lengths
            self._stage1_padding_mask = padding_mask
            stage1_output = padded.float() + gate * self.output_adapter(fused_block1)
            stage1_output = stage1_output.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            ).to(groups[0].dtype)
            return self._pack(stage1_output, lengths), fused_block1
        if stage != 1:
            raise RuntimeError("Language MoE4 replacement has exactly stages 0 and 1")
        if (
            self._stage1_global_input is None
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
            self._stage1_global_input,
            self._stage1_lengths,
            padding_mask,
            block2_input.dtype,
        )
        latent = self.output_norm(
            electronic + self.block2_optical_fusion * optical_delta
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
            electronic.to(delta.device).float()
            + self.block2_optical_fusion * delta.float()
        )
        return torch.cat((latent.mean(dim=0), latent.amax(dim=0)), dim=0)

    def detector_features_from_block2_inputs(
        self, block2_input_groups: list[torch.Tensor], ccd: torch.Tensor
    ) -> torch.Tensor:
        """Batch the exact trainable Language-Global tail after a frozen capture boundary."""

        if not block2_input_groups or any(
            group.ndim != 2 or group.shape[-1] != self.width
            for group in block2_input_groups
        ):
            raise ValueError("Cached Language Block-2 inputs must be non-empty [L,width]")
        lengths = [len(group) for group in block2_input_groups]
        if any(length <= 0 or length > self.max_tokens for length in lengths):
            raise ValueError(f"Invalid cached Language token lengths {lengths}")
        if ccd.ndim != 3 or len(ccd) != len(block2_input_groups):
            raise ValueError("Measured CCD batch must match cached Language groups")
        device = block2_input_groups[0].device
        if ccd.device != device or any(group.device != device for group in block2_input_groups):
            raise ValueError("Cached Language groups and CCD must use the same device")
        max_length = max(lengths)
        padded = block2_input_groups[0].new_zeros(
            len(block2_input_groups), max_length, self.width
        )
        padding_mask = torch.ones(
            len(block2_input_groups), max_length, dtype=torch.bool, device=device
        )
        for index, group in enumerate(block2_input_groups):
            padded[index, : len(group)] = group.float()
            padding_mask[index, : len(group)] = False
        electronic = self.blocks[1](
            padded.float(), padding_mask=padding_mask, causal=True
        )
        delta = self.optical_branch.decode_measured_ccd(
            ccd, padding_mask, electronic.dtype
        )
        latent = self.output_norm(
            electronic + self.block2_optical_fusion * delta
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return torch.stack(
            [
                torch.cat(
                    (
                        latent[index, :length].mean(dim=0),
                        latent[index, :length].amax(dim=0),
                    ),
                    dim=0,
                )
                for index, length in enumerate(lengths)
            ],
            dim=0,
        )


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

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.optical_branch.core.router_losses()


class VisionTwoBlockOpticalCore(ElectronicSequenceCore):
    """Vision 2D Mixer and two optical planes trained in parallel."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(
            hidden_size,
            max_tokens,
            settings,
            settings.electronic_vision_token_mixer_type,
            settings.electronic_vision_token_mixer_kernel_size,
        )
        if len(self.blocks) != 2:
            raise ValueError("Vision dual-fusion optics requires two electronic blocks")
        self.optical_branch = MoE4LanguageTwoBlockOpticalPath(
            self.width, settings, max_tokens=max_tokens
        )
        self.minimum_optical_fusion = float(settings.optical_fusion_minimum)
        initial = _initial_bounded_fusion_logit(
            settings.optical_fusion_initial,
            self.minimum_optical_fusion,
        )
        self.block1_optical_fusion_logit = nn.Parameter(initial.clone())
        self.block2_optical_fusion_logit = nn.Parameter(initial.clone())

    @property
    def block1_optical_fusion(self) -> torch.Tensor:
        return _bounded_fusion(
            self.block1_optical_fusion_logit, self.minimum_optical_fusion
        )

    @property
    def block2_optical_fusion(self) -> torch.Tensor:
        return _bounded_fusion(
            self.block2_optical_fusion_logit, self.minimum_optical_fusion
        )

    def _pad_groups(
        self, groups: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not groups or any(group.ndim != 2 for group in groups):
            raise ValueError("Vision core expects a non-empty list of [T,D] groups")
        lengths = [len(group) for group in groups]
        if any(length <= 0 or length > self.max_tokens for length in lengths):
            raise RuntimeError(f"Invalid Vision token lengths {lengths}")
        padded = groups[0].new_zeros(
            len(groups), max(lengths), self.hidden_size
        )
        padding_mask = torch.ones(
            len(groups), max(lengths), dtype=torch.bool, device=groups[0].device
        )
        for index, group in enumerate(groups):
            padded[index, : len(group)] = group
            padding_mask[index, : len(group)] = False
        return padded, padding_mask, lengths

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        *,
        causal: bool,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if causal or spatial_shapes is None:
            raise RuntimeError("Vision dual-fusion core requires non-causal 2D shapes")
        padded, padding_mask, lengths = self._pad_groups(groups)
        input_latent = self.input_norm(self.input_adapter(padded.float()))
        expert_delta, routing, optical_lengths = self.optical_branch.run_expert_block(
            input_latent, padding_mask
        )
        electronic1 = self.blocks[0](
            input_latent,
            padding_mask=padding_mask,
            causal=False,
            spatial_shapes=spatial_shapes,
        )
        fused1 = (
            electronic1 + self.block1_optical_fusion * expert_delta
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        global_input = self.optical_branch.encode_global_input(
            fused1, padding_mask, routing
        )
        electronic2 = self.blocks[1](
            fused1,
            padding_mask=padding_mask,
            causal=False,
            spatial_shapes=spatial_shapes,
        )
        global_delta = self.optical_branch.run_global_block(
            global_input,
            optical_lengths,
            padding_mask,
            fused1.dtype,
        )
        latent = self.output_norm(
            electronic2 + self.block2_optical_fusion * global_delta
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        residual_gate = torch.sigmoid(self.residual_logit)
        output = padded.float() + residual_gate * self.output_adapter(latent)
        output = output.to(groups[0].dtype)
        self.last_latent_groups = [
            latent[index, :length] for index, length in enumerate(lengths)
        ]
        self.last_routing = routing
        packed = torch.cat(
            [output[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        return packed, latent

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        report.update(
            {
                "implementation": "vision_2d_mixer_moe4_dual_fusion",
                "moe_enabled": True,
                "optical_parameters": sum(
                    parameter.numel()
                    for parameter in self.optical_branch.parameters()
                ),
                "optical_fusion_parameters": 2,
                "minimum_optical_fusion": self.minimum_optical_fusion,
                "fusion_parameterization": (
                    "minimum + (1-minimum)*sigmoid(raw_gate)"
                ),
                "total_parameters": sum(
                    parameter.numel() for parameter in self.parameters()
                ),
            }
        )
        return report


class VisionTwoBlockOpticalReplacement(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.core = VisionTwoBlockOpticalCore(
            hidden_size, settings.max_visual_tokens, settings
        )
        self.tap_stages: tuple[int, ...] = ()
        self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.spatial_shapes: list[tuple[int, int, int]] | None = None

    def set_image_grid_thw(self, image_grid_thw: torch.Tensor | None) -> None:
        if image_grid_thw is None:
            self.spatial_shapes = None
            return
        shapes: list[tuple[int, int, int]] = []
        for frames, height, width in image_grid_thw.detach().cpu().long().tolist():
            shapes.extend((1, int(height), int(width)) for _ in range(int(frames)))
        self.spatial_shapes = shapes

    def compute(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        residual_base: torch.Tensor | None = None,
    ) -> None:
        from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
            lengths_from_cu,
        )

        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        packed, _ = self.core.forward_groups(
            list(hidden_states.split(lengths)),
            causal=False,
            spatial_shapes=self.spatial_shapes,
        )
        if packed.shape != hidden_states.shape:
            raise RuntimeError("Vision optical replacement changed packed shape")
        self.tap_outputs = [packed]
        self.last_output = packed

    def output_for_slot(self, slot: int) -> torch.Tensor:
        if slot in {0, 1} and self.last_output is not None:
            return self.last_output
        raise RuntimeError("Vision optical outputs are unavailable")

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.optical_branch.core.router_losses()

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.optical_branch.set_phase_dropout_active(active)

    def set_intermediate_field_capture(
        self, _enabled: bool, _sample_count: int = 1
    ) -> None:
        return None

    def parameter_breakdown(self) -> dict[str, Any]:
        return self.core.parameter_breakdown()
