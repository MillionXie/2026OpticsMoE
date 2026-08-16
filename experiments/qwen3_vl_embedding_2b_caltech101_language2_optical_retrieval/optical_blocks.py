from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
    LanguageElectronicReplacement,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    AngularSpectrumPropagator,
    PhaseLayer,
)


class RobustCCDNormalizer(nn.Module):
    """Gain/offset robust transform shared by simulated and measured CCD frames."""

    def __init__(self, grid_size: int, settings: Any) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.background_quantile = float(
            settings.language_optical_background_quantile
        )
        self.relative_clip = float(settings.language_optical_normalization_clip)
        self.log_compression = float(settings.language_optical_log_compression)
        self.norm = nn.LayerNorm(self.grid_size, elementwise_affine=True)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (
            self.grid_size,
            self.grid_size,
        ):
            raise ValueError(
                f"CCD intensity must be [B,{self.grid_size},{self.grid_size}], "
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
        compressed = torch.log1p(self.log_compression * relative)
        return self.norm(compressed)


class SinglePlaneLanguageOptics(nn.Module):
    """One physical phase plane driven by the input of Language mixer block 2."""

    def __init__(self, width: int, settings: Any) -> None:
        super().__init__()
        self.width = int(width)
        self.grid_size = int(settings.language_optical_grid_size)
        self.canvas_size = int(settings.language_optical_canvas_size)
        self.offset = (self.canvas_size - self.grid_size) // 2
        self.input_rms = float(settings.language_optical_input_rms)
        self.target_mean = float(settings.language_optical_ccd_target_mean)
        self.max_shift_pixels = int(settings.language_optical_max_shift_pixels)
        self.ccd_shift_pixels = int(settings.language_optical_ccd_shift_pixels)
        self.gain_min = float(settings.language_optical_gain_min)
        self.gain_max = float(settings.language_optical_gain_max)
        self.offset_fraction = float(settings.language_optical_offset_fraction)
        self.read_noise_fraction = float(
            settings.language_optical_read_noise_fraction
        )
        self.input_adapter = nn.Linear(self.width, self.grid_size)
        self.input_norm = nn.LayerNorm(self.grid_size)
        self.phase = PhaseLayer(
            self.grid_size,
            settings.language_optical_phase_parameterization,
            settings.language_optical_phase_init,
            settings.language_optical_phase_init_std,
            settings.language_optical_phase_dropout_mode,
            settings.language_optical_phase_dropout_p,
            settings.language_optical_phase_dropout_block_size,
            True,
        )
        self.propagator = AngularSpectrumPropagator(
            settings.language_optical_wavelength_nm * 1.0e-9,
            settings.language_optical_pixel_pitch_um * 1.0e-6,
            self.canvas_size,
            settings.language_optical_distance_m,
            settings.language_optical_k_space_enabled,
            settings.language_optical_theta_max_deg,
        )
        self.ccd_normalizer = RobustCCDNormalizer(self.grid_size, settings)
        self.decoder = nn.Sequential(
            nn.Linear(self.grid_size, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.current_operating_loss: torch.Tensor | None = None
        self.last_amplitude: torch.Tensor | None = None
        self.last_raw_ccd: torch.Tensor | None = None
        self.last_normalized_ccd: torch.Tensor | None = None

    def _perturb_detector(self, intensity: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return intensity
        if self.ccd_shift_pixels > 0:
            shift_y = int(
                torch.randint(-self.ccd_shift_pixels, self.ccd_shift_pixels + 1, ()).item()
            )
            shift_x = int(
                torch.randint(-self.ccd_shift_pixels, self.ccd_shift_pixels + 1, ()).item()
            )
            intensity = self._translate_zero(intensity, shift_y, shift_x)
        batch = intensity.shape[0]
        gain = torch.empty(batch, 1, 1, device=intensity.device).uniform_(
            self.gain_min, self.gain_max
        )
        reference = intensity.mean(dim=(-2, -1), keepdim=True).detach()
        offset = torch.empty_like(gain).uniform_(0.0, self.offset_fraction) * reference
        noise = torch.randn_like(intensity) * self.read_noise_fraction * reference
        return (gain * intensity + offset + noise).clamp_min(0.0)

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

    def encode_amplitude(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        amplitude = F.softplus(self.input_norm(self.input_adapter(latent.float())))
        amplitude = amplitude.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if amplitude.shape[1] > self.grid_size:
            raise RuntimeError("Language sequence exceeds optical SLM rows")
        if amplitude.shape[1] < self.grid_size:
            amplitude = F.pad(amplitude, (0, 0, 0, self.grid_size - amplitude.shape[1]))
        rms = amplitude.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return amplitude * (self.input_rms / rms)

    def simulate(self, amplitude: torch.Tensor) -> torch.Tensor:
        if self.training and self.max_shift_pixels > 0:
            shift_y = int(torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, ()).item())
            shift_x = int(torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, ()).item())
            amplitude = self._translate_zero(amplitude, shift_y, shift_x)
        modulated = self.phase(torch.complex(amplitude, torch.zeros_like(amplitude)))
        canvas = torch.zeros(
            amplitude.shape[0],
            self.canvas_size,
            self.canvas_size,
            device=amplitude.device,
            dtype=torch.complex64,
        )
        y0 = self.offset
        canvas[:, y0 : y0 + self.grid_size, y0 : y0 + self.grid_size] = modulated
        detector = self.propagator(canvas)
        intensity = detector.abs().square().float()
        return intensity[:, y0 : y0 + self.grid_size, y0 : y0 + self.grid_size]

    def decode_intensity(
        self, intensity: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.ccd_normalizer(intensity)
        delta = self.decoder(normalized)
        self.last_normalized_ccd = normalized.detach()
        token_count = padding_mask.shape[1]
        delta = delta[:, :token_count]
        return delta.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self, latent: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        amplitude = self.encode_amplitude(latent, padding_mask)
        raw_ccd = self.simulate(amplitude)
        clean_mean = raw_ccd.mean(dim=(-2, -1)).clamp_min(1.0e-8)
        target = raw_ccd.new_tensor(self.target_mean).clamp_min(1.0e-8)
        self.current_operating_loss = F.smooth_l1_loss(
            clean_mean.log(), target.log().expand_as(clean_mean)
        )
        perturbed = self._perturb_detector(raw_ccd)
        self.last_amplitude = amplitude.detach()
        self.last_raw_ccd = raw_ccd.detach()
        return self.decode_intensity(perturbed, padding_mask)

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_dropout_active(active)


class LanguageSecondLayerOpticalCore(ElectronicSequenceCore):
    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(
            hidden_size,
            max_tokens,
            settings,
            settings.electronic_language_token_mixer_type,
            settings.electronic_language_token_mixer_kernel_size,
        )
        if len(self.blocks) != 2:
            raise ValueError("Language second-layer optics requires two mixer blocks")
        self.optical_branch = SinglePlaneLanguageOptics(self.width, settings)
        self.optical_fusion_logit = nn.Parameter(
            torch.logit(torch.tensor(float(settings.optical_fusion_initial)))
        )
        self.last_block2_input_groups: list[torch.Tensor] = []
        self.last_electronic_block2_groups: list[torch.Tensor] = []

    @property
    def optical_fusion(self) -> torch.Tensor:
        return torch.sigmoid(self.optical_fusion_logit)

    def parameter_breakdown(self) -> dict[str, Any]:
        report = super().parameter_breakdown()
        report["implementation"] = "electronic_block1_block2_plus_optical_residual"
        report["optical_parameters"] = sum(
            parameter.numel() for parameter in self.optical_branch.parameters()
        )
        report["optical_fusion_parameters"] = self.optical_fusion_logit.numel()
        report["router_parameters"] = 0
        return report

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        *,
        causal: bool,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not causal:
            raise RuntimeError("Language optical core expects causal token mixing")
        lengths = [len(group) for group in groups]
        if not lengths or any(length <= 0 or length > self.max_tokens for length in lengths):
            raise RuntimeError(f"Invalid Language token lengths {lengths}")
        max_length = max(lengths)
        padded = groups[0].new_zeros(len(groups), max_length, self.hidden_size)
        padding_mask = torch.ones(
            len(groups), max_length, dtype=torch.bool, device=groups[0].device
        )
        for index, group in enumerate(groups):
            padded[index, : len(group)] = group
            padding_mask[index, : len(group)] = False
        input_latent = self.input_norm(self.input_adapter(padded.float()))
        block2_input = self.blocks[0](
            input_latent, padding_mask=padding_mask, causal=True
        )
        electronic = self.blocks[1](
            block2_input, padding_mask=padding_mask, causal=True
        )
        optical_delta = self.optical_branch(block2_input, padding_mask)
        latent = electronic + self.optical_fusion * optical_delta
        latent = self.output_norm(latent).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        gate = torch.sigmoid(self.residual_logit)
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
        self.last_routing = {
            "selected_mask": torch.ones(len(groups), 1, dtype=torch.bool, device=padded.device),
            "importance": torch.ones(1, device=padded.device),
            "normalized_entropy": torch.zeros((), device=padded.device),
        }
        packed = torch.cat(
            [output[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        return packed, latent

    def detector_features_from_cached(
        self, electronic: torch.Tensor, ccd: torch.Tensor
    ) -> torch.Tensor:
        if electronic.ndim != 2 or electronic.shape[-1] != self.width:
            raise ValueError("Cached electronic block-2 output must be [L,width]")
        length = electronic.shape[0]
        grid_size = self.optical_branch.grid_size
        mask = torch.zeros(1, grid_size, dtype=torch.bool, device=ccd.device)
        if length < grid_size:
            mask[:, length:] = True
        delta = self.optical_branch.decode_intensity(ccd.unsqueeze(0), mask)[0, :length]
        latent = self.output_norm(electronic.to(delta.device).float() + self.optical_fusion * delta)
        return torch.cat((latent.mean(dim=0), latent.amax(dim=0)), dim=0)


class LanguageSecondLayerOpticalReplacement(LanguageElectronicReplacement):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(hidden_size, settings)
        self.core = LanguageSecondLayerOpticalCore(
            hidden_size, settings.max_language_tokens, settings
        )

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.optical_branch.set_phase_dropout_active(active)
