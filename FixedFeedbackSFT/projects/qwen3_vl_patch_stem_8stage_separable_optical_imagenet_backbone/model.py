from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    OpticalOEOStage,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    QwenStemSlimMixerOpticalImageNetBackbone,
    grid_to_qwen_tokens,
    qwen_tokens_to_grid,
)


OpticalAxis = Literal["token", "channel"]


def qwen_field_to_row_major(
    field: torch.Tensor,
    *,
    token_count: int = 196,
) -> torch.Tensor:
    """Put the active Qwen tokens in true row-major order; preserve pad rows."""

    if field.ndim != 4 or field.shape[-2] < int(token_count):
        raise ValueError(f"Expected [B,C,H,W] with at least {token_count} rows")
    batch, banks, _, width = field.shape
    active = field[:, :, :token_count, :].reshape(
        batch * banks, token_count, width
    )
    grid = qwen_tokens_to_grid(active)
    row_major = grid.permute(0, 2, 3, 1).reshape(
        batch, banks, token_count, width
    )
    return torch.cat((row_major, field[:, :, token_count:, :]), dim=-2)


def row_major_field_to_qwen(
    field: torch.Tensor,
    *,
    token_count: int = 196,
) -> torch.Tensor:
    """Invert :func:`qwen_field_to_row_major`; preserve pad rows."""

    if field.ndim != 4 or field.shape[-2] < int(token_count):
        raise ValueError(f"Expected [B,C,H,W] with at least {token_count} rows")
    batch, banks, _, width = field.shape
    grid_size = math.isqrt(int(token_count))
    if grid_size * grid_size != int(token_count):
        raise ValueError("Token count must be a square")
    grid = field[:, :, :token_count, :].reshape(
        batch * banks, grid_size, grid_size, width
    )
    qwen = grid_to_qwen_tokens(grid.permute(0, 3, 1, 2)).reshape(
        batch, banks, token_count, width
    )
    return torch.cat((qwen, field[:, :, token_count:, :]), dim=-2)


class AxisAngularSpectrumPropagator(nn.Module):
    """One-dimensional angular-spectrum propagation with the other axis relayed.

    ``token`` propagates only along tensor rows; every feature column remains an
    independent optical line. ``channel`` propagates only along tensor columns;
    every token row remains an independent optical line.
    """

    def __init__(
        self,
        size: int,
        wavelength_m: float,
        pixel_size_m: float,
        distance_m: float,
        axis: OpticalAxis,
    ) -> None:
        super().__init__()
        if axis not in {"token", "channel"}:
            raise ValueError(f"Unsupported optical axis: {axis}")
        self.size = int(size)
        self.axis: OpticalAxis = axis
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(
            self.size,
            d=float(pixel_size_m),
            dtype=torch.float64,
        )
        wave_number_sq = (1.0 / float(wavelength_m)) ** 2 - frequency.square()
        propagating = wave_number_sq >= 0.0
        phase = 2.0 * math.pi * self.distance_m * torch.sqrt(
            wave_number_sq.clamp_min(0.0)
        )
        transfer_1d = torch.where(
            propagating,
            torch.exp(1j * phase).to(torch.complex64),
            torch.zeros_like(phase, dtype=torch.complex64),
        )
        if axis == "token":
            transfer = transfer_1d.view(self.size, 1).expand(
                self.size, self.size
            )
        else:
            transfer = transfer_1d.view(1, self.size).expand(
                self.size, self.size
            )
        self.register_buffer(
            "transfer_function",
            transfer.contiguous(),
            persistent=True,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(
                f"Expected [B,C,{self.size},{self.size}], got {tuple(field.shape)}"
            )
        spectrum = torch.fft.fft2(
            field.to(torch.complex64),
            dim=(-2, -1),
            norm="ortho",
        )
        return torch.fft.ifft2(
            spectrum * self.transfer_function,
            dim=(-2, -1),
            norm="ortho",
        ).to(torch.complex64)


class AxisOpticalOEOStage(OpticalOEOStage):
    """OEO stage whose optical branch mixes exactly one semantic tensor axis."""

    def __init__(
        self,
        *,
        optical_axis: OpticalAxis,
        token_count: int,
        **kwargs: Any,
    ) -> None:
        wavelength = float(kwargs["wavelength_m"])
        pixel_size = float(kwargs["pixel_size_m"])
        distance = float(kwargs["distance_m"])
        super().__init__(**kwargs)
        self.optical_axis: OpticalAxis = optical_axis
        self.token_count = int(token_count)
        self.propagator = AxisAngularSpectrumPropagator(
            self.size,
            wavelength,
            pixel_size,
            distance,
            optical_axis,
        )

    def _optical_branch(
        self,
        amplitude: torch.Tensor,
        phase_override: torch.Tensor | None,
        *,
        phase_shift_dy_dx: tuple[float, float] | None = None,
        phase_error_rad: torch.Tensor | None = None,
        detector_noise_relative_rms: float = 0.0,
        detector_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # Token-stage phase tensors live in the row-major physical plane. Only
        # the signal is permuted at the optical/electronic boundary; therefore
        # BP, fixed feedback, random phase, shifts and phase errors all operate
        # in one consistent physical layout inside the parent implementation.
        physical_amplitude = (
            qwen_field_to_row_major(amplitude, token_count=self.token_count)
            if self.optical_axis == "token"
            else amplitude
        )
        physical_output = super()._optical_branch(
            physical_amplitude,
            phase_override,
            phase_shift_dy_dx=phase_shift_dy_dx,
            phase_error_rad=phase_error_rad,
            detector_noise_relative_rms=detector_noise_relative_rms,
            detector_generator=detector_generator,
        )
        if self.optical_axis == "token":
            return row_major_field_to_qwen(
                physical_output,
                token_count=self.token_count,
            )
        return physical_output


class QwenStemSeparableOpticalImageNetBackbone(
    QwenStemSlimMixerOpticalImageNetBackbone
):
    """P11: four token-axis -> channel-axis optical macro blocks."""

    def __init__(self, stem_checkpoint: str | Path, config: dict[str, Any]) -> None:
        super().__init__(stem_checkpoint, config)
        token_distance = float(config.get("token_axis_propagation_distance_m", 0.05))
        channel_distance = float(
            config.get("channel_axis_propagation_distance_m", 0.05)
        )
        if token_distance <= 0.0 or channel_distance <= 0.0:
            raise ValueError("Axis propagation distances must be positive")
        old_stages = list(self.stages)
        stages: list[nn.Module] = []
        schedule: list[dict[str, float | str | int]] = []
        for index, old_stage in enumerate(old_stages):
            axis: OpticalAxis = "token" if index % 2 == 0 else "channel"
            distance = token_distance if axis == "token" else channel_distance
            stage = AxisOpticalOEOStage(
                optical_axis=axis,
                token_count=self.stem.token_count,
                size=self.canvas_size,
                channels=self.optical_channels,
                wavelength_m=float(config.get("wavelength_m", 5.32e-7)),
                pixel_size_m=float(config.get("pixel_size_m", 1.6e-5)),
                distance_m=distance,
                phase_init_std=float(config.get("phase_init_std", 0.10)),
                layernorm_eps=float(config.get("layernorm_eps", 1.0e-5)),
                residual_mode="constrained",
                residual_main_init=float(config.get("optical_gate_init", 0.60)),
                residual_main_min=float(config.get("optical_gate_min", 0.50)),
                normalize_branch_rms=True,
                random_seed=int(config.get("seed", 2026)) + 1009 * index,
                electronic_skip_mode="identity",
                long_skip_enabled=False,
                long_skip_weight_init=0.0,
                long_skip_weight_max=0.0,
            )
            # Reuse the already initialized P09 electronics exactly. This keeps
            # parameter counts, gates and random initialization controlled.
            stage.residual = old_stage.residual
            stage.electronic_skip = old_stage.electronic_skip
            stages.append(stage)
            schedule.append(
                {
                    "stage": index + 1,
                    "axis": axis,
                    "distance_m": distance,
                    "qwen_row_major_physical_layout": axis == "token",
                }
            )
        self.stages = nn.ModuleList(stages)
        self.token_axis_propagation_distance_m = token_distance
        self.channel_axis_propagation_distance_m = channel_distance
        self.axis_schedule = schedule
        self.register_buffer(
            "p11_separable_architecture_signature",
            torch.tensor([11, 1, 2, 4], dtype=torch.int64),
            persistent=True,
        )

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "optical_mixer_variant": "separable_token_channel_axis",
                "optical_macro_blocks": self.num_stages // 2,
                "token_axis_propagation_distance_m": self.token_axis_propagation_distance_m,
                "channel_axis_propagation_distance_m": self.channel_axis_propagation_distance_m,
                "axis_schedule": self.axis_schedule,
                "qwen_token_order_corrected_inside_token_optics": True,
                "adds_trainable_parameters_over_p09": 0,
            }
        )
        return report
