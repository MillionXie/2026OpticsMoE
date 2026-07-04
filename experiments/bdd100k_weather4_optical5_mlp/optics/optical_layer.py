from __future__ import annotations

import math

import torch
from torch import nn

from .angular_spectrum import AngularSpectrumPropagator


class OpticalDetectionLayer(nn.Module):
    """One propagation, modulation, detection, normalization, and re-encoding step."""

    def __init__(
        self,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_cm: float,
        phase_init: str,
        amplitude_mask_enabled: bool,
        phase_dropout: object,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.padding_size = int(padding_size)
        self.eps = float(eps)
        self.phase_dropout = phase_dropout
        self.phase_dropout_active = False
        self.raw_phase = nn.Parameter(torch.empty(self.field_size, self.field_size, dtype=torch.float32))
        if phase_init == "zeros":
            nn.init.zeros_(self.raw_phase)
        elif phase_init == "uniform":
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        else:
            raise ValueError("phase_init must be 'zeros' or 'uniform'")
        self.raw_amplitude = (
            nn.Parameter(torch.full((self.field_size, self.field_size), 4.0, dtype=torch.float32))
            if amplitude_mask_enabled else None
        )
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=float(wavelength_nm) * 1e-9,
            pixel_pitch_m=float(pixel_pitch_um) * 1e-6,
            grid_size=self.padding_size,
            distance_m=float(distance_cm) * 1e-2,
        )

    def phase_wrapped(self) -> torch.Tensor:
        return torch.remainder(self.raw_phase, 2.0 * math.pi)

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase_dropout_active = bool(active)

    def _phase_modulation(self, batch_size: int) -> torch.Tensor:
        phase = self.raw_phase
        modulation = torch.exp(1j * phase).to(torch.complex64).unsqueeze(0)
        cfg = self.phase_dropout
        enabled = self.training and self.phase_dropout_active and cfg.enabled and cfg.p > 0.0 and cfg.mode != "none"
        if not enabled:
            return modulation
        mask_batch = 1 if cfg.batch_shared else batch_size
        keep_probability = 1.0 - float(cfg.p)
        if cfg.mode == "phase_bypass":
            keep = (torch.rand(mask_batch, self.field_size, self.field_size, device=phase.device) < keep_probability).float()
        elif cfg.mode == "block_phase_bypass":
            block = max(1, int(cfg.block_size))
            low = math.ceil(self.field_size / block)
            keep = (torch.rand(mask_batch, low, low, device=phase.device) < keep_probability).float()
            keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)[:, : self.field_size, : self.field_size]
        else:
            raise ValueError(f"Unsupported phase dropout mode: {cfg.mode}")
        keep = keep.to(torch.complex64)
        return keep * modulation + (1.0 - keep)

    def forward(self, amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if amplitude.ndim != 3 or tuple(amplitude.shape[-2:]) != (self.field_size, self.field_size):
            raise ValueError(f"Expected [B,{self.field_size},{self.field_size}], got {tuple(amplitude.shape)}")
        top = (self.padding_size - self.field_size) // 2
        bottom = self.padding_size - self.field_size - top
        field = torch.complex(amplitude.float(), torch.zeros_like(amplitude, dtype=torch.float32))
        # Modulate before free-space propagation. A phase mask placed directly on
        # the detector would cancel under |E|^2 and would therefore be untrainable.
        field = field * self._phase_modulation(amplitude.shape[0])
        if self.raw_amplitude is not None:
            field = field * torch.sigmoid(self.raw_amplitude).unsqueeze(0)
        field = torch.nn.functional.pad(field, (top, bottom, top, bottom))
        field = self.propagator(field)
        field = field[:, top : top + self.field_size, top : top + self.field_size]
        intensity = field.abs().square().float()
        intensity = intensity / intensity.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        intensity = torch.relu(intensity)
        return torch.sqrt(intensity + self.eps), intensity
