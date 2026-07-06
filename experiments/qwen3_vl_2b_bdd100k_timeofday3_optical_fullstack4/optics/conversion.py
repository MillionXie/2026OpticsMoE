from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .angular_spectrum import AngularSpectrumPropagator


class OpticalConversion(nn.Module):
    """One mask/propagation/detection conversion with intensity-to-intensity output."""

    def __init__(self, field_size: int, padding_size: int, wavelength_nm: float, pixel_pitch_um: float,
                 distance_cm: float, amplitude_mask_enabled: bool = True, phase_init: str = "zeros",
                 phase_init_std: float = 0.02, eps: float = 1e-6) -> None:
        super().__init__()
        self.field_size = int(field_size); self.eps = float(eps)
        self.phase_init = str(phase_init); self.phase_init_std = float(phase_init_std)
        self.phase_mask = nn.Parameter(torch.empty(field_size, field_size, dtype=torch.float32))
        self.reset_phase_parameters()
        self.amplitude_mask_logits = nn.Parameter(torch.full((field_size, field_size), 4.0)) if amplitude_mask_enabled else None
        self.detector_bias = nn.Parameter(torch.zeros(()))
        self.propagator = AngularSpectrumPropagator(field_size, padding_size, wavelength_nm, pixel_pitch_um, distance_cm)

    def reset_phase_parameters(self) -> None:
        if self.phase_init in {"zeros", "identity"}:
            nn.init.zeros_(self.phase_mask)
        elif self.phase_init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.phase_mask, 0.0, 2.0 * math.pi)
        elif self.phase_init in {"normal", "small_normal"}:
            nn.init.normal_(self.phase_mask, mean=0.0, std=self.phase_init_std)
        else:
            raise ValueError(f"Unsupported phase_init: {self.phase_init}")

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (self.field_size, self.field_size):
            raise ValueError(f"OpticalConversion expects [B,{self.field_size},{self.field_size}]")
        normalized = F.relu(intensity.float())
        normalized = normalized / normalized.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        field = torch.complex(normalized, torch.zeros_like(normalized))
        modulation = torch.exp(1j * self.phase_mask.float()).to(torch.complex64)
        if self.amplitude_mask_logits is not None:
            modulation = modulation * torch.sigmoid(self.amplitude_mask_logits.float())
        # Mask before propagation: a phase mask immediately adjacent to |E|^2
        # would algebraically cancel and receive no gradient.
        detected = self.propagator(field * modulation).abs().square().float()
        detected = detected / detected.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return F.relu(detected + self.detector_bias.float())

    def wrapped_phase(self) -> torch.Tensor:
        return torch.remainder(self.phase_mask, 2.0 * torch.pi)
