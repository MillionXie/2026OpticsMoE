from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .angular_spectrum import AngularSpectrumPropagator


class OpticalGroup(nn.Module):
    """Differentiable multi-layer optical propagation and detection group."""

    def __init__(
        self,
        layers: int,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        mask_distance_cm: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = int(layers)
        self.field_size = int(field_size)
        self.eps = float(eps)
        self.propagators = nn.ModuleList(
            [
                AngularSpectrumPropagator(
                    field_size,
                    padding_size,
                    wavelength_nm,
                    pixel_pitch_um,
                    mask_distance_cm,
                )
                for _ in range(self.layers)
            ]
        )
        self.phase_masks = nn.Parameter(torch.zeros(self.layers, field_size, field_size))
        self.amplitude_mask_logits = nn.Parameter(
            torch.full((self.layers, field_size, field_size), 4.0)
        )
        self.detection_bias = nn.Parameter(torch.zeros(self.layers))

    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        if amplitude.ndim != 3 or amplitude.shape[-2:] != (
            self.field_size,
            self.field_size,
        ):
            raise ValueError("OpticalGroup input must be [batch, field_size, field_size]")
        amplitude = self._normalize(F.relu(amplitude))
        field = torch.complex(amplitude, torch.zeros_like(amplitude))
        intensity = amplitude.square()
        for index, propagator in enumerate(self.propagators):
            field = propagator(field)
            phase = torch.polar(
                torch.ones_like(self.phase_masks[index]), self.phase_masks[index]
            )
            transmission = torch.sigmoid(self.amplitude_mask_logits[index])
            field = field * phase * transmission
            intensity = field.abs().square()
            intensity = intensity / intensity.mean(dim=(-2, -1), keepdim=True).clamp_min(
                self.eps
            )
            intensity = F.relu(intensity + self.detection_bias[index])
            amplitude = self._normalize(torch.sqrt(intensity + self.eps))
            field = torch.complex(amplitude, torch.zeros_like(amplitude))
        return intensity

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        rms = value.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        return value / rms.clamp_min(self.eps)

