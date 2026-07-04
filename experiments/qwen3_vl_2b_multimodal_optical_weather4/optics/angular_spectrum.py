from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AngularSpectrumPropagator(nn.Module):
    """Band-limited angular-spectrum propagation for a square scalar field."""

    def __init__(
        self,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_cm: float,
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.padding_size = int(padding_size)
        wavelength = float(wavelength_nm) * 1e-9
        pitch = float(pixel_pitch_um) * 1e-6
        distance = float(distance_cm) * 1e-2
        frequencies = torch.fft.fftfreq(self.padding_size, d=pitch)
        fy, fx = torch.meshgrid(frequencies, frequencies, indexing="ij")
        root = 1.0 - (wavelength * fx).square() - (wavelength * fy).square()
        propagating = root >= 0
        phase = (2.0 * torch.pi / wavelength) * distance * torch.sqrt(root.clamp_min(0.0))
        transfer = torch.polar(propagating.to(torch.float32), phase.to(torch.float32))
        self.register_buffer("transfer_function", transfer.to(torch.complex64), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or field.shape[-2:] != (self.field_size, self.field_size):
            raise ValueError(
                f"Expected [batch,{self.field_size},{self.field_size}] field, got {tuple(field.shape)}"
            )
        left = (self.padding_size - self.field_size) // 2
        right = self.padding_size - self.field_size - left
        padded = F.pad(field, (left, right, left, right))
        spectrum = torch.fft.fft2(padded, norm="ortho")
        propagated = torch.fft.ifft2(
            spectrum * self.transfer_function.to(field.device), norm="ortho"
        )
        return propagated[:, left : left + self.field_size, left : left + self.field_size]

