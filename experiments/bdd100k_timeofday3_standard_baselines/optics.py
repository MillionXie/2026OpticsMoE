from __future__ import annotations

import math

import torch
from torch import nn


class AngularSpectrumPropagator(nn.Module):
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
        wavelength = wavelength_nm * 1e-9
        pitch = pixel_pitch_um * 1e-6
        distance = distance_cm * 1e-2
        frequencies = torch.fft.fftfreq(self.padding_size, d=pitch, dtype=torch.float32)
        yy, xx = torch.meshgrid(frequencies, frequencies, indexing="ij")
        argument = 1 - (wavelength * xx).square() - (wavelength * yy).square()
        phase = 2 * math.pi / wavelength * distance * torch.sqrt(argument.clamp_min(0))
        transfer = torch.where(
            argument >= 0,
            torch.exp(1j * phase),
            torch.zeros_like(phase, dtype=torch.complex64),
        )
        self.register_buffer("transfer", transfer.to(torch.complex64), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if tuple(field.shape[-2:]) != (self.field_size, self.field_size):
            raise ValueError(f"Expected field size {self.field_size}, got {tuple(field.shape[-2:])}")
        left = (self.padding_size - self.field_size) // 2
        right = self.padding_size - self.field_size - left
        padded = torch.nn.functional.pad(field.to(torch.complex64), (left, right, left, right))
        propagated = torch.fft.ifft2(torch.fft.fft2(padded) * self.transfer)
        return propagated[:, left : left + self.field_size, left : left + self.field_size]

