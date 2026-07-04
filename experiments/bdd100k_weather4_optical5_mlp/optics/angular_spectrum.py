from __future__ import annotations

import math

import torch
from torch import nn


class AngularSpectrumPropagator(nn.Module):
    """Fixed-distance, zero-evanescent angular-spectrum propagation."""

    def __init__(self, wavelength_m: float, pixel_pitch_m: float, grid_size: int, distance_m: float) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        fy = torch.fft.fftfreq(self.grid_size, d=float(pixel_pitch_m), dtype=torch.float32)
        fx = torch.fft.fftfreq(self.grid_size, d=float(pixel_pitch_m), dtype=torch.float32)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        argument = 1.0 - (float(wavelength_m) * fx_grid).square() - (float(wavelength_m) * fy_grid).square()
        propagating = argument >= 0.0
        phase = 2.0 * math.pi / float(wavelength_m) * float(distance_m) * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer("transfer_function", torch.where(propagating, transfer, torch.zeros_like(transfer)), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.grid_size, self.grid_size):
            raise ValueError(f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}")
        spectrum = torch.fft.fft2(field.to(torch.complex64), dim=(-2, -1))
        return torch.fft.ifft2(spectrum * self.transfer_function, dim=(-2, -1))
