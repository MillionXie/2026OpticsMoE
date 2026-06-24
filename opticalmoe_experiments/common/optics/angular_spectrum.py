import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]


class AngularSpectrumPropagator(nn.Module):
    """Fixed-distance angular spectrum free-space propagation."""

    def __init__(
        self,
        wavelength_m: float,
        pixel_size_m: float,
        grid_size: GridSize,
        distance_m: float,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        if evanescent_mode != "zero":
            raise ValueError("Only evanescent_mode='zero' is supported.")
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.grid_size = (int(height), int(width))
        self.distance_m = float(distance_m)
        self.evanescent_mode = evanescent_mode
        self.register_buffer("transfer_function", self._build_transfer_function(), persistent=False)

    def _build_transfer_function(self) -> torch.Tensor:
        height, width = self.grid_size
        fy = torch.fft.fftfreq(height, d=self.pixel_size_m, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, d=self.pixel_size_m, dtype=torch.float32)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        argument = 1.0 - (self.wavelength_m * fx_grid).square() - (self.wavelength_m * fy_grid).square()
        propagating = argument >= 0.0
        phase = (2.0 * math.pi / self.wavelength_m) * self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        return torch.where(propagating, transfer, torch.zeros_like(transfer))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != self.grid_size:
            raise ValueError(f"Expected grid {self.grid_size}, got {tuple(field.shape[-2:])}")
        field = field.to(torch.complex64)
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        return torch.fft.ifft2(spectrum * self.transfer_function, dim=(-2, -1)).to(torch.complex64)

