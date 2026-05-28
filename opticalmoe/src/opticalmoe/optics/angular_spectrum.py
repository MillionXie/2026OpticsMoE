import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]


class AngularSpectrumPropagator(nn.Module):
    """Angular spectrum free-space propagation.

    Units are meters internally. The input field must be complex with shape
    [B, H, W]. The transfer function is precomputed for a fixed wavelength,
    pixel size, grid size, and propagation distance.
    """

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
            raise ValueError("Only evanescent_mode='zero' is currently supported.")

        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size

        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.grid_size = (int(height), int(width))
        self.distance_m = float(distance_m)
        self.evanescent_mode = evanescent_mode

        transfer_function = self._build_transfer_function()
        self.register_buffer("transfer_function", transfer_function, persistent=False)

    def _build_transfer_function(self) -> torch.Tensor:
        height, width = self.grid_size
        fy = torch.fft.fftfreq(height, d=self.pixel_size_m, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, d=self.pixel_size_m, dtype=torch.float32)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

        wavelength = self.wavelength_m
        argument = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2
        propagating = argument >= 0.0
        sqrt_argument = torch.sqrt(torch.clamp(argument, min=0.0))

        k = 2.0 * math.pi / wavelength
        phase = k * self.distance_m * sqrt_argument
        transfer = torch.exp(1j * phase).to(torch.complex64)

        if self.evanescent_mode == "zero":
            transfer = torch.where(propagating, transfer, torch.zeros_like(transfer))

        return transfer

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"Expected grid size {self.grid_size}, got {tuple(field.shape[-2:])}"
            )

        field = field.to(torch.complex64)
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        propagated = torch.fft.ifft2(spectrum * self.transfer_function, dim=(-2, -1))
        return propagated.to(torch.complex64)
