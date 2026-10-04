import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]
PHASE_DROPOUT_MODES = {"none", "phase_bypass", "block_phase_bypass"}


class AngularSpectrumPropagator(nn.Module):
    """Fixed-distance angular spectrum free-space propagation."""

    def __init__(
        self,
        wavelength_m: float,
        pixel_size_m: float,
        grid_size: GridSize,
        distance_m: float,
        evanescent_mode: str = "zero",
        k_space_constraint_enabled: bool = False,
        theta_max_deg: float = 1.0,
    ):
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
        self.k_space_constraint_enabled = bool(k_space_constraint_enabled)
        self.theta_max_deg = float(theta_max_deg)
        if self.k_space_constraint_enabled and not (0.0 < self.theta_max_deg <= 90.0):
            raise ValueError("theta_max_deg must be in (0, 90] when k-space constraint is enabled.")
        transfer_function, k_space_mask, max_angle_deg = self._build_transfer_function()
        self.max_sampled_angle_deg = float(max_angle_deg)
        self.k_space_pass_fraction = float(k_space_mask.to(torch.float64).mean().item())
        self.register_buffer("transfer_function", transfer_function, persistent=False)
        self.register_buffer("k_space_mask", k_space_mask, persistent=False)

    def _build_transfer_function(self):
        height, width = self.grid_size
        # Match the notebook/NumPy kernel construction in float64.  Computing
        # the multi-million-radian propagation phase in float32 introduces a
        # visible phase error even though shifted and unshifted FFT forms are
        # mathematically equivalent.
        fy = torch.fft.fftfreq(height, d=self.pixel_size_m, dtype=torch.float64)
        fx = torch.fft.fftfreq(width, d=self.pixel_size_m, dtype=torch.float64)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * (
            (1.0 / self.wavelength_m) ** 2 - fx_grid.square() - fy_grid.square()
        )
        propagating = argument >= 0.0
        radial_frequency = torch.sqrt(fx_grid.square() + fy_grid.square())
        wave_number = 2.0 * math.pi / self.wavelength_m
        radial_wave_number = 2.0 * math.pi * radial_frequency
        angle = torch.asin((radial_wave_number / wave_number).clamp(0.0, 1.0))
        max_angle_deg = float(torch.rad2deg(angle.max()).item())
        if self.k_space_constraint_enabled:
            cutoff = wave_number * math.sin(math.radians(self.theta_max_deg))
            k_space_mask = radial_wave_number <= cutoff
        else:
            k_space_mask = torch.ones_like(propagating, dtype=torch.bool)
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        pass_mask = propagating & k_space_mask
        return torch.where(pass_mask, transfer, torch.zeros_like(transfer)), k_space_mask, max_angle_deg

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != self.grid_size:
            raise ValueError(f"Expected grid {self.grid_size}, got {tuple(field.shape[-2:])}")
        field = field.to(torch.complex64)
        # The notebook applies the first mask directly to the input.  Returning
        # here also avoids an unnecessary FFT/IFFT round trip at z=0.
        if self.distance_m == 0.0:
            return field
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        return torch.fft.ifft2(spectrum * self.transfer_function, dim=(-2, -1)).to(torch.complex64)

