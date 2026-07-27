from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class AngularSpectrumPropagator(nn.Module):
    """Band-limited angular-spectrum propagation on a fixed square grid."""

    def __init__(
        self,
        *,
        wavelength_m: float,
        pixel_size_m: float,
        grid_size: int,
        distance_m: float,
        k_space_constraint_enabled: bool = False,
        theta_max_deg: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(
            self.grid_size, d=float(pixel_size_m), dtype=torch.float64
        )
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * (
            (1.0 / float(wavelength_m)) ** 2 - fx.square() - fy.square()
        )
        propagating = argument >= 0
        if k_space_constraint_enabled:
            if not 0.0 < theta_max_deg <= 90.0:
                raise ValueError("theta_max_deg must be in (0,90]")
            radial_wave_number = 2.0 * math.pi * torch.sqrt(
                fx.square() + fy.square()
            )
            cutoff = (2.0 * math.pi / float(wavelength_m)) * math.sin(
                math.radians(theta_max_deg)
            )
            propagating &= radial_wave_number <= cutoff
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(propagating, transfer, torch.zeros_like(transfer)),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (
            self.grid_size,
            self.grid_size,
        ):
            raise ValueError(
                f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}"
            )
        field = field.to(torch.complex64)
        if self.distance_m == 0.0:
            return field
        spectrum = torch.fft.fft2(field)
        return torch.fft.ifft2(spectrum * self.transfer_function).to(torch.complex64)


class PhaseOnlyMask(nn.Module):
    """Trainable phase-only modulation constrained to [0,2π] when requested."""

    def __init__(
        self,
        size: int,
        *,
        parameterization: str,
        init: str,
        init_std: float,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.parameterization = str(parameterization)
        self.raw_phase = nn.Parameter(torch.empty(self.size, self.size))
        if init == "zeros":
            nn.init.zeros_(self.raw_phase)
        elif init == "uniform":
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init == "normal":
            nn.init.normal_(self.raw_phase, 0.0, float(init_std))
        else:
            raise ValueError(f"Unsupported phase initialization {init!r}")

    def phase(self) -> torch.Tensor:
        if self.parameterization == "sigmoid":
            return 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        if self.parameterization == "unconstrained":
            return self.raw_phase
        raise ValueError(f"Unsupported phase parameterization {self.parameterization!r}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.size, self.size)
        if field.ndim != 3 or tuple(field.shape[-2:]) != expected:
            raise ValueError(f"Phase mask expects [B,{self.size},{self.size}]")
        modulation = torch.exp(1j * self.phase()).to(torch.complex64)
        return field.to(torch.complex64) * modulation


@dataclass(frozen=True)
class DetectorRegion:
    class_index: int
    row_index: int
    column_index: int
    y0: int
    y1: int
    x0: int
    x1: int


class TenRegionDetector(nn.Module):
    """Final square-law CCD with ten fixed, equal-area 3/4/3 regions."""

    def __init__(
        self,
        *,
        canvas_size: int,
        active_size: int,
        row_layout: tuple[int, ...],
        region_size: int,
        horizontal_gap: int,
        vertical_gap: int,
        normalize_total_energy: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.canvas_size = int(canvas_size)
        self.active_size = int(active_size)
        self.row_layout = tuple(int(value) for value in row_layout)
        self.region_size = int(region_size)
        self.horizontal_gap = int(horizontal_gap)
        self.vertical_gap = int(vertical_gap)
        self.normalize_total_energy = bool(normalize_total_energy)
        self.eps = float(eps)
        self.active_start = (self.canvas_size - self.active_size) // 2
        self.active_end = self.active_start + self.active_size
        regions = self._build_regions()
        self.regions = regions
        masks = torch.zeros(
            len(regions), self.canvas_size, self.canvas_size, dtype=torch.float32
        )
        for region in regions:
            masks[
                region.class_index,
                region.y0 : region.y1,
                region.x0 : region.x1,
            ] = 1.0
        if masks.sum(0).max().item() > 1:
            raise ValueError("Detector regions overlap")
        self.register_buffer("masks", masks, persistent=False)

    def _build_regions(self) -> tuple[DetectorRegion, ...]:
        total_height = (
            len(self.row_layout) * self.region_size
            + (len(self.row_layout) - 1) * self.vertical_gap
        )
        y_start = (self.canvas_size - total_height) // 2
        regions: list[DetectorRegion] = []
        class_index = 0
        for row_index, count in enumerate(self.row_layout):
            row_width = count * self.region_size + (count - 1) * self.horizontal_gap
            x_start = (self.canvas_size - row_width) // 2
            y0 = y_start + row_index * (self.region_size + self.vertical_gap)
            for column_index in range(count):
                x0 = x_start + column_index * (
                    self.region_size + self.horizontal_gap
                )
                region = DetectorRegion(
                    class_index=class_index,
                    row_index=row_index,
                    column_index=column_index,
                    y0=y0,
                    y1=y0 + self.region_size,
                    x0=x0,
                    x1=x0 + self.region_size,
                )
                if (
                    region.y0 < self.active_start
                    or region.x0 < self.active_start
                    or region.y1 > self.active_end
                    or region.x1 > self.active_end
                ):
                    raise ValueError(
                        f"Detector region {class_index} lies outside the active CCD aperture"
                    )
                regions.append(region)
                class_index += 1
        if class_index != 10:
            raise ValueError(f"Expected ten detector regions, got {class_index}")
        return tuple(regions)

    def forward(
        self, field: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (
            self.canvas_size,
            self.canvas_size,
        ):
            raise ValueError("Detector field has an incompatible shape")
        full_intensity = field.to(torch.complex64).abs().square().float()
        energies = torch.einsum("bhw,chw->bc", full_intensity, self.masks)
        if self.normalize_total_energy:
            active = full_intensity[
                :,
                self.active_start : self.active_end,
                self.active_start : self.active_end,
            ]
            energies = energies / active.sum((-2, -1), keepdim=False).unsqueeze(1).clamp_min(
                self.eps
            )
        detector_intensity = full_intensity[
            :,
            self.active_start : self.active_end,
            self.active_start : self.active_end,
        ]
        return energies, detector_intensity, full_intensity
