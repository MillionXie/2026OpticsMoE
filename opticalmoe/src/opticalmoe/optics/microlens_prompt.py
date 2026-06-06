import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .four_expert_geometry import FourExpertLayout


class MicrolensArrayPrompt(nn.Module):
    """Spatially partitioned four-cell microlens prompt.

    Every non-overlapping cell contains a local thin-lens phase, a local
    grating/prism phase, and one scalar amplitude. The cells are summed only
    after masking, so this is spatial partitioning rather than coherent global
    grating superposition.
    """

    MODES = {"lens_only", "lens_plus_grating", "grating_only", "identity_prompt"}

    def __init__(
        self,
        layout: FourExpertLayout,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        focal_length_m: float = 0.10,
        input_to_prompt_m: float = 0.20,
        amplitudes: Optional[Sequence[float]] = None,
        phase_biases: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        layout.validate()
        self.layout = layout
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.focal_length_m = float(focal_length_m)
        self.input_to_prompt_m = float(input_to_prompt_m)

        amplitudes = [1.0, 1.0, 1.0, 1.0] if amplitudes is None else list(amplitudes)
        phase_biases = [0.0, 0.0, 0.0, 0.0] if phase_biases is None else list(phase_biases)
        if len(amplitudes) != 4:
            raise ValueError("amplitudes must contain exactly four values.")
        if len(phase_biases) != 4:
            raise ValueError("phase_biases must contain exactly four values.")
        if any(value < 0.0 for value in amplitudes):
            raise ValueError("Prompt amplitudes must be non-negative.")

        y_grid, x_grid = layout.physical_grids(pixel_size_m)
        masks = layout.prompt_cell_masks()
        lens_phases = []
        grating_phases = []
        cell_reports = []

        for index, cell in enumerate(layout.prompt_cells):
            offset_y_m, offset_x_m = layout.cell_offset_meters(index, pixel_size_m)
            center_y_m = offset_y_m
            center_x_m = offset_x_m
            x_local = x_grid - center_x_m
            y_local = y_grid - center_y_m

            lens_phase = (
                -math.pi
                / (self.wavelength_m * self.focal_length_m)
                * (x_local ** 2 + y_local ** 2)
            )

            theta_x = math.atan(-offset_x_m / self.input_to_prompt_m)
            theta_y = math.atan(-offset_y_m / self.input_to_prompt_m)
            fx = math.sin(theta_x) / self.wavelength_m
            fy = math.sin(theta_y) / self.wavelength_m
            grating_phase = 2.0 * math.pi * (fx * x_local + fy * y_local)

            lens_phases.append(lens_phase * masks[index])
            grating_phases.append(grating_phase * masks[index])
            cell_reports.append(
                {
                    "cell": cell.name,
                    "expert": f"E{index}",
                    "center_y_px": cell.center[0],
                    "center_x_px": cell.center[1],
                    "offset_y_px": layout.cell_offset_pixels(index)[0],
                    "offset_x_px": layout.cell_offset_pixels(index)[1],
                    "offset_y_m": offset_y_m,
                    "offset_x_m": offset_x_m,
                    "theta_y_deg": math.degrees(theta_y),
                    "theta_x_deg": math.degrees(theta_x),
                    "grating_period_y_px": self._period_pixels(fy),
                    "grating_period_x_px": self._period_pixels(fx),
                    "focal_length_m": self.focal_length_m,
                    "input_to_prompt_m": self.input_to_prompt_m,
                }
            )

        self.register_buffer("cell_masks", masks, persistent=False)
        self.register_buffer("lens_phases", torch.stack(lens_phases, dim=0), persistent=False)
        self.register_buffer("grating_phases", torch.stack(grating_phases, dim=0), persistent=False)
        self.register_buffer(
            "amplitudes",
            torch.tensor(amplitudes, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "phase_biases",
            torch.tensor(phase_biases, dtype=torch.float32),
            persistent=False,
        )
        self.cell_reports = cell_reports

    def _period_pixels(self, spatial_frequency_per_m: float) -> float:
        if abs(spatial_frequency_per_m) < 1e-20:
            return float("inf")
        return 1.0 / (abs(spatial_frequency_per_m) * self.pixel_size_m)

    def set_amplitudes(self, amplitudes: Sequence[float]) -> None:
        values = torch.tensor(list(amplitudes), dtype=torch.float32, device=self.amplitudes.device)
        if values.numel() != 4:
            raise ValueError("amplitudes must contain exactly four values.")
        if torch.any(values < 0.0):
            raise ValueError("Prompt amplitudes must be non-negative.")
        self.amplitudes.copy_(values)

    def amplitude_map(self) -> torch.Tensor:
        return torch.sum(
            self.cell_masks * self.amplitudes.view(4, 1, 1),
            dim=0,
        )

    def lens_phase_map(self) -> torch.Tensor:
        return self.lens_phases.sum(dim=0)

    def grating_phase_map(self) -> torch.Tensor:
        return self.grating_phases.sum(dim=0)

    def phase_map(self, mode: str) -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if mode == "identity_prompt":
            return torch.zeros(self.layout.canvas_shape, dtype=torch.float32, device=self.cell_masks.device)

        phase = torch.zeros(self.layout.canvas_shape, dtype=torch.float32, device=self.cell_masks.device)
        for index in range(4):
            if mode in {"lens_only", "lens_plus_grating"}:
                phase = phase + self.lens_phases[index]
            if mode in {"grating_only", "lens_plus_grating"}:
                phase = phase + self.grating_phases[index]
            phase = phase + self.cell_masks[index] * self.phase_biases[index]
        return phase

    def transmission(self, mode: str = "lens_plus_grating") -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if mode == "identity_prompt":
            return torch.ones(self.layout.canvas_shape, dtype=torch.complex64, device=self.cell_masks.device)

        transmission = torch.zeros(
            self.layout.canvas_shape,
            dtype=torch.complex64,
            device=self.cell_masks.device,
        )
        for index in range(4):
            phase = self.phase_biases[index]
            if mode in {"lens_only", "lens_plus_grating"}:
                phase = phase + self.lens_phases[index]
            if mode in {"grating_only", "lens_plus_grating"}:
                phase = phase + self.grating_phases[index]
            cell = (
                self.cell_masks[index]
                * self.amplitudes[index]
                * torch.exp(1j * phase).to(torch.complex64)
            )
            transmission = transmission + cell
        return transmission.to(torch.complex64)

    def forward(self, field: torch.Tensor, mode: str = "lens_plus_grating") -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != self.layout.canvas_shape:
            raise ValueError(
                f"Expected canvas shape {self.layout.canvas_shape}, got {tuple(field.shape[-2:])}"
            )
        return field.to(torch.complex64) * self.transmission(mode).unsqueeze(0)

    def report(
        self,
        prompt_to_expert_m: float,
    ) -> List[Dict]:
        magnification = float(prompt_to_expert_m) / self.input_to_prompt_m
        reports = []
        for item in self.cell_reports:
            row = dict(item)
            row["prompt_to_expert_m"] = float(prompt_to_expert_m)
            row["magnification"] = magnification
            reports.append(row)
        return reports
