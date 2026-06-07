from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class Aperture:
    """Integer pixel aperture using Python's half-open slice convention."""

    name: str
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.y0 + self.y1) / 2.0, (self.x0 + self.x1) / 2.0)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["center"] = list(self.center)
        return payload


@dataclass(frozen=True)
class FourExpertLayout:
    """Fixed four-expert geometry for the standalone prompt verification."""

    canvas_height: int = 700
    canvas_width: int = 700
    input_size: int = 200
    expert_size: int = 200
    prompt_cell_size: Optional[int] = None
    gap_pixels: int = 100
    outer_margin: int = 100

    def __post_init__(self) -> None:
        # Preserve the original behavior unless an independent cell size is set.
        if self.prompt_cell_size is None:
            object.__setattr__(self, "prompt_cell_size", int(self.expert_size))

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (int(self.canvas_height), int(self.canvas_width))

    @property
    def canvas_center(self) -> Tuple[float, float]:
        return (self.canvas_height / 2.0, self.canvas_width / 2.0)

    @property
    def input_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        y0 = int(round(cy - self.input_size / 2.0))
        x0 = int(round(cx - self.input_size / 2.0))
        return Aperture("input", y0, y0 + self.input_size, x0, x0 + self.input_size)

    @property
    def experts(self) -> List[Aperture]:
        m = self.outer_margin
        e = self.expert_size
        g = self.gap_pixels
        return [
            Aperture("E0", m, m + e, m, m + e),
            Aperture("E1", m, m + e, m + e + g, m + 2 * e + g),
            Aperture("E2", m + e + g, m + 2 * e + g, m, m + e),
            Aperture("E3", m + e + g, m + 2 * e + g, m + e + g, m + 2 * e + g),
        ]

    @property
    def prompt_cells(self) -> List[Aperture]:
        size = int(self.prompt_cell_size)
        half = size // 2
        cells = []
        for index, expert in enumerate(self.experts):
            center_y, center_x = expert.center
            y0 = int(center_y) - half
            x0 = int(center_x) - half
            cells.append(Aperture(f"C{index}", y0, y0 + size, x0, x0 + size))
        return cells

    def validate(self) -> None:
        if self.canvas_shape != (700, 700):
            raise ValueError("This verification layout is defined for a 700 x 700 canvas.")
        if self.input_aperture.center != self.canvas_center:
            raise ValueError("Input aperture must be centered on the canvas.")
        if int(self.prompt_cell_size) <= 0:
            raise ValueError("prompt_cell_size must be positive.")
        if int(self.prompt_cell_size) % 2 != 0:
            raise ValueError(
                "prompt_cell_size must be even so every cell has an integer-pixel "
                "boundary around the fixed expert center."
            )
        expected_centers = [(200.0, 200.0), (200.0, 500.0), (500.0, 200.0), (500.0, 500.0)]
        actual_centers = [item.center for item in self.experts]
        if actual_centers != expected_centers:
            raise ValueError(
                "Four-expert aperture centers do not match the required geometry: "
                f"{actual_centers}"
            )
        for aperture in [self.input_aperture] + self.experts + self.prompt_cells:
            if aperture.height <= 0 or aperture.width <= 0:
                raise ValueError(f"{aperture.name} has an invalid size.")
            if aperture.y0 < 0 or aperture.x0 < 0:
                raise ValueError(f"{aperture.name} starts outside the canvas.")
            if aperture.y1 > self.canvas_height or aperture.x1 > self.canvas_width:
                raise ValueError(f"{aperture.name} ends outside the canvas.")
        cell_masks = self.prompt_cell_masks()
        if torch.any(cell_masks.sum(dim=0) > 1.0):
            raise ValueError(
                "Prompt cells overlap. Reduce prompt_cell_size or change the "
                "geometry before using spatially partitioned transmission."
            )
        prompt_centers = [item.center for item in self.prompt_cells]
        if prompt_centers != actual_centers:
            raise ValueError("Prompt cell centers must remain equal to expert centers.")

    def physical_grids(
        self,
        pixel_size_m: float,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return y/x coordinate grids in meters with origin at canvas center."""

        cy, cx = self.canvas_center
        y = (torch.arange(self.canvas_height, dtype=torch.float32, device=device) - cy) * float(pixel_size_m)
        x = (torch.arange(self.canvas_width, dtype=torch.float32, device=device) - cx) * float(pixel_size_m)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return y_grid, x_grid

    def aperture_mask(
        self,
        aperture: Aperture,
        device: torch.device = None,
    ) -> torch.Tensor:
        mask = torch.zeros(self.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
        return mask

    def prompt_cell_masks(self, device: torch.device = None) -> torch.Tensor:
        return torch.stack(
            [self.aperture_mask(cell, device=device) for cell in self.prompt_cells],
            dim=0,
        )

    def expert_masks(self, device: torch.device = None) -> torch.Tensor:
        return torch.stack(
            [self.aperture_mask(aperture, device=device) for aperture in self.experts],
            dim=0,
        )

    def expert_union_mask(self, device: torch.device = None) -> torch.Tensor:
        return torch.clamp(self.expert_masks(device=device).sum(dim=0), 0.0, 1.0)

    def cell_offset_pixels(self, index: int) -> Tuple[float, float]:
        cy, cx = self.canvas_center
        cell_y, cell_x = self.prompt_cells[index].center
        return (cell_y - cy, cell_x - cx)

    def cell_offset_meters(self, index: int, pixel_size_m: float) -> Tuple[float, float]:
        offset_y_px, offset_x_px = self.cell_offset_pixels(index)
        return (offset_y_px * float(pixel_size_m), offset_x_px * float(pixel_size_m))

    def to_dict(self) -> Dict:
        return {
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": list(self.canvas_center),
            "input_size": self.input_size,
            "expert_size": self.expert_size,
            "prompt_cell_size": int(self.prompt_cell_size),
            "prompt_fill_factor": (
                4.0
                * float(self.prompt_cell_size) ** 2
                / float(self.canvas_height * self.canvas_width)
            ),
            "gap_pixels": self.gap_pixels,
            "outer_margin": self.outer_margin,
            "input_aperture": self.input_aperture.to_dict(),
            "prompt_cells": [item.to_dict() for item in self.prompt_cells],
            "expert_apertures": [item.to_dict() for item in self.experts],
        }
