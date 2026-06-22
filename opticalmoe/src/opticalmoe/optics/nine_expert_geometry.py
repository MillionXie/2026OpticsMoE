from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .four_expert_geometry import Aperture


@dataclass(frozen=True)
class NineExpertFair134Layout:
    """Fixed 9-expert fair134 layout for the AS global-router experiment.

    Coordinates are integer pixel slices. The prompt aperture is the physical
    center 600 x 600 active prompt area; the outer region is padding with zero
    prompt transmission.
    """

    canvas_height: int = 1000
    canvas_width: int = 1000
    input_size: int = 134
    expert_size: int = 134
    expert_pitch: int = 200
    padding: int = 200
    prompt_aperture_size: int = 600

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (int(self.canvas_height), int(self.canvas_width))

    @property
    def canvas_center(self) -> Tuple[int, int]:
        return (int(self.canvas_height // 2), int(self.canvas_width // 2))

    @property
    def gap_px(self) -> int:
        return int(self.expert_pitch - self.expert_size)

    @property
    def half_gap_px(self) -> float:
        return float(self.gap_px) / 2.0

    @property
    def expert_coords(self) -> List[int]:
        first = int(self.padding + self.expert_pitch // 2)
        return [first + index * int(self.expert_pitch) for index in range(3)]

    @property
    def expert_centers(self) -> List[Tuple[int, int]]:
        return [(y, x) for y in self.expert_coords for x in self.expert_coords]

    @property
    def input_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        half = int(self.input_size // 2)
        return Aperture(
            "input",
            cy - half,
            cy - half + int(self.input_size),
            cx - half,
            cx - half + int(self.input_size),
        )

    @property
    def expert_apertures(self) -> List[Aperture]:
        half = int(self.expert_size // 2)
        apertures = []
        for row, y in enumerate(self.expert_coords):
            for col, x in enumerate(self.expert_coords):
                apertures.append(
                    Aperture(
                        f"E{row}{col}",
                        y - half,
                        y - half + int(self.expert_size),
                        x - half,
                        x - half + int(self.expert_size),
                    )
                )
        return apertures

    @property
    def experts(self) -> List[Aperture]:
        """Alias used by existing generic visualization/reporting helpers."""

        return self.expert_apertures

    @property
    def prompt_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        half = int(self.prompt_aperture_size // 2)
        return Aperture(
            "prompt",
            cy - half,
            cy - half + int(self.prompt_aperture_size),
            cx - half,
            cx - half + int(self.prompt_aperture_size),
        )

    def validate(self) -> None:
        if self.canvas_shape != (1000, 1000):
            raise ValueError(f"Expected 1000 x 1000 canvas, got {self.canvas_shape}.")
        if self.canvas_center != (500, 500):
            raise ValueError(f"Expected canvas center (500,500), got {self.canvas_center}.")
        if int(self.input_size) != 134:
            raise ValueError("fair134 layout requires input_size=134.")
        if int(self.expert_size) != 134:
            raise ValueError("fair134 layout requires expert_size=134.")
        if int(self.expert_pitch) != 200:
            raise ValueError("fair134 layout requires expert_pitch=200.")
        if int(self.padding) != 200:
            raise ValueError("fair134 1000 layout requires padding=200.")
        if self.expert_coords != [300, 500, 700]:
            raise ValueError(f"Unexpected expert coords: {self.expert_coords}")
        if self.gap_px != 66:
            raise ValueError(f"Expected expert gap 66 px, got {self.gap_px}.")
        p = self.prompt_aperture
        if (p.y0, p.y1, p.x0, p.x1) != (200, 800, 200, 800):
            raise ValueError(f"Expected prompt aperture 200:800, got {p}.")
        if self.input_aperture.center != self.canvas_center:
            raise ValueError("Input aperture must be centered.")
        for aperture in [self.input_aperture, self.prompt_aperture] + self.expert_apertures:
            if aperture.y0 < 0 or aperture.x0 < 0:
                raise ValueError(f"{aperture.name} starts outside the canvas.")
            if aperture.y1 > self.canvas_height or aperture.x1 > self.canvas_width:
                raise ValueError(f"{aperture.name} ends outside the canvas.")
        masks = self.expert_masks()
        if torch.any(masks.sum(dim=0) > 1.0):
            raise ValueError("Expert masks overlap.")

    def aperture_mask(self, aperture: Aperture, device=None) -> torch.Tensor:
        mask = torch.zeros(self.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
        return mask

    def expert_masks(self, device=None) -> torch.Tensor:
        return torch.stack(
            [self.aperture_mask(aperture, device=device) for aperture in self.expert_apertures],
            dim=0,
        )

    def expert_union_mask(self, device=None) -> torch.Tensor:
        return torch.clamp(self.expert_masks(device=device).sum(dim=0), 0.0, 1.0)

    def prompt_aperture_mask(self, device=None) -> torch.Tensor:
        return self.aperture_mask(self.prompt_aperture, device=device)

    def physical_grids(self, pixel_size_m: float, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return y/x coordinate grids in meters with origin at canvas center."""

        cy, cx = self.canvas_center
        y = (torch.arange(self.canvas_height, dtype=torch.float32, device=device) - cy) * float(pixel_size_m)
        x = (torch.arange(self.canvas_width, dtype=torch.float32, device=device) - cx) * float(pixel_size_m)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return y_grid, x_grid

    def to_dict(self) -> Dict:
        total_9 = 9 * int(self.expert_size) * int(self.expert_size)
        baseline_4 = 4 * 200 * 200
        return {
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": list(self.canvas_center),
            "input_size": int(self.input_size),
            "expert_size": int(self.expert_size),
            "expert_pitch": int(self.expert_pitch),
            "gap_px": self.gap_px,
            "half_gap_px": self.half_gap_px,
            "padding": int(self.padding),
            "prompt_aperture_size": int(self.prompt_aperture_size),
            "prompt_aperture": self.prompt_aperture.to_dict(),
            "input_aperture": self.input_aperture.to_dict(),
            "expert_centers": [list(center) for center in self.expert_centers],
            "expert_apertures": [aperture.to_dict() for aperture in self.expert_apertures],
            "total_9expert_phase_params_per_layer": total_9,
            "baseline_4expert_phase_params_per_layer": baseline_4,
            "relative_param_diff": total_9 / baseline_4 - 1.0,
        }
