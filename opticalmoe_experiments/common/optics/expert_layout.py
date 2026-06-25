from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class Aperture:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "y0": int(self.y0),
            "y1": int(self.y1),
            "x0": int(self.x0),
            "x1": int(self.x1),
            "center": list(self.center),
        }


@dataclass(frozen=True)
class ExpertLayout:
    """Global-router expert layout for 4- or 9-expert OpticalMoE."""

    num_experts: int = 9
    canvas_size: int = 1000
    input_size: int = 134
    expert_size: int = 134
    expert_pitch: int = 200
    padding: int = 200
    prompt_aperture_size: int = 600

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (int(self.canvas_size), int(self.canvas_size))

    @property
    def canvas_center(self) -> Tuple[int, int]:
        return (int(self.canvas_size // 2), int(self.canvas_size // 2))

    @property
    def grid_dim(self) -> int:
        if int(self.num_experts) == 9:
            return 3
        if int(self.num_experts) == 4:
            return 2
        raise ValueError("num_experts must be 4 or 9.")

    @property
    def expert_coords(self) -> List[int]:
        center = self.canvas_center[0]
        if self.grid_dim == 3:
            return [center - self.expert_pitch, center, center + self.expert_pitch]
        return [center - self.expert_pitch // 2, center + self.expert_pitch // 2]

    @property
    def expert_centers(self) -> List[Tuple[int, int]]:
        return [(y, x) for y in self.expert_coords for x in self.expert_coords]

    @property
    def input_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        half = self.input_size // 2
        return Aperture("input", cy - half, cy - half + self.input_size, cx - half, cx - half + self.input_size)

    @property
    def prompt_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        half = self.prompt_aperture_size // 2
        return Aperture("prompt", cy - half, cy + half, cx - half, cx + half)

    @property
    def expert_apertures(self) -> List[Aperture]:
        half = self.expert_size // 2
        apertures = []
        for row, y in enumerate(self.expert_coords):
            for col, x in enumerate(self.expert_coords):
                apertures.append(Aperture(f"E{row}{col}", y - half, y - half + self.expert_size, x - half, x - half + self.expert_size))
        return apertures

    @property
    def expert_union_bounds(self) -> List[int]:
        apertures = self.expert_apertures
        return [
            min(ap.y0 for ap in apertures),
            max(ap.y1 for ap in apertures),
            min(ap.x0 for ap in apertures),
            max(ap.x1 for ap in apertures),
        ]

    @property
    def expert_union_size(self) -> int:
        y0, y1, x0, x1 = self.expert_union_bounds
        return int(max(y1 - y0, x1 - x0))

    @property
    def active_window_size(self) -> int:
        return int(self.prompt_aperture_size)

    @property
    def active_window_aperture(self) -> Aperture:
        cy, cx = self.canvas_center
        half = self.active_window_size // 2
        return Aperture("active_window", cy - half, cy - half + self.active_window_size, cx - half, cx - half + self.active_window_size)

    @property
    def gap_px(self) -> int:
        return int(self.expert_pitch - self.expert_size)

    def validate(self) -> None:
        if self.num_experts not in {4, 9}:
            raise ValueError("num_experts must be 4 or 9.")
        for aperture in [self.input_aperture, self.prompt_aperture] + self.expert_apertures:
            if aperture.y0 < 0 or aperture.x0 < 0 or aperture.y1 > self.canvas_size or aperture.x1 > self.canvas_size:
                raise ValueError(f"Aperture {aperture.name} is outside the canvas.")
        masks = self.expert_masks()
        if torch.any(masks.sum(dim=0) > 1.0):
            raise ValueError("Expert apertures overlap.")
        active = self.active_window_aperture
        if active.y0 < 0 or active.x0 < 0 or active.y1 > self.canvas_size or active.x1 > self.canvas_size:
            raise ValueError("Active optical window is outside the canvas.")
        if self.active_window_size < self.expert_union_size:
            raise ValueError("Active optical window must cover the expert union bounds.")

    def aperture_mask(self, aperture: Aperture, device=None) -> torch.Tensor:
        mask = torch.zeros(self.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = 1.0
        return mask

    def expert_masks(self, device=None) -> torch.Tensor:
        return torch.stack([self.aperture_mask(ap, device=device) for ap in self.expert_apertures], dim=0)

    def expert_union_mask(self, device=None) -> torch.Tensor:
        return self.expert_masks(device=device).sum(dim=0).clamp(0.0, 1.0)

    def prompt_aperture_mask(self, device=None) -> torch.Tensor:
        return self.aperture_mask(self.prompt_aperture, device=device)

    def active_window_mask(self, device=None) -> torch.Tensor:
        return self.aperture_mask(self.active_window_aperture, device=device)

    def physical_grids(self, pixel_size_m: float, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        cy, cx = self.canvas_center
        y = (torch.arange(self.canvas_size, dtype=torch.float32, device=device) - cy) * float(pixel_size_m)
        x = (torch.arange(self.canvas_size, dtype=torch.float32, device=device) - cx) * float(pixel_size_m)
        return torch.meshgrid(y, x, indexing="ij")

    def to_dict(self) -> Dict:
        phase_params = int(self.num_experts) * int(self.expert_size) * int(self.expert_size)
        return {
            "num_experts": int(self.num_experts),
            "grid_dim": int(self.grid_dim),
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": list(self.canvas_center),
            "input_size": int(self.input_size),
            "expert_size": int(self.expert_size),
            "expert_pitch": int(self.expert_pitch),
            "gap_px": int(self.gap_px),
            "padding": int(self.padding),
            "prompt_aperture_size": int(self.prompt_aperture_size),
            "expert_union_bounds": [int(v) for v in self.expert_union_bounds],
            "expert_union_size": int(self.expert_union_size),
            "active_window_size": int(self.active_window_size),
            "input_aperture": self.input_aperture.to_dict(),
            "prompt_aperture": self.prompt_aperture.to_dict(),
            "active_window_aperture": self.active_window_aperture.to_dict(),
            "expert_centers": [list(center) for center in self.expert_centers],
            "expert_apertures": [ap.to_dict() for ap in self.expert_apertures],
            "expert_phase_params_per_layer": phase_params,
            "baseline_4expert_200_phase_params_per_layer": 4 * 200 * 200,
        }
