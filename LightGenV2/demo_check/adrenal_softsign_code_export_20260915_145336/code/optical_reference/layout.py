import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Aperture:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def center(self):
        return ((self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2)

    def to_dict(self):
        return {"name": self.name, "bounds_yx": [self.y0, self.y1, self.x0, self.x1], "center_yx": list(self.center)}


@dataclass(frozen=True)
class MoELayout:
    canvas_size: int = 480
    active_size: int = 478
    input_size: int = 120
    image_size: int = 100
    num_experts: int = 4
    expert_size: int = 224
    expert_pitch: int = 254

    @property
    def expert_grid_size(self):
        return math.isqrt(self.num_experts)

    @property
    def canvas_center(self): return (self.canvas_size // 2, self.canvas_size // 2)

    @property
    def active_start(self): return (self.canvas_size - self.active_size) // 2

    @property
    def active_aperture(self):
        s = self.active_start
        return Aperture(f"active{self.active_size}", s, s + self.active_size, s, s + self.active_size)

    @property
    def input_aperture(self):
        cy, cx = self.canvas_center; half = self.input_size // 2
        return Aperture(f"input{self.input_size}", cy - half, cy - half + self.input_size, cx - half, cx - half + self.input_size)

    @property
    def expert_apertures(self):
        apertures = []
        for row in range(self.expert_grid_size):
            for col in range(self.expert_grid_size):
                y0 = self.active_start + row * self.expert_pitch
                x0 = self.active_start + col * self.expert_pitch
                apertures.append(Aperture(f"E{row}{col}", y0, y0 + self.expert_size, x0, x0 + self.expert_size))
        return apertures

    @property
    def expert_centers(self): return [item.center for item in self.expert_apertures]

    def aperture_mask(self, aperture, device=None):
        mask = torch.zeros(self.canvas_size, self.canvas_size, dtype=torch.float32, device=device)
        mask[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = 1.0
        return mask

    def active_mask(self, device=None): return self.aperture_mask(self.active_aperture, device)

    def expert_masks(self, device=None): return torch.stack([self.aperture_mask(item, device) for item in self.expert_apertures])

    def expert_union_mask(self, device=None): return self.expert_masks(device).sum(0).clamp(0.0, 1.0)

    def validate(self):
        if self.canvas_size <= 0 or self.active_size <= 0 or self.active_size > self.canvas_size:
            raise ValueError("canvas_size and active_size must be positive with active_size <= canvas_size.")
        if self.expert_grid_size**2 != self.num_experts:
            raise ValueError("num_experts must form a square expert grid.")
        if self.expert_size <= 0 or self.expert_pitch < self.expert_size:
            raise ValueError("expert_size must be positive and expert_pitch must be at least expert_size.")
        occupied_size = self.expert_grid_size * self.expert_size + (self.expert_grid_size - 1) * (self.expert_pitch - self.expert_size)
        if occupied_size != self.active_size:
            raise ValueError("The expert grid must exactly span the square global mask.")
        if (self.canvas_size - self.active_size) % 2 != 0:
            raise ValueError("The active expert grid must be symmetrically centered on the canvas.")
        for aperture in self.expert_apertures:
            if aperture.y0 < 0 or aperture.x0 < 0 or aperture.y1 > self.canvas_size or aperture.x1 > self.canvas_size:
                raise ValueError(f"Expert aperture falls outside the canvas: {aperture}")

    def to_dict(self):
        return {
            "canvas_size": self.canvas_size, "active_size": self.active_size, "outer_padding": self.active_start,
            "image_size": self.image_size, "input_size": self.input_size, "input_padding": (self.input_size-self.image_size)//2,
            "num_experts": self.num_experts, "expert_grid_size": self.expert_grid_size,
            "expert_size": self.expert_size, "expert_pitch": self.expert_pitch,
            "expert_gap": self.expert_pitch-self.expert_size, "expert_centers": [list(v) for v in self.expert_centers],
            "expert_apertures": [v.to_dict() for v in self.expert_apertures],
        }
