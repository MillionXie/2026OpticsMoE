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
    active_size: int = 450
    input_size: int = 120
    image_size: int = 100
    num_experts: int = 9
    expert_size: int = 120
    expert_pitch: int = 150

    @property
    def canvas_center(self): return (self.canvas_size // 2, self.canvas_size // 2)

    @property
    def active_start(self): return (self.canvas_size - self.active_size) // 2

    @property
    def active_aperture(self):
        s = self.active_start
        return Aperture("active450", s, s + self.active_size, s, s + self.active_size)

    @property
    def input_aperture(self):
        cy, cx = self.canvas_center; half = self.input_size // 2
        return Aperture("input120", cy - half, cy - half + self.input_size, cx - half, cx - half + self.input_size)

    @property
    def expert_apertures(self):
        margin = (self.expert_pitch - self.expert_size) // 2
        apertures = []
        for row in range(3):
            for col in range(3):
                y0 = self.active_start + row * self.expert_pitch + margin
                x0 = self.active_start + col * self.expert_pitch + margin
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
        if self.canvas_size != 480 or self.active_size != 450: raise ValueError("This experiment requires canvas=480 and active_size=450.")
        if self.num_experts != 9: raise ValueError("This experiment requires 9 experts.")
        if self.expert_pitch != 150 or self.expert_size != 120: raise ValueError("This experiment requires expert_size=120 and pitch=150.")
        if self.expert_pitch - self.expert_size != 30: raise ValueError("Expert gap must be 30 pixels.")
        if self.active_start != 15: raise ValueError("450 active pixels must be centered with 15-pixel outer padding.")
        expected = [(y, x) for y in (90, 240, 390) for x in (90, 240, 390)]
        if self.expert_centers != expected: raise ValueError(f"Unexpected expert centers: {self.expert_centers}")

    def to_dict(self):
        return {
            "canvas_size": self.canvas_size, "active_size": self.active_size, "outer_padding": self.active_start,
            "image_size": self.image_size, "input_size": self.input_size, "input_padding": (self.input_size-self.image_size)//2,
            "num_experts": self.num_experts, "expert_size": self.expert_size, "expert_pitch": self.expert_pitch,
            "expert_gap": self.expert_pitch-self.expert_size, "expert_centers": [list(v) for v in self.expert_centers],
            "expert_apertures": [v.to_dict() for v in self.expert_apertures],
        }
