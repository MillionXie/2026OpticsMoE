from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Aperture:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class MoEGeometry:
    canvas_size: int = 480
    active_size: int = 450
    expert_size: int = 120
    expert_pitch: int = 150
    num_experts: int = 9

    @property
    def active_start(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def input_aperture(self) -> Aperture:
        start = (self.canvas_size - self.expert_size) // 2
        return Aperture(start, start + self.expert_size, start, start + self.expert_size)

    @property
    def active_aperture(self) -> Aperture:
        start = self.active_start
        return Aperture(start, start + self.active_size, start, start + self.active_size)

    @property
    def expert_apertures(self) -> list[Aperture]:
        margin = (self.expert_pitch - self.expert_size) // 2
        result = []
        for row in range(3):
            for column in range(3):
                y0 = self.active_start + row * self.expert_pitch + margin
                x0 = self.active_start + column * self.expert_pitch + margin
                result.append(Aperture(y0, y0 + self.expert_size, x0, x0 + self.expert_size))
        return result

    def active_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.canvas_size, self.canvas_size)
        aperture = self.active_aperture
        mask[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = 1.0
        return mask

    def validate(self) -> None:
        if (self.canvas_size, self.active_size, self.expert_size, self.expert_pitch, self.num_experts) != (480, 450, 120, 150, 9):
            raise ValueError("Expected verified 480/450/120/150/9 homogeneous MoE geometry")
