from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Aperture:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def size(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class MoEGeometry:
    canvas_size: int
    active_size: int
    expert_size: int
    expert_pitch: int
    num_experts: int = 9

    @property
    def active_start(self) -> int:
        return (self.canvas_size - self.active_size) // 2

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

    @property
    def detector_aperture(self) -> Aperture:
        # With the symmetric 3x3 layout this is both the centered 224x224 CCD
        # readout and the exact spatial footprint of expert index four.
        return self.expert_apertures[4]

    def active_mask(self, *, device=None) -> torch.Tensor:
        output = torch.zeros(self.canvas_size, self.canvas_size, device=device)
        aperture = self.active_aperture
        output[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = 1
        return output

    def validate(self) -> None:
        if self.num_experts != 9:
            raise ValueError("MoE9 requires nine experts")
        if self.expert_pitch <= self.expert_size:
            raise ValueError("expert_pitch must exceed expert_size")
        if self.expert_pitch - self.expert_size != 30:
            raise ValueError("Expert gap must be exactly 30 pixels")
        if self.active_size != 3 * self.expert_pitch:
            raise ValueError("active_size must equal 3*expert_pitch")
        if self.canvas_size < self.active_size:
            raise ValueError("canvas_size must cover active_size")
        if (self.canvas_size - self.active_size) % 2:
            raise ValueError("Canvas padding must be symmetric")
        apertures = self.expert_apertures
        for aperture in apertures:
            if min(aperture.y0, aperture.x0) < 0:
                raise ValueError("Expert aperture starts outside canvas")
            if max(aperture.y1, aperture.x1) > self.canvas_size:
                raise ValueError("Expert aperture ends outside canvas")
        if self.detector_aperture.size != self.expert_size:
            raise ValueError("Central detector ROI does not match expert size")

    def report(self) -> dict:
        return {
            "canvas_size": self.canvas_size,
            "active_size": self.active_size,
            "outer_padding_per_side": self.active_start,
            "expert_size": self.expert_size,
            "expert_pitch": self.expert_pitch,
            "expert_gap": self.expert_pitch - self.expert_size,
            "num_experts": self.num_experts,
            "expert_apertures": [
                {"y0": item.y0, "y1": item.y1, "x0": item.x0, "x1": item.x1}
                for item in self.expert_apertures
            ],
            "detector_aperture": {
                "y0": self.detector_aperture.y0,
                "y1": self.detector_aperture.y1,
                "x0": self.detector_aperture.x0,
                "x1": self.detector_aperture.x1,
            },
        }
