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
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True)
class MoEGeometry:
    """Physical geometry for a square homogeneous 224-pixel expert bank.

    Two verified layouts are supported.  MoE16 uses a 4x4, 986-pixel active
    footprint on a 1026 FFT canvas.  The hardware-robust MoE4 variant uses a
    2x2, 478-pixel footprint on a 518 canvas.  Both retain a 30-pixel expert
    gap and a 20-pixel propagation guard band on every side.
    """

    canvas_size: int = 1026
    active_size: int = 986
    expert_size: int = 224
    expert_pitch: int = 254
    num_experts: int = 16
    grid_rows: int = 4
    grid_cols: int = 4

    @property
    def expert_gap(self) -> int:
        return self.expert_pitch - self.expert_size

    @property
    def footprint_height(self) -> int:
        return self.grid_rows * self.expert_size + (self.grid_rows - 1) * self.expert_gap

    @property
    def footprint_width(self) -> int:
        return self.grid_cols * self.expert_size + (self.grid_cols - 1) * self.expert_gap

    @property
    def outer_padding(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def active_start(self) -> int:
        return self.outer_padding

    @property
    def input_aperture(self) -> Aperture:
        start = (self.canvas_size - self.expert_size) // 2
        return Aperture(start, start + self.expert_size, start, start + self.expert_size)

    @property
    def active_aperture(self) -> Aperture:
        start = self.active_start
        return Aperture(start, start + self.active_size, start, start + self.active_size)

    @property
    def detector_aperture(self) -> Aperture:
        """The CCD ROI is the physical 986x986 active area, not the FFT guard band."""

        return self.active_aperture

    @property
    def expert_apertures(self) -> list[Aperture]:
        y_margin = (self.active_size - self.footprint_height) // 2
        x_margin = (self.active_size - self.footprint_width) // 2
        result: list[Aperture] = []
        for row in range(self.grid_rows):
            for column in range(self.grid_cols):
                y0 = self.active_start + y_margin + row * self.expert_pitch
                x0 = self.active_start + x_margin + column * self.expert_pitch
                result.append(Aperture(y0, y0 + self.expert_size, x0, x0 + self.expert_size))
        return result

    def active_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.canvas_size, self.canvas_size)
        aperture = self.active_aperture
        mask[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = 1.0
        return mask

    def validate(self) -> None:
        supported = {
            (1026, 986, 224, 254, 16, 4, 4),
            (518, 478, 224, 254, 4, 2, 2),
        }
        actual = (
            self.canvas_size,
            self.active_size,
            self.expert_size,
            self.expert_pitch,
            self.num_experts,
            self.grid_rows,
            self.grid_cols,
        )
        if actual not in supported:
            raise ValueError(
                "Expected a verified geometry: either MoE16 "
                "canvas1026/active986/expert224/pitch254/16 experts/4x4, "
                "or MoE4 canvas518/active478/expert224/pitch254/4 experts/2x2"
            )
        if self.num_experts != self.grid_rows * self.grid_cols:
            raise ValueError("num_experts must equal grid_rows * grid_cols")
        if self.expert_gap != 30:
            raise ValueError("Adjacent expert apertures must have a 30-pixel gap")
        if self.footprint_height != self.active_size or self.footprint_width != self.active_size:
            raise ValueError("The expert footprint must exactly match the global active area")
        if self.outer_padding != 20:
            raise ValueError("The active area must have a 20-pixel propagation guard band on every side")
