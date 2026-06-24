import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]


class DetectorArray(nn.Module):
    def __init__(
        self,
        num_classes: int,
        grid_size: GridSize,
        detector_size: int = 32,
        layout: str = "grid",
        normalize_total_energy: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        self.num_classes = int(num_classes)
        self.grid_size = (int(height), int(width))
        self.detector_size = int(detector_size)
        self.layout = layout
        self.normalize_total_energy = bool(normalize_total_energy)
        self.eps = float(eps)
        self.register_buffer("masks", self._build_masks(), persistent=False)

    def _centers(self):
        height, width = self.grid_size
        if self.layout == "line":
            xs = torch.linspace(self.detector_size // 2, width - self.detector_size // 2 - 1, steps=self.num_classes)
            return [(height // 2, int(round(float(x)))) for x in xs]
        if self.layout != "grid":
            raise ValueError(f"Unsupported detector layout: {self.layout}")
        rows = int(math.ceil(math.sqrt(self.num_classes)))
        cols = int(math.ceil(float(self.num_classes) / rows))
        ys = torch.linspace(self.detector_size // 2, height - self.detector_size // 2 - 1, steps=rows)
        xs = torch.linspace(self.detector_size // 2, width - self.detector_size // 2 - 1, steps=cols)
        centers = []
        for y in ys:
            for x in xs:
                centers.append((int(round(float(y))), int(round(float(x)))))
                if len(centers) == self.num_classes:
                    return centers
        return centers

    def _build_masks(self) -> torch.Tensor:
        height, width = self.grid_size
        masks = torch.zeros(self.num_classes, height, width, dtype=torch.float32)
        half = self.detector_size // 2
        for index, (cy, cx) in enumerate(self._centers()):
            y0 = max(0, cy - half)
            x0 = max(0, cx - half)
            masks[index, y0:min(height, y0 + self.detector_size), x0:min(width, x0 + self.detector_size)] = 1.0
        return masks

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = torch.abs(field.to(torch.complex64)).square()
        energies = torch.einsum("bhw,chw->bc", intensity, self.masks)
        if self.normalize_total_energy:
            energies = energies / (intensity.sum(dim=(-2, -1), keepdim=False).unsqueeze(1) + self.eps)
        return energies

