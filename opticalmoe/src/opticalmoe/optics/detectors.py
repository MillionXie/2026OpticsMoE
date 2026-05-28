import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]


class DetectorArray(nn.Module):
    """Fixed detector regions that sum output-plane intensity."""

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

        masks = self._build_masks()
        self.register_buffer("masks", masks, persistent=False)

    def _centers_line(self) -> list:
        height, width = self.grid_size
        y = height // 2
        xs = torch.linspace(
            self.detector_size // 2,
            width - self.detector_size // 2 - 1,
            steps=self.num_classes,
        )
        return [(int(y), int(round(float(x)))) for x in xs]

    def _centers_grid(self) -> list:
        height, width = self.grid_size
        rows = int(math.ceil(math.sqrt(self.num_classes)))
        cols = int(math.ceil(self.num_classes / rows))
        ys = torch.linspace(
            self.detector_size // 2,
            height - self.detector_size // 2 - 1,
            steps=rows,
        )
        xs = torch.linspace(
            self.detector_size // 2,
            width - self.detector_size // 2 - 1,
            steps=cols,
        )
        centers = []
        for y in ys:
            for x in xs:
                centers.append((int(round(float(y))), int(round(float(x)))))
                if len(centers) == self.num_classes:
                    return centers
        return centers

    def _build_masks(self) -> torch.Tensor:
        if self.layout == "line":
            centers = self._centers_line()
        elif self.layout == "grid":
            centers = self._centers_grid()
        else:
            raise ValueError(f"Unsupported detector layout: {self.layout}")

        height, width = self.grid_size
        masks = torch.zeros(self.num_classes, height, width, dtype=torch.float32)
        half = self.detector_size // 2

        for idx, (cy, cx) in enumerate(centers):
            y0 = max(0, cy - half)
            y1 = min(height, y0 + self.detector_size)
            x0 = max(0, cx - half)
            x1 = min(width, x0 + self.detector_size)
            masks[idx, y0:y1, x0:x1] = 1.0

        return masks

    def get_masks(self) -> torch.Tensor:
        return self.masks

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")

        intensity = torch.abs(field.to(torch.complex64)) ** 2
        energies = torch.einsum("bhw,chw->bc", intensity, self.masks)

        if self.normalize_total_energy:
            total_energy = intensity.sum(dim=(-2, -1), keepdim=False).unsqueeze(1)
            energies = energies / (total_energy + self.eps)

        return energies
