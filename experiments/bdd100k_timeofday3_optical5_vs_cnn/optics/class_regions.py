from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class ClassRegionDetector(nn.Module):
    """Fixed, non-overlapping class regions on the final optical detector plane."""

    def __init__(
        self,
        field_size: int,
        class_names: Sequence[str],
        region_size: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.field_size = int(field_size)
        self.class_names = list(class_names)
        self.region_size = int(region_size)
        self.temperature = float(temperature)
        self.eps = float(eps)
        if len(self.class_names) < 2:
            raise ValueError("ClassRegionDetector requires at least two classes")
        if not 0 < self.region_size <= self.field_size:
            raise ValueError("detector region_size must satisfy 0 < region_size <= field_size")
        if self.temperature <= 0:
            raise ValueError("detector region temperature must be positive")

        masks = torch.zeros(len(self.class_names), self.field_size, self.field_size)
        boxes: list[dict[str, Any]] = []
        center_y = self.field_size / 2.0
        for index, name in enumerate(self.class_names):
            center_x = self.field_size * (index + 1) / (len(self.class_names) + 1)
            x0 = int(round(center_x - self.region_size / 2.0))
            y0 = int(round(center_y - self.region_size / 2.0))
            x0 = max(0, min(x0, self.field_size - self.region_size))
            y0 = max(0, min(y0, self.field_size - self.region_size))
            x1 = x0 + self.region_size
            y1 = y0 + self.region_size
            masks[index, y0:y1, x0:x1] = 1.0
            boxes.append(
                {
                    "class_index": index,
                    "class_name": name,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": self.region_size,
                    "height": self.region_size,
                }
            )
        if torch.any(masks.sum(dim=0) > 1):
            raise ValueError(
                "Detector class regions overlap; reduce detector_region_size for the field size"
            )
        self.register_buffer("region_masks", masks, persistent=True)
        self.boxes = boxes

    def forward(self, intensity: torch.Tensor) -> dict[str, torch.Tensor]:
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (
            self.field_size,
            self.field_size,
        ):
            raise ValueError(
                f"ClassRegionDetector expects [B,{self.field_size},{self.field_size}]"
            )
        value = intensity.float().clamp_min(0.0)
        region_energy = torch.einsum("bhw,khw->bk", value, self.region_masks.float())
        total_energy = value.sum(dim=(-2, -1)).clamp_min(self.eps)
        region_fractions = region_energy / total_energy.unsqueeze(1)
        detector_fraction = region_fractions.sum(dim=1).clamp(max=1.0)
        region_distribution = region_energy / region_energy.sum(dim=1, keepdim=True).clamp_min(
            self.eps
        )
        region_logits = torch.log(region_distribution.clamp_min(self.eps)) / self.temperature
        return {
            "region_energy": region_energy,
            "region_fractions": region_fractions,
            "region_distribution": region_distribution,
            "region_logits": region_logits,
            "detector_fraction": detector_fraction,
            "outside_fraction": (1.0 - detector_fraction).clamp_min(0.0),
        }

    def specification(self) -> dict[str, Any]:
        return {
            "layout": "horizontal_center",
            "field_size": self.field_size,
            "region_size": self.region_size,
            "temperature": self.temperature,
            "class_order": list(self.class_names),
            "boxes": list(self.boxes),
        }
