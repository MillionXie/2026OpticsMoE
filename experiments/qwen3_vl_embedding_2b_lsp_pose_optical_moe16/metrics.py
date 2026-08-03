from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .datasets import JOINT_NAMES
from .losses import hardargmax_coordinates


@dataclass
class PoseMetricAccumulator:
    pck_threshold: float = 0.2
    pckh_threshold: float = 0.5
    errors: list[float] = field(default_factory=list)
    normalized_errors: list[float] = field(default_factory=list)
    pck_hits: int = 0
    pck_total: int = 0
    pckh_hits: int = 0
    pckh_total: int = 0
    joint_hits: list[int] = field(default_factory=lambda: [0] * len(JOINT_NAMES))
    joint_totals: list[int] = field(default_factory=lambda: [0] * len(JOINT_NAMES))

    def update(
        self,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        visible: torch.Tensor,
        torso_scales: torch.Tensor,
        head_scales: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        predicted = hardargmax_coordinates(heatmaps, image_size).detach().cpu()
        targets = targets.detach().cpu()
        visible = visible.detach().cpu().bool()
        torso_scales = torso_scales.detach().cpu()
        head_scales = head_scales.detach().cpu()
        distances = torch.linalg.vector_norm(
            predicted - torch.nan_to_num(targets, nan=0.0), dim=-1,
        )
        for sample in range(len(predicted)):
            torso = float(torso_scales[sample])
            head = float(head_scales[sample])
            for joint in range(len(JOINT_NAMES)):
                if not bool(visible[sample, joint]):
                    continue
                error = float(distances[sample, joint])
                self.errors.append(error)
                if math.isfinite(torso) and torso > 0:
                    hit = error <= self.pck_threshold * torso
                    self.pck_hits += int(hit)
                    self.pck_total += 1
                    self.joint_hits[joint] += int(hit)
                    self.joint_totals[joint] += 1
                    self.normalized_errors.append(error / torso)
                if math.isfinite(head) and head > 0:
                    self.pckh_hits += int(error <= self.pckh_threshold * head)
                    self.pckh_total += 1
        return predicted

    def compute(self) -> dict[str, Any]:
        mean = lambda values: float(sum(values) / len(values)) if values else None
        return {
            "pck_at_0.2_torso": self.pck_hits / max(self.pck_total, 1),
            "pckh_at_0.5_head": self.pckh_hits / max(self.pckh_total, 1),
            "mean_pixel_error": mean(self.errors),
            "normalized_mean_error_torso": mean(self.normalized_errors),
            "evaluated_joints": len(self.errors),
            "pck_evaluated_joints": self.pck_total,
            "pckh_evaluated_joints": self.pckh_total,
            "per_joint_pck_at_0.2_torso": {
                name: self.joint_hits[index] / max(self.joint_totals[index], 1)
                for index, name in enumerate(JOINT_NAMES)
            },
            "per_joint_counts": {
                name: self.joint_totals[index]
                for index, name in enumerate(JOINT_NAMES)
            },
            "normalization": {
                "PCK": "mean distance of the two shoulder-to-opposite-hip torso diagonals",
                "PCKh": "2 * distance(neck, head_top)",
            },
        }
