from typing import Dict, Tuple

import torch
import torch.nn as nn

from .detectors import DetectorArray
from .grating import build_aperture_mask, build_expert_aperture_union
from .moe_layout import MoeLayout


class TranslatedDetectorArray(nn.Module):
    """Detector masks built locally in a 600x600 expert plane, then translated.

    The forward pass returns both branch-specific detector energies and several
    normalizations. This keeps the model output as a 10-class classifier while
    preserving diagnostics for left/right expert behavior.
    """

    def __init__(
        self,
        num_classes: int,
        layout: MoeLayout,
        detector_size: int = 32,
        detector_layout: str = "grid",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.layout = layout
        self.detector_size = int(detector_size)
        self.detector_layout = detector_layout
        self.eps = float(eps)

        local_detector = DetectorArray(
            num_classes=num_classes,
            grid_size=layout.expert_size,
            detector_size=detector_size,
            layout=detector_layout,
            normalize_total_energy=False,
            eps=eps,
        )
        local_masks = local_detector.get_masks().detach().clone()
        left_masks = self._translate_masks(local_masks, side="left")
        right_masks = self._translate_masks(local_masks, side="right")
        left_aperture_mask = build_aperture_mask(layout.canvas_shape, layout.left)
        right_aperture_mask = build_aperture_mask(layout.canvas_shape, layout.right)
        union_mask = build_expert_aperture_union(layout.canvas_shape, layout.left, layout.right)

        self.register_buffer("local_masks", local_masks, persistent=False)
        self.register_buffer("left_masks", left_masks, persistent=False)
        self.register_buffer("right_masks", right_masks, persistent=False)
        self.register_buffer("left_aperture_mask", left_aperture_mask, persistent=False)
        self.register_buffer("right_aperture_mask", right_aperture_mask, persistent=False)
        self.register_buffer("aperture_union_mask", union_mask, persistent=False)

    def _translate_masks(self, local_masks: torch.Tensor, side: str) -> torch.Tensor:
        aperture = self.layout.aperture_for_side(side)
        masks = torch.zeros(
            local_masks.shape[0],
            self.layout.canvas_height,
            self.layout.canvas_width,
            dtype=torch.float32,
        )
        masks[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = local_masks
        return masks

    def get_masks(self, side: str = "paired") -> torch.Tensor:
        if side == "left":
            return self.left_masks
        if side == "right":
            return self.right_masks
        if side == "paired":
            return torch.clamp(self.left_masks + self.right_masks, 0.0, 1.0)
        raise ValueError("side must be 'left', 'right', or 'paired'")

    def get_aperture_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.left_aperture_mask, self.right_aperture_mask, self.aperture_union_mask

    def forward(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")

        intensity = torch.abs(field.to(torch.complex64)) ** 2
        left_raw = torch.einsum("bhw,chw->bc", intensity, self.left_masks)
        right_raw = torch.einsum("bhw,chw->bc", intensity, self.right_masks)

        left_aperture_energy = torch.einsum("bhw,hw->b", intensity, self.left_aperture_mask)
        right_aperture_energy = torch.einsum("bhw,hw->b", intensity, self.right_aperture_mask)
        total_energy = intensity.sum(dim=(-2, -1))
        outside_energy = torch.clamp(total_energy - left_aperture_energy - right_aperture_energy, min=0.0)

        left_global_norm = left_raw / (total_energy.unsqueeze(1) + self.eps)
        right_global_norm = right_raw / (total_energy.unsqueeze(1) + self.eps)
        left_local_norm = left_raw / (left_aperture_energy.unsqueeze(1) + self.eps)
        right_local_norm = right_raw / (right_aperture_energy.unsqueeze(1) + self.eps)
        paired_sum_raw = left_raw + right_raw
        paired_sum_global = left_global_norm + right_global_norm

        gate_denominator = left_aperture_energy + right_aperture_energy + self.eps
        gate_left = left_aperture_energy / gate_denominator
        gate_right = right_aperture_energy / gate_denominator
        energy_gated_local = (
            gate_left.unsqueeze(1) * left_local_norm
            + gate_right.unsqueeze(1) * right_local_norm
        )

        return {
            "left_raw": left_raw,
            "right_raw": right_raw,
            "left_aperture_energy": left_aperture_energy,
            "right_aperture_energy": right_aperture_energy,
            "total_energy": total_energy,
            "outside_energy": outside_energy,
            "left_global_norm": left_global_norm,
            "right_global_norm": right_global_norm,
            "left_local_norm": left_local_norm,
            "right_local_norm": right_local_norm,
            "paired_sum_raw": paired_sum_raw,
            "paired_sum_global": paired_sum_global,
            "gate_left": gate_left,
            "gate_right": gate_right,
            "energy_gated_local": energy_gated_local,
        }
