from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.modeling import (
    LoadedVisionBackbone,
    build_duts_model as _build_legacy_duts_model,
    load_vision_backbone,
    preprocess_vision,
)
from experiments.vision2_hybrid_dense.modeling import (
    LesionSegmentationDecoder,
    Vision2HybridDenseStudent,
)


class _NoOpRecombiner(nn.Module):
    """Checkpoint facade; both learned fusion gates live inside the core."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.zeros((), device=device))

    def forward(self, value: torch.Tensor, *_: Any) -> torch.Tensor:
        return value


class _BackboneFacade:
    """Expose the legacy ISIC trainer contract without duplicating modules."""

    def __init__(self, student: Vision2HybridDenseStudent, settings: Any) -> None:
        self._student = student
        self.visual = student.visual
        self.core = student.core
        self.recombiner = _NoOpRecombiner(student.device)
        self.device = student.device
        self.settings = settings

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "architecture": "vision2_hybrid_dense_moe4",
            "specification": self.specification(),
            "core_state_dict": self.core.state_dict(),
            "recombiner_state_dict": self.recombiner.state_dict(),
        }

    def specification(self) -> dict[str, Any]:
        return {
            "type": "vision2_hybrid_dense_moe4",
            "qwen_native_vision_blocks_executed": 0,
            "qwen_patch_position_stem_frozen": True,
            "deepstack_enabled": False,
            "physical_ccd_stages": ["vision_expert", "vision_global"],
            "experts_per_stage": 4,
            "top_k": 2,
            "expert_size": 224,
            "active_size": 478,
            "canvas_size": 518,
            "dense_feature_channels": self.settings.electronic_width,
            "fusion": "electronic + sigmoid(gate) * optical_delta per stage",
            "parameter_breakdown": self.core.parameter_breakdown(),
        }


class ISICVision2HybridModel(nn.Module):
    def __init__(self, loaded: LoadedVisionBackbone, settings: Any) -> None:
        super().__init__()
        self.student = Vision2HybridDenseStudent(
            loaded,
            settings,
            LesionSegmentationDecoder(
                input_dim=settings.electronic_width,
                output_size=settings.image_size,
            ),
        )
        object.__setattr__(
            self, "backbone", _BackboneFacade(self.student, settings)
        )

    @property
    def head(self) -> nn.Module:
        return self.student.head

    @property
    def core(self) -> nn.Module:
        return self.student.core

    def train(self, mode: bool = True) -> "ISICVision2HybridModel":
        super().train(mode)
        self.student.train(mode)
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        detach_backbone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not detach_backbone:
            return self.student(pixel_values, image_grid_thw)
        with torch.no_grad():
            _, spatial, detector = self.student(pixel_values, image_grid_thw)
        return self.head(spatial.detach()), spatial.detach(), detector.detach()

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student.router_losses()

    def operating_loss(self) -> torch.Tensor:
        return self.student.operating_loss()


def build_duts_model(
    loaded: LoadedVisionBackbone,
    settings: Any,
    **kwargs: Any,
) -> nn.Module:
    if getattr(settings, "vision2_hybrid_enabled", False):
        if kwargs.get("checkpoint") is not None:
            raise ValueError(
                "Vision2 hybrid initialization uses the ISIC checkpoint loader"
            )
        return ISICVision2HybridModel(loaded, settings)
    return _build_legacy_duts_model(loaded, settings, **kwargs)


__all__ = [
    "ISICVision2HybridModel",
    "LoadedVisionBackbone",
    "build_duts_model",
    "load_vision_backbone",
    "preprocess_vision",
]
