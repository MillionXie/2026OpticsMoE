from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
)
from .optics.hybrid import RobustLanguageOpticalMoE, RobustVisionOpticalMoE
from .optics.replacement import RobustDeepStackMultimodalReplacement


class RobustOpticalRetrievalReadout(nn.Module):
    """Small gated bottleneck before the signed 64-D retrieval projection."""

    def __init__(
        self,
        detector_dim: int,
        embedding_dim: int = 64,
        bottleneck_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.detector_dim = int(detector_dim)
        self.embedding_dim = int(embedding_dim)
        self.norm = nn.LayerNorm(self.detector_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.detector_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, self.detector_dim),
        )
        self.bottleneck_gate = nn.Parameter(torch.tensor(-2.0))
        self.post_norm = nn.LayerNorm(self.detector_dim)
        self.projection = nn.Linear(self.detector_dim, self.embedding_dim)

    def forward_unnormalized(self, detector_features: torch.Tensor) -> torch.Tensor:
        if detector_features.ndim != 2 or detector_features.shape[-1] != self.detector_dim:
            raise RuntimeError(
                f"Detector features must be [B,{self.detector_dim}], got "
                f"{tuple(detector_features.shape)}"
            )
        value = self.norm(detector_features.float())
        value = value + torch.sigmoid(self.bottleneck_gate) * self.bottleneck(value)
        return self.projection(self.post_norm(value))

    def forward(self, detector_features: torch.Tensor) -> torch.Tensor:
        raw = self.forward_unnormalized(detector_features)
        if not torch.isfinite(raw).all() or torch.any(raw.norm(dim=-1) <= 1.0e-12):
            raise RuntimeError("Robust retrieval readout produced an invalid embedding")
        return F.normalize(raw, p=2, dim=-1)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "robust_hybrid_retrieval_readout",
            "architecture": (
                f"LN({self.detector_dim}) -> gated bottleneck residual -> LN -> "
                f"Linear({self.detector_dim},{self.embedding_dim}) -> L2Normalize"
            ),
            "detector_dim": self.detector_dim,
            "embedding_dim": self.embedding_dim,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def build_optical_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[RobustDeepStackMultimodalReplacement, RobustOpticalRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = RobustVisionOpticalMoE(settings.vision_hidden_size, settings).to(
        loaded.device
    )
    language = RobustLanguageOpticalMoE(settings.text_hidden_size, settings).to(
        loaded.device
    )
    replacement = RobustDeepStackMultimodalReplacement(
        loaded.model, vision, language, settings
    )
    readout = RobustOpticalRetrievalReadout(
        settings.detector_output_size,
        settings.embedding_dim,
        settings.readout_bottleneck_dim,
        settings.readout_dropout,
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def unique_trainable_parameters(
    replacement: RobustDeepStackMultimodalReplacement,
    readout: RobustOpticalRetrievalReadout,
) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in [*replacement.trainable_parameters(), *readout.parameters()]:
        if parameter.requires_grad and id(parameter) not in seen:
            values.append(parameter)
            seen.add(id(parameter))
    return values


def trainable_parameter_report(
    model: nn.Module,
    replacement: RobustDeepStackMultimodalReplacement,
    readout: RobustOpticalRetrievalReadout,
) -> dict[str, Any]:
    vision = replacement.vision_surrogate.parameter_breakdown()
    language = replacement.language_surrogate.parameter_breakdown()
    parameters = unique_trainable_parameters(replacement, readout)
    return {
        "teacher_model_id": getattr(model, "name_or_path", type(model).__name__),
        "teacher_parameters_frozen": True,
        "student_architecture": {
            "optical_structure": "expert_phase_then_global_phase",
            "optical_phase_planes_per_modality": 2,
            "learnable_residual_fusions_per_modality": 2,
            "vision_tap_stages": list(replacement.vision_surrogate.tap_stages),
            "student_deepstack_visual_indexes": list(replacement.deepstack_indexes),
            "language_optical_layer_indexes": list(
                replacement.language_optical_layer_indexes
            ),
            "alignment": replacement.alignment_specification(),
        },
        "vision_optical": vision,
        "language_optical": language,
        "retrieval_readout": readout.specification(),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "trainable_tensors": len(parameters),
    }
