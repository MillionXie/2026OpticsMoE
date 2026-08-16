from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.replacement import (
    DeepStackMultimodalReplacement,
)

from .electronic_blocks import (
    LanguageElectronicReplacement,
    VisionElectronicReplacement,
)


class ElectronicRetrievalReadout(nn.Module):
    """Minimal signed projection from electronic features to retrieval space."""

    def __init__(
        self,
        detector_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.detector_dim = int(detector_dim)
        self.embedding_dim = int(embedding_dim)
        self.norm = nn.LayerNorm(self.detector_dim)
        self.projection = nn.Linear(self.detector_dim, self.embedding_dim)

    def forward_unnormalized(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.detector_dim:
            raise RuntimeError(
                f"Electronic features must be [B,{self.detector_dim}], got "
                f"{tuple(features.shape)}"
            )
        return self.projection(self.norm(features.float()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.forward_unnormalized(features)
        if not torch.isfinite(raw).all() or torch.any(raw.norm(dim=-1) <= 1.0e-12):
            raise RuntimeError("Electronic retrieval head produced an invalid embedding")
        return F.normalize(raw, p=2, dim=-1)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "dense_electronic_retrieval_readout",
            "architecture": (
                f"LN({self.detector_dim}) -> Linear({self.detector_dim},"
                f"{self.embedding_dim}) "
                "-> L2Normalize"
            ),
            "detector_dim": self.detector_dim,
            "embedding_dim": self.embedding_dim,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class ElectronicDeepStackReplacement(DeepStackMultimodalReplacement):
    has_optical_phases = False

    def alignment_specification(self) -> dict[str, Any]:
        return {
            "native_pre_attention_enabled": False,
            "transformer_residual_enabled": True,
            "residual_identity_scale": 1.0,
            "residual_branch_scale": "learned_sigmoid",
            "native_deepstack_visual_indexes": list(
                self.native_deepstack_indexes
            ),
            "student_deepstack_visual_indexes": list(self.deepstack_indexes),
            "student_deepstack_auxiliary_count": len(self.deepstack_indexes),
            "language_replacement_layer_indexes": list(
                self.language_optical_layer_indexes
            ),
            "equation": "Y = X + sigmoid(g) * MLP(LayerNorm(X))",
        }

    def student_architecture_report(self) -> dict[str, Any]:
        return {
            "type": "compact_residual_mlp_qwen_replacement",
            "optical_enabled": False,
            "moe_enabled": False,
            "initialization": "independent_random_student",
            "width": self.vision_surrogate.core.width,
            "blocks_per_modality": len(self.vision_surrogate.core.blocks),
            "attention_enabled": False,
            "token_mixing_enabled": False,
            "mlp_expansion": self.vision_surrogate.core.expansion,
            "language_pooling": "mean_over_valid_multimodal_tokens",
            "learnable_identity_residual_per_modality": True,
            "student_deepstack_visual_indexes": list(self.deepstack_indexes),
            "language_replacement_layer_indexes": list(
                self.language_optical_layer_indexes
            ),
            "alignment": self.alignment_specification(),
        }


def build_electronic_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[ElectronicDeepStackReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionElectronicReplacement(settings.vision_hidden_size, settings).to(
        loaded.device
    )
    language = LanguageElectronicReplacement(settings.text_hidden_size, settings).to(
        loaded.device
    )
    replacement = ElectronicDeepStackReplacement(
        loaded.model, vision, language, settings
    )
    readout = ElectronicRetrievalReadout(
        settings.electronic_width,
        settings.embedding_dim,
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


__all__ = [
    "ElectronicDeepStackReplacement",
    "ElectronicRetrievalReadout",
    "LoadedBackbone",
    "build_electronic_student",
    "load_backbone",
]
