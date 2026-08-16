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
    """A compact gated residual head followed by a signed 64-D projection."""

    def __init__(
        self,
        detector_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.detector_dim = int(detector_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.detector_dim)
        self.ffn_in = nn.Linear(self.detector_dim, 2 * self.hidden_dim)
        self.ffn_out = nn.Linear(self.hidden_dim, self.detector_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(self.detector_dim)
        self.projection = nn.Linear(self.detector_dim, self.embedding_dim)

    def forward_unnormalized(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.detector_dim:
            raise RuntimeError(
                f"Electronic features must be [B,{self.detector_dim}], got "
                f"{tuple(features.shape)}"
            )
        value = self.input_norm(features.float())
        gate, content = self.ffn_in(value).chunk(2, dim=-1)
        update = self.ffn_out(F.silu(gate) * content)
        value = value + torch.sigmoid(self.residual_logit) * self.dropout(update)
        return self.projection(self.output_norm(value))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.forward_unnormalized(features)
        if not torch.isfinite(raw).all() or torch.any(raw.norm(dim=-1) <= 1.0e-12):
            raise RuntimeError("Electronic retrieval head produced an invalid embedding")
        return F.normalize(raw, p=2, dim=-1)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "dense_electronic_retrieval_readout",
            "architecture": (
                f"LN({self.detector_dim}) -> gated SwiGLU residual({self.hidden_dim}) "
                f"-> LN -> Linear({self.detector_dim},{self.embedding_dim}) "
                "-> L2Normalize"
            ),
            "detector_dim": self.detector_dim,
            "hidden_dim": self.hidden_dim,
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
            "equation": "Y = X + sigmoid(g) * ElectronicTransformer(X)",
        }

    def student_architecture_report(self) -> dict[str, Any]:
        return {
            "type": "dense_electronic_qwen_replacement",
            "optical_enabled": False,
            "moe_enabled": False,
            "initialization": "independent_random_student",
            "width": self.vision_surrogate.core.width,
            "blocks_per_modality": len(self.vision_surrogate.core.blocks),
            "attention_heads": self.vision_surrogate.core.blocks[0].attention.num_heads,
            "vision_attention": "bidirectional",
            "language_attention": "causal",
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
        settings.electronic_readout_hidden,
        settings.electronic_readout_dropout,
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
