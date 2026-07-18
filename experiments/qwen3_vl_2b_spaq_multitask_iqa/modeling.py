from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LoadedBackbone:
    model: nn.Module
    processor: Any
    load_time_sec: float


class MultitaskRegressionHead(nn.Module):
    """One shared regressor for all four prompt-conditioned tasks.

    Targets are trained on the 0--1 scale. The output intentionally remains
    unbounded during optimization: bounding it with a sigmoid caused the large
    Qwen hidden states to saturate the sigmoid and permanently zero the
    gradients. Evaluation clips the raw prediction to 0--1 before converting
    it back to the original 0--100 score scale.
    """

    SCHEMA_VERSION = 2

    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "shared_multitask_regression_head",
            "schema_version": self.SCHEMA_VERSION,
            "architecture": [
                f"LayerNorm({self.feature_dim})",
                f"Linear({self.feature_dim},{self.hidden_dim})",
                "GELU",
                f"Dropout({self.dropout})",
                f"Linear({self.hidden_dim},1)",
            ],
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "training_target_scale": [0.0, 1.0],
            "output_activation": "none",
            "evaluation_postprocessing": "clamp raw prediction to [0,1], then multiply by 100",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def load_backbone(
    model_id: str,
    cache_dir: Path | None,
    local_files_only: bool,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    processor_cls = transformers.AutoProcessor
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError("Installed transformers does not support Qwen3-VL")
    common: dict[str, Any] = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only,
    }
    processor_kwargs = {key: value for key, value in common.items() if value is not None}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    model_kwargs = {key: value for key, value in common.items() if value is not None}
    model_kwargs.update(
        {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
    )
    started = time.perf_counter()
    processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen backbone must be completely frozen")
    return LoadedBackbone(model=model, processor=processor, load_time_sec=time.perf_counter() - started)


def model_report(model: nn.Module, head: nn.Module, expected_feature_dim: int) -> dict[str, Any]:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", config)
    actual_feature_dim = getattr(text_config, "hidden_size", None)
    if actual_feature_dim is not None and int(actual_feature_dim) != expected_feature_dim:
        raise RuntimeError(
            f"Loaded model text hidden size is {actual_feature_dim}, expected {expected_feature_dim}"
        )
    backbone_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_class": type(model).__name__,
        "backbone_parameters": backbone_parameters,
        "backbone_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "backbone_mode": "eval" if not model.training else "train",
        "full_multimodal_forward": True,
        "generation_used": False,
        "answer_feature": "last non-padding token from final language hidden state",
        "architecture": {
            "vision_depth": getattr(vision_config, "depth", None),
            "vision_hidden_size": getattr(vision_config, "hidden_size", None),
            "text_depth": getattr(text_config, "num_hidden_layers", None),
            "text_hidden_size": actual_feature_dim,
        },
        "regression_head": head.specification() if hasattr(head, "specification") else {},
    }
