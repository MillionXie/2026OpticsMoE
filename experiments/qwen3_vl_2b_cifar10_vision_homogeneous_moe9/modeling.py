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
    device: torch.device
    load_time_sec: float


class NormalizedLinearHead(nn.Module):
    """Small, identical teacher/student head: LayerNorm(1024) -> Linear(10)."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(features.float()))

    def specification(self) -> dict[str, Any]:
        return {
            "type": "normalized_linear",
            "feature_dim": self.norm.normalized_shape[0],
            "num_classes": self.classifier.out_features,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }


def build_head(settings: Any, feature_dim: int, num_classes: int) -> NormalizedLinearHead:
    if settings.head_type != "normalized_linear":
        raise ValueError("Only normalized_linear is supported in this deliberately small-head experiment")
    return NormalizedLinearHead(feature_dim, num_classes)


def load_backbone(model_id: str, cache_dir: Path | None, local_files_only: bool, dtype: torch.dtype,
                  device: torch.device, attn_implementation: str, min_pixels: int, max_pixels: int) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    processor_cls = transformers.AutoProcessor
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError("Installed transformers does not support Qwen3-VL")
    common = {"cache_dir": str(cache_dir) if cache_dir else None, "local_files_only": local_files_only}
    processor_kwargs = {key: value for key, value in common.items() if value is not None}
    processor_kwargs.update({"min_pixels": min_pixels, "max_pixels": max_pixels})
    model_kwargs = {key: value for key, value in common.items() if value is not None}
    model_kwargs.update({"dtype": dtype, "low_cpu_mem_usage": True, "attn_implementation": attn_implementation})
    started = time.perf_counter()
    processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    model.to(device).requires_grad_(False).eval()
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)


def module_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)
