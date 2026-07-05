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


class MLPHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)


def parameter_report(model: nn.Module, head: nn.Module | None = None) -> dict[str, Any]:
    def describe(module: nn.Module | None) -> dict[str, Any]:
        if module is None:
            return {"parameters": 0, "trainable_parameters": 0, "bytes": 0, "gib": 0.0}
        parameters = list(module.parameters())
        count = sum(value.numel() for value in parameters)
        trainable = sum(value.numel() for value in parameters if value.requires_grad)
        size = sum(value.numel() * value.element_size() for value in parameters)
        return {
            "parameters": count,
            "trainable_parameters": trainable,
            "bytes": size,
            "gib": size / 1024**3,
        }

    core = getattr(model, "model", model)
    visual = getattr(core, "visual", None)
    language = getattr(core, "language_model", None)
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", config)
    return {
        "backbone": describe(model),
        "vision": describe(visual),
        "language": describe(language),
        "mlp_head": describe(head),
        "architecture": {
            "vision_hidden_size": getattr(vision_config, "hidden_size", None),
            "vision_depth": getattr(vision_config, "depth", None),
            "vision_out_hidden_size": getattr(vision_config, "out_hidden_size", None),
            "spatial_merge_size": getattr(vision_config, "spatial_merge_size", None),
            "text_hidden_size": getattr(text_config, "hidden_size", None),
            "text_layers": getattr(text_config, "num_hidden_layers", None),
        },
    }

