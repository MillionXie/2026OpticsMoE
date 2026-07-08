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


class VisionFieldProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, head_type: str,
                 hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if head_type not in {"linear", "mlp", "bottleneck"}:
            raise ValueError("head_type must be linear, mlp, or bottleneck")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.head_type = str(head_type)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        if head_type == "linear":
            self.network = nn.Linear(input_dim, num_classes)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def specification(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "input_dim": self.input_dim, "head_type": self.head_type,
            "hidden_dim": None if self.head_type == "linear" else self.hidden_dim,
            "dropout": 0.0 if self.head_type == "linear" else self.dropout,
            "num_classes": self.num_classes, "parameters": parameters,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def load_backbone(model_id: str, cache_dir: Path | None, local_files_only: bool,
                  dtype: torch.dtype, device: torch.device, attn_implementation: str,
                  min_pixels: int, max_pixels: int) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    processor_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only, "min_pixels": min_pixels, "max_pixels": max_pixels,
    }
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only, "dtype": dtype,
        "low_cpu_mem_usage": True, "attn_implementation": attn_implementation,
    }
    if cache_dir is not None:
        processor_kwargs["cache_dir"] = str(cache_dir)
        model_kwargs["cache_dir"] = str(cache_dir)
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError("Installed transformers does not support Qwen3-VL")
    started = time.perf_counter()
    processor = transformers.AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = model_cls.from_pretrained(model_id, **model_kwargs).to(device)
    model.requires_grad_(False)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)

