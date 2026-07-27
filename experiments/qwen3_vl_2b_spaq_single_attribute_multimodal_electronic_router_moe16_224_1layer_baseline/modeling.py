from __future__ import annotations

import importlib
import os
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


class NormalizedLinearRegressionHead(nn.Module):
    """Identical teacher/student attribute head: LayerNorm(D) -> Linear(D,1)."""

    def __init__(self, feature_dim: int, output_activation: str = "linear") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.regressor = nn.Linear(feature_dim, 1)
        self.output_activation = output_activation

    def forward_raw(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.norm(features.float())).squeeze(-1)

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value) if self.output_activation == "sigmoid" else value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activate(self.forward_raw(features))

    def specification(self) -> dict[str, Any]:
        return {
            "type": "normalized_linear_regression",
            "feature_dim": self.norm.normalized_shape[0],
            "output_dim": 1,
            "output_activation": self.output_activation,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }


def build_head(settings: Any, feature_dim: int) -> NormalizedLinearRegressionHead:
    if settings.head_type != "normalized_linear_regression":
        raise ValueError("Only normalized_linear_regression is supported")
    head = NormalizedLinearRegressionHead(feature_dim, settings.head_output_activation)
    head.task_name = settings.task_name
    return head


def load_backbone(model_id: str, cache_dir: Path | None, local_files_only: bool, dtype: torch.dtype,
                  device: torch.device, attn_implementation: str, min_pixels: int, max_pixels: int) -> LoadedBackbone:
    transformers = importlib.import_module("transformers")
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError("Installed transformers does not support Qwen3-VL")
    source = resolve_cached_model_source(model_id, cache_dir)
    using_local_snapshot = source != model_id
    common = {"cache_dir": str(cache_dir) if cache_dir else None,
              "local_files_only": local_files_only or using_local_snapshot}
    model_kwargs = {key: value for key, value in common.items() if value is not None}
    model_kwargs.update({"dtype": dtype, "low_cpu_mem_usage": True, "attn_implementation": attn_implementation})
    started = time.perf_counter()
    processor = load_processor(model_id, cache_dir, local_files_only, min_pixels, max_pixels)
    model = model_cls.from_pretrained(source, **model_kwargs)
    model.to(device).requires_grad_(False).eval()
    return LoadedBackbone(model, processor, device, time.perf_counter() - started)


def load_processor(model_id: str, cache_dir: Path | None, local_files_only: bool,
                   min_pixels: int, max_pixels: int) -> Any:
    transformers = importlib.import_module("transformers")
    source = resolve_cached_model_source(model_id, cache_dir)
    using_local_snapshot = source != model_id
    kwargs = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only or using_local_snapshot,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    return transformers.AutoProcessor.from_pretrained(
        source, **{key: value for key, value in kwargs.items() if value is not None}
    )


def resolve_cached_model_source(model_id: str, cache_dir: Path | None) -> str:
    """Prefer a complete local HF snapshot without changing cache metadata.

    Passing a repository id to some tokenizer versions still performs network
    metadata requests even when all weights are cached.  Loading the resolved
    snapshot directory avoids those requests, while Settings.model_id remains
    the portable repository id used by teacher-cache validation.
    """
    if Path(model_id).is_dir() or "/" not in model_id:
        return model_id
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(Path(cache_dir))
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    repository = "models--" + model_id.replace("/", "--")
    for root in dict.fromkeys(path.resolve() for path in roots):
        directory = root / repository
        snapshots = directory / "snapshots"
        if not snapshots.is_dir():
            continue
        preferred: list[Path] = []
        main_ref = directory / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                preferred.append(snapshots / revision)
        preferred.extend(sorted((path for path in snapshots.iterdir() if path.is_dir()),
                                key=lambda path: path.stat().st_mtime, reverse=True))
        for snapshot in preferred:
            if (snapshot / "config.json").is_file() and (snapshot / "preprocessor_config.json").is_file():
                return str(snapshot.resolve())
    return model_id


def module_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)
