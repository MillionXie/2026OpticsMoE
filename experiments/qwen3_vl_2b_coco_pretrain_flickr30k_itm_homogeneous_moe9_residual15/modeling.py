from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class LoadedBackbone:
    model: nn.Module
    processor: Any
    source: str


class NormalizedBinaryClassificationHead(nn.Module):
    """LayerNorm + scalar linear classifier; forward always returns raw logits."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(features.float())).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(features)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def specification(self) -> dict[str, Any]:
        return {
            "type": "normalized_binary_classification",
            "feature_dim": self.feature_dim,
            "architecture": [f"LayerNorm({self.feature_dim})", f"Linear({self.feature_dim},1)"],
            "output": "raw_logit",
            "parameters": module_parameters(self),
            "trainable_parameters": module_parameters(self, trainable_only=True),
        }


def build_head(settings: Any, feature_dim: int) -> NormalizedBinaryClassificationHead:
    if settings.head_type != "normalized_binary_classification":
        raise ValueError(f"Unsupported binary head_type: {settings.head_type}")
    return NormalizedBinaryClassificationHead(feature_dim)


def load_backbone(model_id: str, cache_dir: Path | None, local_files_only: bool, dtype: torch.dtype,
                  device: torch.device, attn_implementation: str, min_pixels: int,
                  max_pixels: int) -> LoadedBackbone:
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("transformers with Qwen3-VL support is required") from exc
    source = resolve_cached_model_source(model_id, cache_dir) if local_files_only else model_id
    common = {"cache_dir": str(cache_dir) if cache_dir else None, "local_files_only": local_files_only}
    processor = AutoProcessor.from_pretrained(source, min_pixels=min_pixels, max_pixels=max_pixels, **common)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, torch_dtype=dtype, attn_implementation=attn_implementation, **common
    ).to(device)
    model.requires_grad_(False); model.eval()
    return LoadedBackbone(model=model, processor=processor, source=str(source))


def load_processor(model_id: str, cache_dir: Path | None, local_files_only: bool,
                   min_pixels: int, max_pixels: int) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required") from exc
    source = resolve_cached_model_source(model_id, cache_dir) if local_files_only else model_id
    return AutoProcessor.from_pretrained(source, cache_dir=str(cache_dir) if cache_dir else None,
                                         local_files_only=local_files_only,
                                         min_pixels=min_pixels, max_pixels=max_pixels)


def resolve_cached_model_source(model_id: str, cache_dir: Path | None) -> str:
    direct = Path(model_id)
    if direct.is_dir():
        return str(direct)
    roots = [cache_dir] if cache_dir else []
    roots += [Path.home() / ".cache" / "huggingface" / "hub"]
    model_key = "models--" + model_id.replace("/", "--")
    for root in roots:
        if root is None:
            continue
        base = Path(root) / model_key
        refs = base / "refs" / "main"
        if refs.is_file():
            snapshot = base / "snapshots" / refs.read_text(encoding="utf-8").strip()
            if snapshot.is_dir():
                return str(snapshot)
        snapshots = base / "snapshots"
        if snapshots.is_dir():
            choices = sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True)
            if choices:
                return str(choices[0])
    return model_id


def module_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters()
               if not trainable_only or parameter.requires_grad)
