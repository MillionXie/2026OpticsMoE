"""Cache four frozen-Qwen style prompt embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .chair_style_transfer import STYLE_NAMES, STYLE_PROMPTS
from .feature_cache import _encode_caption_rows


def build_style_text_cache(
    output: Path, qwen_checkpoint: Path, device: torch.device, *, force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    features, _ = _encode_caption_rows(model, processor, list(STYLE_PROMPTS), device, 4, "styles")
    payload = {
        "meta": {
            "schema_version": 1, "qwen": str(qwen_checkpoint.resolve()),
            "pooling": "masked mean of final native hidden state", "text_dim": int(features.shape[1]),
        },
        "style_names": list(STYLE_NAMES), "prompts": list(STYLE_PROMPTS),
        "text": features.to(torch.bfloat16).cpu(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload["meta"]


__all__ = ["build_style_text_cache"]
