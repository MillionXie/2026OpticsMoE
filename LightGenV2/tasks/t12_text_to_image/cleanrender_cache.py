"""Cache frozen Qwen text features for the CleanRender manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .dataset import read_manifest, sha256, validate_split_contract
from .feature_cache import _encode_caption_rows


def build_cleanrender_text_cache(
    data_dir: Path,
    qwen_checkpoint: Path,
    device: torch.device,
    *,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    contract = validate_split_contract(data_dir)
    cache_dir = data_dir / "qwen_text_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = [cache_dir / f"{split}.pt" for split in ("train", "val", "test")]
    if all(path.is_file() for path in expected) and not force:
        return contract
    if any(path.exists() for path in expected) and not force:
        raise FileExistsError("Partial CleanRender cache exists; inspect it or pass force=True")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    for split, output in zip(("train", "val", "test"), expected):
        rows = read_manifest(data_dir / f"{split}.jsonl")
        features, unique = _encode_caption_rows(
            model, processor, [row.caption for row in rows], device, batch_size, split
        )
        payload = {
            "meta": {
                "schema_version": 1,
                "split": split,
                "manifest_sha256": sha256(data_dir / f"{split}.jsonl"),
                "qwen": str(qwen_checkpoint.resolve()),
                "qwen_pooling": "masked mean of final native hidden state",
                "unique_captions": unique,
                "text_dim": int(features.shape[1]),
            },
            "sample_ids": [row.sample_id for row in rows],
            "categories": [row.category for row in rows],
            "text": features.to(torch.bfloat16),
        }
        temporary = output.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        output.with_suffix(".json").write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return contract


__all__ = ["build_cleanrender_text_cache"]
