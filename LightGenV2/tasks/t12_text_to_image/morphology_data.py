"""Teacher-generated, full-frame product morphology editing data.

Every target is decoded by an image-editing teacher.  No foreground mask,
pixel paste-back, fixed target photograph, or procedural background is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows
from .half_qwen import load_half_qwen_text_encoder


# The first trained release uses only the category that passed the teacher
# quality gate. Lamp/table manifests remain available for a later stronger
# teacher, but are intentionally excluded from v1 training.
TARGET_CATEGORIES = ("backpack",)


def _manifest_rows(data_dir: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["reference_path"] = (data_dir / row["reference_path"]).resolve()
        row["target_path"] = (data_dir / row["target_path"]).resolve()
    return [row for row in rows if row["category"] in TARGET_CATEGORIES]


def build_morphology_instruction_cache(
    output: Path,
    data_dir: Path,
    qwen_checkpoint: Path,
    device: torch.device,
    *,
    keep_layers: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    prompts = sorted({row["prompt"] for split in ("train", "val", "test") for row in _manifest_rows(data_dir, split)})
    model, processor, qwen_report = load_half_qwen_text_encoder(
        qwen_checkpoint, device, keep_layers=keep_layers
    )
    features, _ = _encode_caption_rows(
        model,
        processor,
        prompts,
        device,
        12,
        "two-layer-qwen-full-frame-product-morphology",
    )
    meta = {
        "schema_version": 1,
        "task": "product RGB + morphology instruction + seed -> fully regenerated RGB",
        "categories": list(TARGET_CATEGORIES),
        "instructions": len(prompts),
        "text_dim": int(features.shape[1]),
        "qwen": str(qwen_checkpoint.resolve()),
        "qwen_pruning": qwen_report,
        "hard_pixel_composite_at_inference": False,
        "fixed_target_catalogue": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(
        {
            "meta": meta,
            "rows": [{"prompt": prompt} for prompt in prompts],
            "text": features.bfloat16().cpu(),
        },
        temporary,
    )
    temporary.replace(output)
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


class ProductMorphologyDataset(Dataset[dict[str, Any]]):
    """Self-contained paired data produced by a generative editing teacher."""

    targets_per_source = 8
    supported_categories = TARGET_CATEGORIES

    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.split = split
        self.image_size = int(image_size)
        self.rows = _manifest_rows(self.data_dir, split)
        if not self.rows:
            raise ValueError(f"Empty morphology split: {split}")
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.text = payload["text"].float()
        self.prompt_lookup = {row["prompt"]: index for index, row in enumerate(payload["rows"])}
        self.sources = []
        seen = set()
        for row in self.rows:
            if row["source_id"] not in seen:
                self.sources.append({"sample_id": row["source_id"], "category": row["category"]})
                seen.add(row["source_id"])
        expected = len(self.sources) * self.targets_per_source
        if len(self.rows) != expected:
            raise ValueError(f"Expected {expected} rows ({self.targets_per_source}/source), got {len(self.rows)}")

    def __len__(self) -> int:
        return len(self.rows)

    def _tensor(self, path: Path) -> torch.Tensor:
        image = ImageOps.fit(
            Image.open(path).convert("RGB"),
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
        )
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "reference": self._tensor(row["reference_path"]),
            "target": self._tensor(row["target_path"]),
            "qwen_text": self.text[self.prompt_lookup[row["prompt"]]],
            "prompt": row["prompt"],
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "target_id": row["morphology_id"],
            "category": row["category"],
            "catalogue_index": int(row["morphology_index"]),
            "seed": int(row["seed"]),
        }


__all__ = [
    "TARGET_CATEGORIES",
    "ProductMorphologyDataset",
    "build_morphology_instruction_cache",
]
