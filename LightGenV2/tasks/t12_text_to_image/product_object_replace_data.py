"""ABO text-guided object replacement while preserving the input scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows
from .half_qwen import load_half_qwen_text_encoder
from .product_scene_data import _fallback_mask, _normalize_product
from .product_scene_replace_data import COMBINATIONS, render_composed_background


TARGET_CATEGORIES = ("chair", "table")


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def _load_rows(data_dir: Path, split: str) -> list[dict[str, Any]]:
    root = data_dir.resolve()
    result = []
    for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["image_path"] = (root / row["image_path"]).resolve()
        row["mask_path"] = (root / row["mask_path"]).resolve() if row.get("mask_path") else None
        result.append(row)
    return result


def _descriptor(row: dict[str, Any]) -> str:
    text = str(row.get("caption") or f"a {row['category']}").strip().lower()
    for suffix in (" on a clean white studio background", " on a white background"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text


def prompt_variants(row: dict[str, Any]) -> tuple[str, ...]:
    target = _descriptor(row)
    return (
        f"Replace the lamp with {target} while keeping the current room and lighting unchanged.",
        f"Remove the lamp and generate {target} in the same position; preserve the entire background.",
        f"Change only the central object into {target}. Do not alter the surrounding scene.",
    )


def instruction_rows(data_dir: Path) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        for row in _load_rows(data_dir, split):
            if row["category"] not in TARGET_CATEGORIES:
                continue
            key = (row["category"], _descriptor(row))
            unique.setdefault(key, row)
    rows = []
    for category, descriptor in sorted(unique):
        example = unique[(category, descriptor)]
        for variant, prompt in enumerate(prompt_variants(example)):
            rows.append({
                "category": category, "descriptor": descriptor,
                "variant": variant, "prompt": prompt,
            })
    return rows


def build_object_instruction_cache(
    output: Path, data_dir: Path, qwen_checkpoint: Path, device: torch.device,
    *, keep_layers: int = 3, force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    rows = instruction_rows(data_dir)
    model, processor, qwen_report = load_half_qwen_text_encoder(
        qwen_checkpoint, device, keep_layers=keep_layers
    )
    features, _ = _encode_caption_rows(
        model, processor, [row["prompt"] for row in rows], device, 12,
        "three-layer-qwen-object-replacement",
    )
    meta = {
        "schema_version": 1, "task": "lamp + text -> new object; preserve scene",
        "qwen": str(qwen_checkpoint.resolve()), "text_dim": int(features.shape[1]),
        "instructions": len(rows), "target_categories": list(TARGET_CATEGORIES),
        "qwen_pruning": qwen_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save({"meta": meta, "rows": rows, "text": features.bfloat16().cpu()}, temporary)
    temporary.replace(output)
    del model, processor
    if device.type == "cuda": torch.cuda.empty_cache()
    return meta


class ProductObjectReplacementDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        rows = _load_rows(data_dir, split)
        self.sources = [row for row in rows if row["category"] == "lamp"]
        self.targets = {
            category: [row for row in rows if row["category"] == category]
            for category in TARGET_CATEGORIES
        }
        if not self.sources or any(not values for values in self.targets.values()):
            raise ValueError(f"Missing lamp/chair/table examples in {data_dir}/{split}.jsonl")
        self.split = split
        self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {
            (row["category"], row["descriptor"], int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }

    def __len__(self) -> int:
        return len(self.sources) * len(TARGET_CATEGORIES)

    @staticmethod
    def _load_product(row: dict[str, Any], size: int) -> tuple[Image.Image, Image.Image]:
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB")
        if row["mask_path"] is None:
            mask = _fallback_mask(image)
        else:
            with Image.open(row["mask_path"]) as handle:
                mask = handle.convert("L")
        # CleanRender masks intentionally retain a generous antialiased edge
        # for white-background presentation.  A mild erosion removes that
        # visible white matte before compositing onto coloured rooms.
        mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(0.65))
        _, normalized_mask, foreground = _normalize_product(image, mask, size)
        return foreground, normalized_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index // len(TARGET_CATEGORIES)]
        category = TARGET_CATEGORIES[index % len(TARGET_CATEGORIES)]
        candidates = self.targets[category]
        target = candidates[_seed(source["sample_id"], category, self.split) % len(candidates)]
        source_rgb, source_mask = self._load_product(source, self.image_size)
        target_rgb, target_mask = self._load_product(target, self.image_size)
        scene = COMBINATIONS[_seed(source["sample_id"], "shared-scene") % len(COMBINATIONS)]
        background = render_composed_background(scene, self.image_size, source["sample_id"], "object-task")
        reference = Image.composite(source_rgb, background, source_mask)
        target_image = Image.composite(target_rgb, background, target_mask)
        union = Image.fromarray(
            np.maximum(np.asarray(source_mask), np.asarray(target_mask)).astype(np.uint8), "L"
        ).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
        edit_mask = torch.from_numpy(np.asarray(union).copy()).float()[None].div(255)
        descriptor = _descriptor(target)
        variant = _seed(source["sample_id"], target["sample_id"], "prompt") % 3
        prompt_index = self.lookup[(category, descriptor, variant)]

        def tensor(image: Image.Image) -> torch.Tensor:
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

        return {
            "reference": tensor(reference), "target": tensor(target_image),
            "edit_mask": edit_mask, "preserve_mask": 1.0 - edit_mask,
            "qwen_text": self.text[prompt_index], "prompt": self.instructions[prompt_index]["prompt"],
            "sample_id": f"{source['sample_id']}:{target['sample_id']}",
            "source_id": source["sample_id"], "target_id": target["sample_id"],
            "target_category": category, "target_category_index": TARGET_CATEGORIES.index(category),
            "background_rgb": tensor(background),
        }


__all__ = [
    "TARGET_CATEGORIES", "ProductObjectReplacementDataset",
    "build_object_instruction_cache", "instruction_rows", "prompt_variants",
]
