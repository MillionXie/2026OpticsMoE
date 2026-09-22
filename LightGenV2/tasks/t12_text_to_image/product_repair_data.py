"""Paired ABO product restoration data where the instruction changes the target.

Each reference contains two missing object regions.  The instruction selects
exactly one region to restore and the target keeps the other region missing.
Consequently the same source image admits different correct outputs and a
model cannot solve the task by copying the input or ignoring the text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

from .dataset import read_manifest
from .feature_cache import _encode_caption_rows


SUPPORTED_CATEGORIES = ("pillow", "cabinet", "dresser", "ottoman")
REGIONS = ("upper", "center", "lower")
REGION_LABELS = {
    "pillow": {"upper": "upper seam", "center": "center fabric", "lower": "lower corner"},
    "cabinet": {"upper": "upper panel", "center": "center door", "lower": "lower base"},
    "dresser": {"upper": "top surface", "center": "drawer area", "lower": "lower base"},
    "ottoman": {"upper": "top cushion", "center": "side panel", "lower": "lower legs"},
}
PROMPT_TEMPLATES = (
    "Restore only the missing {part} of the {category}; leave the other missing area unchanged.",
    "Repair the {category}'s {part} only, without filling the second damaged region.",
    "Complete just the {part} on this {category} and preserve the other blank area.",
)


def instruction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in SUPPORTED_CATEGORIES:
        for region in REGIONS:
            part = REGION_LABELS[category][region]
            for variant, template in enumerate(PROMPT_TEMPLATES):
                rows.append({
                    "category": category,
                    "region": region,
                    "part": part,
                    "variant": variant,
                    "prompt": template.format(part=part, category=category),
                })
    return rows


def build_instruction_cache(
    output: Path,
    qwen_checkpoint: Path,
    device: torch.device,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rows = instruction_rows()
    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    features, _ = _encode_caption_rows(
        model, processor, [row["prompt"] for row in rows], device, 12, "repair-instructions"
    )
    payload = {
        "meta": {
            "schema_version": 1,
            "task": "text_selected_product_restoration",
            "qwen": str(qwen_checkpoint.resolve()),
            "text_dim": int(features.shape[1]),
            "instructions": len(rows),
        },
        "rows": rows,
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


def _foreground_mask(image: torch.Tensor) -> torch.Tensor:
    """Return a conservative single-object mask for white-background ABO renders."""

    rgb = image.add(1).mul(0.5)
    distance = (1.0 - rgb).amax(dim=0)
    mask = (distance > 0.055).float()[None, None]
    mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask[0]


def _region_masks(image: torch.Tensor, severity: float) -> dict[str, torch.Tensor]:
    foreground = _foreground_mask(image)
    positions = torch.nonzero(foreground[0] > 0.5)
    height, width = image.shape[-2:]
    if not len(positions):
        return {region: torch.zeros(1, height, width) for region in REGIONS}
    y0, x0 = positions.amin(0).tolist()
    y1, x1 = positions.amax(0).tolist()
    box_h, box_w = max(4, y1 - y0 + 1), max(4, x1 - x0 + 1)
    yy = torch.arange(height, dtype=torch.float32)[:, None]
    xx = torch.arange(width, dtype=torch.float32)[None, :]
    radius_y = max(3.0, box_h * (0.13 + 0.18 * severity))
    radius_x = max(3.0, box_w * (0.18 + 0.20 * severity))
    centers = {
        "upper": (y0 + 0.24 * box_h, x0 + 0.38 * box_w),
        "center": (y0 + 0.50 * box_h, x0 + 0.60 * box_w),
        "lower": (y0 + 0.76 * box_h, x0 + 0.42 * box_w),
    }
    result = {}
    for region, (cy, cx) in centers.items():
        ellipse = (((yy - cy) / radius_y) ** 2 + ((xx - cx) / radius_x) ** 2 <= 1).float()
        result[region] = ellipse[None] * foreground
    return result


def _fill_missing(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A very light neutral hole resembles an incomplete render while remaining
    # distinct from the pure-white studio background.
    fill = image.new_tensor([0.84, 0.86, 0.88])[:, None, None].mul(2).sub(1)
    return image * (1.0 - mask) + fill * mask


@dataclass(frozen=True)
class RepairPair:
    reference: torch.Tensor
    target: torch.Tensor
    selected_mask: torch.Tensor
    distractor_mask: torch.Tensor
    prompt: str
    qwen_text: torch.Tensor
    sample_id: str
    category: str


class ProductRepairDataset(Dataset[dict[str, Any]]):
    """Deterministic, identity-disjoint restoration pairs from clean ABO renders."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        image_size: int,
        instruction_cache: Path,
        *,
        seed: int = 42,
    ) -> None:
        self.rows = read_manifest(data_dir / f"{split}.jsonl")
        self.image_size = int(image_size)
        self.seed = int(seed)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {
            (row["category"], row["region"], int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }
        unsupported = sorted({row.category for row in self.rows} - set(SUPPORTED_CATEGORIES))
        if unsupported:
            raise ValueError(f"Unsupported restoration categories: {unsupported}")

    def __len__(self) -> int:
        return len(self.rows)

    def _choice(self, index: int, sample_id: str) -> tuple[str, str, int, float]:
        digest = hashlib.sha256(f"{self.seed}:{index}:{sample_id}".encode()).digest()
        selected_index = digest[0] % len(REGIONS)
        distractor_index = (selected_index + 1 + digest[1] % (len(REGIONS) - 1)) % len(REGIONS)
        variant = digest[2] % len(PROMPT_TEMPLATES)
        severity = (0.15, 0.25, 0.35)[digest[3] % 3]
        return REGIONS[selected_index], REGIONS[distractor_index], variant, severity

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row.image_path) as handle:
            image = ImageOps.fit(
                ImageOps.exif_transpose(handle).convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.LANCZOS,
            )
        clean = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)
        selected, distractor, variant, severity = self._choice(index, row.sample_id)
        masks = _region_masks(clean, severity)
        reference = _fill_missing(_fill_missing(clean, masks[selected]), masks[distractor])
        target = _fill_missing(clean, masks[distractor])
        prompt_index = self.lookup[(row.category, selected, variant)]
        prompt = self.instructions[prompt_index]["prompt"]
        return {
            "reference": reference,
            "target": target,
            "selected_mask": masks[selected],
            "distractor_mask": masks[distractor],
            "qwen_text": self.text[prompt_index],
            "prompt": prompt,
            "sample_id": row.sample_id,
            "category": row.category,
            "selected_region": selected,
            "distractor_region": distractor,
        }


def write_pair_manifest(dataset: ProductRepairDataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset.rows):
            selected, distractor, variant, severity = dataset._choice(index, row.sample_id)
            prompt_index = dataset.lookup[(row.category, selected, variant)]
            handle.write(json.dumps({
                "sample_id": row.sample_id,
                "sequence_id": row.sequence_id,
                "category": row.category,
                "source_image": str(row.image_path),
                "selected_region": selected,
                "distractor_region": distractor,
                "severity": severity,
                "prompt": dataset.instructions[prompt_index]["prompt"],
                "license": row.license,
                "source_url": row.source_url,
            }) + "\n")


__all__ = [
    "ProductRepairDataset",
    "REGION_LABELS",
    "REGIONS",
    "SUPPORTED_CATEGORIES",
    "build_instruction_cache",
    "instruction_rows",
    "write_pair_manifest",
]
