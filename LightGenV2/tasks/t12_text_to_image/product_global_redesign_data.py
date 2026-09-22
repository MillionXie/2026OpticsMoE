"""Full-frame text-guided ABO product redesign without pixel compositing at output.

The task maps an existing lamp or table in one scene to a text-selected design
archetype in a different scene.  Masks are used only offline to construct
supervised source/target pairs; inference receives RGB plus text and the model
decodes every output pixel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows
from .half_qwen import load_half_qwen_text_encoder
from .product_object_replace_data import ProductObjectReplacementDataset
from .product_scene_replace_data import render_composed_background


TARGET_CATEGORIES = ("lamp", "table")
CURATED_DESIGNS = {
    "lamp": (
        {
            "sample_id": "lamp-B075X2Y3VK-29",
            "name": "slim dark cylindrical lamp",
            "scene": ("concrete_loft", "cool", "dim", "left"),
        },
        {
            "sample_id": "lamp-B07MF1S1Z2-14",
            "name": "compact white sculptural lamp",
            "scene": ("minimal_bedroom", "warm", "bright", "right"),
        },
    ),
    "table": (
        {
            "sample_id": "table-B07GZY278M-19",
            "name": "light-oak rectangular console table",
            "scene": ("modern_study", "warm", "bright", "left"),
        },
        {
            "sample_id": "table-B07R8WD99Z-08",
            "name": "round cinder-gray metal table",
            "scene": ("boutique_lounge", "neutral", "dim", "right"),
        },
    ),
}


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def _load_rows(data_dir: Path, split: str) -> list[dict[str, Any]]:
    root = data_dir.resolve()
    rows = []
    for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["image_path"] = (root / row["image_path"]).resolve()
        row["mask_path"] = (root / row["mask_path"]).resolve() if row.get("mask_path") else None
        rows.append(row)
    return rows


def _catalogue(data_dir: Path) -> list[dict[str, Any]]:
    rows = _load_rows(data_dir, "train")
    by_id = {row["sample_id"]: row for row in rows}
    selected = []
    for category in TARGET_CATEGORIES:
        fallback = [row for row in rows if row["category"] == category]
        for design_index, design in enumerate(CURATED_DESIGNS[category]):
            # Unit-test/small-data fallback deliberately remains deterministic.
            row = by_id.get(design["sample_id"], fallback[min(design_index, len(fallback)-1)])
            selected.append({**row, "design_name": design["name"], "target_scene": design["scene"]})
    return selected


def prompt_variants(category: str, design_name: str, scene: tuple[str, str, str, str]) -> tuple[str, ...]:
    room, tone, brightness, direction = scene
    room = room.replace("_", " ")
    return (
        f"Regenerate the entire image as a {design_name} in a {tone}, {brightness} {room}, with window light from the {direction}.",
        f"Create a new {category} concept: {design_name}; render a fresh {tone} {room} scene with {brightness} {direction}-side lighting.",
        f"Transform this product photo into a complete new image of a {design_name}, placed in a {brightness} {tone} {room} lit from the {direction}.",
    )


def instruction_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for catalogue_index, target in enumerate(_catalogue(data_dir)):
        for variant, prompt in enumerate(prompt_variants(target["category"], target["design_name"], target["target_scene"])):
            rows.append({
                "catalogue_index": catalogue_index, "variant": variant,
                "category": target["category"], "design_name": target["design_name"],
                "target_id": target["sample_id"], "prompt": prompt,
            })
    return rows


def build_redesign_instruction_cache(
    output: Path, data_dir: Path, qwen_checkpoint: Path, device: torch.device,
    *, keep_layers: int = 2, force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    rows = instruction_rows(data_dir)
    model, processor, qwen_report = load_half_qwen_text_encoder(
        qwen_checkpoint, device, keep_layers=keep_layers
    )
    features, _ = _encode_caption_rows(
        model, processor, [row["prompt"] for row in rows], device, 12,
        "two-layer-qwen-full-frame-product-redesign",
    )
    meta = {
        "schema_version": 1,
        "task": "RGB product scene + text -> fully regenerated product form and scene",
        "categories": list(TARGET_CATEGORIES), "catalogue_size": len(_catalogue(data_dir)),
        "instructions": len(rows), "text_dim": int(features.shape[1]),
        "qwen": str(qwen_checkpoint.resolve()), "qwen_pruning": qwen_report,
        "hard_pixel_composite_at_inference": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save({"meta": meta, "rows": rows, "text": features.bfloat16().cpu()}, temporary)
    temporary.replace(output)
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


class ProductGlobalRedesignDataset(Dataset[dict[str, Any]]):
    """Same-category form redesign with a full-frame RGB target."""

    targets_per_source = 2

    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        self.sources = [row for row in _load_rows(data_dir, split) if row["category"] in TARGET_CATEGORIES]
        self.catalogue = _catalogue(data_dir)
        self.by_category = {
            category: [(index, row) for index, row in enumerate(self.catalogue) if row["category"] == category]
            for category in TARGET_CATEGORIES
        }
        if not self.sources or any(not value for value in self.by_category.values()):
            raise ValueError("Global redesign requires lamp and table rows")
        self.split = split
        self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {
            (int(row["catalogue_index"]), int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }

    def __len__(self) -> int:
        return len(self.sources) * self.targets_per_source

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index // self.targets_per_source]
        offset = index % self.targets_per_source
        catalogue_index, target = self.by_category[source["category"]][offset]
        source_rgb, source_mask = ProductObjectReplacementDataset._load_product(source, self.image_size)
        target_rgb, target_mask = ProductObjectReplacementDataset._load_product(target, self.image_size)
        source_scenes = (
            ("modern_study", "cool", "bright", "right"),
            ("boutique_lounge", "warm", "dim", "left"),
            ("concrete_loft", "neutral", "bright", "right"),
        )
        source_scene = source_scenes[_seed(source["sample_id"], self.split) % len(source_scenes)]
        target_scene = target["target_scene"]
        source_background = render_composed_background(source_scene, self.image_size, source["sample_id"], "redesign-source")
        target_background = render_composed_background(target_scene, self.image_size, source["sample_id"], "redesign-target")
        reference = Image.composite(source_rgb, source_background, source_mask)
        target_image = Image.composite(target_rgb, target_background, target_mask)
        variant = _seed(source["sample_id"], target["sample_id"], "redesign-prompt") % 3
        prompt_index = self.lookup[(catalogue_index, variant)]
        return {
            "reference": self._tensor(reference), "target": self._tensor(target_image),
            "qwen_text": self.text[prompt_index], "prompt": self.instructions[prompt_index]["prompt"],
            "sample_id": f"{source['sample_id']}:{target['sample_id']}",
            "source_id": source["sample_id"], "target_id": target["sample_id"],
            "category": source["category"], "catalogue_index": catalogue_index,
            "source_scene": "__".join(source_scene), "target_scene": "__".join(target_scene),
        }


__all__ = [
    "TARGET_CATEGORIES", "CURATED_DESIGNS", "ProductGlobalRedesignDataset",
    "build_redesign_instruction_cache", "instruction_rows", "prompt_variants",
]
