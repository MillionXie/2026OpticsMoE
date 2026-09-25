"""Expanded ABO lamp/table/pillow editor with four text-named designs each.

The source product is held out by product identity in validation/test. Masks
and procedural rooms make paired supervision only; no mask or design ID enters
the image generator at inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows
from .half_qwen import load_half_qwen_text_encoder
from .product_global_redesign_data import _load_rows, _seed
from .product_object_replace_data import ProductObjectReplacementDataset
from .product_scene_replace_data import (
    TEST_COMBINATIONS, TRAIN_COMBINATIONS, combination_id,
    render_composed_background,
)
from .product_unified_edit_data import _background_prompt, _joint_prompt, _object_prompt


TARGET_CATEGORIES = ("lamp", "table", "pillow")
DESIGNS: dict[str, tuple[dict[str, str], ...]] = {
    "lamp": (
        {"sample_id": "lamp-B075X2Y3VK-29", "name": "beige drum-shade lamp with a textured blue cylindrical base"},
        {"sample_id": "lamp-B07MF1S1Z2-14", "name": "compact white sculptural lamp"},
        {"sequence_id": "B07B4ZBB7R", "name": "slender brass floor lamp"},
        {"sequence_id": "B07B4W2ZRW", "name": "brown woven tripod lamp"},
    ),
    "table": (
        {"sample_id": "table-B07GZY278M-19", "name": "light-oak rectangular console table"},
        {"sample_id": "table-B07R8WD99Z-08", "name": "round cinder-gray metal table"},
        {"sequence_id": "B07H2J7689", "name": "open-shelf natural-wood side table"},
        {"sequence_id": "B07G527L5S", "name": "low black coffee table"},
    ),
    "pillow": (
        {"sequence_id": "B07JL5PNGZ", "name": "round muted-green pleated cushion"},
        {"sequence_id": "B079TXJCP3", "name": "dark navy square cushion"},
        {"sequence_id": "B07C8VQNT6", "name": "tan textured square cushion"},
        {"sequence_id": "B07M6PJ4LX", "name": "cream square cushion with a blue botanical print"},
    ),
}


def _mask_area(row: dict[str, Any]) -> int:
    mask_path = row.get("mask_path")
    if mask_path is None:
        return 0
    with Image.open(mask_path) as handle:
        values = np.asarray(handle.convert("L"), dtype=np.uint8)
    return int(np.count_nonzero(values > 32))


def _catalogue(data_dir: Path) -> list[dict[str, Any]]:
    rows = _load_rows(data_dir, "train")
    by_id = {row["sample_id"]: row for row in rows}
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_id"], []).append(row)
    chosen = []
    for category in TARGET_CATEGORIES:
        for design in DESIGNS[category]:
            if "sample_id" in design:
                row = by_id.get(design["sample_id"])
                if row is None:
                    raise ValueError(f"Missing curated design {design['sample_id']}")
            else:
                options = by_sequence.get(design["sequence_id"], [])
                if not options:
                    raise ValueError(f"Missing curated product {design['sequence_id']}")
                # Front-facing views expose more of a product's actual surface
                # than narrow side views, especially for textile cushions.
                row = max(options, key=_mask_area)
            if row["category"] != category:
                raise ValueError(f"Wrong category for curated design: {row['sample_id']}")
            chosen.append({**row, "design_name": design["name"]})
    return chosen


def instruction_rows(data_dir: Path) -> list[dict[str, Any]]:
    catalogue = _catalogue(data_dir)
    rows = [{"mode": "background", "prompt": _background_prompt(scene),
             "scene": combination_id(scene)}
            for scene in (*TRAIN_COMBINATIONS, *TEST_COMBINATIONS)]
    for index, design in enumerate(catalogue):
        rows.append({"mode": "object", "prompt": _object_prompt(design), "design": index})
        for scene in (*TRAIN_COMBINATIONS, *TEST_COMBINATIONS):
            rows.append({"mode": "joint", "prompt": _joint_prompt(design, scene),
                         "design": index, "scene": combination_id(scene)})
    return rows


@torch.inference_mode()
def build_expanded_instruction_cache(output: Path, data_dir: Path, qwen_checkpoint: Path,
                                     device: torch.device) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = instruction_rows(data_dir)
    model, processor, report = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=2)
    features, _ = _encode_caption_rows(model, processor, [row["prompt"] for row in rows],
                                       device, 12, "qwen-two-block-expanded-unified-product-edit")
    meta = {"schema_version": 2, "modes": ["background", "object", "joint"],
            "categories": list(TARGET_CATEGORIES), "instructions": len(rows),
            "text_dim": int(features.shape[1]), "qwen_pruning": report,
            "catalogue_size": len(_catalogue(data_dir))}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "rows": rows, "text": features.bfloat16().cpu()}, output)
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


class ExpandedUnifiedProductEditDataset(Dataset[dict[str, Any]]):
    targets_per_source = 12
    supported_categories = TARGET_CATEGORIES

    def __init__(self, data_dir: Path, split: str, image_size: int,
                 instruction_cache: Path) -> None:
        self.sources = [row for row in _load_rows(data_dir, split)
                        if row["category"] in TARGET_CATEGORIES]
        self.catalogue = _catalogue(data_dir)
        self.by_category = {
            category: [(i, row) for i, row in enumerate(self.catalogue)
                       if row["category"] == category]
            for category in TARGET_CATEGORIES
        }
        if not self.sources or any(len(value) != 4 for value in self.by_category.values()):
            raise ValueError("Expanded editing needs three categories and four designs each")
        self.split = split
        self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {row["prompt"]: i for i, row in enumerate(self.instructions)}
        self.scenes = TRAIN_COMBINATIONS if split == "train" else TEST_COMBINATIONS

    def __len__(self) -> int:
        return len(self.sources) * self.targets_per_source

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index // self.targets_per_source]
        offset = index % self.targets_per_source
        source_rgb, source_mask = ProductObjectReplacementDataset._load_product(source, self.image_size)
        source_scenes = (
            ("modern_study", "cool", "bright", "right"),
            ("boutique_lounge", "warm", "dim", "left"),
            ("concrete_loft", "neutral", "bright", "right"),
        )
        source_scene = source_scenes[_seed(source["sample_id"], self.split) % len(source_scenes)]
        source_background = render_composed_background(source_scene, self.image_size,
                                                       source["sample_id"], "unified-source")
        reference = Image.composite(source_rgb, source_background, source_mask)
        mode = "background" if offset < 4 else "object" if offset < 8 else "joint"
        design_index, design = self.by_category[source["category"]][offset % 4]
        scene = self.scenes[_seed(source["sample_id"], self.split, str(offset)) % len(self.scenes)]
        if mode == "background":
            background = render_composed_background(scene, self.image_size,
                                                    source["sample_id"], "unified-target")
            target = Image.composite(source_rgb, background, source_mask)
            prompt = _background_prompt(scene)
            label = len(self.catalogue)
        else:
            design_rgb, design_mask = ProductObjectReplacementDataset._load_product(design, self.image_size)
            background = source_background if mode == "object" else render_composed_background(
                scene, self.image_size, source["sample_id"], "unified-target")
            target = Image.composite(design_rgb, background, design_mask)
            prompt = _object_prompt(design) if mode == "object" else _joint_prompt(design, scene)
            label = design_index
        prompt_index = self.lookup[prompt]
        return {
            "reference": self._tensor(reference), "target": self._tensor(target),
            "qwen_text": self.text[prompt_index], "prompt": prompt,
            "sample_id": f"{source['sample_id']}:{mode}:{offset}",
            "source_id": source["sample_id"],
            "target_id": design["sample_id"] if mode != "background" else source["sample_id"],
            "category": source["category"], "mode": mode, "catalogue_index": label,
            "source_scene": combination_id(source_scene),
            "target_scene": combination_id(source_scene if mode == "object" else scene),
        }


__all__ = ["TARGET_CATEGORIES", "DESIGNS", "ExpandedUnifiedProductEditDataset",
           "build_expanded_instruction_cache", "instruction_rows"]
