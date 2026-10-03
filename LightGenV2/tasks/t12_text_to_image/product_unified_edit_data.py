"""Paired ABO edits with independent background and product-form controls.

The masks and procedural rooms only construct supervised targets. The model
receives a full RGB reference and text; no mask or target catalogue ID is
available at inference.
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
from .product_global_redesign_data import (
    TARGET_CATEGORIES, _catalogue, _load_rows, _seed,
)
from .product_object_replace_data import ProductObjectReplacementDataset
from .product_scene_replace_data import (
    TEST_COMBINATIONS, TRAIN_COMBINATIONS, combination_id,
    render_composed_background,
)


def _background_prompt(scene: tuple[str, str, str, str]) -> str:
    room, tone, brightness, direction = scene
    return (f"Keep this exact product and its shape unchanged. Replace only the background "
            f"with a {brightness}, {tone} {room.replace('_', ' ')}; window light from the {direction}.")


def _object_prompt(design: dict[str, Any]) -> str:
    return (f"Keep the existing room, background, and lighting unchanged. Replace only the "
            f"product with a {design['design_name']} of the same category.")


def _joint_prompt(design: dict[str, Any], scene: tuple[str, str, str, str]) -> str:
    room, tone, brightness, direction = scene
    return (f"Generate a complete new product scene with a {design['design_name']} in a "
            f"{brightness}, {tone} {room.replace('_', ' ')}; window light from the {direction}.")


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
def build_unified_instruction_cache(output: Path, data_dir: Path, qwen_checkpoint: Path,
                                    device: torch.device, *, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    rows = instruction_rows(data_dir)
    model, processor, report = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=2)
    features, _ = _encode_caption_rows(model, processor, [row["prompt"] for row in rows],
                                       device, 12, "qwen-two-block-unified-product-edit")
    meta = {"schema_version": 1, "modes": ["background", "object", "joint"],
            "instructions": len(rows), "text_dim": int(features.shape[1]),
            "qwen_pruning": report, "catalogue_size": len(_catalogue(data_dir))}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "rows": rows, "text": features.bfloat16().cpu()}, output)
    del model, processor
    if device.type == "cuda": torch.cuda.empty_cache()
    return meta


class UnifiedProductEditDataset(Dataset[dict[str, Any]]):
    """Four background, two object, and two joint pairs per ABO input view."""

    targets_per_source = 8
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
        if not self.sources or any(len(value) != 2 for value in self.by_category.values()):
            raise ValueError("Unified editing needs lamp and table sources and two designs each")
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
        design_index, design = self.by_category[source["category"]][offset % 2]
        mode = "background" if offset < 4 else "object" if offset < 6 else "joint"
        scene = self.scenes[_seed(source["sample_id"], self.split, str(offset)) % len(self.scenes)]
        if mode == "background":
            target_background = render_composed_background(scene, self.image_size,
                                                            source["sample_id"], "unified-target")
            target = Image.composite(source_rgb, target_background, source_mask)
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
            "source_id": source["sample_id"], "target_id": design["sample_id"] if mode != "background" else source["sample_id"],
            "category": source["category"], "mode": mode, "catalogue_index": label,
            "source_scene": combination_id(source_scene),
            "target_scene": combination_id(source_scene if mode == "object" else scene),
        }


__all__ = ["UnifiedProductEditDataset", "build_unified_instruction_cache", "instruction_rows"]
