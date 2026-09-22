"""CC BY 4.0 product material restyling with editorial studio targets.

This is the high-fidelity counterpart to the controlled product-redesign task.
The source identity and silhouette remain meaningful, while text controls a
category-appropriate material finish and the entire studio treatment.  Masks
are used only to synthesize paired supervision, never by the inference model.
"""

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
from .product_object_replace_data import ProductObjectReplacementDataset
from .product_scene_data import _normalize_product


SUPPORTED_CATEGORIES = ("lamp", "table", "backpack")
STYLE_KEYS = ("heritage", "obsidian", "ivory", "teal")
STYLE_SPECS = {
    "lamp": {
        "heritage": ("brushed champagne brass", (42, 27, 14), (238, 193, 105), "warm limestone"),
        "obsidian": ("matte obsidian black metal", (8, 10, 13), (80, 86, 91), "cool concrete"),
        "ivory": ("hand-finished ivory ceramic", (112, 103, 88), (250, 243, 220), "soft beige gallery"),
        "teal": ("smoked teal glass", (4, 35, 39), (80, 190, 177), "moody blue-gray"),
    },
    "table": {
        "heritage": ("warm hand-oiled walnut", (39, 17, 8), (198, 113, 55), "warm limestone"),
        "obsidian": ("matte obsidian black oak", (7, 9, 11), (67, 73, 78), "cool concrete"),
        "ivory": ("ivory travertine stone", (105, 96, 80), (244, 230, 197), "soft beige gallery"),
        "teal": ("smoked teal lacquer and glass", (5, 31, 35), (67, 166, 159), "moody blue-gray"),
    },
    "backpack": {
        "heritage": ("cognac full-grain leather", (42, 15, 7), (211, 108, 43), "warm limestone"),
        "obsidian": ("matte obsidian technical nylon", (5, 7, 10), (61, 69, 76), "cool concrete"),
        "ivory": ("ivory woven canvas", (105, 98, 82), (241, 231, 204), "soft beige gallery"),
        "teal": ("deep teal performance textile", (3, 29, 33), (54, 156, 153), "moody blue-gray"),
    },
}


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def prompt_variants(category: str, material: str, studio: str) -> tuple[str, ...]:
    return (
        f"Restyle this {category} in {material} and regenerate the full image as a premium {studio} editorial product photograph.",
        f"Keep the product identity, but render the {category} with a {material} finish in an elegant {studio} advertising studio.",
        f"Create a luxury catalogue image of this same {category}, redesigned in {material}, with refined {studio} lighting and background.",
    )


def instruction_rows() -> list[dict[str, Any]]:
    rows = []
    for category_index, category in enumerate(SUPPORTED_CATEGORIES):
        for style_index, style in enumerate(STYLE_KEYS):
            material, _, _, studio = STYLE_SPECS[category][style]
            catalogue_index = category_index * len(STYLE_KEYS) + style_index
            for variant, prompt in enumerate(prompt_variants(category, material, studio)):
                rows.append({
                    "catalogue_index": catalogue_index, "category": category,
                    "style_index": style_index, "style": style, "material": material,
                    "variant": variant, "prompt": prompt,
                })
    return rows


def build_premium_instruction_cache(
    output: Path, _data_dir: Path, qwen_checkpoint: Path, device: torch.device,
    *, keep_layers: int = 2, force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    rows = instruction_rows()
    model, processor, qwen_report = load_half_qwen_text_encoder(qwen_checkpoint, device, keep_layers=keep_layers)
    features, _ = _encode_caption_rows(
        model, processor, [row["prompt"] for row in rows], device, 12,
        "two-layer-qwen-premium-material-restyling",
    )
    meta = {
        "schema_version": 1,
        "task": "RGB product + material prompt -> premium full-frame studio restyling",
        "categories": list(SUPPORTED_CATEGORIES), "styles": list(STYLE_KEYS),
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


def _load_manifest(root: Path, split: str, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["category"] not in categories:
            continue
        row["image_path"] = (root / row["image_path"]).resolve()
        row["mask_path"] = (root / row["mask_path"]).resolve() if row.get("mask_path") else None
        rows.append(row)
    return rows


def _background(size: int, style: str, sample_id: str, role: str) -> Image.Image:
    palettes = {
        "neutral": ((228, 226, 221), (181, 179, 174), (250, 248, 241)),
        "heritage": ((223, 207, 180), (132, 103, 76), (255, 230, 183)),
        "obsidian": ((89, 96, 103), (28, 32, 37), (179, 194, 204)),
        "ivory": ((239, 228, 207), (181, 161, 135), (255, 244, 218)),
        "teal": ((78, 105, 109), (24, 45, 50), (142, 199, 194)),
    }
    top, bottom, light = palettes[style]
    rng = np.random.default_rng(_seed(sample_id, style, role))
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    image = np.broadcast_to(np.asarray(top)[None, None] * (1-y) + np.asarray(bottom)[None, None] * y, (size, size, 3)).copy()
    xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    center = -0.32 if _seed(sample_id, role) % 2 else 0.32
    glow = np.exp(-((xx-center)**2 / .34 + (yy+.30)**2 / .72))[..., None]
    image = image * (1 - .25 * glow) + np.asarray(light)[None, None] * (.25 * glow)
    floor = int(size * .73)
    image[floor:] *= np.linspace(1.0, .77, size-floor)[:, None, None]
    image += rng.normal(0, 0.9, image.shape[:2])[..., None]
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def _materialize(product: Image.Image, style: str, category: str, sample_id: str) -> Image.Image:
    material, dark, light, _ = STYLE_SPECS[category][style]
    rgb = np.asarray(product).astype(np.float32) / 255.0
    luminance = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    low = np.asarray(dark, dtype=np.float32) / 255.0
    high = np.asarray(light, dtype=np.float32) / 255.0
    styled = low + (high-low) * np.power(luminance[..., None], .78)
    height, width = luminance.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    rng = np.random.default_rng(_seed(sample_id, style, material))
    if style == "heritage":
        texture = .035 * np.sin(yy * .62 + rng.uniform(0, 6.28)) + .018 * np.sin(xx * .15)
    elif style == "obsidian":
        texture = rng.normal(0, .009, luminance.shape)
    elif style == "ivory":
        texture = rng.normal(0, .014, luminance.shape) + .010 * np.sin((xx+yy)*.31)
    else:
        texture = .025 * np.sin(xx * .12 + yy * .07) + rng.normal(0, .006, luminance.shape)
    styled = np.clip(styled + texture[..., None], 0, 1)
    # Preserve high-frequency render detail while replacing colour/material.
    local = luminance - np.asarray(Image.fromarray((luminance*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))).astype(np.float32)/255
    styled = np.clip(.92 * styled + .08 * rgb + local[..., None] * .22, 0, 1)
    return Image.fromarray((styled * 255).astype(np.uint8), "RGB")


def _composite_with_shadow(product: Image.Image, mask: Image.Image, background: Image.Image) -> Image.Image:
    shadow = mask.filter(ImageFilter.GaussianBlur(max(3, background.width // 45)))
    shifted = Image.new("L", background.size, 0)
    shifted.paste(shadow, (0, max(2, background.height // 35)))
    shade = Image.new("RGB", background.size, (35, 31, 28))
    softened = Image.composite(shade, background, shifted.point(lambda value: int(value * .18)))
    return Image.composite(product, softened, mask)


def _load_product(row: dict[str, Any], size: int) -> tuple[Image.Image, Image.Image]:
    if row.get("mask_path") is not None:
        return ProductObjectReplacementDataset._load_product(row, size)
    # The backpack subset has no masks. Estimate its neutral studio background
    # from the border instead of using a fixed white threshold, which otherwise
    # retains a visible rectangular source canvas.
    with Image.open(row["image_path"]) as handle:
        image = handle.convert("RGB")
    array = np.asarray(image).astype(np.float32)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(((array - background[None, None]) ** 2).sum(axis=2))
    mask_array = np.where(distance > 68, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_array, "L").filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(.8))
    _, normalized_mask, foreground = _normalize_product(image, mask, size)
    normalized_mask = normalized_mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(.65))
    return foreground, normalized_mask


class PremiumMaterialDataset(Dataset[dict[str, Any]]):
    targets_per_source = len(STYLE_KEYS)
    supported_categories = SUPPORTED_CATEGORIES

    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        clean = _load_manifest(data_dir / "abo_cleanrender_v1", split, ("lamp", "table"))
        backpacks = _load_manifest(data_dir / "abo_backpack_style_v2", split, ("backpack",))
        self.sources = clean + backpacks
        self.split = split; self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]; self.text = payload["text"].float()
        self.lookup = {
            (row["category"], int(row["style_index"]), int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }

    def __len__(self) -> int:
        return len(self.sources) * self.targets_per_source

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index // self.targets_per_source]
        style_index = index % self.targets_per_source; style = STYLE_KEYS[style_index]
        product, mask = _load_product(source, self.image_size)
        reference_background = _background(self.image_size, "neutral", source["sample_id"], "premium-source")
        target_background = _background(self.image_size, style, source["sample_id"], "premium-target")
        reference = _composite_with_shadow(product, mask, reference_background)
        styled = _materialize(product, style, source["category"], source["sample_id"])
        target = _composite_with_shadow(styled, mask, target_background)
        variant = _seed(source["sample_id"], style, "premium-prompt") % 3
        prompt_index = self.lookup[(source["category"], style_index, variant)]
        catalogue_index = SUPPORTED_CATEGORIES.index(source["category"]) * len(STYLE_KEYS) + style_index
        return {
            "reference": self._tensor(reference), "target": self._tensor(target),
            "qwen_text": self.text[prompt_index], "prompt": self.instructions[prompt_index]["prompt"],
            "sample_id": f"{source['sample_id']}:{style}", "source_id": source["sample_id"],
            "target_id": f"{source['category']}:{style}", "category": source["category"],
            "style": style, "catalogue_index": catalogue_index,
        }


__all__ = [
    "SUPPORTED_CATEGORIES", "STYLE_KEYS", "STYLE_SPECS", "PremiumMaterialDataset",
    "build_premium_instruction_cache", "instruction_rows", "prompt_variants",
]
