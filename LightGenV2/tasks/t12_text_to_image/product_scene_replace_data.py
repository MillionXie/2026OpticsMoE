"""Compositional background replacement for ABO CleanRender objects.

Unlike the first scene experiment, the reference already has a background.
The target prompt independently controls room, colour temperature, brightness,
and side-window direction.  Eight exact attribute combinations are excluded
from training and used only by validation/test to measure composition rather
than memorisation of a small list of scene labels.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows
from .half_qwen import load_half_qwen_text_encoder
from .product_scene_data import _fallback_mask, _load_manifest, _normalize_product


ROOMS = ("modern_study", "minimal_bedroom", "boutique_lounge", "concrete_loft")
TONES = ("cool", "warm", "neutral")
BRIGHTNESS = ("dim", "bright")
DIRECTIONS = ("left", "right")

ROOM_LABELS = {
    "modern_study": "modern study",
    "minimal_bedroom": "minimalist bedroom",
    "boutique_lounge": "boutique hotel lounge",
    "concrete_loft": "concrete loft",
}
TONE_LABELS = {"cool": "cool-toned", "warm": "warm-toned", "neutral": "muted neutral"}
BRIGHTNESS_LABELS = {"dim": "dim", "bright": "bright"}

COMBINATIONS: tuple[tuple[str, str, str, str], ...] = tuple(
    itertools.product(ROOMS, TONES, BRIGHTNESS, DIRECTIONS)
)
HOLDOUT_COMBINATIONS = frozenset({
    ("modern_study", "cool", "dim", "left"),
    ("modern_study", "warm", "bright", "right"),
    ("minimal_bedroom", "neutral", "dim", "right"),
    ("minimal_bedroom", "cool", "bright", "left"),
    ("boutique_lounge", "warm", "dim", "left"),
    ("boutique_lounge", "neutral", "bright", "right"),
    ("concrete_loft", "cool", "dim", "right"),
    ("concrete_loft", "warm", "bright", "left"),
})
TRAIN_COMBINATIONS = tuple(value for value in COMBINATIONS if value not in HOLDOUT_COMBINATIONS)
TEST_COMBINATIONS = tuple(value for value in COMBINATIONS if value in HOLDOUT_COMBINATIONS)


def combination_id(value: tuple[str, str, str, str]) -> str:
    return "__".join(value)


def prompt_variants(value: tuple[str, str, str, str]) -> tuple[str, ...]:
    room, tone, brightness, direction = value
    room_label = ROOM_LABELS[room]
    tone_label = TONE_LABELS[tone]
    side = f"soft window light from the {direction}"
    return (
        f"Keep the object unchanged and replace its current background with a {tone_label}, {brightness} {room_label} with {side}.",
        f"Put this exact object in a {brightness}, {tone_label} {room_label}; the main window light enters from the {direction}.",
        f"Change only the surrounding scene to a {room_label}: {tone_label}, {brightness}, with directional window light on the {direction} side.",
    )


def instruction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combination_index, value in enumerate(COMBINATIONS):
        for variant, prompt in enumerate(prompt_variants(value)):
            rows.append({
                "combination_index": combination_index,
                "combination_id": combination_id(value),
                "room": value[0], "tone": value[1], "brightness": value[2],
                "direction": value[3], "variant": variant, "prompt": prompt,
            })
    return rows


def build_half_qwen_instruction_cache(
    output: Path,
    qwen_checkpoint: Path,
    device: torch.device,
    *,
    keep_layers: int = 14,
    force: bool = False,
) -> dict[str, Any]:
    if output.exists() and not force:
        return torch.load(output, map_location="cpu", weights_only=False)["meta"]
    rows = instruction_rows()
    model, processor, qwen_report = load_half_qwen_text_encoder(
        qwen_checkpoint, device, keep_layers=keep_layers
    )
    features, _ = _encode_caption_rows(
        model, processor, [row["prompt"] for row in rows], device, 12,
        "half-qwen-compositional-scene-instructions",
    )
    meta = {
        "schema_version": 2,
        "task": "background-present object + compositional text -> replaced background",
        "qwen": str(qwen_checkpoint.resolve()),
        "text_dim": int(features.shape[1]),
        "instructions": len(rows),
        "combinations_total": len(COMBINATIONS),
        "combinations_train": len(TRAIN_COMBINATIONS),
        "combinations_held_out": len(TEST_COMBINATIONS),
        "qwen_pruning": qwen_report,
    }
    payload = {"meta": meta, "rows": rows, "text": features.to(torch.bfloat16).cpu()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def _palette(tone: str, brightness: str) -> dict[str, tuple[int, int, int]]:
    values = {
        "cool": ((174, 193, 208), (90, 111, 130), (222, 235, 242)),
        "warm": ((210, 185, 154), (125, 91, 65), (249, 220, 172)),
        "neutral": ((193, 190, 181), (106, 103, 98), (231, 226, 211)),
    }[tone]
    factor = 0.62 if brightness == "dim" else 1.08
    return {
        name: tuple(int(np.clip(channel * factor, 8, 250)) for channel in value)
        for name, value in zip(("wall", "floor", "light"), values)
    }


def _gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    array = np.broadcast_to(
        np.asarray(top)[None, None] * (1 - y) + np.asarray(bottom)[None, None] * y,
        (size, size, 3),
    ).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def render_composed_background(
    value: tuple[str, str, str, str], size: int, sample_id: str, role: str,
) -> Image.Image:
    """Render a sample-varying scene whose four controls remain identifiable."""

    room, tone, brightness, direction = value
    rng = np.random.default_rng(_seed(sample_id, combination_id(value), role))
    palette = _palette(tone, brightness)
    jitter = int(rng.integers(-7, 8))
    top = tuple(int(np.clip(v + jitter, 0, 255)) for v in palette["wall"])
    image = _gradient(size, top, palette["floor"])
    draw = ImageDraw.Draw(image, "RGBA")
    floor_y = int(size * (0.70 + rng.uniform(-0.025, 0.025)))
    draw.rectangle((0, floor_y, size, size), fill=(*palette["floor"], 255))
    left_side = direction == "left"
    window_x0 = int((.04 if left_side else .76) * size)
    window_x1 = window_x0 + int(.20 * size)
    window_box = (window_x0, int(.08*size), window_x1, int(.55*size))
    draw.rectangle(window_box, fill=(*palette["light"], 170), outline=(70, 75, 78, 150), width=2)
    draw.line(((window_x0+window_x1)//2, window_box[1], (window_x0+window_x1)//2, window_box[3]), fill=(75, 78, 80, 110), width=1)
    draw.line((window_x0, int(.31*size), window_x1, int(.31*size)), fill=(75, 78, 80, 110), width=1)

    if room == "modern_study":
        other = int((.69 if left_side else .07) * size)
        draw.rectangle((other, int(.17*size), other+int(.23*size), int(.47*size)), outline=(45, 52, 58, 155), width=2)
        for y in (.25, .34, .43):
            draw.line((other, int(y*size), other+int(.23*size), int(y*size)), fill=(48, 53, 57, 130), width=2)
        draw.rectangle((int(.06*size), int(.63*size), int(.94*size), int(.72*size)), fill=(59, 49, 43, 115))
    elif room == "minimal_bedroom":
        draw.rounded_rectangle((int(.07*size), int(.48*size), int(.93*size), int(.78*size)), radius=int(.04*size), fill=(225, 220, 211, 105))
        draw.rectangle((int(.13*size), int(.37*size), int(.87*size), int(.58*size)), fill=(94, 87, 80, 80))
        draw.rectangle((int(.38*size), int(.15*size), int(.62*size), int(.34*size)), outline=(65, 62, 58, 120), width=2)
    elif room == "boutique_lounge":
        for x in (.10, .34, .66, .90):
            draw.line((int(x*size), int(.08*size), int(x*size), floor_y), fill=(174, 135, 73, 90), width=2)
        draw.rounded_rectangle((int(.04*size), int(.53*size), int(.34*size), int(.75*size)), radius=int(.05*size), fill=(74, 52, 48, 105))
        draw.rounded_rectangle((int(.66*size), int(.53*size), int(.96*size), int(.75*size)), radius=int(.05*size), fill=(74, 52, 48, 105))
    else:
        for x in (.31, .63):
            draw.line((int(x*size), 0, int(x*size), floor_y), fill=(78, 81, 82, 70), width=2)
        for y in (.18, .41):
            draw.line((0, int(y*size), size, int(y*size)), fill=(78, 81, 82, 55), width=2)
        draw.rectangle((int(.72*size), int(.56*size), int(.93*size), int(.70*size)), fill=(55, 57, 59, 90))

    beam = Image.new("RGBA", image.size, (0, 0, 0, 0))
    beam_draw = ImageDraw.Draw(beam)
    if left_side:
        polygon = ((0, int(.10*size)), (int(.35*size), 0), (int(.79*size), size), (int(.43*size), size))
    else:
        polygon = ((size, int(.10*size)), (int(.65*size), 0), (int(.21*size), size), (int(.57*size), size))
    beam_draw.polygon(polygon, fill=(*palette["light"], 50 if brightness == "dim" else 76))
    beam = beam.filter(ImageFilter.GaussianBlur(max(5, size//28)))
    image = Image.alpha_composite(image.convert("RGBA"), beam).convert("RGB")
    array = np.asarray(image).astype(np.int16)
    noise = rng.normal(0.0, 1.5, (size, size, 1))
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")


class ProductBackgroundReplacementDataset(Dataset[dict[str, Any]]):
    targets_per_source = 8

    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        self.rows = _load_manifest(data_dir / f"{split}.jsonl")
        self.split = split
        self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {
            (row["combination_id"], int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }
        self.pool = TEST_COMBINATIONS if split in {"val", "test"} else TRAIN_COMBINATIONS

    def __len__(self) -> int:
        return len(self.rows) * self.targets_per_source

    def _target(self, row: dict[str, Any], offset: int) -> tuple[str, str, str, str]:
        base = _seed(row["sample_id"], self.split, "target") % len(self.pool)
        step = 7 if len(self.pool) % 7 else 9
        return self.pool[(base + offset * step) % len(self.pool)]

    def _source(self, sample_id: str, target: tuple[str, str, str, str], offset: int) -> tuple[str, str, str, str]:
        candidates = [value for value in TRAIN_COMBINATIONS if value[0] != target[0] and value[1] != target[1]]
        return candidates[_seed(sample_id, str(offset), "source") % len(candidates)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index // self.targets_per_source]
        offset = index % self.targets_per_source
        target_attributes = self._target(row, offset)
        source_attributes = self._source(row["sample_id"], target_attributes, offset)
        with Image.open(row["image_path"]) as handle:
            product = handle.convert("RGB")
        if row["mask_path"] is not None:
            with Image.open(row["mask_path"]) as handle:
                source_mask = handle.convert("L")
        else:
            source_mask = _fallback_mask(product)
        _, mask_image, foreground = _normalize_product(product, source_mask, self.image_size)
        source_background = render_composed_background(source_attributes, self.image_size, row["sample_id"], "source")
        target_background = render_composed_background(target_attributes, self.image_size, row["sample_id"], "target")
        reference = Image.composite(foreground, source_background, mask_image)
        target = Image.composite(foreground, target_background, mask_image)
        variant = _seed(row["sample_id"], combination_id(target_attributes), "prompt") % 3
        prompt_index = self.lookup[(combination_id(target_attributes), variant)]

        def tensor(image: Image.Image) -> torch.Tensor:
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

        foreground_mask = torch.from_numpy(np.asarray(mask_image).copy()).float()[None].div(255)
        labels = {
            "room_index": ROOMS.index(target_attributes[0]),
            "tone_index": TONES.index(target_attributes[1]),
            "brightness_index": BRIGHTNESS.index(target_attributes[2]),
            "direction_index": DIRECTIONS.index(target_attributes[3]),
        }
        return {
            "reference": tensor(reference), "target": tensor(target),
            "foreground_rgb": tensor(foreground), "foreground_mask": foreground_mask,
            "background_mask": 1.0 - foreground_mask, "qwen_text": self.text[prompt_index],
            "prompt": self.instructions[prompt_index]["prompt"],
            "sample_id": f"{row['sample_id']}:{combination_id(target_attributes)}",
            "source_id": row["sample_id"], "source_scene_id": combination_id(source_attributes),
            "target_scene_id": combination_id(target_attributes),
            "held_out_combination": target_attributes in HOLDOUT_COMBINATIONS,
            **labels,
        }


def write_pair_manifest(dataset: ProductBackgroundReplacementDataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            value = dataset[index]
            row = dataset.rows[index // dataset.targets_per_source]
            handle.write(json.dumps({
                key: value[key] for key in (
                    "sample_id", "source_id", "source_scene_id", "target_scene_id",
                    "prompt", "held_out_combination",
                )
            } | {
                "sequence_id": row["sequence_id"], "source_image": str(row["image_path"]),
                "license": row["license"], "source_url": row["source_url"],
                "background_provenance": "deterministic procedural renderer; no external image",
            }) + "\n")


__all__ = [
    "BRIGHTNESS", "COMBINATIONS", "DIRECTIONS", "HOLDOUT_COMBINATIONS", "ROOMS",
    "TONES", "ProductBackgroundReplacementDataset", "build_half_qwen_instruction_cache",
    "combination_id", "render_composed_background", "write_pair_manifest",
]
