"""Paired text-controlled scene generation for ABO white-background lamps.

Every source render is reused with six target backgrounds.  The object pixels
are copied through the official ABO alpha mask, so the learning problem is
background generation rather than repairing or redrawing the product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

from .feature_cache import _encode_caption_rows


SCENES: tuple[dict[str, Any], ...] = (
    {
        "id": "warm_minimal",
        "label": "warm minimalist interior",
        "prompts": (
            "Place this exact lamp in a warm minimalist cream interior with soft afternoon light.",
            "Keep the lamp unchanged and replace the white background with a calm warm beige studio room.",
            "Show the same lamp in an elegant ivory interior with gentle golden sunlight.",
        ),
    },
    {
        "id": "cool_modern",
        "label": "cool blue modern studio",
        "prompts": (
            "Place this exact lamp in a cool blue modern studio with clean window light.",
            "Keep the lamp unchanged and create a crisp blue-gray contemporary interior behind it.",
            "Show the same lamp in a modern cool-toned room with a large side window.",
        ),
    },
    {
        "id": "dark_luxury",
        "label": "dark luxury interior",
        "prompts": (
            "Place this exact lamp in a dark luxury interior with subtle brass details and a warm halo.",
            "Keep the lamp unchanged and replace the background with an elegant charcoal hotel scene.",
            "Show the same lamp against a dramatic black paneled wall with restrained golden light.",
        ),
    },
    {
        "id": "terracotta_gallery",
        "label": "terracotta gallery",
        "prompts": (
            "Place this exact lamp in a terracotta art gallery with an arched wall and sunbeam.",
            "Keep the lamp unchanged and create a refined clay-colored Mediterranean backdrop.",
            "Show the same lamp in a warm burnt-orange gallery with sculptural sunlight.",
        ),
    },
    {
        "id": "sage_reading",
        "label": "sage green reading nook",
        "prompts": (
            "Place this exact lamp in a quiet sage green reading nook with a plant and framed art.",
            "Keep the lamp unchanged and add a soft green Scandinavian room behind it.",
            "Show the same lamp in a peaceful muted-green interior with natural decor.",
        ),
    },
    {
        "id": "concrete_loft",
        "label": "concrete loft",
        "prompts": (
            "Place this exact lamp in a minimal concrete loft with graphic daylight and a gray floor.",
            "Keep the lamp unchanged and replace the background with a refined industrial loft.",
            "Show the same lamp in a clean architectural concrete room with diagonal window light.",
        ),
    },
)


def instruction_rows() -> list[dict[str, Any]]:
    return [
        {
            "scene_index": scene_index,
            "scene_id": scene["id"],
            "scene_label": scene["label"],
            "variant": variant,
            "prompt": prompt,
        }
        for scene_index, scene in enumerate(SCENES)
        for variant, prompt in enumerate(scene["prompts"])
    ]


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
        model, processor, [row["prompt"] for row in rows], device, 12, "scene-instructions"
    )
    payload = {
        "meta": {
            "schema_version": 1,
            "task": "text_controlled_product_scene_generation",
            "qwen": str(qwen_checkpoint.resolve()),
            "text_dim": int(features.shape[1]),
            "instructions": len(rows),
            "scenes": len(SCENES),
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


def _seed(sample_id: str, scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{sample_id}:{scene_id}".encode()).digest()[:8], "big")


def _gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    a = np.asarray(top, dtype=np.float32)[None, None]
    b = np.asarray(bottom, dtype=np.float32)[None, None]
    array = np.broadcast_to(a * (1 - y) + b * y, (size, size, 3)).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _soft_light(image: Image.Image, box: tuple[int, int, int, int], color: tuple[int, int, int], alpha: int) -> None:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(box, fill=(*color, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(max(6, image.width // 12)))
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


def render_background(scene_index: int, size: int, sample_id: str) -> Image.Image:
    """Render a polished deterministic 2.5-D scene without external imagery."""

    scene = SCENES[scene_index]
    rng = np.random.default_rng(_seed(sample_id, scene["id"]))
    jitter = int(rng.integers(-5, 6))
    floor_y = int(size * (0.70 + rng.uniform(-0.025, 0.025)))

    if scene["id"] == "warm_minimal":
        image = _gradient(size, (246 + jitter, 238 + jitter, 220 + jitter), (216, 193, 162))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(190, 159, 125, 255))
        draw.rounded_rectangle((int(.06*size), int(.12*size), int(.39*size), int(.78*size)),
                               radius=int(.17*size), fill=(232, 213, 185, 190))
        draw.rectangle((int(.78*size), int(.19*size), int(.91*size), int(.44*size)),
                       outline=(145, 116, 83, 150), width=max(1, size//128))
        _soft_light(image, (-size//5, -size//4, int(.85*size), int(.9*size)), (255, 224, 164), 85)
    elif scene["id"] == "cool_modern":
        image = _gradient(size, (211 + jitter, 224 + jitter, 231 + jitter), (145, 170, 187))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(113, 139, 156, 255))
        draw.rectangle((int(.04*size), int(.10*size), int(.30*size), int(.64*size)),
                       fill=(226, 241, 247, 210), outline=(111, 146, 166, 180), width=max(1, size//96))
        draw.line((int(.17*size), int(.10*size), int(.17*size), int(.64*size)), fill=(100, 139, 161, 150), width=2)
        draw.line((int(.04*size), int(.36*size), int(.30*size), int(.36*size)), fill=(100, 139, 161, 150), width=2)
        draw.line((int(.78*size), int(.18*size), int(.78*size), int(.61*size)), fill=(73, 105, 124, 150), width=2)
        draw.line((int(.78*size), int(.25*size), int(.95*size), int(.25*size)), fill=(73, 105, 124, 150), width=2)
    elif scene["id"] == "dark_luxury":
        image = _gradient(size, (42 + jitter, 42 + jitter, 44 + jitter), (15, 16, 19))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(20, 20, 22, 255))
        for x in (int(.15*size), int(.84*size)):
            draw.line((x, int(.08*size), x, floor_y), fill=(181, 145, 74, 120), width=max(1, size//128))
        draw.rectangle((int(.72*size), int(.18*size), int(.90*size), int(.43*size)),
                       outline=(196, 158, 81, 130), width=max(1, size//96))
        _soft_light(image, (int(.18*size), int(.02*size), int(.83*size), int(.85*size)), (207, 149, 70), 70)
    elif scene["id"] == "terracotta_gallery":
        image = _gradient(size, (207 + jitter, 139 + jitter, 103 + jitter), (153, 79, 59))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(126, 66, 53, 255))
        draw.rounded_rectangle((int(.04*size), int(.10*size), int(.37*size), int(.78*size)),
                               radius=int(.17*size), fill=(232, 177, 132, 155))
        draw.polygon(((0, int(.18*size)), (int(.55*size), floor_y),
                      (int(.73*size), floor_y), (int(.15*size), int(.12*size))),
                     fill=(255, 219, 157, 42))
        draw.ellipse((int(.78*size), int(.54*size), int(.91*size), int(.70*size)), fill=(110, 58, 47, 150))
    elif scene["id"] == "sage_reading":
        image = _gradient(size, (191 + jitter, 205 + jitter, 180 + jitter), (123, 147, 119))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(111, 126, 96, 255))
        draw.rectangle((int(.08*size), int(.17*size), int(.27*size), int(.42*size)),
                       fill=(224, 218, 191, 105), outline=(87, 108, 81, 170), width=2)
        stem_x = int(.88*size)
        draw.line((stem_x, int(.43*size), int(.82*size), floor_y), fill=(44, 77, 48, 190), width=max(2, size//64))
        for y0, side in ((.45, -.10), (.50, .08), (.56, -.09), (.61, .07)):
            x0, yy = int(.85*size), int(y0*size)
            draw.ellipse((x0 + int(side*size), yy-int(.05*size), x0+int(side*size)+int(.11*size), yy+int(.04*size)),
                         fill=(52, 91, 57, 175))
    else:
        image = _gradient(size, (201 + jitter, 202 + jitter, 199 + jitter), (126, 129, 130))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, floor_y, size, size), fill=(105, 108, 109, 255))
        for x in (int(.05*size), int(.23*size), int(.41*size)):
            draw.rectangle((x, int(.08*size), x+int(.14*size), int(.55*size)),
                           fill=(216, 224, 226, 155), outline=(93, 99, 101, 120), width=1)
        draw.polygon(((0, int(.20*size)), (int(.48*size), floor_y),
                      (int(.70*size), floor_y), (int(.12*size), int(.12*size))),
                     fill=(255, 255, 246, 42))

    # Fine low-amplitude surface variation avoids a synthetic flat-color look.
    array = np.asarray(image).astype(np.int16)
    noise = rng.normal(0.0, 1.4, (size, size, 1))
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    root = path.parent.resolve()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["category"] != "lamp":
            continue
        row["image_path"] = (root / row["image_path"]).resolve()
        mask_path = row.get("mask_path")
        row["mask_path"] = (root / mask_path).resolve() if mask_path else None
        rows.append(row)
    if not rows:
        raise ValueError(f"No lamp rows in {path}")
    return rows


def _fallback_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image).astype(np.float32) / 255.0
    distance = 1.0 - rgb.min(axis=2)
    mask = Image.fromarray((distance > 0.035).astype(np.uint8) * 255, "L")
    return mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.8))


def _normalize_product(
    image: Image.Image, mask: Image.Image, size: int,
) -> tuple[Image.Image, Image.Image]:
    """Tightly frame the product while retaining its exact pixels and alpha."""

    box = mask.getbbox()
    if box is None:
        return (
            ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS),
            ImageOps.fit(mask, (size, size), method=Image.Resampling.LANCZOS),
        )
    product = image.crop(box)
    product_mask = mask.crop(box)
    max_width, max_height = int(size * 0.68), int(size * 0.78)
    scale = min(max_width / product.width, max_height / product.height)
    resized_size = (
        max(1, round(product.width * scale)),
        max(1, round(product.height * scale)),
    )
    product = product.resize(resized_size, Image.Resampling.LANCZOS)
    product_mask = product_mask.resize(resized_size, Image.Resampling.LANCZOS)
    reference = Image.new("RGB", (size, size), "white")
    normalized_mask = Image.new("L", (size, size), 0)
    left = (size - resized_size[0]) // 2
    top = max(0, int(size * 0.88) - resized_size[1])
    reference.paste(product, (left, top), product_mask)
    normalized_mask.paste(product_mask, (left, top))
    return reference, normalized_mask


class ProductSceneDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_dir: Path, split: str, image_size: int, instruction_cache: Path) -> None:
        self.rows = _load_manifest(data_dir / f"{split}.jsonl")
        self.image_size = int(image_size)
        payload = torch.load(instruction_cache, map_location="cpu", weights_only=False)
        self.instructions = payload["rows"]
        self.text = payload["text"].float()
        self.lookup = {
            (int(row["scene_index"]), int(row["variant"])): index
            for index, row in enumerate(self.instructions)
        }

    def __len__(self) -> int:
        return len(self.rows) * len(SCENES)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index // len(SCENES)]
        scene_index = index % len(SCENES)
        with Image.open(row["image_path"]) as handle:
            source = handle.convert("RGB")
        if row["mask_path"] is not None:
            with Image.open(row["mask_path"]) as handle:
                source_mask = handle.convert("L")
        else:
            source_mask = _fallback_mask(source)
        reference, mask_image = _normalize_product(source, source_mask, self.image_size)
        background = render_background(scene_index, self.image_size, row["sample_id"])
        target = Image.composite(reference, background, mask_image)
        variant = _seed(row["sample_id"], SCENES[scene_index]["id"]) % len(
            SCENES[scene_index]["prompts"]
        )
        prompt_index = self.lookup[(scene_index, variant)]

        def tensor(image: Image.Image) -> torch.Tensor:
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(127.5).sub(1)

        foreground_mask = torch.from_numpy(np.asarray(mask_image).copy()).float()[None].div(255)
        return {
            "reference": tensor(reference),
            "target": tensor(target),
            "foreground_mask": foreground_mask,
            "background_mask": 1.0 - foreground_mask,
            "qwen_text": self.text[prompt_index],
            "prompt": self.instructions[prompt_index]["prompt"],
            "sample_id": f"{row['sample_id']}:{SCENES[scene_index]['id']}",
            "source_id": row["sample_id"],
            "scene_index": scene_index,
            "scene_id": SCENES[scene_index]["id"],
            "scene_label": SCENES[scene_index]["label"],
        }


def write_pair_manifest(dataset: ProductSceneDataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            value = dataset[index]
            row = dataset.rows[index // len(SCENES)]
            handle.write(json.dumps({
                "sample_id": value["sample_id"],
                "source_id": value["source_id"],
                "sequence_id": row["sequence_id"],
                "scene_id": value["scene_id"],
                "scene_label": value["scene_label"],
                "prompt": value["prompt"],
                "source_image": str(row["image_path"]),
                "license": row["license"],
                "source_url": row["source_url"],
                "background_provenance": "deterministic procedural renderer; no external image",
            }) + "\n")


__all__ = [
    "ProductSceneDataset", "SCENES", "build_instruction_cache", "instruction_rows",
    "render_background", "write_pair_manifest",
]
