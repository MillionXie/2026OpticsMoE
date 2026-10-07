from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.product_object_replace_data import (
    TARGET_CATEGORIES,
    ProductObjectReplacementDataset,
    instruction_rows,
)


def _fixture(root: Path) -> Path:
    for split in ("train", "val", "test"):
        rows = []
        for category, box, color in (
            ("lamp", (28, 8, 36, 56), (30, 30, 30)),
            ("chair", (18, 24, 46, 57), (90, 45, 25)),
            ("table", (13, 30, 51, 50), (120, 80, 40)),
        ):
            image_dir = root / "images" / split / category
            mask_dir = root / "masks" / split / category
            image_dir.mkdir(parents=True); mask_dir.mkdir(parents=True)
            image = Image.new("RGB", (64, 64), "white"); ImageDraw.Draw(image).rectangle(box, fill=color)
            mask = Image.new("L", (64, 64), 0); ImageDraw.Draw(mask).rectangle(box, fill=255)
            image.save(image_dir / "item.jpg"); mask.save(mask_dir / "item.png")
            rows.append({
                "sample_id": f"{category}-{split}", "sequence_id": f"{category}-identity-{split}",
                "category": category, "caption": f"a blue {category} on a clean white studio background",
                "image_path": f"images/{split}/{category}/item.jpg",
                "mask_path": f"masks/{split}/{category}/item.png", "license": "CC BY 4.0",
                "source_url": "https://example.test/abo",
            })
        (root/f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    rows = instruction_rows(root)
    cache = root / "object_instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 32)}, cache)
    return cache


def test_object_replacement_preserves_scene_and_changes_object(tmp_path: Path) -> None:
    cache = _fixture(tmp_path)
    dataset = ProductObjectReplacementDataset(tmp_path, "train", 64, cache)
    assert len(dataset) == len(TARGET_CATEGORIES)
    value = dataset[0]
    assert value["target_category"] in TARGET_CATEGORIES
    assert not torch.equal(value["reference"], value["target"])
    assert torch.allclose(value["edit_mask"] + value["preserve_mask"], torch.ones(1,64,64))
    outside = value["preserve_mask"].expand_as(value["reference"]) > 0.99
    assert torch.equal(value["reference"][outside], value["target"][outside])
    assert "preserve" in value["prompt"].lower() or "keeping" in value["prompt"].lower()
