from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.product_global_redesign_data import (
    ProductGlobalRedesignDataset,
    instruction_rows,
)


def _fixture(root: Path) -> Path:
    for split in ("train", "val", "test"):
        rows = []
        for category, box, color in (
            ("lamp", (28, 8, 36, 56), (30, 30, 30)),
            ("table", (13, 30, 51, 50), (120, 80, 40)),
            ("chair", (18, 24, 46, 57), (90, 45, 25)),
        ):
            image_dir = root / "images" / split / category
            mask_dir = root / "masks" / split / category
            image_dir.mkdir(parents=True); mask_dir.mkdir(parents=True)
            image = Image.new("RGB", (64, 64), "white")
            ImageDraw.Draw(image).rectangle(box, fill=color)
            mask = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(mask).rectangle(box, fill=255)
            image.save(image_dir / "item.jpg"); mask.save(mask_dir / "item.png")
            rows.append({
                "sample_id": f"{category}-{split}", "sequence_id": f"{category}-{split}",
                "category": category, "caption": f"a {category}",
                "image_path": f"images/{split}/{category}/item.jpg",
                "mask_path": f"masks/{split}/{category}/item.png", "license": "CC BY 4.0",
            })
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    rows = instruction_rows(root)
    cache = root / "redesign_instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 32)}, cache)
    return cache


def test_redesign_is_full_frame_and_excludes_chairs(tmp_path: Path) -> None:
    cache = _fixture(tmp_path)
    dataset = ProductGlobalRedesignDataset(tmp_path, "train", 64, cache)
    assert len(dataset) == 4
    assert {dataset[index]["category"] for index in range(len(dataset))} == {"lamp", "table"}
    assert all("chair" not in dataset[index]["sample_id"] for index in range(len(dataset)))
    value = dataset[0]
    assert value["reference"].shape == value["target"].shape == (3, 64, 64)
    assert not torch.equal(value["reference"], value["target"])
    assert "entire image" in value["prompt"].lower() or "complete new image" in value["prompt"].lower() or "fresh" in value["prompt"].lower()
