from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.premium_material_data import (
    PremiumMaterialDataset,
    instruction_rows,
)


def _write_dataset(root: Path, categories: tuple[str, ...]) -> None:
    for split in ("train", "val", "test"):
        rows = []
        for index, category in enumerate(categories):
            folder = root / "images" / split / category; folder.mkdir(parents=True)
            image = Image.new("RGB", (64, 64), "white")
            ImageDraw.Draw(image).rounded_rectangle((18, 9, 46, 57), radius=4, fill=(40+index*50, 45, 50))
            image.save(folder / "item.jpg")
            rows.append({
                "sample_id": f"{category}-{split}", "sequence_id": f"{category}-{split}",
                "category": category, "caption": f"a {category}",
                "image_path": f"images/{split}/{category}/item.jpg", "mask_path": None,
                "license": "CC BY 4.0",
            })
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def test_premium_material_pairs_cover_non_chair_categories(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "abo_cleanrender_v1", ("lamp", "table", "chair"))
    _write_dataset(tmp_path / "abo_backpack_style_v2", ("backpack",))
    rows = instruction_rows(); cache = tmp_path / "instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 48)}, cache)
    dataset = PremiumMaterialDataset(tmp_path, "train", 64, cache)
    assert len(dataset) == 12
    categories = {dataset[index]["category"] for index in range(len(dataset))}
    assert categories == {"lamp", "table", "backpack"}
    assert "chair" not in categories
    sample = dataset[0]
    assert sample["reference"].shape == sample["target"].shape == (3, 64, 64)
    assert not torch.equal(sample["reference"], sample["target"])
    assert "premium" in sample["prompt"].lower() or "luxury" in sample["prompt"].lower()
