from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.product_scene_data import (
    ProductSceneDataset,
    SCENES,
    instruction_rows,
    render_background,
)


def _fixture(root: Path) -> Path:
    (root / "images" / "train" / "lamp").mkdir(parents=True)
    (root / "masks" / "train" / "lamp").mkdir(parents=True)
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split / "lamp"
        mask_dir = root / "masks" / split / "lamp"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((26, 12, 38, 53), fill=(25, 25, 25))
        mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(mask).rectangle((26, 12, 38, 53), fill=255)
        image.save(image_dir / "lamp.jpg")
        mask.save(mask_dir / "lamp.png")
        row = {
            "sample_id": f"lamp-{split}", "sequence_id": f"identity-{split}",
            "category": "lamp", "caption": "a lamp", "image_path": f"images/{split}/lamp/lamp.jpg",
            "mask_path": f"masks/{split}/lamp/lamp.png", "license": "CC BY 4.0",
            "source_url": "https://example.test/abo",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = instruction_rows()
    cache = root / "instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 24)}, cache)
    return cache


def test_scene_dataset_has_same_input_and_six_text_selected_targets(tmp_path: Path) -> None:
    cache = _fixture(tmp_path)
    dataset = ProductSceneDataset(tmp_path, "train", 64, cache)
    assert len(dataset) == len(SCENES)
    values = [dataset[index] for index in range(len(SCENES))]
    assert all(torch.equal(values[0]["reference"], value["reference"]) for value in values[1:])
    assert len({value["prompt"] for value in values}) == len(SCENES)
    assert any(not torch.equal(values[0]["target"], value["target"]) for value in values[1:])
    assert torch.equal(values[0]["foreground_mask"] + values[0]["background_mask"], torch.ones(1, 64, 64))


def test_scene_backgrounds_are_deterministic_and_distinct() -> None:
    first = [render_background(index, 64, "lamp-1") for index in range(len(SCENES))]
    second = [render_background(index, 64, "lamp-1") for index in range(len(SCENES))]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
    assert len({image.tobytes() for image in first}) == len(SCENES)
