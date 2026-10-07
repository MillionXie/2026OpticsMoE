from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.prepare_uco3d import prepare


def test_prepare_creates_clean_identity_disjoint_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for sequence_index in range(3):
        for frame_index in range(2):
            image_path = source / f"rgb_{sequence_index}_{frame_index}.png"
            mask_path = source / f"mask_{sequence_index}_{frame_index}.png"
            image = Image.new("RGB", (40, 30), "navy")
            mask = Image.new("L", (40, 30), 0)
            ImageDraw.Draw(mask).rectangle((10, 5, 29, 24), fill=255)
            image.save(image_path)
            mask.save(mask_path)
            rows.append({
                "sequence_id": f"object-{sequence_index}", "category": "mug",
                "caption": "a blue ceramic mug", "frame_path": image_path.name,
                "mask_path": mask_path.name, "source_url": "https://uco3d.github.io/",
                "license": "CC BY 4.0",
            })
    index = source / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "prepared"
    report = prepare(
        index, output, ["mug"], train_instances=1, val_instances=1,
        test_instances=1, frames_per_instance=2, image_size=64, seed=3,
    )
    assert report["splits"] == {"train": 2, "val": 2, "test": 2}
    images = list((output / "images").rglob("*.jpg"))
    assert len(images) == 6
    assert all(Image.open(path).size == (64, 64) for path in images)
