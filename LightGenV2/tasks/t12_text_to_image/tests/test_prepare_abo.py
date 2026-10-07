from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw

from LightGenV2.tasks.t12_text_to_image.prepare_abo import prepare


def test_prepare_abo_uses_one_product_identity_per_split(tmp_path: Path) -> None:
    root = tmp_path / "abo"
    (root / "images" / "metadata").mkdir(parents=True)
    (root / "images" / "small" / "aa").mkdir(parents=True)
    (root / "listings" / "metadata").mkdir(parents=True)
    (root / "LICENSE-CC-BY-4.0.txt").write_text("CC BY 4.0", encoding="utf-8")

    image_rows = []
    listings = []
    for index in range(3):
        image_id = f"image-{index}"
        path = f"aa/{index}.jpg"
        image = Image.new("RGB", (160, 160), "white")
        ImageDraw.Draw(image).rectangle((40, 30, 120, 130), fill="navy")
        image.save(root / "images" / "small" / path)
        image_rows.append({"image_id": image_id, "height": 160, "width": 160, "path": path})
        listings.append({
            "item_id": f"item-{index}",
            "main_image_id": image_id,
            "product_type": [{"value": "CHAIR"}],
            "item_name": [{"language_tag": "en_US", "value": f"Blue chair {index}"}],
            "color": [{"language_tag": "en_US", "value": "Navy Blue"}],
            "material": [{"language_tag": "en_US", "value": "Wood"}],
        })

    with gzip.open(root / "images" / "metadata" / "images.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "height", "width", "path"))
        writer.writeheader()
        writer.writerows(image_rows)
    with gzip.open(root / "listings" / "metadata" / "listings_0.json.gz", "wt") as handle:
        for row in listings:
            handle.write(json.dumps(row) + "\n")

    output = tmp_path / "prepared"
    report = prepare(
        root, output, ["CHAIR"], train_instances=1, val_instances=1,
        test_instances=1, image_size=64, seed=5,
    )
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    assert report["one_main_image_per_product"] is True
    assert len(list((output / "images").rglob("*.jpg"))) == 3
    assert "navy blue wood chair" in (output / "train.jsonl").read_text(encoding="utf-8")
