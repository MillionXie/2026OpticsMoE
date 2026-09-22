from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

from PIL import Image

from LightGenV2.tasks.t12_text_to_image.dataset import read_manifest
from LightGenV2.tasks.t12_text_to_image.prepare_abo_cleanrender import (
    PREFIX,
    _category_title_allowed,
    prepare,
)


def _fake_abo(root: Path, archive: Path) -> None:
    (root / "listings" / "metadata").mkdir(parents=True)
    (root / "LICENSE-CC-BY-4.0.txt").write_text("CC BY 4.0 test fixture", encoding="utf-8")
    rows = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for category in ("CHAIR", "LAMP", "TABLE"):
            for identity in range(3):
                item = f"{category}-{identity}"
                rows.append({
                    "item_id": item,
                    "product_type": [{"value": category}],
                    "item_name": [{"language_tag": "en_US", "value": f"Blue {category} {identity}"}],
                    "color": [{"language_tag": "en_US", "value": "navy blue"}],
                    "material": [{"language_tag": "en_US", "value": "wood"}],
                })
                for view in range(4):
                    image = Image.new("RGBA", (48, 32), (20 + identity * 20, 40, 180, 255))
                    payload = __import__("io").BytesIO()
                    image.save(payload, format="PNG")
                    output.writestr(f"{PREFIX}{item}/{item}_{view:02d}.png", payload.getvalue())
        output.writestr(f"{PREFIX}README.txt", "fixture")
    with gzip.open(root / "listings" / "metadata" / "listings_0.json.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_cleanrender_fetches_views_and_splits_by_object(tmp_path: Path) -> None:
    root, archive = tmp_path / "abo", tmp_path / "renders.zip"
    _fake_abo(root, archive)
    output = tmp_path / "prepared"
    report = prepare(
        root, output, ["CHAIR", "LAMP", "TABLE"], archive_source=archive,
        train_instances=1, val_instances=1, test_instances=1,
        train_views=3, eval_views=2, image_size=32, seed=7,
    )
    assert report["splits"] == {"train": 9, "val": 6, "test": 6}
    assert report["object_sequences"] == {"train": 3, "val": 3, "test": 3}
    identities = {
        split: {row.sequence_id for row in read_manifest(output / f"{split}.jsonl")}
        for split in ("train", "val", "test")
    }
    assert not identities["train"] & identities["val"]
    assert not identities["train"] & identities["test"]
    assert not identities["val"] & identities["test"]
    with Image.open(next((output / "images").rglob("*.jpg"))) as image:
        assert image.size == (32, 32)
    assert (output / "contact_sheet.jpg").is_file()
    assert "navy blue wood" in (output / "train.jsonl").read_text(encoding="utf-8")


def test_pillow_alias_filter_removes_chair_like_products() -> None:
    assert not _category_title_allowed("PILLOW", "Outdoor high back patio chair cushion")
    assert not _category_title_allowed("PILLOW", "Red lounger patio cushion")
    assert _category_title_allowed("PILLOW", "Modern geometric throw pillow")
