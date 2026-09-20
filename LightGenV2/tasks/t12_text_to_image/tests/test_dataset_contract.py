from __future__ import annotations

from pathlib import Path

import pytest

from LightGenV2.tasks.t12_text_to_image.dataset import validate_split_contract, write_manifest


def row(split: str, index: int, sequence: str, license_name: str = "CC BY 4.0"):
    return {
        "sample_id": f"{split}_{index}",
        "sequence_id": sequence,
        "category": "mug",
        "caption": "a red ceramic mug",
        "image_path": f"images/{split}_{index}.jpg",
        "license": license_name,
        "source_url": "https://uco3d.github.io/",
    }


def test_manifest_requires_exact_license_and_object_disjoint_splits(tmp_path: Path) -> None:
    write_manifest(tmp_path / "train.jsonl", [row("train", 0, "object-a")])
    write_manifest(tmp_path / "val.jsonl", [row("val", 0, "object-b")])
    write_manifest(tmp_path / "test.jsonl", [row("test", 0, "object-c")])
    report = validate_split_contract(tmp_path, verify_images=False)
    assert report["license"] == "CC BY 4.0"
    assert report["identity_split"] is True


def test_manifest_rejects_identity_leakage(tmp_path: Path) -> None:
    write_manifest(tmp_path / "train.jsonl", [row("train", 0, "same")])
    write_manifest(tmp_path / "val.jsonl", [row("val", 0, "same")])
    write_manifest(tmp_path / "test.jsonl", [row("test", 0, "other")])
    with pytest.raises(ValueError, match="identity leakage"):
        validate_split_contract(tmp_path, verify_images=False)


def test_manifest_rejects_non_exact_cc_by_version(tmp_path: Path) -> None:
    write_manifest(tmp_path / "train.jsonl", [row("train", 0, "a", "CC BY 2.0")])
    write_manifest(tmp_path / "val.jsonl", [row("val", 0, "b")])
    write_manifest(tmp_path / "test.jsonl", [row("test", 0, "c")])
    with pytest.raises(ValueError, match="not explicitly CC BY 4.0"):
        validate_split_contract(tmp_path, verify_images=False)
