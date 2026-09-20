"""Manifest and cached-feature datasets for the curated CC BY 4.0 subset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = {
    "sample_id", "sequence_id", "category", "caption", "image_path",
    "license", "source_url",
}
CC_BY_4_IDENTIFIERS = {
    "CC BY 4.0",
    "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sequence_id: str
    category: str
    caption: str
    image_path: Path
    license: str
    source_url: str


def read_manifest(path: Path, *, verify_images: bool = True) -> list[Sample]:
    root = path.parent.resolve()
    samples: list[Sample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
        if row["license"] not in CC_BY_4_IDENTIFIERS:
            raise ValueError(f"{path}:{line_number} is not explicitly CC BY 4.0")
        image_value = Path(row["image_path"])
        if image_value.is_absolute():
            raise ValueError(f"{path}:{line_number} image_path must be relative to the dataset root")
        image = (root / image_value).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"{path}:{line_number} image_path escapes the dataset root")
        if verify_images and not image.is_file():
            raise FileNotFoundError(image)
        samples.append(Sample(
            sample_id=str(row["sample_id"]), sequence_id=str(row["sequence_id"]),
            category=str(row["category"]), caption=str(row["caption"]),
            image_path=image, license=str(row["license"]), source_url=str(row["source_url"]),
        ))
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError(f"Duplicate sample_id in {path}")
    return samples


def validate_split_contract(data_dir: Path, *, verify_images: bool = True) -> dict[str, Any]:
    by_split = {
        split: read_manifest(data_dir / f"{split}.jsonl", verify_images=verify_images)
        for split in ("train", "val", "test")
    }
    sequence_sets = {split: {x.sequence_id for x in rows} for split, rows in by_split.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sequence_sets[left] & sequence_sets[right]
        if overlap:
            raise ValueError(f"Object identity leakage between {left}/{right}: {sorted(overlap)[:5]}")
    categories = {split: {x.category for x in rows} for split, rows in by_split.items()}
    if not categories["train"] or any(value != categories["train"] for value in categories.values()):
        raise ValueError("Train/val/test must contain the same non-empty category set")
    return {
        "schema_version": 1,
        "dataset": "curated CC BY 4.0 single-object T2I subset",
        "license": "CC BY 4.0",
        "splits": {split: len(rows) for split, rows in by_split.items()},
        "object_sequences": {split: len(sequence_sets[split]) for split in by_split},
        "categories": sorted(categories["train"]),
        "manifest_sha256": {split: sha256(data_dir / f"{split}.jsonl") for split in by_split},
        "identity_split": True,
    }


class CachedLatentDataset(Dataset[dict[str, Any]]):
    """Memory-mapped-style access to one immutable split feature cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        required = {"meta", "sample_ids", "text", "latent", "categories"}
        if not required.issubset(self.payload):
            raise ValueError(f"Invalid T12 cache {path}; missing {sorted(required - self.payload.keys())}")
        length = len(self.payload["sample_ids"])
        if any(len(self.payload[key]) != length for key in ("text", "latent", "categories")):
            raise ValueError(f"Inconsistent cache lengths in {path}")

    def __len__(self) -> int:
        return len(self.payload["sample_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": self.payload["sample_ids"][index],
            "text": self.payload["text"][index].float(),
            "latent": self.payload["latent"][index].float(),
            "category": self.payload["categories"][index],
        }


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = [
    "CC_BY_4_IDENTIFIERS", "CachedLatentDataset", "Sample", "read_manifest",
    "sha256", "validate_split_contract", "write_manifest",
]
