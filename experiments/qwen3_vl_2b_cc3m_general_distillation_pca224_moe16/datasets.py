from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: Path
    caption: str
    source_line: int


class ImageCaptionDataset(Dataset):
    def __init__(self, rows: Sequence[ManifestRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, int]:
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            rgb = image.convert("RGB").copy()
        return rgb, row.caption, index


@dataclass
class DatasetBundle:
    train: ImageCaptionDataset
    validation: ImageCaptionDataset
    calibration: ImageCaptionDataset
    metadata: dict[str, Any]


def load_jsonl_dataset(settings: Any, *, persist_split: bool = True) -> DatasetBundle:
    rows, manifest_digest = read_manifest(
        settings.manifest_path,
        row_limit=settings.sample_limit,
    )
    if len(rows) < 2:
        raise RuntimeError("The manifest must contain at least two valid image-caption rows")
    if settings.validation_manifest_path is not None:
        validation_rows, validation_digest = read_manifest(settings.validation_manifest_path)
        train_rows = rows
        split_kind = "explicit_validation_manifest"
    else:
        order = list(range(len(rows)))
        random.Random(settings.seed).shuffle(order)
        validation_count = max(1, round(len(rows) * settings.validation_fraction))
        validation_indexes = set(order[:validation_count])
        train_rows = [row for index, row in enumerate(rows) if index not in validation_indexes]
        validation_rows = [row for index, row in enumerate(rows) if index in validation_indexes]
        validation_digest = manifest_digest
        split_kind = "deterministic_manifest_row_split"
    overlap = {row.sample_id for row in train_rows} & {row.sample_id for row in validation_rows}
    if overlap:
        raise RuntimeError(f"Train/validation manifests share sample_id values: {sorted(overlap)[:5]}")
    calibration_count = min(int(settings.calibration_sample_count), len(train_rows))
    calibration_order = list(range(len(train_rows)))
    random.Random(settings.seed + 17).shuffle(calibration_order)
    calibration_rows = [train_rows[index] for index in calibration_order[:calibration_count]]
    split_payload = {
        "dataset": "cc3m_jsonl",
        "manifest_path": str(settings.manifest_path),
        "manifest_sha256": manifest_digest,
        "validation_manifest_path": (
            str(settings.validation_manifest_path) if settings.validation_manifest_path else None
        ),
        "validation_manifest_sha256": validation_digest,
        "split_kind": split_kind,
        "seed": settings.seed,
        "sample_limit": settings.sample_limit,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "calibration_samples": len(calibration_rows),
        "train_sample_ids_sha256": _ids_digest(train_rows),
        "validation_sample_ids_sha256": _ids_digest(validation_rows),
    }
    split_digest = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    split_payload["split_digest"] = split_digest
    settings.manifest_digest = split_digest
    if persist_split:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_dir / "data_split.json").write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return DatasetBundle(
        train=ImageCaptionDataset(train_rows),
        validation=ImageCaptionDataset(validation_rows),
        calibration=ImageCaptionDataset(calibration_rows),
        metadata=split_payload,
    )


def read_manifest(
    path: Path,
    row_limit: int | None = None,
) -> tuple[list[ManifestRow], str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CC3M JSONL manifest does not exist: {path}. Each line must contain "
            "sample_id, image_path, and caption."
        )
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
    digest = hasher.hexdigest()
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    missing_images: list[str] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            missing_fields = [
                name for name in ("sample_id", "image_path", "caption")
                if name not in value
            ]
            if missing_fields:
                raise RuntimeError(
                    f"{path} line {line_number} is missing fields: {missing_fields}"
                )
            sample_id = str(value["sample_id"]).strip()
            caption = str(value["caption"]).strip()
            if not sample_id or not caption:
                raise RuntimeError(
                    f"{path} line {line_number} has an empty sample_id or caption"
                )
            if sample_id in seen:
                raise RuntimeError(
                    f"Duplicate sample_id {sample_id!r} in {path} line {line_number}"
                )
            seen.add(sample_id)
            image_path = Path(str(value["image_path"])).expanduser()
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            if not image_path.is_file():
                missing_images.append(str(image_path))
                if len(missing_images) >= 10:
                    break
            rows.append(ManifestRow(sample_id, image_path, caption, line_number))
            if row_limit is not None and len(rows) >= int(row_limit):
                break
    if missing_images:
        raise FileNotFoundError(
            "Manifest image files are missing. First attempted paths:\n" + "\n".join(missing_images)
        )
    return rows, digest


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    import torch

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _collate(batch: Sequence[tuple[Image.Image, str, int]]):
    images, captions, indexes = zip(*batch)
    return list(images), list(captions), list(indexes)


def _ids_digest(rows: Sequence[ManifestRow]) -> str:
    return hashlib.sha256("\n".join(row.sample_id for row in rows).encode("utf-8")).hexdigest()
