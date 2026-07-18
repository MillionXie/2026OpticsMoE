from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .io_utils import write_json
from .settings import Settings


IMAGE_COLUMNS = (
    "image_name", "image", "filename", "file_name", "img", "image_path", "dist_img"
)
TASK_COLUMN_ALIASES = {
    "Brightness": ("brightness",),
    "Colorfulness": ("colorfulness", "colourfulness"),
    "Contrast": ("contrast",),
}
SPREADSHEET_SUFFIXES = {".csv", ".xlsx", ".xls"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SPAQImageRecord:
    image_name: str
    image_path: Path
    scores: dict[str, float]


@dataclass(frozen=True)
class DatasetBundle:
    train: "SPAQSingleAttributeDataset"
    test: "SPAQSingleAttributeDataset"
    train_records: list[SPAQImageRecord]
    test_records: list[SPAQImageRecord]
    metadata: dict[str, Any]
    cache_identity: dict[str, Any]


class SPAQSingleAttributeDataset(Dataset[tuple[Image.Image, float]]):
    """RGB SPAQ images paired with one configured attribute normalized to [0,1]."""

    def __init__(self, records: Sequence[SPAQImageRecord], task_name: str) -> None:
        self.records = list(records)
        self.task_name = task_name
        self.targets = [float(record.scores[task_name]) / 100.0 for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, float]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB").copy()
        return rgb, self.targets[index]

    def sample_metadata(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {"image_name": record.image_name, "image_path": str(record.image_path),
                "task": self.task_name, "score": float(record.scores[self.task_name]),
                "normalized_score": self.targets[index]}


def load_spaq(settings: Settings, persist_split: bool = True) -> DatasetBundle:
    annotation_path, frame, columns = discover_annotations(
        settings.data_root, settings.annotations_file, settings.task_name
    )
    image_column = _required_column(columns, IMAGE_COLUMNS, "image filename")
    task_name = settings.task_name
    task_columns = {task_name: _required_column(
        columns, TASK_COLUMN_ALIASES[task_name], f"{task_name} score"
    )}
    records = _build_records(
        frame,
        image_column,
        task_columns,
        settings.data_root,
        settings.image_dir,
    )
    split_path = settings.output_dir / "data_split.json"
    train_names, test_names, split_digest = _load_or_create_split(
        records,
        split_path,
        settings.train_fraction,
        settings.seed,
        persist_split,
    )
    by_name = {record.image_name: record for record in records}
    full_train = [by_name[name] for name in train_names]
    full_test = [by_name[name] for name in test_names]
    train_records = _limit_images(full_train, settings.train_image_limit)
    test_records = _limit_images(full_test, settings.test_image_limit)
    metadata = {
        "dataset": "SPAQ",
        "task": task_name,
        "input_color_mode": "RGB",
        "annotation_file": str(annotation_path),
        "image_root": str(settings.image_dir or settings.data_root),
        "image_column": image_column,
        "task_columns": task_columns,
        "source_images": len(records),
        "full_train_images": len(full_train),
        "full_test_images": len(full_test),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_fraction": settings.train_fraction,
        "test_fraction": 1.0 - settings.train_fraction,
        "validation_images": 0,
        "split_seed": settings.seed,
        "split_digest": split_digest,
        "language_model_used": False,
        "prompt_used": False,
        "label_scale": [0.0, 100.0],
        "training_label_scale": [0.0, 1.0],
        "score_statistics": _score_statistics(records),
        "train_image_limit": settings.train_image_limit, "test_image_limit": settings.test_image_limit,
        "train_samples_per_epoch": settings.train_samples_per_epoch,
    }
    cache_identity = {
        "dataset": "spaq_single_attribute",
        "task": task_name,
        "annotation_file": str(annotation_path),
        "image_column": image_column,
        "task_columns": task_columns,
        "split_digest": split_digest,
    }
    return DatasetBundle(
        train=SPAQSingleAttributeDataset(train_records, task_name),
        test=SPAQSingleAttributeDataset(test_records, task_name),
        train_records=train_records,
        test_records=test_records,
        metadata=metadata,
        cache_identity=cache_identity,
    )


def discover_annotations(
    data_root: Path, configured_path: Path | None, task_name: str
) -> tuple[Path, Any, list[str]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"SPAQ data_root does not exist: {data_root}")
    candidates = [configured_path] if configured_path is not None else sorted(
        path for path in data_root.rglob("*") if path.suffix.lower() in SPREADSHEET_SUFFIXES
    )
    if configured_path is not None and not configured_path.is_file():
        raise FileNotFoundError(f"Configured annotations_file does not exist: {configured_path}")
    inspections: list[tuple[Path, list[str], Any]] = []
    valid: list[tuple[Path, Any, list[str]]] = []
    for path in candidates:
        try:
            frame = _read_table(path)
            columns = [str(column) for column in frame.columns]
        except Exception as exc:
            inspections.append((path, [f"<read error: {exc}>"], None))
            continue
        inspections.append((path, columns, frame))
        if _has_required_columns(columns, task_name):
            valid.append((path, frame, columns))
    if configured_path is not None and inspections and not valid:
        raise RuntimeError(_annotation_error(data_root, inspections))
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        # SPAQ also ships EXIF_tags.xlsx with a camera-metadata column named
        # Brightness. Prefer the subjective IQA table, identifiable by having
        # more of the supported perceptual score columns.
        quality_column_counts = [_quality_column_count(columns) for _, _, columns in valid]
        best_count = max(quality_column_counts)
        preferred = [entry for entry, count in zip(valid, quality_column_counts) if count == best_count]
        if len(preferred) == 1:
            return preferred[0]
        details = "\n".join(f"- {path}: {columns}" for path, _, columns in valid)
        raise RuntimeError(
            "Multiple SPAQ annotation files contain all required columns. Set annotations_file "
            f"explicitly. Candidates:\n{details}"
        )
    raise RuntimeError(_annotation_error(data_root, inspections))


def targets_of(dataset: Dataset[Any]) -> list[float]:
    if hasattr(dataset, "targets"):
        return [float(value) for value in dataset.targets]
    if isinstance(dataset, Subset):
        parent = targets_of(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset does not expose normalized single-attribute targets")


def sample_metadata(dataset: Dataset[Any], index: int) -> dict[str, Any]:
    if isinstance(dataset, Subset):
        return sample_metadata(dataset.dataset, int(dataset.indices[index]))
    if hasattr(dataset, "sample_metadata"):
        return dataset.sample_metadata(index)
    return {"sample_index": index}


class IndexedDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, float, int]:
        image, target = self.dataset[index]
        return image, float(target), index


def indexed_collate(batch: Sequence[Any]) -> tuple[list[Image.Image], torch.Tensor, torch.Tensor]:
    images, targets, indices = zip(*batch)
    return list(images), torch.tensor(targets, dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def make_indexed_loader(dataset: Dataset[Any], batch_size: int, workers: int,
                        shuffle: bool, seed: int) -> DataLoader[Any]:
    return DataLoader(
        IndexedDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=indexed_collate, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _read_table(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read SPAQ CSV/Excel annotations") from exc
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _normalize_column(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def _column_lookup(columns: Sequence[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column(column)
        if normalized in lookup:
            raise RuntimeError(
                f"Annotation columns {lookup[normalized]!r} and {column!r} normalize to the same name"
            )
        lookup[normalized] = str(column)
    return lookup


def _required_column(columns: Sequence[str], aliases: Sequence[str], purpose: str) -> str:
    lookup = _column_lookup(columns)
    matches = [lookup[_normalize_column(alias)] for alias in aliases if _normalize_column(alias) in lookup]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {purpose} column from aliases {list(aliases)}; "
            f"found {matches}. Annotation columns: {list(columns)}"
        )
    return matches[0]


def _has_required_columns(columns: Sequence[str], task_name: str) -> bool:
    try:
        _required_column(columns, IMAGE_COLUMNS, "image filename")
        _required_column(columns, TASK_COLUMN_ALIASES[task_name], f"{task_name} score")
    except RuntimeError:
        return False
    return True


def _quality_column_count(columns: Sequence[str]) -> int:
    lookup = _column_lookup(columns)
    return sum(
        any(_normalize_column(alias) in lookup for alias in aliases)
        for aliases in TASK_COLUMN_ALIASES.values()
    )


def _annotation_error(data_root: Path, inspections: Sequence[tuple[Path, list[str], Any]]) -> str:
    if inspections:
        found = "\n".join(f"- {path}: {columns}" for path, columns, _ in inspections)
    else:
        discovered = sorted(path for path in data_root.rglob("*") if path.is_file())
        found = "\n".join(f"- {path}" for path in discovered[:100]) or "- <no files found>"
    return (
        "Could not identify a SPAQ annotation table containing one unambiguous image filename "
        "column plus at least one of Brightness, Colorfulness, or Contrast. No labels were guessed. "
        f"Files/columns discovered under {data_root}:\n{found}"
    )


def _build_records(
    frame: Any,
    image_column: str,
    task_columns: dict[str, str],
    data_root: Path,
    configured_image_dir: Path | None,
) -> list[SPAQImageRecord]:
    image_index = _index_images(data_root, configured_image_dir)
    records: list[SPAQImageRecord] = []
    seen: set[str] = set()
    errors: list[str] = []
    for row_index, row in frame.iterrows():
        image_name = str(row[image_column]).strip()
        if not image_name or image_name.lower() == "nan":
            errors.append(f"row {row_index}: empty image filename")
            continue
        if image_name in seen:
            errors.append(f"row {row_index}: duplicate image filename {image_name!r}")
            continue
        try:
            image_path = _resolve_image(image_name, data_root, configured_image_dir, image_index)
            scores = {task: float(row[column]) for task, column in task_columns.items()}
            invalid = {
                task: score
                for task, score in scores.items()
                if not math.isfinite(score) or not 0.0 <= score <= 100.0
            }
            if invalid:
                raise ValueError(f"scores outside finite [0,100]: {invalid}")
        except Exception as exc:
            errors.append(f"row {row_index}, image {image_name!r}: {exc}")
            continue
        seen.add(image_name)
        records.append(SPAQImageRecord(image_name, image_path, scores))
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:25])
        suffix = f"\n... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise RuntimeError(f"SPAQ annotations/images failed validation:\n{preview}{suffix}")
    if len(records) < 2:
        raise RuntimeError("SPAQ needs at least two valid original images for a train/test split")
    return records


def _index_images(data_root: Path, configured_image_dir: Path | None) -> dict[str, list[Path]]:
    root = configured_image_dir or data_root
    if not root.is_dir():
        raise FileNotFoundError(f"Configured image_dir does not exist: {root}")
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.lower(), []).append(path.resolve())
    return index


def _resolve_image(
    image_name: str,
    data_root: Path,
    image_dir: Path | None,
    index: dict[str, list[Path]],
) -> Path:
    value = Path(image_name)
    attempts = []
    if image_dir is not None:
        attempts.append((image_dir / value).resolve())
    attempts.append((data_root / value).resolve())
    for path in attempts:
        if path.is_file():
            return path
    matches = index.get(value.name.lower(), [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous image basename; matches: {[str(path) for path in matches]}")
    raise FileNotFoundError(f"image not found; attempted: {[str(path) for path in attempts]}")


def _load_or_create_split(
    records: Sequence[SPAQImageRecord],
    path: Path,
    train_fraction: float,
    seed: int,
    persist: bool,
) -> tuple[list[str], list[str], str]:
    names = sorted(record.image_name for record in records)
    source_digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if path.is_file():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "seed": seed,
            "train_fraction": train_fraction,
            "source_image_digest": source_digest,
            "source_image_count": len(names),
        }
        mismatches = {
            key: {"saved": payload.get(key), "current": value}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        saved_names = payload.get("train_images", []) + payload.get("test_images", [])
        if sorted(saved_names) != names:
            mismatches["split_members"] = "saved split does not match current source images"
        if mismatches:
            raise RuntimeError(
                f"Existing data_split.json is incompatible with current SPAQ data: {mismatches}. "
                "Use a new output_dir or deliberately remove the stale split file."
            )
        train_names = list(payload["train_images"])
        test_names = list(payload["test_images"])
    else:
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * (1.0 - train_fraction)))))
        test_names = sorted(shuffled[:test_count])
        train_names = sorted(shuffled[test_count:])
        if persist:
            write_json(
                path,
                {
                    "schema_version": 1,
                    "dataset": "SPAQ",
                    "split_unit": "original_image",
                    "seed": seed,
                    "train_fraction": train_fraction,
                    "test_fraction": 1.0 - train_fraction,
                    "source_image_count": len(names),
                    "source_image_digest": source_digest,
                    "train_image_count": len(train_names),
                    "test_image_count": len(test_names),
                    "train_images": train_names,
                    "test_images": test_names,
                },
            )
    split_digest = hashlib.sha256(
        ("train\n" + "\n".join(train_names) + "\ntest\n" + "\n".join(test_names)).encode("utf-8")
    ).hexdigest()
    return train_names, test_names, split_digest


def _limit_images(records: list[SPAQImageRecord], limit: int | None) -> list[SPAQImageRecord]:
    return records if limit is None else records[: min(limit, len(records))]


def _score_statistics(records: Sequence[SPAQImageRecord]) -> dict[str, dict[str, float]]:
    return {
        task: {
            "min": min(record.scores[task] for record in records),
            "max": max(record.scores[task] for record in records),
            "mean": sum(record.scores[task] for record in records) / len(records),
        }
        for task in records[0].scores
    }
