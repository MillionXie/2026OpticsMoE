from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .io_utils import write_json
from .settings import Settings


DISTORTED_IMAGE_COLUMNS = (
    "dist_img",
    "distorted_image",
    "image",
    "image_name",
    "filename",
    "file_name",
    "img",
)
REFERENCE_IMAGE_COLUMNS = (
    "ref_img",
    "reference_image",
    "ref_image",
    "reference",
    "ref",
    "ref_id",
    "reference_id",
)
SCORE_COLUMNS = ("dmos", "dmos_mean", "mos", "mos_mean", "quality_score", "score")
VARIANCE_COLUMNS = ("var", "variance", "dmos_var", "score_variance")
DISTORTION_TYPE_COLUMNS = ("distortion_type", "dist_type", "distortion", "type")
DISTORTION_LEVEL_COLUMNS = ("distortion_level", "dist_level", "level", "severity")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DISTORTED_NAME = re.compile(r"^[Ii](?P<reference>\d+)[_-](?P<type>\d+)[_-](?P<level>\d+)$")


@dataclass(frozen=True)
class KADIDImageRecord:
    image_name: str
    image_path: Path
    reference_image: str
    reference_id: str
    dmos: float
    variance: float | None
    distortion_type: int | str | None
    distortion_level: int | None


@dataclass(frozen=True)
class DatasetBundle:
    train: "KADIDMOSDataset"
    test: "KADIDMOSDataset"
    train_records: list[KADIDImageRecord]
    test_records: list[KADIDImageRecord]
    metadata: dict[str, Any]
    cache_identity: dict[str, Any]


class KADIDMOSDataset(Dataset[tuple[Image.Image, float]]):
    """KADID RGB distorted images with official DMOS normalized from [1,5] to [0,1]."""

    def __init__(self, records: Sequence[KADIDImageRecord], score_min: float, score_max: float) -> None:
        self.records = list(records)
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        scale = self.score_max - self.score_min
        self.targets = [(float(record.dmos) - self.score_min) / scale for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, float]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB").copy()
        return rgb, self.targets[index]

    def sample_metadata(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image_name": record.image_name,
            "image_path": str(record.image_path),
            "reference_image": record.reference_image,
            "reference_id": record.reference_id,
            "quality_score": record.dmos,
            "dmos": record.dmos,
            "score_variance": record.variance,
            "distortion_type": record.distortion_type,
            "distortion_level": record.distortion_level,
            "normalized_score": self.targets[index],
        }


def load_kadid10k(settings: Settings, persist_split: bool = True) -> DatasetBundle:
    annotation_path, frame, columns = discover_metadata(
        settings.data_root,
        settings.annotations_file,
    )
    image_column = _required_column(columns, DISTORTED_IMAGE_COLUMNS, "distorted image")
    reference_column = _required_column(columns, REFERENCE_IMAGE_COLUMNS, "reference image")
    score_column = _required_column(columns, SCORE_COLUMNS, "DMOS/MOS score")
    variance_column = _optional_column(columns, VARIANCE_COLUMNS, "score variance")
    distortion_type_column = _optional_column(
        columns,
        DISTORTION_TYPE_COLUMNS,
        "distortion type",
    )
    distortion_level_column = _optional_column(
        columns,
        DISTORTION_LEVEL_COLUMNS,
        "distortion level",
    )
    if "dmos" not in _normalize_column(score_column):
        raise RuntimeError(
            f"KADID-10k is expected to use its official DMOS column; selected {score_column!r}. "
            f"Available columns: {columns}"
        )
    records = _build_records(
        frame,
        image_column,
        reference_column,
        score_column,
        variance_column,
        distortion_type_column,
        distortion_level_column,
        settings,
    )
    reference_ids = sorted({record.reference_id for record in records})
    if settings.require_official_counts:
        expected_samples = settings.official_distorted_images
        expected_references = settings.official_reference_images
        if len(records) != expected_samples or len(reference_ids) != expected_references:
            raise RuntimeError(
                "KADID-10k official-count validation failed: "
                f"samples={len(records)} (expected {expected_samples}), "
                f"references={len(reference_ids)} (expected {expected_references}). "
                "Check data_root/annotations_file/image_dir, or set "
                "dataset.require_official_counts=false only for a deliberate synthetic test."
            )
    split_path = settings.output_dir / "data_split.json"
    train_references, test_references, split_digest = _load_or_create_reference_split(
        records,
        split_path,
        settings.train_fraction,
        settings.seed,
        persist_split,
    )
    train_set = set(train_references)
    test_set = set(test_references)
    full_train = [record for record in records if record.reference_id in train_set]
    full_test = [record for record in records if record.reference_id in test_set]
    if {record.reference_id for record in full_train} & {record.reference_id for record in full_test}:
        raise RuntimeError("Reference-disjoint split construction failed")
    train_records = _diverse_limit(full_train, settings.train_image_limit)
    test_records = _diverse_limit(full_test, settings.test_image_limit)
    metadata = {
        "dataset": "KADID-10k",
        "task": "DMOS regression",
        "input_color_mode": "RGB",
        "annotation_file": str(annotation_path),
        "image_root": str(settings.image_dir or settings.data_root),
        "image_column": image_column,
        "reference_column": reference_column,
        "score_column": score_column,
        "variance_column": variance_column,
        "distortion_type_column": distortion_type_column,
        "distortion_level_column": distortion_level_column,
        "source_images": len(records),
        "reference_count_total": len(reference_ids),
        "reference_count_train": len(train_references),
        "reference_count_test": len(test_references),
        "full_train_images": len(full_train),
        "full_test_images": len(full_test),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_fraction": settings.train_fraction,
        "test_fraction": 1.0 - settings.train_fraction,
        "validation_images": 0,
        "split_unit": "reference_image",
        "reference_disjoint_train_test": True,
        "split_seed": settings.seed,
        "split_digest": split_digest,
        "train_references": train_references,
        "test_references": test_references,
        "quality_score_higher_is_better": True,
        "label_scale": [settings.quality_score_min, settings.quality_score_max],
        "training_label_scale": [0.0, 1.0],
        "normalization": (
            f"(dmos - {settings.quality_score_min}) / "
            f"{settings.quality_score_max - settings.quality_score_min}"
        ),
        "score_statistics": _score_statistics(records),
        "distortion_type_counts": _value_counts(records, "distortion_type"),
        "distortion_level_counts": _value_counts(records, "distortion_level"),
        "language_model_used": True,
        "prompt_used": True,
        "classification_prompt": settings.classification_prompt,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "train_image_limit": settings.train_image_limit,
        "test_image_limit": settings.test_image_limit,
        "train_samples_per_epoch": settings.train_samples_per_epoch,
        "train_epoch_partitions": settings.train_epoch_partitions,
    }
    cache_identity = {
        "dataset": settings.dataset,
        "task": settings.task_name,
        "annotation_file": str(annotation_path),
        "image_column": image_column,
        "reference_column": reference_column,
        "score_column": score_column,
        "score_range": [settings.quality_score_min, settings.quality_score_max],
        "quality_score_higher_is_better": True,
        "split_unit": "reference_image",
        "split_digest": split_digest,
    }
    return DatasetBundle(
        train=KADIDMOSDataset(
            train_records,
            settings.quality_score_min,
            settings.quality_score_max,
        ),
        test=KADIDMOSDataset(
            test_records,
            settings.quality_score_min,
            settings.quality_score_max,
        ),
        train_records=train_records,
        test_records=test_records,
        metadata=metadata,
        cache_identity=cache_identity,
    )


def discover_metadata(
    data_root: Path,
    configured_path: Path | None,
) -> tuple[Path, Any, list[str]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"KADID-10k data_root does not exist: {data_root}")
    if configured_path is not None:
        candidates = [configured_path]
        if not configured_path.is_file():
            matches = sorted(data_root.rglob(configured_path.name))
            if len(matches) == 1:
                candidates = matches
            else:
                raise FileNotFoundError(
                    f"Configured KADID annotations_file does not exist: {configured_path}. "
                    f"Recursive matches: {[str(path) for path in matches]}"
                )
    else:
        candidates = sorted(data_root.rglob("*.csv"))
    inspections: list[tuple[Path, list[str]]] = []
    valid: list[tuple[Path, Any, list[str]]] = []
    for path in candidates:
        try:
            frame = _read_table(path)
            columns = [str(column) for column in frame.columns]
        except Exception as exc:
            inspections.append((path, [f"<read error: {exc}>"]))
            continue
        inspections.append((path, columns))
        try:
            _required_column(columns, DISTORTED_IMAGE_COLUMNS, "distorted image")
            _required_column(columns, REFERENCE_IMAGE_COLUMNS, "reference image")
            score = _required_column(columns, SCORE_COLUMNS, "DMOS/MOS score")
        except RuntimeError:
            continue
        if "dmos" in _normalize_column(score):
            valid.append((path, frame, columns))
    if len(valid) == 1:
        return valid[0]
    details = "\n".join(f"- {path}: {columns}" for path, columns in inspections)
    if not details:
        details = "- <no CSV files found>"
    if len(valid) > 1:
        raise RuntimeError(
            "Multiple KADID metadata CSV files are valid. Set annotations_file explicitly. "
            f"Candidates:\n{details}"
        )
    raise RuntimeError(
        "Could not identify KADID dmos.csv. Required columns are one distorted-image "
        "column, one reference-image column, and one DMOS column. No labels were guessed. "
        f"Files/columns discovered under {data_root}:\n{details}"
    )


def targets_of(dataset: Dataset[Any]) -> list[float]:
    if hasattr(dataset, "targets"):
        return [float(value) for value in dataset.targets]
    if isinstance(dataset, Subset):
        parent = targets_of(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    raise TypeError("Dataset does not expose normalized KADID DMOS targets")


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
    return (
        list(images),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long),
    )


def make_indexed_loader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=indexed_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _read_table(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read KADID dmos.csv") from exc
    return pd.read_csv(path)


def _normalize_column(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def _column_lookup(columns: Sequence[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column(column)
        if normalized in lookup:
            raise RuntimeError(
                f"Metadata columns {lookup[normalized]!r} and {column!r} normalize to the same name"
            )
        lookup[normalized] = str(column)
    return lookup


def _required_column(columns: Sequence[str], aliases: Sequence[str], purpose: str) -> str:
    lookup = _column_lookup(columns)
    matches = [
        lookup[_normalize_column(alias)]
        for alias in aliases
        if _normalize_column(alias) in lookup
    ]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {purpose} column from aliases {list(aliases)}; "
            f"found {matches}. Metadata columns: {list(columns)}"
        )
    return matches[0]


def _optional_column(columns: Sequence[str], aliases: Sequence[str], purpose: str) -> str | None:
    lookup = _column_lookup(columns)
    matches = [
        lookup[_normalize_column(alias)]
        for alias in aliases
        if _normalize_column(alias) in lookup
    ]
    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous {purpose} columns {matches}. Metadata columns: {list(columns)}"
        )
    return matches[0] if matches else None


def _build_records(
    frame: Any,
    image_column: str,
    reference_column: str,
    score_column: str,
    variance_column: str | None,
    distortion_type_column: str | None,
    distortion_level_column: str | None,
    settings: Settings,
) -> list[KADIDImageRecord]:
    image_index = _index_images(settings.data_root, settings.image_dir)
    records: list[KADIDImageRecord] = []
    seen: set[str] = set()
    errors: list[str] = []
    for row_index, row in frame.iterrows():
        image_name = str(row[image_column]).strip()
        reference_image = str(row[reference_column]).strip()
        try:
            if not image_name or image_name.lower() == "nan":
                raise ValueError("empty distorted image filename")
            if not reference_image or reference_image.lower() == "nan":
                raise ValueError("empty reference image identifier")
            if image_name.lower() in seen:
                raise ValueError(f"duplicate distorted image filename {image_name!r}")
            image_path = _resolve_image(
                image_name,
                settings.data_root,
                settings.image_dir,
                image_index,
            )
            dmos = float(row[score_column])
            if not math.isfinite(dmos) or not (
                settings.quality_score_min <= dmos <= settings.quality_score_max
            ):
                raise ValueError(
                    f"DMOS {dmos!r} is outside finite "
                    f"[{settings.quality_score_min},{settings.quality_score_max}]"
                )
            variance = _optional_float(row[variance_column]) if variance_column else None
            parsed_reference, parsed_type, parsed_level = _parse_distorted_name(image_name)
            reference_id = _reference_id(reference_image, parsed_reference)
            distortion_type = (
                _clean_scalar(row[distortion_type_column])
                if distortion_type_column
                else parsed_type
            )
            distortion_level = (
                _optional_int(row[distortion_level_column])
                if distortion_level_column
                else parsed_level
            )
            if distortion_level is not None and not 1 <= distortion_level <= 5:
                raise ValueError(f"distortion level must be in [1,5], got {distortion_level}")
        except Exception as exc:
            errors.append(f"row {row_index}, image {image_name!r}: {exc}")
            continue
        seen.add(image_name.lower())
        records.append(
            KADIDImageRecord(
                image_name=image_name,
                image_path=image_path,
                reference_image=reference_image,
                reference_id=reference_id,
                dmos=dmos,
                variance=variance,
                distortion_type=distortion_type,
                distortion_level=distortion_level,
            )
        )
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:25])
        suffix = f"\n... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise RuntimeError(f"KADID metadata/images failed validation:\n{preview}{suffix}")
    if len(records) < 2:
        raise RuntimeError("KADID needs at least two valid distorted images")
    return sorted(records, key=lambda record: (record.reference_id, record.image_name.lower()))


def _index_images(data_root: Path, configured_image_dir: Path | None) -> dict[str, list[Path]]:
    root = configured_image_dir or data_root
    if not root.is_dir():
        raise FileNotFoundError(f"Configured KADID image_dir does not exist: {root}")
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.lower(), []).append(path.resolve())
    if not index:
        raise FileNotFoundError(f"No KADID images were found under {root}")
    return index


def _resolve_image(
    image_name: str,
    data_root: Path,
    image_dir: Path | None,
    index: dict[str, list[Path]],
) -> Path:
    value = Path(image_name)
    attempts: list[Path] = []
    if image_dir is not None:
        attempts.append((image_dir / value).resolve())
    attempts.extend(
        [
            (data_root / value).resolve(),
            (data_root / "images" / value).resolve(),
            (data_root / "image" / value).resolve(),
        ]
    )
    for path in attempts:
        if path.is_file():
            return path
    matches = index.get(value.name.lower(), [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous image basename; matches: {[str(path) for path in matches]}")
    raise FileNotFoundError(f"image not found; attempted: {[str(path) for path in attempts]}")


def _parse_distorted_name(image_name: str) -> tuple[str | None, int | None, int | None]:
    match = DISTORTED_NAME.match(Path(image_name).stem)
    if not match:
        return None, None, None
    return (
        str(int(match.group("reference"))),
        int(match.group("type")),
        int(match.group("level")),
    )


def _reference_id(reference_image: str, parsed_reference: str | None) -> str:
    stem = Path(reference_image).stem.strip().lower()
    numeric = re.search(r"(\d+)", stem)
    if numeric:
        return str(int(numeric.group(1)))
    if parsed_reference is not None:
        return parsed_reference
    return stem


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _clean_scalar(value: Any) -> int | str | None:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except ValueError:
        pass
    return text


def _load_or_create_reference_split(
    records: Sequence[KADIDImageRecord],
    path: Path,
    train_fraction: float,
    seed: int,
    persist: bool,
) -> tuple[list[str], list[str], str]:
    references = sorted({record.reference_id for record in records})
    source_rows = [
        f"{record.image_name}\t{record.reference_id}\t{record.dmos:.12g}"
        for record in records
    ]
    source_digest = hashlib.sha256("\n".join(source_rows).encode("utf-8")).hexdigest()
    expected = {
        "schema_version": 1,
        "dataset": "KADID-10k",
        "split_unit": "reference_image",
        "seed": seed,
        "train_fraction": train_fraction,
        "source_digest": source_digest,
        "source_image_count": len(records),
        "source_reference_count": len(references),
    }
    if path.is_file():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"saved": payload.get(key), "current": value}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        saved_references = payload.get("train_references", []) + payload.get(
            "test_references",
            [],
        )
        if sorted(saved_references) != references:
            mismatches["split_members"] = "saved references do not match current metadata"
        if mismatches:
            raise RuntimeError(
                f"Existing data_split.json is incompatible with current KADID data: {mismatches}. "
                "Use a new output_dir or deliberately remove the stale split file."
            )
        train_references = list(payload["train_references"])
        test_references = list(payload["test_references"])
    else:
        shuffled = list(references)
        random.Random(seed).shuffle(shuffled)
        test_count = max(
            1,
            min(
                len(shuffled) - 1,
                int(round(len(shuffled) * (1.0 - train_fraction))),
            ),
        )
        test_references = sorted(shuffled[:test_count])
        train_references = sorted(shuffled[test_count:])
        if persist:
            train_set = set(train_references)
            test_set = set(test_references)
            write_json(
                path,
                {
                    **expected,
                    "test_fraction": 1.0 - train_fraction,
                    "train_reference_count": len(train_references),
                    "test_reference_count": len(test_references),
                    "train_image_count": sum(
                        record.reference_id in train_set for record in records
                    ),
                    "test_image_count": sum(
                        record.reference_id in test_set for record in records
                    ),
                    "train_references": train_references,
                    "test_references": test_references,
                },
            )
    split_digest = hashlib.sha256(
        (
            "train_references\n"
            + "\n".join(train_references)
            + "\ntest_references\n"
            + "\n".join(test_references)
        ).encode("utf-8")
    ).hexdigest()
    return train_references, test_references, split_digest


def _diverse_limit(
    records: list[KADIDImageRecord],
    limit: int | None,
) -> list[KADIDImageRecord]:
    if limit is None or limit >= len(records):
        return list(records)
    groups: dict[str, list[KADIDImageRecord]] = {}
    for record in records:
        groups.setdefault(record.reference_id, []).append(record)
    selected: list[KADIDImageRecord] = []
    offsets = {reference: 0 for reference in groups}
    references = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for reference in references:
            offset = offsets[reference]
            if offset < len(groups[reference]):
                selected.append(groups[reference][offset])
                offsets[reference] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def _score_statistics(records: Sequence[KADIDImageRecord]) -> dict[str, float]:
    scores = [record.dmos for record in records]
    mean = sum(scores) / len(scores)
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores)),
    }


def _value_counts(records: Sequence[KADIDImageRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = getattr(record, field)
        key = "unknown" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
