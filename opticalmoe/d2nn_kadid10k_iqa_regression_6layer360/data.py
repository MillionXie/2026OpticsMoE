import csv
import importlib.util
import random
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import REPO_ROOT, resolve_path

IMAGE_COLUMNS = ("distorted_image", "dist_img", "image", "image_name", "filename", "file_name", "img")
REFERENCE_COLUMNS = ("reference_image", "ref_image", "ref_img", "reference", "ref", "ref_id", "reference_id")
SCORE_COLUMNS = ("dmos", "dmos_mean", "mos", "mos_mean", "score", "quality_score")
LEVEL_COLUMNS = ("distortion_level", "level", "dist_level", "severity")
TYPE_COLUMNS = ("distortion_type", "dist_type", "distortion")


@dataclass(frozen=True)
class QualityRecord:
    image_path: Path
    image_name: str
    reference_image: str
    score: float
    target: float = 0.0
    distortion_level: int | None = None
    distortion_type: str | None = None


@dataclass
class DataBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    metadata: dict


class KADIDRegressionDataset(Dataset):
    def __init__(self, records, resize_size=300):
        self.records = list(records); self.resize_size = int(resize_size)
        self.references = [record.reference_image for record in self.records]

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            gray = image.convert("L").resize((self.resize_size, self.resize_size), Image.Resampling.BICUBIC)
            width, height = gray.size
            tensor = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).view(height, width).unsqueeze(0).float().div_(255.0)
        return tensor, torch.tensor(record.target, dtype=torch.float32), int(index)

    def sample_metadata(self, index):
        record = self.records[int(index)]
        return {
            "image_path": str(record.image_path), "image_name": record.image_name,
            "reference_image": record.reference_image, "quality_score": record.score,
            "quality_target": record.target, "distortion_level": record.distortion_level,
            "distortion_type": record.distortion_type,
        }


def _load_prepare_module():
    path = REPO_ROOT / "experiments" / "qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual" / "data_prepare.py"
    spec = importlib.util.spec_from_file_location("kadid_data_prepare_for_pure_optical", path)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot load KADID preparation helper: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _find_column(columns, candidates, purpose, configured=None, required=True):
    lookup = {str(column).strip().lower(): column for column in columns}
    if configured:
        key = str(configured).strip().lower()
        if key not in lookup: raise RuntimeError(f"Configured {purpose} column {configured!r} not found. CSV columns: {list(columns)}")
        return lookup[key]
    for candidate in candidates:
        if candidate in lookup: return lookup[candidate]
    if required: raise RuntimeError(f"Missing {purpose} column. Accepted names: {list(candidates)}. CSV columns: {list(columns)}")
    return None


def _resolve_image(root, image_dir, filename, row_number):
    name = Path(str(filename).replace("\\", "/")); directory = Path(image_dir)
    candidates = [name] if name.is_absolute() else [directory / name if directory.is_absolute() else root / directory / name, root / name]
    for candidate in candidates:
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f"KADID image not found at CSV row {row_number}: {filename}. Tried: {[str(path) for path in candidates]}")


def _filename_distortion_metadata(image_name):
    parts = Path(image_name).stem.split("_"); distortion_type = None; level = None
    if len(parts) >= 3:
        distortion_type = parts[-2]
        try: level = int(parts[-1])
        except ValueError: level = None
    return distortion_type, level


def _read_records(root, csv_path, image_dir, cfg):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames: raise RuntimeError(f"KADID metadata CSV has no header: {csv_path}")
        columns = [str(value).strip() for value in reader.fieldnames]; rows = list(reader)
    image_column = _find_column(columns, IMAGE_COLUMNS, "distorted image", cfg.get("image_column"))
    reference_column = _find_column(columns, REFERENCE_COLUMNS, "reference image", cfg.get("reference_column"))
    score_column = _find_column(columns, SCORE_COLUMNS, "quality score", cfg.get("score_column"))
    level_column = _find_column(columns, LEVEL_COLUMNS, "distortion level", required=False)
    type_column = _find_column(columns, TYPE_COLUMNS, "distortion type", required=False)
    records = []
    for row_number, row in enumerate(rows, 2):
        image_name = str(row.get(image_column, "") or "").strip(); reference = str(row.get(reference_column, "") or "").strip()
        if not image_name or not reference: raise RuntimeError(f"Empty image/reference field in {csv_path} at row {row_number}")
        try: score = float(str(row.get(score_column, "")).strip())
        except ValueError as exc: raise RuntimeError(f"Invalid quality score at CSV row {row_number}: {row.get(score_column)!r}") from exc
        inferred_type, inferred_level = _filename_distortion_metadata(image_name)
        if type_column:
            distortion_type = str(row.get(type_column, "") or "").strip() or inferred_type
        else:
            distortion_type = inferred_type
        if level_column and str(row.get(level_column, "") or "").strip():
            try: distortion_level = int(float(str(row[level_column]).strip()))
            except ValueError as exc: raise RuntimeError(f"Invalid distortion level at CSV row {row_number}") from exc
        else: distortion_level = inferred_level
        records.append(QualityRecord(_resolve_image(root, image_dir, image_name, row_number), image_name, reference, score, distortion_level=distortion_level, distortion_type=distortion_type))
    if not records: raise RuntimeError(f"No KADID samples found in {csv_path}")
    return records, {"image_column": image_column, "reference_column": reference_column, "score_column": score_column, "level_column": level_column, "type_column": type_column, "csv_columns": columns}


def _split_references(records, test_fraction, validation_fraction, seed):
    references = sorted({record.reference_image for record in records}); random.Random(seed).shuffle(references)
    if len(references) < 3: raise RuntimeError("Reference-disjoint train/validation/test split requires at least three references")
    test_count = min(max(round(len(references) * float(test_fraction)), 1), len(references) - 2)
    test_refs = set(references[:test_count]); remaining = references[test_count:]
    validation_count = min(max(round(len(remaining) * float(validation_fraction)), 1), len(remaining) - 1)
    validation_refs = set(remaining[:validation_count]); train_refs = set(remaining[validation_count:])
    return (
        [record for record in records if record.reference_image in train_refs],
        [record for record in records if record.reference_image in validation_refs],
        [record for record in records if record.reference_image in test_refs],
    )


def _limit(records, limit, seed):
    if limit is None or int(limit) >= len(records): return list(records)
    order = list(range(len(records))); random.Random(seed).shuffle(order)
    return [records[index] for index in sorted(order[:int(limit)])]


def load_data(config, smoke_test=False):
    cfg = config.get("dataset", {}); root = resolve_path(cfg.get("data_root"))
    metadata_csv = str(cfg.get("metadata_csv", "dmos.csv")); image_dir = str(cfg.get("image_dir", "images")); preparation = None
    if bool(cfg.get("download", True)):
        preparation = _load_prepare_module().ensure_kadid10k_dataset(root, metadata_csv, image_dir, cfg.get("dataset_download_url"))
        metadata_csv = preparation["metadata_csv"]; image_dir = preparation["image_dir"]
    csv_path = Path(metadata_csv); csv_path = csv_path if csv_path.is_absolute() else root / csv_path
    if not csv_path.is_file(): raise FileNotFoundError(f"KADID metadata CSV not found: {csv_path}")
    records, columns = _read_records(root, csv_path.resolve(), image_dir, cfg)
    train, validation, test = _split_references(records, cfg.get("test_reference_fraction", .2), cfg.get("validation_reference_fraction", .1), int(config.get("seed", 42)))
    score_min = min(record.score for record in train); score_max = max(record.score for record in train)
    if score_max <= score_min: raise RuntimeError("Training quality scores have zero range")
    higher_is_better = cfg.get("quality_score_higher_is_better")
    if not isinstance(higher_is_better, bool): raise RuntimeError("quality_score_higher_is_better must be explicitly true or false for regression")
    def normalize(record):
        value = (record.score - score_min) / (score_max - score_min)
        if not higher_is_better: value = 1.0 - value
        return replace(record, target=float(max(0.0, min(1.0, value))))
    train = [normalize(record) for record in train]; validation = [normalize(record) for record in validation]; test = [normalize(record) for record in test]
    if smoke_test:
        train_limit, validation_limit, test_limit = 24, 12, 12
    else:
        train_limit, validation_limit, test_limit = cfg.get("train_limit"), cfg.get("validation_limit"), cfg.get("test_limit")
    train = _limit(train, train_limit, 43); validation = _limit(validation, validation_limit, 44); test = _limit(test, test_limit, 45)
    resize = int(cfg.get("resize_size", 300)); datasets = [KADIDRegressionDataset(part, resize) for part in (train, validation, test)]
    train_refs, validation_refs, test_refs = [set(dataset.references) for dataset in datasets]
    metadata = {
        "dataset": "kadid10k_iqa_regression", "data_root": str(root), "metadata_csv": str(csv_path.resolve()), "image_dir": image_dir,
        "samples_total": len(records), "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
        "reference_count_total": len({record.reference_image for record in records}), "reference_count_train": len(train_refs), "reference_count_validation": len(validation_refs), "reference_count_test": len(test_refs),
        "reference_disjoint_train_validation_test": not (train_refs & validation_refs or train_refs & test_refs or validation_refs & test_refs),
        "score_column": columns["score_column"], "score_higher_is_better": higher_is_better, "train_score_min": score_min, "train_score_max": score_max,
        "score_range_total": [min(record.score for record in records), max(record.score for record in records)],
        "score_range_train": [min(record.score for record in train), max(record.score for record in train)],
        "score_range_validation": [min(record.score for record in validation), max(record.score for record in validation)],
        "score_range_test": [min(record.score for record in test), max(record.score for record in test)],
        "normalized_target": "0=worst, 1=best; clipping uses train-only score range", "columns": columns, "preparation": preparation,
    }
    return DataBundle(*datasets, metadata)


def make_loader(dataset, batch_size, workers, shuffle, seed):
    workers = int(workers); kwargs = {"batch_size": int(batch_size), "num_workers": workers, "shuffle": bool(shuffle), "pin_memory": torch.cuda.is_available(), "generator": torch.Generator().manual_seed(int(seed))}
    if workers > 0: kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs)
