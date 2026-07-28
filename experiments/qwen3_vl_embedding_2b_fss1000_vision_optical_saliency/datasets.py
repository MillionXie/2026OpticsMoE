from __future__ import annotations

import os
import random
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .io_utils import write_csv, write_json


@dataclass(frozen=True)
class FSSRecord:
    sample_index: int
    split: str
    class_name: str
    image_path: Path
    mask_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.class_name}/{self.image_path.stem}"


@dataclass(frozen=True)
class DatasetBundle:
    train_records: tuple[FSSRecord, ...]
    test_records: tuple[FSSRecord, ...]
    train_classes: tuple[str, ...]
    test_classes: tuple[str, ...]
    data_directory: Path
    metadata: dict[str, Any]


def prepare_fss1000(settings: Any, *, persist: bool = True) -> DatasetBundle:
    data_dir = _locate_data_directory(settings.data_root)
    if data_dir is None and settings.download:
        _download_and_extract(settings)
        data_dir = _locate_data_directory(settings.data_root)
    if data_dir is None:
        discovered = [
            str(path.relative_to(settings.data_root))
            for path in settings.data_root.rglob("*")
            if path.is_dir()
        ] if settings.data_root.exists() else []
        raise FileNotFoundError(
            f"FSS-1000 class directory was not found under {settings.data_root}. "
            "Expected fewshot_data/<class>/{1.jpg,1.png,...}. "
            f"Discovered directories (first 30): {discovered[:30]}"
        )

    test_names = _load_official_test_classes(settings)
    all_classes = sorted(
        path.name for path in data_dir.iterdir()
        if path.is_dir() and _class_has_pairs(path)
    )
    missing_test = sorted(set(test_names) - set(all_classes))
    if missing_test:
        raise RuntimeError(
            f"Official test list contains {len(missing_test)} classes absent from data: "
            f"{missing_test[:20]}"
        )
    test_classes = sorted(set(test_names))
    train_classes = sorted(set(all_classes) - set(test_classes))
    if set(train_classes) & set(test_classes):
        raise RuntimeError("FSS-1000 train/test class leakage detected")
    if settings.test_class_limit is not None:
        test_classes = test_classes[: int(settings.test_class_limit)]
    if settings.train_class_limit is not None:
        train_classes = train_classes[: int(settings.train_class_limit)]

    records: list[FSSRecord] = []
    ignored_geometry_mismatches: list[dict[str, Any]] = []
    sample_index = 0
    for split, classes in (("train", train_classes), ("test", test_classes)):
        for class_name in classes:
            pairs = _paired_files(data_dir / class_name)
            if settings.images_per_class_limit is not None:
                pairs = pairs[: int(settings.images_per_class_limit)]
            if not pairs:
                raise RuntimeError(f"No image/mask pairs found for class {class_name!r}")
            for image_path, mask_path in pairs:
                image_size, mask_size = _pair_geometry(image_path, mask_path)
                if image_size != mask_size:
                    ignored_geometry_mismatches.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "sample_id": f"{class_name}/{image_path.stem}",
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "image_width": image_size[0],
                            "image_height": image_size[1],
                            "mask_width": mask_size[0],
                            "mask_height": mask_size[1],
                            "reason": "source_image_mask_geometry_mismatch",
                        }
                    )
                    continue
                records.append(
                    FSSRecord(sample_index, split, class_name, image_path, mask_path)
                )
                sample_index += 1
    train_records = tuple(record for record in records if record.split == "train")
    test_records = tuple(record for record in records if record.split == "test")
    metadata = {
        "dataset": "FSS-1000",
        "task": "class_agnostic_binary_saliency",
        "official_test_class_list": settings.official_test_list_url,
        "official_split_policy": (
            "official 240 test classes; remaining classes are training because this "
            "experiment intentionally has no validation split"
        ),
        "merge_official_validation_into_train": settings.merge_official_validation_into_train,
        "data_directory": str(data_dir),
        "all_discovered_classes": len(all_classes),
        "train_classes": len(train_classes),
        "test_classes": len(test_classes),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "class_disjoint": not bool(set(train_classes) & set(test_classes)),
        "mask_resize_interpolation": "nearest",
        "ignored_geometry_mismatch": len(ignored_geometry_mismatches),
        "ignored_geometry_mismatch_manifest": (
            "manifests/ignored_samples.csv" if ignored_geometry_mismatches else None
        ),
    }
    if not train_records or not test_records:
        raise RuntimeError(f"Empty FSS split after limits: {metadata}")
    if persist:
        write_json(settings.output_dir / "dataset.json", metadata)
        write_json(
            settings.output_dir / "manifests" / "class_split.json",
            {"train": train_classes, "test": test_classes},
        )
        write_csv(
            settings.output_dir / "manifests" / "samples.csv",
            [
                {
                    "sample_index": record.sample_index,
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "class_name": record.class_name,
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                }
                for record in records
            ],
            ["sample_index", "sample_id", "split", "class_name", "image_path", "mask_path"],
        )
        write_csv(
            settings.output_dir / "manifests" / "ignored_samples.csv",
            ignored_geometry_mismatches,
            [
                "split", "class_name", "sample_id", "image_path", "mask_path",
                "image_width", "image_height", "mask_width", "mask_height", "reason",
            ],
        )
    if ignored_geometry_mismatches:
        print(
            "WARNING: ignored "
            f"{len(ignored_geometry_mismatches)} FSS-1000 image/mask pair(s) whose "
            "source geometries do not match. No mask was force-resized against a "
            "different-aspect-ratio image. See manifests/ignored_samples.csv.",
            flush=True,
        )
    return DatasetBundle(
        train_records, test_records, tuple(train_classes), tuple(test_classes),
        data_dir, metadata,
    )


class FSSSaliencyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: tuple[FSSRecord, ...],
        settings: Any,
        *,
        training: bool,
    ) -> None:
        self.records = records
        self.settings = settings
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        with Image.open(record.mask_path) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise RuntimeError(
                f"Image/mask geometry mismatch for {record.sample_id}: "
                f"image={image.size}, mask={mask.size}"
            )
        image, mask = _paired_transform(
            image, mask, self.settings, training=self.training
        )
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        unique = set(torch.unique(mask_tensor).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise RuntimeError(f"Mask is not binary after nearest resize: {unique}")
        return {
            "image": image,
            "mask": mask_tensor,
            "sample_index": record.sample_index,
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
        }


def collate_saliency(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "masks": torch.stack([item["mask"] for item in batch]),
        "sample_indices": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "class_names": [item["class_name"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
    }


def _paired_transform(
    image: Image.Image,
    mask: Image.Image,
    settings: Any,
    *,
    training: bool,
) -> tuple[Image.Image, Image.Image]:
    size = int(settings.image_size)
    if training and settings.augmentation_enabled:
        rng = random
        scale = rng.uniform(float(settings.crop_scale_min), 1.0)
        crop_w = max(1, round(image.width * scale))
        crop_h = max(1, round(image.height * scale))
        left = rng.randint(0, max(0, image.width - crop_w))
        top = rng.randint(0, max(0, image.height - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        image = image.crop(box)
        mask = mask.crop(box)
        if rng.random() < float(settings.horizontal_flip_probability):
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        angle = rng.uniform(-float(settings.rotation_degrees), float(settings.rotation_degrees))
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        if settings.brightness_jitter > 0:
            factor = rng.uniform(1.0 - settings.brightness_jitter, 1.0 + settings.brightness_jitter)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if settings.contrast_jitter > 0:
            factor = rng.uniform(1.0 - settings.contrast_jitter, 1.0 + settings.contrast_jitter)
            image = ImageEnhance.Contrast(image).enhance(factor)
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return image, mask


def _locate_data_directory(root: Path) -> Path | None:
    candidates = (
        root / "fewshot_data",
        root / "FSS-1000" / "fewshot_data",
        root / "FSS-1000-master" / "fewshot_data",
        root,
    )
    for candidate in candidates:
        if candidate.is_dir():
            class_count = sum(
                1 for path in candidate.iterdir()
                if path.is_dir() and _class_has_pairs(path)
            )
            if class_count >= 100:
                return candidate
    if root.exists():
        for candidate in root.rglob("fewshot_data"):
            if candidate.is_dir():
                return candidate
    return None


def _class_has_pairs(path: Path) -> bool:
    return bool(_paired_files(path))


def _paired_files(class_dir: Path) -> list[tuple[Path, Path]]:
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in class_dir.iterdir() if class_dir.is_dir() else ():
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".bmp"}:
            images[path.stem] = path
        elif suffix == ".png":
            masks[path.stem.removesuffix("_mask")] = path
    return [(images[key], masks[key]) for key in sorted(set(images) & set(masks))]


def _pair_geometry(
    image_path: Path,
    mask_path: Path,
) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        with Image.open(image_path) as source:
            image_size = tuple(int(value) for value in source.size)
        with Image.open(mask_path) as source:
            mask_size = tuple(int(value) for value in source.size)
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect FSS-1000 pair geometry: image={image_path}, "
            f"mask={mask_path}: {type(exc).__name__}: {exc}"
        ) from exc
    return image_size, mask_size


def _load_official_test_classes(settings: Any) -> list[str]:
    local = settings.data_root / "fss_test_set.txt"
    bundled = Path(__file__).resolve().parent / "splits" / "fss_test_set.txt"
    if not local.is_file():
        if bundled.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
        elif not settings.download:
            raise FileNotFoundError(
                f"Official FSS test list missing: {local}. Enable dataset.download."
            )
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(settings.official_test_list_url, local)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download official FSS test list from "
                    f"{settings.official_test_list_url}: {exc}"
                ) from exc
    names = [line.strip() for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 240 or len(set(names)) != len(names):
        raise RuntimeError(
            f"Official FSS test list must contain 240 unique classes, got "
            f"{len(names)} lines/{len(set(names))} unique"
        )
    return names


def _download_and_extract(settings: Any) -> None:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    if settings.download_source in {"auto", "huggingface"}:
        try:
            _materialize_huggingface_dataset(settings)
            return
        except Exception as exc:
            errors.append(f"Hugging Face {settings.huggingface_dataset_id}: {type(exc).__name__}: {exc}")
            if settings.download_source == "huggingface":
                raise RuntimeError(errors[-1]) from exc
    if settings.download_source not in {"auto", "google_drive"}:
        raise RuntimeError(f"Unsupported download source {settings.download_source!r}")
    archive = settings.data_root / "fewshot_data.zip"
    if not archive.is_file():
        try:
            # A short preflight prevents gdown from hanging indefinitely on
            # servers where Google Drive is firewalled.
            urllib.request.urlopen(
                f"https://drive.google.com/uc?export=download&id={settings.download_file_id}",
                timeout=20,
            ).close()
            subprocess.run(
                [
                    sys.executable, "-m", "gdown", "--id", settings.download_file_id,
                    "--output", str(archive),
                ],
                check=True,
                timeout=3600,
            )
        except Exception as exc:
            errors.append(f"Google Drive: {type(exc).__name__}: {exc}")
            raise RuntimeError(
                "Automatic FSS-1000 download failed from every configured source. "
                f"Attempts: {errors}. You can manually place fewshot_data.zip under "
                f"{settings.data_root}."
            ) from exc
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Downloaded FSS archive is not a valid zip: {archive}")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(settings.data_root)


def _materialize_huggingface_dataset(settings: Any) -> None:
    """Materialize a verified public mirror into the official paired-file layout."""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = settings.huggingface_endpoint
    from datasets import load_dataset

    target = settings.data_root / "fewshot_data"
    if _locate_data_directory(settings.data_root) is not None:
        return
    cache = settings.data_root / "huggingface_cache"
    dataset = load_dataset(
        settings.huggingface_dataset_id,
        split="train",
        cache_dir=str(cache),
    )
    required = {"image", "mask", "class_name"}
    missing = required - set(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"Mirror is missing fields {sorted(missing)}; available={dataset.column_names}"
        )
    if len(dataset) != 10_000:
        raise RuntimeError(
            f"FSS-1000 mirror must contain 10,000 samples, got {len(dataset)}"
        )
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for index, row in enumerate(dataset, start=1):
        class_name = str(row["class_name"])
        number = counts.get(class_name, 0) + 1
        counts[class_name] = number
        folder = target / class_name
        folder.mkdir(parents=True, exist_ok=True)
        image = row["image"]
        mask = row["mask"]
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise RuntimeError(
                f"Mirror sample {index} did not decode image/mask as PIL images"
            )
        image.convert("RGB").save(folder / f"{number}.jpg", quality=95)
        mask.convert("L").save(folder / f"{number}.png")
        if index % 500 == 0 or index == len(dataset):
            print(f"[fss1000_download] materialized={index:,}/{len(dataset):,}", flush=True)
    if len(counts) != 1000 or set(counts.values()) != {10}:
        raise RuntimeError(
            f"Expected 1,000 classes x 10 images; got classes={len(counts)}, "
            f"per-class-counts={sorted(set(counts.values()))}"
        )
