from __future__ import annotations

import random
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .io_utils import sha256_strings, write_csv, write_json


COCO_EXPECTED = {"train": 118_287, "val": 5_000}
DUTS_EXPECTED = {"train": 10_553, "test": 5_019}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    sample_index: int
    sample_id: str
    split: str
    image_path: Path
    mask_path: Path | None = None


@dataclass(frozen=True)
class CocoBundle:
    train_records: tuple[ImageRecord, ...]
    val_records: tuple[ImageRecord, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DutsBundle:
    train_records: tuple[ImageRecord, ...]
    test_records: tuple[ImageRecord, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetBundle:
    coco: CocoBundle
    duts: DutsBundle


def prepare_datasets(settings: Any, *, persist: bool = True) -> DatasetBundle:
    coco = prepare_coco(settings, persist=persist)
    duts = prepare_duts(settings, persist=persist)
    if persist:
        write_json(
            settings.output_dir / "dataset.json",
            {"coco": coco.metadata, "duts": duts.metadata},
        )
    return DatasetBundle(coco=coco, duts=duts)


def prepare_coco(settings: Any, *, persist: bool = True) -> CocoBundle:
    train_dir = _locate_directory(settings.coco_root, "train2017")
    val_dir = _locate_directory(settings.coco_root, "val2017")
    if (train_dir is None or val_dir is None) and settings.auto_download:
        settings.coco_root.mkdir(parents=True, exist_ok=True)
        if train_dir is None:
            _download_and_extract_zip(
                settings.coco_train_url,
                settings.coco_root / "train2017.zip",
                settings.coco_root,
                remove_archive=settings.remove_archives_after_extract,
            )
        if val_dir is None:
            _download_and_extract_zip(
                settings.coco_val_url,
                settings.coco_root / "val2017.zip",
                settings.coco_root,
                remove_archive=settings.remove_archives_after_extract,
            )
        train_dir = _locate_directory(settings.coco_root, "train2017")
        val_dir = _locate_directory(settings.coco_root, "val2017")
    if train_dir is None or val_dir is None:
        raise FileNotFoundError(
            "COCO 2017 images were not found. Expected train2017 and val2017 "
            f"under {settings.coco_root}. auto_download={settings.auto_download}."
        )
    train_paths = _image_files(train_dir)
    val_paths = _image_files(val_dir)
    _validate_count("COCO train2017", len(train_paths), COCO_EXPECTED["train"])
    _validate_count("COCO val2017", len(val_paths), COCO_EXPECTED["val"])
    if settings.coco_train_limit is not None:
        train_paths = train_paths[: settings.coco_train_limit]
    if settings.coco_val_limit is not None:
        val_paths = val_paths[: settings.coco_val_limit]
    train_records = _records("train", train_paths)
    val_records = _records("val", val_paths)
    digest = sha256_strings(
        record.sample_id for record in (*train_records, *val_records)
    )
    metadata = {
        "dataset": "COCO 2017",
        "task": "unlabeled_general_vision_feature_distillation",
        "train_images": len(train_records),
        "val_images": len(val_records),
        "full_train_images": COCO_EXPECTED["train"],
        "full_val_images": COCO_EXPECTED["val"],
        "train_directory": str(train_dir),
        "val_directory": str(val_dir),
        "manifest_sha256": digest,
        "resize_mode": settings.coco_resize_mode,
        "image_size": settings.image_size,
        "validation_usage": "observation_only_not_checkpoint_selection",
    }
    if not train_records or not val_records:
        raise RuntimeError(f"COCO split became empty after configured limits: {metadata}")
    if persist:
        _write_manifest(
            settings.output_dir / "manifests" / "coco_images.csv",
            (*train_records, *val_records),
        )
        write_json(settings.output_dir / "coco_dataset.json", metadata)
    return CocoBundle(train_records, val_records, metadata)


def prepare_duts(settings: Any, *, persist: bool = True) -> DutsBundle:
    train_root = _locate_directory(settings.duts_root, "DUTS-TR")
    test_root = _locate_directory(settings.duts_root, "DUTS-TE")
    if (train_root is None or test_root is None) and settings.auto_download:
        settings.duts_root.mkdir(parents=True, exist_ok=True)
        if train_root is None:
            _download_and_extract_zip(
                settings.duts_train_url,
                settings.duts_root / "DUTS-TR.zip",
                settings.duts_root,
                remove_archive=settings.remove_archives_after_extract,
            )
        if test_root is None:
            _download_and_extract_zip(
                settings.duts_test_url,
                settings.duts_root / "DUTS-TE.zip",
                settings.duts_root,
                remove_archive=settings.remove_archives_after_extract,
            )
        train_root = _locate_directory(settings.duts_root, "DUTS-TR")
        test_root = _locate_directory(settings.duts_root, "DUTS-TE")
    if train_root is None or test_root is None:
        raise FileNotFoundError(
            "DUTS was not found. Expected DUTS-TR and DUTS-TE under "
            f"{settings.duts_root}. auto_download={settings.auto_download}."
        )
    train_pairs = _duts_pairs(train_root, "DUTS-TR")
    test_pairs = _duts_pairs(test_root, "DUTS-TE")
    _validate_count("DUTS-TR", len(train_pairs), DUTS_EXPECTED["train"])
    _validate_count("DUTS-TE", len(test_pairs), DUTS_EXPECTED["test"])
    if settings.duts_train_limit is not None:
        train_pairs = train_pairs[: settings.duts_train_limit]
    if settings.duts_test_limit is not None:
        test_pairs = test_pairs[: settings.duts_test_limit]
    train_records = tuple(
        ImageRecord(index, f"train/{image.stem}", "train", image, mask)
        for index, (image, mask) in enumerate(train_pairs)
    )
    test_records = tuple(
        ImageRecord(index, f"test/{image.stem}", "test", image, mask)
        for index, (image, mask) in enumerate(test_pairs)
    )
    digest = sha256_strings(
        f"{record.sample_id}:{record.mask_path.name if record.mask_path else ''}"
        for record in (*train_records, *test_records)
    )
    metadata = {
        "dataset": "DUTS",
        "task": "class_agnostic_binary_saliency",
        "train_images": len(train_records),
        "test_images": len(test_records),
        "full_train_images": DUTS_EXPECTED["train"],
        "full_test_images": DUTS_EXPECTED["test"],
        "train_directory": str(train_root),
        "test_directory": str(test_root),
        "manifest_sha256": digest,
        "mask_resize_interpolation": "nearest",
        "test_usage": "observation_only_not_checkpoint_selection",
    }
    if not train_records or not test_records:
        raise RuntimeError(f"DUTS split became empty after configured limits: {metadata}")
    if persist:
        _write_manifest(
            settings.output_dir / "manifests" / "duts_pairs.csv",
            (*train_records, *test_records),
        )
        write_json(settings.output_dir / "duts_dataset.json", metadata)
    return DutsBundle(train_records, test_records, metadata)


class CocoImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: tuple[ImageRecord, ...], settings: Any) -> None:
        self.records = records
        self.settings = settings

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        size = int(self.settings.image_size)
        if self.settings.coco_resize_mode == "center_crop":
            image = ImageOps.fit(
                image,
                (size, size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        return {
            "image": image,
            "sample_id": record.sample_id,
            "sample_index": record.sample_index,
            "split": record.split,
            "image_path": str(record.image_path),
        }


class DUTSSaliencyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: tuple[ImageRecord, ...],
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
        if record.mask_path is None:
            raise RuntimeError(f"DUTS record has no mask: {record.sample_id}")
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        with Image.open(record.mask_path) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise RuntimeError(
                f"DUTS source geometry mismatch for {record.sample_id}: "
                f"image={image.size}, mask={mask.size}"
            )
        image, mask = paired_saliency_transform(
            image,
            mask,
            self.settings,
            training=self.training,
        )
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        unique = set(torch.unique(mask_tensor).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise RuntimeError(f"Mask is not binary after nearest resize: {unique}")
        return {
            "image": image,
            "mask": mask_tensor,
            "sample_id": record.sample_id,
            "sample_index": record.sample_index,
            "split": record.split,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
        }


def collate_coco(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "sample_indices": torch.tensor(
            [item["sample_index"] for item in batch],
            dtype=torch.long,
        ),
        "splits": [item["split"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def collate_duts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "masks": torch.stack([item["mask"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "sample_indices": torch.tensor(
            [item["sample_index"] for item in batch],
            dtype=torch.long,
        ),
        "splits": [item["split"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
    }


def paired_saliency_transform(
    image: Image.Image,
    mask: Image.Image,
    settings: Any,
    *,
    training: bool,
) -> tuple[Image.Image, Image.Image]:
    size = int(settings.image_size)
    if training and settings.augmentation_enabled:
        scale = random.uniform(float(settings.crop_scale_min), 1.0)
        crop_width = max(1, round(image.width * scale))
        crop_height = max(1, round(image.height * scale))
        left = random.randint(0, max(0, image.width - crop_width))
        top = random.randint(0, max(0, image.height - crop_height))
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box)
        mask = mask.crop(box)
        if random.random() < float(settings.horizontal_flip_probability):
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        angle = random.uniform(
            -float(settings.rotation_degrees),
            float(settings.rotation_degrees),
        )
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )
        mask = mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        if settings.brightness_jitter > 0:
            factor = random.uniform(
                1.0 - settings.brightness_jitter,
                1.0 + settings.brightness_jitter,
            )
            image = ImageEnhance.Brightness(image).enhance(factor)
        if settings.contrast_jitter > 0:
            factor = random.uniform(
                1.0 - settings.contrast_jitter,
                1.0 + settings.contrast_jitter,
            )
            image = ImageEnhance.Contrast(image).enhance(factor)
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return image, mask


def _records(split: str, paths: list[Path]) -> tuple[ImageRecord, ...]:
    return tuple(
        ImageRecord(index, f"{split}/{path.stem}", split, path)
        for index, path in enumerate(paths)
    )


def _write_manifest(path: Path, records: Iterable[ImageRecord]) -> None:
    write_csv(
        path,
        (
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "split": record.split,
                "image_path": str(record.image_path),
                "mask_path": "" if record.mask_path is None else str(record.mask_path),
            }
            for record in records
        ),
        ["sample_index", "sample_id", "split", "image_path", "mask_path"],
    )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _duts_pairs(root: Path, prefix: str) -> list[tuple[Path, Path]]:
    image_dir = _locate_directory(root, f"{prefix}-Image")
    mask_dir = _locate_directory(root, f"{prefix}-Mask")
    if image_dir is None or mask_dir is None:
        discovered = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_dir()
        ][:50]
        raise FileNotFoundError(
            f"Could not locate {prefix}-Image and {prefix}-Mask under {root}. "
            f"Discovered directories: {discovered}"
        )
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"{prefix} image/mask pairing failed: missing_masks={missing_masks[:20]}, "
            f"missing_images={missing_images[:20]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def _locate_directory(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    if root.is_dir() and root.name == name:
        return root
    if root.exists():
        matches = sorted(path for path in root.rglob(name) if path.is_dir())
        if matches:
            return matches[0]
    return None


def _validate_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{name} must contain exactly {expected:,} images/pairs before limits, "
            f"found {actual:,}. The archive may be incomplete or incorrectly extracted."
        )


def _download_and_extract_zip(
    url: str,
    archive: Path,
    destination: Path,
    *,
    remove_archive: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        _download_file(url, archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {archive}")
    print(f"Extracting {archive.name} into {destination}", flush=True)
    with zipfile.ZipFile(archive) as handle:
        _safe_extract(handle, destination)
    if remove_archive:
        archive.unlink(missing_ok=True)


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    existing = temporary.stat().st_size if temporary.is_file() else 0
    headers = {
        "User-Agent": "Mozilla/5.0 2026OpticsMoE dataset downloader",
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    action = f"Resuming at {existing / (1024 ** 3):.2f} GiB" if existing else "Downloading"
    print(f"{action}: {url} -> {path}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_status = int(getattr(response, "status", 200))
            resumed = bool(existing and response_status == 206)
            if existing and not resumed:
                print(
                    "  Server ignored the Range header; restarting this archive.",
                    flush=True,
                )
            copied = existing if resumed else 0
            response_bytes = int(response.headers.get("Content-Length", 0))
            total = copied + response_bytes if resumed and response_bytes else response_bytes
            report_step = 512 * 1024 * 1024
            next_report = ((copied // report_step) + 1) * report_step
            with temporary.open("ab" if resumed else "wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    copied += len(chunk)
                    if copied >= next_report:
                        suffix = (
                            f"/{total / (1024 ** 3):.2f} GiB"
                            if total
                            else ""
                        )
                        print(
                            f"  downloaded {copied / (1024 ** 3):.2f} GiB{suffix}",
                            flush=True,
                        )
                        next_report += 512 * 1024 * 1024
        temporary.replace(path)
    except urllib.error.HTTPError as exc:
        # A fully downloaded .part can receive 416 if interruption happened
        # after the last byte but before the atomic rename.
        if exc.code == 416 and temporary.is_file() and zipfile.is_zipfile(temporary):
            temporary.replace(path)
            return
        raise RuntimeError(
            f"Automatic dataset download failed: {url}. Partial data is kept "
            f"at {temporary} and will be resumed next time. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Automatic dataset download failed: {url}. Partial data is kept "
            f"at {temporary} and will be resumed next time. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _safe_extract(handle: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in handle.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination_resolved)
        except ValueError as exc:
            raise RuntimeError(
                f"Unsafe path in ZIP archive: {member.filename}"
            ) from exc
    handle.extractall(destination)
