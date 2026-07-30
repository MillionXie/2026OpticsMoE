from __future__ import annotations

import random
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.io_utils import (
    sha256_file,
    write_csv,
    write_json,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVES = {
    "train_images": "ISBI2016_ISIC_Part1_Training_Data.zip",
    "train_masks": "ISBI2016_ISIC_Part1_Training_GroundTruth.zip",
    "test_images": "ISBI2016_ISIC_Part1_Test_Data.zip",
    "test_masks": "ISBI2016_ISIC_Part1_Test_GroundTruth.zip",
}
DIRECTORIES = {
    "train_images": "ISBI2016_ISIC_Part1_Training_Data",
    "train_masks": "ISBI2016_ISIC_Part1_Training_GroundTruth",
    "test_images": "ISBI2016_ISIC_Part1_Test_Data",
    "test_masks": "ISBI2016_ISIC_Part1_Test_GroundTruth",
}


@dataclass(frozen=True)
class ISICRecord:
    sample_index: int
    sample_id: str
    split: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class DatasetBundle:
    train_records: tuple[ISICRecord, ...]
    test_records: tuple[ISICRecord, ...]
    metadata: dict[str, Any]


def prepare_isic2016(settings: Any, *, persist: bool = True) -> DatasetBundle:
    directories = _locate_all(settings.data_root)
    if directories is None and settings.auto_download:
        _download_all(settings)
        directories = _locate_all(settings.data_root)
    if directories is None:
        discovered = (
            [
                str(path.relative_to(settings.data_root))
                for path in settings.data_root.rglob("*")
                if path.is_dir()
            ][:50]
            if settings.data_root.exists()
            else []
        )
        raise FileNotFoundError(
            "ISBI 2016 Task 1 data are incomplete under "
            f"{settings.data_root}. Enable dataset.auto_download or place the "
            "official four extracted archives there. "
            f"Discovered directories: {discovered}"
        )

    full_train = _pair_split(
        "train",
        directories["train_images"],
        directories["train_masks"],
    )
    full_test = _pair_split(
        "test",
        directories["test_images"],
        directories["test_masks"],
    )
    if len(full_train) != settings.expected_train_samples:
        raise RuntimeError(
            f"Official ISIC train split must contain "
            f"{settings.expected_train_samples} pairs, found {len(full_train)}"
        )
    if len(full_test) != settings.expected_test_samples:
        raise RuntimeError(
            f"Official ISIC test split must contain "
            f"{settings.expected_test_samples} pairs, found {len(full_test)}"
        )
    train_pairs = _limit(full_train, settings.train_limit, settings.random_seed)
    test_pairs = _limit(full_test, settings.test_limit, settings.random_seed + 1)
    train_records = tuple(
        ISICRecord(index, image.stem, "train", image, mask)
        for index, (image, mask) in enumerate(train_pairs)
    )
    test_records = tuple(
        ISICRecord(index, image.stem, "test", image, mask)
        for index, (image, mask) in enumerate(test_pairs)
    )
    train_ids = {record.sample_id for record in train_records}
    test_ids = {record.sample_id for record in test_records}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise RuntimeError(f"ISIC train/test image leakage detected: {overlap[:10]}")

    metadata = {
        "dataset": "ISBI2016_ISIC_Task1",
        "task": "skin_lesion_boundary_binary_segmentation",
        "license": "CC0",
        "official_split": True,
        "validation_split": False,
        "full_train_samples": len(full_train),
        "full_test_samples": len(full_test),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_test_overlap": len(overlap),
        "image_size": settings.image_size,
        "mask_resize_interpolation": "nearest",
        "checkpoint_selection": "minimum_training_loss",
        "test_used_for_checkpoint_selection": False,
        "directories": {key: str(value) for key, value in directories.items()},
        "official_urls": {
            "train_images": settings.train_image_url,
            "train_masks": settings.train_mask_url,
            "test_images": settings.test_image_url,
            "test_masks": settings.test_mask_url,
        },
    }
    if persist:
        write_json(settings.output_dir / "dataset.json", metadata)
        _write_manifest(
            settings.output_dir / "manifests" / "samples.csv",
            (*train_records, *test_records),
        )
        write_json(
            settings.output_dir / "manifests" / "official_split.json",
            {
                "train": [record.sample_id for record in train_records],
                "test": [record.sample_id for record in test_records],
            },
        )
    return DatasetBundle(train_records, test_records, metadata)


class ISICSegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: tuple[ISICRecord, ...],
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
                f"ISIC source geometry mismatch for {record.sample_id}: "
                f"image={image.size}, mask={mask.size}"
            )
        image, mask, transform = paired_transform(
            image,
            mask,
            self.settings,
            training=self.training,
        )
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        if not set(torch.unique(mask_tensor).tolist()).issubset({0.0, 1.0}):
            raise RuntimeError("Nearest-neighbor transformed mask is not binary")
        return {
            "image": image,
            "mask": mask_tensor,
            "sample_index": record.sample_index,
            "sample_id": record.sample_id,
            "split": record.split,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "geometry_transform": transform,
        }


def collate_segmentation(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "masks": torch.stack([item["mask"] for item in batch]),
        "sample_indices": torch.tensor(
            [item["sample_index"] for item in batch], dtype=torch.long
        ),
        "sample_ids": [item["sample_id"] for item in batch],
        "splits": [item["split"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
        "geometry_transforms": [item["geometry_transform"] for item in batch],
    }


def build_loaders(
    bundle: DatasetBundle,
    settings: Any,
) -> tuple[DataLoader, DataLoader]:
    common: dict[str, Any] = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "collate_fn": collate_segmentation,
    }
    if settings.num_workers > 0:
        common["prefetch_factor"] = 2
    train_loader = DataLoader(
        ISICSegmentationDataset(bundle.train_records, settings, training=True),
        batch_size=settings.student_batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        ISICSegmentationDataset(bundle.test_records, settings, training=False),
        batch_size=settings.inference_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader


def paired_transform(
    image: Image.Image,
    mask: Image.Image,
    settings: Any,
    *,
    training: bool,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    transform: dict[str, Any] = {
        "crop_box_normalized": [0.0, 0.0, 1.0, 1.0],
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotation_degrees": 0.0,
    }
    if training and settings.augmentation_enabled:
        source_width, source_height = image.size
        scale = random.uniform(float(settings.crop_scale_min), 1.0)
        crop_width = max(1, round(source_width * scale))
        crop_height = max(1, round(source_height * scale))
        left = random.randint(0, max(0, source_width - crop_width))
        top = random.randint(0, max(0, source_height - crop_height))
        box = (left, top, left + crop_width, top + crop_height)
        transform["crop_box_normalized"] = [
            left / source_width,
            top / source_height,
            (left + crop_width) / source_width,
            (top + crop_height) / source_height,
        ]
        image, mask = image.crop(box), mask.crop(box)
        if random.random() < settings.horizontal_flip_probability:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            transform["horizontal_flip"] = True
        if random.random() < settings.vertical_flip_probability:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            transform["vertical_flip"] = True
        angle = random.uniform(
            -float(settings.rotation_degrees),
            float(settings.rotation_degrees),
        )
        transform["rotation_degrees"] = angle
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
    size = int(settings.image_size)
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return image, mask, transform


def _download_all(settings: Any) -> None:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    urls = {
        "train_images": settings.train_image_url,
        "train_masks": settings.train_mask_url,
        "test_images": settings.test_image_url,
        "test_masks": settings.test_mask_url,
    }
    archive_dir = settings.data_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    download_manifest: list[dict[str, Any]] = []
    for key, url in urls.items():
        archive = archive_dir / ARCHIVES[key]
        if not archive.is_file():
            _download_file(url, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f"Downloaded archive is not a valid ZIP: {archive}")
        _safe_extract(archive, settings.data_root)
        download_manifest.append(
            {
                "component": key,
                "url": url,
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            }
        )
    write_json(settings.data_root / "download_manifest.json", download_manifest)
    if settings.remove_archives_after_extract:
        shutil.rmtree(archive_dir, ignore_errors=True)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "2026OpticsMoE-ISIC2016/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=destination.parent,
                delete=False,
                suffix=".part",
            ) as handle:
                temporary = Path(handle.name)
                copied = 0
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    copied += len(chunk)
                    if total:
                        print(
                            f"[ISIC download] {destination.name}: "
                            f"{copied / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                            flush=True,
                        )
        temporary.replace(destination)
    except Exception as exc:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(
                    f"Unsafe path in ISIC archive {archive}: {member.filename}"
                )
        handle.extractall(destination)


def _locate_all(root: Path) -> dict[str, Path] | None:
    found: dict[str, Path] = {}
    for key, name in DIRECTORIES.items():
        direct = root / name
        candidates = [direct] if direct.is_dir() else []
        if root.exists():
            candidates.extend(
                path for path in root.rglob(name) if path.is_dir()
            )
        unique = sorted(set(candidates))
        if not unique:
            return None
        found[key] = unique[0]
    return found


def _pair_split(
    split: str,
    image_directory: Path,
    mask_directory: Path,
) -> list[tuple[Path, Path]]:
    images = {
        path.stem: path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    masks = {
        path.stem.removesuffix("_Segmentation"): path
        for path in mask_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"ISIC {split} pairing failed: missing masks={missing_masks[:20]}, "
            f"missing images={missing_images[:20]}"
        )
    pairs = [(images[key], masks[key]) for key in sorted(images)]
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        if image_size != mask_size:
            raise RuntimeError(
                f"ISIC official pair geometry mismatch: image={image_path} "
                f"{image_size}, mask={mask_path} {mask_size}"
            )
    return pairs


def _limit(
    pairs: list[tuple[Path, Path]],
    limit: int | None,
    seed: int,
) -> list[tuple[Path, Path]]:
    if limit is None or limit >= len(pairs):
        return list(pairs)
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    return [pairs[index] for index in sorted(indices[:limit])]


def _write_manifest(path: Path, records: Iterable[ISICRecord]) -> None:
    write_csv(
        path,
        (
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "split": record.split,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
            }
            for record in records
        ),
        ["sample_index", "sample_id", "split", "image_path", "mask_path"],
    )


__all__ = [
    "DatasetBundle",
    "ISICRecord",
    "ISICSegmentationDataset",
    "build_loaders",
    "collate_segmentation",
    "paired_transform",
    "prepare_isic2016",
]
