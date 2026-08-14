from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .io_utils import sha256_records, write_csv, write_json
from .settings import Settings, load_settings


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_AWA2_CLASS_COUNT = 50
EXPECTED_AWA2_IMAGE_COUNT = 37_322
SPLIT_ALGORITHM = "sha256_per_class_gallery_query_train_v1"


@dataclass(frozen=True)
class AwA2Sample:
    sample_id: str
    image_path: Path
    class_id: int
    class_name: str
    class_index: int
    split: str
    source_split: str
    is_gallery: bool

    def manifest_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path.resolve())
        return value


@dataclass(frozen=True)
class AwA2RetrievalBundle:
    train_samples: tuple[AwA2Sample, ...]
    test_samples: tuple[AwA2Sample, ...]
    gallery_samples: tuple[AwA2Sample, ...]
    class_names: tuple[str, ...]
    manifest_digest: str
    metadata: dict[str, Any]

    def all_samples(self) -> tuple[AwA2Sample, ...]:
        return self.gallery_samples + self.train_samples + self.test_samples


class AwA2RetrievalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[AwA2Sample],
        image_size: int,
        *,
        augment: bool = False,
        crop_scale_min: float = 0.9,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
        rotation_degrees: float = 5.0,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.crop_scale_min = float(crop_scale_min)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.rotation_degrees = float(rotation_degrees)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        return {"image": self._transform(image), "sample": sample, "dataset_index": index}

    def _transform(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        import random

        scale = random.uniform(self.crop_scale_min, 1.0)
        crop_w = max(1, round(image.width * scale))
        crop_h = max(1, round(image.height * scale))
        left = random.randint(0, max(0, image.width - crop_w))
        top = random.randint(0, max(0, image.height - crop_h))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if random.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
        if self.rotation_degrees:
            image = image.rotate(
                random.uniform(-self.rotation_degrees, self.rotation_degrees),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0, 0, 0),
            )
        if self.brightness_jitter:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            )
        if self.contrast_jitter:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            )
        return image


def collate_awa2(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "samples": [item["sample"] for item in items],
        "dataset_indices": [int(item["dataset_index"]) for item in items],
    }


def prepare_awa2_subset(
    settings: Settings, *, persist: bool = True
) -> AwA2RetrievalBundle:
    image_root = _ensure_dataset(settings)
    available = _discover_classes(image_root)
    if settings.use_all_classes:
        class_names = tuple(sorted(available))
        if len(class_names) != EXPECTED_AWA2_CLASS_COUNT:
            raise RuntimeError(
                f"AwA2 all-class mode requires {EXPECTED_AWA2_CLASS_COUNT} classes, "
                f"but discovered {len(class_names)} under {image_root}"
            )
    else:
        class_names = settings.selected_classes
        missing = [name for name in class_names if name not in available]
        if missing:
            raise RuntimeError(
                f"Configured AwA2 classes are absent: {missing}. Available classes: "
                f"{sorted(available)}"
            )

    train: list[AwA2Sample] = []
    test: list[AwA2Sample] = []
    galleries: list[AwA2Sample] = []
    per_class_source_counts: dict[str, int] = {}
    official_ids = {name: index + 1 for index, name in enumerate(sorted(available))}
    for class_index, class_name in enumerate(class_names):
        images = available[class_name]
        if len(images) < settings.minimum_images_per_class:
            raise RuntimeError(
                f"AwA2 class {class_name!r} has only {len(images)} images; "
                f"minimum_images_per_class={settings.minimum_images_per_class}"
            )
        per_class_source_counts[class_name] = len(images)
        ranked = sorted(
            images,
            key=lambda path: _stable_rank(
                settings.random_seed, class_name, path.relative_to(image_root).as_posix()
            ),
        )
        query_count = max(1, round(len(ranked) * settings.test_fraction))
        gallery_count = settings.gallery_images_per_class
        if gallery_count + query_count >= len(ranked):
            raise RuntimeError(
                f"Split leaves no training images for {class_name}: total={len(ranked)}, "
                f"gallery={gallery_count}, query={query_count}"
            )
        gallery_paths = ranked[:gallery_count]
        test_paths = ranked[gallery_count : gallery_count + query_count]
        train_paths = ranked[gallery_count + query_count :]
        train_paths = _limit_paths(train_paths, settings.train_limit_per_class)
        test_paths = _limit_paths(test_paths, settings.test_limit_per_class)
        class_id = official_ids[class_name]
        galleries.extend(
            _make_samples(
                gallery_paths, class_id, class_name, class_index, "gallery", True, image_root
            )
        )
        train.extend(
            _make_samples(
                train_paths, class_id, class_name, class_index, "train", False, image_root
            )
        )
        test.extend(
            _make_samples(
                test_paths, class_id, class_name, class_index, "test", False, image_root
            )
        )

    _validate_partition(train, test, galleries, class_names)
    records = [
        sample.manifest_record()
        for sample in sorted(
            galleries + train + test,
            key=lambda item: (item.split, item.class_index, item.sample_id),
        )
    ]
    digest = sha256_records(records)
    metadata = {
        "dataset": "Animals with Attributes 2 (AwA2)",
        "dataset_version": "1.0 (2017-06-09)",
        "dataset_source": "https://cvml.ista.ac.at/AwA2/",
        "dataset_root": str(settings.dataset_root),
        "image_root": str(image_root),
        "task_definition": "class-level image-to-image animal retrieval",
        "identity_unit": "animal species/class; AwA2 does not provide individual-animal IDs",
        "selected_classes": list(class_names),
        "use_all_classes": settings.use_all_classes,
        "split_policy": (
            "Per class, stable SHA256 ordering selects disjoint gallery and query images; "
            "all remaining images are train. Query images are excluded from both stages."
        ),
        "split_algorithm": SPLIT_ALGORITHM,
        "test_fraction": settings.test_fraction,
        "validation_split_created": False,
        "seed": settings.random_seed,
        "manifest_sha256": digest,
        "official_expected_counts": {
            "classes": EXPECTED_AWA2_CLASS_COUNT,
            "images": EXPECTED_AWA2_IMAGE_COUNT,
        },
        "source_counts": {
            "classes": len(available),
            "images": sum(len(paths) for paths in available.values()),
            "selected_images": sum(per_class_source_counts.values()),
        },
        "counts": {
            "train": len(train),
            "test": len(test),
            "gallery": len(galleries),
        },
        "per_class_counts": {
            split: dict(sorted(Counter(sample.class_name for sample in values).items()))
            for split, values in (("train", train), ("test", test), ("gallery", galleries))
        },
    }
    bundle = AwA2RetrievalBundle(
        tuple(train), tuple(test), tuple(galleries), class_names, digest, metadata
    )
    if persist:
        write_csv(
            settings.subset_manifest_path,
            [sample.manifest_record() for sample in bundle.all_samples()],
            [
                "sample_id",
                "image_path",
                "class_id",
                "class_name",
                "class_index",
                "split",
                "source_split",
                "is_gallery",
            ],
        )
        write_json(settings.output_dir / "dataset.json", metadata)
        write_json(
            settings.output_dir
            / "manifests"
            / f"{settings.dataset_variant}_subset_metadata.json",
            metadata,
        )
    return bundle


def _ensure_dataset(settings: Settings) -> Path:
    found = _find_image_root(settings.dataset_root)
    if found is not None:
        return found
    if not settings.download:
        raise FileNotFoundError(
            f"AwA2 JPEGImages directory was not found under {settings.dataset_root}. "
            "Enable dataset.download or place the official AwA2-data archive contents there."
        )
    settings.dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = settings.dataset_root.parent / "AwA2-data.zip"
    _download_resumable(
        settings.download_url,
        archive_path,
        timeout=settings.download_timeout_sec,
    )
    _safe_extract_zip(archive_path, settings.dataset_root)
    found = _find_image_root(settings.dataset_root)
    if found is None:
        raise RuntimeError(
            f"Downloaded AwA2 archive did not contain a usable JPEGImages directory: {archive_path}"
        )
    if settings.delete_archive_after_extract:
        archive_path.unlink(missing_ok=True)
    return found


def _download_resumable(url: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.is_file() else 0
    headers = {"User-Agent": "2026OpticsMoE-AwA2Retrieval/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming AwA2 download at {existing / (1024 ** 3):.2f} GiB: {destination}")
    else:
        print(f"Downloading official AwA2 image archive (~13 GB): {url}")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            append = existing > 0 and status == 206
            if existing > 0 and not append:
                print("Server ignored Range; restarting the partial AwA2 archive safely")
            mode = "ab" if append else "wb"
            downloaded = existing if append else 0
            with destination.open(mode) as target:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    downloaded += len(chunk)
                    if downloaded // (512 * 1024 * 1024) != (
                        downloaded - len(chunk)
                    ) // (512 * 1024 * 1024):
                        print(f"[AwA2 download] {downloaded / (1024 ** 3):.2f} GiB")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Failed to download AwA2 from {url}. The partial archive is retained at "
            f"{destination} for a later resume. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if not zipfile.is_zipfile(destination):
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP archive: {destination}. "
            "Remove it only if the server returned an error page rather than a partial ZIP."
        )


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    print(f"Extracting AwA2 archive to {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((destination, target)) != str(destination):
                raise RuntimeError(f"Unsafe path in AwA2 ZIP: {member.filename}")
        archive.extractall(destination)


def _find_image_root(root: Path) -> Path | None:
    candidates = [
        root / "JPEGImages",
        root / "Animals_with_Attributes2" / "JPEGImages",
        root,
    ]
    candidates.extend(path for path in root.glob("*/JPEGImages") if path.is_dir())
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        if len(class_dirs) >= 2 and any(_image_files(path) for path in class_dirs[:3]):
            return candidate.resolve()
    return None


def _discover_classes(image_root: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        images = _image_files(class_dir)
        if images:
            classes[class_dir.name] = images
    if len(classes) < 2:
        raise RuntimeError(f"Fewer than two AwA2 class directories found under {image_root}")
    return classes


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _stable_rank(seed: int, class_name: str, relative_path: str) -> str:
    payload = f"{SPLIT_ALGORITHM}|{seed}|{class_name}|{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _limit_paths(paths: Sequence[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return list(paths)
    if int(limit) <= 0:
        raise ValueError("Per-class limits must be positive or null")
    return list(paths[: int(limit)])


def _make_samples(
    paths: Sequence[Path],
    class_id: int,
    class_name: str,
    class_index: int,
    split: str,
    is_gallery: bool,
    image_root: Path,
) -> list[AwA2Sample]:
    return [
        AwA2Sample(
            sample_id=f"{split}:{class_name}:{path.relative_to(image_root).as_posix()}",
            image_path=path,
            class_id=class_id,
            class_name=class_name,
            class_index=class_index,
            split=split,
            source_split=f"deterministic_{split}",
            is_gallery=is_gallery,
        )
        for path in paths
    ]


def _validate_partition(
    train: Sequence[AwA2Sample],
    test: Sequence[AwA2Sample],
    gallery: Sequence[AwA2Sample],
    class_names: Sequence[str],
) -> None:
    expected = set(class_names)
    for name, values in (("train", train), ("test", test), ("gallery", gallery)):
        missing = sorted(expected - {sample.class_name for sample in values})
        if missing:
            raise RuntimeError(f"{name} split has no samples for classes: {missing}")
    path_sets = {
        "train": {sample.image_path for sample in train},
        "test": {sample.image_path for sample in test},
        "gallery": {sample.image_path for sample in gallery},
    }
    leakage = {
        "train_test": path_sets["train"] & path_sets["test"],
        "train_gallery": path_sets["train"] & path_sets["gallery"],
        "test_gallery": path_sets["test"] & path_sets["gallery"],
    }
    bad = {name: sorted(map(str, paths)) for name, paths in leakage.items() if paths}
    if bad:
        raise RuntimeError(f"AwA2 image leakage detected: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed AwA2 retrieval manifests")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_awa2_subset(settings, persist=True)
    print(
        f"Prepared AwA2-{len(bundle.class_names)}: train={len(bundle.train_samples):,} "
        f"query={len(bundle.test_samples):,} gallery={len(bundle.gallery_samples):,} "
        f"manifest={bundle.manifest_digest[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
