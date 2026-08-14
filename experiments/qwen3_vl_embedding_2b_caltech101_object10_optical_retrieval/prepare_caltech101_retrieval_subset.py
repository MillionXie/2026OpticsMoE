from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
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
EXPECTED_OBJECT_CLASS_COUNT = 101
OFFICIAL_ARCHIVE_MD5 = "3138e1922a9193bfa496528edbbc45d0"
BACKGROUND_CLASS = "BACKGROUND_Google"
SPLIT_ALGORITHM = "sha256_per_class_gallery_query_train_v1"


@dataclass(frozen=True)
class Caltech101Sample:
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
class Caltech101RetrievalBundle:
    train_samples: tuple[Caltech101Sample, ...]
    test_samples: tuple[Caltech101Sample, ...]
    gallery_samples: tuple[Caltech101Sample, ...]
    class_names: tuple[str, ...]
    manifest_digest: str
    metadata: dict[str, Any]

    def all_samples(self) -> tuple[Caltech101Sample, ...]:
        return self.gallery_samples + self.train_samples + self.test_samples


class Caltech101RetrievalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Caltech101Sample],
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


def collate_caltech101(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "samples": [item["sample"] for item in items],
        "dataset_indices": [int(item["dataset_index"]) for item in items],
    }


def prepare_caltech101_subset(
    settings: Settings, *, persist: bool = True
) -> Caltech101RetrievalBundle:
    image_root = _ensure_dataset(settings)
    available = _discover_classes(image_root)
    if settings.use_all_classes:
        class_names = tuple(sorted(available))
        if len(class_names) != EXPECTED_OBJECT_CLASS_COUNT:
            raise RuntimeError(
                f"Caltech-101 all-class mode requires {EXPECTED_OBJECT_CLASS_COUNT} object "
                f"classes after excluding {BACKGROUND_CLASS!r}, "
                f"but discovered {len(class_names)} under {image_root}"
            )
    else:
        class_names = settings.selected_classes
        missing = [name for name in class_names if name not in available]
        if missing:
            raise RuntimeError(
                f"Configured Caltech101 classes are absent: {missing}. Available classes: "
                f"{sorted(available)}"
            )

    train: list[Caltech101Sample] = []
    test: list[Caltech101Sample] = []
    galleries: list[Caltech101Sample] = []
    per_class_source_counts: dict[str, int] = {}
    official_ids = {name: index + 1 for index, name in enumerate(sorted(available))}
    for class_index, class_name in enumerate(class_names):
        images = available[class_name]
        if len(images) < settings.minimum_images_per_class:
            raise RuntimeError(
                f"Caltech101 class {class_name!r} has only {len(images)} images; "
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
        "dataset": "Caltech-101",
        "dataset_version": "CaltechDATA record 20086, version 1.0",
        "dataset_source": "https://data.caltech.edu/records/mzrjq-6wc02",
        "dataset_root": str(settings.dataset_root),
        "image_root": str(image_root),
        "task_definition": "class-level image-to-image object retrieval",
        "identity_unit": "object category; Caltech-101 does not provide instance identity IDs",
        "excluded_directories": [BACKGROUND_CLASS],
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
            "object_classes": EXPECTED_OBJECT_CLASS_COUNT,
            "images_per_class": "about 40 to 800; most categories about 50",
            "archive_md5": OFFICIAL_ARCHIVE_MD5,
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
    bundle = Caltech101RetrievalBundle(
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
            f"Caltech-101 101_ObjectCategories directory was not found under "
            f"{settings.dataset_root}. Enable dataset.download or place the official archive "
            "contents there."
        )
    settings.dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = settings.dataset_root.parent / "caltech-101.zip"
    _download_resumable(
        settings.download_url,
        archive_path,
        timeout=settings.download_timeout_sec,
    )
    _safe_extract_zip(archive_path, settings.dataset_root)
    _extract_nested_object_archive(settings.dataset_root)
    found = _find_image_root(settings.dataset_root)
    if found is None:
        raise RuntimeError(
            f"Downloaded Caltech-101 archive did not contain a usable "
            f"101_ObjectCategories directory: {archive_path}"
        )
    if settings.delete_archive_after_extract:
        archive_path.unlink(missing_ok=True)
    return found


def _download_resumable(url: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.is_file() else 0
    headers = {"User-Agent": "2026OpticsMoE-Caltech101Retrieval/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(
            f"Resuming Caltech-101 download at {existing / (1024 ** 2):.1f} MiB: "
            f"{destination}", flush=True
        )
    else:
        print(f"Downloading official Caltech-101 archive (137.4 MB): {url}", flush=True)
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            append = existing > 0 and status == 206
            if existing > 0 and not append:
                print("Server ignored Range; restarting the partial archive safely", flush=True)
            mode = "ab" if append else "wb"
            downloaded = existing if append else 0
            with destination.open(mode) as target:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    downloaded += len(chunk)
                    if downloaded // (32 * 1024 * 1024) != (
                        downloaded - len(chunk)
                    ) // (32 * 1024 * 1024):
                        print(
                            f"[Caltech-101 download] {downloaded / (1024 ** 2):.1f} MiB",
                            flush=True,
                        )
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Failed to download Caltech101 from {url}. The partial archive is retained at "
            f"{destination} for a later resume. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if not zipfile.is_zipfile(destination):
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP archive: {destination}. "
            "Remove it only if the server returned an error page rather than a partial ZIP."
        )
    digest = _md5(destination)
    if digest != OFFICIAL_ARCHIVE_MD5:
        raise RuntimeError(
            f"Caltech-101 archive MD5 mismatch: expected {OFFICIAL_ARCHIVE_MD5}, got "
            f"{digest}. Delete the corrupt archive and retry."
        )


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    print(f"Extracting Caltech101 archive to {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((destination, target)) != str(destination):
                raise RuntimeError(f"Unsafe path in Caltech101 ZIP: {member.filename}")
        archive.extractall(destination)


def _extract_nested_object_archive(root: Path) -> None:
    if _find_image_root(root) is not None:
        return
    archives = sorted(root.rglob("101_ObjectCategories.tar.gz"))
    archives.extend(root.rglob("101_ObjectCategories.tar"))
    if not archives:
        return
    archive_path = archives[0]
    destination = archive_path.parent.resolve()
    print(f"Extracting nested object archive {archive_path}", flush=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath((destination, target)) != str(destination):
                raise RuntimeError(f"Unsafe path in Caltech-101 TAR: {member.name}")
        archive.extractall(destination)


def _find_image_root(root: Path) -> Path | None:
    candidates = [
        root / "101_ObjectCategories",
        root / "caltech-101" / "101_ObjectCategories",
        root,
    ]
    candidates.extend(path for path in root.glob("*/101_ObjectCategories") if path.is_dir())
    candidates.extend(path for path in root.glob("*/*/101_ObjectCategories") if path.is_dir())
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
        if class_dir.name == BACKGROUND_CLASS:
            continue
        images = _image_files(class_dir)
        if images:
            classes[class_dir.name] = images
    if len(classes) < 2:
        raise RuntimeError(f"Fewer than two Caltech101 class directories found under {image_root}")
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


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> list[Caltech101Sample]:
    return [
        Caltech101Sample(
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
    train: Sequence[Caltech101Sample],
    test: Sequence[Caltech101Sample],
    gallery: Sequence[Caltech101Sample],
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
        raise RuntimeError(f"Caltech101 image leakage detected: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed Caltech101 retrieval manifests")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_caltech101_subset(settings, persist=True)
    print(
        f"Prepared Caltech101-{len(bundle.class_names)}: train={len(bundle.train_samples):,} "
        f"query={len(bundle.test_samples):,} gallery={len(bundle.gallery_samples):,} "
        f"manifest={bundle.manifest_digest[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
