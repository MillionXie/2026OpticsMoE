from __future__ import annotations

import hashlib
import random
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    sha256_records,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GrocerySample,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import Settings


ARCHIVE_MD5 = "3138e1922a9193bfa496528edbbc45d0"
BACKGROUND_CATEGORY = "BACKGROUND_Google"


def prepare_caltech101_subset(
    settings: Settings, *, persist: bool = True
) -> GroceryRetrievalBundle:
    categories_root = _ensure_dataset(settings)
    available = {
        path.name: path
        for path in categories_root.iterdir()
        if path.is_dir() and path.name != BACKGROUND_CATEGORY
    }
    missing = [name for name in settings.selected_skus if name not in available]
    if missing:
        raise RuntimeError(f"Caltech101 categories are missing: {missing}")

    train: list[GrocerySample] = []
    test: list[GrocerySample] = []
    gallery: list[GrocerySample] = []
    for class_index, class_name in enumerate(settings.selected_skus):
        paths = sorted(available[class_name].glob("*.jpg"))
        required = settings.gallery_images_per_sku + 2
        if len(paths) < required:
            raise RuntimeError(
                f"{class_name} has {len(paths)} images, fewer than required {required}"
            )
        random.Random(f"{settings.random_seed}:{class_name}").shuffle(paths)
        gallery_paths = paths[: settings.gallery_images_per_sku]
        remainder = paths[settings.gallery_images_per_sku :]
        train_count = (
            len(remainder) - 1
            if settings.train_limit_per_sku is None
            else min(int(settings.train_limit_per_sku), len(remainder) - 1)
        )
        train_paths = remainder[:train_count]
        test_paths = remainder[train_count:]
        if settings.test_limit_per_sku is not None:
            test_paths = test_paths[: int(settings.test_limit_per_sku)]
        if not train_paths or not test_paths:
            raise RuntimeError(f"{class_name} produced an empty train or test partition")
        gallery.extend(
            _samples(gallery_paths, class_index, class_name, "gallery", True)
        )
        train.extend(_samples(train_paths, class_index, class_name, "train", False))
        test.extend(_samples(test_paths, class_index, class_name, "test", False))

    records = [
        sample.manifest_record()
        for sample in sorted(
            gallery + train + test,
            key=lambda item: (item.split, item.sku_index, item.sample_id),
        )
    ]
    digest = sha256_records(records)
    metadata = {
        "dataset": "Caltech101",
        "dataset_source": "https://data.caltech.edu/records/mzrjq-6wc02",
        "dataset_root": str(settings.dataset_root),
        "selected_categories": list(settings.selected_skus),
        "excluded_categories": [BACKGROUND_CATEGORY],
        "split_policy": (
            "seeded per-category shuffle; disjoint gallery, train, then test; "
            "class imbalance capped by configured per-category limits"
        ),
        "seed": settings.random_seed,
        "manifest_sha256": digest,
        "counts": {
            "train": len(train),
            "test": len(test),
            "gallery": len(gallery),
        },
        "per_category_counts": {
            split: dict(sorted(Counter(item.sku_name for item in values).items()))
            for split, values in (
                ("train", train),
                ("test", test),
                ("gallery", gallery),
            )
        },
    }
    bundle = GroceryRetrievalBundle(
        tuple(train),
        tuple(test),
        tuple(gallery),
        tuple(settings.selected_skus),
        digest,
        metadata,
    )
    if persist:
        write_csv(
            settings.subset_manifest_path,
            [sample.manifest_record() for sample in bundle.all_samples()],
            [
                "sample_id",
                "image_path",
                "sku_id",
                "sku_name",
                "sku_index",
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


def _samples(
    paths: Iterable[Path],
    class_index: int,
    class_name: str,
    split: str,
    is_gallery: bool,
) -> list[GrocerySample]:
    return [
        GrocerySample(
            sample_id=f"caltech101:{class_name}:{path.stem}",
            image_path=path,
            sku_id=class_index,
            sku_name=class_name,
            sku_index=class_index,
            split=split,
            source_split="official_all",
            is_gallery=is_gallery,
        )
        for path in paths
    ]


def _ensure_dataset(settings: Settings) -> Path:
    found = _find_categories_root(settings.dataset_root)
    if found is not None:
        return found
    if not settings.download:
        raise FileNotFoundError(
            f"Caltech101 101_ObjectCategories was not found below {settings.dataset_root}"
        )

    settings.dataset_root.parent.mkdir(parents=True, exist_ok=True)
    settings.dataset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=settings.dataset_root.parent) as temporary:
        archive_path = Path(temporary) / "caltech-101.zip"
        request = urllib.request.Request(
            settings.download_url,
            headers={"User-Agent": "2026OpticsMoE-Caltech101Retrieval/1.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, archive_path.open(
            "wb"
        ) as target:
            shutil.copyfileobj(response, target)
        if _md5(archive_path) != ARCHIVE_MD5:
            raise RuntimeError("Caltech101 archive MD5 mismatch")
        _safe_extract_zip(archive_path, settings.dataset_root)

    found = _find_categories_root(settings.dataset_root)
    if found is None:
        for archive in settings.dataset_root.rglob("101_ObjectCategories.tar.gz"):
            _safe_extract_tar(archive, archive.parent)
        found = _find_categories_root(settings.dataset_root)
    if found is None:
        raise RuntimeError("Downloaded Caltech101 archive has no 101_ObjectCategories")
    return found


def _find_categories_root(root: Path) -> Path | None:
    direct = (root / "101_ObjectCategories", root / "caltech-101" / "101_ObjectCategories")
    for candidate in direct:
        if candidate.is_dir():
            return candidate
    return next((path for path in root.rglob("101_ObjectCategories") if path.is_dir()), None)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        source.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if member.issym() or member.islnk() or (target != root and root not in target.parents):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
        source.extractall(destination, members=members)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
