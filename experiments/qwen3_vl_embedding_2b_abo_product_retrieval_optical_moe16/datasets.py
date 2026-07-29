from __future__ import annotations

import csv
import gzip
import json
import math
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .io_utils import canonical_digest, read_json, rows_digest, stable_u64, write_json


MANIFEST_COLUMNS = (
    "image_id",
    "item_id",
    "relative_image_path",
    "product_type",
    "is_main",
    "original_width",
    "original_height",
    "quality_score",
    "stage1_train",
    "stage2_split",
)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    width: int
    height: int
    is_main: bool


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    product_type: str
    images: tuple[ImageRecord, ...]


@dataclass(frozen=True)
class ABOSample:
    image_id: str
    item_id: str
    item_index: int
    image_path: Path
    product_type: str
    split: str
    quality_score: float


class ABORetrievalDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: Sequence[ABOSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            with Image.open(sample.image_path) as image:
                rgb = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode ABO image {sample.image_id}: {sample.image_path}"
            ) from exc
        return {
            "image": rgb,
            "image_id": sample.image_id,
            "item_id": sample.item_id,
            "item_index": sample.item_index,
            "image_path": str(sample.image_path),
            "product_type": sample.product_type,
            "split": sample.split,
            "quality_score": sample.quality_score,
        }


@dataclass(frozen=True)
class ABOBundle:
    stage1_train: ABORetrievalDataset
    stage2_train: ABORetrievalDataset
    gallery: ABORetrievalDataset
    query: ABORetrievalDataset
    stage1_item_ids: tuple[str, ...]
    stage2_item_ids: tuple[str, ...]
    manifest_digest: str
    manifest_path: Path
    metadata: dict[str, Any]


def collate_abo(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [row["image"] for row in rows],
        "image_ids": [row["image_id"] for row in rows],
        "item_ids": [row["item_id"] for row in rows],
        "item_indices": torch.tensor(
            [row["item_index"] for row in rows], dtype=torch.long
        ),
        "image_paths": [row["image_path"] for row in rows],
        "product_types": [row["product_type"] for row in rows],
        "splits": [row["split"] for row in rows],
    }


def prepare_abo(settings: Any, *, persist: bool = True) -> ABOBundle:
    settings.dataset_root.mkdir(parents=True, exist_ok=True)
    settings.artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.manifest_csv.is_file() and settings.manifest_metadata_json.is_file():
        metadata = read_json(settings.manifest_metadata_json)
        expected = canonical_digest(settings.split_identity())
        if metadata.get("split_identity_digest") != expected:
            if not settings.rebuild_manifest:
                raise RuntimeError(
                    "Existing ABO Query/Gallery manifest was created with different split "
                    "settings. It is intentionally not overwritten because this would change "
                    "the held-out images. Set dataset.rebuild_manifest=true only when starting "
                    "a new experiment."
                )
        else:
            return _bundle_from_manifest(settings, metadata)

    if not settings.download and not _official_layout_exists(settings.dataset_root):
        raise FileNotFoundError(_missing_data_message(settings.dataset_root))
    ensure_abo_downloaded(settings)
    items = load_catalog_items(settings)
    rows, selection = build_fixed_manifest(items, settings)
    digest = rows_digest(rows)
    assert_no_image_leakage(rows)
    metadata = {
        "dataset": "Amazon Berkeley Objects catalog images",
        "manifest_version": "abo_fixed_split_v1",
        "manifest_sha256": digest,
        "split_identity": settings.split_identity(),
        "split_identity_digest": canonical_digest(settings.split_identity()),
        "leakage_policy": (
            "Gallery and query image_ids are fixed before Stage 1 and are forbidden "
            "from both Stage-1 pretraining and Stage-2 fine-tuning."
        ),
        **selection,
    }
    if persist:
        _write_manifest(settings.manifest_csv, rows)
        write_json(settings.manifest_metadata_json, metadata)
        write_json(
            settings.output_dir / "dataset.json",
            {
                **metadata,
                "canonical_manifest_csv": str(settings.manifest_csv),
                "canonical_manifest_metadata": str(settings.manifest_metadata_json),
            },
        )
    return _bundle_from_rows(settings, rows, metadata)


def ensure_abo_downloaded(settings: Any) -> None:
    root = settings.dataset_root
    if not _listing_files(root):
        if not settings.download:
            raise FileNotFoundError(_missing_data_message(root))
        archive = root / "downloads" / "abo-listings.tar"
        _download_resumable(settings.listings_url, archive)
        _safe_extract_tar(archive, root)
    if _image_metadata(root) is None or _small_image_root(root) is None:
        if not settings.download:
            raise FileNotFoundError(_missing_data_message(root))
        archive = root / "downloads" / "abo-images-small.tar"
        _download_resumable(settings.images_url, archive)
        _safe_extract_tar(archive, root)
    if not _official_layout_exists(root):
        raise RuntimeError(_missing_data_message(root))


def load_catalog_items(settings: Any) -> list[CatalogItem]:
    metadata_path = _image_metadata(settings.dataset_root)
    small_root = _small_image_root(settings.dataset_root)
    if metadata_path is None or small_root is None:
        raise FileNotFoundError(_missing_data_message(settings.dataset_root))
    image_metadata: dict[str, tuple[str, int, int]] = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"image_id", "height", "width", "path"}
        if not expected.issubset(reader.fieldnames or ()):
            raise RuntimeError(
                f"ABO image metadata columns are {reader.fieldnames}; expected {sorted(expected)}"
            )
        for row in reader:
            image_metadata[str(row["image_id"])] = (
                str(row["path"]),
                int(row["width"]),
                int(row["height"]),
            )

    raw_items: list[tuple[str, str, list[tuple[str, bool]]]] = []
    image_owners: dict[str, set[str]] = defaultdict(set)
    for path in _listing_files(settings.dataset_root):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {path}:{line_number}") from exc
                item_id = str(row.get("item_id", "")).strip()
                main = str(row.get("main_image_id", "")).strip()
                others = [
                    str(value).strip()
                    for value in (row.get("other_image_id") or [])
                    if str(value).strip()
                ]
                if not item_id or not main:
                    continue
                product_types = row.get("product_type") or []
                product_type = "UNKNOWN"
                if product_types and isinstance(product_types[0], dict):
                    product_type = str(product_types[0].get("value") or "UNKNOWN")
                references = [(main, True)] + [
                    (image_id, False) for image_id in others if image_id != main
                ]
                references = list(dict.fromkeys(references))
                raw_items.append((item_id, product_type, references))
                for image_id, _ in references:
                    image_owners[image_id].add(item_id)

    ambiguous = {image_id for image_id, owners in image_owners.items() if len(owners) > 1}
    items: list[CatalogItem] = []
    missing_files = 0
    for item_id, product_type, references in raw_items:
        images: list[ImageRecord] = []
        for image_id, is_main in references:
            if image_id in ambiguous or image_id not in image_metadata:
                continue
            relative_path, width, height = image_metadata[image_id]
            if min(width, height) < settings.minimum_original_short_side:
                continue
            image_path = small_root / relative_path
            if not image_path.is_file():
                missing_files += 1
                continue
            images.append(
                ImageRecord(
                    image_id=image_id,
                    relative_path=relative_path,
                    width=width,
                    height=height,
                    is_main=is_main,
                )
            )
        if images:
            items.append(CatalogItem(item_id, product_type, tuple(images)))
    if missing_files:
        print(f"WARNING: ignored {missing_files:,} metadata rows whose small image is missing")
    if not items:
        raise RuntimeError("No usable ABO catalog items remain after metadata validation")
    return items


def build_fixed_manifest(
    catalog_items: Sequence[CatalogItem], settings: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_stage2 = [
        item
        for item in catalog_items
        if len(item.images) >= settings.stage2_min_images_per_item
    ]
    if len(eligible_stage2) < settings.stage2_item_count:
        raise RuntimeError(
            f"Requested {settings.stage2_item_count:,} Stage-2 items with >="
            f"{settings.stage2_min_images_per_item} images, but only "
            f"{len(eligible_stage2):,} are eligible"
        )
    selected_stage2, quality_scores = _select_stage2_items(eligible_stage2, settings)
    selected_stage2_ids = {item.item_id for item in selected_stage2}
    stage2_assignments: dict[str, dict[str, str]] = {}
    selected_images_by_item: dict[str, tuple[ImageRecord, ...]] = {}
    held_out_ids: set[str] = set()
    for item in selected_stage2:
        selected = _rank_item_images(item, quality_scores, settings)[
            : settings.stage2_max_images_per_item
        ]
        assignments = split_stage2_images(
            item.item_id,
            [image.image_id for image in selected],
            settings.random_seed,
            settings.stage2_train_fraction,
            settings.stage2_gallery_fraction,
            settings.stage2_query_fraction,
            preferred_gallery_ids=[
                image.image_id for image in selected if image.is_main
            ],
        )
        stage2_assignments[item.item_id] = assignments
        selected_images_by_item[item.item_id] = tuple(selected)
        held_out_ids.update(
            image_id
            for image_id, split in assignments.items()
            if split in {"gallery", "query"}
        )

    eligible_stage1 = []
    for item in catalog_items:
        available = [image for image in item.images if image.image_id not in held_out_ids]
        if len(available) >= settings.stage1_min_images_per_item:
            eligible_stage1.append(item)
    extras = sorted(
        (item for item in eligible_stage1 if item.item_id not in selected_stage2_ids),
        key=lambda item: (
            -len(item.images),
            stable_u64(settings.random_seed, "stage1-item", item.item_id),
        ),
    )
    stage1_items = list(selected_stage2)
    stage1_items.extend(extras[: max(0, settings.stage1_item_count - len(stage1_items))])
    if len(stage1_items) != settings.stage1_item_count:
        raise RuntimeError(
            f"Could select only {len(stage1_items):,}/{settings.stage1_item_count:,} "
            "Stage-1 items after excluding fixed Query/Gallery images"
        )
    stage1_image_ids = _allocate_stage1_images(
        stage1_items, held_out_ids, settings
    )

    by_item = {item.item_id: item for item in catalog_items}
    rows: list[dict[str, Any]] = []
    all_item_ids = sorted({item.item_id for item in stage1_items} | selected_stage2_ids)
    for item_id in all_item_ids:
        item = by_item[item_id]
        stage2_map = stage2_assignments.get(item_id, {})
        stage2_images = {
            image.image_id: image
            for image in selected_images_by_item.get(item_id, ())
        }
        included_ids = {
            image.image_id
            for image in item.images
            if image.image_id in stage1_image_ids
        } | set(stage2_map)
        image_lookup = {image.image_id: image for image in item.images}
        for image_id in sorted(included_ids):
            image = stage2_images.get(image_id, image_lookup[image_id])
            score = quality_scores.get(
                image_id, _metadata_quality(image.width, image.height)
            )
            rows.append(
                {
                    "image_id": image.image_id,
                    "item_id": item.item_id,
                    "relative_image_path": image.relative_path,
                    "product_type": item.product_type,
                    "is_main": int(image.is_main),
                    "original_width": image.width,
                    "original_height": image.height,
                    "quality_score": round(float(score), 8),
                    "stage1_train": int(image.image_id in stage1_image_ids),
                    "stage2_split": stage2_map.get(image.image_id, ""),
                }
            )
    rows.sort(key=lambda row: (row["item_id"], row["image_id"]))
    assert_no_image_leakage(rows)
    split_counts = Counter(
        row["stage2_split"] for row in rows if row["stage2_split"]
    )
    return rows, {
        "catalog_items_parsed": len(catalog_items),
        "stage1_items": len(stage1_items),
        "stage1_images": sum(int(row["stage1_train"]) for row in rows),
        "stage2_items": len(selected_stage2),
        "stage2_train_images": split_counts["train"],
        "gallery_images": split_counts["gallery"],
        "query_images": split_counts["query"],
        "stage2_split_counts": dict(split_counts),
        "product_type_counts_stage2": dict(
            Counter(item.product_type for item in selected_stage2)
        ),
        "fixed_gallery_image_ids_sha256": canonical_digest(
            sorted(
                row["image_id"] for row in rows if row["stage2_split"] == "gallery"
            )
        ),
        "fixed_query_image_ids_sha256": canonical_digest(
            sorted(
                row["image_id"] for row in rows if row["stage2_split"] == "query"
            )
        ),
    }


def split_stage2_images(
    item_id: str,
    image_ids: Sequence[str],
    seed: int,
    train_fraction: float = 0.6,
    gallery_fraction: float = 0.2,
    query_fraction: float = 0.2,
    preferred_gallery_ids: Sequence[str] = (),
) -> dict[str, str]:
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image_id within item {item_id}")
    if len(image_ids) < 4:
        raise ValueError("Stage-2 split requires at least four images")
    if abs(train_fraction + gallery_fraction + query_fraction - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to one")
    ordered = sorted(
        image_ids,
        key=lambda image_id: stable_u64(seed, "stage2-image", item_id, image_id),
    )
    total = len(ordered)
    gallery_count = max(1, int(total * gallery_fraction + 0.5))
    query_count = max(1, int(total * query_fraction + 0.5))
    while total - gallery_count - query_count < 2:
        if gallery_count >= query_count and gallery_count > 1:
            gallery_count -= 1
        elif query_count > 1:
            query_count -= 1
        else:
            raise RuntimeError(f"Cannot create a 2/1/1 split for item {item_id}")
    train_count = total - gallery_count - query_count
    preferred = [
        image_id
        for image_id in preferred_gallery_ids
        if image_id in set(ordered)
    ]
    gallery_ids = list(dict.fromkeys(preferred))[:gallery_count]
    gallery_ids.extend(
        image_id
        for image_id in ordered
        if image_id not in gallery_ids
    )
    gallery_ids = gallery_ids[:gallery_count]
    remaining = [image_id for image_id in ordered if image_id not in gallery_ids]
    # Keep query selection deterministic and disjoint from the catalog-main
    # Gallery preference. The remainder becomes Stage-2 training views.
    query_ids = remaining[-query_count:]
    train_ids = remaining[:train_count]
    assignments: dict[str, str] = {}
    for image_id in train_ids:
        assignments[image_id] = "train"
    for image_id in gallery_ids:
        assignments[image_id] = "gallery"
    for image_id in query_ids:
        assignments[image_id] = "query"
    return assignments


def assert_no_image_leakage(rows: Sequence[dict[str, Any]]) -> None:
    stage1 = {str(row["image_id"]) for row in rows if int(row["stage1_train"])}
    stage2_train = {
        str(row["image_id"]) for row in rows if row["stage2_split"] == "train"
    }
    gallery = {
        str(row["image_id"]) for row in rows if row["stage2_split"] == "gallery"
    }
    query = {str(row["image_id"]) for row in rows if row["stage2_split"] == "query"}
    if gallery & query:
        raise RuntimeError("Gallery and Query share image IDs")
    if (stage1 | stage2_train) & (gallery | query):
        overlap = sorted((stage1 | stage2_train) & (gallery | query))[:10]
        raise RuntimeError(
            f"Image-level leakage: held-out Gallery/Query images occur in training: {overlap}"
        )
    train_items = {
        str(row["item_id"]) for row in rows if row["stage2_split"] == "train"
    }
    gallery_items = {
        str(row["item_id"]) for row in rows if row["stage2_split"] == "gallery"
    }
    query_items = {
        str(row["item_id"]) for row in rows if row["stage2_split"] == "query"
    }
    if train_items != gallery_items or train_items != query_items:
        raise RuntimeError(
            "Stage-2 identity sets differ across train/gallery/query; every evaluation "
            "item must have training images and held-out gallery/query views"
        )


def _select_stage2_items(
    eligible: Sequence[CatalogItem], settings: Any
) -> tuple[list[CatalogItem], dict[str, float]]:
    by_type: dict[str, list[CatalogItem]] = defaultdict(list)
    for item in eligible:
        by_type[item.product_type].append(item)
    preferred = [value for value in settings.preferred_product_types if value in by_type]
    remaining_types = sorted(
        (value for value in by_type if value not in preferred),
        key=lambda value: (-len(by_type[value]), value),
    )
    type_order = (preferred + remaining_types)[: settings.stage2_product_type_count]
    if not type_order:
        raise RuntimeError("No product types remain for Stage-2 selection")
    candidate_limit = max(
        settings.stage2_item_count,
        settings.stage2_item_count * settings.quality_candidate_multiplier,
    )
    candidate_per_type = max(1, math.ceil(candidate_limit / len(type_order)))
    candidates: list[CatalogItem] = []
    for product_type in type_order:
        ranked = sorted(
            by_type[product_type],
            key=lambda item: (
                -_item_metadata_quality(item),
                stable_u64(settings.random_seed, "quality-candidate", item.item_id),
            ),
        )
        candidates.extend(ranked[:candidate_per_type])
    if len(candidates) < settings.stage2_item_count:
        existing_ids = {candidate.item_id for candidate in candidates}
        remaining = [
            item
            for item in eligible
            if item.item_id not in existing_ids
        ]
        remaining.sort(
            key=lambda item: (
                -_item_metadata_quality(item),
                stable_u64(settings.random_seed, "quality-fill", item.item_id),
            )
        )
        candidates.extend(remaining[: settings.stage2_item_count - len(candidates)])
    # When the configured dense types do not contain enough eligible products,
    # explicitly extend the type order rather than silently discarding fillers.
    for item in candidates:
        if item.product_type not in type_order:
            type_order.append(item.product_type)

    quality_scores: dict[str, float] = {}
    if settings.quality_scan_enabled:
        small_root = _small_image_root(settings.dataset_root)
        assert small_root is not None
        for index, item in enumerate(candidates, start=1):
            for image in item.images[: settings.stage2_max_images_per_item]:
                quality_scores[image.image_id] = _visual_quality(
                    small_root / image.relative_path,
                    image.width,
                    image.height,
                )
            if index % 2_000 == 0:
                print(f"[prepare_data] quality-scanned {index:,}/{len(candidates):,} items")
    else:
        for item in candidates:
            for image in item.images:
                quality_scores[image.image_id] = _metadata_quality(
                    image.width, image.height
                )

    ranked_by_type: dict[str, list[CatalogItem]] = defaultdict(list)
    for item in candidates:
        ranked_by_type[item.product_type].append(item)
    for product_type in ranked_by_type:
        ranked_by_type[product_type].sort(
            key=lambda item: (
                -np.mean(
                    sorted(
                        (
                            quality_scores.get(
                                image.image_id,
                                _metadata_quality(image.width, image.height),
                            )
                            for image in item.images
                        ),
                        reverse=True,
                    )[: settings.stage2_min_images_per_item]
                ),
                stable_u64(settings.random_seed, "quality-final", item.item_id),
            )
        )
    selected: list[CatalogItem] = []
    cursor = {product_type: 0 for product_type in type_order}
    while len(selected) < settings.stage2_item_count:
        changed = False
        for product_type in type_order:
            values = ranked_by_type.get(product_type, [])
            if cursor[product_type] < len(values):
                selected.append(values[cursor[product_type]])
                cursor[product_type] += 1
                changed = True
                if len(selected) == settings.stage2_item_count:
                    break
        if not changed:
            raise RuntimeError(
                f"Only {len(selected):,}/{settings.stage2_item_count:,} Stage-2 items "
                "could be selected from the configured product types"
            )
    return selected, quality_scores


def _rank_item_images(
    item: CatalogItem, scores: dict[str, float], settings: Any
) -> list[ImageRecord]:
    return sorted(
        item.images,
        key=lambda image: (
            -scores.get(
                image.image_id, _metadata_quality(image.width, image.height)
            ),
            not image.is_main,
            stable_u64(settings.random_seed, "item-image", item.item_id, image.image_id),
        ),
    )


def _allocate_stage1_images(
    items: Sequence[CatalogItem], held_out: set[str], settings: Any
) -> set[str]:
    pools: dict[str, list[ImageRecord]] = {}
    selected: set[str] = set()
    for item in items:
        available = sorted(
            (image for image in item.images if image.image_id not in held_out),
            key=lambda image: (
                not image.is_main,
                -_metadata_quality(image.width, image.height),
                stable_u64(settings.random_seed, "stage1-view", item.item_id, image.image_id),
            ),
        )[: settings.stage1_max_images_per_item]
        if len(available) < settings.stage1_min_images_per_item:
            raise RuntimeError(
                f"Item {item.item_id} has only {len(available)} Stage-1 views after "
                "holding out Query/Gallery"
            )
        pools[item.item_id] = available
        selected.update(
            image.image_id for image in available[: settings.stage1_min_images_per_item]
        )
    target = min(
        settings.stage1_target_image_count,
        sum(len(values) for values in pools.values()),
    )
    if target < len(selected):
        raise RuntimeError("Stage-1 image target is smaller than the per-item minimum")
    cursor = settings.stage1_min_images_per_item
    while len(selected) < target:
        changed = False
        for item in items:
            values = pools[item.item_id]
            if cursor < len(values):
                selected.add(values[cursor].image_id)
                changed = True
                if len(selected) == target:
                    break
        if not changed:
            break
        cursor += 1
    return selected


def _visual_quality(path: Path, width: int, height: int) -> float:
    """Deterministic catalog-quality proxy, not a learned semantic selector."""
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((128, 128), Image.Resampling.BILINEAR)
            array = np.asarray(rgb, dtype=np.float32)
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            edges = np.asarray(
                rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
            )
    except Exception:
        return _metadata_quality(width, height) - 10.0
    if array.size == 0:
        return _metadata_quality(width, height) - 10.0
    border = np.concatenate(
        [array[0], array[-1], array[:, 0], array[:, -1]], axis=0
    )
    border_uniformity = 1.0 / (1.0 + float(border.std()) / 32.0)
    sharpness = math.log1p(float(edges.var()))
    contrast = math.log1p(float(gray.std()))
    return (
        _metadata_quality(width, height)
        + 0.45 * sharpness
        + 0.20 * contrast
        + 0.35 * border_uniformity
    )


def _metadata_quality(width: int, height: int) -> float:
    return math.log1p(max(1, min(int(width), int(height))))


def _item_metadata_quality(item: CatalogItem) -> float:
    values = sorted(
        (_metadata_quality(image.width, image.height) for image in item.images),
        reverse=True,
    )
    return float(np.mean(values[: min(4, len(values))]))


def _bundle_from_manifest(settings: Any, metadata: dict[str, Any]) -> ABOBundle:
    rows: list[dict[str, Any]] = []
    with settings.manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "is_main": int(row["is_main"]),
                    "original_width": int(row["original_width"]),
                    "original_height": int(row["original_height"]),
                    "quality_score": float(row["quality_score"]),
                    "stage1_train": int(row["stage1_train"]),
                }
            )
    if rows_digest(rows) != metadata.get("manifest_sha256"):
        raise RuntimeError(
            "ABO manifest SHA256 mismatch. Refusing to train on a modified split."
        )
    assert_no_image_leakage(rows)
    return _bundle_from_rows(settings, rows, metadata)


def _bundle_from_rows(
    settings: Any, rows: Sequence[dict[str, Any]], metadata: dict[str, Any]
) -> ABOBundle:
    small_root = _small_image_root(settings.dataset_root)
    if small_root is None:
        raise FileNotFoundError(_missing_data_message(settings.dataset_root))
    stage1_ids = sorted(
        {str(row["item_id"]) for row in rows if int(row["stage1_train"])}
    )
    stage2_ids = sorted(
        {str(row["item_id"]) for row in rows if row["stage2_split"] == "train"}
    )
    stage1_index = {item_id: index for index, item_id in enumerate(stage1_ids)}
    stage2_index = {item_id: index for index, item_id in enumerate(stage2_ids)}

    def make(row: dict[str, Any], split: str, index: int) -> ABOSample:
        path = small_root / str(row["relative_image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        return ABOSample(
            image_id=str(row["image_id"]),
            item_id=str(row["item_id"]),
            item_index=index,
            image_path=path,
            product_type=str(row["product_type"]),
            split=split,
            quality_score=float(row["quality_score"]),
        )

    stage1 = [
        make(row, "stage1_train", stage1_index[str(row["item_id"])])
        for row in rows
        if int(row["stage1_train"])
    ]
    stage2 = [
        make(row, "stage2_train", stage2_index[str(row["item_id"])])
        for row in rows
        if row["stage2_split"] == "train"
    ]
    gallery = [
        make(row, "gallery", stage2_index[str(row["item_id"])])
        for row in rows
        if row["stage2_split"] == "gallery"
    ]
    query = [
        make(row, "query", stage2_index[str(row["item_id"])])
        for row in rows
        if row["stage2_split"] == "query"
    ]
    return ABOBundle(
        stage1_train=ABORetrievalDataset(stage1),
        stage2_train=ABORetrievalDataset(stage2),
        gallery=ABORetrievalDataset(gallery),
        query=ABORetrievalDataset(query),
        stage1_item_ids=tuple(stage1_ids),
        stage2_item_ids=tuple(stage2_ids),
        manifest_digest=str(metadata["manifest_sha256"]),
        manifest_path=settings.manifest_csv,
        metadata=metadata,
    )


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _download_resumable(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    print(
        f"[download] {url} -> {destination} "
        f"(resume offset={offset / (1024 ** 2):.1f} MiB)"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        partial = getattr(response, "status", None) == 206
        mode = "ab" if offset and partial else "wb"
        if mode == "wb":
            offset = 0
        downloaded = offset
        next_report = downloaded + 256 * 1024 * 1024
        with part.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"[download] received {downloaded / (1024 ** 3):.2f} GiB")
                    next_report += 256 * 1024 * 1024
    part.replace(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    print(f"[extract] {archive} -> {destination}")
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            target = (destination / member.name).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in ABO archive: {member.name}")
            handle.extract(member, destination)


def _listing_files(root: Path) -> list[Path]:
    candidates = (
        root / "listings" / "metadata",
        root / "abo-listings" / "listings" / "metadata",
    )
    for directory in candidates:
        values = sorted(directory.glob("listings_*.json.gz"))
        if values:
            return values
    return []


def _image_metadata(root: Path) -> Path | None:
    candidates = (
        root / "images" / "metadata" / "images.csv.gz",
        root / "abo-images-small" / "images" / "metadata" / "images.csv.gz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _small_image_root(root: Path) -> Path | None:
    candidates = (
        root / "images" / "small",
        root / "abo-images-small" / "images" / "small",
    )
    return next((path for path in candidates if path.is_dir()), None)


def _official_layout_exists(root: Path) -> bool:
    return bool(_listing_files(root)) and _image_metadata(root) is not None and _small_image_root(root) is not None


def _missing_data_message(root: Path) -> str:
    return (
        f"ABO data is incomplete under {root}. Expected official paths "
        "listings/metadata/listings_*.json.gz, images/metadata/images.csv.gz, "
        "and images/small/<relative path>. Enable dataset.download to fetch the "
        "83 MiB listings archive and 3 GiB 256-pixel image archive."
    )
