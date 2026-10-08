"""Frozen Qwen3-VL-Embedding-2B baseline for ABO similarity-10 retrieval.

The online query is one held-out product image.  It ranks 120 precomputed
gallery-product centroids, each built from the 12 training views of one product.
Product identities are disjoint across train/validation/test; relevance is the
shared category, so each query has exactly 12 relevant gallery products.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

import hashlib


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    IGNORED_MODEL_INPUTS,
    move_inputs,
    teacher_embeddings,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
INSTRUCTION = (
    "Represent this catalog product image for category-aware visual similarity "
    "retrieval."
)
IMAGE_PIXELS = 224 * 224
EMBEDDING_DIM = 2048
POWER_SAMPLE_INTERVAL_MS = 10
EXPECTED_SPLIT_IMAGES = {"train": 1440, "val": 480, "test": 480}
EXPECTED_SPLIT_PRODUCTS = {"train": 120, "val": 40, "test": 40}
EXPECTED_PRODUCTS_PER_CATEGORY = {"train": 12, "val": 4, "test": 4}
EXPECTED_VIEWS_PER_PRODUCT = 12
EXPECTED_CATEGORIES = 10
EXPECTED_ARCHIVE_SHA256 = "c8f0f79cbd8ceb3c42162092136254a1f44f6d13b3329e0e3b1c5dfc458c1485"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    product_id: str
    category_id: int
    category_name: str
    split: str
    image_path: Path


@dataclass(frozen=True)
class GalleryItem:
    product_id: str
    category_id: int
    category_name: str
    view_count: int


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_contract(data_root: Path) -> tuple[list[Sample], dict[int, str]]:
    data_root = data_root.resolve()
    manifest = data_root / "data" / "abo_similarity10_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"ABO similarity-10 manifest is missing: {manifest}")
    samples: list[Sample] = []
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_path = (data_root / row["image_path"]).resolve()
            if not image_path.is_file() or not image_path.is_relative_to(data_root):
                raise FileNotFoundError(f"Invalid package-relative image: {image_path}")
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    product_id=row["product_id"],
                    category_id=int(row["category_id"]),
                    category_name=row["class_name"],
                    split=row["split"],
                    image_path=image_path,
                )
            )
    if len(samples) != sum(EXPECTED_SPLIT_IMAGES.values()):
        raise RuntimeError(f"Expected 2400 manifest rows, found {len(samples)}")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("ABO sample_id values are not unique")
    split_images = Counter(sample.split for sample in samples)
    if dict(split_images) != EXPECTED_SPLIT_IMAGES:
        raise RuntimeError(f"Unexpected split image counts: {dict(split_images)}")

    category_names: dict[int, str] = {}
    split_products: dict[str, set[str]] = defaultdict(set)
    product_rows: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        previous = category_names.setdefault(sample.category_id, sample.category_name)
        if previous != sample.category_name:
            raise RuntimeError("A category ID maps to multiple names")
        split_products[sample.split].add(sample.product_id)
        product_rows[sample.product_id].append(sample)
    if sorted(category_names) != list(range(EXPECTED_CATEGORIES)):
        raise RuntimeError(f"Expected category IDs 0..9, found {sorted(category_names)}")
    if {key: len(value) for key, value in split_products.items()} != EXPECTED_SPLIT_PRODUCTS:
        raise RuntimeError("Unexpected split product counts")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_products[left] & split_products[right]:
            raise RuntimeError(f"Product identity leakage between {left} and {right}")
    for product_id, rows in product_rows.items():
        if len(rows) != EXPECTED_VIEWS_PER_PRODUCT:
            raise RuntimeError(f"Product {product_id} has {len(rows)} views")
        if len({row.split for row in rows}) != 1 or len({row.category_id for row in rows}) != 1:
            raise RuntimeError(f"Product {product_id} has inconsistent metadata")
    for split, expected in EXPECTED_PRODUCTS_PER_CATEGORY.items():
        counts = Counter(
            product_rows[product_id][0].category_id for product_id in split_products[split]
        )
        if set(counts) != set(range(EXPECTED_CATEGORIES)) or set(counts.values()) != {expected}:
            raise RuntimeError(f"Unexpected per-category product counts in {split}: {counts}")
    return samples, category_names


def _template(processor: Any, image: Image.Image) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _inputs(processor: Any, image: Image.Image) -> dict[str, torch.Tensor]:
    values = processor(
        text=[_template(processor, image)],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {
        name: value
        for name, value in values.items()
        if torch.is_tensor(value) and name not in IGNORED_MODEL_INPUTS
    }


@torch.inference_mode()
def _embed_image(loaded: Any, sample: Sample) -> torch.Tensor:
    with Image.open(sample.image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    inputs = move_inputs(_inputs(loaded.processor, image), loaded.device)
    return teacher_embeddings(loaded.model, inputs, EMBEDDING_DIM)[0]


def _gallery_centroids(
    train_samples: Sequence[Sample], embeddings: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, list[GalleryItem]]:
    if len(train_samples) != len(embeddings):
        raise ValueError("Sample and embedding counts differ")
    grouped: dict[str, list[tuple[Sample, torch.Tensor]]] = defaultdict(list)
    for sample, embedding in zip(train_samples, embeddings, strict=True):
        grouped[sample.product_id].append((sample, embedding.float()))
    vectors: list[torch.Tensor] = []
    metadata: list[GalleryItem] = []
    for product_id in sorted(grouped):
        records = grouped[product_id]
        if len(records) != EXPECTED_VIEWS_PER_PRODUCT:
            raise RuntimeError(f"Gallery product {product_id} does not have 12 views")
        first = records[0][0]
        if any(record[0].category_id != first.category_id for record in records):
            raise RuntimeError(f"Gallery product {product_id} spans categories")
        vectors.append(F.normalize(torch.stack([record[1] for record in records]).mean(0), dim=0))
        metadata.append(
            GalleryItem(product_id, first.category_id, first.category_name, len(records))
        )
    gallery = F.normalize(torch.stack(vectors), dim=-1)
    return gallery, metadata


def _category_prototypes(
    gallery: torch.Tensor, metadata: Sequence[GalleryItem]
) -> torch.Tensor:
    labels = torch.tensor([item.category_id for item in metadata], dtype=torch.long)
    prototypes = [
        F.normalize(gallery[labels == category].mean(0), dim=0)
        for category in range(EXPECTED_CATEGORIES)
    ]
    return F.normalize(torch.stack(prototypes), dim=-1)


def _ranking_metrics(relevant: np.ndarray) -> dict[str, float | int]:
    if relevant.ndim != 2 or relevant.shape[0] == 0:
        raise ValueError("Relevance must be a non-empty [queries, candidates] matrix")
    values = relevant.astype(np.float64, copy=False)
    total_relevant = np.maximum(values.sum(axis=1), 1.0)
    result: dict[str, float | int] = {}
    for cutoff in (1, 5, 10):
        used = min(cutoff, values.shape[1])
        hits = values[:, :used].sum(axis=1)
        result[f"precision_at_{cutoff}"] = float(np.mean(hits / used))
        result[f"positive_recall_at_{cutoff}"] = float(
            np.mean(hits / total_relevant)
        )
        result[f"hit_at_{cutoff}"] = float(np.mean(hits > 0))
        result[f"r_at_{cutoff}"] = result[f"hit_at_{cutoff}"]
    used = min(10, values.shape[1])
    positions = np.arange(1, used + 1, dtype=np.float64)
    top = values[:, :used]
    precision = np.cumsum(top, axis=1) / positions[None, :]
    ap = np.sum(precision * top, axis=1) / np.minimum(total_relevant, used)
    discounts = 1.0 / np.log2(positions + 1.0)
    dcg = np.sum(top * discounts[None, :], axis=1)
    ideal = np.asarray(
        [discounts[: int(min(count, used))].sum() for count in total_relevant]
    )
    result["map_at_10"] = float(np.mean(ap))
    result["ndcg_at_10"] = float(np.mean(dcg / np.maximum(ideal, 1.0e-12)))
    result["query_count"] = int(values.shape[0])
    result["candidate_count"] = int(values.shape[1])
    result["relevant_candidates_per_query"] = float(np.mean(total_relevant))
    return result


def _balanced_timing_queries(samples: Sequence[Sample], count: int) -> list[Sample]:
    if count <= 0:
        raise ValueError("--timing-samples must be positive")
    by_category: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_category[sample.category_id].append(sample)
    selected: list[Sample] = []
    offset = 0
    target = min(count, len(samples))
    while len(selected) < target:
        added = False
        for category in sorted(by_category):
            candidates = by_category[category]
            if offset < len(candidates):
                selected.append(candidates[offset])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        offset += 1
    return selected


def _evaluate(
    query_embeddings: torch.Tensor,
    query_samples: Sequence[Sample],
    gallery: torch.Tensor,
    gallery_metadata: Sequence[GalleryItem],
    prototypes: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scores = (F.normalize(query_embeddings.float(), dim=-1) @ gallery.float().T).numpy()
    order = np.argsort(-scores, axis=1)
    gallery_categories = np.asarray([item.category_id for item in gallery_metadata])
    query_categories = np.asarray([sample.category_id for sample in query_samples])
    relevant = gallery_categories[order] == query_categories[:, None]
    metrics: dict[str, Any] = _ranking_metrics(relevant)
    route_scores = (F.normalize(query_embeddings.float(), dim=-1) @ prototypes.float().T).numpy()
    route_prediction = np.argmax(route_scores, axis=1)
    metrics["category_route_accuracy"] = float(np.mean(route_prediction == query_categories))
    metrics["unrestricted_top1_category_accuracy"] = metrics["precision_at_1"]

    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(query_samples):
        top10 = order[index, :10].tolist()
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "query_product_id": sample.product_id,
                "query_category_id": sample.category_id,
                "query_category_name": sample.category_name,
                "route_prediction_id": int(route_prediction[index]),
                "route_correct": bool(route_prediction[index] == sample.category_id),
                "top1_relevant": bool(relevant[index, 0]),
                "top10_product_ids": json.dumps(
                    [gallery_metadata[value].product_id for value in top10]
                ),
                "top10_category_ids": json.dumps(
                    [gallery_metadata[value].category_id for value in top10]
                ),
                "top10_scores": json.dumps([float(scores[index, value]) for value in top10]),
            }
        )
    per_category: list[dict[str, Any]] = []
    for category in range(EXPECTED_CATEGORIES):
        mask = query_categories == category
        local = relevant[mask]
        row = _ranking_metrics(local)
        row.update(
            {
                "category_id": category,
                "category_name": query_samples[int(np.flatnonzero(mask)[0])].category_name,
                "category_route_accuracy": float(
                    np.mean(route_prediction[mask] == query_categories[mask])
                ),
            }
        )
        per_category.append(row)
    return metrics, predictions, per_category
