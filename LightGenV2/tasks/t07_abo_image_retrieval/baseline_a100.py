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

from LightGenV2.common.baseline_measurement import (
    FirstBlockTimer,
    NvidiaSmiPowerSampler,
    environment_report,
    gpu_power_limit_w,
    power_report,
    save_power_samples,
    sha256_file,
    summarize,
    validate_cuda_device,
    write_json,
)
from LightGenV2.tasks.t01_object_retrieval.modeling import load_backbone
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


def _plot(
    run_dir: Path,
    report: dict[str, Any],
    per_category: Sequence[dict[str, Any]],
    power_samples: Sequence[Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    performance = report["performance"]
    names = ["Route", "P@1", "P@5", "P@10", "mAP@10", "NDCG@10"]
    values = [
        performance["category_route_accuracy"],
        performance["precision_at_1"],
        performance["precision_at_5"],
        performance["precision_at_10"],
        performance["map_at_10"],
        performance["ndcg_at_10"],
    ]
    axes[0].bar(names, values, color="#1778b5")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("a  Similar-product retrieval", loc="left", fontweight="bold")
    axes[0].set_ylabel("score")
    axes[1].bar(
        [row["category_name"] for row in per_category],
        [row["precision_at_1"] for row in per_category],
        color="#3a923a",
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_title("b  Per-category P@1", loc="left", fontweight="bold")
    active = [sample for sample in power_samples if sample.phase.startswith("active:")]
    if active:
        origin = active[0].host_monotonic_s
        axes[2].plot(
            [sample.host_monotonic_s - origin for sample in active],
            [sample.watts for sample in active],
            lw=0.7,
            color="#c44e52",
        )
    axes[2].set_xlabel("active-window time (s)")
    axes[2].set_ylabel("GPU board power (W)")
    axes[2].set_title("c  A100 power", loc="left", fontweight="bold")
    figure.savefig(run_dir / "baseline_overview.png", dpi=220)
    figure.savefig(run_dir / "baseline_overview.pdf")
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    gpu_name = validate_cuda_device(args.expected_gpu)
    rated_power_w = gpu_power_limit_w()
    if args.warmup_forwards < 0:
        raise ValueError("--warmup-forwards cannot be negative")
    data_root = args.data_root.expanduser().resolve()
    archive_path = args.dataset_archive.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive is missing: {archive_path}")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Dataset archive SHA256 mismatch: {archive_sha256} != {EXPECTED_ARCHIVE_SHA256}"
        )
    run_dir = args.run_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    all_samples, category_names = _load_contract(data_root)
    train_samples = [sample for sample in all_samples if sample.split == "train"]
    test_samples = [sample for sample in all_samples if sample.split == "test"]
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        test_samples = test_samples[: args.max_queries]

    settings = SimpleNamespace(
        model_id=str(model_path),
        cache_dir=None,
        local_files_only=True,
        processor_min_pixels=IMAGE_PIXELS,
        processor_max_pixels=IMAGE_PIXELS,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    loaded = load_backbone(settings, torch.device("cuda:0"))
    parameter_count = sum(parameter.numel() for parameter in loaded.model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in loaded.model.parameters()
        if parameter.requires_grad
    )
    if trainable_count != 0 or loaded.model.training:
        raise RuntimeError("Frozen-Qwen baseline must have zero trainable parameters")

    gallery_views: list[torch.Tensor] = []
    for index, sample in enumerate(train_samples):
        gallery_views.append(_embed_image(loaded, sample).detach().cpu().to(torch.float16))
        if (index + 1) % 120 == 0:
            print(f"[gallery] {index + 1}/{len(train_samples)}", flush=True)
    gallery_cpu, gallery_metadata = _gallery_centroids(train_samples, gallery_views)
    prototypes_cpu = _category_prototypes(gallery_cpu, gallery_metadata)
    gallery = gallery_cpu.to(loaded.device)
    prototypes = prototypes_cpu.to(loaded.device)

    query_vectors: list[torch.Tensor] = []
    for index, sample in enumerate(test_samples):
        query_vectors.append(_embed_image(loaded, sample).detach().cpu().to(torch.float16))
        if (index + 1) % 120 == 0:
            print(f"[performance] {index + 1}/{len(test_samples)}", flush=True)
    query_embeddings = torch.stack(query_vectors)
    performance, predictions, per_category = _evaluate(
        query_embeddings,
        test_samples,
        gallery_cpu,
        gallery_metadata,
        prototypes_cpu,
    )
    timing_queries = _balanced_timing_queries(test_samples, int(args.timing_samples))
    first_block = loaded.model.model.visual.blocks[0]
    sampler = NvidiaSmiPowerSampler(interval_ms=POWER_SAMPLE_INTERVAL_MS)
    active_phase: dict[str, str | None] = {"value": None}
    power_hook = first_block.register_forward_pre_hook(
        lambda _module, _hook_args: sampler.set_phase(active_phase["value"])
    )
    measurements: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    time.sleep(float(args.cooldown_seconds))
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(float(args.idle_seconds))
    sampler.set_phase(None)
    timer: FirstBlockTimer | None = None
    try:
        for index in range(int(args.warmup_forwards)):
            sample = timing_queries[index % len(timing_queries)]
            embedding = _embed_image(loaded, sample)
            (embedding.float() @ gallery.float().T).argsort(descending=True)
            (embedding.float() @ prototypes.float().T).argmax()
        torch.cuda.synchronize()
        timer = FirstBlockTimer(first_block)
        for index, sample in enumerate(timing_queries):
            with Image.open(sample.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = move_inputs(_inputs(loaded.processor, image), loaded.device)
            timer.reset()
            active_phase["value"] = f"active:{index}"
            try:
                embedding = teacher_embeddings(loaded.model, inputs, EMBEDDING_DIM)[0]
                order = (embedding.float() @ gallery.float().T).argsort(descending=True)
                route = (embedding.float() @ prototypes.float().T).argmax()
                timing = timer.finish()
            finally:
                sampler.set_phase(None)
                active_phase["value"] = None
            measurements.append(
                {
                    "sample_index": index,
                    "sample_id": sample.sample_id,
                    "category_id": sample.category_id,
                    "cuda_ms": timing["cuda_ms"],
                    "host_ms": timing["host_ms"],
                    "first_block_calls": timing["first_block_calls"],
                    "first_block_input_shape": json.dumps(timing["first_block_input_shape"]),
                    "top_gallery_index": int(order[0]),
                    "route_prediction": int(route),
                }
            )
            if (index + 1) % 50 == 0:
                print(f"[timing] {index + 1}/{len(timing_queries)}", flush=True)
    finally:
        power_hook.remove()
        if timer is not None:
            timer.close()
        power_samples = sampler.stop()

    latency_cuda = [float(row["cuda_ms"]) for row in measurements]
    measured_power = power_report(
        power_samples, latency_cuda, power_limit_w=rated_power_w
    )
    measured_power["sampling_interval_ms"] = POWER_SAMPLE_INTERVAL_MS
    measured_power["pre_idle_cooldown_seconds"] = float(args.cooldown_seconds)
    manifest_path = data_root / "data" / "abo_similarity10_manifest.csv"
    report_path = data_root / "data" / "abo_similarity10_manifest.report.json"
    model_index = model_path / "model.safetensors.index.json"
    report = {
        "schema_version": 1,
        "status": "complete" if args.max_queries is None else "diagnostic_subset",
        "task_id": "T07",
        "task": "ABO similarity-10 image-to-similar-product retrieval",
        "retrieval_direction": "held-out_image_query_to_train_product_centroids",
        "relevance": "shared category; exactly 12 relevant gallery products per query",
        "model": str(model_path),
        "model_family": "Qwen3-VL-Embedding-2B",
        "qwen_frozen": True,
        "optimization": "none; no fine-tuning and no trained readout head",
        "model_parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "batch_size": 1,
        "embedding_dim": EMBEDDING_DIM,
        "image_size": [224, 224],
        "instruction": INSTRUCTION,
        "dataset": {
            "images": len(all_samples),
            "categories": category_names,
            "split_images": EXPECTED_SPLIT_IMAGES,
            "split_products": EXPECTED_SPLIT_PRODUCTS,
            "views_per_product": EXPECTED_VIEWS_PER_PRODUCT,
            "gallery_products": len(gallery_metadata),
            "test_queries": len(test_samples),
        },
        "gallery_contract": (
            "Each of 120 train products is represented by the L2-normalized mean "
            "of its 12 frozen-Qwen view embeddings; gallery construction is offline"
        ),
        "performance": performance,
        "per_category": per_category,
        "timing_samples": len(timing_queries),
        "timing_subset": "deterministic round-robin by category without replacement",
        "explicit_warmup_forwards": int(args.warmup_forwards),
        "first_measured_sample_included_in_statistics": True,
        "timing_boundary": (
            "native Vision Transformer block 0 input through all native Vision/"
            "Language blocks, valid-token 2048-D normalization, cosine against 120 "
            "precomputed gallery centroids and 10 category prototypes, and full ranking"
        ),
        "excluded_from_online_timing": (
            "file I/O, image decode, processor/tokenizer, patch embedding, gallery/"
            "prototype precomputation, and model loading"
        ),
        "latency_cuda_ms": summarize(latency_cuda),
        "latency_host_ms": summarize([float(row["host_ms"]) for row in measurements]),
        "power": measured_power,
        "dataset_sha256": {
            "archive": archive_sha256,
            "manifest_csv": sha256_file(manifest_path),
            "manifest_report_json": sha256_file(report_path),
        },
        "model_index_sha256": sha256_file(model_index) if model_index.is_file() else None,
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(power_limit_w=rated_power_w),
        "hardware_contract": {
            "expected_gpu_name_substring": args.expected_gpu,
            "actual_gpu_name": gpu_name,
        },
        "model_load_seconds": loaded.load_time_sec,
    }
    write_json(run_dir / "baseline_report.json", report)
    _write_rows(run_dir / "retrieval_predictions.csv", predictions)
    _write_rows(run_dir / "timing_per_sample.csv", measurements)
    _write_rows(run_dir / "per_category_metrics.csv", per_category)
    save_power_samples(run_dir / "power_samples.csv", power_samples)
    torch.save(
        {
            "embeddings": gallery_cpu.to(torch.float16),
            "metadata": [item.__dict__ for item in gallery_metadata],
            "category_prototypes": prototypes_cpu.to(torch.float16),
            "instruction": INSTRUCTION,
            "manifest_sha256": sha256_file(manifest_path),
        },
        run_dir / "gallery_index.pt",
    )
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    _plot(run_dir, report, per_category, power_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--warmup-forwards", type=int, default=50)
    parser.add_argument("--timing-samples", type=int, default=200)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--expected-gpu", default="NVIDIA A100-PCIE-40GB")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
