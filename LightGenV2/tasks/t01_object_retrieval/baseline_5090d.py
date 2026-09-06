"""Frozen Qwen3-VL-Embedding-2B Caltech101 baseline on RTX 5090 D."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from LightGenV2.common.baseline_measurement import (
    FirstBlockTimer,
    NvidiaSmiPowerSampler,
    environment_report,
    power_report,
    save_power_samples,
    summarize,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    TeacherEmbeddingStore,
    build_teacher_embedding_cache,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    preprocess_images,
    teacher_embeddings,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalDataset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import (
    evaluate_embeddings,
)

from .modeling import load_backbone
from .settings import load_settings


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _configure(args: argparse.Namespace) -> Any:
    settings = load_settings(TASK_DIR / "configs" / "qwen_frozen_embedding.yaml")
    settings.dataset_root = args.data_root.expanduser().resolve()
    settings.download = False
    settings.model_id = str(args.model.expanduser().resolve())
    settings.local_files_only = True
    settings.cache_dir = None
    settings.output_dir = args.run_dir.expanduser().resolve()
    settings.shared_teacher_cache_dir = settings.output_dir / "teacher_cache"
    settings.teacher_batch_size = int(args.embedding_batch_size)
    settings.inference_batch_size = 1
    settings.num_workers = int(args.num_workers)
    return settings


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This formal baseline requires NVIDIA GeForce RTX 5090 D")
    settings = _configure(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = prepare_caltech101_subset(settings, persist=True)
    loaded = load_backbone(settings, torch.device("cuda:0"))
    build_teacher_embedding_cache(loaded, bundle, settings, force=False)
    store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    query_embeddings = store.lookup(bundle.test_samples)
    gallery_embeddings = store.lookup(bundle.gallery_samples)
    evaluation = evaluate_embeddings(
        query_embeddings,
        bundle.test_samples,
        gallery_embeddings,
        bundle.gallery_samples,
        bundle.class_names,
        settings.gallery_aggregation,
        system_name="frozen_qwen3_vl_embedding_2b_5090d",
    )

    gallery = F.normalize(gallery_embeddings.float(), dim=-1)
    gallery_labels = torch.tensor([sample.sku_index for sample in bundle.gallery_samples])
    prototypes = []
    for class_index in range(len(bundle.class_names)):
        prototypes.append(F.normalize(gallery[gallery_labels.eq(class_index)].mean(0), dim=0))
    prototypes_gpu = torch.stack(prototypes).to(loaded.device)
    dataset = GroceryRetrievalDataset(bundle.test_samples, settings.image_size, augment=False)
    timer = FirstBlockTimer(loaded.model.model.visual.blocks[0])
    sampler = NvidiaSmiPowerSampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    measurements: list[dict[str, Any]] = []
    try:
        for index in range(min(args.timing_samples, len(dataset))):
            item = dataset[index]
            inputs = preprocess_images(
                loaded.processor, [item["image"]], settings.instruction
            )
            validate_token_budgets(inputs, settings)
            inputs = move_inputs(inputs, loaded.device)
            torch.cuda.synchronize()
            timer.reset()
            sampler.set_phase(f"active:{index}")
            try:
                embedding = teacher_embeddings(
                    loaded.model, inputs, settings.embedding_dim
                )
                scores = embedding @ prototypes_gpu.T
                _ranking = scores.argsort(dim=-1, descending=True)
                timing = timer.finish()
            finally:
                sampler.set_phase(None)
            measurements.append(
                {
                    "sample_index": index,
                    "sample_id": item["sample"].sample_id,
                    "cuda_ms": timing["cuda_ms"],
                    "host_ms": timing["host_ms"],
                    "first_block_input_shape": json.dumps(
                        timing["first_block_input_shape"]
                    ),
                }
            )
    finally:
        timer.close()
        power_samples = sampler.stop()

    latencies = [row["cuda_ms"] for row in measurements]
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "Caltech101 image retrieval",
        "model": settings.model_id,
        "qwen_frozen": True,
        "trainable_parameters": 0,
        "test_samples": len(bundle.test_samples),
        "timing_samples": len(measurements),
        "explicit_warmup_forwards": 0,
        "first_test_sample_included": True,
        "timing_boundary": (
            "input to native Vision Transformer block 0 through all native Vision/"
            "Language blocks, 64-D Matryoshka normalization, fixed-gallery similarity "
            "and Top-K ranking"
        ),
        "performance": evaluation.metrics,
        "latency_cuda_ms": summarize(latencies),
        "latency_host_ms": summarize([row["host_ms"] for row in measurements]),
        "power": power_report(power_samples, latencies),
        "split_manifest_sha256": bundle.manifest_digest,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(),
        "model_load_seconds": loaded.load_time_sec,
    }
    write_json(settings.output_dir / "baseline_report.json", report)
    _write_rows(settings.output_dir / "timing_per_sample.csv", measurements)
    _write_rows(settings.output_dir / "retrieval_results.csv", evaluation.rows)
    save_power_samples(settings.output_dir / "power_samples.csv", power_samples)
    (settings.output_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timing-samples", type=int, default=200)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
