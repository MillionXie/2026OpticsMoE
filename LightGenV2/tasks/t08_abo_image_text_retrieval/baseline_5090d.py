"""Frozen Qwen3-VL-Embedding-2B ABO image-to-title baseline on RTX 5090 D.

The model is never fine-tuned.  All image and title embeddings are produced by
the original frozen checkpoint with batch size one.  Online latency starts at
the first native Vision Transformer block and ends after 2048-D normalization,
100-title cosine similarity, and ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from LightGenV2.common.baseline_measurement import (
    FirstBlockTimer,
    NvidiaSmiPowerSampler,
    environment_report,
    power_report,
    save_power_samples,
    sha256_file,
    summarize,
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
QUERY_INSTRUCTION = (
    "Retrieve the product title that best describes this product image."
)
DOCUMENT_INSTRUCTION = "Represent the user's input."
IMAGE_PIXELS = 224 * 224
EMBEDDING_DIM = 2048
POWER_SAMPLE_INTERVAL_MS = 10


@dataclass(frozen=True)
class Query:
    sample_id: str
    product_id: str
    label: int
    image_path: Path


@dataclass(frozen=True)
class Title:
    label: int
    product_id: str
    text: str


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


def _load_contract(data_root: Path, max_queries: int | None) -> tuple[list[Query], list[Title]]:
    data_root = data_root.resolve()
    with (data_root / "titles.csv").open(encoding="utf-8", newline="") as stream:
        titles = [
            Title(int(row["label"]), row["product_id"], row["title"])
            for row in csv.DictReader(stream)
        ]
    if len(titles) != 100 or [row.label for row in titles] != list(range(100)):
        raise RuntimeError("ABO easy100 requires exactly 100 titles labelled 0..99")
    if len({row.product_id for row in titles}) != len(titles):
        raise RuntimeError("ABO title candidate product IDs are not unique")
    label_to_product = {row.label: row.product_id for row in titles}
    with (data_root / "test.csv").open(encoding="utf-8", newline="") as stream:
        queries = []
        for row in csv.DictReader(stream):
            label = int(row["label"])
            image_path = (data_root / row["image_path"]).resolve()
            if label_to_product.get(label) != row["product_id"]:
                raise RuntimeError("ABO label/product mapping differs between test and titles")
            if not image_path.is_file() or not image_path.is_relative_to(data_root):
                raise FileNotFoundError(f"Invalid ABO image path: {image_path}")
            queries.append(Query(row["sample_id"], row["product_id"], label, image_path))
    if len(queries) != 2400:
        raise RuntimeError(f"ABO easy100 requires 2400 test images, found {len(queries)}")
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        queries = queries[:max_queries]
    return queries, titles


def _template(processor: Any, *, instruction: str, image: Image.Image | None = None,
              text: str | None = None) -> str:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:
        content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _inputs(processor: Any, *, image: Image.Image | None = None,
            text: str | None = None, instruction: str) -> dict[str, torch.Tensor]:
    prompt = _template(processor, instruction=instruction, image=image, text=text)
    arguments: dict[str, Any] = {
        "text": [prompt],
        "padding": True,
        "return_tensors": "pt",
    }
    if image is not None:
        arguments["images"] = [image]
    values = processor(**arguments)
    return {
        name: value
        for name, value in values.items()
        if torch.is_tensor(value) and name not in IGNORED_MODEL_INPUTS
    }


@torch.inference_mode()
def _embed_title(loaded: Any, title: str) -> torch.Tensor:
    inputs = _inputs(
        loaded.processor,
        text=title,
        instruction=DOCUMENT_INSTRUCTION,
    )
    return teacher_embeddings(
        loaded.model, move_inputs(inputs, loaded.device), EMBEDDING_DIM
    )[0]


def _metrics(ranks: np.ndarray) -> dict[str, float]:
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def _plot(run_dir: Path, report: dict[str, Any], ranks: np.ndarray,
          power_samples: list[Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    performance = report["performance"]
    names = ["R@1", "R@5", "R@10", "MRR"]
    values = [performance[key] for key in
              ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")]
    axes[0].bar(names, values, color="#1778b5")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("score")
    axes[0].set_title("a  Image-to-title retrieval", loc="left", fontweight="bold")
    axes[1].hist(ranks, bins=np.arange(0.5, 101.5, 1), color="#3a923a")
    axes[1].set_xlabel("true-title rank")
    axes[1].set_ylabel("test images")
    axes[1].set_title("b  Rank distribution", loc="left", fontweight="bold")
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
    axes[2].set_title("c  RTX 5090 D power", loc="left", fontweight="bold")
    figure.savefig(run_dir / "baseline_overview.png", dpi=220)
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Formal speed/power measurement requires RTX 5090 D")
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    queries, titles = _load_contract(data_root, args.max_queries)
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
        parameter.numel() for parameter in loaded.model.parameters()
        if parameter.requires_grad
    )
    if trainable_count != 0 or loaded.model.training:
        raise RuntimeError("Frozen-Qwen baseline must have zero trainable parameters in eval mode")

    title_embeddings = torch.stack(
        [_embed_title(loaded, title.text) for title in titles]
    )
    torch.save(
        {
            "embeddings": title_embeddings.detach().cpu().to(torch.float16),
            "labels": [title.label for title in titles],
            "product_ids": [title.product_id for title in titles],
            "titles": [title.text for title in titles],
            "instruction": DOCUMENT_INSTRUCTION,
        },
        run_dir / "title_embeddings.pt",
    )

    first_block = loaded.model.model.visual.blocks[0]
    timer = FirstBlockTimer(first_block)
    # The online window is about 25--35 ms/image on a 5090 D, so 20 Hz would
    # undersample individual forwards.  Request 100 Hz and retain every raw
    # sample; nvidia-smi may still be limited by the board sensor update rate.
    sampler = NvidiaSmiPowerSampler(interval_ms=POWER_SAMPLE_INTERVAL_MS)
    active_phase: dict[str, str | None] = {"value": None}
    power_hook = first_block.register_forward_pre_hook(
        lambda _module, _args: sampler.set_phase(active_phase["value"])
    )
    torch.cuda.synchronize()
    time.sleep(float(args.cooldown_seconds))
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(float(args.idle_seconds))
    sampler.set_phase(None)
    measurements: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    image_embeddings: list[torch.Tensor] = []
    ranks: list[int] = []
    try:
        for index, query in enumerate(queries):
            with Image.open(query.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = _inputs(
                loaded.processor,
                image=image,
                instruction=QUERY_INSTRUCTION,
            )
            inputs = move_inputs(inputs, loaded.device)
            timer.reset()
            active_phase["value"] = f"active:{index}"
            try:
                embedding = teacher_embeddings(
                    loaded.model, inputs, EMBEDDING_DIM
                )[0]
                scores = embedding.float() @ title_embeddings.float().T
                order = scores.argsort(descending=True)
                timing = timer.finish()
            finally:
                sampler.set_phase(None)
                active_phase["value"] = None
            matching = torch.nonzero(order.eq(query.label), as_tuple=False)
            if matching.numel() != 1:
                raise RuntimeError("True title did not occur exactly once in ranking")
            rank = int(matching.item()) + 1
            top10 = order[:10].detach().cpu().tolist()
            ranks.append(rank)
            image_embeddings.append(embedding.detach().cpu().to(torch.float16))
            measurements.append(
                {
                    "sample_index": index,
                    "sample_id": query.sample_id,
                    "product_id": query.product_id,
                    "cuda_ms": timing["cuda_ms"],
                    "host_ms": timing["host_ms"],
                    "first_block_calls": timing["first_block_calls"],
                    "first_block_input_shape": json.dumps(
                        timing["first_block_input_shape"]
                    ),
                }
            )
            predictions.append(
                {
                    "sample_id": query.sample_id,
                    "true_product_id": query.product_id,
                    "true_label": query.label,
                    "true_rank": rank,
                    "predicted_product_id": titles[top10[0]].product_id,
                    "predicted_label": top10[0],
                    "top10_labels": json.dumps(top10),
                    "top10_product_ids": json.dumps(
                        [titles[value].product_id for value in top10]
                    ),
                    "top10_scores": json.dumps(
                        [float(scores[value]) for value in top10]
                    ),
                }
            )
            if (index + 1) % 100 == 0:
                print(f"[query] {index + 1}/{len(queries)}", flush=True)
    finally:
        power_hook.remove()
        timer.close()
        power_samples = sampler.stop()

    rank_array = np.asarray(ranks, dtype=np.int64)
    latency_cuda = [float(row["cuda_ms"]) for row in measurements]
    performance = _metrics(rank_array)
    per_product = []
    for title in titles:
        selected = rank_array[
            np.asarray([query.label == title.label for query in queries], dtype=bool)
        ]
        if selected.size:
            per_product.append(
                {"label": title.label, "product_id": title.product_id,
                 "queries": int(selected.size), **_metrics(selected)}
            )
    model_file = model_path / "model.safetensors"
    measured_power = power_report(power_samples, latency_cuda)
    measured_power["sampling_interval_ms"] = POWER_SAMPLE_INTERVAL_MS
    measured_power["pre_idle_cooldown_seconds"] = float(args.cooldown_seconds)
    report = {
        "schema_version": 1,
        "status": "complete" if args.max_queries is None else "diagnostic_subset",
        "task_id": "T08",
        "task": "ABO easy100 image-to-title retrieval",
        "retrieval_direction": "image_query_to_fixed_title_candidates",
        "model": str(model_path),
        "model_family": "Qwen3-VL-Embedding-2B",
        "qwen_frozen": True,
        "optimization": "none; no fine-tuning and no trained readout head",
        "model_parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "batch_size": 1,
        "embedding_dim": EMBEDDING_DIM,
        "image_size": [224, 224],
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "candidate_titles_precomputed": len(titles),
        "test_samples": len(queries),
        "first_test_sample_included_in_timing": True,
        "explicit_warmup_forwards": 0,
        "timing_boundary": (
            "native Vision Transformer block 0 input through all native Vision/"
            "Language blocks, final valid-token 2048-D normalization, cosine "
            "similarity against 100 precomputed title embeddings, and full ranking"
        ),
        "excluded_from_online_timing": (
            "file I/O, image decode, processor/tokenizer, patch embedding, title-bank "
            "precomputation, and model loading"
        ),
        "performance": performance,
        "latency_cuda_ms": summarize(latency_cuda),
        "latency_host_ms": summarize(
            [float(row["host_ms"]) for row in measurements]
        ),
        "power": measured_power,
        "dataset_sha256": {
            "test_csv": sha256_file(data_root / "test.csv"),
            "titles_csv": sha256_file(data_root / "titles.csv"),
            "manifest_csv": sha256_file(data_root / "manifest.csv"),
        },
        "model_safetensors_sha256": (
            sha256_file(model_file) if model_file.is_file() else None
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
        "environment": environment_report(),
        "model_load_seconds": loaded.load_time_sec,
    }
    write_json(run_dir / "baseline_report.json", report)
    write_json(run_dir / "per_product_metrics.json", per_product)
    _write_rows(run_dir / "timing_per_sample.csv", measurements)
    _write_rows(run_dir / "retrieval_predictions.csv", predictions)
    save_power_samples(run_dir / "power_samples.csv", power_samples)
    torch.save(
        {
            "embeddings": torch.stack(image_embeddings),
            "sample_ids": [query.sample_id for query in queries],
            "labels": [query.label for query in queries],
            "product_ids": [query.product_id for query in queries],
            "instruction": QUERY_INSTRUCTION,
        },
        run_dir / "test_image_embeddings.pt",
    )
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    _plot(run_dir, report, rank_array, power_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
