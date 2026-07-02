from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch import nn

from .features import image_token_features, move_inputs, pool_tokens, preprocess_images
from .io_utils import synchronize, write_csv, write_json
from .metrics import classification_metrics
from .timing import summarize_timings


def benchmark_inference(
    model: nn.Module,
    processor: Any,
    head: nn.Module,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    warmup_batches: int,
    benchmark_batches: int | None,
    progress: bool,
) -> dict[str, Any]:
    model.eval()
    head.eval()
    _warmup(model, processor, head, loader, device, warmup_batches)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    top5_all: list[list[int]] = []
    shape_record: dict[str, Any] | None = None
    iterator = iter(loader)
    available = len(loader) if hasattr(loader, "__len__") else benchmark_batches
    count = available if benchmark_batches is None else min(available or benchmark_batches, benchmark_batches)
    indices: Any = range(count or 0)
    if progress:
        try:
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc="End-to-end inference")
        except ImportError:
            pass

    with torch.inference_mode():
        for batch_index in indices:
            end_to_end_start = time.perf_counter()
            started = time.perf_counter()
            images, labels = next(iterator)
            data_loading = time.perf_counter() - started

            started = time.perf_counter()
            cpu_inputs = preprocess_images(processor, images)
            preprocess = time.perf_counter() - started

            synchronize(device)
            started = time.perf_counter()
            gpu_inputs = move_inputs(cpu_inputs, device)
            synchronize(device)
            transfer = time.perf_counter() - started

            started = time.perf_counter()
            tokens = image_token_features(model, gpu_inputs)
            synchronize(device)
            vision = time.perf_counter() - started

            started = time.perf_counter()
            pooled = pool_tokens(tokens)
            synchronize(device)
            pooling = time.perf_counter() - started

            started = time.perf_counter()
            logits = head(pooled)
            synchronize(device)
            mlp = time.perf_counter() - started

            started = time.perf_counter()
            cpu_logits = logits.float().cpu()
            predictions = cpu_logits.argmax(dim=-1)
            top5 = cpu_logits.topk(min(5, len(class_names)), dim=-1).indices
            postprocess = time.perf_counter() - started
            end_to_end = time.perf_counter() - end_to_end_start

            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())
            top5_all.extend(top5.tolist())
            rows.append(
                {
                    "batch": batch_index,
                    "samples": len(images),
                    "data_loading_sec": data_loading,
                    "image_preprocess_sec": preprocess,
                    "host_to_device_sec": transfer,
                    "vision_forward_sec": vision,
                    "pooling_sec": pooling,
                    "mlp_forward_sec": mlp,
                    "model_inference_sec": vision + pooling + mlp,
                    "postprocess_sec": postprocess,
                    "pipeline_sec": end_to_end - data_loading,
                    "end_to_end_sec": end_to_end,
                }
            )
            if shape_record is None:
                shape_record = {
                    "pixel_values": list(cpu_inputs["pixel_values"].shape),
                    "image_grid_thw": cpu_inputs["image_grid_thw"].tolist(),
                    "per_image_visual_tokens": [list(value.shape) for value in tokens],
                    "pooled_features": list(pooled.shape),
                    "logits": list(logits.shape),
                }

    metrics = classification_metrics(labels_all, predictions_all, top5_all, class_names)
    timing = summarize_timings(rows)
    peak_memory = _peak_memory(device)
    report = {
        "metrics": vars(metrics),
        "timing": timing,
        "feature_shapes": shape_record or {},
        "warmup_batches": warmup_batches,
        "peak_cuda_memory": peak_memory,
    }
    write_csv(output_dir / "metrics" / "inference_batches.csv", rows, list(rows[0]) if rows else [])
    write_json(output_dir / "metrics" / "inference.json", report)
    _write_predictions(output_dir / "metrics" / "predictions.csv", labels_all, predictions_all, class_names)
    _write_confusion(output_dir / "metrics" / "confusion_matrix.csv", metrics.confusion_matrix, class_names)
    return report


def _warmup(
    model: nn.Module,
    processor: Any,
    head: nn.Module,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    device: torch.device,
    batches: int,
) -> None:
    iterator = iter(loader)
    with torch.inference_mode():
        for _ in range(batches):
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _ = next(iterator)
            inputs = move_inputs(preprocess_images(processor, images), device)
            logits = head(pool_tokens(image_token_features(model, inputs)))
            del logits
    synchronize(device)


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {
        "allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _write_predictions(
    path: Path, labels: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> None:
    rows = [
        {
            "index": index,
            "label": label,
            "prediction": prediction,
            "label_name": class_names[label],
            "prediction_name": class_names[prediction],
            "correct": int(label == prediction),
        }
        for index, (label, prediction) in enumerate(zip(labels, predictions))
    ]
    write_csv(path, rows, list(rows[0]) if rows else [])


def _write_confusion(path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])
