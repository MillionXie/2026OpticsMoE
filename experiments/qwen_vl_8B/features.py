from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch import nn

from .io_utils import synchronize, write_csv, write_json
from .timing import summarize_timings


def preprocess_images(processor: Any, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Qwen processor does not expose image_processor")
    values = image_processor(images=list(images), return_tensors="pt")
    required = ("pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Image processor did not return: {', '.join(missing)}")
    return {name: values[name] for name in required}


def move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}


def image_token_features(
    model: nn.Module, inputs: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    output = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    primary = output
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (tuple, list)):
        primary = output[0]
    if isinstance(primary, (tuple, list)) and all(torch.is_tensor(item) for item in primary):
        return list(primary)
    if hasattr(primary, "last_hidden_state"):
        primary = primary.last_hidden_state
    if isinstance(primary, tuple) and primary and torch.is_tensor(primary[0]):
        primary = primary[0]
    if not torch.is_tensor(primary):
        raise RuntimeError("Unable to identify Qwen visual token features")
    batch_size = int(inputs["image_grid_thw"].shape[0])
    if primary.ndim == 3 and primary.shape[0] == batch_size:
        return list(primary.unbind(0))
    if primary.ndim == 2 and primary.shape[0] == batch_size:
        return [row.unsqueeze(0) for row in primary]
    if primary.ndim != 2:
        raise RuntimeError(f"Unexpected visual feature shape: {tuple(primary.shape)}")
    lengths = _merged_lengths(model, inputs["image_grid_thw"])
    if sum(lengths) != primary.shape[0]:
        raise RuntimeError(
            f"Packed token count {primary.shape[0]} does not match expected per-image lengths {lengths}"
        )
    return list(primary.split(lengths, dim=0))


def pool_tokens(tokens: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.float().mean(dim=0) for value in tokens], dim=0)


def extract_and_cache(
    model: nn.Module,
    processor: Any,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    device: torch.device,
    split: str,
    output_dir: Path,
    metadata: Mapping[str, Any],
    cache_dtype: str,
    progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    timings: list[dict[str, Any]] = []
    shape_record: dict[str, Any] | None = None
    iterator = iter(loader)
    total = len(loader) if hasattr(loader, "__len__") else None
    if progress:
        try:
            from tqdm.auto import tqdm

            indices = tqdm(range(total or 0), desc=f"Extract {split} features")
        except ImportError:
            indices = range(total or 0)
    else:
        indices = range(total or 0)
    with torch.inference_mode():
        for batch_index in indices:
            end_to_end_start = time.perf_counter()
            fetch_start = time.perf_counter()
            images, labels = next(iterator)
            data_loading = time.perf_counter() - fetch_start

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
            end_to_end = time.perf_counter() - end_to_end_start
            feature_chunks.append(pooled.cpu())
            label_chunks.append(labels.cpu())
            timings.append(
                {
                    "batch": batch_index,
                    "samples": len(images),
                    "data_loading_sec": data_loading,
                    "image_preprocess_sec": preprocess,
                    "host_to_device_sec": transfer,
                    "vision_forward_sec": vision,
                    "pooling_sec": pooling,
                    "mlp_forward_sec": 0.0,
                    "model_inference_sec": vision + pooling,
                    "postprocess_sec": 0.0,
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
                    "feature_dimension": int(pooled.shape[-1]),
                }

    features = torch.cat(feature_chunks)
    labels = torch.cat(label_chunks)
    stored = features.half() if cache_dtype == "float16" else features.float()
    payload = {"features": stored, "labels": labels, "metadata": dict(metadata)}
    cache_path = output_dir / "features" / f"{split}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    fields = list(timings[0]) if timings else []
    write_csv(output_dir / "metrics" / f"feature_extraction_{split}_batches.csv", timings, fields)
    summary = summarize_timings(timings)
    summary["shape"] = shape_record or {}
    summary["cache_path"] = str(cache_path)
    summary["cache_dtype"] = cache_dtype
    write_json(output_dir / "metrics" / f"feature_extraction_{split}.json", summary)
    return features, labels, summary


def load_feature_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["features"].float(), payload["labels"].long(), dict(payload["metadata"])


def _merged_lengths(model: nn.Module, grid: torch.Tensor) -> list[int]:
    config = getattr(getattr(model, "config", None), "vision_config", None)
    merge = int(getattr(config, "spatial_merge_size", 1))
    return [int(value) // (merge**2) for value in grid.prod(dim=-1).cpu().tolist()]
