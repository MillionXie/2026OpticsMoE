from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn

from .models import backbone_core
from .utils import cuda_synchronize


TIMING_FIELDS = (
    "dataset_fetch_sec",
    "prompt_build_sec",
    "image_preprocess_sec",
    "tokenizer_sec",
    "processor_framework_sec",
    "processor_total_sec",
    "host_to_device_sec",
    "model_generate_sec",
    "decode_postprocess_sec",
    "complete_inference_sec",
    "end_to_end_sec",
)


@dataclass
class BatchTiming:
    batch_index: int
    image_count: int
    input_tokens: int
    generated_tokens: int
    dataset_fetch_sec: float
    prompt_build_sec: float
    image_preprocess_sec: float
    tokenizer_sec: float
    processor_framework_sec: float
    processor_total_sec: float
    host_to_device_sec: float
    model_generate_sec: float
    decode_postprocess_sec: float
    complete_inference_sec: float
    end_to_end_sec: float

    def as_dict(self) -> dict[str, int | float]:
        return dict(vars(self))


@dataclass
class ComponentTimer:
    total_sec: float = 0.0
    calls: int = 0

    def add(self, elapsed: float) -> None:
        self.total_sec += elapsed
        self.calls += 1


class _TimedComponent:
    def __init__(self, component: Any, timer: ComponentTimer) -> None:
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_timer", timer)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = self._component(*args, **kwargs)
        self._timer.add(time.perf_counter() - start)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._component, name, value)


@contextmanager
def timed_processor_components(
    processor: Any,
) -> Iterator[tuple[ComponentTimer, ComponentTimer]]:
    original_image_processor = processor.image_processor
    original_tokenizer = processor.tokenizer
    image_timer = ComponentTimer()
    tokenizer_timer = ComponentTimer()
    processor.image_processor = _TimedComponent(original_image_processor, image_timer)
    processor.tokenizer = _TimedComponent(original_tokenizer, tokenizer_timer)
    try:
        yield image_timer, tokenizer_timer
    finally:
        processor.image_processor = original_image_processor
        processor.tokenizer = original_tokenizer


def summarize_batch_timings(rows: Sequence[BatchTiming]) -> dict[str, Any]:
    return {field: _summarize_field(rows, field) for field in TIMING_FIELDS}


def _summarize_field(rows: Sequence[BatchTiming], field: str) -> dict[str, float | int]:
    values = [float(getattr(row, field)) for row in rows]
    per_image = [value / row.image_count for value, row in zip(values, rows)]
    total_images = sum(row.image_count for row in rows)
    total = sum(values)
    return {
        "batches": len(values),
        "images": total_images,
        "total_sec": total,
        "mean_batch_sec": statistics.fmean(values) if values else 0.0,
        "mean_ms_per_image": 1000.0 * statistics.fmean(per_image) if per_image else 0.0,
        "median_ms_per_image": 1000.0 * statistics.median(per_image)
        if per_image
        else 0.0,
        "p90_ms_per_image": 1000.0 * _percentile(per_image, 0.90),
        "p95_ms_per_image": 1000.0 * _percentile(per_image, 0.95),
        "p99_ms_per_image": 1000.0 * _percentile(per_image, 0.99),
        "std_ms_per_image": (
            1000.0 * statistics.pstdev(per_image) if len(per_image) > 1 else 0.0
        ),
        "images_per_second": total_images / total if total > 0 else 0.0,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
    }


def parameter_statistics(model: nn.Module) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, int]] = {}
    by_device: dict[str, dict[str, int]] = {}
    total_parameters = 0
    trainable_parameters = 0
    total_bytes = 0
    for parameter in model.parameters():
        count = parameter.numel()
        size = count * parameter.element_size()
        total_parameters += count
        total_bytes += size
        if parameter.requires_grad:
            trainable_parameters += count
        dtype_key = str(parameter.dtype).removeprefix("torch.")
        device_key = str(parameter.device)
        _add_parameter_group(by_dtype, dtype_key, count, size)
        _add_parameter_group(by_device, device_key, count, size)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_bytes": total_bytes,
        "parameter_gib": total_bytes / (1024**3),
        "by_dtype": by_dtype,
        "by_device": by_device,
    }


def _add_parameter_group(
    groups: dict[str, dict[str, int]], key: str, count: int, size: int
) -> None:
    group = groups.setdefault(key, {"parameters": 0, "bytes": 0})
    group["parameters"] += count
    group["bytes"] += size


def architecture_statistics(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", config)
    vision = getattr(config, "vision_config", None)
    return {
        "text": _config_fields(
            text,
            (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            ),
        ),
        "vision": _config_fields(
            vision,
            (
                "hidden_size",
                "intermediate_size",
                "depth",
                "num_heads",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
                "out_hidden_size",
                "deepstack_visual_indexes",
            ),
        ),
    }


def _config_fields(config: Any, names: Sequence[str]) -> dict[str, Any]:
    if config is None:
        return {}
    return {name: getattr(config, name, None) for name in names}


@dataclass
class ShapeRecorder:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def hook(self, name: str, module: nn.Module) -> Any:
        self.records[name] = {
            "module_type": type(module).__name__,
            "calls": 0,
            "observations": [],
        }

        def callback(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            record = self.records[name]
            record["calls"] += 1
            if tensor is None:
                return
            descriptor = tensor_descriptor(tensor)
            for existing in record["observations"]:
                if (
                    existing["shape"] == descriptor["shape"]
                    and existing["dtype"] == descriptor["dtype"]
                    and existing["device"] == descriptor["device"]
                ):
                    existing["calls"] += 1
                    return
            descriptor["calls"] = 1
            record["observations"].append(descriptor)

        return module.register_forward_hook(callback)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def audit_model_features(
    model: nn.Module,
    inputs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    core = backbone_core(model)
    recorder = ShapeRecorder()
    handles = _register_shape_hooks(core, recorder)
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"token_type_ids", "mm_token_type_ids"}
    }
    input_tensors = {
        key: tensor_descriptor(value)
        for key, value in model_inputs.items()
        if torch.is_tensor(value)
    }
    try:
        cuda_synchronize(device)
        vision_start = time.perf_counter()
        with torch.inference_mode():
            image_output = model.get_image_features(
                pixel_values=model_inputs["pixel_values"],
                image_grid_thw=model_inputs["image_grid_thw"],
            )
        cuda_synchronize(device)
        vision_elapsed = time.perf_counter() - vision_start
        image_features, deepstack_features = _normalize_image_features(image_output)
        pooled = torch.stack([feature.mean(dim=0) for feature in image_features])

        cuda_synchronize(device)
        multimodal_start = time.perf_counter()
        with torch.inference_mode():
            output = core(**model_inputs, return_dict=True, use_cache=False)
        cuda_synchronize(device)
        multimodal_elapsed = time.perf_counter() - multimodal_start
        hidden = getattr(output, "last_hidden_state", _first_tensor(output))
    finally:
        for handle in handles:
            handle.remove()

    return {
        "architecture": architecture_statistics(model),
        "input_tensors": input_tensors,
        "vision_encoder_forward_sec": vision_elapsed,
        "multimodal_prefill_forward_sec": multimodal_elapsed,
        "image_feature_count": len(image_features),
        "per_image_features": [tensor_descriptor(value) for value in image_features],
        "pooled_feature": tensor_descriptor(pooled),
        "feature_dimension": int(pooled.shape[-1]),
        "deepstack_features": [
            tensor_descriptor(value)
            for value in deepstack_features
            if torch.is_tensor(value)
        ],
        "multimodal_last_hidden_state": (
            tensor_descriptor(hidden) if torch.is_tensor(hidden) else None
        ),
        "intermediate_modules": recorder.records,
    }


def _normalize_image_features(
    output: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    deepstack: list[torch.Tensor] = []
    primary = output
    if isinstance(output, tuple) and len(output) == 2:
        primary, raw_deepstack = output
        if isinstance(raw_deepstack, (tuple, list)):
            deepstack = list(raw_deepstack)
    if torch.is_tensor(primary):
        features = [primary]
    elif isinstance(primary, (tuple, list)) and all(
        torch.is_tensor(item) for item in primary
    ):
        features = list(primary)
    else:
        raise RuntimeError(
            "Unable to identify per-image tensors from get_image_features output."
        )
    return features, deepstack


def _register_shape_hooks(core: nn.Module, recorder: ShapeRecorder) -> list[Any]:
    handles = []
    visual = getattr(core, "visual", None)
    vision_blocks = getattr(visual, "blocks", None)
    if visual is not None:
        for name in ("patch_embed", "merger"):
            module = getattr(visual, name, None)
            if isinstance(module, nn.Module):
                handles.append(recorder.hook(f"vision.{name}", module))
    if vision_blocks is not None:
        for index in _representative_indices(len(vision_blocks)):
            block = vision_blocks[index]
            handles.append(recorder.hook(f"vision.blocks.{index}", block))
            linear_fc1 = getattr(getattr(block, "mlp", None), "linear_fc1", None)
            if isinstance(linear_fc1, nn.Module):
                handles.append(
                    recorder.hook(f"vision.blocks.{index}.mlp.linear_fc1", linear_fc1)
                )

    language = getattr(core, "language_model", None)
    text_layers = getattr(language, "layers", None)
    if text_layers is not None:
        for index in _representative_indices(len(text_layers)):
            layer = text_layers[index]
            handles.append(recorder.hook(f"text.layers.{index}", layer))
            gate_proj = getattr(getattr(layer, "mlp", None), "gate_proj", None)
            if isinstance(gate_proj, nn.Module):
                handles.append(
                    recorder.hook(f"text.layers.{index}.mlp.gate_proj", gate_proj)
                )
    return handles


def _representative_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    return sorted({0, length // 4, length // 2, (3 * length) // 4, length - 1})
