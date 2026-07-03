from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch import nn

from .io_utils import synchronize, write_csv, write_json
from .timing import summarize_timings


IGNORED_MODEL_INPUTS = {"token_type_ids", "mm_token_type_ids"}


def preprocess_image_text(
    processor: Any,
    images: Sequence[Image.Image],
    classification_prompt: str,
) -> dict[str, torch.Tensor]:
    """Build native Qwen3-VL image/text inputs, including image placeholders."""

    texts = [
        _apply_chat_template(processor, image, classification_prompt)
        for image in images
    ]
    values = processor(
        text=texts,
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Qwen3-VL processor did not return: {', '.join(missing)}")
    return {
        name: value
        for name, value in values.items()
        if torch.is_tensor(value) and name not in IGNORED_MODEL_INPUTS
    }


def move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in inputs.items()
    }


def multimodal_forward_features(
    model: nn.Module, inputs: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Run the complete Qwen3-VL vision-language forward and return final hidden states."""

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("Full Qwen3-VL forward did not return language hidden states")
    hidden = hidden_states[-1]
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected [batch, sequence, hidden] tensor, got {tuple(hidden.shape)}")
    return hidden


def pool_answer_hidden_state(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the last non-padding token, i.e. the position ready to emit the answer."""

    if attention_mask.ndim != 2 or attention_mask.shape[:2] != hidden.shape[:2]:
        raise RuntimeError(
            "attention_mask and final hidden state must share [batch, sequence] dimensions"
        )
    sequence_positions = torch.arange(hidden.shape[1], device=hidden.device)
    sequence_positions = sequence_positions.unsqueeze(0).expand(hidden.shape[0], -1)
    answer_positions = sequence_positions.masked_fill(attention_mask.eq(0), -1).max(dim=1).values
    if torch.any(answer_positions < 0):
        raise RuntimeError("Every sample must contain at least one non-padding token")
    batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
    features = hidden[batch_positions, answer_positions].float()
    return features, answer_positions


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
    classification_prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    timings: list[dict[str, Any]] = []
    shape_record: dict[str, Any] | None = None
    iterator = iter(loader)
    total = len(loader) if hasattr(loader, "__len__") else 0
    indices: Any = range(total)
    if progress:
        try:
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc=f"Extract {split} multimodal features")
        except ImportError:
            pass

    with torch.inference_mode():
        for batch_index in indices:
            end_to_end_start = time.perf_counter()
            started = time.perf_counter()
            images, labels = next(iterator)
            data_loading = time.perf_counter() - started

            started = time.perf_counter()
            cpu_inputs = preprocess_image_text(processor, images, classification_prompt)
            preprocess = time.perf_counter() - started

            synchronize(device)
            started = time.perf_counter()
            gpu_inputs = move_inputs(cpu_inputs, device)
            synchronize(device)
            transfer = time.perf_counter() - started

            started = time.perf_counter()
            hidden = multimodal_forward_features(model, gpu_inputs)
            synchronize(device)
            forward = time.perf_counter() - started

            started = time.perf_counter()
            features, answer_positions = pool_answer_hidden_state(
                hidden, gpu_inputs["attention_mask"]
            )
            synchronize(device)
            pooling = time.perf_counter() - started
            end_to_end = time.perf_counter() - end_to_end_start

            feature_chunks.append(features.cpu())
            label_chunks.append(labels.cpu())
            timings.append(
                {
                    "batch": batch_index,
                    "samples": len(images),
                    "data_loading_sec": data_loading,
                    "multimodal_preprocess_sec": preprocess,
                    "host_to_device_sec": transfer,
                    "multimodal_forward_sec": forward,
                    "hidden_pooling_sec": pooling,
                    "mlp_forward_sec": 0.0,
                    "postprocess_sec": 0.0,
                    "pipeline_sec": end_to_end - data_loading,
                    "end_to_end_sec": end_to_end,
                }
            )
            if shape_record is None:
                shape_record = _shape_record(
                    cpu_inputs,
                    hidden,
                    features,
                    answer_positions,
                    classification_prompt,
                )

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
    write_csv(
        output_dir / "metrics" / f"feature_extraction_{split}_batches.csv",
        timings,
        fields,
    )
    summary = summarize_timings(timings)
    summary["shape"] = shape_record or {}
    summary["cache_path"] = str(cache_path)
    summary["cache_dtype"] = cache_dtype
    write_json(output_dir / "metrics" / f"feature_extraction_{split}.json", summary)
    return features, labels, summary


def load_feature_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["features"].float(), payload["labels"].long(), dict(payload["metadata"])


def _apply_chat_template(processor: Any, image: Image.Image, prompt: str) -> str:
    apply_template = getattr(processor, "apply_chat_template", None)
    if apply_template is None:
        return prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return apply_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _shape_record(
    inputs: Mapping[str, torch.Tensor],
    hidden: torch.Tensor,
    features: torch.Tensor,
    answer_positions: torch.Tensor,
    prompt: str,
) -> dict[str, Any]:
    return {
        "input_ids": list(inputs["input_ids"].shape),
        "attention_mask": list(inputs["attention_mask"].shape),
        "pixel_values": list(inputs["pixel_values"].shape),
        "image_grid_thw": list(inputs["image_grid_thw"].shape),
        "image_grid_thw_values": inputs["image_grid_thw"].tolist(),
        "last_hidden_state": list(hidden.shape),
        "answer_positions": answer_positions.detach().cpu().tolist(),
        "answer_hidden_features": list(features.shape),
        "feature_dimension": int(features.shape[-1]),
        "prompt": prompt,
    }

