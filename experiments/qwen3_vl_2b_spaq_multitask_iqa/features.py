from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn

from .io_utils import write_json


IGNORED_MODEL_INPUTS = {"token_type_ids", "mm_token_type_ids"}
CACHE_SCHEMA_VERSION = 1


def preprocess_image_text(
    processor: Any,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
) -> dict[str, torch.Tensor]:
    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same batch length")
    texts = [_apply_chat_template(processor, image, prompt) for image, prompt in zip(images, prompts)]
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
    return {name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}


def full_multimodal_features(
    model: nn.Module, inputs: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.training:
        raise RuntimeError("Frozen Qwen backbone must remain in eval mode during feature extraction")
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
    attention_mask = inputs["attention_mask"]
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise RuntimeError(
            f"Unexpected hidden/mask shapes: hidden={tuple(hidden.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    positions = positions.expand(hidden.shape[0], -1).masked_fill(attention_mask.eq(0), -1)
    answer_positions = positions.max(dim=1).values
    if torch.any(answer_positions < 0):
        raise RuntimeError("Every sample must have at least one valid prompt token")
    batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_positions, answer_positions].float(), answer_positions


def extract_and_cache(
    model: nn.Module,
    processor: Any,
    loader: Any,
    device: torch.device,
    split: str,
    cache_path: Path,
    expected_metadata: Mapping[str, Any],
    cache_dtype: str,
    expected_feature_dim: int,
    progress: bool,
) -> dict[str, Any]:
    model.requires_grad_(False)
    model.eval()
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    task_chunks: list[torch.Tensor] = []
    sample_index_chunks: list[torch.Tensor] = []
    image_names: list[str] = []
    image_paths: list[str] = []
    tasks: list[str] = []
    grid_values: list[list[int]] = []
    answer_positions_all: list[int] = []
    started = time.perf_counter()
    iterator: Any = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=f"Extract {split} SPAQ image-prompt features")
        except ImportError:
            pass
    with torch.inference_mode():
        for batch in iterator:
            cpu_inputs = preprocess_image_text(processor, batch["images"], batch["prompts"])
            gpu_inputs = move_inputs(cpu_inputs, device)
            features, answer_positions = full_multimodal_features(model, gpu_inputs)
            if features.shape[-1] != expected_feature_dim:
                raise RuntimeError(
                    f"Qwen answer hidden dimension is {features.shape[-1]}, "
                    f"expected {expected_feature_dim}"
                )
            feature_chunks.append(features.cpu())
            label_chunks.append(batch["normalized_scores"].cpu())
            score_chunks.append(batch["scores"].cpu())
            task_chunks.append(batch["task_indices"].cpu())
            sample_index_chunks.append(batch["sample_indices"].cpu())
            image_names.extend(batch["image_names"])
            image_paths.extend(batch["image_paths"])
            tasks.extend(batch["tasks"])
            grid_values.extend(cpu_inputs["image_grid_thw"].tolist())
            answer_positions_all.extend(answer_positions.cpu().tolist())
    if not feature_chunks:
        raise RuntimeError(f"Cannot cache empty split: {split}")
    features = torch.cat(feature_chunks)
    stored_features = features.half() if cache_dtype == "float16" else features.float()
    payload = {
        "features": stored_features,
        "normalized_scores": torch.cat(label_chunks).float(),
        "scores": torch.cat(score_chunks).float(),
        "task_indices": torch.cat(task_chunks).long(),
        "sample_indices": torch.cat(sample_index_chunks).long(),
        "image_names": image_names,
        "image_paths": image_paths,
        "tasks": tasks,
        "image_grid_thw": torch.tensor(grid_values, dtype=torch.long),
        "answer_positions": torch.tensor(answer_positions_all, dtype=torch.long),
        "metadata": dict(expected_metadata),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    summary = {
        "split": split,
        "samples": len(features),
        "original_images": len(set(image_names)),
        "feature_shape": list(features.shape),
        "cache_dtype": cache_dtype,
        "cache_path": str(cache_path),
        "elapsed_sec": time.perf_counter() - started,
        "full_multimodal_forward": True,
        "generation_used": False,
        "answer_positions_min": min(answer_positions_all),
        "answer_positions_max": max(answer_positions_all),
        "image_grid_thw_unique": sorted({tuple(value) for value in grid_values}),
    }
    write_json(cache_path.parent.parent / "metrics" / f"feature_extraction_{split}.json", summary)
    return payload


def load_feature_cache(
    path: Path,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen feature cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "features", "normalized_scores", "scores", "task_indices", "sample_indices",
        "image_names", "image_paths", "tasks", "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Feature cache {path} is missing keys: {missing}")
    if expected_metadata is not None:
        saved = payload["metadata"]
        mismatches = {
            key: {"saved": saved.get(key), "current": value}
            for key, value in expected_metadata.items()
            if saved.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Feature cache metadata mismatch for {path}: {mismatches}. "
                "Delete the stale cache or use a new output_dir."
            )
    payload["features"] = payload["features"].float()
    return payload


def cache_metadata(
    split: str,
    split_samples: int,
    model_id: str,
    processor_min_pixels: int | None,
    processor_max_pixels: int | None,
    dtype: str,
    attn_implementation: str,
    expected_feature_dim: int,
    dataset_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "samples": split_samples,
        "model_id": model_id,
        "processor_min_pixels": processor_min_pixels,
        "processor_max_pixels": processor_max_pixels,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "feature_dim": expected_feature_dim,
        "feature_source": "full_qwen3_vl_final_language_hidden_last_valid_prompt_token",
        **dict(dataset_identity),
    }


def _apply_chat_template(processor: Any, image: Image.Image, prompt: str) -> str:
    apply_template = getattr(processor, "apply_chat_template", None)
    if apply_template is None:
        raise RuntimeError(
            "Qwen processor has no apply_chat_template method; native image placeholder alignment "
            "cannot be guaranteed"
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return apply_template(messages, tokenize=False, add_generation_prompt=True)
