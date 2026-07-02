from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch import nn

from .models import backbone_core
from .progress import progress_iter, progress_total
from .utils import cuda_synchronize


FEATURE_SOURCES = (
    "visual_tokens_mean",
    "vision_pooler",
    "multimodal_image_tokens_mean",
    "last_hidden_mean",
)
MULTIMODAL_PROMPT = "Represent this image for CIFAR-10 image classification."


class FeatureSourceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: torch.Tensor
    labels: torch.Tensor
    elapsed_sec: float
    image_count: int


def extract_feature_batch(
    model: nn.Module,
    processor: Any,
    images: Sequence[Image.Image],
    feature_source: str,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    context = nullcontext() if requires_grad else torch.inference_mode()
    with context:
        if feature_source in {"visual_tokens_mean", "vision_pooler"}:
            features = _extract_vision_features(
                model, processor, images, feature_source, device
            )
        elif feature_source in {"multimodal_image_tokens_mean", "last_hidden_mean"}:
            features = _extract_multimodal_features(
                model, processor, images, feature_source, device
            )
        else:
            raise ValueError(f"Unknown feature source: {feature_source}")
    if features.ndim != 2 or features.shape[0] != len(images):
        raise RuntimeError(
            f"Feature source {feature_source} returned shape {tuple(features.shape)} for "
            f"a batch of {len(images)} images."
        )
    return features.to(device)


def _extract_vision_features(
    model: nn.Module,
    processor: Any,
    images: Sequence[Image.Image],
    feature_source: str,
    device: torch.device,
) -> torch.Tensor:
    if not hasattr(model, "get_image_features"):
        raise FeatureSourceUnavailable(
            "This transformers/Qwen3-VL implementation does not expose get_image_features. "
            "Use --feature-source multimodal_image_tokens_mean or last_hidden_mean."
        )
    # Some 4.57-era Qwen3-VL processors run image-token replacement even when text is None.
    # Empty strings keep this public processor path version-compatible; text outputs are ignored.
    inputs = processor(
        images=list(images),
        text=[""] * len(images),
        return_tensors="pt",
        padding=True,
    )
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise FeatureSourceUnavailable(
            "The processor did not return pixel_values and image_grid_thw. Use a current Qwen3-VL "
            "processor or select --feature-source multimodal_image_tokens_mean."
        )
    pixel_values = inputs["pixel_values"].to(device)
    grid = inputs["image_grid_thw"].to(device)
    output = model.get_image_features(pixel_values=pixel_values, image_grid_thw=grid)

    # Transformers 4.57 returns (tuple[per-image merged tokens], deepstack features).
    if isinstance(output, tuple) and output and isinstance(output[0], (tuple, list)):
        if feature_source == "vision_pooler":
            raise FeatureSourceUnavailable(
                "vision_pooler is unavailable in this transformers Qwen3-VL implementation. "
                "Use --feature-source visual_tokens_mean (recommended) or "
                "multimodal_image_tokens_mean."
            )
        return torch.stack([tokens.mean(dim=0) for tokens in output[0]], dim=0).float()

    if feature_source == "vision_pooler":
        pooler = getattr(output, "pooler_output", None)
        if pooler is None:
            raise FeatureSourceUnavailable(
                "Qwen3-VL did not expose pooler_output. Use --feature-source visual_tokens_mean "
                "or multimodal_image_tokens_mean."
            )
        return _pool_packed_or_batched(pooler, grid, model, merged=True).float()

    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return _pool_packed_or_batched(hidden, grid, model, merged=False).float()
    if torch.is_tensor(output):
        return _pool_packed_or_batched(output, grid, model, merged=True).float()
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return _pool_packed_or_batched(output[0], grid, model, merged=True).float()
    raise FeatureSourceUnavailable(
        "Unable to identify visual token hidden states in get_image_features output. "
        "Use --feature-source multimodal_image_tokens_mean or last_hidden_mean."
    )


def _pool_packed_or_batched(
    hidden: torch.Tensor,
    grid: torch.Tensor,
    model: nn.Module,
    merged: bool,
) -> torch.Tensor:
    batch_size = int(grid.shape[0])
    if hidden.ndim == 3 and hidden.shape[0] == batch_size:
        return hidden.mean(dim=1)
    if hidden.ndim == 2 and hidden.shape[0] == batch_size:
        return hidden
    if hidden.ndim != 2:
        raise FeatureSourceUnavailable(
            f"Expected 2D packed or 3D batched vision features, got {hidden.shape}."
        )

    lengths = grid.prod(dim=-1)
    if merged:
        merge_size = _spatial_merge_size(model)
        lengths = lengths // (merge_size**2)
    length_list = [int(value) for value in lengths.tolist()]
    if sum(length_list) != hidden.shape[0]:
        raise FeatureSourceUnavailable(
            "Cannot split packed vision features: expected "
            f"{sum(length_list)} tokens from image_grid_thw, "
            f"received {hidden.shape[0]}. Try --feature-source multimodal_image_tokens_mean."
        )
    return torch.stack(
        [part.mean(dim=0) for part in torch.split(hidden, length_list)], dim=0
    )


def _spatial_merge_size(model: nn.Module) -> int:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = (
        getattr(getattr(base, "model", None), "visual", None),
        getattr(base, "visual", None),
    )
    for visual in candidates:
        if visual is not None and hasattr(visual, "spatial_merge_size"):
            return int(visual.spatial_merge_size)
    vision_config = getattr(getattr(base, "config", None), "vision_config", None)
    return int(getattr(vision_config, "spatial_merge_size", 1))


def _extract_multimodal_features(
    model: nn.Module,
    processor: Any,
    images: Sequence[Image.Image],
    feature_source: str,
    device: torch.device,
) -> torch.Tensor:
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": MULTIMODAL_PROMPT},
                ],
            }
        ]
        for image in images
    ]
    try:
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
    except (TypeError, ValueError) as exc:
        raise FeatureSourceUnavailable(
            "The installed processor cannot batch PIL images through apply_chat_template. "
            "Upgrade transformers or use --feature-source visual_tokens_mean."
        ) from exc
    inputs = _move_inputs(inputs, device)
    inputs.pop("token_type_ids", None)
    core = backbone_core(model)
    output = core(**inputs, return_dict=True, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        raise FeatureSourceUnavailable(
            "The multimodal backbone did not return last_hidden_state; use visual_tokens_mean."
        )

    if feature_source == "last_hidden_mean":
        mask = inputs.get(
            "attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)
        ).bool()
    else:
        mm_types = inputs.get("mm_token_type_ids")
        if mm_types is not None:
            mask = mm_types.eq(1)
        else:
            input_ids = inputs.get("input_ids")
            config = getattr(core, "config", getattr(model, "config", None))
            image_token_id = getattr(config, "image_token_id", None)
            if input_ids is None or image_token_id is None:
                raise FeatureSourceUnavailable(
                    "Cannot identify image-token positions (no mm_token_type_ids/image_token_id). "
                    "Use --feature-source last_hidden_mean or visual_tokens_mean."
                )
            mask = input_ids.eq(int(image_token_id))
        if not torch.all(mask.sum(dim=1) > 0):
            raise FeatureSourceUnavailable(
                "At least one sample has no image tokens in the multimodal sequence. "
                "Use visual_tokens_mean or upgrade the Qwen3-VL processor."
            )
    return _masked_mean(hidden, mask).float()


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def _move_inputs(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def extract_dataset_features(
    model: nn.Module,
    processor: Any,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    feature_source: str,
    device: torch.device,
    description: str = "feature extraction",
    show_progress: bool = True,
) -> FeatureExtractionResult:
    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    image_count = 0
    cuda_synchronize(device)
    start = time.perf_counter()
    batches = progress_iter(
        loader,
        description=description,
        enabled=show_progress,
        total=progress_total(loader),
    )
    for images, labels in batches:
        batch_features = extract_feature_batch(
            model, processor, images, feature_source, device
        )
        feature_parts.append(batch_features.detach().cpu())
        label_parts.append(labels.cpu())
        image_count += len(images)
    cuda_synchronize(device)
    elapsed = time.perf_counter() - start
    if not feature_parts:
        raise RuntimeError("Feature extraction received an empty data loader.")
    return FeatureExtractionResult(
        features=torch.cat(feature_parts).float(),
        labels=torch.cat(label_parts).long(),
        elapsed_sec=elapsed,
        image_count=image_count,
    )


def cache_metadata(
    model_id: str,
    feature_source: str,
    image_size: int,
    resize_to: int | None,
    dtype: str,
    split: str,
    sample_count: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": "CIFAR10",
        "split": split,
        "sample_count": sample_count,
        "class_names": list(class_names),
        "model_id": model_id,
        "feature_source": feature_source,
        "image_size": image_size,
        "resize_to": resize_to,
        "dtype": dtype,
    }


def load_feature_cache(
    path: Path, expected: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        payload = torch.load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or "features" not in payload
        or "labels" not in payload
    ):
        return None
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    features = payload["features"]
    labels = payload["labels"]
    if not torch.is_tensor(features) or not torch.is_tensor(labels):
        return None
    if (
        len(features) != expected["sample_count"]
        or len(labels) != expected["sample_count"]
    ):
        return None
    return features.float(), labels.long()


def save_feature_cache(
    path: Path,
    result: FeatureExtractionResult,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update(
        {
            "features": result.features.cpu(),
            "labels": result.labels.cpu(),
            "feature_dim": int(result.features.shape[1]),
            "extraction_time_sec": result.elapsed_sec,
            "extraction_images_per_second": (
                result.image_count / result.elapsed_sec
                if result.elapsed_sec > 0
                else 0.0
            ),
        }
    )
    torch.save(payload, path)
