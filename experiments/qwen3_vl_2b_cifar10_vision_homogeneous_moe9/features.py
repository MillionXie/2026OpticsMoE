from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn


def preprocess_images(processor: Any, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Qwen processor does not expose image_processor")
    values = image_processor(images=list(images), return_tensors="pt")
    missing = [name for name in ("pixel_values", "image_grid_thw") if name not in values]
    if missing:
        raise RuntimeError(f"Qwen image processor did not return: {', '.join(missing)}")
    return {name: values[name] for name in ("pixel_values", "image_grid_thw")}


def move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}


def run_visual(model: nn.Module, inputs: Mapping[str, torch.Tensor]) -> Any:
    return model.get_image_features(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])


def pool_token_groups(groups: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([group.float().mean(0) for group in groups], dim=0)

