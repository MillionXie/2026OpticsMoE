from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn


LAST_HIDDEN_ATTRIBUTE = "_flickr30k_itm_optical_last_hidden"


def apply_chat_template(processor: Any, image: Image.Image, prompt: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def preprocess_image_text(processor: Any, images: Sequence[Image.Image],
                          prompts: Sequence[str]) -> dict[str, torch.Tensor]:
    if len(images) != len(prompts):
        raise ValueError(f"images/prompts length mismatch: {len(images)} != {len(prompts)}")
    if not images:
        raise ValueError("preprocess_image_text requires at least one image/prompt pair")
    texts = [apply_chat_template(processor, image, prompt) for image, prompt in zip(images, prompts)]
    encoded = processor(text=texts, images=list(images), padding=True, return_tensors="pt")
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def multimodal_forward_features(model: nn.Module, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, LAST_HIDDEN_ATTRIBUTE):
        setattr(model, LAST_HIDDEN_ATTRIBUTE, None)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    hidden = hidden_states[-1] if hidden_states else getattr(model, LAST_HIDDEN_ATTRIBUTE, None)
    if hidden is None:
        raise RuntimeError("Full Qwen3-VL forward did not return or expose final language hidden states")
    return hidden


def pool_answer_hidden_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or attention_mask.ndim != 2 or hidden.shape[:2] != attention_mask.shape:
        raise RuntimeError(f"Hidden/attention shapes are incompatible: {tuple(hidden.shape)} vs {tuple(attention_mask.shape)}")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if torch.any(last_positions < 0):
        raise RuntimeError("Every image-caption sample must contain at least one valid language token")
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, last_positions], last_positions
