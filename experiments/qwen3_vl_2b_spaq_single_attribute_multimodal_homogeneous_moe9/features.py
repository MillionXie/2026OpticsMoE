from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn


IGNORED_MODEL_INPUTS = {"token_type_ids", "mm_token_type_ids"}


def apply_chat_template(processor: Any, image: Image.Image, prompt: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        messages[0]["content"][0] = {"type": "image"}
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def preprocess_image_text(processor: Any, images: Sequence[Image.Image], prompt: str) -> dict[str, torch.Tensor]:
    texts = [apply_chat_template(processor, image, prompt) for image in images]
    values = processor(text=texts, images=list(images), return_tensors="pt", padding=True)
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Qwen3-VL processor did not return: {', '.join(missing)}")
    return {name: value for name, value in values.items()
            if torch.is_tensor(value) and name not in IGNORED_MODEL_INPUTS}


def move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}


def multimodal_forward_features(model: nn.Module, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "_spaq_optical_last_hidden"):
        setattr(model, "_spaq_optical_last_hidden", None)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = getattr(outputs, "hidden_states", None)
    hidden = hidden_states[-1] if hidden_states else getattr(model, "_spaq_optical_last_hidden", None)
    if hidden is None:
        raise RuntimeError("Full Qwen3-VL forward did not expose final language hidden states")
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected final language hidden [B,S,D], got {tuple(hidden.shape)}")
    return hidden


def pool_answer_hidden_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask.ndim != 2 or attention_mask.shape != hidden.shape[:2]:
        raise RuntimeError("attention_mask and hidden must share [batch, sequence] dimensions")
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
    positions = positions.masked_fill(attention_mask.eq(0), -1)
    answer_positions = positions.max(dim=1).values
    if torch.any(answer_positions < 0):
        raise RuntimeError("Every sample must contain a valid answer-position token")
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, answer_positions].float(), answer_positions
