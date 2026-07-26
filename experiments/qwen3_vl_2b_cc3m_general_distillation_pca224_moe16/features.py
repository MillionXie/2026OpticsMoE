from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn


IGNORED_MODEL_INPUTS = {"token_type_ids", "mm_token_type_ids"}


def apply_chat_template(processor: Any, image: Image.Image, text_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt},
        ],
    }]
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        messages[0]["content"][0] = {"type": "image"}
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def preprocess_image_text(
    processor: Any,
    images: Sequence[Image.Image],
    captions: Sequence[str],
    prompt_template: str,
) -> dict[str, torch.Tensor]:
    if len(images) != len(captions):
        raise ValueError("images and captions must have equal length")
    prompts = [prompt_template.format(caption=caption) for caption in captions]
    texts = [
        apply_chat_template(processor, image, prompt)
        for image, prompt in zip(images, prompts)
    ]
    values = processor(text=texts, images=list(images), return_tensors="pt", padding=True)
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
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}


def run_multimodal_forward(model: nn.Module, inputs: Mapping[str, torch.Tensor]) -> None:
    model(
        **inputs,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
        logits_to_keep=1,
    )
