from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch import nn

from .utils import cuda_synchronize


@dataclass(frozen=True)
class GenerationResult:
    labels: list[int]
    predictions: list[int]
    raw_outputs: list[str]
    elapsed_sec: float
    image_count: int


def generate_batch(
    model: nn.Module,
    processor: Any,
    images: Sequence[Image.Image],
    class_names: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    labels = ", ".join(class_names)
    prompt = (
        "Classify this CIFAR-10 image. Reply with exactly one class name from: "
        f"{labels}. Do not add any explanation."
    )
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()
    }
    inputs.pop("token_type_ids", None)
    input_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[:, input_length:]
    return list(
        processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def run_generation(
    model: nn.Module,
    processor: Any,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    class_names: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
) -> GenerationResult:
    labels: list[int] = []
    predictions: list[int] = []
    outputs: list[str] = []
    cuda_synchronize(device)
    start = time.perf_counter()
    for images, batch_labels in loader:
        batch_outputs = generate_batch(
            model, processor, images, class_names, device, max_new_tokens
        )
        outputs.extend(batch_outputs)
        labels.extend(batch_labels.tolist())
        predictions.extend(parse_class_name(value, class_names) for value in batch_outputs)
    cuda_synchronize(device)
    elapsed = time.perf_counter() - start
    return GenerationResult(labels, predictions, outputs, elapsed, len(labels))


def parse_class_name(text: str, class_names: Sequence[str]) -> int:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    aliases = {
        "plane": "airplane",
        "aircraft": "airplane",
        "car": "automobile",
        "auto": "automobile",
        "vehicle": "automobile",
    }
    if normalized in class_names:
        return list(class_names).index(normalized)
    if normalized in aliases:
        return list(class_names).index(aliases[normalized])
    tokens = normalized.split()
    for index, name in enumerate(class_names):
        if name in tokens:
            return index
    for alias, name in aliases.items():
        if alias in tokens:
            return list(class_names).index(name)
    return -1
