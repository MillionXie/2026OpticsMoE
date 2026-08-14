from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn

from .modeling import OpticalRetrievalReadout, official_mrl_embedding
from .optics.replacement import DeepStackMultimodalReplacement


IGNORED_MODEL_INPUTS = {"token_type_ids", "mm_token_type_ids"}


def apply_embedding_template(processor: Any, image: Image.Image, instruction: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": image}],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        messages[1]["content"][0] = {"type": "image"}
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def preprocess_images(
    processor: Any,
    images: Sequence[Image.Image],
    instruction: str,
) -> dict[str, torch.Tensor]:
    texts = [apply_embedding_template(processor, image, instruction) for image in images]
    values = processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors="pt",
    )
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Qwen3-VL processor did not return: {missing}")
    return {
        name: tensor
        for name, tensor in values.items()
        if torch.is_tensor(tensor) and name not in IGNORED_MODEL_INPUTS
    }


def validate_token_budgets(inputs: Mapping[str, torch.Tensor], settings: Any) -> None:
    lengths = inputs["attention_mask"].long().sum(dim=1)
    maximum_language = int(lengths.max().item())
    if maximum_language > settings.max_language_tokens:
        raise RuntimeError(
            f"language sequence length {maximum_language} exceeds max_language_tokens="
            f"{settings.max_language_tokens}. Shorten the instruction or lower the visual "
            "token budget; silent truncation is forbidden."
        )
    grids = inputs["image_grid_thw"].long()
    visual_counts = grids.prod(dim=-1)
    maximum_visual = int(visual_counts.max().item())
    if maximum_visual > settings.max_visual_tokens:
        raise RuntimeError(
            f"visual token count {maximum_visual} exceeds max_visual_tokens="
            f"{settings.max_visual_tokens}. Lower processor_max_pixels; silent crop, pooling, "
            "or token truncation is forbidden."
        )


def move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()
    }


def forward_base_hidden(
    model: nn.Module, inputs: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    # Qwen3VLForEmbedding wraps Qwen3VLModel as .model. Calling the base avoids
    # any task-specific wrapper while retaining the official final RMSNorm.
    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "visual"):
        core = model
    outputs = core(
        **inputs,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(model, "_awa2_retrieval_optical_last_hidden", None)
    if hidden is None or hidden.ndim != 3:
        raise RuntimeError("Qwen base forward did not expose [B,S,D] last_hidden_state")
    return hidden


@torch.no_grad()
def teacher_embeddings(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    embedding_dim: int,
) -> torch.Tensor:
    model.eval()
    hidden = forward_base_hidden(model, inputs)
    return official_mrl_embedding(hidden, inputs["attention_mask"], embedding_dim)


def student_embeddings(
    model: nn.Module,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    replacement.use_student()
    replacement.prepare_student_batch(inputs["attention_mask"])
    forward_base_hidden(model, inputs)
    detector_features = replacement.language_surrogate.retrieval_detector_features()
    if detector_features.ndim != 2:
        raise RuntimeError(
            f"Expected student detector features [B,D], got {tuple(detector_features.shape)}"
        )
    embedding = readout(detector_features)
    return embedding, detector_features
