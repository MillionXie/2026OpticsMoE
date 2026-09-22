"""Text-only, half-depth Qwen3-VL encoder used by the image editor.

The generator never consumes image tokens from Qwen.  We therefore detach the
language model from the vision tower and LM head, and retain only the first
half of its Transformer blocks.  This makes the architectural claim explicit
instead of merely skipping work after a full Qwen forward pass.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _language_model(full_model: nn.Module) -> nn.Module:
    core = getattr(full_model, "model", None)
    language = getattr(core, "language_model", None)
    if language is None or not hasattr(language, "layers"):
        raise TypeError("Expected model.model.language_model.layers in Qwen3-VL")
    return language


def retain_language_layers(language: nn.Module, keep_layers: int) -> dict[str, Any]:
    """Physically remove all Qwen language blocks after ``keep_layers``."""

    layers = language.layers
    original = len(layers)
    if not 0 < keep_layers <= original:
        raise ValueError(f"keep_layers must be in [1, {original}], got {keep_layers}")
    layer_parameters = [sum(p.numel() for p in layer.parameters()) for layer in layers]
    language.layers = nn.ModuleList(list(layers[:keep_layers]))
    if hasattr(language, "config") and hasattr(language.config, "num_hidden_layers"):
        language.config.num_hidden_layers = keep_layers
    return {
        "language_layers_original": original,
        "language_layers_retained": keep_layers,
        "language_transformer_parameters_original": sum(layer_parameters),
        "language_transformer_parameters_retained": sum(layer_parameters[:keep_layers]),
        "language_transformer_parameters_removed": sum(layer_parameters[keep_layers:]),
        "language_transformer_depth_fraction": keep_layers / original,
    }


def load_half_qwen_text_encoder(
    checkpoint: Path,
    device: torch.device,
    *,
    keep_layers: int = 14,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load Qwen, keep its text encoder only, and truncate 28 blocks to 14."""

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    full_parameters = sum(p.numel() for p in full.parameters())
    language = _language_model(full)
    report = retain_language_layers(language, keep_layers)
    retained_parameters = sum(p.numel() for p in language.parameters())
    report.update({
        "checkpoint_parameters": full_parameters,
        "retained_text_encoder_parameters": retained_parameters,
        "vision_tower_used": False,
        "language_model_head_used": False,
        "feature_pooling": "attention-mask-weighted mean of retained layer 14 output",
    })

    # A child module has no parent pointer.  Keeping this reference while
    # deleting the full model releases both the vision tower and the LM head.
    del full
    gc.collect()
    language = language.to(device).eval().requires_grad_(False)
    return language, processor, report


__all__ = ["load_half_qwen_text_encoder", "retain_language_layers"]
