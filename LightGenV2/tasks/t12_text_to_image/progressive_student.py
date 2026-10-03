"""Progressively narrowed one-pass optical UNets for sub-300M editors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import torch
from torch import nn

from .compact_product_model import prepare_compact_optical_unet
from .product_repair_model import RepairModelConfig


PROGRESSIVE_WIDTHS = (
    (256, 512, 1024),
    (224, 448, 896),
    (192, 384, 768),
)


def _attention_heads(widths: Iterable[int]) -> tuple[int, ...]:
    """Keep approximately 64 channels per attention head."""

    return tuple(max(1, int(width) // 64) for width in widths)


def build_narrow_optical_unet(
    base_config: dict[str, Any], widths: Iterable[int], model_config: RepairModelConfig,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Construct the same BK-SDM topology with structured channel pruning."""

    from diffusers import UNet2DConditionModel

    selected = tuple(int(value) for value in widths)
    if len(selected) != 3 or any(value <= 0 or value % 32 for value in selected):
        raise ValueError("Narrow UNet widths must contain three positive multiples of 32")
    config = deepcopy(dict(base_config))
    config.update({
        "block_out_channels": list(selected),
        "attention_head_dim": list(_attention_heads(selected)),
        "in_channels": 8,
    })
    unet = UNet2DConditionModel.from_config(config)
    optical, pruning = prepare_compact_optical_unet(unet, model_config)
    pruning.update({
        "structured_widths": list(selected),
        "original_widths": list(base_config["block_out_channels"]),
        "width_fraction": selected[-1] / int(base_config["block_out_channels"][-1]),
    })
    return unet, optical, pruning


def copy_overlapping_state(
    module: nn.Module, source: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Warm-start a narrower network by copying every overlapping tensor slab.

    This is deterministic structured channel pruning: the student keeps a
    contiguous, shape-compatible subspace of every teacher tensor instead of
    being randomly reinitialized.  Subsequent local distillation repairs the
    channel interactions changed by the narrower topology.
    """

    target = module.state_dict()
    copied, exact, partial, missing = 0, 0, 0, []
    with torch.no_grad():
        for name, destination in target.items():
            value = source.get(name)
            if value is None or value.ndim != destination.ndim:
                missing.append(name)
                continue
            if value.shape == destination.shape:
                destination.copy_(value.to(dtype=destination.dtype))
                exact += destination.numel()
                copied += destination.numel()
                continue
            slices = tuple(slice(0, min(left, right)) for left, right in zip(destination.shape, value.shape))
            destination[slices].copy_(value[slices].to(dtype=destination.dtype))
            count = 1
            for piece in slices:
                count *= int(piece.stop)
            partial += count
            copied += count
    module.load_state_dict(target)
    return {
        "target_parameters": sum(tensor.numel() for tensor in target.values()),
        "copied_values": copied,
        "exact_values": exact,
        "partial_values": partial,
        "coverage": copied / max(1, sum(tensor.numel() for tensor in target.values())),
        "missing_keys": missing,
    }


def counted_student_parameters(
    *, unet: nn.Module, adapter: nn.Module, router: nn.Module,
    qwen_counted: int, vae_encoder: int, vae_decoder: int,
) -> dict[str, int]:
    result = {
        "qwen_transformer_and_norm": int(qwen_counted),
        "vae_encoder": int(vae_encoder),
        "unet_including_optics": sum(parameter.numel() for parameter in unet.parameters()),
        "adapter": sum(parameter.numel() for parameter in adapter.parameters()),
        "router": sum(parameter.numel() for parameter in router.parameters()),
        "vae_decoder": int(vae_decoder),
    }
    result["total"] = sum(result.values())
    return result


__all__ = [
    "PROGRESSIVE_WIDTHS", "build_narrow_optical_unet",
    "copy_overlapping_state", "counted_student_parameters",
]
