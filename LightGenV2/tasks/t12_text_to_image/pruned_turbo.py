"""Structured attention pruning for the one-call compact Turbo UNet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn


DEFAULT_PRUNE_SPEC = (
    {"side": "up", "block": 0, "attention": 1},
)


class AttentionBypass(nn.Module):
    """Diffusers-compatible identity for a removed Transformer2DModel."""

    def forward(
        self, hidden_states: torch.Tensor, *args, return_dict: bool = True, **kwargs
    ):
        del args, kwargs
        if return_dict:
            return type("Transformer2DModelOutput", (), {"sample": hidden_states})()
        return (hidden_states,)


def apply_attention_pruning(
    unet: nn.Module, spec: tuple[dict[str, Any], ...] = DEFAULT_PRUNE_SPEC
) -> dict[str, Any]:
    """Replace complete attention modules while preserving every tensor shape."""

    removed = 0
    applied = []
    for item in spec:
        side = str(item["side"])
        block_index = int(item["block"])
        attention_index = int(item["attention"])
        if side not in {"down", "up"}:
            raise ValueError(f"Unsupported UNet side: {side}")
        blocks = unet.down_blocks if side == "down" else unet.up_blocks
        attentions = blocks[block_index].attentions
        module = attentions[attention_index]
        if isinstance(module, AttentionBypass):
            raise ValueError(f"Attention is already pruned: {item}")
        parameters = sum(parameter.numel() for parameter in module.parameters())
        attentions[attention_index] = AttentionBypass()
        removed += parameters
        applied.append({**item, "removed_parameters": parameters})
    remaining = sum(parameter.numel() for parameter in unet.parameters())
    return {
        "spec": applied,
        "removed_parameters": removed,
        "remaining_parameters": remaining,
    }


def save_pruned_unet(
    unet: nn.Module,
    output_dir: Path,
    *,
    base_unet: Path,
    prune_report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().half().contiguous()
        for name, value in unet.state_dict().items()
    }
    save_file(state, output_dir / "model.fp16.safetensors")
    manifest = {
        "schema_version": 1,
        "variant": "qwen_bksdm_v2_tiny_pruned_deep_up_attention",
        "base_unet": str(base_unet),
        "pruning": prune_report,
        **metadata,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_pruned_unet(
    checkpoint_dir: Path, device: torch.device, *, dtype: torch.dtype = torch.float16
):
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        manifest["base_unet"], variant="fp16", torch_dtype=dtype, local_files_only=True
    )
    spec = tuple(
        {key: item[key] for key in ("side", "block", "attention")}
        for item in manifest["pruning"]["spec"]
    )
    apply_attention_pruning(unet, spec)
    state = load_file(checkpoint_dir / "model.fp16.safetensors", device="cpu")
    unet.load_state_dict(state, strict=True)
    return unet.to(device).eval(), manifest


__all__ = [
    "AttentionBypass",
    "DEFAULT_PRUNE_SPEC",
    "apply_attention_pruning",
    "load_pruned_unet",
    "save_pruned_unet",
]
