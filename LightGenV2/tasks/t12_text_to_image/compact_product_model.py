"""Sub-500M product editor components with an identity-residual optical mid block."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import CompactFourierOptics, ScaleMatchedFusion
from .product_repair_model import RepairModelConfig
from .pruned_turbo import apply_attention_pruning


# Retain text attention only at the deepest down block and the first attention
# of the deepest up block.  This removes 72.095M duplicated attention weights
# from BK-SDM-v2-Tiny while preserving its convolutional image capacity.
COMPACT_ATTENTION_PRUNE_SPEC = (
    {"side": "down", "block": 0, "attention": 0},
    {"side": "down", "block": 1, "attention": 0},
    {"side": "up", "block": 0, "attention": 1},
    {"side": "up", "block": 1, "attention": 0},
    {"side": "up", "block": 1, "attention": 1},
    {"side": "up", "block": 2, "attention": 0},
    {"side": "up", "block": 2, "attention": 1},
)


class IdentityResidualOpticalMidBlock(nn.Module):
    """Fill BK-SDM's deleted mid residual transform with physical optics.

    The electronic branch is exactly the identity tensor.  It has no learned
    transform and is excluded from the hardware latency budget.  The optical
    branch consumes the same input and is RMS-fused with that residual.
    """

    has_cross_attention = True

    def __init__(
        self,
        *,
        channels: int,
        timestep_dim: int,
        condition_dim: int,
        config: RepairModelConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.channels = int(channels)
        self.grid = int(config.optical_grid)
        width = int(config.optical_width)
        self.input_norm = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, width)
        self.timestep_projection = nn.Linear(timestep_dim, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.optical = CompactFourierOptics(
            width, self.grid, experts=config.optical_experts, top_k=config.optical_top_k
        )
        self.expert_gate = nn.Parameter(torch.tensor(-1.25))
        self.global_gate = nn.Parameter(torch.tensor(-1.25))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, channels)
        self.output_gate = nn.Parameter(torch.tensor(-1.5))
        self.fusion = ScaleMatchedFusion(
            config.alpha_initial, config.alpha_minimum,
            config.alpha_maximum, config.rms_epsilon,
        )
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def optical_parameters(self):
        yield from self.named_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, cross_attention_kwargs, encoder_attention_mask
        if temb is None or encoder_hidden_states is None:
            raise ValueError("Optical mid block requires timestep and text conditions")
        batch, channels, height, width = hidden_states.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        pooled = F.adaptive_avg_pool2d(hidden_states.float(), (self.grid, self.grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.input_projection(self.input_norm(tokens))
        tokens = tokens + self.timestep_projection(temb.float())[:, None]
        text = encoder_hidden_states.float().mean(dim=1)
        tokens = F.silu(tokens + self.condition_projection(text)[:, None])
        tokens = tokens + torch.sigmoid(self.expert_gate) * self.optical.expert(tokens)
        tokens = tokens + torch.sigmoid(self.global_gate) * self.optical.global_block(tokens)
        delta = self.output_projection(self.output_norm(tokens))
        delta = delta.transpose(1, 2).reshape(batch, channels, self.grid, self.grid)
        delta = F.interpolate(delta, (height, width), mode="bilinear", align_corners=False)
        optical = hidden_states.float() + torch.sigmoid(self.output_gate) * delta
        return self.fusion(hidden_states, optical.to(hidden_states.dtype))

    def architecture_report(self) -> dict[str, Any]:
        return {
            "location": "deleted BK-SDM mid residual transform at decoder entrance",
            "parallel_contract": "identity residual(shared_input) || optical(shared_input), then RMS fusion",
            "electronic_and_optical_are_parallel": True,
            "electronic_branch": "parameter-free identity residual; excluded from latency",
            "optical_backend": "differentiable phase-only FFT simulation",
            "optical_grid": [self.grid, self.grid],
            "optical_parameters": sum(p.numel() for p in self.parameters()),
            "alpha": float(self.fusion.alpha.detach()),
            "alpha_minimum": self.fusion.minimum,
            "alpha_maximum": self.fusion.maximum,
            "physical_latency_ms": 1.0447 * 6,
        }


def prepare_compact_optical_unet(
    unet: nn.Module, config: RepairModelConfig,
) -> tuple[IdentityResidualOpticalMidBlock, dict[str, Any]]:
    if unet.mid_block is not None:
        raise ValueError("Compact optical contract expects BK-SDM with a deleted mid block")
    pruning = apply_attention_pruning(unet, COMPACT_ATTENTION_PRUNE_SPEC)
    optical = IdentityResidualOpticalMidBlock(
        channels=int(unet.config.block_out_channels[-1]),
        timestep_dim=int(unet.time_embedding.linear_2.out_features),
        condition_dim=int(unet.config.cross_attention_dim),
        config=config,
    )
    reference = next(unet.parameters())
    optical.to(device=reference.device, dtype=reference.dtype)
    unet.mid_block = optical
    pruning["parameters_after_optical"] = sum(p.numel() for p in unet.parameters())
    return optical, pruning


def load_legacy_scene_warm_start(unet: nn.Module, checkpoint: Path) -> dict[str, Any]:
    """Load the learned electronic editor while dropping its old optical wrapper."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    converted: dict[str, torch.Tensor] = {}
    for name, value in payload["unet"].items():
        if name.startswith("up_blocks.0.electronic."):
            converted[name.replace("up_blocks.0.electronic.", "up_blocks.0.", 1)] = value
        elif name.startswith("up_blocks.0."):
            continue
        else:
            converted[name] = value
    result = unet.load_state_dict(converted, strict=False)
    return {
        "loaded_keys": len(converted),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "adapter": payload.get("adapter"),
    }


__all__ = [
    "COMPACT_ATTENTION_PRUNE_SPEC", "IdentityResidualOpticalMidBlock",
    "load_legacy_scene_warm_start", "prepare_compact_optical_unet",
]
