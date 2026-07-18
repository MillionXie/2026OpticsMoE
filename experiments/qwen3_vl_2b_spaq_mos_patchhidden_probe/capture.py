from __future__ import annotations

from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.optics.replacement import locate_visual


class PatchHiddenCapture(nn.Module):
    """Identity vision block that records the frozen patch-embedding hidden."""

    def __init__(self) -> None:
        super().__init__()
        self.last_hidden: torch.Tensor | None = None
        self.last_token_counts: list[int] = []

    def forward(self, hidden_states: torch.Tensor, *args: Any,
                cu_seqlens: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        if cu_seqlens is None:
            cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and args and torch.is_tensor(args[0]):
            cu_seqlens = args[0]
        if cu_seqlens is None:
            raise RuntimeError("Qwen visual block did not provide cu_seqlens")
        boundaries = cu_seqlens.detach().cpu().long().tolist()
        self.last_token_counts = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        if sum(self.last_token_counts) != int(hidden_states.shape[0]):
            raise RuntimeError("cu_seqlens do not match patch-hidden rows")
        self.last_hidden = hidden_states.detach()
        return hidden_states


class VisionPatchBypass:
    """Temporarily bypass every vision transformer block while capturing its input."""

    def __init__(self, model: nn.Module) -> None:
        self.visual = locate_visual(model)
        self.blocks = self.visual.blocks
        self.original = list(self.blocks)
        self.capture = PatchHiddenCapture()
        self.activate()

    def activate(self) -> None:
        self.blocks[0] = self.capture
        for index in range(1, len(self.blocks)):
            self.blocks[index] = _IdentityVisionBlock()

    def close(self) -> None:
        for index, block in enumerate(self.original):
            self.blocks[index] = block


class _IdentityVisionBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return hidden_states
