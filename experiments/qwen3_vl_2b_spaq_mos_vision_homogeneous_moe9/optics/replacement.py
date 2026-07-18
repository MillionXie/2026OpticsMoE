from __future__ import annotations

from typing import Any

import torch
from torch import nn


def locate_visual(model: nn.Module) -> nn.Module:
    for candidate in (getattr(model, "visual", None), getattr(getattr(model, "model", None), "visual", None)):
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class VisionStackReplacement:
    def __init__(self, model: nn.Module, surrogate: nn.Module) -> None:
        self.model = model
        self.visual = locate_visual(model)
        self.blocks = self.visual.blocks
        if not len(self.blocks):
            raise RuntimeError("Qwen vision stack is empty")
        self.original = list(self.blocks)
        self.surrogate = surrogate
        self.bypasses = [VisionBypass() for _ in range(len(self.blocks) - 1)]
        self.teacher_output: torch.Tensor | None = None
        self.teacher_cu_seqlens: torch.Tensor | None = None
        self._handles = [
            self.original[0].register_forward_pre_hook(self._capture_input, with_kwargs=True),
            self.original[-1].register_forward_hook(self._capture_output),
        ]

    def _capture_input(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        value = kwargs.get("cu_seqlens")
        if value is None and len(args) > 1 and torch.is_tensor(args[1]):
            value = args[1]
        self.teacher_cu_seqlens = value.detach() if value is not None else None

    def _capture_output(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        self.teacher_output = value.detach()

    def use_teacher(self) -> None:
        for index, layer in enumerate(self.original):
            self.blocks[index] = layer

    def use_student(self) -> None:
        self.blocks[0] = self.surrogate
        for index, bypass in enumerate(self.bypasses, start=1):
            self.blocks[index] = bypass

    def teacher_token_counts(self) -> list[int]:
        if self.teacher_cu_seqlens is None:
            raise RuntimeError("Teacher vision stack did not expose cu_seqlens")
        boundaries = self.teacher_cu_seqlens.cpu().long().tolist()
        return [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]

    def close(self) -> None:
        self.use_teacher()
        for handle in self._handles:
            handle.remove()
