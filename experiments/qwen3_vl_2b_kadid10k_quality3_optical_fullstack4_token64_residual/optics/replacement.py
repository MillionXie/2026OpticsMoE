from __future__ import annotations

import inspect
from typing import Any, Iterable
import torch
from torch import nn

from .stacks import LanguageBypass, LanguageOpticalStackSurrogate, VisionBypass, VisionOpticalStackSurrogate


def locate_visual(model: nn.Module) -> nn.Module:
    for candidate in (getattr(model, "visual", None), getattr(getattr(model, "model", None), "visual", None)):
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")


def locate_language(model: nn.Module) -> nn.Module:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL language decoder layers; tried model.model.language_model.layers, model.language_model.layers, and model.model.layers")


class FullStackReplacement:
    def __init__(self, model: nn.Module, vision: VisionOpticalStackSurrogate, language: LanguageOpticalStackSurrogate) -> None:
        self.model = model; self.visual = locate_visual(model); self.language_model = locate_language(model)
        self.vision_blocks = self.visual.blocks; self.language_layers = self.language_model.layers
        if not len(self.vision_blocks) or not len(self.language_layers):
            raise RuntimeError("Vision and language stacks must be non-empty")
        self.original_vision = list(self.vision_blocks); self.original_language = list(self.language_layers)
        self.vision_surrogate = vision; self.language_surrogate = language
        self.vision_bypasses = [VisionBypass() for _ in range(len(self.vision_blocks)-1)]
        tuple_mode = _decoder_returns_tuple(self.original_language[0])
        self.language_surrogate.return_tuple = tuple_mode
        self.language_bypasses = [LanguageBypass(tuple_mode) for _ in range(len(self.language_layers)-1)]
        self.teacher_vision_output: torch.Tensor | None = None
        self.teacher_cu_seqlens: torch.Tensor | None = None
        self.last_language_hidden: torch.Tensor | None = None
        self._handles = [
            self.original_vision[0].register_forward_pre_hook(self._capture_vision_input, with_kwargs=True),
            self.original_vision[-1].register_forward_hook(self._capture_vision_output),
            self.language_model.norm.register_forward_hook(self._capture_language_norm_output),
        ]

    def _capture_vision_input(self, _module: nn.Module, args: tuple[Any,...], kwargs: dict[str,Any]) -> None:
        value = kwargs.get("cu_seqlens")
        if value is None and len(args)>1 and torch.is_tensor(args[1]): value=args[1]
        self.teacher_cu_seqlens = value.detach() if value is not None else None

    def _capture_vision_output(self, _module: nn.Module, _args: tuple[Any,...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        self.teacher_vision_output = value.detach()

    def _capture_language_norm_output(self, _module: nn.Module, _args: tuple[Any,...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(value):
            raise RuntimeError("Qwen final language norm did not return a tensor")
        # Do not detach: student CE/KD/answer losses must backpropagate through
        # the frozen final norm into the language and vision optical stacks.
        self.last_language_hidden = value
        setattr(self.model, "_optical_fullstack_last_hidden", value)

    def use_teacher(self) -> None:
        for i, layer in enumerate(self.original_vision): self.vision_blocks[i] = layer
        for i, layer in enumerate(self.original_language): self.language_layers[i] = layer

    def use_student(self) -> None:
        self.vision_blocks[0] = self.vision_surrogate
        for i, bypass in enumerate(self.vision_bypasses, start=1): self.vision_blocks[i] = bypass
        self.language_layers[0] = self.language_surrogate
        for i, bypass in enumerate(self.language_bypasses, start=1): self.language_layers[i] = bypass

    def prepare_student_batch(self, attention_mask: torch.Tensor) -> None:
        self.language_surrogate.set_attention_mask(attention_mask)

    def teacher_token_counts(self) -> list[int]:
        if self.teacher_cu_seqlens is None:
            raise RuntimeError("Teacher vision stack did not expose cu_seqlens")
        boundaries=self.teacher_cu_seqlens.cpu().long().tolist()
        return [b-a for a,b in zip(boundaries[:-1],boundaries[1:])]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.vision_surrogate.parameters(); yield from self.language_surrogate.parameters()

    def close(self) -> None:
        self.use_teacher()
        for handle in self._handles: handle.remove()
        if hasattr(self.model, "_optical_fullstack_last_hidden"):
            delattr(self.model, "_optical_fullstack_last_hidden")


def _decoder_returns_tuple(layer: nn.Module) -> bool:
    try:
        source = inspect.getsource(inspect.unwrap(layer.forward))
    except (OSError, TypeError):
        return False
    return (
        "return (hidden_states" in source
        or "return hidden_states," in source
        or "outputs = (hidden_states," in source
    )
