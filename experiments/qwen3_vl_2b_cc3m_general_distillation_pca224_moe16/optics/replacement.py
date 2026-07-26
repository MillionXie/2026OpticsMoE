from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .moe import LanguagePCAOpticalMoE, VisionPCAOpticalMoE


def locate_visual(model: nn.Module) -> nn.Module:
    for candidate in (
        getattr(model, "visual", None),
        getattr(getattr(model, "model", None), "visual", None),
    ):
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")


def locate_language(model: nn.Module) -> nn.Module:
    for candidate in (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "norm"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL language decoder layers")


def _tensor(value: Any, name: str) -> torch.Tensor:
    value = value[0] if isinstance(value, tuple) else value
    if not torch.is_tensor(value):
        raise RuntimeError(f"{name} did not return a hidden-state tensor")
    return value


class TeacherTapCapture:
    """Capture stack inputs/taps in their original frozen Qwen dimensions."""

    def __init__(self, model: nn.Module, language_tap_indexes: tuple[int, ...]) -> None:
        self.model = model
        self.visual = locate_visual(model)
        self.language = locate_language(model)
        self.vision_blocks = self.visual.blocks
        self.language_layers = self.language.layers
        self.deepstack_indexes = tuple(int(value) for value in self.visual.deepstack_visual_indexes)
        self.vision_tap_indexes = (*self.deepstack_indexes, len(self.vision_blocks) - 1)
        self.language_tap_indexes = tuple(int(value) for value in language_tap_indexes)
        if len(self.vision_tap_indexes) != 4 or len(self.language_tap_indexes) != 4:
            raise RuntimeError("Teacher capture requires exactly four vision and four language taps")
        self.vision_input: torch.Tensor | None = None
        self.vision_cu_seqlens: torch.Tensor | None = None
        self.vision_taps: dict[int, torch.Tensor] = {}
        self.language_input: torch.Tensor | None = None
        self.language_taps: dict[int, torch.Tensor] = {}
        self.final_language_hidden: torch.Tensor | None = None
        self._handles = [
            self.vision_blocks[0].register_forward_pre_hook(
                self._capture_vision_input, with_kwargs=True
            ),
            self.language_layers[0].register_forward_pre_hook(
                self._capture_language_input, with_kwargs=True
            ),
            self.language.norm.register_forward_hook(self._capture_final_norm),
        ]
        for index in self.vision_tap_indexes:
            self._handles.append(
                self.vision_blocks[index].register_forward_hook(
                    self._capture_output(self.vision_taps, index, "vision")
                )
            )
        for index in self.language_tap_indexes:
            self._handles.append(
                self.language_layers[index].register_forward_hook(
                    self._capture_output(self.language_taps, index, "language")
                )
            )

    def clear(self) -> None:
        self.vision_input = None
        self.vision_cu_seqlens = None
        self.vision_taps.clear()
        self.language_input = None
        self.language_taps.clear()
        self.final_language_hidden = None

    def _capture_vision_input(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        self.vision_input = _tensor(hidden, "vision stack input").detach()
        self.vision_cu_seqlens = cu.detach() if torch.is_tensor(cu) else None

    def _capture_language_input(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        self.language_input = _tensor(hidden, "language stack input").detach()

    @staticmethod
    def _capture_output(store: dict[int, torch.Tensor], index: int, stack: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            store[index] = _tensor(output, f"{stack} tap {index}").detach()

        return hook

    def _capture_final_norm(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        self.final_language_hidden = _tensor(output, "final language norm").detach()

    def vision_lengths(self) -> list[int]:
        if self.vision_input is None or self.vision_cu_seqlens is None:
            raise RuntimeError("Teacher forward did not expose vision input/cu_seqlens")
        boundaries = self.vision_cu_seqlens.cpu().long().tolist()
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        if sum(lengths) != len(self.vision_input):
            raise RuntimeError("Teacher vision cu_seqlens do not match packed hidden")
        return lengths

    def validate_complete(self) -> None:
        missing_vision = [i for i in self.vision_tap_indexes if i not in self.vision_taps]
        missing_language = [i for i in self.language_tap_indexes if i not in self.language_taps]
        if self.vision_input is None or self.language_input is None or missing_vision or missing_language:
            raise RuntimeError(
                f"Incomplete teacher taps: vision_missing={missing_vision}, "
                f"language_missing={missing_language}"
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


class VisionStartBlock(nn.Module):
    def __init__(self, surrogate: VisionPCAOpticalMoE) -> None:
        super().__init__()
        self.surrogate = surrogate

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        self.surrogate.compute(hidden_states, cu_seqlens)
        return hidden_states


class VisionTapBlock(nn.Module):
    def __init__(self, surrogate: VisionPCAOpticalMoE, slot: int) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.slot = int(slot)

    def forward(self, _hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.surrogate.output_for_slot(self.slot)


class Bypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class LanguageStageBlock(nn.Module):
    def __init__(self, surrogate: LanguagePCAOpticalMoE, stage: int) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.stage = int(stage)

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.surrogate.forward_stage(self.stage, hidden_states)


class PCAMultimodalReplacement:
    """Replace both full stacks while retaining native Qwen merger/injection/norm."""

    def __init__(
        self,
        model: nn.Module,
        vision: VisionPCAOpticalMoE,
        language: LanguagePCAOpticalMoE,
        settings: Any,
    ) -> None:
        self.model = model
        self.visual = locate_visual(model)
        self.language_model = locate_language(model)
        self.vision_blocks = self.visual.blocks
        self.language_layers = self.language_model.layers
        self.original_vision = list(self.vision_blocks)
        self.original_language = list(self.language_layers)
        self.vision_surrogate = vision
        self.language_surrogate = language
        self.deepstack_indexes = tuple(int(value) for value in self.visual.deepstack_visual_indexes)
        self.provider_indexes = (*self.deepstack_indexes, len(self.vision_blocks) - 1)
        self.language_tap_indexes = tuple(settings.language_tap_indexes)
        self.capture = TeacherTapCapture(model, self.language_tap_indexes)

        self.student_vision_modules: list[nn.Module] = [Bypass() for _ in self.vision_blocks]
        self.student_vision_modules[0] = VisionStartBlock(vision)
        for slot, index in enumerate(self.provider_indexes):
            self.student_vision_modules[index] = VisionTapBlock(vision, slot)
        self.student_language_modules: list[nn.Module] = [Bypass() for _ in self.language_layers]
        for stage in range(4):
            self.student_language_modules[stage] = LanguageStageBlock(language, stage)
        self.use_teacher()

    def use_teacher(self) -> None:
        for index, module in enumerate(self.original_vision):
            self.vision_blocks[index] = module
        for index, module in enumerate(self.original_language):
            self.language_layers[index] = module

    def use_student(self, *, vision: bool = True, language: bool = True) -> None:
        vision_modules = self.student_vision_modules if vision else self.original_vision
        language_modules = self.student_language_modules if language else self.original_language
        for index, module in enumerate(vision_modules):
            self.vision_blocks[index] = module
        for index, module in enumerate(language_modules):
            self.language_layers[index] = module

    def prepare_student_batch(self, attention_mask: torch.Tensor) -> None:
        self.language_surrogate.set_attention_mask(attention_mask)

    def trainable_parameters(self, mode: str) -> Iterable[nn.Parameter]:
        if mode in {"vision", "joint"}:
            yield from self.vision_surrogate.parameters()
        if mode in {"language", "joint"}:
            yield from self.language_surrogate.parameters()

    def router_losses(self) -> dict[str, torch.Tensor]:
        vision_balance, vision_importance = self.vision_surrogate.router_losses()
        language_balance, language_importance = self.language_surrogate.router_losses()
        return {
            "vision_balance": vision_balance,
            "vision_importance": vision_importance,
            "language_balance": language_balance,
            "language_importance": language_importance,
        }

    def close(self) -> None:
        self.use_teacher()
        self.capture.close()
