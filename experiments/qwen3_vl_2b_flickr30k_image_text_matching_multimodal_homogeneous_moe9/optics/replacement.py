from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .moe import LanguageDeepStackHomogeneousMoE, VisionDeepStackHomogeneousMoE


def locate_visual(model: nn.Module) -> nn.Module:
    for candidate in (getattr(model, "visual", None), getattr(getattr(model, "model", None), "visual", None)):
        if candidate is not None and hasattr(candidate, "blocks"): return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")


def locate_language(model: nn.Module) -> nn.Module:
    for candidate in (getattr(getattr(model, "model", None), "language_model", None),
                      getattr(model, "language_model", None), getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "norm"): return candidate
    raise RuntimeError("Unable to locate Qwen3-VL language decoder layers")


class VisionStartBlock(nn.Module):
    def __init__(self, surrogate: VisionDeepStackHomogeneousMoE) -> None:
        super().__init__(); self.surrogate = surrogate
    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        self.surrogate.compute(hidden_states, cu_seqlens); return hidden_states


class VisionTapBlock(nn.Module):
    def __init__(self, surrogate: VisionDeepStackHomogeneousMoE, slot: int) -> None:
        super().__init__(); self.surrogate = surrogate; self.slot = slot
    def forward(self, _hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return self.surrogate.output_for_slot(self.slot)


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return hidden_states


class LanguageStageBlock(nn.Module):
    def __init__(self, surrogate: LanguageDeepStackHomogeneousMoE, stage: int) -> None:
        super().__init__(); self.surrogate = surrogate; self.stage = stage
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.surrogate.forward_stage(self.stage, hidden_states)


class LanguageBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return hidden_states


class DeepStackMultimodalReplacement:
    """Preserve native Qwen DeepStack merger/injection timing while replacing selected stacks."""

    def __init__(self, model: nn.Module, vision: VisionDeepStackHomogeneousMoE,
                 language: LanguageDeepStackHomogeneousMoE, language_mode: str) -> None:
        self.model = model; self.visual = locate_visual(model); self.language_model = locate_language(model)
        self.vision_blocks = self.visual.blocks; self.language_layers = self.language_model.layers
        self.original_vision = list(self.vision_blocks); self.original_language = list(self.language_layers)
        self.vision_surrogate = vision; self.language_surrogate = language; self.language_mode = language_mode
        self.deepstack_indexes = tuple(int(value) for value in self.visual.deepstack_visual_indexes)
        if len(self.deepstack_indexes) != 3: raise RuntimeError(f"Expected 3 DeepStack indexes, got {self.deepstack_indexes}")
        self.language_surrogate.set_deepstack_injection_count(len(self.deepstack_indexes))
        final_index = len(self.vision_blocks) - 1; provider_indexes = (*self.deepstack_indexes, final_index)
        if len(set(provider_indexes)) != 4: raise RuntimeError("DeepStack indexes overlap final vision block")
        self.student_vision_modules: list[nn.Module] = [VisionBypass() for _ in self.vision_blocks]
        self.student_vision_modules[0] = VisionStartBlock(vision)
        for slot, index in enumerate(provider_indexes): self.student_vision_modules[index] = VisionTapBlock(vision, slot)
        self.student_language_modules: list[nn.Module] = [LanguageBypass() for _ in self.language_layers]
        for stage in range(len(language.core.expert_layers)):
            self.student_language_modules[stage] = LanguageStageBlock(language, stage)
        self.teacher_cu_seqlens: torch.Tensor | None = None; self.teacher_vision_taps: dict[int, torch.Tensor] = {}
        self.last_language_hidden: torch.Tensor | None = None
        self._handles = [self.original_vision[0].register_forward_pre_hook(self._capture_cu, with_kwargs=True),
                         self.language_model.norm.register_forward_hook(self._capture_language)]
        for index in provider_indexes:
            self._handles.append(self.original_vision[index].register_forward_hook(self._capture_tap(index)))

    def _capture_cu(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        value = kwargs.get("cu_seqlens")
        if value is None and len(args) > 1 and torch.is_tensor(args[1]): value = args[1]
        self.teacher_cu_seqlens = value.detach() if value is not None else None

    def _capture_tap(self, index: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            self.teacher_vision_taps[index] = value.detach()
        return hook

    def _capture_language(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(value): raise RuntimeError("Qwen final language norm did not return a tensor")
        self.last_language_hidden = value; setattr(self.model, "_flickr30k_itm_optical_last_hidden", value)

    def use_teacher(self) -> None:
        for index, layer in enumerate(self.original_vision): self.vision_blocks[index] = layer
        for index, layer in enumerate(self.original_language): self.language_layers[index] = layer

    def use_student(self) -> None:
        for index, layer in enumerate(self.student_vision_modules): self.vision_blocks[index] = layer
        layers = self.student_language_modules if self.language_mode == "optical_moe" else self.original_language
        for index, layer in enumerate(layers): self.language_layers[index] = layer

    def prepare_student_batch(self, attention_mask: torch.Tensor) -> None:
        if self.language_mode == "optical_moe": self.language_surrogate.set_attention_mask(attention_mask)

    def teacher_token_counts(self) -> list[int]:
        if self.teacher_cu_seqlens is None: raise RuntimeError("Teacher vision did not expose cu_seqlens")
        boundaries = self.teacher_cu_seqlens.cpu().long().tolist()
        return [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.vision_surrogate.parameters()
        if self.language_mode == "optical_moe": yield from self.language_surrogate.parameters()

    def router_losses(self) -> dict[str, torch.Tensor]:
        vb, vi = self.vision_surrogate.router_losses()
        if self.language_mode == "optical_moe": lb, li = self.language_surrogate.router_losses()
        else:
            lb = vb.new_zeros(()); li = vb.new_zeros(())
        return {"vision_balance": vb, "vision_importance": vi,
                "language_balance": lb, "language_importance": li}

    def set_phase_dropout_active(self, active: bool) -> None:
        self.vision_surrogate.set_phase_dropout_active(active)
        if self.language_mode == "optical_moe": self.language_surrogate.set_phase_dropout_active(active)

    def close(self) -> None:
        self.use_teacher()
        for handle in self._handles: handle.remove()
        if hasattr(self.model, "_flickr30k_itm_optical_last_hidden"): delattr(self.model, "_flickr30k_itm_optical_last_hidden")
