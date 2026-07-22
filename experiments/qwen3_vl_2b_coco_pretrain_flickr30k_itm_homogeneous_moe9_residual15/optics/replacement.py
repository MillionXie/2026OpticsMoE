from __future__ import annotations

import copy
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


def _tensor_output(value: Any, module_name: str) -> torch.Tensor:
    output = value[0] if isinstance(value, tuple) else value
    if not torch.is_tensor(output):
        raise RuntimeError(f"{module_name} did not return a hidden-state tensor")
    return output


class VisionNativeAttentionPrelude(nn.Module):
    """One copied native Qwen vision attention sub-layer, without its electronic MLP."""

    def __init__(self, source_block: nn.Module) -> None:
        super().__init__()
        for name in ("norm1", "attn", "norm2"):
            if not hasattr(source_block, name):
                raise RuntimeError(f"Qwen vision source block has no {name}; cannot build Transformer-aligned prelude")
        self.norm1 = copy.deepcopy(source_block.norm1)
        self.attn = copy.deepcopy(source_block.attn)
        self.norm2 = copy.deepcopy(source_block.norm2)
        self.last_attention_output: torch.Tensor | None = None
        self.last_residual_base: torch.Tensor | None = None
        self.last_optical_input: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        attention_kwargs = dict(kwargs)
        if cu_seqlens is not None:
            attention_kwargs["cu_seqlens"] = cu_seqlens
        attention = _tensor_output(self.attn(self.norm1(hidden_states), **attention_kwargs), "vision attention")
        residual_base = hidden_states + attention
        optical_input = self.norm2(residual_base)
        self.last_attention_output = attention
        self.last_residual_base = residual_base
        self.last_optical_input = optical_input
        return residual_base, optical_input


class LanguageNativeAttentionPrelude(nn.Module):
    """One copied native Qwen decoder attention sub-layer, preserving masks and RoPE kwargs."""

    def __init__(self, source_layer: nn.Module) -> None:
        super().__init__()
        for name in ("input_layernorm", "self_attn", "post_attention_layernorm"):
            if not hasattr(source_layer, name):
                raise RuntimeError(f"Qwen language source layer has no {name}; cannot build Transformer-aligned prelude")
        self.input_layernorm = copy.deepcopy(source_layer.input_layernorm)
        self.self_attn = copy.deepcopy(source_layer.self_attn)
        self.post_attention_layernorm = copy.deepcopy(source_layer.post_attention_layernorm)
        self.last_attention_output: torch.Tensor | None = None
        self.last_residual_base: torch.Tensor | None = None
        self.last_optical_input: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        attention = _tensor_output(
            self.self_attn(hidden_states=self.input_layernorm(hidden_states), **kwargs),
            "language attention",
        )
        residual_base = hidden_states + attention
        optical_input = self.post_attention_layernorm(residual_base)
        self.last_attention_output = attention
        self.last_residual_base = residual_base
        self.last_optical_input = optical_input
        return residual_base, optical_input


class VisionNativeNormPrelude(nn.Module):
    """Frozen pre-FFN normalization used by X + OpticalMoE(Norm(X))."""

    def __init__(self, source_block: nn.Module) -> None:
        super().__init__()
        if not hasattr(source_block, "norm2"):
            raise RuntimeError("Qwen vision source block has no norm2; cannot build norm-only prelude")
        self.norm = copy.deepcopy(source_block.norm2).requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor, *_: Any, **__: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, self.norm(hidden_states)


class LanguageNativeNormPrelude(nn.Module):
    """Frozen pre-MLP normalization used by X + OpticalMoE(Norm(X))."""

    def __init__(self, source_layer: nn.Module) -> None:
        super().__init__()
        if not hasattr(source_layer, "post_attention_layernorm"):
            raise RuntimeError("Qwen language source layer lacks post_attention_layernorm")
        self.norm = copy.deepcopy(source_layer.post_attention_layernorm).requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, self.norm(hidden_states)


class VisionStartBlock(nn.Module):
    def __init__(self, surrogate: VisionDeepStackHomogeneousMoE,
                 pre_attention: VisionNativeAttentionPrelude | None,
                 residual_enabled: bool) -> None:
        super().__init__(); self.surrogate = surrogate; self.pre_attention = pre_attention
        self.residual_enabled = bool(residual_enabled)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        if self.pre_attention is None:
            residual_base, optical_input = hidden_states, hidden_states
        else:
            residual_base, optical_input = self.pre_attention(hidden_states, cu_seqlens, **kwargs)
        self.surrogate.compute(
            optical_input,
            cu_seqlens,
            residual_base=residual_base if self.residual_enabled else None,
        )
        return hidden_states


class VisionTapBlock(nn.Module):
    def __init__(self, surrogate: VisionDeepStackHomogeneousMoE, slot: int) -> None:
        super().__init__(); self.surrogate = surrogate; self.slot = slot
    def forward(self, _hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return self.surrogate.output_for_slot(self.slot)


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return hidden_states


class LanguageStageBlock(nn.Module):
    def __init__(self, surrogate: LanguageDeepStackHomogeneousMoE, stage: int,
                 pre_attention: LanguageNativeAttentionPrelude | None = None,
                 residual_enabled: bool = False) -> None:
        super().__init__(); self.surrogate = surrogate; self.stage = stage; self.pre_attention = pre_attention
        self.residual_enabled = bool(residual_enabled)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.stage != 0:
            return self.surrogate.forward_stage(self.stage, hidden_states)
        if self.pre_attention is None:
            residual_base, optical_input = hidden_states, hidden_states
        else:
            residual_base, optical_input = self.pre_attention(hidden_states, **kwargs)
        return self.surrogate.forward_stage(
            self.stage,
            hidden_states,
            optical_input=optical_input,
            residual_base=residual_base if self.residual_enabled else None,
        )


class LanguageBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor: return hidden_states


class DeepStackMultimodalReplacement:
    """Preserve native Qwen DeepStack merger/injection timing while replacing selected stacks."""

    def __init__(self, model: nn.Module, vision: VisionDeepStackHomogeneousMoE,
                 language: LanguageDeepStackHomogeneousMoE, settings: Any) -> None:
        self.model = model; self.visual = locate_visual(model); self.language_model = locate_language(model)
        self.vision_blocks = self.visual.blocks; self.language_layers = self.language_model.layers
        self.original_vision = list(self.vision_blocks); self.original_language = list(self.language_layers)
        self.vision_surrogate = vision; self.language_surrogate = language
        self.language_mode = settings.student_language_mode
        self.native_pre_attention_enabled = bool(settings.native_pre_attention_enabled)
        self.native_pre_norm_enabled = bool(settings.native_pre_norm_enabled)
        self.native_pre_attention_trainable = bool(settings.native_pre_attention_trainable)
        self.transformer_residual_enabled = bool(settings.transformer_residual_enabled)
        self.vision_attention_source_layer = int(settings.vision_attention_source_layer)
        self.language_attention_source_layer = int(settings.language_attention_source_layer)
        vision_source = self.original_vision[self.vision_attention_source_layer]
        language_source = self.original_language[self.language_attention_source_layer]
        self.vision_pre_attention = (VisionNativeAttentionPrelude(vision_source)
                                     if self.native_pre_attention_enabled else
                                     VisionNativeNormPrelude(vision_source) if self.native_pre_norm_enabled else None)
        self.language_pre_attention = (
            (LanguageNativeAttentionPrelude(language_source) if self.native_pre_attention_enabled
             else LanguageNativeNormPrelude(language_source) if self.native_pre_norm_enabled else None)
            if self.language_mode == "optical_moe" else None
        )
        for module in (self.vision_pre_attention, self.language_pre_attention):
            if module is not None:
                module.requires_grad_(self.native_pre_attention_trainable)
                module.train(self.native_pre_attention_trainable)
        self.deepstack_indexes = tuple(int(value) for value in self.visual.deepstack_visual_indexes)
        if len(self.deepstack_indexes) != 3: raise RuntimeError(f"Expected 3 DeepStack indexes, got {self.deepstack_indexes}")
        self.language_surrogate.set_deepstack_injection_count(len(self.deepstack_indexes))
        final_index = len(self.vision_blocks) - 1; provider_indexes = (*self.deepstack_indexes, final_index)
        if len(set(provider_indexes)) != 4: raise RuntimeError("DeepStack indexes overlap final vision block")
        self.student_vision_modules: list[nn.Module] = [VisionBypass() for _ in self.vision_blocks]
        self.student_vision_modules[0] = VisionStartBlock(
            vision, self.vision_pre_attention, self.transformer_residual_enabled
        )
        for slot, index in enumerate(provider_indexes): self.student_vision_modules[index] = VisionTapBlock(vision, slot)
        self.student_language_modules: list[nn.Module] = [LanguageBypass() for _ in self.language_layers]
        for stage in range(language.core.logical_stages):
            self.student_language_modules[stage] = LanguageStageBlock(
                language,
                stage,
                self.language_pre_attention if stage == 0 else None,
                self.transformer_residual_enabled,
            )
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
        if self.native_pre_attention_trainable:
            if self.vision_pre_attention is not None: yield from self.vision_pre_attention.parameters()
            if self.language_pre_attention is not None: yield from self.language_pre_attention.parameters()

    def configure_student_trainability(self) -> None:
        self.vision_surrogate.requires_grad_(True)
        if self.language_mode == "optical_moe": self.language_surrogate.requires_grad_(True)
        for module in (self.vision_pre_attention, self.language_pre_attention):
            if module is not None: module.requires_grad_(self.native_pre_attention_trainable)

    def set_student_train_mode(self) -> None:
        self.vision_surrogate.train()
        if self.language_mode == "optical_moe": self.language_surrogate.train()
        for module in (self.vision_pre_attention, self.language_pre_attention):
            if module is not None: module.train(self.native_pre_attention_trainable)

    @staticmethod
    def _module_parameters(module: nn.Module | None, trainable_only: bool = False) -> int:
        if module is None: return 0
        return sum(parameter.numel() for parameter in module.parameters()
                   if not trainable_only or parameter.requires_grad)

    def alignment_specification(self) -> dict[str, Any]:
        return {
            "native_pre_attention_enabled": self.native_pre_attention_enabled,
            "native_pre_norm_enabled": self.native_pre_norm_enabled,
            "native_pre_attention_trainable": self.native_pre_attention_trainable,
            "transformer_residual_enabled": self.transformer_residual_enabled,
            "residual_identity_scale": 1.0,
            "residual_identity_scale_trainable": False,
            "vision_attention_source_layer": self.vision_attention_source_layer,
            "language_attention_source_layer": self.language_attention_source_layer,
            "vision_attention_parameters": self._module_parameters(self.vision_pre_attention),
            "vision_attention_trainable_parameters": self._module_parameters(self.vision_pre_attention, True),
            "language_attention_parameters": self._module_parameters(self.language_pre_attention),
            "language_attention_trainable_parameters": self._module_parameters(self.language_pre_attention, True),
            "prelude_type": ("native_attention_and_norm" if self.native_pre_attention_enabled else
                             "frozen_norm_only" if self.native_pre_norm_enabled else "none"),
            "vision_prelude_parameters": self._module_parameters(self.vision_pre_attention),
            "vision_prelude_trainable_parameters": self._module_parameters(self.vision_pre_attention, True),
            "language_prelude_parameters": self._module_parameters(self.language_pre_attention),
            "language_prelude_trainable_parameters": self._module_parameters(self.language_pre_attention, True),
            "post_residual_activation": False,
            "logical_optical_stages": self.vision_surrogate.core.logical_stages,
            "physical_layers_per_logical_stage": self.vision_surrogate.core.physical_layers_per_logical_stage,
            "total_physical_layers": self.vision_surrogate.core.total_physical_layers,
            "equation": ("A = X + Attention(Norm1(X)); Y = A + OpticalMoE(Norm2(A))"
                         if self.native_pre_attention_enabled else
                         "Y = X + OpticalMoE(Norm(X))" if self.native_pre_norm_enabled else
                         "Y = X + OpticalMoE(X)"),
        }

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
