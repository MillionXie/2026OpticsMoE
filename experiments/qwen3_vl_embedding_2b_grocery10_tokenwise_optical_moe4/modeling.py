from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    forward_base_hidden,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
    load_backbone,
    official_mrl_embedding,
)

from .optics import TokenwiseOpticalMoE


def locate_visual(model: nn.Module) -> nn.Module:
    for candidate in (
        getattr(model, "visual", None),
        getattr(getattr(model, "model", None), "visual", None),
    ):
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")


class TokenwiseVisionBlock(nn.Module):
    def __init__(self, core: TokenwiseOpticalMoE) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.core(hidden_states, cu_seqlens)


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class TokenwiseVisionReplacement:
    """Replace Qwen vision blocks with one adapter-free token-wise optical core.

    Qwen patch/position embedding, merger, all DeepStack mergers, and the full
    frozen electronic language model stay native. Since the optical core emits
    one spatial hidden tensor, Qwen's three native DeepStack taps observe the
    same tensor through bypass blocks rather than fabricated per-stage tensors.
    """

    def __init__(self, model: nn.Module, core: TokenwiseOpticalMoE) -> None:
        self.model = model
        self.visual = locate_visual(model)
        self.blocks = self.visual.blocks
        self.original_blocks = list(self.blocks)
        self.native_deepstack_indexes = tuple(
            int(v) for v in getattr(self.visual, "deepstack_visual_indexes", [])
        )
        self.vision_surrogate = core
        self.student_blocks: list[nn.Module] = [VisionBypass() for _ in self.blocks]
        self.student_blocks[0] = TokenwiseVisionBlock(core)
        self.configure_student_trainability()

    def use_teacher(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.blocks[index] = block

    def use_student(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.blocks[index] = block

    def configure_student_trainability(self) -> None:
        self.model.requires_grad_(False)
        self.vision_surrogate.requires_grad_(True)

    def set_student_train_mode(self) -> None:
        self.model.eval()
        self.vision_surrogate.train()

    def prepare_student_batch(self, _attention_mask: torch.Tensor) -> None:
        return None

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.vision_surrogate.parameters()

    def router_losses(self) -> dict[str, torch.Tensor]:
        balance, importance = self.vision_surrogate.router_losses()
        zero = balance * 0.0
        return {
            "vision_balance": balance,
            "vision_importance": importance,
            "language_balance": zero,
            "language_importance": zero,
            "balance": balance,
            "importance": importance,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"vision_tokenwise_optical": self.vision_surrogate.state_dict()}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = payload.get("vision_tokenwise_optical", payload)
        self.vision_surrogate.load_state_dict(state)

    def close(self) -> None:
        self.use_teacher()

    def architecture_report(self) -> dict[str, Any]:
        return {
            "student_language_mode": "frozen_native_electronic",
            "vision_replacement": "adapter_free_per_token_topk_optical_moe",
            "native_deepstack_visual_indexes": list(self.native_deepstack_indexes),
            "deepstack_behavior": "same single optical vision output observed at native tap indexes",
            "native_qwen_vision_blocks_executed": 0,
            "native_qwen_language_layers_frozen_and_executed": True,
            "transformer_residual_enabled": self.vision_surrogate.settings.residual_enabled,
            **self.vision_surrogate.parameter_breakdown(),
        }


def build_student(
    loaded: LoadedBackbone, settings: Any
) -> TokenwiseVisionReplacement:
    settings.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    core = TokenwiseOpticalMoE(settings.hidden_size, settings).to(loaded.device)
    replacement = TokenwiseVisionReplacement(loaded.model, core)
    return replacement


def student_embeddings(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
    inputs: Mapping[str, torch.Tensor],
    embedding_dim: int,
) -> torch.Tensor:
    replacement.use_student()
    replacement.prepare_student_batch(inputs["attention_mask"])
    hidden = forward_base_hidden(loaded.model, inputs)
    output = official_mrl_embedding(hidden, inputs["attention_mask"], embedding_dim)
    if output.shape != (inputs["attention_mask"].shape[0], embedding_dim):
        raise RuntimeError(f"Unexpected student embedding shape {tuple(output.shape)}")
    return output


def trainable_parameter_report(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
) -> dict[str, Any]:
    ids = {id(p) for p in replacement.trainable_parameters() if p.requires_grad}
    names = []
    for name, parameter in replacement.vision_surrogate.named_parameters():
        if id(parameter) in ids:
            names.append({
                "name": f"vision_tokenwise_optical.{name}",
                "shape": list(parameter.shape),
                "parameters": parameter.numel(),
            })
    native_trainable = [
        name for name, parameter in loaded.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in ids
    ]
    if native_trainable:
        raise RuntimeError(f"Native Qwen parameters unexpectedly trainable: {native_trainable[:10]}")
    return {
        "teacher_model_id": getattr(loaded.model, "name_or_path", type(loaded.model).__name__),
        "qwen_native_parameters_trainable": 0,
        "input_adapter_parameters": 0,
        "output_adapter_parameters": 0,
        "trainable_parameters": sum(p.numel() for p in replacement.trainable_parameters()),
        "trainable_tensors": len(names),
        "trainable_parameter_list": names,
        "architecture": replacement.architecture_report(),
    }


__all__ = [
    "LoadedBackbone",
    "TokenwiseVisionReplacement",
    "build_student",
    "load_backbone",
    "student_embeddings",
    "trainable_parameter_report",
]
