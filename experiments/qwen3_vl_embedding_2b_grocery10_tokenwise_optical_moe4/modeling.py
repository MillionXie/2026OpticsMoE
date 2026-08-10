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


def locate_language(model: nn.Module) -> nn.Module:
    for candidate in (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "norm"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL language decoder layers")


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


class LanguageBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class TokenwiseLanguageSurrogate(nn.Module):
    """2048-D multimodal tokens -> 1024-D optical core -> 2048-D Qwen space.

    The two Linear layers are the minimum dimensional bridge required by the
    32x32 token field.  All valid visual/text tokens pass through one shared
    optical language stack after Qwen has performed its single normal visual
    token injection. Padding never enters the optical panel.
    """

    def __init__(self, language_hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.language_hidden_size = int(language_hidden_size)
        self.max_tokens = int(settings.max_language_tokens)
        self.input_norm = nn.LayerNorm(
            self.language_hidden_size,
            elementwise_affine=bool(settings.language_adapter_layernorm_affine),
        )
        self.input_adapter = nn.Linear(self.language_hidden_size, settings.hidden_size)
        self.optical_core = TokenwiseOpticalMoE(settings.hidden_size, settings)
        self.output_adapter = nn.Linear(settings.hidden_size, self.language_hidden_size)
        self._attention_mask: torch.Tensor | None = None
        self.last_token_counts: list[int] = []

    def prepare_batch(self, attention_mask: torch.Tensor) -> None:
        if attention_mask.ndim != 2:
            raise ValueError("Language attention mask must be [B,S]")
        counts = attention_mask.to(torch.bool).sum(dim=-1)
        if int(counts.max()) > self.max_tokens:
            raise RuntimeError(
                f"language token count {int(counts.max())} exceeds optical panel capacity "
                f"{self.max_tokens}; shorten the prompt. Silent truncation is forbidden."
            )
        if int(counts.min()) <= 0:
            raise RuntimeError("Every language sample must contain at least one valid token")
        self._attention_mask = attention_mask.to(torch.bool)
        self.last_token_counts = counts.detach().cpu().tolist()

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        if self._attention_mask is None:
            raise RuntimeError("Call prepare_student_batch before the optical language layer")
        mask = self._attention_mask.to(hidden_states.device)
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Language hidden/mask mismatch: hidden={tuple(hidden_states.shape)} "
                f"mask={tuple(mask.shape)}"
            )
        original_dtype = hidden_states.dtype
        normalized = self.input_norm(hidden_states.float())
        latent = self.input_adapter(normalized)
        packed = latent[mask]
        counts = mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [counts.new_zeros(1), counts.cumsum(dim=0)], dim=0
        )
        optical = self.optical_core(packed, cu_seqlens)
        decoded = self.output_adapter(optical.float())
        output = hidden_states.float().clone()
        # Linear follows the active autocast policy (BF16 on the server), while
        # the numerically stable scatter buffer is deliberately FP32. Boolean
        # indexed assignment requires exact dtype equality, unlike arithmetic.
        output[mask] = decoded.to(dtype=output.dtype)
        return output.to(original_dtype)


class TokenwiseVisionReplacement:
    """Vision token MoE plus optional language token MoE.

    In the new optical-language mode DeepStack is disabled. Qwen injects the
    final merged visual tokens into the text sequence exactly once, then the
    first decoder slot applies the optical language stack; all other native
    decoder layers are bypassed. The native final RMSNorm and official MRL
    embedding readout remain frozen and active.
    """

    def __init__(
        self,
        model: nn.Module,
        vision_core: TokenwiseOpticalMoE,
        settings: Any,
        language_surrogate: TokenwiseLanguageSurrogate | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.visual = locate_visual(model)
        self.blocks = self.visual.blocks
        self.original_blocks = list(self.blocks)
        self.native_deepstack_indexes = tuple(
            int(v) for v in getattr(self.visual, "deepstack_visual_indexes", [])
        )
        self.vision_surrogate = vision_core
        self.student_blocks: list[nn.Module] = [VisionBypass() for _ in self.blocks]
        self.student_blocks[0] = TokenwiseVisionBlock(vision_core)

        self.language_model = locate_language(model)
        self.language_layers = self.language_model.layers
        self.original_language_layers = list(self.language_layers)
        self.language_surrogate = language_surrogate
        self.student_language_layers: list[nn.Module] = [
            LanguageBypass() for _ in self.language_layers
        ]
        if language_surrogate is not None:
            self.student_language_layers[0] = language_surrogate
        self.configure_student_trainability()

    @property
    def language_is_optical(self) -> bool:
        return self.language_surrogate is not None

    def use_teacher(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.blocks[index] = block
        for index, layer in enumerate(self.original_language_layers):
            self.language_layers[index] = layer
        self.visual.deepstack_visual_indexes = list(self.native_deepstack_indexes)

    def use_student(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.blocks[index] = block
        if self.language_is_optical:
            for index, layer in enumerate(self.student_language_layers):
                self.language_layers[index] = layer
        else:
            for index, layer in enumerate(self.original_language_layers):
                self.language_layers[index] = layer
        self.visual.deepstack_visual_indexes = (
            list(self.native_deepstack_indexes)
            if self.settings.student_deepstack_enabled
            else []
        )

    def configure_student_trainability(self) -> None:
        self.model.requires_grad_(False)
        self.vision_surrogate.requires_grad_(True)
        if self.language_surrogate is not None:
            self.language_surrogate.requires_grad_(True)

    def set_student_train_mode(self) -> None:
        self.model.eval()
        self.vision_surrogate.train()
        if self.language_surrogate is not None:
            self.language_surrogate.train()

    def prepare_student_batch(self, attention_mask: torch.Tensor) -> None:
        if self.language_surrogate is not None:
            self.language_surrogate.prepare_batch(attention_mask)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.vision_surrogate.parameters()
        if self.language_surrogate is not None:
            yield from self.language_surrogate.parameters()

    def router_losses(self) -> dict[str, torch.Tensor]:
        vision_balance, vision_importance = self.vision_surrogate.router_losses()
        if self.language_surrogate is None:
            language_balance = vision_balance * 0.0
            language_importance = vision_importance * 0.0
            balance, importance = vision_balance, vision_importance
        else:
            language_balance, language_importance = (
                self.language_surrogate.optical_core.router_losses()
            )
            balance = 0.5 * (vision_balance + language_balance)
            importance = 0.5 * (vision_importance + language_importance)
        return {
            "vision_balance": vision_balance,
            "vision_importance": vision_importance,
            "language_balance": language_balance,
            "language_importance": language_importance,
            "balance": balance,
            "importance": importance,
        }

    def phase_dc_loss(self) -> torch.Tensor:
        loss = self.vision_surrogate.phase_dc_loss()
        if self.language_surrogate is not None:
            loss = loss + self.language_surrogate.optical_core.phase_dc_loss()
        return loss

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vision_tokenwise_optical": self.vision_surrogate.state_dict()
        }
        if self.language_surrogate is not None:
            payload["language_tokenwise_optical"] = self.language_surrogate.state_dict()
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        vision_state = payload.get("vision_tokenwise_optical", payload)
        self.vision_surrogate.load_state_dict(vision_state)
        if self.language_surrogate is not None:
            language_state = payload.get("language_tokenwise_optical")
            if language_state is None:
                raise RuntimeError("Checkpoint does not contain language_tokenwise_optical")
            self.language_surrogate.load_state_dict(language_state)

    def close(self) -> None:
        self.use_teacher()

    def architecture_report(self) -> dict[str, Any]:
        language = None
        if self.language_surrogate is not None:
            language = {
                "input_adapter": [2048, self.settings.hidden_size],
                "optical_core": self.language_surrogate.optical_core.parameter_breakdown(),
                "output_adapter": [self.settings.hidden_size, 2048],
                "max_tokens": self.settings.max_language_tokens,
            }
        return {
            "student_language_mode": (
                "tokenwise_optical_moe" if self.language_is_optical else "frozen_native_electronic"
            ),
            "vision_replacement": "adapter_free_per_token_topk_optical_moe",
            "native_deepstack_visual_indexes": list(self.native_deepstack_indexes),
            "student_deepstack_visual_indexes": (
                list(self.native_deepstack_indexes)
                if self.settings.student_deepstack_enabled else []
            ),
            "deepstack_behavior": (
                "native" if self.settings.student_deepstack_enabled
                else "disabled; final visual tokens are injected into the language sequence once"
            ),
            "native_qwen_vision_blocks_executed": 0,
            "native_qwen_language_layers_executed": (
                len(self.original_language_layers) if not self.language_is_optical else 0
            ),
            "language": language,
            "transformer_residual_enabled": self.vision_surrogate.settings.residual_enabled,
            "vision": self.vision_surrogate.parameter_breakdown(),
        }


def _language_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(locate_language(model).norm, "weight").numel()
    return int(value)


def build_student(
    loaded: LoadedBackbone, settings: Any
) -> TokenwiseVisionReplacement:
    settings.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    vision_core = TokenwiseOpticalMoE(settings.hidden_size, settings).to(loaded.device)
    language_surrogate = None
    if settings.student_language_mode == "optical_moe":
        language_surrogate = TokenwiseLanguageSurrogate(
            _language_hidden_size(loaded.model), settings
        ).to(loaded.device)
    return TokenwiseVisionReplacement(
        loaded.model, vision_core, settings, language_surrogate
    )


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
    modules: list[tuple[str, nn.Module]] = [
        ("vision_tokenwise_optical", replacement.vision_surrogate)
    ]
    if replacement.language_surrogate is not None:
        modules.append(("language_tokenwise_optical", replacement.language_surrogate))
    seen: set[int] = set()
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if id(parameter) in ids and id(parameter) not in seen:
                seen.add(id(parameter))
                names.append({
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "parameters": parameter.numel(),
                })
    native_trainable = [
        name for name, parameter in loaded.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in ids
    ]
    if native_trainable:
        raise RuntimeError(f"Native Qwen parameters unexpectedly trainable: {native_trainable[:10]}")
    input_adapters = sum(
        row["parameters"] for row in names if ".input_adapter." in row["name"]
    )
    output_adapters = sum(
        row["parameters"] for row in names if ".output_adapter." in row["name"]
    )
    return {
        "teacher_model_id": getattr(loaded.model, "name_or_path", type(loaded.model).__name__),
        "qwen_native_parameters_trainable": 0,
        "input_adapter_parameters": input_adapters,
        "output_adapter_parameters": output_adapters,
        "trainable_parameters": sum(row["parameters"] for row in names),
        "trainable_tensors": len(names),
        "trainable_parameter_list": names,
        "architecture": replacement.architecture_report(),
    }


__all__ = [
    "LoadedBackbone",
    "TokenwiseLanguageSurrogate",
    "TokenwiseVisionReplacement",
    "build_student",
    "load_backbone",
    "student_embeddings",
    "trainable_parameter_report",
]
