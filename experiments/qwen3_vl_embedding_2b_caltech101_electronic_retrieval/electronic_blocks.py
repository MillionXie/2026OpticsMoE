from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    lengths_from_cu,
)


class ElectronicTransformerBlock(nn.Module):
    """Pre-norm attention and SwiGLU FFN at a compact token width."""

    def __init__(
        self,
        width: int,
        heads: int,
        ff_multiplier: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        ff_width = int(round(width * ff_multiplier))
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_in = nn.Linear(width, 2 * ff_width)
        self.ffn_out = nn.Linear(ff_width, width)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        causal_mask = (
            torch.ones(
                hidden.shape[1],
                hidden.shape[1],
                dtype=torch.bool,
                device=hidden.device,
            ).triu(1)
            if causal
            else None
        )
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attention)
        gate, value = self.ffn_in(self.ffn_norm(hidden)).chunk(2, dim=-1)
        update = self.ffn_out(F.silu(gate) * value)
        hidden = hidden + self.ffn_dropout(update)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class _CompatibilityRouter(nn.Module):
    """Parameter-free single path used only by shared training diagnostics."""


class _CompatibilityPhase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("raw_phase", torch.zeros(1, 1), persistent=False)
        self.parameterization = "sigmoid"


class _CompatibilityExpertLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_CompatibilityPhase()])


class _CompatibilityGlobalPhase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phase = _CompatibilityPhase()


class ElectronicSequenceCore(nn.Module):
    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_tokens = int(max_tokens)
        self.width = int(settings.electronic_width)
        self.input_adapter = nn.Linear(self.hidden_size, self.width)
        self.input_norm = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList(
            [
                ElectronicTransformerBlock(
                    self.width,
                    settings.electronic_heads,
                    settings.electronic_ff_multiplier,
                    settings.electronic_dropout,
                    settings.electronic_attention_dropout,
                )
                for _ in range(settings.electronic_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.width)
        self.output_adapter = nn.Linear(self.width, self.hidden_size)
        initial = float(settings.electronic_initial_residual_weight)
        self.residual_logit = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(initial))))
        )

        # The shared trainer expects these diagnostic attributes. They contain
        # no trainable parameters and never participate in the forward path.
        self.router = _CompatibilityRouter()
        self.expert_layers = nn.ModuleList([_CompatibilityExpertLayer()])
        self.global_phase = _CompatibilityGlobalPhase()
        self.last_routing: dict[str, torch.Tensor] = {}
        self.last_latent_groups: list[torch.Tensor] = []

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not groups or any(group.ndim != 2 for group in groups):
            raise ValueError("Electronic core expects a non-empty list of [T,D] groups")
        lengths = [len(group) for group in groups]
        if any(length <= 0 or length > self.max_tokens for length in lengths):
            raise RuntimeError(
                f"Electronic token lengths {lengths} exceed configured maximum {self.max_tokens}"
            )
        max_length = max(lengths)
        padded = groups[0].new_zeros(len(groups), max_length, self.hidden_size)
        padding_mask = torch.ones(
            len(groups), max_length, dtype=torch.bool, device=groups[0].device
        )
        for index, group in enumerate(groups):
            padded[index, : len(group)] = group
            padding_mask[index, : len(group)] = False
        input_latent = self.input_norm(self.input_adapter(padded.float()))
        latent = input_latent
        for block in self.blocks:
            latent = block(latent, padding_mask=padding_mask, causal=causal)
        latent = self.output_norm(latent)
        gate = torch.sigmoid(self.residual_logit)
        latent = input_latent + gate * (latent - input_latent)
        delta = self.output_adapter(latent)
        output = padded.float() + gate * delta
        output = output.to(groups[0].dtype)
        self.last_latent_groups = [
            latent[index, :length] for index, length in enumerate(lengths)
        ]
        device = padded.device
        self.last_routing = {
            "selected_mask": torch.ones(
                len(groups), 1, dtype=torch.bool, device=device
            ),
            "importance": torch.ones(1, device=device),
            "normalized_entropy": torch.zeros((), device=device),
        }
        packed = torch.cat(
            [output[index, :length] for index, length in enumerate(lengths)], dim=0
        )
        return packed, latent

    def router_response_consistency_loss(self) -> torch.Tensor:
        return self.residual_logit.new_zeros(())

    def parameter_breakdown(self) -> dict[str, Any]:
        attention = sum(
            parameter.numel()
            for block in self.blocks
            for parameter in block.attention.parameters()
        )
        ffn = sum(
            parameter.numel()
            for block in self.blocks
            for module in (block.ffn_in, block.ffn_out)
            for parameter in module.parameters()
        )
        adapters = sum(
            parameter.numel()
            for module in (self.input_adapter, self.output_adapter)
            for parameter in module.parameters()
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "implementation": "dense_electronic_transformer",
            "moe_enabled": False,
            "optical_parameters": 0,
            "router_parameters": 0,
            "attention_parameters": attention,
            "ffn_parameters": ffn,
            "adapter_parameters": adapters,
            "residual_gate_parameters": self.residual_logit.numel(),
            "total_parameters": total,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class VisionElectronicReplacement(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.core = ElectronicSequenceCore(
            hidden_size, settings.max_visual_tokens, settings
        )
        self.tap_stages = (1,)
        self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None

    def compute(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        residual_base: torch.Tensor | None = None,
    ) -> None:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        packed, _ = self.core.forward_groups(
            list(hidden_states.split(lengths)), causal=False
        )
        if hidden_states.shape != packed.shape:
            raise RuntimeError("Vision electronic replacement changed packed shape")
        # The core already contains a learnable gated identity residual.
        self.tap_outputs = [packed]
        self.last_output = packed

    def output_for_slot(self, slot: int) -> torch.Tensor:
        if slot == 0 and self.tap_outputs:
            return self.tap_outputs[0]
        if slot == 1 and self.last_output is not None:
            return self.last_output
        raise RuntimeError("Vision electronic outputs are unavailable")

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.core.residual_logit.new_zeros(())
        return zero, zero

    def set_phase_dropout_active(self, _active: bool) -> None:
        return None

    def set_intermediate_field_capture(self, _enabled: bool, _sample_count: int = 1) -> None:
        return None

    def parameter_breakdown(self) -> dict[str, Any]:
        return self.core.parameter_breakdown()


class LanguageElectronicReplacement(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.core = ElectronicSequenceCore(
            hidden_size, settings.max_language_tokens, settings
        )
        self.valid_mask: torch.Tensor | None = None
        self.lengths: list[int] = []
        self.deepstack_injection_count = 0

    def set_attention_mask(self, mask: torch.Tensor) -> None:
        self.valid_mask = mask.bool()
        self.lengths = [int(value) for value in mask.long().sum(1).tolist()]
        if max(self.lengths) > self.core.max_tokens:
            raise RuntimeError("Language tokens exceed electronic maximum")

    def set_deepstack_injection_count(self, count: int) -> None:
        self.deepstack_injection_count = int(count)

    def forward_stage(
        self,
        stage: int,
        hidden_states: torch.Tensor,
        optical_input: torch.Tensor | None = None,
        residual_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stage != 0:
            raise RuntimeError("Dense electronic replacement has exactly one Qwen stage")
        branch = hidden_states if optical_input is None else optical_input
        if self.valid_mask is None or self.valid_mask.shape != branch.shape[:2]:
            raise RuntimeError("Call prepare_student_batch before language replacement")
        mask = self.valid_mask.to(branch.device)
        groups = [branch[index, mask[index]] for index in range(branch.shape[0])]
        packed, _ = self.core.forward_groups(groups, causal=True)
        output = torch.zeros_like(hidden_states)
        output[mask] = packed
        return output

    def retrieval_detector_features(self) -> torch.Tensor:
        if len(self.core.last_latent_groups) != len(self.lengths):
            raise RuntimeError("Language electronic features are unavailable")
        features = torch.stack(
            [group[-1] for group in self.core.last_latent_groups], dim=0
        )
        if not torch.isfinite(features).all():
            raise RuntimeError("Language electronic features contain NaN or Inf")
        return features

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.core.residual_logit.new_zeros(())
        return zero, zero

    def set_phase_dropout_active(self, _active: bool) -> None:
        return None

    def set_intermediate_field_capture(self, _enabled: bool, _sample_count: int = 1) -> None:
        return None

    def parameter_breakdown(self) -> dict[str, Any]:
        return self.core.parameter_breakdown()
