from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    lengths_from_cu,
)


class ElectronicResidualMLPBlock(nn.Module):
    """Attention-free token mixer followed by a channel-wise residual MLP."""

    def __init__(
        self,
        width: int,
        expansion: float,
        dropout: float,
        initial_residual_weight: float,
        token_mixer_enabled: bool = False,
        token_mixer_kernel_size: int = 5,
        token_mixer_type: str = "depthwise_conv1d",
    ) -> None:
        super().__init__()
        hidden_width = int(round(width * expansion))
        self.token_mixer_enabled = bool(token_mixer_enabled)
        self.token_mixer_kernel_size = int(token_mixer_kernel_size)
        self.token_mixer_type = str(token_mixer_type)
        if self.token_mixer_enabled:
            self.token_norm = nn.LayerNorm(width)
            convolution = nn.Conv2d if self.token_mixer_type == "depthwise_conv2d" else nn.Conv1d
            self.token_depthwise = convolution(
                width, width, kernel_size=self.token_mixer_kernel_size, groups=width, bias=False
            )
            self.token_pointwise = nn.Linear(width, width)
            self.token_dropout = nn.Dropout(dropout)
            self.token_residual_logit = nn.Parameter(
                torch.logit(torch.tensor(float(initial_residual_weight)))
            )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
            nn.Dropout(dropout),
        )
        self.residual_logit = nn.Parameter(
            torch.logit(torch.tensor(float(initial_residual_weight)))
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        causal: bool,
        spatial_shapes: list[tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        if self.token_mixer_enabled:
            token_input = self.token_norm(hidden).masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
            if self.token_mixer_type == "depthwise_conv2d":
                if causal or spatial_shapes is None:
                    raise RuntimeError("2D token mixing requires non-causal spatial shapes")
                token_update = self._mix_spatial_tokens(token_input, spatial_shapes)
            else:
                token_input = token_input.transpose(1, 2)
                if causal:
                    token_input = F.pad(
                        token_input, (self.token_mixer_kernel_size - 1, 0)
                    )
                else:
                    left = self.token_mixer_kernel_size // 2
                    right = self.token_mixer_kernel_size - 1 - left
                    token_input = F.pad(token_input, (left, right))
                token_update = self.token_depthwise(token_input).transpose(1, 2)
            token_update = self.token_pointwise(F.gelu(token_update))
            token_update = self.token_dropout(token_update)
            hidden = hidden + torch.sigmoid(self.token_residual_logit) * token_update
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        update = self.mlp(self.norm(hidden))
        hidden = hidden + torch.sigmoid(self.residual_logit) * update
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _mix_spatial_tokens(
        self,
        token_input: torch.Tensor,
        spatial_shapes: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(spatial_shapes) != token_input.shape[0]:
            raise RuntimeError("Spatial shape count does not match the vision batch")
        merge_size = 2
        mixed = torch.zeros_like(token_input)
        padding = self.token_mixer_kernel_size // 2
        for index, (frames, height, width) in enumerate(spatial_shapes):
            token_count = frames * height * width
            if height % merge_size or width % merge_size:
                raise RuntimeError("Vision grid must be divisible by Qwen spatial merge size 2")
            if token_count > token_input.shape[1]:
                raise RuntimeError("Vision grid contains more tokens than the padded sequence")
            # Qwen stores patches block-major as [t,H/2,W/2,2,2,C]. Restore
            # the true image grid before convolution and restore Qwen order after it.
            grid = (
                token_input[index, :token_count]
                .view(
                    frames,
                    height // merge_size,
                    width // merge_size,
                    merge_size,
                    merge_size,
                    token_input.shape[-1],
                )
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(frames, token_input.shape[-1], height, width)
            )
            grid = self.token_depthwise(F.pad(grid, (padding, padding, padding, padding)))
            restored = (
                grid.view(
                    frames,
                    token_input.shape[-1],
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                )
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(token_count, token_input.shape[-1])
            )
            mixed[index, :token_count] = restored
        return mixed


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
    def __init__(
        self,
        hidden_size: int,
        max_tokens: int,
        settings: Any,
        token_mixer_type: str | None = None,
        token_mixer_kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_tokens = int(max_tokens)
        self.width = int(settings.electronic_width)
        self.expansion = float(settings.electronic_expansion)
        self.token_mixer_enabled = bool(settings.electronic_token_mixer_enabled)
        self.token_mixer_kernel_size = int(
            token_mixer_kernel_size or settings.electronic_token_mixer_kernel_size
        )
        self.token_mixer_type = str(
            token_mixer_type or settings.electronic_vision_token_mixer_type
        )
        self.input_adapter = nn.Linear(self.hidden_size, self.width)
        self.input_norm = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList(
            [
                ElectronicResidualMLPBlock(
                    self.width,
                    self.expansion,
                    settings.electronic_dropout,
                    settings.electronic_initial_residual_weight,
                    self.token_mixer_enabled,
                    self.token_mixer_kernel_size,
                    self.token_mixer_type,
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
        spatial_shapes: list[tuple[int, int, int]] | None = None,
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
            latent = block(
                latent,
                padding_mask=padding_mask,
                causal=causal,
                spatial_shapes=spatial_shapes,
            )
        latent = self.output_norm(latent)
        gate = torch.sigmoid(self.residual_logit)
        latent = input_latent + gate * (latent - input_latent)
        latent = latent.masked_fill(padding_mask.unsqueeze(-1), 0.0)
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
        ffn = sum(
            parameter.numel()
            for block in self.blocks
            for parameter in block.mlp.parameters()
        )
        token_mixer = sum(
            parameter.numel()
            for block in self.blocks
            if block.token_mixer_enabled
            for module in (
                block.token_norm,
                block.token_depthwise,
                block.token_pointwise,
            )
            for parameter in module.parameters()
        )
        adapters = sum(
            parameter.numel()
            for module in (self.input_adapter, self.output_adapter)
            for parameter in module.parameters()
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "implementation": (
                "depthwise_token_mixer_residual_mlp"
                if self.token_mixer_enabled
                else "shared_tokenwise_residual_mlp"
            ),
            "moe_enabled": False,
            "attention_enabled": False,
            "token_mixing_enabled": self.token_mixer_enabled,
            "token_mixer_type": (
                f"{self.token_mixer_type}_pointwise_linear"
                if self.token_mixer_enabled
                else "none"
            ),
            "token_mixer_kernel_size": self.token_mixer_kernel_size,
            "token_mixer_parameters": token_mixer,
            "optical_parameters": 0,
            "router_parameters": 0,
            "attention_parameters": 0,
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
            hidden_size,
            settings.max_visual_tokens,
            settings,
            settings.electronic_vision_token_mixer_type,
            settings.electronic_vision_token_mixer_kernel_size,
        )
        self.tap_stages = (1,) if settings.electronic_deepstack_enabled else ()
        self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.spatial_shapes: list[tuple[int, int, int]] | None = None

    def set_image_grid_thw(self, image_grid_thw: torch.Tensor | None) -> None:
        if image_grid_thw is None:
            self.spatial_shapes = None
            return
        shapes: list[tuple[int, int, int]] = []
        for frames, height, width in image_grid_thw.detach().cpu().long().tolist():
            shapes.extend((1, int(height), int(width)) for _ in range(int(frames)))
        self.spatial_shapes = shapes

    def compute(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        residual_base: torch.Tensor | None = None,
    ) -> None:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        packed, _ = self.core.forward_groups(
            list(hidden_states.split(lengths)),
            causal=False,
            spatial_shapes=(
                self.spatial_shapes
                if self.core.token_mixer_type == "depthwise_conv2d"
                else None
            ),
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
            hidden_size,
            settings.max_language_tokens,
            settings,
            settings.electronic_language_token_mixer_type,
            settings.electronic_language_token_mixer_kernel_size,
        )
        self.valid_mask: torch.Tensor | None = None
        self.lengths: list[int] = []
        self.deepstack_injection_count = 0
        self.pooling = str(settings.electronic_pooling)

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
        if self.pooling == "mean":
            features = torch.stack(
                [group.mean(dim=0) for group in self.core.last_latent_groups], dim=0
            )
        elif self.pooling == "mean_max":
            features = torch.stack(
                [
                    torch.cat((group.mean(dim=0), group.amax(dim=0)), dim=0)
                    for group in self.core.last_latent_groups
                ],
                dim=0,
            )
        else:
            raise RuntimeError(f"Unsupported electronic pooling: {self.pooling}")
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
