from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .moe import OpticalMoECore


class ResidualScale(nn.Module):
    def __init__(self, value: float, trainable: bool) -> None:
        super().__init__()
        tensor = torch.tensor(float(value), dtype=torch.float32)
        if trainable:
            self.value = nn.Parameter(tensor)
        else:
            self.register_buffer("value", tensor)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return delta * self.value.to(device=delta.device, dtype=delta.dtype)


class FoldedOpticalMixerBlock(nn.Module):
    """Mixer-shaped token/channel residual block using one five-stage MoE9 core.

    Stage 0-1 forms the token-mixing branch. Stage 2-4 forms the
    channel-mixing branch. The router is evaluated once from the token field and
    its exact top-k selection/weights are reused after the mid-block reload.
    """

    def __init__(self, settings: Any) -> None:
        super().__init__()
        hidden = settings.model.hidden_size
        self.token_count = settings.model.token_count
        self.field_size = settings.geometry.expert_size
        self.token_stage_count = settings.model.token_stages_per_block
        self.channel_stage_count = settings.model.channel_stages_per_block
        self.token_norm = nn.LayerNorm(hidden, eps=settings.model.final_layernorm_eps)
        self.channel_norm = nn.LayerNorm(hidden, eps=settings.model.final_layernorm_eps)
        self.nonnegative = nn.Softplus()
        self.core = OpticalMoECore(settings)
        self.token_residual_scale = ResidualScale(
            settings.model.residual_scale, settings.model.residual_scale_trainable
        )
        self.channel_residual_scale = ResidualScale(
            settings.model.residual_scale, settings.model.residual_scale_trainable
        )
        self.last_token_delta: torch.Tensor | None = None
        self.last_channel_delta: torch.Tensor | None = None

    def _token_field(self, hidden: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> [B,C,T], with exact zeros for the 28 non-token columns.
        valid = self.nonnegative(self.token_norm(hidden)).transpose(1, 2)
        if valid.shape[1] != self.field_size or valid.shape[2] > self.field_size:
            raise RuntimeError(
                f"Token optical mapping expected [B,{self.field_size},T<=field], "
                f"got {tuple(valid.shape)}"
            )
        return F.pad(valid, (0, self.field_size - valid.shape[2], 0, 0), value=0)

    def _channel_field(self, hidden: torch.Tensor) -> torch.Tensor:
        # [B,T,C], with exact zeros for the 28 non-token rows.
        valid = self.nonnegative(self.channel_norm(hidden))
        if valid.shape[2] != self.field_size or valid.shape[1] > self.field_size:
            raise RuntimeError(
                f"Channel optical mapping expected [B,T<=field,{self.field_size}], "
                f"got {tuple(valid.shape)}"
            )
        return F.pad(valid, (0, 0, 0, self.field_size - valid.shape[1]), value=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (
            self.token_count,
            self.field_size,
        ):
            raise ValueError(
                f"Expected [B,{self.token_count},{self.field_size}], got {tuple(hidden.shape)}"
            )
        token_field = self._token_field(hidden)
        field, routing = self.core.begin(token_field)
        for stage in range(self.token_stage_count):
            field = self.core.run_stage(stage, field, routing)
        token_readout = self.core.global_readout(field, "token")
        token_delta = token_readout[:, :, : self.token_count].transpose(1, 2)
        mixed = hidden + self.token_residual_scale(token_delta.to(hidden.dtype))

        channel_field = self._channel_field(mixed)
        field = self.core.reload_with_same_routing(channel_field, routing)
        start = self.token_stage_count
        stop = start + self.channel_stage_count
        for stage in range(start, stop):
            field = self.core.run_stage(stage, field, routing)
        channel_readout = self.core.global_readout(field, "channel")
        channel_delta = channel_readout[:, : self.token_count, :]
        output = mixed + self.channel_residual_scale(channel_delta.to(hidden.dtype))
        if self.core.capture_debug:
            self.last_token_delta = token_delta.detach().cpu()
            self.last_channel_delta = channel_delta.detach().cpu()
            self.core.last_debug["token_delta"] = self.last_token_delta
            self.core.last_debug["channel_delta"] = self.last_channel_delta
            self.core.last_debug["block_output"] = output.detach().cpu()
        return output

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.core.last_routing["balance_loss"],
            self.core.last_routing["importance_loss"],
        )

    def set_phase_dropout_active(self, active: bool) -> None:
        self.core.set_phase_dropout_active(active)


@dataclass
class OpticalMixerOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    pooled_hidden: torch.Tensor
    router_balance_loss: torch.Tensor
    router_importance_loss: torch.Tensor
    router_statistics: list[dict[str, torch.Tensor]]


class OpticalMixerMoE9(nn.Module):
    def __init__(self, settings: Any) -> None:
        super().__init__()
        model = settings.model
        self.settings = settings
        self.patch_embed = nn.Conv2d(
            3,
            model.hidden_size,
            kernel_size=model.patch_size,
            stride=model.patch_size,
        )
        self.blocks = nn.ModuleList(
            [FoldedOpticalMixerBlock(settings) for _ in range(model.num_blocks)]
        )
        self.final_norm = nn.LayerNorm(
            model.hidden_size, eps=model.final_layernorm_eps
        )
        self.clip_projection = nn.Linear(
            model.hidden_size, model.clip_projection_dim
        )
        self.classifier = nn.Linear(model.clip_projection_dim, model.num_classes)

    def forward(self, images: torch.Tensor) -> OpticalMixerOutput:
        hidden = self.patch_embed(images).flatten(2).transpose(1, 2)
        expected = (
            self.settings.model.token_count,
            self.settings.model.hidden_size,
        )
        if tuple(hidden.shape[1:]) != expected:
            raise RuntimeError(
                f"Patch embedding produced {tuple(hidden.shape)}, expected [B,{expected[0]},{expected[1]}]"
            )
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.final_norm(hidden).mean(dim=1)
        embedding = F.normalize(self.clip_projection(pooled), dim=-1)
        logits = self.classifier(embedding)
        balance_losses = []
        importance_losses = []
        router_statistics = []
        for block in self.blocks:
            balance, importance = block.router_losses()
            balance_losses.append(balance)
            importance_losses.append(importance)
            routing = block.core.last_routing
            router_statistics.append(
                {
                    "load": routing["load"],
                    "importance": routing["importance"],
                    "weights_mean": routing["weights"].mean(0),
                    "selected_count": routing["selected_mask"].float().sum(0),
                    "normalized_entropy": routing["normalized_entropy"],
                }
            )
        return OpticalMixerOutput(
            logits=logits,
            embedding=embedding,
            pooled_hidden=pooled,
            router_balance_loss=torch.stack(balance_losses).mean(),
            router_importance_loss=torch.stack(importance_losses).mean(),
            router_statistics=router_statistics,
        )

    def set_phase_dropout_active(self, active: bool) -> None:
        for block in self.blocks:
            block.set_phase_dropout_active(active)

    def set_debug_capture(self, block_indices: list[int], enabled: bool) -> None:
        selected = set(int(index) for index in block_indices)
        for index, block in enumerate(self.blocks):
            block.core.set_debug_capture(enabled and index in selected)

    def debug_state(self) -> dict[int, dict]:
        return {
            index: block.core.last_debug
            for index, block in enumerate(self.blocks)
            if block.core.capture_debug and block.core.last_debug
        }

    def parameter_breakdown(self) -> dict[str, Any]:
        blocks = [block.core.parameter_breakdown() for block in self.blocks]
        expert_phase = sum(item["expert_phase_parameters"] for item in blocks)
        global_phase = sum(item["global_phase_parameters"] for item in blocks)
        router = sum(item["router_parameters"] for item in blocks)
        oeo = sum(item["oeo_parameters"] for item in blocks)
        patch = sum(parameter.numel() for parameter in self.patch_embed.parameters())
        norms = sum(
            parameter.numel()
            for module in [self.final_norm]
            + [item.token_norm for item in self.blocks]
            + [item.channel_norm for item in self.blocks]
            for parameter in module.parameters()
        )
        projection = sum(parameter.numel() for parameter in self.clip_projection.parameters())
        classifier = sum(parameter.numel() for parameter in self.classifier.parameters())
        residual = sum(
            parameter.numel()
            for block in self.blocks
            for module in (block.token_residual_scale, block.channel_residual_scale)
            for parameter in module.parameters()
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        optical = expert_phase + global_phase
        electronic = total - optical
        return {
            "model_name": "OpticalMixerMoE9",
            "optical": {
                "expert_phase_parameters": expert_phase,
                "global_phase_parameters": global_phase,
                "total_phase_parameters": optical,
                "parameters_per_block": [
                    item["optical_phase_parameters"] for item in blocks
                ],
            },
            "electronic": {
                "patch_embedding_parameters": patch,
                "router_parameters": router,
                "oeo_affine_parameters": oeo,
                "normalization_parameters": norms,
                "clip_projection_parameters": projection,
                "classifier_parameters": classifier,
                "trainable_residual_scale_parameters": residual,
                "total_parameters": electronic,
            },
            "total_parameters": total,
            "total_trainable_parameters": trainable,
            "optical_phase_ratio": optical / max(total, 1),
        }
