from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .physical import AngularSpectrumPropagator, PhaseTensor, tokenwise_layer_norm


@dataclass(frozen=True)
class TokenwiseLayout:
    token_rows: int
    token_cols: int
    feature_side: int
    expert_rows: int
    expert_cols: int
    expert_gap: int
    token_gap: int
    padding: int

    @property
    def num_experts(self) -> int:
        return self.expert_rows * self.expert_cols

    @property
    def max_tokens(self) -> int:
        return self.token_rows * self.token_cols

    @property
    def group_height(self) -> int:
        return self.expert_rows * self.feature_side + (self.expert_rows - 1) * self.expert_gap

    @property
    def group_width(self) -> int:
        return self.expert_cols * self.feature_side + (self.expert_cols - 1) * self.expert_gap

    @property
    def active_height(self) -> int:
        return self.token_rows * self.group_height + (self.token_rows - 1) * self.token_gap

    @property
    def active_width(self) -> int:
        return self.token_cols * self.group_width + (self.token_cols - 1) * self.token_gap

    @property
    def canvas_size(self) -> int:
        if self.active_height != self.active_width:
            raise ValueError("Token-wise optical layout must be square")
        return self.active_height + 2 * self.padding

    def linear_indices(self) -> torch.Tensor:
        """Return unique [token, expert, y, x] indexes into the padded canvas."""
        side, canvas = self.feature_side, self.canvas_size
        result = torch.empty(
            self.max_tokens, self.num_experts, side, side, dtype=torch.long
        )
        yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        for token in range(self.max_tokens):
            token_row, token_col = divmod(token, self.token_cols)
            token_y = self.padding + token_row * (self.group_height + self.token_gap)
            token_x = self.padding + token_col * (self.group_width + self.token_gap)
            for expert in range(self.num_experts):
                expert_row, expert_col = divmod(expert, self.expert_cols)
                y0 = token_y + expert_row * (side + self.expert_gap)
                x0 = token_x + expert_col * (side + self.expert_gap)
                result[token, expert] = (yy + y0) * canvas + (xx + x0)
        if result.unique().numel() != result.numel():
            raise RuntimeError("Token/expert apertures overlap")
        return result


class PerTokenTopKRouter(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.num_experts = int(settings.num_experts)
        self.top_k = int(settings.top_k)
        self.temperature = float(settings.router_temperature)
        self.noise_std = float(settings.router_noise_std)
        self.norm = (
            nn.LayerNorm(
                hidden_size,
                elementwise_affine=bool(settings.router_layernorm_affine),
            )
            if settings.router_layernorm_enabled
            else nn.Identity()
        )
        self.gate = nn.Linear(hidden_size, self.num_experts)
        nn.init.normal_(self.gate.weight, 0.0, float(settings.router_gate_init_std))
        nn.init.zeros_(self.gate.bias)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("Router expects tokens [B,T,H] and valid_mask [B,T]")
        logits = self.gate(self.norm(tokens.float()))
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        indices = probabilities.topk(self.top_k, dim=-1).indices
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, indices, True)
        weights = probabilities * selected
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        valid = valid_mask[..., None]
        probabilities = probabilities * valid
        weights = weights * valid
        selected = selected & valid
        denominator = valid_mask.sum().clamp_min(1).to(probabilities.dtype)
        importance = probabilities.sum(dim=(0, 1)) / denominator
        load = selected.float().sum(dim=(0, 1)) / (denominator * self.top_k)
        balance = self.num_experts * (importance * load).sum()
        importance_loss = self.num_experts * importance.square().sum()
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1)
        entropy = (entropy * valid_mask).sum() / denominator
        return {
            "logits": logits,
            "probabilities": probabilities,
            "selected_mask": selected,
            "weights": weights,
            "indices": indices,
            "balance_loss": balance,
            "importance_loss": importance_loss,
            "normalized_entropy": entropy / math.log(self.num_experts),
            "importance": importance,
            "load": load,
        }


class TokenwiseOpticalMoE(nn.Module):
    """Adapter-free per-token optical MoE for packed Qwen vision hidden states.

    Every 1024-D token is reshaped exactly to 32x32. A shared electronic router
    selects experts independently for every token. The physical SLM contains a
    fixed 14x14 grid of token groups, with four 32x32 expert apertures per group.
    """

    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.settings = settings
        self.layout = TokenwiseLayout(
            settings.token_grid_rows,
            settings.token_grid_cols,
            settings.token_feature_side,
            settings.expert_grid_rows,
            settings.expert_grid_cols,
            settings.expert_gap,
            settings.token_group_gap,
            settings.propagation_padding,
        )
        if self.hidden_size != self.layout.feature_side**2:
            raise ValueError(
                f"No adapter is allowed: hidden_size {hidden_size} must equal "
                f"feature_side^2={self.layout.feature_side**2}"
            )
        self.router = PerTokenTopKRouter(hidden_size, settings)
        self.register_buffer("aperture_indices", self.layout.linear_indices(), persistent=False)
        phase_kwargs = dict(
            parameterization=settings.phase_parameterization,
            init=settings.phase_init,
            init_std=settings.phase_init_std,
            dropout_mode=settings.phase_dropout_mode,
            dropout_p=settings.phase_dropout_p,
            dropout_block_size=settings.phase_dropout_block_size,
            dropout_batch_shared=settings.phase_dropout_batch_shared,
        )
        first_shape = (
            (self.layout.num_experts, self.layout.feature_side, self.layout.feature_side)
            if settings.share_expert_phase_across_tokens
            else (
                self.layout.max_tokens,
                self.layout.num_experts,
                self.layout.feature_side,
                self.layout.feature_side,
            )
        )
        self.first_expert_phase = PhaseTensor(first_shape, **phase_kwargs)
        if settings.second_plane_mode == "expert":
            self.second_phase = PhaseTensor(first_shape, **phase_kwargs)
        else:
            self.second_phase = PhaseTensor(
                (self.layout.active_height, self.layout.active_width), **phase_kwargs
            )
        self.propagator = AngularSpectrumPropagator(
            settings.wavelength_nm * 1e-9,
            settings.pixel_pitch_um * 1e-6,
            self.layout.canvas_size,
            settings.propagation_distance_m,
            k_space_enabled=settings.k_space_enabled,
            theta_max_deg=settings.theta_max_deg,
        )
        affine_shape = (
            self.layout.num_experts,
            self.layout.feature_side,
            self.layout.feature_side,
        )
        if settings.oeo_elementwise_affine:
            self.oeo_weight = nn.Parameter(torch.ones(affine_shape))
            self.oeo_bias = nn.Parameter(torch.zeros(affine_shape))
        else:
            self.register_parameter("oeo_weight", None)
            self.register_parameter("oeo_bias", None)
        if settings.final_layernorm_affine:
            self.final_weight = nn.Parameter(torch.ones(affine_shape))
            self.final_bias = nn.Parameter(torch.zeros(affine_shape))
        else:
            self.register_parameter("final_weight", None)
            self.register_parameter("final_bias", None)
        if settings.residual_scale_trainable:
            self.residual_scale = nn.Parameter(torch.tensor(float(settings.residual_scale)))
        else:
            self.register_buffer(
                "residual_scale", torch.tensor(float(settings.residual_scale)), persistent=True
            )
        self.last_routing: dict[str, torch.Tensor] = {}
        self.last_token_counts: list[int] = []
        self.last_input_amplitude: torch.Tensor | None = None
        self.last_amplitude_canvas: torch.Tensor | None = None
        self.last_first_detector_intensity: torch.Tensor | None = None
        self.last_reload_amplitude: torch.Tensor | None = None
        self.last_final_detector_intensity: torch.Tensor | None = None
        self.last_signed_readout: torch.Tensor | None = None

    def _groups(self, hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected packed hidden [sum(T),{self.hidden_size}], got {tuple(hidden.shape)}")
        if cu_seqlens is None:
            raise RuntimeError("Per-image cu_seqlens are required for token-wise routing")
        boundaries = cu_seqlens.detach().cpu().long().tolist()
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        if not lengths or sum(lengths) != len(hidden) or min(lengths) <= 0:
            raise RuntimeError("cu_seqlens do not match packed visual tokens")
        if max(lengths) > self.layout.max_tokens:
            raise RuntimeError(
                f"visual token count {max(lengths)} exceeds token panel capacity "
                f"{self.layout.max_tokens}; reduce processor_max_pixels. Silent truncation is forbidden."
            )
        padded = hidden.new_zeros((len(lengths), self.layout.max_tokens, self.hidden_size))
        valid = torch.zeros((len(lengths), self.layout.max_tokens), dtype=torch.bool, device=hidden.device)
        start = 0
        for row, length in enumerate(lengths):
            padded[row, :length] = hidden[start : start + length]
            valid[row, :length] = True
            start += length
        return padded, valid, lengths

    def _input_amplitude(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = tokens.float()
        if self.settings.input_normalization == "layernorm":
            values = F.layer_norm(values, (self.hidden_size,))
        if self.settings.input_nonlinearity == "softplus":
            values = F.softplus(values)
        elif self.settings.input_nonlinearity == "relu":
            values = F.relu(values)
        else:
            values = values.abs()
        amplitude = values.reshape(
            len(values), self.layout.max_tokens,
            self.layout.feature_side, self.layout.feature_side,
        )
        if self.settings.input_amplitude_normalization == "per_token_max":
            amplitude = amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        elif self.settings.input_amplitude_normalization == "per_token_rms":
            rms = amplitude.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
            amplitude = amplitude / rms
        return amplitude * valid[..., None, None]

    def _route_scales(self, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = routing["weights"].clamp_min(0.0)
        return weights if self.settings.amplitude_weight_domain == "amplitude" else weights.sqrt()

    def _scatter_crops(self, crops: torch.Tensor) -> torch.Tensor:
        batch = crops.shape[0]
        values = crops.reshape(batch, -1)
        indexes = self.aperture_indices.reshape(-1)
        canvas = crops.new_zeros((batch, self.layout.canvas_size**2))
        canvas = canvas.scatter(1, indexes[None].expand(batch, -1), values)
        return canvas.reshape(batch, self.layout.canvas_size, self.layout.canvas_size)

    def _gather_crops(self, canvas: torch.Tensor) -> torch.Tensor:
        batch = canvas.shape[0]
        indexes = self.aperture_indices.reshape(-1)
        return canvas.flatten(1).index_select(1, indexes).reshape(
            batch,
            self.layout.max_tokens,
            self.layout.num_experts,
            self.layout.feature_side,
            self.layout.feature_side,
        )

    def _apply_expert_phase(self, field: torch.Tensor, phase: PhaseTensor) -> torch.Tensor:
        crops = self._gather_crops(field.to(torch.complex64))
        modulation = phase.modulation(len(field), token_axis=not self.settings.share_expert_phase_across_tokens)
        if self.settings.share_expert_phase_across_tokens:
            if modulation.ndim == 3:
                modulation = modulation[None, None]
            else:
                modulation = modulation[:, None]
        else:
            if modulation.ndim == 4:
                modulation = modulation[None]
        return self._scatter_crops(crops * modulation)

    def _apply_global_phase(self, field: torch.Tensor) -> torch.Tensor:
        phase = self.second_phase
        modulation = phase.modulation(len(field), token_axis=False)
        if modulation.ndim == 2:
            modulation = modulation[None]
        y0 = x0 = self.layout.padding
        y1, x1 = y0 + self.layout.active_height, x0 + self.layout.active_width
        output = torch.zeros_like(field, dtype=torch.complex64)
        output[:, y0:y1, x0:x1] = field[:, y0:y1, x0:x1].to(torch.complex64) * modulation
        return output

    def _oeo_reload(
        self,
        field: torch.Tensor,
        routing: dict[str, torch.Tensor],
        valid: torch.Tensor,
    ) -> torch.Tensor:
        intensity = field.abs().square().float()
        crops = self._gather_crops(intensity)
        weight = self.oeo_weight[None, None] if self.oeo_weight is not None else None
        bias = self.oeo_bias[None, None] if self.oeo_bias is not None else None
        signed = tokenwise_layer_norm(crops, self.settings.oeo_layernorm_eps, weight, bias)
        amplitude = F.relu(signed) if self.settings.oeo_nonlinearity == "relu" else F.softplus(signed)
        if self.settings.oeo_reapply_routing_weights:
            amplitude = amplitude * self._route_scales(routing)[..., None, None]
        if self.settings.oeo_hard_route_mask:
            amplitude = amplitude * routing["selected_mask"][..., None, None]
        amplitude = amplitude * valid[:, :, None, None, None]
        if self.settings.capture_intermediate_fields:
            count = min(self.settings.visualization_sample_count, len(intensity))
            self.last_first_detector_intensity = intensity[:count].detach().cpu()
            self.last_reload_amplitude = amplitude[:count].detach().cpu()
        else:
            self.last_first_detector_intensity = None
            self.last_reload_amplitude = None
        reload_canvas = self._scatter_crops(amplitude)
        return torch.complex(reload_canvas, torch.zeros_like(reload_canvas))

    def _final_readout(
        self,
        field: torch.Tensor,
        routing: dict[str, torch.Tensor],
        valid: torch.Tensor,
    ) -> torch.Tensor:
        intensity = field.abs().square().float()
        crops = self._gather_crops(intensity)
        weight = self.final_weight[None, None] if self.final_weight is not None else None
        bias = self.final_bias[None, None] if self.final_bias is not None else None
        signed = tokenwise_layer_norm(crops, self.settings.final_layernorm_eps, weight, bias)
        if self.settings.final_aggregation == "routing_weighted_sum":
            coefficients = routing["weights"]
        else:
            coefficients = routing["selected_mask"].float() / float(self.settings.top_k)
        output = (signed * coefficients[..., None, None]).sum(dim=2)
        output = output * valid[..., None, None]
        if self.settings.capture_intermediate_fields:
            count = min(self.settings.visualization_sample_count, len(intensity))
            self.last_final_detector_intensity = intensity[:count].detach().cpu()
            self.last_signed_readout = output[:count].detach().cpu()
        else:
            self.last_final_detector_intensity = None
            self.last_signed_readout = None
        return output.flatten(-2)

    def forward(self, hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
        tokens, valid, lengths = self._groups(hidden, cu_seqlens)
        self.last_token_counts = lengths
        routing = self.router(tokens, valid)
        self.last_routing = routing
        amplitude = self._input_amplitude(tokens, valid)
        scales = self._route_scales(routing)
        routed = amplitude[:, :, None] * scales[..., None, None]
        canvas = self._scatter_crops(routed)
        if self.settings.capture_intermediate_fields:
            count = min(self.settings.visualization_sample_count, len(amplitude))
            self.last_input_amplitude = amplitude[:count].detach().cpu()
            self.last_amplitude_canvas = canvas[:count].detach().cpu()
        else:
            self.last_input_amplitude = None
            self.last_amplitude_canvas = None
        field = torch.complex(canvas, torch.zeros_like(canvas))
        field = self.propagator(self._apply_expert_phase(field, self.first_expert_phase))
        field = self._oeo_reload(field, routing, valid)
        if self.settings.second_plane_mode == "expert":
            field = self._apply_expert_phase(field, self.second_phase)
        else:
            field = self._apply_global_phase(field)
        field = self.propagator(field)
        padded_output = self._final_readout(field, routing, valid)
        packed = torch.cat(
            [padded_output[row, :length] for row, length in enumerate(lengths)], dim=0
        ).to(hidden.dtype)
        if self.settings.residual_enabled:
            packed = hidden + self.residual_scale.to(hidden.dtype) * packed
        if packed.shape != hidden.shape:
            raise RuntimeError(f"Optical output shape {tuple(packed.shape)} != input {tuple(hidden.shape)}")
        return packed

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.last_routing:
            zero = self.first_expert_phase.raw_phase.sum() * 0.0
            return zero, zero
        return self.last_routing["balance_loss"], self.last_routing["importance_loss"]

    def phase_dc_loss(self) -> torch.Tensor:
        return self.first_expert_phase.dc_loss() + self.second_phase.dc_loss()

    def parameter_breakdown(self) -> dict[str, Any]:
        router = sum(p.numel() for p in self.router.parameters())
        first = self.first_expert_phase.raw_phase.numel()
        second = self.second_phase.raw_phase.numel()
        affine = sum(
            p.numel() for name, p in self.named_parameters()
            if name.startswith("oeo_") or name.startswith("final_")
        )
        return {
            "architecture": "per_token_topk_adapter_free_optical_moe",
            "hidden_size": self.hidden_size,
            "input_adapter_parameters": 0,
            "output_adapter_parameters": 0,
            "router_parameters": router,
            "first_expert_phase_parameters": first,
            "second_plane_mode": self.settings.second_plane_mode,
            "second_phase_parameters": second,
            "normalization_affine_parameters": affine,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "active_panel_size": [self.layout.active_height, self.layout.active_width],
            "canvas_size": self.layout.canvas_size,
            "token_group_size": [self.layout.group_height, self.layout.group_width],
            "max_tokens": self.layout.max_tokens,
            "num_experts": self.layout.num_experts,
            "top_k": self.settings.top_k,
            "shared_expert_phase_across_tokens": self.settings.share_expert_phase_across_tokens,
            "k_space_enabled": self.settings.k_space_enabled,
            "k_space_pass_fraction": self.propagator.pass_fraction,
        }
