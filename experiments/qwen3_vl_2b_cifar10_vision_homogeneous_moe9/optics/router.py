from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InputTopKRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int, pool_size: int, temperature: float,
                 input_layernorm_enabled: bool = True, input_layernorm_eps: float = 1e-5) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        self.input_norm = (nn.LayerNorm(self.pool_size * self.pool_size, eps=float(input_layernorm_eps),
                                        elementwise_affine=False)
                           if input_layernorm_enabled else nn.Identity())
        self.gate = nn.Linear(self.pool_size * self.pool_size, self.num_experts)
        nn.init.normal_(self.gate.weight, 0.0, 0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(fields.float().unsqueeze(1), (self.pool_size, self.pool_size)).flatten(1)
        router_input = self.input_norm(pooled)
        logits = self.gate(router_input)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        _, indices = torch.topk(probabilities, self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        sparse = probabilities * selected
        weights = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        importance = probabilities.mean(0)
        load = selected.float().mean(0) / float(self.top_k)
        balance = float(self.num_experts) * torch.sum(importance * load)
        importance_loss = float(self.num_experts) * torch.sum(importance.square()) - 1.0
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1).mean() / math.log(float(self.num_experts))
        return {"logits": logits, "probabilities": probabilities, "weights": weights, "selected_mask": selected,
                "selected_indices": indices, "balance_loss": balance, "importance_loss": importance_loss,
                "normalized_entropy": entropy, "importance": importance, "load": load}


class GlobalRouterPrompt(nn.Module):
    """Uniform per-cell routing amplitude and one continuous global quadratic phase."""

    def __init__(self, geometry, wavelength_m: float, pixel_size_m: float, focal_length_m: float,
                 top_k: int, pool_size: int, temperature: float, input_layernorm_enabled: bool = True,
                 input_layernorm_eps: float = 1e-5) -> None:
        super().__init__()
        self.geometry = geometry
        self.router = InputTopKRouter(geometry.num_experts, top_k, pool_size, temperature,
                                      input_layernorm_enabled, input_layernorm_eps)
        cell_masks = []
        for row in range(3):
            for column in range(3):
                y0 = geometry.active_start + row * geometry.expert_pitch
                x0 = geometry.active_start + column * geometry.expert_pitch
                mask = torch.zeros(geometry.canvas_size, geometry.canvas_size)
                mask[y0:y0 + geometry.expert_pitch, x0:x0 + geometry.expert_pitch] = 1.0
                cell_masks.append(mask)
        center = geometry.canvas_size // 2
        axis = (torch.arange(geometry.canvas_size, dtype=torch.float64) - center) * float(pixel_size_m)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        phase = -math.pi * (x.square() + y.square()) / (float(wavelength_m) * float(focal_length_m))
        phase = torch.remainder(phase, 2.0 * math.pi).float() * geometry.active_mask()
        self.register_buffer("cell_masks", torch.stack(cell_masks), persistent=False)
        self.register_buffer("global_lens_phase", phase, persistent=False)

    def forward(self, input_fields: torch.Tensor) -> dict[str, torch.Tensor]:
        routing = self.router(input_fields)
        amplitude = torch.einsum("be,ehw->bhw", routing["weights"], self.cell_masks)
        transmission = amplitude.to(torch.complex64) * torch.exp(1j * self.global_lens_phase).to(torch.complex64)
        return {**routing, "prompt_amplitude": amplitude, "prompt_phase": self.global_lens_phase, "transmission": transmission}

