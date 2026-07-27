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


class ElectronicAmplitudeRouter(nn.Module):
    """Electronic top-k gate for a directly addressed amplitude SLM.

    This module deliberately has no optical prompt, grating, lens phase, or
    complex transmission.  It only predicts sparse sample-dependent weights.
    The optical core places weighted copies into the selected expert apertures.
    """

    def __init__(self, geometry, top_k: int, pool_size: int, temperature: float,
                 input_layernorm_enabled: bool = True,
                 input_layernorm_eps: float = 1e-5) -> None:
        super().__init__()
        self.geometry = geometry
        self.router = InputTopKRouter(
            geometry.num_experts,
            top_k,
            pool_size,
            temperature,
            input_layernorm_enabled,
            input_layernorm_eps,
        )

    def forward(self, input_fields: torch.Tensor) -> dict[str, torch.Tensor]:
        routing = self.router(input_fields)
        return {
            **routing,
            "router_implementation": "electronic_amplitude_topk",
            "phase_prompt_used": False,
            "amplitude_phase_relay": "ideal_4f_identity",
        }
