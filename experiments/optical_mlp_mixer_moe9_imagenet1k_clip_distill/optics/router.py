from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ElectronicTopKRouter(nn.Module):
    """Input-dependent electronic router for direct amplitude-SLM loading."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        pool_size: int,
        temperature: float,
        input_layernorm_enabled: bool,
        input_layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        self.input_norm = (
            nn.LayerNorm(
                self.pool_size * self.pool_size,
                eps=float(input_layernorm_eps),
                elementwise_affine=False,
            )
            if input_layernorm_enabled
            else nn.Identity()
        )
        self.gate = nn.Linear(self.pool_size * self.pool_size, self.num_experts)
        nn.init.normal_(self.gate.weight, mean=0, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(
            fields.float().unsqueeze(1), (self.pool_size, self.pool_size)
        ).flatten(1)
        logits = self.gate(self.input_norm(pooled))
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        indices = torch.topk(probabilities, self.top_k, dim=-1).indices
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        sparse = probabilities * selected
        weights = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        importance = probabilities.mean(0)
        load = selected.float().mean(0) / float(self.top_k)
        balance_loss = float(self.num_experts) * torch.sum(importance * load)
        importance_loss = float(self.num_experts) * torch.sum(importance.square()) - 1
        entropy = -(
            probabilities.clamp_min(1e-12).log() * probabilities
        ).sum(-1).mean() / math.log(float(self.num_experts))
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "importance": importance,
            "load": load,
            "balance_loss": balance_loss,
            "importance_loss": importance_loss,
            "normalized_entropy": entropy,
            "router_implementation": "electronic_direct_amplitude_topk",
        }
