from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InputTopKRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int, pool_size: int, temperature: float,
                 input_layernorm_enabled: bool = True, input_layernorm_eps: float = 1e-5,
                 noise_std: float = 0.0, gate_init_std: float = 0.01) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.gate_init_std = float(gate_init_std)
        self.input_norm = (nn.LayerNorm(self.pool_size * self.pool_size, eps=float(input_layernorm_eps),
                                        elementwise_affine=False)
                           if input_layernorm_enabled else nn.Identity())
        self.gate = nn.Linear(self.pool_size * self.pool_size, self.num_experts)
        nn.init.normal_(self.gate.weight, 0.0, self.gate_init_std)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(fields.float().unsqueeze(1), (self.pool_size, self.pool_size)).flatten(1)
        router_input = self.input_norm(pooled)
        logits = self.gate(router_input)
        # Noisy top-k gating (Shazeer et al. 2017): train-time Gaussian logit
        # noise forces exploration so the router escapes the uniform attractor.
        # At eval time noise is disabled and routing is deterministic.
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        _, indices = torch.topk(probabilities, self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        sparse = probabilities * selected
        weights = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        importance = probabilities.mean(0)
        load = selected.float().mean(0) / float(self.top_k)
        balance = float(self.num_experts) * torch.sum(importance * load)
        importance_loss = float(self.num_experts) * torch.sum(importance.square()) - 1.0
        # The ordinary importance loss can be almost zero even when deterministic
        # Top-k selection has collapsed: probabilities may be close to uniform
        # while the same experts win by a very small margin for every sample.
        # Preserve the exact hard-load value in the forward pass, but use the
        # soft probabilities as a straight-through gradient surrogate.  This is
        # opt-in at the training-loss level, so legacy configurations are
        # numerically unchanged.
        hard_load = selected.float().mean(0) / float(self.top_k)
        soft_load = probabilities.mean(0)
        load_st = hard_load + soft_load - soft_load.detach()
        hard_load_balance_loss = (
            float(self.num_experts) * torch.sum(load_st.square()) - 1.0
        )
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1).mean() / math.log(float(self.num_experts))
        return {"logits": logits, "probabilities": probabilities, "weights": weights, "selected_mask": selected,
                "selected_indices": indices, "balance_loss": balance, "importance_loss": importance_loss,
                "hard_load_balance_loss": hard_load_balance_loss,
                "normalized_entropy": entropy, "importance": importance, "load": load}

    @torch.no_grad()
    def update_load_bias(self, selected_mask: torch.Tensor, update_rate: float) -> None:
        """Move the existing gate bias toward balanced deterministic Top-k load.

        This auxiliary-loss-free update is computed from training-batch routing
        only. It changes which experts win Top-k without introducing a new
        inference parameter; selected routing weights are still computed from
        the ordinary logits/probabilities.
        """
        rate = float(update_rate)
        if rate <= 0.0:
            return
        if selected_mask.ndim != 2 or selected_mask.shape[1] != self.num_experts:
            raise ValueError("selected_mask has an incompatible shape")
        selection_frequency = selected_mask.float().mean(dim=0)
        target = float(self.top_k) / float(self.num_experts)
        correction = rate * (target - selection_frequency)
        correction = correction - correction.mean()
        self.gate.bias.add_(correction.to(self.gate.bias))


class ElectronicAmplitudeRouter(nn.Module):
    """Electronic top-k gate for a directly addressed amplitude SLM.

    This module deliberately has no optical prompt, grating, lens phase, or
    complex transmission.  It only predicts sparse sample-dependent weights.
    The optical core places weighted copies into the selected expert apertures.
    """

    def __init__(self, geometry, top_k: int, pool_size: int, temperature: float,
                 input_layernorm_enabled: bool = True,
                 input_layernorm_eps: float = 1e-5,
                 noise_std: float = 0.0, gate_init_std: float = 0.01) -> None:
        super().__init__()
        self.geometry = geometry
        self.router = InputTopKRouter(
            geometry.num_experts,
            top_k,
            pool_size,
            temperature,
            input_layernorm_enabled,
            input_layernorm_eps,
            noise_std,
            gate_init_std,
        )

    def forward(self, input_fields: torch.Tensor) -> dict[str, torch.Tensor]:
        routing = self.router(input_fields)
        return {
            **routing,
            "router_implementation": "electronic_amplitude_topk",
            "phase_prompt_used": False,
            "amplitude_phase_relay": "ideal_4f_identity",
        }

    def set_noise_std(self, value: float) -> None:
        """Update train-time exploration noise (used for annealing schedules)."""
        self.router.noise_std = max(0.0, float(value))

    def update_load_bias(self, selected_mask: torch.Tensor, update_rate: float) -> None:
        self.router.update_load_bias(selected_mask, update_rate)
