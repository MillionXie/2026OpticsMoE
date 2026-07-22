from __future__ import annotations

from typing import Iterable

import torch


class SAMController:
    """Two-step sharpness-aware update around an existing optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, parameters: Iterable[torch.nn.Parameter],
                 rho: float = 0.05, adaptive: bool = False) -> None:
        if rho <= 0: raise ValueError("SAM rho must be positive")
        self.optimizer = optimizer; self.parameters = [p for p in parameters if p.requires_grad]
        self.rho = float(rho); self.adaptive = bool(adaptive); self._perturbations: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        norms = []
        for parameter in self.parameters:
            if parameter.grad is None: continue
            scale = parameter.detach().abs() if self.adaptive else 1.0
            norms.append((scale * parameter.grad).norm(p=2))
        if not norms: raise RuntimeError("SAM first_step found no gradients")
        grad_norm = torch.stack([value.to(norms[0].device) for value in norms]).norm(p=2).clamp_min(1e-12)
        self._perturbations.clear()
        for parameter in self.parameters:
            if parameter.grad is None: continue
            direction = parameter.detach().square() * parameter.grad if self.adaptive else parameter.grad
            perturbation = direction * (self.rho / grad_norm)
            parameter.add_(perturbation); self._perturbations[id(parameter)] = perturbation
        if zero_grad: self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        for parameter in self.parameters:
            perturbation = self._perturbations.get(id(parameter))
            if perturbation is not None: parameter.sub_(perturbation)
        self._perturbations.clear(); self.optimizer.step()
        if zero_grad: self.optimizer.zero_grad(set_to_none=True)
