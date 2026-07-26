from __future__ import annotations

from typing import Any

import torch


class SharpnessAwareMinimizer:
    """Two-step SAM controller around an already configured base optimizer.

    The base optimizer owns learning rates, Adam moments and per-group weight
    decay.  This controller only applies and removes the sharpness
    perturbation, so schedulers continue to operate on ``base_optimizer``.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
    ) -> None:
        if rho <= 0:
            raise ValueError("SAM rho must be positive")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.eps = float(eps)
        self._perturbations: dict[torch.Tensor, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> float:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eps)
        self._perturbations.clear()
        for group in self.base_optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                multiplier: torch.Tensor | float
                multiplier = parameter.detach().square() if self.adaptive else 1.0
                perturbation = multiplier * parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)
                self._perturbations[parameter] = perturbation
        if not self._perturbations:
            raise RuntimeError("SAM first_step found no gradients")
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.detach().cpu())

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        if not self._perturbations:
            raise RuntimeError("SAM second_step called before first_step")
        for parameter, perturbation in self._perturbations.items():
            parameter.sub_(perturbation)
        self._perturbations.clear()
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def cancel_step(self, zero_grad: bool = True) -> None:
        """Restore unperturbed parameters after a failed second evaluation."""
        for parameter, perturbation in self._perturbations.items():
            parameter.sub_(perturbation)
        self._perturbations.clear()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        device: torch.device | None = None
        for group in self.base_optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                device = parameter.device if device is None else device
                multiplier: torch.Tensor | float
                multiplier = parameter.detach().abs() if self.adaptive else 1.0
                norms.append((multiplier * parameter.grad).norm(p=2).to(device))
        if not norms:
            if device is None:
                device = torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.stack(norms).norm(p=2)

    def specification(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "rho": self.rho,
            "adaptive": self.adaptive,
            "base_optimizer": type(self.base_optimizer).__name__,
        }
