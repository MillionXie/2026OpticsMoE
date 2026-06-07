import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]


class PhaseLayer(nn.Module):
    """Trainable phase-only optical modulation layer."""

    def __init__(
        self,
        grid_size: GridSize,
        parameterization: str = "unconstrained",
        init: str = "uniform",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size

        self.grid_size = (int(height), int(width))
        self.parameterization = parameterization
        self.raw_phase = nn.Parameter(torch.empty(self.grid_size, dtype=torch.float32))
        self.reset_parameters(init=init, init_std=init_std)

    def reset_parameters(self, init: str, init_std: float) -> None:
        if init in {"zeros", "identity"}:
            nn.init.zeros_(self.raw_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "small_normal"}:
            nn.init.normal_(self.raw_phase, mean=0.0, std=init_std)
        elif init == "kaiming_phase":
            # Experimental ablation only. Phase masks act through exp(i*phase),
            # so this real-valued initialization has no default optical meaning.
            nn.init.kaiming_uniform_(self.raw_phase, a=math.sqrt(5.0))
        else:
            raise ValueError(f"Unsupported phase init: {init}")

    def get_phase(self) -> torch.Tensor:
        if self.parameterization == "unconstrained":
            return self.raw_phase
        if self.parameterization == "sigmoid":
            return 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        if self.parameterization == "cos":
            return math.pi * (torch.cos(self.raw_phase) + 1.0)
        raise ValueError(f"Unsupported phase parameterization: {self.parameterization}")

    def get_phase_wrapped(self) -> torch.Tensor:
        return torch.remainder(self.get_phase(), 2.0 * math.pi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected field shape [B, H, W], got {tuple(field.shape)}")
        phase = self.get_phase().to(field.device)
        modulation = torch.exp(1j * phase).to(torch.complex64)
        return field.to(torch.complex64) * modulation
