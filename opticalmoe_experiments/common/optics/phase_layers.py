import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]
PHASE_DROPOUT_MODES = {"none", "phase_bypass", "block_phase_bypass"}


class PhaseLayer(nn.Module):
    """Trainable phase-only optical modulation with optional phase bypass dropout."""

    def __init__(
        self,
        grid_size: GridSize,
        parameterization: str = "unconstrained",
        init: str = "identity",
        init_std: float = 0.02,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        if phase_dropout_mode not in PHASE_DROPOUT_MODES:
            raise ValueError(f"Unsupported phase dropout mode: {phase_dropout_mode}")
        if not 0.0 <= float(phase_dropout_p) < 1.0:
            raise ValueError("phase_dropout_p must satisfy 0 <= p < 1.")
        self.grid_size = (int(height), int(width))
        self.parameterization = parameterization
        self.phase_dropout_mode = phase_dropout_mode
        self.phase_dropout_p = float(phase_dropout_p)
        self.phase_dropout_block_size = int(phase_dropout_block_size)
        self.phase_dropout_batch_shared = bool(phase_dropout_batch_shared)
        self.phase_dropout_active = True
        self.last_phase_dropout_mask = None
        self.raw_phase = nn.Parameter(torch.empty(self.grid_size, dtype=torch.float32))
        self.reset_parameters(init, init_std)

    def reset_parameters(self, init: str, init_std: float) -> None:
        if init in {"identity", "zeros"}:
            nn.init.zeros_(self.raw_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "small_normal"}:
            nn.init.normal_(self.raw_phase, 0.0, init_std)
        elif init == "kaiming_phase":
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

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase_dropout_active = bool(active)

    def _dropout_enabled(self) -> bool:
        return (
            self.training
            and self.phase_dropout_active
            and self.phase_dropout_mode != "none"
            and self.phase_dropout_p > 0.0
        )

    def _sample_mask(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        mask_batch = 1 if self.phase_dropout_batch_shared else int(batch_size)
        keep_prob = 1.0 - self.phase_dropout_p
        if self.phase_dropout_mode == "phase_bypass":
            return (torch.rand((mask_batch, height, width), device=device) < keep_prob).float()
        if self.phase_dropout_mode == "block_phase_bypass":
            block = max(1, int(self.phase_dropout_block_size))
            low_h = int(math.ceil(float(height) / block))
            low_w = int(math.ceil(float(width) / block))
            keep = (torch.rand((mask_batch, low_h, low_w), device=device) < keep_prob).float()
            return keep.repeat_interleave(block, -2).repeat_interleave(block, -1)[:, :height, :width]
        raise RuntimeError(f"Unexpected phase dropout mode: {self.phase_dropout_mode}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        field = field.to(torch.complex64)
        phase = self.get_phase().to(device=field.device, dtype=torch.float32)
        modulation = torch.exp(1j * phase).to(torch.complex64)
        if not self._dropout_enabled():
            self.last_phase_dropout_mask = None
            return field * modulation
        keep = self._sample_mask(field.shape[0], phase.shape[-2], phase.shape[-1], field.device)
        self.last_phase_dropout_mask = keep.detach()
        keep_complex = keep.to(torch.complex64)
        modulation = keep_complex * modulation.unsqueeze(0) + (1.0 - keep_complex)
        return field * modulation

