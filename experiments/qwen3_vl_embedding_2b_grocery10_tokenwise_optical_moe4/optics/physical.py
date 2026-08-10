from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AngularSpectrumPropagator(nn.Module):
    """Differentiable angular-spectrum propagation on a square sampled grid."""

    def __init__(
        self,
        wavelength_m: float,
        pixel_pitch_m: float,
        grid_size: int,
        distance_m: float,
        *,
        k_space_enabled: bool = False,
        theta_max_deg: float = 0.65,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(
            self.grid_size, d=float(pixel_pitch_m), dtype=torch.float64
        )
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wave_number = 2.0 * math.pi / float(wavelength_m)
        kx, ky = 2.0 * math.pi * fx, 2.0 * math.pi * fy
        argument = wave_number**2 - kx.square() - ky.square()
        passing = argument >= 0.0
        if k_space_enabled:
            if not 0.0 < theta_max_deg <= 90.0:
                raise ValueError("theta_max_deg must be in (0,90]")
            radial = torch.sqrt(kx.square() + ky.square())
            passing &= radial <= wave_number * math.sin(math.radians(theta_max_deg))
        kz = torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * self.distance_m * kz).to(torch.complex64)
        transfer = torch.where(passing, transfer, torch.zeros_like(transfer))
        self.register_buffer("transfer_function", transfer, persistent=False)
        self.register_buffer("pass_mask", passing, persistent=False)

    @property
    def pass_fraction(self) -> float:
        return float(self.pass_mask.float().mean())

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_size, self.grid_size)
        if field.ndim != 3 or tuple(field.shape[-2:]) != expected:
            raise ValueError(f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}")
        field = field.to(torch.complex64)
        if self.distance_m == 0.0:
            return field
        return torch.fft.ifft2(torch.fft.fft2(field) * self.transfer_function).to(torch.complex64)


class PhaseTensor(nn.Module):
    """Phase-only tensor with raw-parameter initialization and optional dropout."""

    def __init__(
        self,
        shape: Sequence[int],
        *,
        parameterization: str,
        init: str,
        init_std: float,
        dropout_mode: str,
        dropout_p: float,
        dropout_block_size: int,
        dropout_batch_shared: bool,
    ) -> None:
        super().__init__()
        self.parameterization = str(parameterization)
        self.dropout_mode = str(dropout_mode)
        self.dropout_p = float(dropout_p)
        self.dropout_block_size = int(dropout_block_size)
        self.dropout_batch_shared = bool(dropout_batch_shared)
        self.raw_phase = nn.Parameter(torch.empty(tuple(int(v) for v in shape)))
        if init == "zeros":
            nn.init.zeros_(self.raw_phase)
        elif init == "uniform":
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init == "normal":
            nn.init.normal_(self.raw_phase, 0.0, float(init_std))
        else:
            raise ValueError(f"Unsupported phase init {init!r}")

    def phase(self) -> torch.Tensor:
        if self.parameterization == "sigmoid":
            return 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        if self.parameterization == "unconstrained":
            return self.raw_phase
        raise ValueError(f"Unsupported phase parameterization {self.parameterization!r}")

    def modulation(self, batch_size: int, *, token_axis: bool) -> torch.Tensor:
        modulation = torch.exp(1j * self.phase()).to(torch.complex64)
        if not self.training or self.dropout_mode == "none" or self.dropout_p <= 0.0:
            return modulation
        prefix = 1 if self.dropout_batch_shared else int(batch_size)
        spatial = modulation.shape[-2:]
        leading = modulation.shape[:-2]
        if self.dropout_mode == "phase_bypass":
            keep = torch.rand((prefix, *leading, *spatial), device=modulation.device) >= self.dropout_p
        elif self.dropout_mode == "block_phase_bypass":
            block = max(1, self.dropout_block_size)
            low_h, low_w = math.ceil(spatial[0] / block), math.ceil(spatial[1] / block)
            keep = torch.rand((prefix, *leading, low_h, low_w), device=modulation.device) >= self.dropout_p
            keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)
            keep = keep[..., : spatial[0], : spatial[1]]
        else:
            raise RuntimeError(f"Unsupported active phase dropout {self.dropout_mode!r}")
        expanded = modulation.unsqueeze(0)
        return torch.where(keep, expanded, torch.ones_like(expanded))

    def dc_loss(self) -> torch.Tensor:
        phasor = torch.exp(1j * self.phase())
        return phasor.mean(dim=(-2, -1)).abs().square().mean()


def tokenwise_layer_norm(
    values: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """LayerNorm the final 2-D optical crop; optional affine broadcasts by expert."""
    normalized = F.layer_norm(values.float(), values.shape[-2:], eps=float(eps))
    if weight is not None:
        normalized = normalized * weight + bias
    return normalized
