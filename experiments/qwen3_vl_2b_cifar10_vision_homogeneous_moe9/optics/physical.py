from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AngularSpectrumPropagator(nn.Module):
    def __init__(self, wavelength_m: float, pixel_size_m: float, grid_size: int, distance_m: float,
                 k_space_constraint_enabled: bool = False, theta_max_deg: float = 1.0) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(self.grid_size, d=float(pixel_size_m), dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * ((1.0 / float(wavelength_m)) ** 2 - fx.square() - fy.square())
        propagating = argument >= 0
        if k_space_constraint_enabled:
            if not 0.0 < theta_max_deg <= 90.0:
                raise ValueError("theta_max_deg must be in (0,90]")
            radial_wave_number = 2.0 * math.pi * torch.sqrt(fx.square() + fy.square())
            cutoff = (2.0 * math.pi / float(wavelength_m)) * math.sin(math.radians(theta_max_deg))
            propagating &= radial_wave_number <= cutoff
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer("transfer_function", torch.where(propagating, transfer, torch.zeros_like(transfer)), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.grid_size, self.grid_size):
            raise ValueError(f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}")
        field = field.to(torch.complex64)
        if self.distance_m == 0.0:
            return field
        return torch.fft.ifft2(torch.fft.fft2(field) * self.transfer_function).to(torch.complex64)


class PhaseLayer(nn.Module):
    def __init__(self, size: int, parameterization: str = "sigmoid", init: str = "zeros", init_std: float = 0.02,
                 dropout_mode: str = "none", dropout_p: float = 0.0, dropout_block_size: int = 8,
                 dropout_batch_shared: bool = True) -> None:
        super().__init__()
        self.size = int(size)
        self.parameterization = str(parameterization)
        self.dropout_mode = str(dropout_mode)
        self.dropout_p = float(dropout_p)
        self.dropout_block_size = int(dropout_block_size)
        self.dropout_batch_shared = bool(dropout_batch_shared)
        self.dropout_active = False
        self.raw_phase = nn.Parameter(torch.empty(self.size, self.size))
        if init in {"zeros", "identity"}:
            nn.init.zeros_(self.raw_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "small_normal"}:
            nn.init.normal_(self.raw_phase, 0.0, init_std)
        else:
            raise ValueError(f"Unsupported phase_init={init!r}")

    def phase(self) -> torch.Tensor:
        if self.parameterization == "sigmoid":
            return 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        if self.parameterization == "unconstrained":
            return self.raw_phase
        raise ValueError(f"Unsupported phase parameterization {self.parameterization!r}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        modulation = torch.exp(1j * self.phase()).to(torch.complex64)
        if self.training and self.dropout_active and self.dropout_mode != "none" and self.dropout_p > 0.0:
            batch = 1 if self.dropout_batch_shared else field.shape[0]
            if self.dropout_mode == "phase_bypass":
                keep = torch.rand(batch, self.size, self.size, device=field.device) >= self.dropout_p
            elif self.dropout_mode == "block_phase_bypass":
                block = max(1, self.dropout_block_size)
                low = math.ceil(self.size / block)
                keep = torch.rand(batch, low, low, device=field.device) >= self.dropout_p
                keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)[:, :self.size, :self.size]
            else:
                raise RuntimeError(f"Unsupported active phase dropout mode {self.dropout_mode!r}")
            keep = keep.to(torch.complex64)
            modulation = keep * modulation.unsqueeze(0) + (1.0 - keep)
        return field.to(torch.complex64) * modulation

    def set_dropout_active(self, active: bool) -> None:
        self.dropout_active = bool(active)


class SquareDetectionLayerNormReload(nn.Module):
    """Per-expert, non-affine LayerNorm followed by activation and zero-phase reload."""

    def __init__(self, canvas_size: int, apertures: list, eps: float, nonlinearity: str,
                 per_expert_enabled: bool = True, elementwise_affine: bool = False) -> None:
        super().__init__()
        self.apertures = apertures
        self.eps = float(eps)
        self.nonlinearity = str(nonlinearity)
        self.per_expert_enabled = bool(per_expert_enabled)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            size = apertures[0].y1 - apertures[0].y0 if self.per_expert_enabled else int(canvas_size)
            count = len(apertures) if self.per_expert_enabled else 1
            self.affine_weight = nn.Parameter(torch.ones(count, size, size))
            self.affine_bias = nn.Parameter(torch.zeros(count, size, size))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = field.to(torch.complex64).abs().square().float()
        if self.per_expert_enabled:
            output = torch.zeros_like(intensity)
            for index, aperture in enumerate(self.apertures):
                crop = intensity[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
                normalized = F.layer_norm(crop, crop.shape[-2:], weight=None, bias=None, eps=self.eps)
                if self.affine_weight is not None:
                    normalized = normalized * self.affine_weight[index] + self.affine_bias[index]
                activated = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
                output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = activated
        else:
            normalized = F.layer_norm(intensity, intensity.shape[-2:], weight=None, bias=None, eps=self.eps)
            if self.affine_weight is not None:
                normalized = normalized * self.affine_weight[0] + self.affine_bias[0]
            output = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
        return torch.complex(output, torch.zeros_like(output))
