from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def aperture_linear_indices(canvas_size: int, apertures: list) -> torch.Tensor:
    groups = []
    for aperture in apertures:
        rows = torch.arange(aperture.y0, aperture.y1, dtype=torch.long)
        columns = torch.arange(aperture.x0, aperture.x1, dtype=torch.long)
        groups.append((rows[:, None] * int(canvas_size) + columns[None, :]).reshape(-1))
    result = torch.stack(groups)
    if result.unique().numel() != result.numel():
        raise ValueError("Expert apertures overlap")
    return result


class AngularSpectrumPropagator(nn.Module):
    def __init__(
        self,
        *,
        wavelength_m: float,
        pixel_size_m: float,
        grid_size: int,
        distance_m: float,
        k_space_constraint_enabled: bool = False,
        theta_max_deg: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(self.grid_size, d=float(pixel_size_m), dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * (
            (1.0 / float(wavelength_m)) ** 2 - fx.square() - fy.square()
        )
        propagating = argument >= 0
        if k_space_constraint_enabled:
            if not 0.0 < theta_max_deg <= 90.0:
                raise ValueError("theta_max_deg must be in (0,90]")
            radial_wave_number = 2.0 * math.pi * torch.sqrt(fx.square() + fy.square())
            cutoff = (2.0 * math.pi / float(wavelength_m)) * math.sin(math.radians(theta_max_deg))
            propagating &= radial_wave_number <= cutoff
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(propagating, transfer, torch.zeros_like(transfer)),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}"
            )
        field = field.to(torch.complex64)
        if self.distance_m == 0:
            return field
        return torch.fft.ifft2(
            torch.fft.fft2(field) * self.transfer_function
        ).to(torch.complex64)


class PhaseLayer(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        parameterization: str,
        init: str,
        init_std: float,
        dropout_mode: str = "none",
        dropout_p: float = 0.0,
        dropout_block_size: int = 8,
        dropout_batch_shared: bool = True,
    ) -> None:
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
            nn.init.uniform_(self.raw_phase, 0, 2 * math.pi)
        elif init in {"normal", "small_normal"}:
            nn.init.normal_(self.raw_phase, 0, init_std)
        else:
            raise ValueError(f"Unsupported phase initialization {init!r}")

    def phase(self) -> torch.Tensor:
        if self.parameterization == "sigmoid":
            return 2 * math.pi * torch.sigmoid(self.raw_phase)
        if self.parameterization == "unconstrained":
            return self.raw_phase
        raise ValueError(f"Unsupported phase parameterization {self.parameterization!r}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        modulation = torch.exp(1j * self.phase()).to(torch.complex64)
        if self.training and self.dropout_active and self.dropout_mode != "none" and self.dropout_p > 0:
            batch = 1 if self.dropout_batch_shared else field.shape[0]
            if self.dropout_mode == "phase_bypass":
                keep = torch.rand(batch, self.size, self.size, device=field.device) >= self.dropout_p
            elif self.dropout_mode == "block_phase_bypass":
                block = max(1, self.dropout_block_size)
                low = math.ceil(self.size / block)
                keep = torch.rand(batch, low, low, device=field.device) >= self.dropout_p
                keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)
                keep = keep[:, : self.size, : self.size]
            else:
                raise RuntimeError(f"Unsupported phase dropout mode {self.dropout_mode!r}")
            keep = keep.to(torch.complex64)
            modulation = keep * modulation.unsqueeze(0) + (1 - keep)
        return field.to(torch.complex64) * modulation

    def set_dropout_active(self, active: bool) -> None:
        self.dropout_active = bool(active)


class ExpertSquareDetectionReload(nn.Module):
    """Square-law detector, independent expert LN, activation and zero-phase reload."""

    def __init__(
        self,
        canvas_size: int,
        apertures: list,
        *,
        eps: float,
        nonlinearity: str,
        per_expert_enabled: bool,
        elementwise_affine: bool,
    ) -> None:
        super().__init__()
        self.canvas_size = int(canvas_size)
        self.apertures = apertures
        self.expert_size = apertures[0].size
        self.eps = float(eps)
        self.nonlinearity = str(nonlinearity)
        self.per_expert_enabled = bool(per_expert_enabled)
        self.elementwise_affine = bool(elementwise_affine)
        self.register_buffer(
            "aperture_indices",
            aperture_linear_indices(canvas_size, apertures),
            persistent=False,
        )
        if self.elementwise_affine:
            count = len(apertures) if self.per_expert_enabled else 1
            size = self.expert_size if self.per_expert_enabled else self.canvas_size
            self.affine_weight = nn.Parameter(torch.ones(count, size, size))
            self.affine_bias = nn.Parameter(torch.zeros(count, size, size))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def forward(
        self,
        field: torch.Tensor,
        *,
        selected_experts: torch.Tensor | None,
        routing_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        intensity = field.to(torch.complex64).abs().square().float()
        if not self.per_expert_enabled:
            if selected_experts is not None or routing_weights is not None:
                raise RuntimeError("Hard routing requires independent per-expert normalization")
            normalized = F.layer_norm(
                intensity, intensity.shape[-2:], eps=self.eps
            )
            if self.affine_weight is not None:
                normalized = normalized * self.affine_weight[0] + self.affine_bias[0]
            output = (
                F.relu(normalized)
                if self.nonlinearity == "relu"
                else F.softplus(normalized)
            )
            return torch.complex(output, torch.zeros_like(output))

        batch = intensity.shape[0]
        flat_indices = self.aperture_indices.reshape(-1)
        crops = intensity.flatten(1).index_select(1, flat_indices).reshape(
            batch, len(self.apertures), self.expert_size, self.expert_size
        )
        normalized = F.layer_norm(crops, crops.shape[-2:], eps=self.eps)
        if self.affine_weight is not None:
            normalized = (
                normalized * self.affine_weight.unsqueeze(0)
                + self.affine_bias.unsqueeze(0)
            )
        activated = (
            F.relu(normalized)
            if self.nonlinearity == "relu"
            else F.softplus(normalized)
        )
        if routing_weights is not None:
            activated = activated * routing_weights.to(activated)[:, :, None, None]
        if selected_experts is not None:
            activated = activated * selected_experts.to(activated)[:, :, None, None]
        output = intensity.new_zeros(batch, self.canvas_size * self.canvas_size)
        output = output.scatter(
            1,
            flat_indices.unsqueeze(0).expand(batch, -1),
            activated.reshape(batch, -1),
        ).reshape_as(intensity)
        return torch.complex(output, torch.zeros_like(output))
