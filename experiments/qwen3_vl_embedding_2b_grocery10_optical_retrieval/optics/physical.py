from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def aperture_linear_indices(canvas_size: int, apertures: list) -> torch.Tensor:
    """Return unique flattened canvas indexes for ordered square apertures."""
    groups = []
    for aperture in apertures:
        rows = torch.arange(aperture.y0, aperture.y1, dtype=torch.long)
        columns = torch.arange(aperture.x0, aperture.x1, dtype=torch.long)
        groups.append((rows[:, None] * int(canvas_size) + columns[None, :]).reshape(-1))
    result = torch.stack(groups)
    if result.unique().numel() != result.numel():
        raise ValueError("Expert apertures must not overlap")
    return result


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


def _iter_phase_layers(module: object):
    """Yield unique phase planes from a module or DeepStack hook wrapper."""
    seen: set[int] = set()
    roots: list[nn.Module] = []
    if isinstance(module, nn.Module):
        roots.append(module)
    else:
        # DeepStackMultimodalReplacement is deliberately not an nn.Module; it
        # owns the two registered optical modules used by the forward hooks.
        for name in ("vision_surrogate", "language_surrogate", "backbone"):
            candidate = getattr(module, name, None)
            if isinstance(candidate, nn.Module):
                roots.append(candidate)
    for root in roots:
        for child in root.modules():
            if isinstance(child, PhaseLayer) and id(child) not in seen:
                seen.add(id(child))
                yield child


def phase_dc_loss(module: object) -> torch.Tensor:
    """Return the mean coherent zero-order power of all phase planes.

    The loss is evaluated independently for every physical phase plane before
    averaging.  Combining phasors from different masks first would allow two
    unrelated planes to cancel one another and would not describe either
    mask's zero-order diffraction efficiency.

    ``|mean(exp(i*phase))|^2`` is stationary at a perfectly uniform phase.  A
    nonzero initialization jitter is therefore required when this loss is
    expected to break a zero-initialized mask's spatial symmetry.
    """
    losses = []
    for child in _iter_phase_layers(module):
        phase = child.phase().float()
        mean_real = torch.cos(phase).mean()
        mean_imag = torch.sin(phase).mean()
        losses.append(mean_real.square() + mean_imag.square())
    if not losses:
        raise RuntimeError("phase_dc_loss requires at least one PhaseLayer")
    return torch.stack(losses).mean()


@torch.no_grad()
def phase_dc_statistics(module: object) -> dict[str, float | int]:
    """Detached per-model diagnostics for coherent phase-mask DC power."""
    rho_squared = []
    for child in _iter_phase_layers(module):
        phase = child.phase().detach().float()
        mean_real = torch.cos(phase).mean()
        mean_imag = torch.sin(phase).mean()
        rho_squared.append(mean_real.square() + mean_imag.square())
    if not rho_squared:
        raise RuntimeError("phase_dc_statistics requires at least one PhaseLayer")
    values = torch.stack(rho_squared)
    return {
        "phase_dc_current_loss": float(values.mean()),
        "phase_dc_rho_mean": float(values.sqrt().mean()),
        "phase_dc_rho_max": float(values.sqrt().max()),
        "phase_dc_plane_count": int(values.numel()),
    }


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
        self.canvas_size = int(canvas_size)
        self.expert_size = apertures[0].y1 - apertures[0].y0
        self.register_buffer(
            "aperture_indices",
            aperture_linear_indices(self.canvas_size, apertures),
            persistent=False,
        )
        if self.elementwise_affine:
            size = apertures[0].y1 - apertures[0].y0 if self.per_expert_enabled else int(canvas_size)
            count = len(apertures) if self.per_expert_enabled else 1
            self.affine_weight = nn.Parameter(torch.ones(count, size, size))
            self.affine_bias = nn.Parameter(torch.zeros(count, size, size))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self.capture_intermediate = False
        self.capture_sample_count = 1
        self.last_input_complex_field: torch.Tensor | None = None
        self.last_input_intensity: torch.Tensor | None = None
        self.last_output_amplitude: torch.Tensor | None = None

    def forward(
        self,
        field: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.capture_intermediate:
            count = min(self.capture_sample_count, len(field))
            self.last_input_complex_field = field[:count].detach().cpu().to(torch.complex64)
        intensity = field.to(torch.complex64).abs().square().float()
        return self.forward_intensity(
            intensity,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
        )

    def forward_intensity(
        self,
        intensity: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the electronic OEO transform to a measured CCD intensity.

        Hardware CCD files already contain square-law intensity.  Keeping this
        entry point separate prevents the common replay error of squaring a
        captured intensity a second time.
        """
        if intensity.ndim != 3 or tuple(intensity.shape[-2:]) != (
            self.canvas_size,
            self.canvas_size,
        ):
            raise ValueError(
                f"Measured expert CCD intensity must be [B,{self.canvas_size},"
                f"{self.canvas_size}], got {tuple(intensity.shape)}"
            )
        intensity = intensity.float()
        if not torch.isfinite(intensity).all():
            raise RuntimeError("Measured expert CCD intensity contains NaN or Inf")
        if torch.any(intensity < -1.0e-7):
            raise RuntimeError("Measured expert CCD intensity must be nonnegative")
        intensity = intensity.clamp_min(0.0)
        if self.capture_intermediate:
            count = min(self.capture_sample_count, len(intensity))
            self.last_input_intensity = intensity[:count].detach().cpu()
        expected = (intensity.shape[0], len(self.apertures))
        if selected_experts is not None:
            if tuple(selected_experts.shape) != expected:
                raise ValueError(f"selected_experts must have shape {expected}, got {tuple(selected_experts.shape)}")
            selected_experts = selected_experts.to(device=intensity.device, dtype=torch.bool)
        if routing_weights is not None:
            if tuple(routing_weights.shape) != expected:
                raise ValueError(f"routing_weights must have shape {expected}, got {tuple(routing_weights.shape)}")
            routing_weights = routing_weights.to(device=intensity.device, dtype=intensity.dtype)
        if self.per_expert_enabled:
            batch = intensity.shape[0]
            flat_indices = self.aperture_indices.reshape(-1)
            crops = intensity.flatten(1).index_select(1, flat_indices).reshape(
                batch, len(self.apertures), self.expert_size, self.expert_size
            )
            normalized = F.layer_norm(crops, crops.shape[-2:], weight=None, bias=None, eps=self.eps)
            if self.affine_weight is not None:
                normalized = normalized * self.affine_weight.unsqueeze(0) + self.affine_bias.unsqueeze(0)
            activated = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
            # Per-expert LayerNorm removes the incoming amplitude scale. Restore the
            # sample-dependent router coefficient only after normalization/activation.
            if routing_weights is not None:
                activated = activated * routing_weights[:, :, None, None]
            if selected_experts is not None:
                activated = activated * selected_experts[:, :, None, None].to(activated.dtype)
            output = intensity.new_zeros((batch, self.canvas_size * self.canvas_size)).scatter(
                1,
                flat_indices.unsqueeze(0).expand(batch, -1),
                activated.reshape(batch, -1),
            ).reshape_as(intensity)
        else:
            if selected_experts is not None or routing_weights is not None:
                raise RuntimeError("Hard expert gating and routing-weight restoration require per-expert LayerNorm")
            normalized = F.layer_norm(intensity, intensity.shape[-2:], weight=None, bias=None, eps=self.eps)
            if self.affine_weight is not None:
                normalized = normalized * self.affine_weight[0] + self.affine_bias[0]
            output = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
        if self.capture_intermediate:
            count = min(self.capture_sample_count, len(output))
            self.last_output_amplitude = output[:count].detach().cpu()
        return torch.complex(output, torch.zeros_like(output))

    def set_intermediate_capture(self, enabled: bool, sample_count: int = 1) -> None:
        if sample_count <= 0:
            raise ValueError("capture sample_count must be positive")
        self.capture_intermediate = bool(enabled)
        self.capture_sample_count = int(sample_count)
        if not enabled:
            self.last_input_complex_field = None
            self.last_input_intensity = None
            self.last_output_amplitude = None
