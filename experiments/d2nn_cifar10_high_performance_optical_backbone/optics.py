from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def physical_phase(raw_phase: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained parameter to the phase-only SLM interval [0, 2pi]."""

    return 2.0 * math.pi * torch.sigmoid(raw_phase)


def _spatial_rms(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value.square().mean(dim=(-2, -1), keepdim=True).add(eps).sqrt()


def rms_normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / _spatial_rms(value, eps)


class AngularSpectrumPropagator(nn.Module):
    """Band-limited angular-spectrum free-space propagation."""

    def __init__(self, size: int, wavelength_m: float, pixel_size_m: float, distance_m: float) -> None:
        super().__init__()
        self.size = int(size)
        frequency = torch.fft.fftfreq(self.size, d=float(pixel_size_m), dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wave_number_sq = (1.0 / float(wavelength_m)) ** 2 - fx.square() - fy.square()
        propagating = wave_number_sq >= 0.0
        phase = 2.0 * math.pi * float(distance_m) * torch.sqrt(wave_number_sq.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(propagating, transfer, torch.zeros_like(transfer)),
            persistent=True,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"Expected [B,C,{self.size},{self.size}], got {tuple(field.shape)}")
        spectrum = torch.fft.fft2(field.to(torch.complex64), dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(
            spectrum * self.transfer_function,
            dim=(-2, -1),
            norm="ortho",
        ).to(torch.complex64)


class ResidualMixer(nn.Module):
    """Mix optical and bypass amplitudes while exposing the optical fraction."""

    def __init__(self, mode: str, main_init: float, main_min: float) -> None:
        super().__init__()
        self.mode = str(mode)
        self.main_min = float(main_min)
        if self.mode == "fixed":
            self.register_buffer("fixed_main", torch.tensor(float(main_init)), persistent=True)
        elif self.mode == "learned":
            initial = torch.tensor([float(main_init), 1.0 - float(main_init)]).clamp_min(1e-6)
            self.logits = nn.Parameter(initial.log())
        elif self.mode == "constrained":
            span = max(1.0 - self.main_min, 1e-8)
            probability = min(max((float(main_init) - self.main_min) / span, 1e-5), 1.0 - 1e-5)
            self.logit = nn.Parameter(torch.tensor(math.log(probability / (1.0 - probability))))
        elif self.mode != "none":
            raise ValueError(f"Unsupported residual mode: {self.mode}")

    def main_weight(self) -> torch.Tensor:
        if self.mode == "none":
            return torch.ones((), device=next(self.buffers(), torch.empty(0)).device)
        if self.mode == "fixed":
            return self.fixed_main
        if self.mode == "learned":
            return torch.softmax(self.logits, dim=0)[0]
        return self.main_min + (1.0 - self.main_min) * torch.sigmoid(self.logit)

    def forward(self, optical: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        main = self.main_weight().to(device=optical.device, dtype=optical.dtype)
        return main * optical + (1.0 - main) * skip


class OpticalOEOStage(nn.Module):
    """Phase mask -> propagation -> square-law CCD -> electronic nonlinearity -> reload."""

    def __init__(
        self,
        *,
        size: int,
        channels: int,
        wavelength_m: float,
        pixel_size_m: float,
        distance_m: float,
        phase_init_std: float,
        layernorm_eps: float,
        residual_mode: str,
        residual_main_init: float,
        residual_main_min: float,
        normalize_branch_rms: bool,
        random_seed: int,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.channels = int(channels)
        self.eps = float(layernorm_eps)
        self.normalize_branch_rms = bool(normalize_branch_rms)
        generator = torch.Generator().manual_seed(int(random_seed))
        self.raw_phase = nn.Parameter(
            torch.randn(self.channels, self.size, self.size, generator=generator) * float(phase_init_std)
        )
        self.propagator = AngularSpectrumPropagator(size, wavelength_m, pixel_size_m, distance_m)
        self.residual = ResidualMixer(residual_mode, residual_main_init, residual_main_min)
        self.register_buffer(
            "random_phase",
            2.0 * math.pi * torch.rand(self.channels, self.size, self.size, generator=generator),
            persistent=True,
        )

    def phase(self) -> torch.Tensor:
        return physical_phase(self.raw_phase)

    def _optical_branch(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # Complex FFTs remain float32 even when the electronic head uses AMP.
        with torch.autocast(device_type=amplitude.device.type, enabled=False):
            value = amplitude.float()
            modulation = torch.exp(1j * phase.float()).to(torch.complex64)
            field = torch.complex(value, torch.zeros_like(value)) * modulation.unsqueeze(0)
            intensity = self.propagator(field).abs().square().float()
            mean = intensity.mean(dim=(-2, -1), keepdim=True)
            variance = intensity.var(dim=(-2, -1), keepdim=True, unbiased=False)
            activated = F.relu((intensity - mean) * torch.rsqrt(variance + self.eps))
            if self.normalize_branch_rms:
                activated = rms_normalize(activated, self.eps)
            return activated

    def forward(
        self,
        amplitude: torch.Tensor,
        *,
        phase_override: torch.Tensor | None = None,
        optical_off: bool = False,
        return_details: bool = False,
    ):
        expected = (self.channels, self.size, self.size)
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{self.channels},{self.size},{self.size}], got {tuple(amplitude.shape)}")
        skip = rms_normalize(amplitude.float(), self.eps) if self.normalize_branch_rms else amplitude.float()
        if optical_off:
            output = skip
            optical = torch.zeros_like(skip)
        else:
            optical = self._optical_branch(amplitude, self.phase() if phase_override is None else phase_override)
            output = self.residual(optical, skip)
        if not return_details:
            return output
        return output, {
            "optical_rms": _spatial_rms(optical, self.eps).mean().detach(),
            "skip_rms": _spatial_rms(skip, self.eps).mean().detach(),
            "output_rms": _spatial_rms(output, self.eps).mean().detach(),
            "optical_weight": self.residual.main_weight().detach(),
        }
