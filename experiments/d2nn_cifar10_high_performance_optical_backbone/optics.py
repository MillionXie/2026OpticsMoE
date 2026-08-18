from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


FeedbackMode = Literal["bp", "fa_pretrained", "fa_random"]


def physical_phase(raw_phase: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained parameter to the phase-only SLM interval [0, 2pi]."""

    return 2.0 * math.pi * torch.sigmoid(raw_phase)


def _spatial_rms(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value.square().mean(dim=(-2, -1), keepdim=True).add(eps).sqrt()


def rms_normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / _spatial_rms(value, eps)


def _propagate(field: torch.Tensor, transfer_function: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(field.to(torch.complex64), dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(
        spectrum * transfer_function,
        dim=(-2, -1),
        norm="ortho",
    ).to(torch.complex64)


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
        return _propagate(field, self.transfer_function)


class _FixedFeedbackOptical(torch.autograd.Function):
    """Use the current phase locally and a frozen phase for the preceding-stage error connector."""

    @staticmethod
    def forward(
        ctx,
        amplitude: torch.Tensor,
        raw_phase: torch.Tensor,
        feedback_phase: torch.Tensor,
        transfer_function: torch.Tensor,
    ) -> torch.Tensor:
        modulation = torch.exp(1j * physical_phase(raw_phase)).to(torch.complex64)
        field = torch.complex(amplitude.float(), torch.zeros_like(amplitude.float())) * modulation.unsqueeze(0)
        output = _propagate(field, transfer_function)
        ctx.save_for_backward(amplitude, raw_phase, feedback_phase, transfer_function)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        amplitude, raw_phase, feedback_phase, transfer_function = ctx.saved_tensors
        upstream = grad_output.to(torch.complex64)
        with torch.enable_grad():
            # The local phase update remains the exact gradient of the current
            # forward operator. Only the connector to the preceding stage is fixed.
            phase_variable = raw_phase.detach().requires_grad_(True)
            current_modulation = torch.exp(1j * physical_phase(phase_variable)).to(torch.complex64)
            current_field = (
                torch.complex(amplitude.detach(), torch.zeros_like(amplitude))
                * current_modulation.unsqueeze(0)
            )
            current_output = _propagate(current_field, transfer_function)
            (phase_gradient,) = torch.autograd.grad(
                current_output,
                phase_variable,
                grad_outputs=upstream,
                retain_graph=False,
                create_graph=False,
            )

            amplitude_variable = amplitude.detach().requires_grad_(True)
            feedback_modulation = torch.exp(1j * feedback_phase.detach()).to(torch.complex64)
            feedback_field = (
                torch.complex(amplitude_variable, torch.zeros_like(amplitude_variable))
                * feedback_modulation.unsqueeze(0)
            )
            feedback_output = _propagate(feedback_field, transfer_function)
            (amplitude_gradient,) = torch.autograd.grad(
                feedback_output,
                amplitude_variable,
                grad_outputs=upstream,
                retain_graph=False,
                create_graph=False,
            )
        return amplitude_gradient, phase_gradient, None, None


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


def _bounded_logit(initial: float, maximum: float) -> torch.Tensor:
    if maximum <= 0.0:
        return torch.tensor(-20.0)
    probability = min(max(float(initial) / float(maximum), 1e-5), 1.0 - 1e-5)
    return torch.tensor(math.log(probability / (1.0 - probability)))


class LowResolutionElectronicTransform(nn.Module):
    """Process the bypass at reduced resolution to spend parameters without excessive full-plane MACs."""

    def __init__(self, channels: int, hidden_channels: int, downsample_factor: int) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.downsample_factor = int(downsample_factor)
        groups = 8 if hidden % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        target_size = value.shape[-2:]
        reduced = F.avg_pool2d(
            value,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        delta = self.layers(reduced)
        return F.interpolate(delta, size=target_size, mode="bilinear", align_corners=False)


class ElectronicSkipProcessor(nn.Module):
    """Apply a deliberately small electronic transform to the bounded bypass branch."""

    def __init__(
        self,
        *,
        channels: int,
        mode: str,
        hidden_channels: int,
        downsample_factor: int,
        scale_init: float,
        scale_max: float,
        long_skip_enabled: bool,
        long_skip_weight_init: float,
        long_skip_weight_max: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.scale_max = float(scale_max)
        self.long_skip_enabled = bool(long_skip_enabled)
        self.long_skip_weight_max = float(long_skip_weight_max)
        self.eps = float(eps)
        hidden = int(hidden_channels)
        if self.mode == "identity":
            self.transform = None
        elif self.mode == "pointwise":
            self.transform = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            )
        elif self.mode == "depthwise":
            self.transform = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GroupNorm(channels, channels),
                nn.GELU(),
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            )
        elif self.mode == "lowres":
            self.transform = LowResolutionElectronicTransform(
                channels,
                hidden,
                downsample_factor,
            )
        else:
            raise ValueError(f"Unsupported electronic skip mode: {self.mode}")
        if self.transform is not None:
            # Start from the pretrained identity bypass. The new branch first
            # learns its final projection and only then perturbs earlier layers.
            if isinstance(self.transform, nn.Sequential):
                nn.init.zeros_(self.transform[-1].weight)
            self.scale_logit = nn.Parameter(_bounded_logit(scale_init, scale_max))
        if self.long_skip_enabled:
            self.long_skip_logit = nn.Parameter(
                _bounded_logit(long_skip_weight_init, long_skip_weight_max)
            )

    def transform_scale(self) -> torch.Tensor:
        if self.transform is None:
            return torch.zeros(())
        return self.scale_max * torch.sigmoid(self.scale_logit)

    def long_skip_weight(self) -> torch.Tensor:
        if not self.long_skip_enabled:
            return torch.zeros(())
        return self.long_skip_weight_max * torch.sigmoid(self.long_skip_logit)

    def forward(
        self,
        value: torch.Tensor,
        *,
        long_skip: torch.Tensor | None = None,
        disable_transform: bool = False,
        disable_long_skip: bool = False,
    ) -> torch.Tensor:
        base = rms_normalize(value.float(), self.eps)
        if self.long_skip_enabled and long_skip is not None and not disable_long_skip:
            source = rms_normalize(long_skip.float(), self.eps)
            weight = self.long_skip_weight().to(device=base.device, dtype=base.dtype)
            base = rms_normalize((1.0 - weight) * base + weight * source, self.eps)
        if self.transform is None or disable_transform:
            return base
        scale = self.transform_scale().to(device=base.device, dtype=base.dtype)
        processed = F.relu(base + scale * self.transform(base))
        return rms_normalize(processed, self.eps)


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
        electronic_skip_mode: str = "identity",
        electronic_skip_hidden_channels: int = 12,
        electronic_skip_downsample_factor: int = 4,
        electronic_skip_scale_init: float = 0.10,
        electronic_skip_scale_max: float = 0.25,
        long_skip_enabled: bool = False,
        long_skip_weight_init: float = 0.10,
        long_skip_weight_max: float = 0.25,
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
        self.electronic_skip = ElectronicSkipProcessor(
            channels=self.channels,
            mode=electronic_skip_mode,
            hidden_channels=electronic_skip_hidden_channels,
            downsample_factor=electronic_skip_downsample_factor,
            scale_init=electronic_skip_scale_init,
            scale_max=electronic_skip_scale_max,
            long_skip_enabled=long_skip_enabled,
            long_skip_weight_init=long_skip_weight_init,
            long_skip_weight_max=long_skip_weight_max,
            eps=self.eps,
        )
        self.register_buffer(
            "random_phase",
            2.0 * math.pi * torch.rand(self.channels, self.size, self.size, generator=generator),
            persistent=True,
        )
        self.register_buffer(
            "feedback_phase",
            torch.full((self.channels, self.size, self.size), math.pi, dtype=torch.float32),
            # Runtime connector state is reconstructed from the frozen source
            # checkpoint and must not change forward-checkpoint compatibility.
            persistent=False,
        )
        self.feedback_mode: FeedbackMode = "bp"

    def phase(self) -> torch.Tensor:
        return physical_phase(self.raw_phase)

    def set_feedback(self, mode: FeedbackMode, phase: torch.Tensor | None = None) -> None:
        if mode not in {"bp", "fa_pretrained", "fa_random"}:
            raise ValueError(f"Unsupported feedback mode: {mode}")
        if mode != "bp":
            if phase is None or tuple(phase.shape) != (self.channels, self.size, self.size):
                raise ValueError(
                    f"A [{self.channels},{self.size},{self.size}] feedback phase is required for {mode}"
                )
            self.feedback_phase.copy_(phase.detach().to(self.feedback_phase))
        self.feedback_mode = mode

    def _optical_branch(self, amplitude: torch.Tensor, phase_override: torch.Tensor | None) -> torch.Tensor:
        # Complex FFTs remain float32 even when the electronic head uses AMP.
        with torch.autocast(device_type=amplitude.device.type, enabled=False):
            value = amplitude.float()
            if self.feedback_mode == "bp" or phase_override is not None:
                phase = self.phase() if phase_override is None else phase_override
                modulation = torch.exp(1j * phase.float()).to(torch.complex64)
                field = torch.complex(value, torch.zeros_like(value)) * modulation.unsqueeze(0)
                propagated = self.propagator(field)
            else:
                propagated = _FixedFeedbackOptical.apply(
                    value,
                    self.raw_phase,
                    self.feedback_phase,
                    self.propagator.transfer_function,
                )
            intensity = propagated.abs().square().float()
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
        long_skip: torch.Tensor | None = None,
        disable_electronic_skip: bool = False,
        disable_long_skip: bool = False,
        return_details: bool = False,
    ):
        expected = (self.channels, self.size, self.size)
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{self.channels},{self.size},{self.size}], got {tuple(amplitude.shape)}")
        skip = self.electronic_skip(
            amplitude,
            long_skip=long_skip,
            disable_transform=disable_electronic_skip,
            disable_long_skip=disable_long_skip,
        )
        if not self.normalize_branch_rms:
            skip = amplitude.float()
        if optical_off:
            output = skip
            optical = torch.zeros_like(skip)
        else:
            optical = self._optical_branch(amplitude, phase_override)
            output = self.residual(optical, skip)
        if not return_details:
            return output
        return output, {
            "optical_rms": _spatial_rms(optical, self.eps).mean().detach(),
            "skip_rms": _spatial_rms(skip, self.eps).mean().detach(),
            "output_rms": _spatial_rms(output, self.eps).mean().detach(),
            "optical_weight": self.residual.main_weight().detach(),
            "electronic_skip_scale": self.electronic_skip.transform_scale().detach(),
            "long_skip_weight": self.electronic_skip.long_skip_weight().detach(),
        }
