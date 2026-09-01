from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


FeedbackMode = Literal["bp", "fa_pretrained", "fa_random"]


def physical_phase(raw_phase: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.sigmoid(raw_phase)


def propagate(field: torch.Tensor, transfer_function: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(field, dim=(-2, -1)) * transfer_function, dim=(-2, -1)).to(
        torch.complex64
    )


class AngularSpectrumPropagator(nn.Module):
    """Unpadded angular-spectrum propagation with no optional k-space cutoff."""

    def __init__(self, size: int, wavelength_m: float, pixel_size_m: float, distance_m: float) -> None:
        super().__init__()
        self.size = int(size)
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.distance_m = float(distance_m)
        frequency = torch.fft.fftfreq(self.size, d=self.pixel_size_m, dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * ((1.0 / self.wavelength_m) ** 2 - fx.square() - fy.square())
        propagating = argument >= 0.0
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(propagating, transfer, torch.zeros_like(transfer)),
            persistent=True,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"Expected [B,{self.size},{self.size}], got {tuple(field.shape)}")
        return propagate(field.to(torch.complex64), self.transfer_function)


class _FixedFeedbackOptical(torch.autograd.Function):
    """Current forward/local phase gradient with a frozen input-feedback connector."""

    @staticmethod
    def forward(
        ctx,
        amplitude: torch.Tensor,
        raw_phase: torch.Tensor,
        feedback_phase: torch.Tensor,
        transfer_function: torch.Tensor,
    ) -> torch.Tensor:
        phase = physical_phase(raw_phase)
        modulation = torch.exp(1j * phase).to(torch.complex64)
        field = torch.complex(amplitude.float(), torch.zeros_like(amplitude.float())) * modulation
        output = propagate(field, transfer_function)
        ctx.save_for_backward(amplitude, raw_phase, feedback_phase, transfer_function)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        amplitude, raw_phase, feedback_phase, transfer_function = ctx.saved_tensors
        grad_output = grad_output.to(torch.complex64)
        with torch.enable_grad():
            # The local phase update uses the current phase and current forward input,
            # exactly as the local weight update in feedback alignment does.
            phase_variable = raw_phase.detach().requires_grad_(True)
            current_modulation = torch.exp(1j * physical_phase(phase_variable)).to(torch.complex64)
            current_field = torch.complex(amplitude.detach(), torch.zeros_like(amplitude)) * current_modulation
            current_output = propagate(current_field, transfer_function)
            (phase_gradient,) = torch.autograd.grad(
                current_output,
                phase_variable,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )

            # Only the error connector sent to the preceding stage is replaced.
            amplitude_variable = amplitude.detach().requires_grad_(True)
            feedback_modulation = torch.exp(1j * feedback_phase.detach()).to(torch.complex64)
            feedback_field = torch.complex(
                amplitude_variable, torch.zeros_like(amplitude_variable)
            ) * feedback_modulation
            feedback_output = propagate(feedback_field, transfer_function)
            (amplitude_gradient,) = torch.autograd.grad(
                feedback_output,
                amplitude_variable,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )
        return amplitude_gradient, phase_gradient, None, None


class LearnableResidualMixer(nn.Module):
    """Positive normalized optical/skip weights, both trained with ordinary BP."""

    def __init__(self, main_init: float, skip_init: float) -> None:
        super().__init__()
        if main_init <= 0.0 or skip_init <= 0.0 or abs(main_init + skip_init - 1.0) > 1e-6:
            raise ValueError("Residual weights must be positive and sum to one")
        initial = torch.tensor([float(main_init), float(skip_init)], dtype=torch.float32)
        self.logits = nn.Parameter(initial.log())

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, optical_amplitude: torch.Tensor, previous_amplitude: torch.Tensor) -> torch.Tensor:
        weights = self.weights()
        return weights[0] * optical_amplitude + weights[1] * previous_amplitude


class OpticalOEOStage(nn.Module):
    """Phase modulation -> propagation -> CCD -> LN -> ReLU -> residual reload."""

    def __init__(
        self,
        *,
        size: int,
        wavelength_m: float,
        pixel_size_m: float,
        distance_m: float,
        layernorm_eps: float,
        residual_main_init: float,
        residual_skip_init: float,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.layernorm_eps = float(layernorm_eps)
        self.raw_phase = nn.Parameter(torch.zeros(self.size, self.size, dtype=torch.float32))
        self.propagator = AngularSpectrumPropagator(size, wavelength_m, pixel_size_m, distance_m)
        self.residual = LearnableResidualMixer(residual_main_init, residual_skip_init)
        self.register_buffer("feedback_phase", torch.full((self.size, self.size), math.pi), persistent=True)
        self.feedback_mode: FeedbackMode = "bp"

    def phase(self) -> torch.Tensor:
        return physical_phase(self.raw_phase)

    def set_feedback(self, mode: FeedbackMode, phase: torch.Tensor | None = None) -> None:
        if mode not in {"bp", "fa_pretrained", "fa_random"}:
            raise ValueError(f"Unsupported feedback mode: {mode}")
        if mode != "bp":
            if phase is None or tuple(phase.shape) != (self.size, self.size):
                raise ValueError(f"A [{self.size},{self.size}] feedback phase is required for {mode}")
            self.feedback_phase.copy_(phase.detach().to(self.feedback_phase))
        self.feedback_mode = mode

    def optical_forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        if self.feedback_mode == "bp":
            modulation = torch.exp(1j * self.phase()).to(torch.complex64)
            field = torch.complex(amplitude.float(), torch.zeros_like(amplitude.float())) * modulation
            return self.propagator(field)
        return _FixedFeedbackOptical.apply(
            amplitude,
            self.raw_phase,
            self.feedback_phase,
            self.propagator.transfer_function,
        )

    def forward(self, amplitude: torch.Tensor, *, return_details: bool = False):
        if amplitude.ndim != 3 or tuple(amplitude.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"Expected [B,{self.size},{self.size}], got {tuple(amplitude.shape)}")
        field = self.optical_forward(amplitude)
        intensity = field.abs().square().float()
        normalized = F.layer_norm(
            intensity,
            normalized_shape=(self.size, self.size),
            weight=None,
            bias=None,
            eps=self.layernorm_eps,
        )
        activated = F.relu(normalized)
        reloaded = self.residual(activated, amplitude)
        if not return_details:
            return reloaded
        return reloaded, {
            "field": field,
            "intensity": intensity,
            "normalized": normalized,
            "activated": activated,
            "reloaded": reloaded,
            "residual_weights": self.residual.weights(),
        }


def phasor_operator_distance(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (torch.exp(1j * current) - torch.exp(1j * reference)).abs().square().mean().sqrt()


def phasor_operator_coherence(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.cos(current - reference).mean()
