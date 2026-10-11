"""Frozen *intensity-response* feedback for the P11 optical interfaces.

The forward always uses the current mask.  A source response computes both
amplitude credit and mask updates at a frozen physical phase, using the current
input amplitude and current intensity error.  It is a surrogate update after
the current mask moves; it is not the derivative of the current forward.

Only the ideal optical branch is replaced.  Qwen token ordering, intensity
normalization, activation, residual gates, and electronic skip paths retain the
P11 contract.  This module does not change any shared FixedFeedbackSFT class.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MethodType
from typing import Literal

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


ResponseMode = Literal[
    "current_response", "source_full_response", "random_full_response"
]
OpticalAxis = Literal["token", "channel"]
RESPONSE_MODES = (
    "current_response", "source_full_response", "random_full_response"
)


def _axis_dimension(axis: OpticalAxis) -> int:
    if axis == "token":
        return -2
    if axis == "channel":
        return -1
    raise ValueError(f"Unsupported optical axis: {axis}")


def axis_transfer_1d(transfer: torch.Tensor, axis: OpticalAxis) -> torch.Tensor:
    """Extract a separable P11 transfer without silently accepting a 2-D optic."""

    dimension = _axis_dimension(axis)
    if transfer.ndim == 1:
        return transfer.detach().clone()
    if transfer.ndim != 2:
        raise ValueError("Transfer must be one-dimensional or a P11 2-D transfer")
    if dimension == -2:
        line = transfer[:, 0]
        expected = line[:, None].expand_as(transfer)
    else:
        line = transfer[0, :]
        expected = line[None, :].expand_as(transfer)
    if not torch.equal(transfer, expected):
        raise ValueError("Transfer is not constant along the relayed P11 axis")
    return line.detach().clone()


def axis_propagate(
    field: torch.Tensor,
    transfer_1d: torch.Tensor,
    axis: OpticalAxis,
    *,
    adjoint: bool = False,
) -> torch.Tensor:
    """One-axis FFT propagation; the adjoint conjugates H, never divides by H."""

    dimension = _axis_dimension(axis)
    if transfer_1d.ndim != 1 or transfer_1d.numel() != field.shape[dimension]:
        raise ValueError("One-dimensional transfer must match the propagation axis")
    complex_dtype = torch.complex128 if field.dtype in {
        torch.float64, torch.complex128
    } else torch.complex64
    field = field.to(complex_dtype)
    transfer = transfer_1d.to(device=field.device, dtype=complex_dtype)
    if adjoint:
        transfer = transfer.conj()
    shape = [1] * field.ndim
    shape[dimension] = transfer.numel()
    spectrum = torch.fft.fft(field, dim=dimension, norm="ortho")
    return torch.fft.ifft(
        spectrum * transfer.reshape(shape), dim=dimension, norm="ortho"
    )


def intensity_forward(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    transfer_1d: torch.Tensor,
    axis: OpticalAxis,
) -> torch.Tensor:
    """Ordinary differentiable intensity forward, also used by verification."""

    field = amplitude * torch.exp(1j * phase)
    return axis_propagate(field, transfer_1d, axis).abs().square()


def response_vjp(
    amplitude: torch.Tensor,
    response_phase: torch.Tensor,
    intensity_error: torch.Tensor,
    transfer_1d: torch.Tensor,
    axis: OpticalAxis,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real amplitude/physical-phase VJPs at the selected response mask.

    q = exp(-i phi_ref) P^H [e P(exp(i phi_ref) a)].  The phase VJP is
    summed over broadcast dimensions (notably the batch); amplitude credit is
    per sample.  The source is evaluated at live ``a`` and ``e`` every call.
    """

    modulation = torch.exp(1j * response_phase)
    source_field = axis_propagate(amplitude * modulation, transfer_1d, axis)
    q = modulation.conj() * axis_propagate(
        intensity_error * source_field, transfer_1d, axis, adjoint=True
    )
    amplitude_gradient = 2.0 * q.real
    phase_gradient = (2.0 * amplitude * q.imag).sum_to_size(response_phase.shape)
    return amplitude_gradient, phase_gradient


class _IntensityResponse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, amplitude, phase, response_phase, transfer_1d, axis):
        # In frozen modes no current phase or current complex field is saved.
        # Current phase determines only the forward intensity returned here.
        output = intensity_forward(amplitude, phase, transfer_1d, axis)
        ctx.save_for_backward(amplitude, response_phase, transfer_1d)
        ctx.axis = axis
        ctx.phase_shape = phase.shape
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, intensity_error):
        amplitude, response_phase, transfer_1d = ctx.saved_tensors
        amplitude_gradient, phase_gradient = response_vjp(
            amplitude, response_phase, intensity_error, transfer_1d, ctx.axis
        )
        # Broadcast reduction follows the trainable current phase's shape.
        # Its parameterization chain rule is outside this custom Function.
        if response_phase.shape != ctx.phase_shape:
            raise RuntimeError("Current and response phase shapes must match")
        return amplitude_gradient, phase_gradient, None, None, None


def intensity_response(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    transfer_1d: torch.Tensor,
    axis: OpticalAxis,
    *,
    mode: ResponseMode = "current_response",
    response_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Current intensity with either its exact VJP or a frozen complete VJP.

    ``phase`` and ``response_phase`` contain physical radians, not raw logits.
    No derivative is returned for the frozen reference or propagation transfer.
    Only first-order training is supported by this custom backward.
    """

    if mode not in RESPONSE_MODES:
        raise ValueError(f"Unsupported response mode: {mode}")
    if amplitude.dtype not in {torch.float32, torch.float64}:
        raise TypeError("Amplitude must be real float32 or float64")
    if phase.dtype != amplitude.dtype or phase.device != amplitude.device:
        raise ValueError("Current phase and amplitude must share dtype and device")
    if transfer_1d.requires_grad:
        raise ValueError("Propagation transfer is fixed and cannot require gradients")
    if mode == "current_response":
        selected_phase = phase.detach()
    else:
        if response_phase is None or response_phase.shape != phase.shape:
            raise ValueError("A frozen response phase matching the current phase is required")
        selected_phase = response_phase.detach().to(phase)
    return _IntensityResponse.apply(
        amplitude, phase, selected_phase, transfer_1d, axis
    )


def snapshot_source_phases(backbone: nn.Module) -> tuple[torch.Tensor, ...]:
    """Capture independent physical-phase tensors in the stages' physical layout."""

    return tuple(stage.phase().detach().clone() for stage in backbone.stages)


def _full_response_optical_branch(
    self,
    amplitude: torch.Tensor,
    phase_override: torch.Tensor | None,
    *,
    phase_shift_dy_dx: tuple[float, float] | None = None,
    phase_error_rad: torch.Tensor | None = None,
    detector_noise_relative_rms: float = 0.0,
    detector_generator: torch.Generator | None = None,
) -> torch.Tensor:
    from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
        rms_normalize,
    )
    from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
        qwen_field_to_row_major,
        row_major_field_to_qwen,
    )

    if (
        phase_override is not None
        or phase_shift_dy_dx is not None
        or phase_error_rad is not None
        or float(detector_noise_relative_rms) != 0.0
    ):
        raise ValueError("FFA pilot supports ideal optics only; deployment overrides/errors/noise are unsupported")
    with torch.autocast(device_type=amplitude.device.type, enabled=False):
        value = amplitude.float()
        physical_amplitude = (
            qwen_field_to_row_major(value, token_count=self.token_count)
            if self.optical_axis == "token" else value
        )
        intensity = intensity_response(
            physical_amplitude,
            self.phase().float(),
            self.ffa_transfer_1d,
            self.optical_axis,
            mode=self.ffa_response_mode,
            response_phase=self.ffa_response_phase,
        )
        # Exact copy of the P11 parent intensity/electronic boundary contract.
        mean = intensity.mean(dim=(-2, -1), keepdim=True)
        variance = intensity.var(dim=(-2, -1), keepdim=True, unbiased=False)
        activated = F.relu((intensity - mean) * torch.rsqrt(variance + self.eps))
        if self.normalize_branch_rms:
            activated = rms_normalize(activated, self.eps)
        return (
            row_major_field_to_qwen(activated, token_count=self.token_count)
            if self.optical_axis == "token" else activated
        )


def install_full_response(
    backbone: nn.Module,
    mode: ResponseMode,
    source_phases: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> None:
    """Replace only this backbone's P11 optical branches, preserving checkpoints.

    For source mode, omitted phases snapshot the masks at installation time.
    For random mode, omitted phases use the existing seeded P11 random buffers.
    Pass explicit source snapshots when reinstalling after a diagnostic switch.
    All added buffers are nonpersistent, so original forward checkpoints still
    load strictly; the experiment must record reference identity separately.
    """

    from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
        AxisOpticalOEOStage,
    )

    if mode not in RESPONSE_MODES:
        raise ValueError(f"Unsupported response mode: {mode}")
    stages = tuple(backbone.stages)
    if not stages:
        raise ValueError("No optical stages found")
    if source_phases is not None and len(source_phases) != len(stages):
        raise ValueError("One physical response phase per optical stage is required")
    prepared = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, AxisOpticalOEOStage):
            raise TypeError("FFA installation requires P11 AxisOpticalOEOStage instances")
        phase = (
            source_phases[index] if source_phases is not None else
            stage.random_phase if mode == "random_full_response" else stage.phase()
        )
        if phase.shape != stage.raw_phase.shape:
            raise ValueError(f"Response phase shape mismatch at stage {index}")
        if not torch.isfinite(phase).all():
            raise ValueError(f"Non-finite response phase at stage {index}")
        transfer = axis_transfer_1d(
            stage.propagator.transfer_function, stage.optical_axis
        )
        prepared.append((stage, phase.detach().to(stage.raw_phase).clone(), transfer))
    # Validate the entire backbone before changing any instance.
    for stage, phase, transfer in prepared:
        if not getattr(stage, "ffa_response_enabled", False):
            stage._ffa_had_instance_branch = "_optical_branch" in stage.__dict__
            stage._ffa_original_optical_branch = stage._optical_branch
        for name, value in (("ffa_response_phase", phase), ("ffa_transfer_1d", transfer)):
            if name in stage._buffers:
                setattr(stage, name, value)
            else:
                stage.register_buffer(name, value, persistent=False)
        stage.ffa_response_mode = mode
        stage.ffa_response_enabled = True
        stage._optical_branch = MethodType(_full_response_optical_branch, stage)


def uninstall_full_response(backbone: nn.Module) -> None:
    """Restore the exact previous instance/class branch for BP/legacy diagnostics."""

    for stage in backbone.stages:
        if not getattr(stage, "ffa_response_enabled", False):
            continue
        if stage._ffa_had_instance_branch:
            stage._optical_branch = stage._ffa_original_optical_branch
        else:
            delattr(stage, "_optical_branch")
        delattr(stage, "_ffa_original_optical_branch")
        delattr(stage, "_ffa_had_instance_branch")
        stage.ffa_response_enabled = False
