"""Six-pass logical CCD adapter, BEFORE readout and AFTER grid remapping.

This deliberately does not use the teacher's propagator-level OpticalBoundary:
that boundary assumed a 518 grid and is not suitable for the 1101 device grid.
No device SDK calls or automatic SLM switching are performed here.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import types

STAGES = ("vision_router", "vision_expert", "vision_global",
          "language_router", "language_expert", "language_global")
OWNERS = ("parallel_router", "parallel_optics", "parallel_optics",
          "serial_router", "serial_optics", "serial_optics")


@contextmanager
def measured_ccd_boundary(model, measured=None):
    """Substitute a contiguous measured prefix and capture logical inputs.

    Intensities are linear, nonnegative, [B,478,478] or [B,518,518]; conversion
    from camera ADC/ROI to these values belongs to a separately verified session.
    """
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import phase_planes
    measured = measured or {}
    if tuple(measured) != STAGES[:len(measured)]:
        raise ValueError("Measured CCDs must form the ordered six-stage prefix")
    if model.training:
        raise ValueError("Measured replay requires eval mode")
    planes, supports = phase_planes(model, "temporal")
    tap = {"amplitudes": {}, "detectors": {}, "stages": []}
    originals = []

    def make_detector(owner, original):
        def detector(module, field):
            index = len(tap["stages"])
            if index >= len(STAGES) or OWNERS[index] != owner:
                raise RuntimeError("Unexpected optical stage order")
            stage = STAGES[index]
            tap["stages"].append(stage)
            g = module.geometry
            margin, n = g.active_margin, g.active_size
            active = field[:, margin:margin+n, margin:margin+n]
            phase = torch.as_tensor(planes[stage], device=field.device)
            support = torch.as_tensor(supports[stage], device=field.device)
            levels = module.settings.phase_quantization_levels
            if levels:
                step = 2 * math.pi / (levels - 1)
                phase = (phase / step).round() * step
            eta = module.settings.unmodulated_power_fraction_eval
            modulation = math.sqrt(1-eta) * torch.exp(1j*phase) + math.sqrt(eta)
            modulation = torch.where(support, modulation, torch.ones_like(modulation))
            if modulation.abs().min() < 1e-6:
                raise ValueError("Near-zero modulation prevents safe amplitude recovery")
            amplitude = active / modulation
            if amplitude.imag.abs().max() > 1e-4 or amplitude.real.min() < -1e-5:
                raise RuntimeError("Cannot recover nonnegative amplitude; check phase/session identity")
            tap["amplitudes"][stage] = amplitude.real.clamp_min(0).detach()
            if stage in measured:
                value = measured[stage].to(field.device).float()
                if value.shape == active.shape:
                    value = torch.nn.functional.pad(value, (margin, margin, margin, margin))
                if value.shape != field.shape or not torch.isfinite(value).all() or value.min() < 0:
                    raise ValueError("Measured CCD shape/range does not match logical readout grid")
                guard = module._guard_energy(value)
            else:
                value, guard = original(field)
            tap["detectors"][stage] = value.detach()
            return value, guard
        return detector

    try:
        for owner in dict.fromkeys(OWNERS):
            module = getattr(model, owner)
            original = module._detector
            originals.append((module, original))
            module._detector = types.MethodType(make_detector(owner, original), module)
        yield tap
    finally:
        for module, original in originals:
            module._detector = original


def rasterize_phase(phase, *, model_pitch_um=17.0, device_pitch_um=8.0):
    """Nearest phasor mapping, not bilinear interpolation of wrapped angles."""
    import torch
    from torch.nn import functional as F
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1] or device_pitch_um <= 0:
        raise ValueError("Expected square phase plane and positive pitch")
    size = round(phase.shape[0] * model_pitch_um / device_pitch_um)
    unit = torch.exp(1j * phase.float())
    real = F.interpolate(unit.real[None, None], size=(size, size), mode="nearest")[0, 0]
    imag = F.interpolate(unit.imag[None, None], size=(size, size), mode="nearest")[0, 0]
    return torch.atan2(imag, real).remainder(2 * math.pi)
