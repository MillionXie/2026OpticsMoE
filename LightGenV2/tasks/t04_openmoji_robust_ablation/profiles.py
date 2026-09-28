"""One optical/BMP amplitude contract; cumulative training-only perturbations."""
from __future__ import annotations

import torch
from torch.nn import functional as F

PROFILES = {
    "r0_base": {"ccd": False, "dc30": False, "grid": False},
    "r1_ccd": {"ccd": True, "dc30": False, "grid": False},
    "r2_ccd_dc30": {"ccd": True, "dc30": True, "grid": False},
    "r3_ccd_dc30_grid": {"ccd": True, "dc30": True, "grid": True},
}


def bounded(x: torch.Tensor) -> torch.Tensor:
    """Preserve complex phase and exact zeros; output amplitude in [0,1]."""
    a = x.abs()
    return x * (torch.tanh(a / 0.5) / a.clamp_min(1e-8))


def bmp_amplitude(field: torch.Tensor) -> torch.Tensor:
    """Quantize the *same* bounded input used by simulation, with no peak rescale."""
    amplitude = bounded(field).abs()
    return torch.round(255.0 * amplitude).to(torch.uint8)


def grid_roundtrip(x: torch.Tensor, native: int) -> torch.Tensor:
    """Differentiable 17→8→17 μm raster approximation, not 8 μm propagation."""
    if x.is_complex():
        return torch.complex(grid_roundtrip(x.real, native), grid_roundtrip(x.imag, native))
    shape = x.shape[-2:]
    flat = x.reshape(-1, 1, *shape)
    up = F.interpolate(flat, size=(native, native), mode="bilinear", align_corners=False)
    down = F.interpolate(up, size=shape, mode="area")
    return down.reshape(x.shape)


def install(model, group: str) -> None:
    """Patch propagation entry points before training and every inference load."""
    profile = PROFILES[group]
    for path in model._optical_paths():
        original = path._simulate_detector_roi
        path.gain_min = 1.0
        path.gain_max = 1.0
        path.offset_fraction = 0.03 if profile["ccd"] else 0.0
        path.read_noise_fraction = 0.01 if profile["ccd"] else 0.0
        path.ccd_noise_distribution = "none"
        path.zero_order_enabled = profile["dc30"]
        path.amplitude_zero_order_intensity_min = 0.0
        path.amplitude_zero_order_intensity_max = 0.0
        path.phase_zero_order_intensity_min = 0.30 if profile["dc30"] else 0.0
        path.phase_zero_order_intensity_max = path.phase_zero_order_intensity_min

        def detector(field, modulation, shifts, *, phase_support=None, original=original):
            field = bounded(field)
            if profile["grid"]:
                field = grid_roundtrip(field, 1101 if field.shape[-1] == 518 else 1016)
            return original(field, modulation, shifts, phase_support=phase_support)

        path._simulate_detector_roi = detector
        router = path.core.router
        original_router = router._simulate
        router.input_shift_pixels = 0
        router.phase_shift_pixels = 0
        router.ccd_shift_pixels = 0

        def route(fields, original=original_router):
            fields = bounded(fields)
            if profile["grid"]:
                fields = grid_roundtrip(fields, 1016)
            intensity = original(fields)
            if model.training and profile["ccd"]:
                ref = intensity.mean(dim=(-2, -1), keepdim=True).detach()
                intensity = (intensity + 0.03 * ref + 0.01 * ref * torch.randn_like(intensity)).clamp_min(0)
            return intensity

        router._simulate = route


def assert_contract() -> None:
    z = torch.tensor([0.0, 0.01, 1.0, 100.0], requires_grad=True)
    a = bounded(z)
    a.sum().backward()
    assert a[0] == 0 and a.min() >= 0 and a.max() <= 1
    assert torch.isfinite(z.grad).all()
    assert (torch.round(255 * a) / 255 - a).abs().max() <= 0.5 / 255 + 1e-6
    assert torch.equal(bmp_amplitude(z.detach()), torch.round(255 * a.detach()).to(torch.uint8))
    x = torch.randn(1, 8, 8, requires_grad=True)
    y = grid_roundtrip(x, 17)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(x.grad).all()
