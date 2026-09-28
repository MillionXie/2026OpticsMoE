"""Explicit Poisson signal + independent zero-mean read noise in electron units.

No borrowed ACCEL camera constants. Calibration maps model intensity to electrons;
the supplied defaults are a disclosed pilot, not an identified physical camera.
"""
from contextlib import contextmanager


def validate_profile(profile):
    for name in ("electrons_per_intensity_unit", "read_noise_electrons", "dark_electrons"):
        value = float(profile[name])
        if not __import__("math").isfinite(value) or value < 0:
            raise ValueError(f"Invalid camera parameter: {name}")
    if profile["electrons_per_intensity_unit"] <= 0:
        raise ValueError("Camera conversion must be positive")


def sample_camera(clean, profile, *, scale=1.0):
    """STE gradient only: sampled output is not a differentiable Poisson draw.

    X ~ Poisson(k*max(I,0)+d), R ~ Normal(0,sigma_e**2).
    Y = max((X + R - d)/k, 0). Dark subtraction is expected-value subtraction.
    scale is a controlled sensitivity scan, not a calibrated illumination scan.
    """
    import torch
    validate_profile(profile)
    if not __import__("math").isfinite(scale) or scale < 0:
        raise ValueError("Noise scale must be finite/nonnegative")
    if scale == 0:
        return clean
    k = float(profile["electrons_per_intensity_unit"]) / scale**2
    dark = float(profile["dark_electrons"])
    sigma = float(profile["read_noise_electrons"]) / scale
    expected = clean.detach().clamp_min(0) * k + dark
    noisy = (torch.poisson(expected) + torch.randn_like(clean) * sigma - dark) / k
    sampled = noisy.clamp_min(0)
    return clean + (sampled - clean).detach()


@contextmanager
def camera_operator(profile, *, evaluation_scale=None):
    """Patch only detector noise; preserve dropout mode."""
    from . import model as optics
    original = optics._perturb_ccd

    def perturb(clean, settings, *, training):
        enabled = training and settings.ccd_noise_enabled if evaluation_scale is None else evaluation_scale > 0
        if not enabled:
            return clean
        return sample_camera(clean, profile, scale=1.0 if evaluation_scale is None else evaluation_scale)

    optics._perturb_ccd = perturb
    try:
        yield
    finally:
        optics._perturb_ccd = original
