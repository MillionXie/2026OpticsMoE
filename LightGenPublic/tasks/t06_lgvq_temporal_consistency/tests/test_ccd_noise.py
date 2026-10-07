from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _torch_and_perturb():
    try:
        import torch
    except (ImportError, OSError) as error:
        pytest.skip(f"PyTorch runtime is unavailable on this host: {error}")
    sys.path.insert(0, str(ROOT / "runtime"))
    from LightGenV2.tasks.t06_video_quality_assessment.models.multivideo9x4 import (
        _perturb_ccd,
        _resample_complex_field,
    )

    return torch, _perturb_ccd, _resample_complex_field


def _settings(enabled: bool):
    return SimpleNamespace(
        ccd_noise_enabled=enabled,
        ccd_biased_gaussian_enabled=True,
        ccd_noise_mean_fraction=0.03,
        ccd_noise_std_fraction=0.03,
        ccd_noise_min_fraction=-0.03,
        ccd_noise_max_fraction=0.12,
        ccd_shot_noise_enabled=True,
        ccd_shot_photons_per_mean=4096.0,
    )


def test_disabled_ccd_noise_is_bitwise_identity_and_consumes_no_rng() -> None:
    torch, perturb, _ = _torch_and_perturb()
    clean = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    torch.manual_seed(17)
    before = torch.random.get_rng_state().clone()
    actual = perturb(clean, _settings(False), training=True)
    after = torch.random.get_rng_state()
    assert actual is clean
    assert torch.equal(before, after)


def test_ccd_noise_is_training_only_and_nonnegative() -> None:
    torch, perturb, _ = _torch_and_perturb()
    clean = torch.ones(2, 32, 32)
    settings = _settings(True)
    assert perturb(clean, settings, training=False) is clean
    torch.manual_seed(17)
    noisy = perturb(clean, settings, training=True)
    assert not torch.equal(noisy, clean)
    assert torch.isfinite(noisy).all()
    assert float(noisy.min()) >= 0.0


def test_complex_resampling_preserves_constant_field_and_gradient() -> None:
    torch, _, resample = _torch_and_perturb()
    phase = torch.full((1, 4, 4), 0.7, requires_grad=True)
    amplitude = torch.full((1, 4, 4), 2.0, requires_grad=True)
    field = amplitude * torch.exp(1j * phase)
    output = resample(field, 9)
    assert output.shape == (1, 9, 9)
    assert torch.allclose(output.abs(), torch.full_like(output.abs(), 2.0))
    assert torch.allclose(
        torch.angle(output), torch.full_like(torch.angle(output), 0.7), atol=1.0e-6
    )
    output.real.mean().backward()
    assert amplitude.grad is not None and torch.isfinite(amplitude.grad).all()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()
