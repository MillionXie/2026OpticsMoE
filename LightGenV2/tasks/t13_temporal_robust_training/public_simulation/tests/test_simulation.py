from pathlib import Path

import torch

from lgvq_temporal.camera import sample_camera
from lgvq_temporal.model import bounded_amplitude
from lgvq_temporal.settings import load_settings
from lgvq_temporal.simulation import synthetic_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_default_physics():
    settings = load_settings(ROOT / "src/lgvq_temporal/configs/simulation.yaml")
    assert settings.modulator_pixel_pitch_um == 8.0
    assert settings.unmodulated_power_fraction_min == 0.30
    assert settings.unmodulated_power_fraction_max == 0.30
    assert settings.ccd_noise_enabled
    assert settings.input_shift_pixels == settings.phase_shift_pixels == settings.ccd_shift_pixels == 0


def test_bounded_amplitude_and_gradient():
    value = torch.tensor([0.0, 0.25, 0.5, 2.0], requires_grad=True)
    actual = bounded_amplitude(value)
    assert torch.allclose(actual, torch.tanh(value / 0.5))
    assert bool((actual >= 0).all()) and bool((actual <= 1).all())
    actual.sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def test_camera_sampling_uses_finite_gradient():
    profile = {"electrons_per_intensity_unit": 3072, "read_noise_electrons": 10,
               "dark_electrons": 0}
    value = torch.full((2, 8, 8), 0.25, requires_grad=True)
    torch.manual_seed(7)
    noisy = sample_camera(value, profile)
    noisy.sum().backward()
    assert bool(torch.isfinite(noisy).all())
    assert value.grad is not None and bool(torch.all(value.grad == 1))


def test_full_field_smoke():
    settings = load_settings(ROOT / "src/lgvq_temporal/configs/simulation.yaml", synthetic=True)
    torch.manual_seed(163)
    assert synthetic_smoke(settings)["status"] == "passed"
