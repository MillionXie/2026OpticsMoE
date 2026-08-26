from pathlib import Path

import torch

from ..optical_blocks import _bounded_fusion, _initial_bounded_fusion_logit
from ..settings import load_settings


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "caltech101_four_layer_optical_joint_17um_10cm_robust.yaml"
)


def test_release_contract_is_explicitly_10cm_and_robust() -> None:
    settings = load_settings(CONFIG)
    assert settings.language_optical_distance_m == 0.10
    assert settings.phase_learning_rate == 0.006
    assert settings.optical_fusion_initial == 0.20
    assert settings.optical_fusion_minimum == 0.10
    assert settings.language_optical_k_space_enabled is True
    assert settings.language_optical_max_shift_pixels == 16
    assert settings.language_optical_phase_shift_pixels == 16
    assert settings.language_optical_ccd_shift_pixels == 16
    assert settings.hardware_amplitude_slm_pixel_pitch_um == 17.0
    assert settings.hardware_phase_slm_pixel_pitch_um == 8.0
    assert settings.hardware_phase_flip_vertical is True


def test_bounded_gate_starts_at_requested_fraction_and_never_crosses_floor() -> None:
    raw = _initial_bounded_fusion_logit(0.20, 0.10).requires_grad_(True)
    fusion = _bounded_fusion(raw, 0.10)
    torch.testing.assert_close(fusion, torch.tensor(0.20), atol=1.0e-7, rtol=0.0)
    fusion.backward()
    assert raw.grad is not None and raw.grad > 0
    values = _bounded_fusion(torch.tensor([-100.0, 0.0, 100.0]), 0.10)
    assert torch.all(values >= 0.10)
    assert torch.all(values <= 1.0)
