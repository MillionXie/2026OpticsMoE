from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .. import optical_blocks
from ..modeling import FourLayerOpticalReplacement
from ..optical_blocks import (
    MoE4LanguageTwoBlockOpticalPath,
    _bounded_fusion,
    _initial_bounded_fusion_logit,
    _shift_full_detector_then_crop,
    _translate_with_fill,
)
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
    assert FourLayerOpticalReplacement.checkpoint_architecture.endswith("_v2")


def test_bounded_gate_starts_at_requested_fraction_and_never_crosses_floor() -> None:
    raw = _initial_bounded_fusion_logit(0.20, 0.10).requires_grad_(True)
    fusion = _bounded_fusion(raw, 0.10)
    torch.testing.assert_close(fusion, torch.tensor(0.20), atol=1.0e-7, rtol=0.0)
    fusion.backward()
    assert raw.grad is not None and raw.grad > 0
    values = _bounded_fusion(torch.tensor([-100.0, 0.0, 100.0]), 0.10)
    assert torch.all(values >= 0.10)
    assert torch.all(values <= 1.0)


def test_input_phase_and_ccd_offsets_are_sampled_independently() -> None:
    path = SimpleNamespace(
        input_shift_pixels=16,
        phase_shift_pixels=16,
        ccd_shift_pixels=16,
        training=True,
        last_sampled_shifts={},
    )
    draws = [(1, 2), (-3, 4), (5, -6)]
    with patch.object(optical_blocks, "_sample_integer_shift", side_effect=draws) as sample:
        shifts = MoE4LanguageTwoBlockOpticalPath._draw_stage_shifts(
            path, "expert"
        )
    assert shifts == {
        "input": (1, 2),
        "phase": (-3, 4),
        "ccd": (5, -6),
    }
    assert path.last_sampled_shifts["expert"] == shifts
    assert sample.call_count == 3


def test_phase_translation_moves_modulation_and_identity_fills_edges() -> None:
    modulation = torch.ones(1, 5, 5, dtype=torch.complex64)
    modulation[:, 2, 2] = -1.0 + 0.0j
    shifted = _translate_with_fill(
        modulation,
        1,
        -1,
        fill_value=1.0 + 0.0j,
    )
    assert shifted[0, 3, 1] == -1.0 + 0.0j
    assert torch.all(shifted[:, 0, :] == 1.0 + 0.0j)
    assert torch.all(shifted[:, :, -1] == 1.0 + 0.0j)


def test_phase_translation_preserves_finite_nonzero_gradient() -> None:
    raw_phase = torch.linspace(-0.4, 0.6, 25).reshape(1, 5, 5)
    raw_phase.requires_grad_(True)
    modulation = torch.exp(1j * raw_phase)
    shifted = _translate_with_fill(
        modulation,
        1,
        -1,
        fill_value=1.0 + 0.0j,
    )
    weights = torch.linspace(0.5, 1.5, 25).reshape(1, 5, 5)
    loss = (shifted.real * weights).sum()
    loss.backward()
    assert raw_phase.grad is not None
    assert torch.isfinite(raw_phase.grad).all()
    assert torch.count_nonzero(raw_phase.grad) > 0


def test_expert_modulation_supports_dropout_disabled_three_dimensional_stack() -> None:
    apertures = [
        SimpleNamespace(y0=0, y1=1, x0=0, x1=1),
        SimpleNamespace(y0=2, y1=3, x0=2, x1=3),
    ]
    stacked = torch.tensor(
        [[[2.0 + 0.0j]], [[3.0 + 0.0j]]], dtype=torch.complex64
    )
    layer = SimpleNamespace(_stacked_modulation=lambda _batch: stacked)
    path = SimpleNamespace(
        core=SimpleNamespace(
            expert_layers=[layer],
            geometry=SimpleNamespace(expert_apertures=apertures),
        )
    )
    field = torch.zeros(4, 3, 3, dtype=torch.complex64)
    modulation = MoE4LanguageTwoBlockOpticalPath._expert_phase_modulation(
        path, field
    )
    assert tuple(modulation.shape) == (4, 3, 3)
    assert torch.all(modulation[:, 0, 0] == 2.0 + 0.0j)
    assert torch.all(modulation[:, 2, 2] == 3.0 + 0.0j)
    # Exported expert mosaics use phase=0 outside learned apertures, i.e. unit
    # complex modulation. Shifting the phase map must preserve that semantics.
    assert torch.all(modulation[:, 1, :] == 1.0 + 0.0j)


def test_global_modulation_uses_the_same_identity_background_contract() -> None:
    def global_phase(identity: torch.Tensor) -> torch.Tensor:
        modulation = identity.clone()
        modulation[:, 1:3, 1:3] = -1.0 + 0.0j
        return modulation

    path = SimpleNamespace(
        core=SimpleNamespace(global_phase=global_phase)
    )
    field = torch.zeros(2, 4, 4, dtype=torch.complex64)
    modulation = MoE4LanguageTwoBlockOpticalPath._global_phase_modulation(
        path, field
    )
    assert torch.all(modulation[:, 1:3, 1:3] == -1.0 + 0.0j)
    assert torch.all(modulation[:, 0, :] == 1.0 + 0.0j)
    shifted = _translate_with_fill(
        modulation, 1, 0, fill_value=1.0 + 0.0j
    )
    assert torch.all(shifted[:, 0, :] == 1.0 + 0.0j)


def test_ccd_translation_occurs_on_full_canvas_before_active_crop() -> None:
    full = torch.zeros(1, 5, 5)
    # This signal starts outside the active rows [1,4), then enters the ROI
    # after a +1 detector translation. Crop-first implementations lose it.
    full[0, 0, 2] = 7.0
    cropped = _shift_full_detector_then_crop(
        full,
        shift_y=1,
        shift_x=0,
        y0=1,
        y1=4,
        x0=1,
        x1=4,
    )
    assert tuple(cropped.shape) == (1, 3, 3)
    assert cropped[0, 0, 1] == 7.0
