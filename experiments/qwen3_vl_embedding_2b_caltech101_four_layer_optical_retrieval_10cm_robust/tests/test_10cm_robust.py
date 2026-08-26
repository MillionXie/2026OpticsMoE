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
FAST_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "caltech101_four_layer_optical_joint_17um_10cm_robust_fast_2h.yaml"
)
RECOVERY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "caltech101_four_layer_optical_joint_17um_10cm_robust_fast_remaining20.yaml"
)
QUICK_HARDWARE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "caltech101_four_layer_optical_quick_last_stage_10x10.yaml"
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
    assert settings.batch_size == 30
    assert settings.pk_skus_per_batch == 10
    assert settings.pk_images_per_sku == 3
    assert settings.router_noise_std == 0.10
    assert settings.lambda_router_balance == 0.05
    assert settings.hardware_amplitude_slm_pixel_pitch_um == 17.0
    assert settings.hardware_phase_slm_pixel_pitch_um == 8.0
    assert settings.hardware_phase_flip_vertical is True
    assert FourLayerOpticalReplacement.checkpoint_architecture.endswith("_v2")


def test_fast_release_only_changes_time_budget() -> None:
    full = load_settings(CONFIG)
    fast = load_settings(FAST_CONFIG)
    assert fast.epochs == 25
    assert fast.optimizer_steps_per_epoch == 15
    assert fast.batch_size == 30
    assert fast.pk_skus_per_batch == 10
    assert fast.pk_images_per_sku == 3
    assert fast.language_optical_distance_m == full.language_optical_distance_m
    assert fast.phase_learning_rate == full.phase_learning_rate
    assert fast.optical_fusion_minimum == full.optical_fusion_minimum
    assert fast.output_dir == full.output_dir


def test_router_recovery_continuation_preserves_absolute_end_epoch() -> None:
    recovery = load_settings(RECOVERY_CONFIG)
    assert recovery.epochs == 20
    assert recovery.optimizer_steps_per_epoch == 15
    assert recovery.phase_focus_warmup_epochs == 0
    assert recovery.phase_focus_interval_epochs == 2
    assert recovery.router_noise_std == 0.10
    assert recovery.lambda_router_balance == 0.05


def test_quick_last_stage_uses_isolated_small_dataset_metadata() -> None:
    full = load_settings(CONFIG)
    quick = load_settings(QUICK_HARDWARE_CONFIG)
    assert quick.gallery_images_per_sku == 1
    assert quick.train_limit_per_sku == 10
    assert quick.test_limit_per_sku == 10
    assert quick.reserve_test_before_train is True
    assert quick.output_dir != full.output_dir
    assert quick.language_optical_distance_m == full.language_optical_distance_m
    assert quick.phase_learning_rate == full.phase_learning_rate
    assert quick.optical_fusion_minimum == full.optical_fusion_minimum


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
