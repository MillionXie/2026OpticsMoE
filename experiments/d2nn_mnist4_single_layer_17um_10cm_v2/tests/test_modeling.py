from types import SimpleNamespace

import torch

from ..modeling import RobustRawCCDMNIST4D2NN, translate_zero_fill


def _tiny_settings() -> SimpleNamespace:
    return SimpleNamespace(
        classes=(0, 1, 2, 3),
        active_size=16,
        input_size=12,
        input_guard=2,
        canvas_size=20,
        canvas_guard=2,
        propagation_grid_size=32,
        propagation_guard=6,
        wavelength_nm=532.0,
        logical_pixel_pitch_um=17.0,
        detector_distance_m=0.10,
        k_space_enabled=True,
        k_space_theta_max_deg=0.65,
        robustness_enabled=True,
        robustness_probability=1.0,
        robustness_warmup_epochs=0,
        input_shift_max_px=2,
        phase_shift_max_px=2,
        pre_ccd_shift_max_px=2,
        loss_mode="notebook_full_plane_mse",
        notebook_full_plane_mse_scale=100.0,
        target_region_mse_weight=1.0,
        background_mse_weight=0.5,
        detector_bounds=lambda: (
            (2, 2, 6, 6),
            (10, 2, 14, 6),
            (2, 10, 6, 14),
            (10, 10, 14, 14),
        ),
    )


def test_zero_fill_translation_has_no_circular_wrap() -> None:
    source = torch.zeros(1, 4, 4)
    source[0, 0, 0] = 1.0
    shifted = translate_zero_fill(source, dy=-1, dx=0)
    assert shifted.sum() == 0
    source.zero_()
    source[0, 1, 1] = 2.0
    shifted = translate_zero_fill(source, dy=1, dx=-1)
    assert shifted[0, 2, 0] == 2.0
    assert shifted.sum() == 2.0


def test_raw_zero_means_pi_and_kspace_is_active() -> None:
    model = RobustRawCCDMNIST4D2NN(_tiny_settings())
    torch.testing.assert_close(
        model.phase(), torch.full_like(model.raw_phase, torch.pi)
    )
    assert 0.0 < model.propagator.pass_fraction < 1.0
    assert not bool(model.propagator.k_space_pass_mask.all())


def test_raw_ccd_has_no_post_detection_normalization_or_nonlinear_readout() -> None:
    model = RobustRawCCDMNIST4D2NN(_tiny_settings()).eval()
    image = torch.rand(2, 1, 12, 12) * 0.4
    zero = {"input": (0, 0), "phase": (0, 0), "pre_ccd": (0, 0)}
    output = model(image, forced_shifts=zero)
    scaled = model(2.0 * image, forced_shifts=zero)
    assert set(output) == {
        "ccd_intensity",
        "detector_energy",
        "active_amplitude",
        "applied_shifts",
    }
    assert output["ccd_intensity"].shape == (2, 16, 16)
    torch.testing.assert_close(
        scaled["ccd_intensity"], 4.0 * output["ccd_intensity"], rtol=2e-5, atol=1e-6
    )
    torch.testing.assert_close(
        scaled["detector_energy"],
        4.0 * output["detector_energy"],
        rtol=2e-5,
        atol=1e-6,
    )


def test_notebook_full_plane_loss_and_diagnostics_match_raw_pixel_definition() -> None:
    settings = _tiny_settings()
    model = RobustRawCCDMNIST4D2NN(settings)
    intensity = torch.zeros(1, 16, 16)
    intensity[:, 2:6, 2:6] = 0.25
    output = {"ccd_intensity": intensity}
    total, target, background = model.raw_ccd_loss(output, torch.tensor([0]))
    torch.testing.assert_close(target, torch.tensor((0.25 - 1.0) ** 2))
    torch.testing.assert_close(background, torch.tensor(0.0))
    expected_full_plane = 100.0 * 16.0 * (0.25 - 1.0) ** 2 / (16.0 * 16.0)
    torch.testing.assert_close(total, torch.tensor(expected_full_plane))


def test_robustness_can_be_disabled_during_training_warmup() -> None:
    model = RobustRawCCDMNIST4D2NN(_tiny_settings()).train()
    model.set_robustness_training_active(False)
    output = model(torch.rand(1, 1, 12, 12))
    assert output["applied_shifts"] == {
        "input": (0, 0),
        "phase": (0, 0),
        "pre_ccd": (0, 0),
    }


def test_forced_pre_ccd_shifts_are_cardinal_and_phase_gradient_survives() -> None:
    model = RobustRawCCDMNIST4D2NN(_tiny_settings()).train()
    forced = {"input": (0, 2), "phase": (-1, 0), "pre_ccd": (1, 0)}
    output = model(torch.rand(1, 1, 12, 12), forced_shifts=forced)
    assert output["applied_shifts"] == forced
    loss, _, _ = model.raw_ccd_loss(output, torch.tensor([2]))
    loss.backward()
    assert model.raw_phase.grad is not None
    assert torch.isfinite(model.raw_phase.grad).all()
    assert bool(torch.any(model.raw_phase.grad != 0))
