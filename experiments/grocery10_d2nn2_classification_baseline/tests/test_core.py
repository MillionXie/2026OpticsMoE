from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.grocery10_d2nn2_classification_baseline.modeling import (
    TwoPlaneD2NNClassifier,
    _encode_rgb_tensor,
)
from experiments.grocery10_d2nn2_classification_baseline.settings import load_settings
from experiments.grocery10_d2nn2_classification_baseline.training import (
    detector_plane_mse_loss,
    detector_region_cross_entropy,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_settings() -> SimpleNamespace:
    return SimpleNamespace(
        selected_skus=tuple(f"sku_{index}" for index in range(10)),
        input_encoding="grayscale_amplitude",
        canvas_size=64,
        active_size=56,
        image_size=16,
        first_phase_size=16,
        wavelength_nm=532.0,
        pixel_pitch_um=8.0,
        input_to_first_phase_distance_m=0.0,
        first_to_second_phase_distance_m=0.005,
        second_phase_to_detector_distance_m=0.005,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        phase_init_std=0.02,
        k_space_constraint_enabled=False,
        theta_max_deg=1.0,
        detector_row_layout=(3, 4, 3),
        detector_size=4,
        detector_horizontal_gap=4,
        detector_vertical_gap=4,
        detector_normalize_total_energy=True,
        detector_eps=1e-8,
        loss_type="detector_region_cross_entropy",
        detector_plane_mse_scale=100.0,
        normalize_detector_plane_mse=True,
        detector_plane_mse_normalization_eps=1e-8,
        loss_eps=1e-8,
    )


def test_formal_config_matches_experimental_optical_path() -> None:
    settings = load_settings(ROOT / "configs" / "grocery10_d2nn2.yaml")
    assert settings.canvas_size == 1026
    assert settings.active_size == 986
    assert settings.first_phase_size == settings.image_size == 224
    assert settings.pixel_pitch_um == 8.0
    assert settings.wavelength_nm == 532.0
    assert settings.input_to_first_phase_distance_m == 0.0
    assert settings.first_to_second_phase_distance_m == 0.10
    assert settings.second_phase_to_detector_distance_m == 0.10
    assert settings.detector_row_layout == (3, 4, 3)
    assert len(settings.selected_skus) == 10


def test_two_phase_forward_shape_and_final_intensity_nonnegative() -> None:
    model = TwoPlaneD2NNClassifier(_tiny_settings())
    logits, details = model(
        torch.rand(2, 1, 16, 16), return_intermediates=True
    )
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    assert details["detector_region_energies"].shape == (2, 10)
    assert torch.all(details["detector_region_energies"] >= 0)
    assert torch.all(details["detector_intensity"] >= 0)
    assert details["detector_intensity"].shape == (2, 56, 56)


def test_only_two_phase_masks_are_trainable_and_both_receive_gradients() -> None:
    model = TwoPlaneD2NNClassifier(_tiny_settings())
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(trainable) == {
        "first_phase.raw_phase",
        "second_global_phase.raw_phase",
    }
    assert sum(parameter.numel() for parameter in trainable.values()) == 16**2 + 56**2
    logits = model(torch.rand(3, 1, 16, 16))
    loss = detector_region_cross_entropy(logits, torch.tensor([0, 4, 9]))
    loss.backward()
    for parameter in trainable.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_model_contains_no_electronic_readout_or_intermediate_nonlinearity() -> None:
    model = TwoPlaneD2NNClassifier(_tiny_settings())
    forbidden = (nn.Linear, nn.LayerNorm, nn.ReLU, nn.GELU, nn.Softplus)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    names = " ".join(name.lower() for name, _ in model.named_modules())
    assert "router" not in names
    assert "moe" not in names
    report = model.parameter_report()
    assert report["phase_planes"] == 2
    assert report["intermediate_oeo_conversions"] == 0
    assert report["electronic_trainable_parameters"] == 0
    assert report["similarity_embedding_dim"] is None


def test_detector_regions_are_equal_area_centered_and_nonoverlapping() -> None:
    model = TwoPlaneD2NNClassifier(_tiny_settings())
    assert [sum(model.settings.detector_row_layout[:index]) for index in range(3)] == [
        0,
        3,
        7,
    ]
    assert len(model.detector.regions) == 10
    assert model.detector.masks.sum(0).max().item() == 1.0
    assert torch.all(model.detector.masks.sum((-2, -1)) == 16)
    for region in model.detector.regions:
        assert model.active_start <= region.x0 < region.x1 <= model.active_end
        assert model.active_start <= region.y0 < region.y1 <= model.active_end


def test_detector_region_cross_entropy_is_not_similarity_learning() -> None:
    logits = torch.rand(4, 10, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss = detector_region_cross_entropy(logits, labels)
    loss.backward()
    assert loss.ndim == 0
    assert logits.grad is not None


def test_detector_plane_mse_is_energy_scale_invariant_and_has_gradients() -> None:
    intensity = torch.rand(2, 56, 56, requires_grad=True)
    target = torch.zeros_like(intensity)
    target[0, 4:8, 7:11] = 1.0
    target[1, 30:34, 35:39] = 1.0
    loss = detector_plane_mse_loss(
        intensity,
        target,
        scale=100.0,
        normalize=True,
        eps=1e-8,
    )
    scaled_loss = detector_plane_mse_loss(
        intensity * 9.0,
        target,
        scale=100.0,
        normalize=True,
        eps=1e-8,
    )
    assert torch.allclose(loss, scaled_loss, rtol=1e-5, atol=1e-6)
    loss.backward()
    assert intensity.grad is not None
    assert torch.isfinite(intensity.grad).all()


def test_detector_intensity_only_path_avoids_intermediate_field_capture() -> None:
    model = TwoPlaneD2NNClassifier(_tiny_settings())
    energies, detector_intensity = model(
        torch.rand(2, 1, 16, 16), return_detector_intensity=True
    )
    assert energies.shape == (2, 10)
    assert detector_intensity.shape == (2, 56, 56)
    assert detector_intensity.requires_grad
    assert torch.all(detector_intensity >= 0)


def test_rgb_quadrant_encoding_is_fixed_nonnegative_and_preserves_channels() -> None:
    rgb = torch.zeros(3, 16, 16)
    rgb[0] = 0.2
    rgb[1] = 0.5
    rgb[2] = 0.8
    amplitude = _encode_rgb_tensor(rgb, "rgb_quadrant_amplitude")
    assert amplitude.shape == (1, 16, 16)
    assert torch.all(amplitude >= 0)
    assert torch.allclose(amplitude[0, :8, :8], torch.full((8, 8), 0.2))
    assert torch.allclose(amplitude[0, :8, 8:], torch.full((8, 8), 0.5))
    assert torch.allclose(amplitude[0, 8:, :8], torch.full((8, 8), 0.8))
