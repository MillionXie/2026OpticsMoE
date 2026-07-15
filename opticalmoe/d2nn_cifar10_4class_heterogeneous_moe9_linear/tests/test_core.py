import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts import D2NNExpert, FiberArrayExpert, FourierExpert, HeterogeneousExpertBank
from layout import MoELayout
from model import HeterogeneousOpticalMoEClassifier
from train import detector_plane_mse_loss
from utils import load_yaml


def optics(size=16):
    return {
        "wavelength_m": 5.32e-7,
        "pixel_size_m": 16.0e-6,
        "phase_param": "sigmoid",
        "phase_init": "zeros",
        "init_std": 0.02,
        "evanescent_mode": "zero",
        "k_space_constraint_enabled": False,
        "theta_max_deg": 1.0,
    }


def test_detector_plane_mse_normalization_is_configurable():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    assert config["loss"]["normalize_detector_plane_mse"] is True
    intensity=torch.rand(2,8,8);target=torch.zeros_like(intensity);target[:,2:5,3:6]=1
    assert torch.allclose(
        detector_plane_mse_loss(intensity,target,100.0,True,1.0e-8),
        detector_plane_mse_loss(5*intensity,target,100.0,True,1.0e-8),
        rtol=1e-5,atol=1e-6,
    )
    assert not torch.allclose(
        detector_plane_mse_loss(intensity,target,100.0,False,1.0e-8),
        detector_plane_mse_loss(5*intensity,target,100.0,False,1.0e-8),
    )


def assert_complex_expert_backward(expert, size=16):
    field = torch.randn(2, size, size, dtype=torch.complex64, requires_grad=True)
    output = expert(field)
    assert output.shape == field.shape
    assert output.dtype == torch.complex64
    assert torch.isfinite(output.real).all()
    assert torch.isfinite(output.imag).all()
    output.abs().square().mean().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad.real).all()
    trainable = [parameter for parameter in expert.parameters() if parameter.requires_grad]
    assert trainable and any(parameter.grad is not None for parameter in trainable)


def test_d2nn_expert_shape_dtype_finite_and_gradient_with_padded_propagation():
    expert = D2NNExpert(
        16,
        {"num_layers": 3, "propagation_padding": 4, "inter_layer_distance_m": 0.01, "phase_init": "zeros"},
        optics(),
        {"gain_enabled": True, "phase_bias_enabled": True},
    )
    assert len(expert.phase_layers) == 3
    assert all(propagator.propagator.grid_size == (24, 24) for propagator in expert.propagators)
    assert_complex_expert_backward(expert)


def test_fourier_expert_is_orthonormal_complex_linear_transform_and_has_gradient():
    expert = FourierExpert(16, {"phase_init": "zeros"}, {"gain_enabled": False, "phase_bias_enabled": False})
    field = torch.randn(2, 16, 16, dtype=torch.complex64)
    output = expert(field)
    assert torch.allclose(output, field, atol=2.0e-5, rtol=2.0e-5)
    assert_complex_expert_backward(expert)


def test_fiber_expert_uses_coherent_complex_projection_and_bounded_amplitude():
    expert = FiberArrayExpert(
        16,
        {"mode_grid": [4, 4], "mode_sigma_pixels": 1.2, "mode_center_margin_pixels": 1.5, "amplitude_min": 0.0, "amplitude_max": 1.0, "amplitude_init": 0.9},
        {"gain_enabled": True, "phase_bias_enabled": True},
    )
    norms = expert.mode_bank.abs().square().sum((-2, -1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)
    assert torch.all((expert.mode_amplitude() > 0) & (expert.mode_amplitude() < 1))
    assert_complex_expert_backward(expert)


def test_configured_row_major_expert_map_and_parameter_counts():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    expected = ["d2nn", "fourier", "fiber", "fiber", "d2nn", "fourier", "fourier", "fiber", "d2nn"]
    assert config["experts"]["types"] == expected
    layout = MoELayout()
    bank = HeterogeneousExpertBank(layout, config["experts"], config["optics"])
    assert bank.expert_types == expected
    assert [expert.expert_type for expert in bank.experts] == expected
    report = bank.parameter_report()
    assert report["by_type"]["d2nn"]["count"] == 3
    assert report["by_type"]["fourier"]["count"] == 3
    assert report["by_type"]["fiber"]["count"] == 3
    assert report["by_type"]["d2nn"]["trainable_parameters"] == 3 * (5 * 120 * 120 + 2)
    assert report["by_type"]["fourier"]["trainable_parameters"] == 3 * (120 * 120 + 2)
    assert report["by_type"]["fiber"]["trainable_parameters"] == 3 * (100 + 100 + 2)


def test_expert_bank_has_no_activation_normalization_or_detector_modules():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    bank = HeterogeneousExpertBank(MoELayout(), config["experts"], config["optics"])
    forbidden = (nn.ReLU, nn.GELU, nn.SiLU, nn.Softplus, nn.Sigmoid, nn.LayerNorm, nn.BatchNorm2d)
    assert not any(isinstance(module, forbidden) for module in bank.modules())
    source = (ROOT / "experts.py").read_text(encoding="utf-8")
    assert "abs().square()" not in source.split("class HeterogeneousExpertBank", 1)[0]
    assert "layer_norm" not in source.lower()


def test_bank_reassembles_nine_complex_local_outputs_and_propagates_gradients():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    small = deepcopy(config["experts"])
    small["d2nn"].update({"num_layers": 2, "propagation_padding": 0, "inter_layer_distance_m": 0.0})
    small["fiber"].update({"mode_grid": [3, 3], "mode_sigma_pixels": 4.0})
    layout = MoELayout()
    bank = HeterogeneousExpertBank(layout, small, config["optics"])
    field = torch.randn(1, 480, 480, dtype=torch.complex64, requires_grad=True)
    output, details = bank(field, return_details=True)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert details["local_outputs"].shape == (1, 9, 120, 120)
    assert details["input_power"].shape == (1, 9)
    assert details["output_power"].shape == (1, 9)
    assert torch.isfinite(output.real).all() and torch.isfinite(output.imag).all()
    output.abs().square().mean().backward()
    assert field.grad is not None


def test_full_model_output_and_intermediate_contract_without_nonlinear_expert_path():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    config = deepcopy(config)
    config["optics"]["distances_m"]["expert_to_global_fc"] = 0.0
    config["optics"]["distances_m"]["global_fc_to_detector"] = 0.0
    config["experts"]["d2nn"].update({"num_layers": 1, "propagation_padding": 0, "inter_layer_distance_m": 0.0})
    model = HeterogeneousOpticalMoEClassifier(config, 4)
    logits, items = model(torch.rand(1, 1, 32, 32), return_intermediates=True)
    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()
    assert items["expert_local_outputs"].shape == (1, 9, 120, 120)
    assert items["expert_bank_output"].dtype == torch.complex64
    assert items["detector_intensity"].min() >= 0

