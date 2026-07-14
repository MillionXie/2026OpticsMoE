import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts import D2NNExpert, FiberArrayExpert, FourierExpert, HeterogeneousExpertBank
from layout import MoELayout
from model import DeepHeterogeneousOpticalMoEClassifier
from utils import load_yaml


def optics():
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


def backward_and_check_all(parameters, output):
    loss = output.abs().square().mean() + 0.01 * output.real.mean() + 0.02 * output.imag.mean()
    loss.backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_yaml_assignment_and_deep_structure_controls():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    expected = ["d2nn", "fourier", "fiber", "fiber", "d2nn", "fourier", "fourier", "fiber", "d2nn"]
    assert config["expert_bank"]["assignments"] == expected
    assert config["expert_bank"]["d2nn"]["num_layers"] == 5
    assert config["expert_bank"]["fourier"]["num_conv_blocks"] == 3
    assert config["expert_bank"]["fourier"]["num_tail_spatial_layers"] == 2
    assert config["expert_bank"]["fiber"]["num_pre_d2nn_layers"] == 2
    assert config["expert_bank"]["fiber"]["num_post_d2nn_layers"] == 2
    assert config["expert_bank"]["fiber"]["mode_bank_trainable"] is False


def test_d2nn_five_layer_complex_shape_finite_and_all_mask_gradients():
    expert = D2NNExpert(
        16,
        {"num_layers": 5, "propagation_padding": 3, "inter_layer_distance_m": 0.01},
        optics(),
        {"gain_enabled": True, "phase_bias_enabled": True},
    )
    field = torch.randn(2, 16, 16, dtype=torch.complex64, requires_grad=True)
    output, details = expert(field, return_details=True)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert len(details["spatial_layer_fields"]) == 5
    assert torch.isfinite(output.real).all() and torch.isfinite(output.imag).all()
    backward_and_check_all([layer.raw_phase for layer in expert.phase_layers], output)
    assert field.grad is not None


def test_fourier_three_blocks_two_spatial_layers_intermediates_and_all_gradients():
    expert = FourierExpert(
        16,
        {
            "num_conv_blocks": 3,
            "num_tail_spatial_layers": 2,
            "inter_block_distance_m": 0.01,
            "tail_spatial_distance_m": 0.01,
            "propagation_padding": 3,
            "phase_only": True,
            "phase_init": "zeros",
        },
        optics(),
        {"gain_enabled": True, "phase_bias_enabled": True},
    )
    field = torch.randn(2, 16, 16, dtype=torch.complex64, requires_grad=True)
    output, details = expert(field, return_details=True)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert torch.isfinite(output.real).all() and torch.isfinite(output.imag).all()
    assert len(details["fourier_block_fields"]) == 3
    assert len(details["tail_spatial_fields"]) == 2
    assert len(expert.inter_block_propagators) == 2
    assert len({block.raw_frequency_phase.data_ptr() for block in expert.convolution_blocks}) == 3
    parameters = [block.raw_frequency_phase for block in expert.convolution_blocks]
    parameters += [layer.raw_phase for layer in expert.tail_spatial_layers]
    backward_and_check_all(parameters, output)
    assert field.grad is not None


def test_fourier_blocks_contain_explicit_finite_aperture_and_cannot_be_mask_product_only():
    source = (ROOT / "experts.py").read_text(encoding="utf-8")
    block_source = source.split("class FourierConvolutionBlock", 1)[1].split("class FourierExpert", 1)[0]
    assert "finite_spectrum = torch.zeros_like(spectrum)" in block_source
    assert "spatial[:, start:stop, start:stop]" in block_source
    expert_source = source.split("class FourierExpert", 1)[1].split("class FiberArrayExpert", 1)[0]
    assert "inter_block_propagators" in expert_source
    assert "PaddedLocalPropagator" in expert_source


def test_fiber_encoder_bottleneck_decoder_metrics_and_all_trainable_gradients():
    expert = FiberArrayExpert(
        16,
        {
            "num_pre_d2nn_layers": 2,
            "num_post_d2nn_layers": 2,
            "inter_layer_distance_m": 0.01,
            "propagation_padding": 3,
            "fibers_per_axis": 4,
            "mode_sigma_px": 1.2,
            "mode_center_margin_px": 1.5,
            "trainable_mode_phase": True,
            "trainable_mode_amplitude": True,
            "mode_bank_trainable": False,
            "amplitude_min": 0.0,
            "amplitude_max": 1.0,
            "amplitude_init": 0.9,
        },
        optics(),
        {"gain_enabled": True, "phase_bias_enabled": True},
    )
    field = torch.randn(2, 16, 16, dtype=torch.complex64, requires_grad=True)
    output, details = expert(field, return_details=True)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert len(details["encoder_fields"]) == 2
    assert len(details["decoder_fields"]) == 2
    assert details["encoded_field"].shape == field.shape
    assert details["reconstructed_field"].shape == field.shape
    assert details["mode_power_distribution"].shape == (2, 16)
    assert torch.allclose(details["mode_power_distribution"].sum(-1), torch.ones(2), atol=1.0e-5)
    assert torch.isfinite(details["coupling_efficiency"]).all()
    assert torch.isfinite(details["effective_mode_number"]).all()
    assert not isinstance(expert.mode_bank, nn.Parameter)
    parameters = [layer.raw_phase for layer in list(expert.pre_layers) + list(expert.post_layers)]
    parameters += [expert.raw_mode_phase, expert.raw_mode_amplitude]
    backward_and_check_all(parameters, output)
    assert field.grad is not None


def test_parameter_counts_by_type_are_reported_separately():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    bank = HeterogeneousExpertBank(MoELayout(), config["expert_bank"], config["optics"])
    report = bank.parameter_report()["by_type"]
    assert report["d2nn"]["trainable_parameters"] == 3 * (5 * 120 * 120 + 2)
    assert report["fourier"]["trainable_parameters"] == 3 * ((3 + 2) * 120 * 120 + 2)
    assert report["fiber"]["trainable_parameters"] == 3 * (4 * 120 * 120 + 100 + 100 + 2)
    assert report["d2nn"]["trainable_parameters"] == report["fourier"]["trainable_parameters"]


def test_expert_bank_contains_no_activation_detection_or_normalization_modules():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    bank = HeterogeneousExpertBank(MoELayout(), config["expert_bank"], config["optics"])
    forbidden = (nn.ReLU, nn.GELU, nn.SiLU, nn.Softplus, nn.Sigmoid, nn.LayerNorm, nn.BatchNorm2d)
    assert not any(isinstance(module, forbidden) for module in bank.modules())
    source = (ROOT / "experts.py").read_text(encoding="utf-8").lower()
    assert "layer_norm" not in source
    assert "relu(" not in source


def test_bank_output_contract_and_fiber_metric_shapes():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    compact = deepcopy(config["expert_bank"])
    compact["d2nn"].update({"num_layers": 1, "propagation_padding": 0})
    compact["fourier"].update({"num_conv_blocks": 1, "num_tail_spatial_layers": 1, "propagation_padding": 0})
    compact["fiber"].update({"num_pre_d2nn_layers": 1, "num_post_d2nn_layers": 1, "fibers_per_axis": 3, "propagation_padding": 0})
    bank = HeterogeneousExpertBank(MoELayout(), compact, config["optics"])
    field = torch.randn(1, 480, 480, dtype=torch.complex64, requires_grad=True)
    output, details = bank(field, return_details=True)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert details["local_outputs"].shape == (1, 9, 120, 120)
    assert details["fiber_coupling_efficiency"].shape == (1, 9)
    assert details["fiber_effective_mode_number"].shape == (1, 9)
    assert details["fiber_mode_power_distribution"].shape == (1, 9, 9)
    assert torch.isfinite(output.real).all() and torch.isfinite(output.imag).all()
    output.abs().square().mean().backward()
    assert field.grad is not None


def test_full_model_shape_and_intermediate_contract():
    config = deepcopy(load_yaml(ROOT / "configs" / "config.yaml"))
    config["optics"]["distances_m"]["expert_to_global_fc"] = 0.0
    config["optics"]["distances_m"]["global_fc_to_detector"] = 0.0
    config["expert_bank"]["d2nn"].update({"num_layers": 1, "propagation_padding": 0})
    config["expert_bank"]["fourier"].update({"num_conv_blocks": 1, "num_tail_spatial_layers": 1, "propagation_padding": 0})
    config["expert_bank"]["fiber"].update({"num_pre_d2nn_layers": 1, "num_post_d2nn_layers": 1, "fibers_per_axis": 3, "propagation_padding": 0})
    model = DeepHeterogeneousOpticalMoEClassifier(config, 4)
    logits, items = model(torch.rand(1, 1, 32, 32), return_intermediates=True)
    assert logits.shape == (1, 4)
    assert items["expert_local_outputs"].shape == (1, 9, 120, 120)
    assert items["expert_bank_output"].dtype == torch.complex64
    assert items["detector_intensity"].min() >= 0

