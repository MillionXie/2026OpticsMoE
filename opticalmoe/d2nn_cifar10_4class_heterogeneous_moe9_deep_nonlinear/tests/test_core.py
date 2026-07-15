import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts import D2NNExpert, FiberArrayExpert, FourierExpert, StageGlobalOEO
from model import DeepHeterogeneousOpticalMoENonlinearClassifier
from train import detector_plane_mse_loss
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


def nonlinear(per_expert_enabled=True, elementwise_affine=False, affine_sharing="per_expert"):
    return {
        "enabled": True,
        "type": "intensity_layernorm_relu",
        "normalization": {
            "type": "layernorm",
            "per_expert_enabled": per_expert_enabled,
            "aperture": "nonlinear_enabled_expert_regions",
            "eps": 1.0e-6,
            "elementwise_affine": elementwise_affine,
            "affine_sharing": affine_sharing,
        },
        "activation": {"type": "relu"},
        "reencoding": {
            "amplitude_source": "relu_output",
            "zero_phase": True,
            "reapply_routing_amplitude": False,
        },
    }


def fiber_cfg():
    return {
        "num_pre_d2nn_layers": 2,
        "num_post_d2nn_layers": 2,
        "nonlinear_schedule": [True, False, True, True, True],
        "inter_layer_distance_m": 0.01,
        "propagation_padding": 2,
        "fibers_per_axis": 3,
        "mode_sigma_px": 1.2,
        "mode_center_margin_px": 1.5,
        "trainable_mode_phase": True,
        "trainable_mode_amplitude": True,
        "mode_bank_trainable": False,
        "amplitude_min": 0.0,
        "amplitude_max": 1.0,
        "amplitude_init": 0.9,
    }


def test_config_maps_required_schedules_and_simple_nonlinearity():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    assert config["expert_bank"]["assignments"] == [
        "d2nn", "fourier", "fiber", "fiber", "d2nn", "fourier", "fourier", "fiber", "d2nn"
    ]
    assert config["expert_bank"]["d2nn"]["nonlinear_schedule"] == [True] * 5
    assert config["expert_bank"]["fourier"]["nonlinear_schedule"] == [True] * 5
    assert config["expert_bank"]["fiber"]["nonlinear_schedule"] == [True, False, True, True, True]
    assert config["nonlinearity"]["type"] == "intensity_layernorm_relu"
    assert config["nonlinearity"]["normalization"]["per_expert_enabled"] is True
    assert config["nonlinearity"]["normalization"]["elementwise_affine"] is True
    assert config["nonlinearity"]["normalization"]["affine_sharing"] == "per_expert"
    assert config["nonlinearity"]["reencoding"]["reapply_routing_amplitude"] is False
    assert config["nonlinearity"]["activation"] == {"type": "relu"}
    assert config["loss"]["detector_ce_weight"] == 0.0
    assert config["loss"]["router_balance_weight"] == 0.2
    assert config["loss"]["router_importance_weight"] == 0.0


def test_three_expert_types_have_five_complex_linear_stages():
    d2nn = D2NNExpert(12, {"num_layers": 5, "nonlinear_schedule": [True] * 5, "propagation_padding": 2, "inter_layer_distance_m": 0.01}, optics())
    fourier = FourierExpert(12, {"num_conv_blocks": 3, "num_tail_spatial_layers": 2, "nonlinear_schedule": [True] * 5, "propagation_padding": 2, "inter_block_distance_m": 0.01, "phase_only": True}, optics())
    fiber = FiberArrayExpert(12, fiber_cfg(), optics())
    for expert in (d2nn, fourier, fiber):
        field = torch.randn(1, 12, 12, dtype=torch.complex64)
        assert expert.num_stages == 5
        for stage in range(5):
            field = expert.forward_stage(stage, field)
            assert field.shape == (1, 12, 12)
            assert field.dtype == torch.complex64
            assert torch.isfinite(field.real).all() and torch.isfinite(field.imag).all()


def test_fiber_stage2_bypasses_oeo_exactly_and_preserves_complex_phase():
    expert = FiberArrayExpert(12, fiber_cfg(), optics())
    field = torch.randn(2, 12, 12, dtype=torch.complex64, requires_grad=True)
    linear_stage2 = expert.forward_stage(1, field)
    outputs, details = StageGlobalOEO(nonlinear(), 1)([linear_stage2], [False], capture_fields=True)
    assert outputs[0] is linear_stage2 and torch.equal(outputs[0], linear_stage2)
    assert details["pre_intensity"] is None
    assert expert.nonlinear_schedule == [True, False, True, True, True]
    assert torch.angle(outputs[0]).abs().sum() > 0
    projected = expert.forward_stage(2, outputs[0])
    (projected.abs().square().mean() + projected.real.mean()).backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert expert.pre_layers[1].raw_phase.grad is not None


def test_fiber_has_no_extra_coupling_phase_parameter():
    expert = FiberArrayExpert(12, fiber_cfg(), optics())
    names = [name for name, _ in expert.named_parameters()]
    assert not any("coupling_phase" in name for name in names)
    assert sum(name.endswith("raw_phase") for name in names) == 4
    assert "raw_mode_phase" in names
    assert expert.parameter_summary()["extra_fiber_coupling_phase_parameters"] == 0


def test_per_expert_layernorm_statistics_are_independent_and_bypass_is_preserved():
    module = StageGlobalOEO(nonlinear(), 0)
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert not hasattr(module, "gain") and not hasattr(module, "threshold")
    low = torch.ones(1, 4, 4, dtype=torch.complex64)
    high = 2.0 * torch.ones_like(low)
    bypass = (1.0 + 2.0j) * torch.ones_like(low)
    outputs, details = module([low, high, bypass], [True, True, False], capture_fields=True)
    assert details["normalization_scope"] == "per_expert"
    assert details["routing_amplitude_reapplied"] is False
    assert torch.allclose(details["normalization_mean"], torch.tensor([[1.0, 4.0]]), atol=1e-6)
    assert torch.allclose(details["normalization_std"], torch.full((1, 2), 1.0e-3), atol=1e-6)
    assert torch.count_nonzero(details["normalized_intensity"]) == 0
    assert torch.count_nonzero(outputs[0]) == 0
    assert torch.count_nonzero(outputs[1]) == 0
    assert outputs[2] is bypass and torch.equal(outputs[2], bypass)
    assert torch.all(outputs[0].imag == 0) and torch.all(outputs[1].imag == 0)


def test_stage_global_statistics_remain_available_as_ablation():
    module = StageGlobalOEO(nonlinear(per_expert_enabled=False), 0)
    low = torch.ones(1, 4, 4, dtype=torch.complex64)
    high = 2.0 * torch.ones_like(low)
    outputs, details = module([low, high], [True, True], capture_fields=True)
    assert details["normalization_scope"] == "stage_global"
    assert torch.allclose(details["normalization_mean"], torch.tensor([[2.5]]), atol=1e-6)
    assert torch.count_nonzero(outputs[0]) == 0
    assert torch.allclose(outputs[1].real, torch.ones_like(outputs[1].real), atol=1e-5)


def test_affine_layernorm_is_independent_per_expert_trainable_and_identity_initialized():
    module = StageGlobalOEO(nonlinear(elementwise_affine=True), 0, field_size=4, num_experts=3)
    assert sum(parameter.numel() for parameter in module.parameters()) == 2 * 3 * 4 * 4
    assert module.affine_weight.shape == (3, 4, 4)
    assert module.affine_bias.shape == (3, 4, 4)
    assert torch.all(module.affine_weight == 1)
    assert torch.all(module.affine_bias == 0)
    base = torch.randn(2, 4, 4, dtype=torch.complex64)
    fields = [base.clone() for _ in range(3)]
    with torch.no_grad():
        module.affine_bias[0].fill_(0.0)
        module.affine_bias[1].fill_(1.0)
        module.affine_bias[2].fill_(2.0)
    outputs, details = module(fields, [True, True, True])
    assert outputs[0].real.mean() < outputs[1].real.mean() < outputs[2].real.mean()
    sum(output.real.mean() for output in outputs).backward()
    assert module.affine_weight.grad is not None
    assert module.affine_bias.grad is not None
    assert details["elementwise_affine"] is True
    assert details["affine_sharing"] == "per_expert"


def test_shared_stage_affine_remains_available_as_lower_parameter_ablation():
    module = StageGlobalOEO(
        nonlinear(elementwise_affine=True, affine_sharing="per_stage"), 0, field_size=4
    )
    assert module.affine_weight.shape == (1, 4, 4)
    assert sum(parameter.numel() for parameter in module.parameters()) == 2 * 4 * 4


def test_invalid_layernorm_configs_fail_clearly():
    invalid = nonlinear()
    invalid["normalization"]["eps"] = 0.0
    with pytest.raises(ValueError, match="eps"):
        StageGlobalOEO(invalid, 0)
    invalid = nonlinear(elementwise_affine=True)
    with pytest.raises(ValueError, match="field_size"):
        StageGlobalOEO(invalid, 0)
    invalid = nonlinear()
    invalid["normalization"]["affine_sharing"] = "global"
    with pytest.raises(ValueError, match="affine_sharing"):
        StageGlobalOEO(invalid, 0)
    invalid = nonlinear(elementwise_affine=True)
    with pytest.raises(ValueError, match="num_experts"):
        StageGlobalOEO(invalid, 0, field_size=4)
    invalid = nonlinear()
    invalid["reencoding"]["reapply_routing_amplitude"] = True
    with pytest.raises(ValueError, match="routing amplitude"):
        StageGlobalOEO(invalid, 0)
    invalid = nonlinear()
    invalid["activation"]["type"] = "gelu"
    with pytest.raises(ValueError, match="activation.type"):
        StageGlobalOEO(invalid, 0)


def test_all_d2nn_fourier_and_fiber_stage_parameters_receive_gradients():
    experts = [
        D2NNExpert(12, {"num_layers": 5, "nonlinear_schedule": [True] * 5, "propagation_padding": 2, "inter_layer_distance_m": 0.01}, optics()),
        FourierExpert(12, {"num_conv_blocks": 3, "num_tail_spatial_layers": 2, "nonlinear_schedule": [True] * 5, "propagation_padding": 2, "inter_block_distance_m": 0.01, "phase_only": True}, optics()),
        FiberArrayExpert(12, fiber_cfg(), optics()),
    ]
    for expert in experts:
        field = torch.randn(2, 12, 12, dtype=torch.complex64)
        modules = [StageGlobalOEO(nonlinear(), stage) for stage in range(5)]
        for stage in range(5):
            field = expert.forward_stage(stage, field)
            field = modules[stage]([field], [expert.nonlinear_enabled(stage)])[0][0]
        expert.apply_output_scalar(field).abs().square().mean().backward()
        for parameter in expert.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()


def test_detector_plane_mse_normalization_is_optional_and_scale_invariant():
    torch.manual_seed(3)
    intensity = torch.rand(2, 8, 8)
    target = torch.zeros_like(intensity)
    target[:, 2:5, 3:6] = 1.0
    normalized = detector_plane_mse_loss(intensity, target, 100.0, True, 1.0e-8)
    normalized_scaled = detector_plane_mse_loss(7.0 * intensity, target, 100.0, True, 1.0e-8)
    raw = detector_plane_mse_loss(intensity, target, 100.0, False, 1.0e-8)
    raw_scaled = detector_plane_mse_loss(7.0 * intensity, target, 100.0, False, 1.0e-8)
    assert torch.allclose(normalized, normalized_scaled, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(raw, raw_scaled)


def test_ten_class_config_uses_centered_3_4_3_detector_with_4class_box_size():
    config = load_yaml(ROOT / "configs" / "config_cifar10_10class.yaml")
    model = DeepHeterogeneousOpticalMoENonlinearClassifier(config, 10)
    assert config["dataset"]["class_indices"] == list(range(10))
    assert config["detector"]["detector_size"] == 50
    assert config["detector"]["N_det_sets"] == [3, 4, 3]
    assert config["dataset"]["train_samples_per_class_per_epoch"] == 1000
    assert config["loss"]["router_balance_weight"] == 0.2
    assert config["loss"]["router_importance_weight"] == 0.0
    assert config["nonlinearity"]["normalization"]["per_expert_enabled"] is True
    assert config["nonlinearity"]["normalization"]["elementwise_affine"] is True
    assert config["nonlinearity"]["normalization"]["affine_sharing"] == "per_expert"
    bounds = []
    for mask in model.detector.masks:
        points = mask.nonzero()
        bounds.append((int(points[:, 0].min()), int(points[:, 0].max() + 1), int(points[:, 1].min()), int(points[:, 1].max() + 1)))
    assert bounds[:3] == [(115,165,115,165),(115,165,215,265),(115,165,315,365)]
    assert bounds[3:7] == [(215,265,65,115),(215,265,165,215),(215,265,265,315),(215,265,365,415)]
    assert bounds[7:] == [(315,365,115,165),(315,365,215,265),(315,365,315,365)]


def test_fourier_stages_keep_explicit_finite_aperture_and_padding():
    source = (ROOT / "experts.py").read_text(encoding="utf-8")
    block = source.split("class FourierConvolutionBlock", 1)[1].split("class FourierExpert", 1)[0]
    assert "finite_spectrum = torch.zeros_like(spectrum)" in block
    assert "spatial[:, start:stop, start:stop]" in block
    assert 'norm="ortho"' in block
    assert "PaddedLocalPropagator" in source


def test_model_source_has_no_post_global_oeo():
    source = (ROOT / "model.py").read_text(encoding="utf-8")
    forward = source.split("def forward(self, images", 1)[1]
    assert "at_global_fc = self.expert_to_global_fc(bank_output)" in forward
    assert "after_global_fc = self.global_fc(at_global_fc)" in forward
    assert "detector_field = self.to_detector(after_global_fc)" in forward
    segment = forward.split("after_global_fc = self.global_fc(at_global_fc)", 1)[1].split("detector_intensity", 1)[0]
    assert "stage_nonlinear" not in segment.lower()
    assert '"post_global_oeo_applied": False' in forward


def test_oeo_affine_parameter_groups_are_reported():
    config = deepcopy(load_yaml(ROOT / "configs" / "config.yaml"))
    model = DeepHeterogeneousOpticalMoENonlinearClassifier(config, 4)
    report = model.nonlinearity_parameter_report()
    assert report["normalization_scope"] == "per_expert"
    assert report["elementwise_affine"] is True
    assert report["routing_amplitude_reapplied"] is False
    assert report["affine_sharing"] == "per_expert"
    assert report["trainable_parameters"] == 5 * 9 * 2 * 120 * 120
    assert report["parameters"] == 5 * 9 * 2 * 120 * 120
    assert len(report["per_stage"]) == 5
    assert all(stage["trainable_parameters"] == 9 * 2 * 120 * 120 for stage in report["per_stage"])
    expert_report = model.expert_parameter_report()["by_type"]
    # Per expert: independent masks plus one enabled scalar phase bias. Output
    # amplitude gains are disabled and therefore are persistent buffers only.
    assert expert_report["d2nn"]["trainable_parameters"] == 3 * (5 * 120 * 120 + 1)
    assert expert_report["fourier"]["trainable_parameters"] == 3 * (5 * 120 * 120 + 1)
    assert expert_report["fiber"]["trainable_parameters"] == 3 * (4 * 120 * 120 + 100 + 100 + 1)
