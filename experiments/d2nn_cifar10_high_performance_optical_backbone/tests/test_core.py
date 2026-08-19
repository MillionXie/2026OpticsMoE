from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.fixed_feedback_training import (
    _paired_test_accuracy_deltas,
    _sample_summary,
)
from experiments.d2nn_cifar10_high_performance_optical_backbone.datasets import _split
from experiments.d2nn_cifar10_high_performance_optical_backbone.formal_settings import load_formal_settings
from experiments.d2nn_cifar10_high_performance_optical_backbone.model import OpticalClassifier
from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    ElectronicSkipProcessor,
    OpticalOEOStage,
    ResidualMixer,
    physical_phase,
)
from experiments.d2nn_cifar10_high_performance_optical_backbone.settings import load_settings


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_settings_and_stratified_split() -> None:
    settings = load_settings(CONFIG)
    assert settings.num_classes == 10
    train, validation = _split([0] * 10 + [1] * 10, validation_per_class=2, seed=42)
    assert len(train) == 16
    assert len(validation) == 4
    assert not set(train) & set(validation)


def test_constrained_residual_initialization_and_bound() -> None:
    mixer = ResidualMixer("constrained", main_init=0.5, main_min=0.35)
    assert torch.isclose(mixer.main_weight(), torch.tensor(0.5), atol=1e-5)
    mixer.logit.data.fill_(-100.0)
    assert float(mixer.main_weight()) >= 0.35 - 1e-6


def test_electronic_skip_processor_is_bounded_and_shape_preserving() -> None:
    processor = ElectronicSkipProcessor(
        channels=3,
        mode="depthwise",
        hidden_channels=12,
        downsample_factor=4,
        scale_init=0.10,
        scale_max=0.25,
        long_skip_enabled=True,
        long_skip_weight_init=0.10,
        long_skip_weight_max=0.25,
        eps=1e-5,
    )
    value = torch.rand(2, 3, 8, 8)
    source = torch.rand_like(value)
    output = processor(value, long_skip=source)
    assert output.shape == value.shape
    assert torch.all(output >= 0.0)
    assert 0.0 <= float(processor.transform_scale()) <= 0.25
    assert 0.0 <= float(processor.long_skip_weight()) <= 0.25
    spatial_rms = output.square().mean(dim=(-2, -1)).sqrt()
    torch.testing.assert_close(spatial_rms, torch.ones_like(spatial_rms), rtol=2e-4, atol=2e-4)


def test_phase_parameterization_is_physical() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    phase = physical_phase(raw)
    assert torch.all(phase >= 0.0)
    assert torch.all(phase <= 2.0 * torch.pi)


def test_forward_backward_and_ablation_shapes() -> None:
    settings = load_settings(CONFIG)
    optical = replace(settings.optical, canvas_size=16, pool_size=2, hidden_dim=16)
    model = OpticalClassifier(optical, num_classes=10)
    images = torch.rand(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert all(stage.raw_phase.grad is not None for stage in model.stages)
    assert model(images, ablation="optical_off").shape == (2, 10)
    assert model(images, ablation="phase_random").shape == (2, 10)
    assert model(images, ablation="phase_shuffle").shape == (2, 10)
    assert model(images, ablation="electronic_skip_off").shape == (2, 10)
    assert model(images, ablation="long_skip_off").shape == (2, 10)


def test_conv_readout_and_long_skip_forward_backward() -> None:
    settings = load_settings(CONFIG)
    optical = replace(
        settings.optical,
        canvas_size=16,
        pool_size=4,
        readout_mode="conv",
        conv_channels=8,
        electronic_skip_mode="depthwise",
        electronic_skip_hidden_channels=6,
        long_skip_enabled=True,
    )
    model = OpticalClassifier(optical, num_classes=10)
    images = torch.rand(2, 3, 32, 32)
    logits = model(images)
    logits.square().mean().backward()
    assert logits.shape == (2, 10)
    assert all(stage.raw_phase.grad is not None for stage in model.stages)
    assert any(
        parameter.grad is not None
        for stage in model.stages
        for parameter in stage.electronic_skip.parameters()
    )


def test_dual_pool_readout_forward_backward() -> None:
    settings = load_settings(CONFIG)
    optical = replace(
        settings.optical,
        canvas_size=16,
        pool_size=4,
        hidden_dim=16,
        readout_mode="dual_pool",
    )
    model = OpticalClassifier(optical, num_classes=10)
    images = torch.rand(2, 3, 32, 32)
    logits = model(images)
    logits.mean().backward()
    assert logits.shape == (2, 10)
    assert all(stage.raw_phase.grad is not None for stage in model.stages)


def test_low_resolution_electronic_budget_and_forward() -> None:
    root = CONFIG.parent
    settings = load_settings(root / "a13_lowres_electronic_residual.yaml")
    optical = replace(settings.optical, canvas_size=16, pool_size=2, hidden_dim=16)
    model = OpticalClassifier(optical, num_classes=10)
    logits = model(torch.rand(2, 3, 32, 32))
    residual_parameters = sum(parameter.numel() for parameter in model.residual_parameters())
    assert logits.shape == (2, 10)
    assert 300_000 <= residual_parameters <= 400_000
    assert model.estimated_electronic_macs() > 0


def test_a13_replication_changes_only_the_seed_list() -> None:
    root = CONFIG.parent
    screening = load_settings(root / "a13_lowres_electronic_residual.yaml")
    replication = load_settings(root / "a13_multiseed_replication.yaml")
    assert replication.training.seeds == (1234, 2026, 2027)
    assert replace(replication.training, seeds=screening.training.seeds) == screening.training
    assert replication.output_dir == screening.output_dir
    assert replication.optical == screening.optical
    assert replication.data == screening.data
    assert replication.optimizer == screening.optimizer


def test_optical_off_does_not_depend_on_phase() -> None:
    settings = load_settings(CONFIG)
    optical = replace(settings.optical, canvas_size=16, pool_size=2, hidden_dim=16, dropout=0.0)
    model = OpticalClassifier(optical, num_classes=10).eval()
    images = torch.rand(2, 3, 32, 32)
    before = model(images, ablation="optical_off")
    for stage in model.stages:
        stage.raw_phase.data.normal_()
    after = model(images, ablation="optical_off")
    torch.testing.assert_close(before, after)


def _stage() -> OpticalOEOStage:
    return OpticalOEOStage(
        size=8,
        channels=2,
        wavelength_m=5.32e-7,
        pixel_size_m=1.6e-5,
        distance_m=0.05,
        phase_init_std=0.05,
        layernorm_eps=1e-5,
        residual_mode="constrained",
        residual_main_init=0.5,
        residual_main_min=0.35,
        normalize_branch_rms=True,
        random_seed=17,
    )


def test_fixed_feedback_matches_bp_when_connector_matches_current_phase() -> None:
    bp = _stage()
    fixed = _stage()
    fixed.load_state_dict(bp.state_dict())
    fixed.set_feedback("fa_pretrained", bp.phase().detach())
    bp_input = torch.rand(2, 2, 8, 8, requires_grad=True)
    fixed_input = bp_input.detach().clone().requires_grad_(True)
    bp(bp_input).square().mean().backward()
    fixed(fixed_input).square().mean().backward()
    torch.testing.assert_close(bp_input.grad, fixed_input.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(bp.raw_phase.grad, fixed.raw_phase.grad, rtol=2e-4, atol=2e-5)


def test_random_feedback_changes_connector_but_not_forward() -> None:
    bp = _stage()
    fixed = _stage()
    fixed.load_state_dict(bp.state_dict())
    random_phase = 2.0 * torch.pi * torch.rand_like(bp.phase())
    fixed.set_feedback("fa_random", random_phase)
    bp_input = torch.rand(2, 2, 8, 8, requires_grad=True)
    fixed_input = bp_input.detach().clone().requires_grad_(True)
    bp_output = bp(bp_input)
    fixed_output = fixed(fixed_input)
    torch.testing.assert_close(bp_output, fixed_output)
    bp_output.square().mean().backward()
    fixed_output.square().mean().backward()
    assert not torch.allclose(bp_input.grad, fixed_input.grad)
    torch.testing.assert_close(bp.raw_phase.grad, fixed.raw_phase.grad, rtol=2e-4, atol=2e-5)


def test_formal_pilot_has_only_the_expected_contract() -> None:
    formal = load_formal_settings(CONFIG.parent / "formal_pilot.yaml")
    assert formal.formal.finetune_seeds == (2026, 2027, 2028)
    assert formal.formal.head_warmup_epochs == 10
    assert formal.formal.finetune_epochs == 20
    assert len(formal.formal.source_checkpoint_sha256) == 64


def test_formal_a13_matches_the_locked_backbone_contract() -> None:
    root = CONFIG.parent
    a13 = load_settings(root / "a13_lowres_electronic_residual.yaml")
    formal = load_formal_settings(root / "formal_a13_high_performance.yaml")
    assert formal.base.optical == a13.optical
    assert formal.base.data == a13.data
    assert formal.base.optimizer == a13.optimizer
    assert formal.formal.finetune_seeds == (2026, 2027, 2028)
    assert formal.formal.finetune_epochs == a13.training.epochs == 50
    assert formal.formal.phase_learning_rate == a13.optimizer.phase_learning_rate
    assert formal.formal.residual_learning_rate == a13.optimizer.residual_learning_rate
    assert formal.formal.electronic_learning_rate == a13.optimizer.electronic_learning_rate


def test_a07_changes_only_the_optical_floor_contract() -> None:
    baseline = load_settings(CONFIG.parent / "a04_cifar100_to_cifar10.yaml")
    candidate = load_settings(CONFIG.parent / "a07_high_optical_cifar100_to_cifar10.yaml")
    assert baseline.optical.residual_main_min == 0.35
    assert candidate.optical.residual_main_min == 0.50
    assert baseline.optical.residual_main_init == candidate.optical.residual_main_init == 0.50
    assert baseline.optimizer == candidate.optimizer
    assert baseline.training == candidate.training


def test_a08_a10_keep_a07_training_budget_and_optical_floor() -> None:
    root = CONFIG.parent
    a07 = load_settings(root / "a07_high_optical_cifar100_to_cifar10.yaml")
    a08 = load_settings(root / "a08_pointwise_electronic_residual.yaml")
    a09 = load_settings(root / "a09_depthwise_electronic_residual.yaml")
    a10 = load_settings(root / "a10_depthwise_unet_skips.yaml")
    assert all(candidate.optical.residual_main_min == 0.50 for candidate in (a08, a09, a10))
    assert all(candidate.optimizer == a07.optimizer for candidate in (a08, a09, a10))
    assert all(candidate.training == a07.training for candidate in (a08, a09, a10))
    assert a08.optical.electronic_skip_mode == "pointwise"
    assert a09.optical.electronic_skip_mode == "depthwise"
    assert a10.optical.electronic_skip_mode == "depthwise"
    assert not a08.optical.long_skip_enabled
    assert not a09.optical.long_skip_enabled
    assert a10.optical.long_skip_enabled


def test_formal_comparison_helpers_use_sample_statistics_and_paired_seeds() -> None:
    summary = _sample_summary([0.7, 0.8, 0.9])
    assert summary["n"] == 3
    assert abs(float(summary["mean"]) - 0.8) < 1.0e-12
    assert abs(float(summary["std"]) - 0.1) < 1.0e-12

    rows = [
        {"method": method, "seed": seed, "test_accuracy": accuracy}
        for method, seed, accuracy in (
            ("noft", 2026, 0.50),
            ("noft", 2027, 0.51),
            ("bp", 2026, 0.72),
            ("bp", 2027, 0.73),
            ("fa_pretrained", 2026, 0.70),
            ("fa_pretrained", 2027, 0.71),
            ("fa_random", 2026, 0.60),
            ("fa_random", 2027, 0.61),
        )
    ]
    contrasts = _paired_test_accuracy_deltas(rows)
    assert all(
        abs(delta - 0.02) < 1.0e-12
        for delta in contrasts["bp_minus_fa_pretrained"]["by_seed"].values()
    )
    assert all(
        abs(delta - 0.1) < 1.0e-12
        for delta in contrasts["fa_pretrained_minus_fa_random"]["by_seed"].values()
    )
