from pathlib import Path

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
)

from ..optical_blocks import (
    LanguageTwoBlockOpticalCore,
    VisionTwoBlockOpticalCore,
)
from ..hardware_bridge import _downstream_optimizer, _downstream_parameters
from ..settings import load_settings


EXPERIMENT = Path(__file__).resolve().parents[1]


def _settings():
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "release"
        / "caltech101_four_layer_optical_joint.yaml"
    )
    settings.electronic_width = 6
    settings.electronic_expansion = 2.0
    settings.electronic_dropout = 0.0
    settings.language_optical_max_shift_pixels = 0
    settings.language_optical_phase_shift_pixels = 0
    settings.language_optical_ccd_shift_pixels = 0
    settings.k_space_constraint_enabled = False
    return settings


def test_vision_and_language_have_two_independent_fusion_gates() -> None:
    settings = _settings()
    vision = VisionTwoBlockOpticalCore(10, 224, settings).train()
    language = LanguageTwoBlockOpticalCore(12, 224, settings).train()
    vision_output, _ = vision.forward_groups(
        [torch.randn(4, 10)], causal=False, spatial_shapes=[(1, 2, 2)]
    )
    language_output, _ = language.forward_groups(
        [torch.randn(5, 12)], causal=True
    )
    vision_balance, vision_importance = vision.optical_branch.core.router_losses()
    language_balance, language_importance = language.optical_branch.core.router_losses()
    loss = (
        vision_output.square().mean()
        + language_output.square().mean()
        + 0.02 * (vision_balance + language_balance)
        + 0.005 * (vision_importance + language_importance)
    )
    loss.backward()
    assert vision_output.shape == (4, 10)
    assert language_output.shape == (5, 12)
    for core in (vision, language):
        assert core.block1_optical_fusion_logit.grad is not None
        assert core.block2_optical_fusion_logit.grad is not None
        phases = [
            parameter
            for name, parameter in core.optical_branch.named_parameters()
            if "raw_phase" in name
        ]
        assert phases and all(parameter.grad is not None for parameter in phases)
        assert all(torch.isfinite(parameter.grad).all() for parameter in phases)
        router_parameters = list(core.optical_branch.core.router.parameters())
        assert router_parameters
        assert all(parameter.grad is not None for parameter in router_parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in router_parameters)


def test_17um_release_restores_phase_and_router_training_controls() -> None:
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "release"
        / "caltech101_four_layer_optical_joint_17um.yaml"
    )
    assert settings.language_optical_pixel_pitch_um == 17.0
    assert settings.phase_learning_rate == 5.0e-4
    assert settings.router_learning_rate == 2.0e-4
    assert settings.lambda_router_balance == 0.02
    assert settings.lambda_router_importance == 0.005
    assert settings.language_optical_max_shift_pixels == 12
    assert settings.language_optical_phase_shift_pixels == 12
    assert settings.language_optical_ccd_shift_pixels == 12
    assert settings.hardware_amplitude_slm_pixel_pitch_um == 17.0
    assert settings.hardware_phase_slm_pixel_pitch_um == 8.0
    assert settings.hardware_phase_flip_vertical is True
    assert settings.hardware_phase_slm_center_x == 980.0
    assert settings.hardware_phase_slm_center_y == 590.0


def test_strong_phase_release_uses_focus_epochs_and_stronger_fusion() -> None:
    settings = load_settings(
        EXPERIMENT
        / "configs"
        / "release"
        / "caltech101_four_layer_optical_joint_17um_strong_phase.yaml"
    )
    assert settings.phase_learning_rate == 4.0e-3
    assert settings.phase_focus_enabled is True
    assert settings.phase_focus_warmup_epochs == 5
    assert settings.phase_focus_interval_epochs == 3
    assert settings.optical_fusion_initial == 0.15
    assert settings.output_dir.name == (
        "caltech101_four_layer_moe4_joint_17um_strong_phase"
    )


def test_logical_measured_ccd_replaces_both_simulated_boundaries() -> None:
    settings = _settings()
    core = VisionTwoBlockOpticalCore(10, 224, settings).eval()
    measured_expert = torch.full((1, 478, 478), 2.0)
    measured_global = torch.full((1, 478, 478), 3.0)
    core.optical_branch.set_measured_ccd(
        expert=measured_expert, global_=measured_global
    )
    output, _ = core.forward_groups(
        [torch.randn(4, 10)], causal=False, spatial_shapes=[(1, 2, 2)]
    )
    assert output.shape == (4, 10)
    assert torch.equal(core.optical_branch.last_raw_expert_ccd, measured_expert)
    assert torch.equal(core.optical_branch.last_raw_ccd, measured_global)


def test_nested_optical_core_is_compatible_with_phase_artifacts() -> None:
    settings = _settings()
    core = VisionTwoBlockOpticalCore(10, 224, settings)
    tensors = phase_tensors(core.optical_branch.core)
    assert tensors["physical_expert_mosaic_rad"].shape == (478, 478)
    assert tensors["physical_global_phase_rad"].shape == (478, 478)


def test_hardware_finetune_keeps_phase_and_router_learning_rates_small() -> None:
    settings = _settings()

    class Surrogate(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

    class Replacement:
        def __init__(self) -> None:
            self.vision_surrogate = Surrogate(
                VisionTwoBlockOpticalCore(10, 224, settings)
            )
            self.language_surrogate = Surrogate(
                LanguageTwoBlockOpticalCore(12, 224, settings)
            )

        def phase_parameter_groups(self):
            vision = self.vision_surrogate.core.optical_branch.core
            language = self.language_surrogate.core.optical_branch.core
            return {
                "vision": [
                    parameter
                    for name, parameter in vision.named_parameters()
                    if "raw_phase" in name
                ],
                "language": [
                    parameter
                    for name, parameter in language.named_parameters()
                    if "raw_phase" in name
                ],
            }

        def router_parameters(self):
            return [
                *self.vision_surrogate.core.optical_branch.core.router.parameters(),
                *self.language_surrogate.core.optical_branch.core.router.parameters(),
            ]

    replacement = Replacement()
    readout = nn.Linear(12, 4)
    parameters = _downstream_parameters(
        replacement, readout, stage="vision_expert"
    )
    optimizer = _downstream_optimizer(
        parameters, replacement, readout, settings
    )
    learning_rates = {
        group["group_name"]: group["lr"] for group in optimizer.param_groups
    }
    assert learning_rates["downstream_electronic"] == settings.learning_rate
    assert learning_rates["downstream_phases"] == settings.phase_learning_rate
    assert learning_rates["downstream_routers"] == settings.router_learning_rate
    assert learning_rates["retrieval_readout"] == settings.readout_learning_rate


def test_language_detector_objective_reaches_every_declared_trainable_tensor() -> None:
    settings = _settings()
    core = LanguageTwoBlockOpticalCore(12, 224, settings).train()
    # These only form the Qwen hidden output. Retrieval deliberately reads the
    # cached 192-wide latent before this residual adapter.
    core.output_adapter.requires_grad_(False)
    core.residual_logit.requires_grad_(False)
    core.forward_groups([torch.randn(5, 12)], causal=True)
    detector = torch.cat(
        (
            core.last_latent_groups[0].mean(dim=0),
            core.last_latent_groups[0].amax(dim=0),
        )
    )
    detector.square().mean().backward()
    missing = [
        name
        for name, parameter in core.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
