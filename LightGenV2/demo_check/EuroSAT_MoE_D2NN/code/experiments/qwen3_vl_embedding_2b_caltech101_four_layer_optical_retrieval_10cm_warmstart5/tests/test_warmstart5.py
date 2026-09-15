from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.offline_tail import (
    LanguageGlobalOfflineTail,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    RobustCCDNormalizer,
)

from ..modeling import (
    GATE_KEYS,
    apply_stage_trainability,
    merge_surrogate_states,
)
from ..settings import load_settings


def _states():
    target = {
        "core.input_adapter.weight": torch.zeros(2, 3),
        "core.blocks.0.weight": torch.zeros(2, 2),
        "core.optical_branch.core.raw_phase": torch.zeros(4, 4),
        "core.optical_branch.readout.weight": torch.zeros(2, 4),
        "core.block1_optical_fusion_logit": torch.tensor(-5.0),
        "core.block2_optical_fusion_logit": torch.tensor(-5.0),
    }
    electronic = {
        "core.input_adapter.weight": torch.ones(2, 3),
        "core.blocks.0.weight": torch.full((2, 2), 2.0),
    }
    optical = {
        **electronic,
        "core.optical_branch.core.raw_phase": torch.full((4, 4), 3.0),
        "core.optical_branch.readout.weight": torch.full((2, 4), 4.0),
        "core.block1_optical_fusion_logit": torch.tensor(99.0),
        "core.block2_optical_fusion_logit": torch.tensor(99.0),
    }
    return target, electronic, optical


def test_strict_merge_uses_electronics_optics_and_resets_gates() -> None:
    target, electronic, optical = _states()
    merged, report = merge_surrogate_states(target, electronic, optical)
    torch.testing.assert_close(merged["core.input_adapter.weight"], electronic["core.input_adapter.weight"])
    torch.testing.assert_close(
        merged["core.optical_branch.core.raw_phase"],
        optical["core.optical_branch.core.raw_phase"],
    )
    for key in GATE_KEYS:
        torch.testing.assert_close(merged[key], target[key])
    assert report == {
        "electronic_tensor_count": 2,
        "optical_tensor_count": 2,
        "gate_tensor_count": 2,
        "gate_keys_reset_not_loaded": sorted(GATE_KEYS),
    }


def test_strict_merge_rejects_wrong_electronic_architecture_and_shape() -> None:
    target, electronic, optical = _states()
    with pytest.raises(RuntimeError, match="exact 2D/no-DeepStack"):
        merge_surrogate_states(target, {**electronic, "unexpected": torch.zeros(1)}, optical)
    bad = dict(electronic)
    bad["core.blocks.0.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        merge_surrogate_states(target, bad, optical)


class _FakeReplacement:
    def __init__(self) -> None:
        self.vision_surrogate = _FakeSurrogate()
        self.language_surrogate = _FakeSurrogate()

    def configure_student_trainability(self) -> None:
        self.vision_surrogate.requires_grad_(True)
        self.language_surrogate.requires_grad_(True)

    def trainable_parameters(self):
        yield from self.vision_surrogate.parameters()
        yield from self.language_surrogate.parameters()


class _FakeSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Module()
        self.core.electronic = nn.Linear(2, 2)
        self.core.optical_branch = nn.Linear(2, 2)
        self.core.residual_logit = nn.Parameter(torch.zeros(()))


def test_stage_a_freezes_electronics_head_and_leaves_only_optics() -> None:
    replacement = _FakeReplacement()
    readout = nn.Linear(2, 2)
    apply_stage_trainability(replacement, readout, "optical_calibration")
    assert not any(
        parameter.requires_grad
        for parameter in replacement.vision_surrogate.core.electronic.parameters()
    )
    assert all(
        parameter.requires_grad
        for surrogate in (replacement.vision_surrogate, replacement.language_surrogate)
        for parameter in surrogate.core.optical_branch.parameters()
    )
    assert not any(parameter.requires_grad for parameter in readout.parameters())


def test_stage_b_unfreezes_joint_path_but_keeps_language_unused_residual_frozen() -> None:
    replacement = _FakeReplacement()
    readout = nn.Linear(2, 2)
    apply_stage_trainability(replacement, readout, "joint")
    assert all(parameter.requires_grad for parameter in readout.parameters())
    assert all(
        parameter.requires_grad
        for parameter in replacement.vision_surrogate.parameters()
    )
    assert not replacement.language_surrogate.core.residual_logit.requires_grad


def test_release_configs_seal_test_and_fix_fusion_floor() -> None:
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    stage_a = load_settings(project / "configs/release/stage1_optical_calibration.yaml")
    stage_b = load_settings(project / "configs/release/stage2_joint_sealed_test.yaml")
    quick = load_settings(project / "configs/release/quick_last_stage_10x10.yaml")
    hardware = load_settings(
        project / "configs/release/stage2_joint_hardware_canonical_ccd.yaml"
    )
    quick_hardware = load_settings(
        project / "configs/release/quick_last_stage_10x10_canonical_ccd.yaml"
    )
    assert stage_a.warmstart_stage == "optical_calibration"
    assert stage_b.warmstart_stage == "joint"
    assert not stage_a.evaluate_test_each_epoch
    assert not stage_b.evaluate_test_each_epoch
    assert stage_a.optical_fusion_minimum == stage_b.optical_fusion_minimum == 0.05
    assert stage_a.optical_fusion_initial == stage_b.optical_fusion_initial == 0.055
    assert stage_a.optimizer_steps_per_epoch == stage_b.optimizer_steps_per_epoch == 12
    assert quick.gallery_images_per_sku == 1
    assert quick.train_limit_per_sku == quick.test_limit_per_sku == 10
    assert hardware.hardware_ccd_flip_vertical is False
    assert hardware.hardware_ccd_flip_horizontal is False
    assert quick_hardware.hardware_ccd_flip_vertical is False
    assert quick_hardware.hardware_ccd_flip_horizontal is False


def test_simulation_and_offline_hardware_use_identical_ccd_normalization() -> None:
    settings = SimpleNamespace(
        active_size=4,
        language_optical_normalization_clip=12.0,
        language_optical_log_compression=1.0,
    )
    simulation_normalizer = RobustCCDNormalizer(settings)
    offline_contract = SimpleNamespace(
        detector_size=4,
        ccd_relative_clip=12.0,
        ccd_log_compression=1.0,
    )
    intensity = torch.tensor(
        [
            [
                [-2.0, 0.0, 1.0, 2.0],
                [4.0, 8.0, 16.0, 32.0],
                [0.5, 1.5, 3.0, 6.0],
                [12.0, 24.0, 48.0, 96.0],
            ]
        ]
    )

    simulation = simulation_normalizer(intensity)
    measured = LanguageGlobalOfflineTail._normalize_ccd(
        offline_contract, intensity
    )

    torch.testing.assert_close(simulation, measured, rtol=0.0, atol=0.0)
