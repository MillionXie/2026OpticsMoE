from __future__ import annotations

from pathlib import Path

import torch
import pytest

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    QwenStemSlimMixerOpticalImageNetBackbone,
)

from ..model import QwenStemDualScaleOpticalImageNetBackbone


def fake_stem(path: Path) -> Path:
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": torch.zeros(1024, 3, 16, 16),
            "conv2d_bias": torch.zeros(1024),
            "position_embedding": torch.randn(196, 1024) * 0.01,
            "metadata": {
                "image_size": 224,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        },
        path,
    )
    return path


def model_config() -> dict[str, object]:
    return {
        "canvas_size": 224,
        "optical_channels": 3,
        "num_stages": 8,
        "token_dim": 224,
        "num_classes": 1000,
        "head_hidden_dim": 448,
        "phase_init_std": 0.10,
        "optical_gate_init": 0.60,
        "optical_gate_min": 0.50,
        "mixer_width": 96,
        "mixer_expansion": 2.0,
        "mixer_kernel_size": 3,
        "mixer_dropout": 0.10,
        "local_propagation_distance_m": 0.005,
        "global_propagation_distance_m": 0.05,
    }


def test_dual_scale_schedule_and_parameter_budget(tmp_path: Path) -> None:
    model = QwenStemDualScaleOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    report = model.parameter_report()
    assert [stage.optical_mixing_role for stage in model.stages] == [
        "local",
        "global",
    ] * 4
    assert [stage.propagation_distance_m for stage in model.stages] == [
        0.005,
        0.05,
    ] * 4
    assert report["optical_mixer_variant"] == "dual_scale_serial_local_global"
    assert report["optical_phase_parameters"] == 8 * 3 * 224 * 224
    assert report["residual_electronic_parameters"] == 733_472
    assert report["optical_fraction_of_backbone_trainable"] >= 0.50
    assert report["adds_trainable_parameters_over_p09"] == 0


def test_local_and_global_transfer_functions_are_distinct(tmp_path: Path) -> None:
    model = QwenStemDualScaleOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    local = model.stages[0].propagator.transfer_function
    global_ = model.stages[1].propagator.transfer_function
    assert local.shape == global_.shape == (224, 224)
    assert not torch.allclose(local, global_)


def _impulse_energy_radius(propagator, fraction: float) -> float:
    field = torch.zeros(1, 1, 224, 224, dtype=torch.complex64)
    field[0, 0, 112, 112] = 1.0
    energy = propagator(field).abs().square()[0, 0]
    coordinate = torch.arange(224)
    distance = (coordinate - 112).abs()
    distance = torch.minimum(distance, 224 - distance).float()
    yy, xx = torch.meshgrid(distance, distance, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square()).flatten()
    order = torch.argsort(radius)
    cumulative = torch.cumsum(energy.flatten()[order], dim=0) / energy.sum()
    index = int(torch.searchsorted(cumulative, torch.tensor(float(fraction))))
    return float(radius[order[index]])


def test_local_kernel_has_smaller_measured_receptive_field(tmp_path: Path) -> None:
    model = QwenStemDualScaleOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    local_r90 = _impulse_energy_radius(model.stages[0].propagator, 0.90)
    global_r90 = _impulse_energy_radius(model.stages[1].propagator, 0.90)
    assert 2.0 < local_r90 < 10.0
    assert global_r90 > 40.0
    assert local_r90 < global_r90


def test_p10_trainable_initialization_matches_p09(tmp_path: Path) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    torch.manual_seed(123)
    p10 = QwenStemDualScaleOpticalImageNetBackbone(stem, model_config())
    torch.manual_seed(123)
    p09 = QwenStemSlimMixerOpticalImageNetBackbone(stem, model_config())
    p10_parameters = dict(p10.named_parameters())
    p09_parameters = dict(p09.named_parameters())
    assert p10_parameters.keys() == p09_parameters.keys()
    for name in p09_parameters:
        torch.testing.assert_close(p10_parameters[name], p09_parameters[name])


def test_p09_strict_load_rejects_p10_checkpoint(tmp_path: Path) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    p10 = QwenStemDualScaleOpticalImageNetBackbone(stem, model_config())
    p09 = QwenStemSlimMixerOpticalImageNetBackbone(stem, model_config())
    with pytest.raises(RuntimeError, match="p10_dual_scale_architecture_signature"):
        p09.load_state_dict(p10.state_dict(), strict=True)
