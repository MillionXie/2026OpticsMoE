from __future__ import annotations

from pathlib import Path

import torch

from ..model import QwenStemOpticalImageNetBackbone
from ..stem import STEM_FORMAT, StaticQwenPatchStem


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
        "residual_hidden_channels": 64,
        "residual_downsample_factor": 7,
    }


def test_extracted_stem_is_frozen_and_has_expected_shape(tmp_path: Path) -> None:
    stem = StaticQwenPatchStem(fake_stem(tmp_path / "stem.pt"))
    output = stem(torch.rand(2, 3, 224, 224))
    assert output.shape == (2, 196, 1024)
    assert list(stem.parameters()) == []
    assert stem.parameter_report()["contains_transformer"] is False


def test_locked_model_has_million_scale_optics_and_no_qwen_parameters(tmp_path: Path) -> None:
    model = QwenStemOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    report = model.parameter_report()
    assert report["optical_phase_parameters"] == 8 * 3 * 224 * 224
    assert report["optical_fraction_of_trainable"] >= 0.50
    assert report["minimum_optical_gate"] >= 0.50
    assert report["contains_electronic_transformer"] is False
    assert all(not parameter.requires_grad for parameter in model.stem.parameters())


def test_adapter_packs_196_tokens_and_copies_three_latent_banks(tmp_path: Path) -> None:
    model = QwenStemOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    fields, qwen_tokens = model.optical_input(torch.zeros(1, 3, 224, 224))
    assert qwen_tokens.shape == (1, 196, 1024)
    assert fields.shape == (1, 3, 224, 224)
    assert torch.equal(fields[:, 0], fields[:, 1])
    assert torch.equal(fields[:, 1], fields[:, 2])
    assert torch.count_nonzero(fields[:, :, 196:]) == 0
