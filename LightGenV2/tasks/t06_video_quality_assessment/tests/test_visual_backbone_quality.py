from __future__ import annotations

import torch

from LightGenV2.tasks.t06_video_quality_assessment import visual_backbone_quality as subject


def test_xavier_rows_are_deterministic_and_have_expected_shape() -> None:
    first = subject.deterministic_xavier_rows(42)
    second = subject.deterministic_xavier_rows(42)
    different = subject.deterministic_xavier_rows(43)
    assert first.shape == (5, 2048)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_quality_head_has_qwen_matched_parameter_budget() -> None:
    head = subject.FiveQualityRows(torch.zeros(5, 2048))
    assert sum(parameter.numel() for parameter in head.parameters()) == 10240
    output = head(torch.ones(3, 2048))
    assert output.shape == (3, 5)


def test_config_contract_accepts_clip_and_yolo() -> None:
    for backbone in subject.BACKBONES:
        raw = {
            "task": {"targets": ["temporal", "spatial"]},
            "input": {
                "frame_count": 4,
                "frame_fractions": [0.10, 0.37, 0.63, 0.90],
            },
            "model": {
                "backbone": backbone,
                "backbone_frozen": True,
                "trainable_module": "five_bias_free_quality_rows",
                "frame_feature_width": 512,
                "readout_width": 2048,
                "trainable_parameters": 10240,
            },
            "training": {"epochs": 50},
        }
        subject.validate_config(raw)
