from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    FLIP_PERMUTATION,
    JOINT_NAMES,
    PoseRecord,
    make_heatmaps,
    normalize_joints_array,
    pose_scales,
    split_standard_protocol,
    transform_person,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import (
    hardargmax_coordinates,
    masked_coordinate_loss,
    masked_heatmap_mse,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import (
    PoseMetricAccumulator,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
    LightweightPoseHead,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import (
    load_settings,
)


def _record(index: int, source: str) -> PoseRecord:
    points = np.zeros((14, 2), dtype=np.float32)
    return PoseRecord(
        sample_id=f"{source}_{index}", source=source, split="unassigned",
        source_index=index, image_path=Path(f"/{source}/{index}.jpg"),
        keypoints=points, raw_visibility=np.zeros(14, dtype=np.float32),
    )


def test_joint_mat_layouts_are_normalized() -> None:
    canonical = np.arange(5 * 14 * 3, dtype=np.float32).reshape(5, 14, 3)
    assert np.array_equal(normalize_joints_array(canonical), canonical)
    assert np.array_equal(normalize_joints_array(canonical.transpose(2, 1, 0)), canonical)
    assert np.array_equal(normalize_joints_array(canonical.transpose(1, 2, 0)), canonical)
    assert np.array_equal(normalize_joints_array(canonical.transpose(0, 2, 1)), canonical)

    # HR-LSPET has x/y coordinates only; the loader supplies a neutral third
    # channel so the rest of the data pipeline stays format-compatible.
    xy = canonical[..., :2]
    normalized_xy = normalize_joints_array(xy.transpose(1, 2, 0))
    assert np.array_equal(normalized_xy[..., :2], xy)
    assert np.count_nonzero(normalized_xy[..., 2]) == 0


def test_standard_protocol_is_11000_train_1000_test_and_disjoint() -> None:
    lsp = [_record(index, "lsp") for index in range(2000)]
    lspet = [_record(index, "lspet") for index in range(10000)]
    train, test = split_standard_protocol(lsp, lspet)
    assert len(train) == 11000
    assert len(test) == 1000
    assert {r.source_index for r in train if r.source == "lsp"} == set(range(1000))
    assert {r.source_index for r in test} == set(range(1000, 2000))
    assert {r.sample_id for r in train}.isdisjoint({r.sample_id for r in test})


def test_person_transform_and_flip_preserve_joint_semantics() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    points = np.stack((np.linspace(20, 80, 14), np.linspace(25, 75, 14)), axis=1).astype(np.float32)
    transformed, flipped, visible, _, was_flipped = transform_person(
        image, points, image_size=224, crop_margin=1.25, training=True,
        scale_jitter=0.0, center_jitter=0.0, flip_probability=1.0,
        brightness_jitter=0.0, contrast_jitter=0.0,
    )
    assert transformed.size == (224, 224)
    assert visible.all()
    assert was_flipped is True
    # Right ankle after a flip comes from the former left ankle (index 5).
    _, unflipped, _, _, was_flipped = transform_person(
        image, points, image_size=224, crop_margin=1.25, training=False,
        scale_jitter=0.0, center_jitter=0.0, flip_probability=0.0,
        brightness_jitter=0.0, contrast_jitter=0.0,
    )
    assert flipped[0, 0] == pytest.approx(223.0 - unflipped[FLIP_PERMUTATION[0], 0])
    assert flipped[0, 1] == pytest.approx(unflipped[FLIP_PERMUTATION[0], 1])
    assert was_flipped is False


def test_heatmaps_shape_mask_and_peak() -> None:
    points = np.full((14, 2), np.nan, dtype=np.float32)
    points[0] = (112.0, 112.0)
    visible = np.zeros(14, dtype=bool)
    visible[0] = True
    heatmaps = make_heatmaps(points, visible, 224, 56, 2.0)
    assert heatmaps.shape == (14, 56, 56)
    assert heatmaps[0].max() == pytest.approx(1.0)
    assert heatmaps[1:].count_nonzero() == 0
    prediction = hardargmax_coordinates(heatmaps[None], 224)
    assert torch.allclose(prediction[0, 0], torch.tensor([113.5, 113.5]), atol=2.0)


def test_masked_losses_ignore_invisible_joints_and_backward() -> None:
    prediction = torch.randn(2, 14, 56, 56, requires_grad=True)
    target = torch.zeros_like(prediction)
    visible = torch.zeros(2, 14, dtype=torch.bool)
    visible[:, 0] = True
    keypoints = torch.full((2, 14, 2), float("nan"))
    keypoints[:, 0] = torch.tensor([100.0, 120.0])
    loss = masked_heatmap_mse(prediction, target, visible)
    loss = loss + masked_coordinate_loss(prediction, keypoints, visible, 224)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert prediction.grad[:, 1:].abs().sum() == 0


def test_pose_head_shape_parameters_and_gradients() -> None:
    head = LightweightPoseHead(224, projection_dim=32, decoder_channels=(32, 16), heatmap_size=56)
    output = head(torch.randn(2, 224, 14, 14))
    assert output.shape == (2, 14, 56, 56)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_perfect_predictions_have_perfect_pck() -> None:
    heatmaps = torch.full((1, 14, 56, 56), -100.0)
    target = torch.zeros(1, 14, 2)
    for joint in range(14):
        y, x = divmod(joint + 10, 56)
        heatmaps[0, joint, y, x] = 100.0
        target[0, joint] = hardargmax_coordinates(heatmaps[:, joint:joint + 1], 224)[0, 0]
    visible = torch.ones(1, 14, dtype=torch.bool)
    accumulator = PoseMetricAccumulator()
    accumulator.update(heatmaps, target, visible, torch.tensor([100.0]), torch.tensor([50.0]), 224)
    metrics = accumulator.compute()
    assert metrics["pck_at_0.2_torso"] == 1.0
    assert metrics["pckh_at_0.5_head"] == 1.0
    assert metrics["mean_pixel_error"] == 0.0


def test_pose_scale_definitions() -> None:
    points = np.zeros((14, 2), dtype=np.float32)
    points[8], points[3] = (0, 0), (3, 4)
    points[9], points[2] = (0, 0), (6, 8)
    points[12], points[13] = (0, 0), (0, 4)
    torso, head = pose_scales(points, np.ones(14, dtype=bool))
    assert torso == pytest.approx(7.5)
    assert head == pytest.approx(8.0)


def test_formal_settings_keep_validated_optics_and_paths_inside_experiment() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root / "configs" / "lsp_pose.yaml")
    assert settings.num_experts == 16
    assert settings.expert_layers == 1
    assert settings.top_k == 4
    assert settings.canvas_size == 1026
    assert settings.detector_output_size == 224
    assert settings.output_dir.parent == root / "runs"
    assert settings.data_root.name == "lsp_pose"
    assert settings.lspet_expected_count == 9428
    assert len(JOINT_NAMES) == 14
