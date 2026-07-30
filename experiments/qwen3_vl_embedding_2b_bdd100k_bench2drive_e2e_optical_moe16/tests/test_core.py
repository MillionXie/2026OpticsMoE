from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ..datasets_bdd100k import rasterize_targets
from ..datasets_bench2drive import normalized_driving_state, parse_annotation
from ..modeling import (
    CCDLinearRecombiner,
    DrivingActor,
    RoadStructureAuxiliaryHead,
    TwinQCritic,
    decode_normalized_action,
    encode_control_target,
    pool_packed_tokens,
    raw_ccd_readout,
)
from ..objectives import (
    auxiliary_structure_loss,
    behavior_cloning_loss,
    normalized_feature_loss,
    shaped_reward,
)
from ..settings import EXPERIMENT_DIR, load_settings
from ..smoke import run_smoke


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_default_run_paths_are_owned_by_experiment() -> None:
    settings = load_settings(EXPERIMENT_DIR / "configs" / "bench2drive.yaml")
    expected = (
        EXPERIMENT_DIR
        / "runs"
        / "qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16"
    ).resolve()
    assert settings.output_dir == expected
    assert settings.pretrained_backbone_checkpoint == (
        expected / "checkpoints" / "bdd_optical_backbone_best.pt"
    )


def test_settings_fix_requested_optical_geometry() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "bench2drive_smoke.yaml")
    assert settings.expert_layers == 1
    assert settings.num_experts == 16
    assert settings.top_k == 4
    assert settings.expert_size == 224
    assert settings.active_size == 986
    assert settings.canvas_size == 1026
    assert settings.detector_output_size == 224


def test_bdd_rasterizes_drivable_lane_and_participant() -> None:
    labels = [
        {
            "category": "drivable area",
            "poly2d": [
                {"vertices": [[0, 50], [99, 50], [99, 99], [0, 99]], "closed": True}
            ],
        },
        {
            "category": "lane",
            "poly2d": [{"vertices": [[50, 0], [50, 99]], "closed": False}],
        },
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}},
    ]
    targets = rasterize_targets(
        (100, 100),
        labels,
        lane_width=3,
        participant_categories={"car"},
    )
    assert np.asarray(targets["drivable"]).sum() > 0
    assert np.asarray(targets["lane"]).sum() > 0
    assert np.asarray(targets["road_participant"]).sum() > 0


def test_bench2drive_annotation_target_is_ego_local(tmp_path: Path) -> None:
    annotation = {
        "speed": 5.0,
        "steer": 0.1,
        "throttle": 0.5,
        "brake": 0.0,
        "x": 10.0,
        "y": 20.0,
        "theta": np.pi / 2,
        "x_target": 10.0,
        "y_target": 30.0,
        "next_command": 4,
    }
    row = parse_annotation(
        annotation,
        sample_id="route/00000",
        route_id="route",
        frame_id="00000",
        image_path=tmp_path / "00000.jpg",
        annotation_path=tmp_path / "00000.json.gz",
    )
    assert row.command == 3
    assert row.target_x_local == pytest.approx(10.0)
    assert row.target_y_local == pytest.approx(0.0, abs=1e-6)


def test_ccd_recombiner_and_actor_shapes_and_gradients() -> None:
    recombiner = CCDLinearRecombiner()
    ccd = torch.rand(2, 224, 224, requires_grad=True)
    signed = recombiner(ccd)
    assert signed.shape == (2, 224, 224)
    pooled = signed.mean(dim=1)
    conditioning = normalized_driving_state(
        torch.tensor([2.0, 5.0]),
        torch.tensor([0, 5]),
        torch.tensor([[10.0, 1.0], [5.0, -2.0]]),
        speed_scale=12.0,
        target_clip=50.0,
    )
    state = torch.cat([pooled, conditioning], dim=-1)
    actor = DrivingActor(hidden_dims=(32, 16))
    normalized = actor.forward_normalized(state)
    controls = decode_normalized_action(normalized)
    assert normalized.shape == controls.shape == (2, 3)
    assert torch.all((-1 <= controls[:, 0]) & (controls[:, 0] <= 1))
    assert torch.all((0 <= controls[:, 1:]) & (controls[:, 1:] <= 1))
    controls.mean().backward()
    assert ccd.grad is not None
    assert recombiner.linear.weight.grad is not None
    assert actor.control_head.weight.grad is not None


def test_real_moe16_optical_forward_backward() -> None:
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
        HomogeneousMoEOpticalCore,
    )

    settings = load_settings(EXPERIMENT / "configs" / "bench2drive_smoke.yaml")
    core = HomogeneousMoEOpticalCore(1024, 224, settings)
    recombiner = CCDLinearRecombiner()
    input_field = core.encode_groups([torch.randn(4, 1024)])
    field, routing = core.begin(input_field)
    field = core.run_stage(0, field, routing)
    field = core.propagator(core.global_phase(field))
    ccd = raw_ccd_readout(field, core)
    signed = recombiner(ccd)
    loss = signed.square().mean() + 0.03 * routing["balance_loss"]
    loss.backward()
    assert ccd.shape == (1, 224, 224)
    assert torch.all(ccd >= 0)
    assert routing["selected_mask"].sum().item() == 4
    assert core.expert_layers[0].experts[0].raw_phase.grad is not None
    assert core.global_phase.phase.raw_phase.grad is not None
    assert recombiner.linear.weight.grad is not None


def test_losses_and_padding_free_pooling() -> None:
    lengths = [3, 2]
    packed = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    pooled = pool_packed_tokens(packed, lengths)
    assert pooled.shape == (2, 4)
    student = torch.randn(5, 224, requires_grad=True)
    teacher = student.detach().clone()
    feature_loss, parts = normalized_feature_loss(
        student, teacher, cosine_weight=1.0, smooth_l1_weight=0.5
    )
    assert float(feature_loss) == pytest.approx(0.0, abs=1e-6)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    auxiliary, _ = auxiliary_structure_loss(
        logits, targets, weights=(0.25, 0.25, 0.15)
    )
    predicted = torch.randn(2, 3, requires_grad=True).tanh()
    controls = torch.tensor([[0.0, 0.5, 0.0], [0.1, 0.3, 0.2]])
    bc, _ = behavior_cloning_loss(
        predicted,
        controls,
        steer_weight=1,
        throttle_weight=1,
        brake_weight=1,
        exclusion_weight=0.1,
    )
    (feature_loss + auxiliary + bc).backward()


def test_reward_contains_requested_terms() -> None:
    settings = SimpleNamespace(
        reward_speed_scale=3.0,
        reward_lane_scale=2.0,
        reward_weights={
            "route_progress": 1.0,
            "target_speed": 0.25,
            "lane_keep": 0.2,
            "collision": 5.0,
            "offroad": 2.0,
            "red_light": 2.0,
            "control_smoothness": 0.05,
        },
    )
    reward, parts = shaped_reward(
        {
            "route_progress": 1.0,
            "speed": 8.0,
            "target_speed": 8.0,
            "lane_offset": 0.0,
            "collision": True,
            "offroad": False,
            "red_light": False,
        },
        [0.0, 0.5, 0.0],
        [0.1, 0.4, 0.0],
        settings,
    )
    assert set(parts) == {
        "route_progress",
        "target_speed",
        "lane_keep",
        "collision",
        "offroad",
        "red_light",
        "control_smoothness",
    }
    assert parts["collision"] == -5.0
    with pytest.raises(RuntimeError, match="reward signals"):
        shaped_reward({}, [0, 0, 0], None, settings)


def test_dependency_free_training_smoke(tmp_path: Path) -> None:
    settings = load_settings(EXPERIMENT / "configs" / "bench2drive_smoke.yaml")
    settings.output_dir = tmp_path / "runs"
    result = run_smoke(settings)
    assert result["status"] == "passed"
    assert result["actor_updated"]
    assert (settings.output_dir / "metrics" / "smoke.json").is_file()
