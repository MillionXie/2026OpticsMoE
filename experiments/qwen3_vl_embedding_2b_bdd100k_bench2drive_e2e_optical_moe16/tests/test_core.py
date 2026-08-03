from __future__ import annotations

import gzip
import io
import json
import tarfile
import threading
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ..datasets_bdd100k import rasterize_targets
from ..datasets_bench2drive import normalized_driving_state, parse_annotation
from ..behavior_cloning import (
    CommandBalancedEpochSampler,
    _build_bc_optimizer,
    _require_finite,
    _set_policy_trainability,
)
from ..carla_bridge import RemoteCarlaEnv
from ..prepare_bench2drive_base import (
    archives_from_official_manifest,
    commit_staging,
    extract_front_rgb_and_annotations,
)
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


def test_scratch_config_skips_pretrained_checkpoint_and_trains_optics() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "bench2drive_base_scratch.yaml"
    )
    assert settings.bc_backbone_initialization == "scratch"
    assert settings.pretrained_backbone_checkpoint is None
    assert settings.bc_train_optical_from_stage1 is True
    assert settings.phase_init == "small_normal"
    assert settings.phase_init_std == pytest.approx(0.2)
    assert settings.k_space_constraint_enabled
    assert settings.theta_max_deg == pytest.approx(2.0)
    assert settings.bc_phase_dc_weight == pytest.approx(5.0)
    assert settings.bc_batch_size == 8
    assert settings.bc_samples_per_command_per_epoch == 2000
    assert settings.bc_stage1_optical_learning_rate == pytest.approx(2e-4)
    assert settings.bc_stage2_optical_learning_rate == pytest.approx(1e-4)
    assert settings.bc_stage1_phase_learning_rate == pytest.approx(2e-3)
    assert settings.bc_stage2_phase_learning_rate == pytest.approx(1e-3)
    assert settings.bc_stage1_router_learning_rate == pytest.approx(2e-4)
    assert settings.bc_gradient_clip_norm == pytest.approx(1.0)
    assert settings.bc_checkpoint_interval_batches == 250


def test_command_sampler_rotates_and_covers_every_record() -> None:
    records = [
        SimpleNamespace(command=command)
        for command, count in enumerate((9, 5, 2))
        for _ in range(count)
    ]
    sampler = CommandBalancedEpochSampler(records, max_per_command=3, seed=42)
    assert len(sampler) == 3 + 3 + 2
    assert sampler.full_coverage_epochs == 3
    seen: set[int] = set()
    for epoch in range(sampler.full_coverage_epochs):
        sampler.set_epoch(epoch)
        indexes = list(sampler)
        assert len(indexes) == len(set(indexes))
        seen.update(indexes)
    assert seen == set(range(len(records)))


def test_nonfinite_batch_guard_reports_sample_ids() -> None:
    with pytest.raises(RuntimeError, match="sample-a"):
        _require_finite(
            "loss",
            torch.tensor(float("nan")),
            {"sample_ids": ["sample-a"]},
        )


def test_joint_bc_trainability_keeps_native_qwen_frozen() -> None:
    class FakeCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_adapter = torch.nn.Linear(4, 4)
            self.router = torch.nn.Linear(4, 2)
            self.raw_phase = torch.nn.Parameter(torch.zeros(4, 4))

    actor = DrivingActor(hidden_dims=(8,))
    policy = SimpleNamespace(
        actor=actor,
        backbone=SimpleNamespace(
            visual=torch.nn.Linear(4, 4),
            core=FakeCore(),
            recombiner=torch.nn.Linear(4, 4),
        ),
    )
    settings = SimpleNamespace(
        bc_actor_learning_rate=1e-3,
        bc_linear_learning_rate=2e-4,
        bc_weight_decay=0.0,
    )
    _set_policy_trainability(policy, train_optical=True)
    assert not any(p.requires_grad for p in policy.backbone.visual.parameters())
    assert all(p.requires_grad for p in policy.backbone.core.parameters())
    assert all(p.requires_grad for p in policy.backbone.recombiner.parameters())
    assert not policy.actor.log_std.requires_grad
    optimizer = _build_bc_optimizer(
        policy,
        settings,
        train_optical=True,
        optical_learning_rate=2e-4,
        phase_learning_rate=2e-3,
        router_learning_rate=3e-4,
    )
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "actor",
        "ccd_recombiner",
        "optical_adapters_oeo",
        "optical_phases",
        "optical_router",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-3, 2e-4, 2e-4, 2e-3, 3e-4]
    )


def test_python311_remote_bridge_protocol_without_carla() -> None:
    listener = Listener(("127.0.0.1", 0), authkey=b"test-key")
    host, port = listener.address

    def server() -> None:
        connection = listener.accept()
        while True:
            request = connection.recv()
            if request["op"] == "hello":
                result = {"protocol_version": 1}
            elif request["op"] == "reset":
                result = {
                    "observation": {
                        "rgb_front": np.zeros((8, 8, 3), dtype=np.uint8),
                        "speed": 1.0,
                        "command": 3,
                        "target_point": [2.0, 0.0],
                    },
                    "info": {"seed": request["seed"]},
                }
            elif request["op"] == "step":
                result = {
                    "observation": {
                        "rgb_front": np.ones((8, 8, 3), dtype=np.uint8),
                        "speed": 2.0,
                        "command": 3,
                        "target_point": [3.0, 0.0],
                    },
                    "reward": 0.5,
                    "terminated": False,
                    "truncated": False,
                    "info": {"route_progress": 0.5},
                }
            else:
                connection.send({"ok": True, "result": {}})
                break
            connection.send({"ok": True, "result": result})
        connection.close()
        listener.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    settings = SimpleNamespace(
        carla_bridge_host=host,
        carla_bridge_port=port,
        carla_bridge_authkey="test-key",
        carla_bridge_timeout_seconds=2.0,
    )
    environment = RemoteCarlaEnv(settings)
    observation, info = environment.reset(seed=7)
    assert observation["rgb_front"].shape == (8, 8, 3)
    assert info["seed"] == 7
    _observation, reward, terminated, truncated, step_info = environment.step(
        [0.0, 0.5, 0.0]
    )
    assert reward == pytest.approx(0.5)
    assert not terminated and not truncated
    assert step_info["route_progress"] == pytest.approx(0.5)
    environment.close()
    thread.join(timeout=2.0)


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


def test_selective_bench2drive_archive_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "clip.tar.gz"
    members = {
        "clip/camera/rgb_front/00000.jpg": b"front",
        "clip/anno/00000.json.gz": b"annotation",
        "clip/camera/rgb_back/00000.jpg": b"back",
        "clip/lidar/00000.laz": b"lidar",
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    output = tmp_path / "base"
    counts = extract_front_rgb_and_annotations(archive_path, output)
    assert counts == {
        "selected_files": 2,
        "rgb_front_images": 1,
        "annotations": 1,
    }
    assert (output / "clip" / "camera" / "rgb_front" / "00000.jpg").read_bytes() == b"front"
    assert (output / "clip" / "anno" / "00000.json.gz").read_bytes() == b"annotation"
    assert not (output / "clip" / "camera" / "rgb_back" / "00000.jpg").exists()


def test_official_archive_manifest_avoids_hub_listing(tmp_path: Path) -> None:
    path = tmp_path / "base.json"
    path.write_text(
        json.dumps(
            {
                "b.tar.gz": {"sha256": "b", "size": 2},
                "README.md": {},
                "a.tar.gz": {"sha256": "a", "size": 1},
            }
        ),
        encoding="utf-8",
    )
    assert archives_from_official_manifest(path) == ["a.tar.gz", "b.tar.gz"]


def test_staged_archive_commit_hides_partial_routes(tmp_path: Path) -> None:
    output = tmp_path / "base"
    staging = output / ".extracting" / "archive"
    route = staging / "route"
    route.mkdir(parents=True)
    (route / "complete.txt").write_text("yes", encoding="utf-8")
    old = output / "route"
    old.mkdir(parents=True)
    (old / "partial.txt").write_text("old", encoding="utf-8")
    commit_staging(staging, output)
    assert (output / "route" / "complete.txt").read_text(encoding="utf-8") == "yes"
    assert not (output / "route" / "partial.txt").exists()
    assert not staging.exists()


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
    assert torch.count_nonzero(core.expert_layers[0].experts[0].raw_phase) > 0
    assert torch.count_nonzero(core.global_phase.phase.raw_phase) > 0
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
