from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
)

from .datasets_bdd100k import BDD100KSpatialDataset, build_bdd_records
from .datasets_bench2drive import build_bench2drive_splits
from .io_utils import write_json
from .modeling import (
    CCDLinearRecombiner,
    DrivingActor,
    RoadStructureAuxiliaryHead,
    TwinQCritic,
    raw_ccd_readout,
)
from .objectives import (
    auxiliary_structure_loss,
    behavior_cloning_loss,
    normalized_feature_loss,
    shaped_reward,
)
from .sac import train_sac


class SyntheticDrivingEnv:
    """Tiny deterministic API smoke environment; not a scientific benchmark."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.step_index = 0
        self.rng = np.random.default_rng(settings.random_seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_index = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        self.step_index += 1
        speed = float(6.0 + action[1] * 2.0 - action[2] * 3.0)
        observation = self._observation(speed=speed)
        done = self.step_index >= 8
        info = {
            "route_progress": 0.5,
            "speed": speed,
            "target_speed": 8.0,
            "lane_offset": float(action[0]) * 0.2,
            "collision": False,
            "offroad": False,
            "red_light": False,
        }
        return observation, 0.0, done, False, info

    def _observation(self, speed: float = 6.0) -> dict[str, Any]:
        return {
            "visual_feature": self.rng.normal(size=224).astype(np.float32),
            "speed": speed,
            "command": self.step_index % 6,
            "target_point": np.array([10.0, 0.5], dtype=np.float32),
        }


def make_synthetic_env(settings: Any) -> SyntheticDrivingEnv:
    return SyntheticDrivingEnv(settings)


def run_smoke(settings: Any) -> dict[str, Any]:
    """Run lightweight train/backward checks without Qwen, BDD100K or CARLA."""
    torch.manual_seed(settings.random_seed)
    _prepare_synthetic_data(settings)
    bdd_train = build_bdd_records(settings, settings.bdd_train_split)
    bdd_test = build_bdd_records(settings, settings.bdd_test_split)
    bench_train, bench_test = build_bench2drive_splits(settings)
    bdd_item = BDD100KSpatialDataset(
        bdd_train,
        settings.image_size,
        settings.bdd_lane_width,
        settings.road_participant_categories,
    )[0]
    if bdd_item["targets"].shape != (3, 224, 224):
        raise RuntimeError("Synthetic BDD loader did not produce [3,224,224] targets")
    optical_core = HomogeneousMoEOpticalCore(1024, 224, settings)
    optical_recombiner = CCDLinearRecombiner()
    optical_input = optical_core.encode_groups([torch.randn(4, 1024)])
    optical_field, optical_routing = optical_core.begin(optical_input)
    optical_field = optical_core.run_stage(
        0, optical_field, optical_routing
    )
    optical_field = optical_core.propagator(
        optical_core.global_phase(optical_field)
    )
    optical_ccd = raw_ccd_readout(optical_field, optical_core)
    optical_signed = optical_recombiner(optical_ccd)
    optical_loss = (
        optical_signed.square().mean()
        + 0.03 * optical_routing["balance_loss"]
    )
    optical_loss.backward()
    optical_gradients_ok = all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in (
            optical_core.expert_layers[0].experts[0].raw_phase.grad,
            optical_core.global_phase.phase.raw_phase.grad,
            optical_recombiner.linear.weight.grad,
        )
    )
    feature_student = torch.randn(12, 224, requires_grad=True)
    feature_teacher = torch.randn(12, 224)
    feature_loss, _ = normalized_feature_loss(
        feature_student,
        feature_teacher,
        cosine_weight=1.0,
        smooth_l1_weight=0.5,
    )
    auxiliary_head = RoadStructureAuxiliaryHead()
    spatial = torch.randn(2, 224, 14, 14, requires_grad=True)
    auxiliary_logits = auxiliary_head(spatial, output_size=32)
    auxiliary_target = torch.randint(0, 2, (2, 3, 32, 32)).float()
    auxiliary_loss, _ = auxiliary_structure_loss(
        auxiliary_logits,
        auxiliary_target,
        weights=(0.25, 0.25, 0.15),
    )
    (feature_loss + auxiliary_loss).backward()

    actor = DrivingActor(hidden_dims=(32, 16))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    state = torch.randn(8, 233)
    controls = torch.cat(
        [
            torch.rand(8, 1) * 2.0 - 1.0,
            torch.rand(8, 1),
            torch.rand(8, 1),
        ],
        dim=-1,
    )
    before = actor.forward_normalized(state).detach()
    for _ in range(3):
        actor_optimizer.zero_grad(set_to_none=True)
        predicted = actor.forward_normalized(state)
        bc_loss, _ = behavior_cloning_loss(
            predicted,
            controls,
            steer_weight=1.0,
            throttle_weight=1.0,
            brake_weight=1.0,
            exclusion_weight=0.1,
        )
        bc_loss.backward()
        actor_optimizer.step()
    after = actor.forward_normalized(state).detach()

    critic = TwinQCritic(hidden_dim=32)
    q1, q2 = critic(state, after)
    critic_loss = q1.square().mean() + q2.square().mean()
    critic_loss.backward()
    reward, components = shaped_reward(
        {
            "route_progress": 0.5,
            "speed": 7.0,
            "target_speed": 8.0,
            "lane_offset": 0.1,
            "collision": False,
            "offroad": False,
            "red_light": False,
        },
        [0.0, 0.5, 0.0],
        [0.1, 0.4, 0.0],
        settings,
    )
    synthetic_policy = _SyntheticPolicy()
    bc_checkpoint = settings.output_dir / "checkpoints" / "smoke_bc_policy.pt"
    bc_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": synthetic_policy.backbone.checkpoint_state(),
            "actor_state_dict": synthetic_policy.actor.state_dict(),
        },
        bc_checkpoint,
    )
    sac_result = train_sac(
        synthetic_policy,
        processor=None,
        settings=settings,
        device=torch.device("cpu"),
        env=make_synthetic_env(settings),
        bc_checkpoint=bc_checkpoint,
    )
    result = {
        "status": "passed",
        "scope": (
            "dependency-free tensor/backward smoke; real Qwen/BDD/Bench2Drive "
            "phases require their external model/data"
        ),
        "feature_loss": float(feature_loss.detach()),
        "real_optical_moe16": {
            "ccd_shape": list(optical_ccd.shape),
            "ccd_nonnegative": bool(torch.all(optical_ccd >= 0)),
            "selected_experts": int(optical_routing["selected_mask"].sum()),
            "loss": float(optical_loss.detach()),
            "phase_and_linear_gradients_finite": optical_gradients_ok,
        },
        "auxiliary_loss": float(auxiliary_loss.detach()),
        "bc_loss": float(bc_loss.detach()),
        "actor_updated": not torch.equal(before, after),
        "critic_gradients_finite": all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in critic.parameters()
        ),
        "reward": reward,
        "reward_components": components,
        "sac": sac_result,
        "synthetic_data": {
            "bdd_train": len(bdd_train),
            "bdd_test": len(bdd_test),
            "bench_train": len(bench_train),
            "bench_validation": len(bench_test),
            "route_disjoint": not (
                {row.route_id for row in bench_train}
                & {row.route_id for row in bench_test}
            ),
        },
    }
    if (
        not result["actor_updated"]
        or not result["critic_gradients_finite"]
        or not optical_gradients_ok
    ):
        raise RuntimeError(f"Smoke training failed: {result}")
    write_json(settings.output_dir / "metrics" / "smoke.json", result)
    return result


class _SyntheticBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(1, 1)
        self.recombiner = nn.Linear(1, 1)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "core_state_dict": self.core.state_dict(),
            "recombiner_state_dict": self.recombiner.state_dict(),
            "architecture": {"type": "synthetic_smoke_only"},
        }


class _SyntheticPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SyntheticBackbone()
        self.actor = DrivingActor(hidden_dims=(32, 16))


def _prepare_synthetic_data(settings: Any) -> None:
    """Create explicit `_smoke` fixtures; never used by a formal config."""
    labels_root = settings.bdd_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    for split, count in (
        (settings.bdd_train_split, 4),
        (settings.bdd_test_split, 2),
    ):
        image_root = settings.bdd_root / settings.bdd_image_dir / split
        image_root.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(count):
            name = f"{split}_{index:04d}.jpg"
            array = np.zeros((96, 160, 3), dtype=np.uint8)
            array[..., 1] = 70 + index * 10
            array[45:, :, 0] = 80
            Image.fromarray(array, mode="RGB").save(image_root / name)
            frames.append(
                {
                    "name": name,
                    "labels": [
                        {
                            "category": "drivable area",
                            "poly2d": [
                                {
                                    "vertices": [
                                        [0, 48],
                                        [159, 48],
                                        [159, 95],
                                        [0, 95],
                                    ],
                                    "closed": True,
                                }
                            ],
                        },
                        {
                            "category": "lane",
                            "poly2d": [
                                {
                                    "vertices": [[80, 48], [80, 95]],
                                    "closed": False,
                                }
                            ],
                        },
                        {
                            "category": "car",
                            "box2d": {"x1": 60, "y1": 35, "x2": 100, "y2": 65},
                        },
                    ],
                }
            )
        (labels_root / f"bdd100k_labels_images_{split}.json").write_text(
            json.dumps(frames), encoding="utf-8"
        )
    for route_index in range(2):
        route = settings.bench2drive_root / f"route_{route_index:02d}"
        rgb_root = route / "camera" / "rgb_front"
        anno_root = route / "anno"
        rgb_root.mkdir(parents=True, exist_ok=True)
        anno_root.mkdir(parents=True, exist_ok=True)
        for frame in range(4):
            frame_id = f"{frame:05d}"
            array = np.full(
                (64, 96, 3), 30 + route_index * 30 + frame, dtype=np.uint8
            )
            Image.fromarray(array, mode="RGB").save(rgb_root / f"{frame_id}.jpg")
            annotation = {
                "speed": 5.0 + frame,
                "steer": 0.05 * (frame - 1),
                "throttle": 0.4,
                "brake": 0.0,
                "x": float(frame),
                "y": float(route_index),
                "theta": 0.0,
                "x_target": float(frame + 10),
                "y_target": float(route_index),
                "next_command": 4,
            }
            with gzip.open(
                anno_root / f"{frame_id}.json.gz", "wt", encoding="utf-8"
            ) as handle:
                json.dump(annotation, handle)
