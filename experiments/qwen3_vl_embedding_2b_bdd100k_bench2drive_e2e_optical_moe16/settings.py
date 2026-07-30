from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .io_utils import write_json


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]


def _nested(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _read_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = raw.pop("base_config", None)
    if parent is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_yaml(parent_path, seen), raw)


def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def _run_path(value: str | Path | None) -> Path | None:
    """Resolve generated run artifacts inside this experiment.

    Dataset and reusable cache paths remain repository-relative. A conventional
    ``runs/...`` setting, however, belongs to this experiment rather than the
    repository root. Absolute paths remain supported.
    """
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "runs":
        return (EXPERIMENT_DIR / path).resolve()
    return (PROJECT_DIR / path).resolve()


class Settings:
    """Resolved settings, including the flat names consumed by Optical MoE16."""

    def __init__(self, raw: dict[str, Any], config_path: Path) -> None:
        self.raw = copy.deepcopy(raw)
        self.config_path = config_path.resolve()
        self.output_dir = _required_path(
            _run_path(_nested(raw, "paths.output_dir"))
        )
        self.artifact_cache_dir = _required_path(
            _path(_nested(raw, "paths.artifact_cache_dir"))
        )
        self.bdd_root = _required_path(_path(_nested(raw, "paths.bdd100k_root")))
        self.bench2drive_root = _required_path(
            _path(_nested(raw, "paths.bench2drive_root"))
        )
        self.pretrained_backbone_checkpoint = _required_path(
            _run_path(_nested(raw, "paths.pretrained_backbone_checkpoint"))
        )

        self.model_id = str(
            _nested(raw, "qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")
        )
        self.cache_dir = _path(_nested(raw, "qwen.cache_dir"))
        self.local_files_only = bool(_nested(raw, "qwen.local_files_only", False))
        self.processor_min_pixels = int(
            _nested(raw, "qwen.processor_min_pixels", 50_176)
        )
        self.processor_max_pixels = int(
            _nested(raw, "qwen.processor_max_pixels", 50_176)
        )
        self.dtype = str(_nested(raw, "qwen.dtype", "bfloat16"))
        self.attn_implementation = str(
            _nested(raw, "qwen.attn_implementation", "sdpa")
        )
        self.device = str(_nested(raw, "qwen.device", "cuda"))
        self.image_size = int(_nested(raw, "data.image_size", 224))
        self.random_seed = int(_nested(raw, "training.seed", 42))
        self.num_workers = int(_nested(raw, "training.num_workers", 8))
        self.amp_enabled = bool(_nested(raw, "training.amp_enabled", True))
        self.log_interval_batches = int(
            _nested(raw, "training.log_interval_batches", 250)
        )

        self.bdd_train_split = str(_nested(raw, "bdd100k.train_split", "train"))
        self.bdd_test_split = str(_nested(raw, "bdd100k.test_split", "val"))
        self.bdd_image_dir = str(_nested(raw, "bdd100k.image_dir", "images/100k"))
        self.bdd_annotation_jsons = tuple(
            str(item) for item in _nested(raw, "bdd100k.annotation_jsons", [])
        )
        self.bdd_drivable_mask_dir = str(
            _nested(raw, "bdd100k.drivable_mask_dir", "")
        )
        self.bdd_lane_mask_dir = str(_nested(raw, "bdd100k.lane_mask_dir", ""))
        self.bdd_require_auxiliary_labels = bool(
            _nested(raw, "bdd100k.require_auxiliary_labels", True)
        )
        self.bdd_train_limit = _optional_int(
            _nested(raw, "bdd100k.train_limit", None)
        )
        self.bdd_test_limit = _optional_int(_nested(raw, "bdd100k.test_limit", None))
        self.bdd_lane_width = int(_nested(raw, "bdd100k.lane_width", 5))
        self.road_participant_categories = tuple(
            str(value).lower()
            for value in _nested(
                raw,
                "bdd100k.road_participant_categories",
                ["car", "bus", "truck", "person", "rider", "bike", "motor"],
            )
        )

        self.pca_rank = int(_nested(raw, "pca.rank", 224))
        self.pca_calibration_images = int(
            _nested(raw, "pca.calibration_images", 2_000)
        )
        self.pca_tokens_per_image = int(
            _nested(raw, "pca.tokens_per_image", 64)
        )
        self.pca_max_tokens = int(_nested(raw, "pca.max_tokens", 100_000))
        self.pca_oversample = int(_nested(raw, "pca.oversample", 32))
        self.pca_niter = int(_nested(raw, "pca.niter", 4))
        self.pca_device = str(_nested(raw, "pca.device", "cpu"))

        self.pretrain_batch_size = int(
            _nested(raw, "pretrain.batch_size", 2)
        )
        self.pretrain_epochs = int(_nested(raw, "pretrain.epochs", 30))
        self.pretrain_learning_rate = float(
            _nested(raw, "pretrain.learning_rate", 2e-3)
        )
        self.pretrain_weight_decay = float(
            _nested(raw, "pretrain.weight_decay", 0.0)
        )
        self.lambda_feature_cosine = float(
            _nested(raw, "pretrain.loss.feature_cosine", 1.0)
        )
        self.lambda_feature_smooth_l1 = float(
            _nested(raw, "pretrain.loss.feature_smooth_l1", 0.5)
        )
        self.lambda_drivable = float(
            _nested(raw, "pretrain.loss.drivable", 0.25)
        )
        self.lambda_lane = float(_nested(raw, "pretrain.loss.lane", 0.25))
        self.lambda_participant = float(
            _nested(raw, "pretrain.loss.road_participant", 0.15)
        )
        self.lambda_router_balance = float(
            _nested(raw, "pretrain.loss.router_balance", 0.03)
        )
        self.lambda_router_importance = float(
            _nested(raw, "pretrain.loss.router_importance", 0.0)
        )

        self.bench_frame_stride = int(
            _nested(raw, "bench2drive.frame_stride", 5)
        )
        self.bench_train_fraction = float(
            _nested(raw, "bench2drive.train_fraction", 0.95)
        )
        self.bench_train_limit = _optional_int(
            _nested(raw, "bench2drive.train_limit", None)
        )
        self.bench_test_limit = _optional_int(
            _nested(
                raw,
                "bench2drive.holdout_limit",
                _nested(raw, "bench2drive.test_limit", None),
            )
        )
        self.num_commands = int(_nested(raw, "bench2drive.num_commands", 6))
        self.speed_normalization_mps = float(
            _nested(raw, "bench2drive.speed_normalization_mps", 12.0)
        )
        self.target_point_clip_m = float(
            _nested(raw, "bench2drive.target_point_clip_m", 50.0)
        )

        self.bc_batch_size = int(_nested(raw, "behavior_cloning.batch_size", 4))
        self.bc_stage1_epochs = int(
            _nested(raw, "behavior_cloning.stage1_epochs", 20)
        )
        self.bc_stage2_epochs = int(
            _nested(raw, "behavior_cloning.stage2_epochs", 30)
        )
        self.bc_actor_learning_rate = float(
            _nested(raw, "behavior_cloning.actor_learning_rate", 1e-3)
        )
        self.bc_linear_learning_rate = float(
            _nested(raw, "behavior_cloning.linear_learning_rate", 2e-4)
        )
        self.bc_optical_learning_rate = float(
            _nested(raw, "behavior_cloning.optical_learning_rate", 2e-5)
        )
        self.bc_weight_decay = float(
            _nested(raw, "behavior_cloning.weight_decay", 0.0)
        )
        self.actor_hidden_dims = tuple(
            int(value)
            for value in _nested(
                raw, "behavior_cloning.actor_hidden_dims", [256, 128]
            )
        )
        self.bc_steer_weight = float(
            _nested(raw, "behavior_cloning.loss.steer", 1.0)
        )
        self.bc_throttle_weight = float(
            _nested(raw, "behavior_cloning.loss.throttle", 1.0)
        )
        self.bc_brake_weight = float(
            _nested(raw, "behavior_cloning.loss.brake", 1.0)
        )
        self.bc_exclusion_weight = float(
            _nested(raw, "behavior_cloning.loss.throttle_brake_exclusion", 0.1)
        )
        self.bc_router_balance_weight = float(
            _nested(raw, "behavior_cloning.loss.router_balance", 0.03)
        )
        self.bc_router_importance_weight = float(
            _nested(raw, "behavior_cloning.loss.router_importance", 0.0)
        )

        self.sac_env_factory = str(_nested(raw, "sac.env_factory", "")).strip()
        self.sac_total_steps = int(_nested(raw, "sac.total_steps", 100_000))
        self.sac_random_steps = int(_nested(raw, "sac.random_steps", 1_000))
        self.sac_batch_size = int(_nested(raw, "sac.batch_size", 64))
        self.sac_replay_capacity = int(
            _nested(raw, "sac.replay_capacity", 100_000)
        )
        self.sac_gamma = float(_nested(raw, "sac.gamma", 0.99))
        self.sac_tau = float(_nested(raw, "sac.tau", 0.005))
        self.sac_learning_rate = float(_nested(raw, "sac.learning_rate", 3e-4))
        self.sac_initial_alpha = float(_nested(raw, "sac.initial_alpha", 0.2))
        self.sac_autotune_alpha = bool(_nested(raw, "sac.autotune_alpha", True))
        self.sac_freeze_backbone_steps = int(
            _nested(raw, "sac.unfreeze.freeze_backbone_steps", 100_000)
        )
        self.sac_unfreeze_linear_step = _optional_int(
            _nested(raw, "sac.unfreeze.linear_step", None)
        )
        self.sac_unfreeze_phase_step = _optional_int(
            _nested(raw, "sac.unfreeze.phase_step", None)
        )
        self.sac_store_images = bool(_nested(raw, "sac.replay_store_images", False))
        reward = _nested(raw, "sac.reward", {})
        self.reward_weights = {
            "route_progress": float(reward.get("route_progress", 1.0)),
            "target_speed": float(reward.get("target_speed", 0.25)),
            "lane_keep": float(reward.get("lane_keep", 0.2)),
            "collision": float(reward.get("collision", 5.0)),
            "offroad": float(reward.get("offroad", 2.0)),
            "red_light": float(reward.get("red_light", 2.0)),
            "control_smoothness": float(reward.get("control_smoothness", 0.05)),
        }
        self.reward_speed_scale = float(
            _nested(raw, "sac.reward.target_speed_scale_mps", 3.0)
        )
        self.reward_lane_scale = float(
            _nested(raw, "sac.reward.lane_offset_scale_m", 2.0)
        )

        # Flat interface expected by the already-validated MoE16 core.
        self.input_adapter_dim = int(_nested(raw, "optical.input_adapter_dim", 224))
        self.max_visual_tokens = int(
            _nested(raw, "optical.max_visual_tokens", 224)
        )
        self.max_language_tokens = 224
        self.canvas_size = int(_nested(raw, "optical.geometry.canvas_size", 1026))
        self.active_size = int(_nested(raw, "optical.geometry.active_size", 986))
        self.expert_size = int(_nested(raw, "optical.geometry.expert_size", 224))
        self.expert_pitch = int(_nested(raw, "optical.geometry.expert_pitch", 254))
        self.num_experts = int(_nested(raw, "optical.geometry.num_experts", 16))
        self.expert_grid_rows = int(_nested(raw, "optical.geometry.grid_rows", 4))
        self.expert_grid_cols = int(_nested(raw, "optical.geometry.grid_cols", 4))
        self.expert_layers = int(
            _nested(raw, "optical.geometry.layers_per_expert", 1)
        )
        self.top_k = int(_nested(raw, "optical.router.top_k", 4))
        self.router_pool_size = int(_nested(raw, "optical.router.pool_size", 14))
        self.router_temperature = float(
            _nested(raw, "optical.router.temperature", 1.0)
        )
        self.router_input_layernorm_enabled = bool(
            _nested(raw, "optical.router.input_layernorm_enabled", True)
        )
        self.router_input_layernorm_eps = float(
            _nested(raw, "optical.router.input_layernorm_eps", 1e-5)
        )
        self.amplitude_slm_weight_domain = str(
            _nested(raw, "optical.router.amplitude_weight_domain", "amplitude")
        )
        self.amplitude_slm_input_normalization = str(
            _nested(raw, "optical.router.input_normalization", "none")
        )
        self.amplitude_phase_relay = "ideal_4f_coplanar"
        self.wavelength_nm = float(
            _nested(raw, "optical.physics.wavelength_nm", 532.0)
        )
        self.pixel_pitch_um = float(
            _nested(raw, "optical.physics.pixel_pitch_um", 8.0)
        )
        self.expert_interlayer_distance_m = float(
            _nested(raw, "optical.physics.expert_to_global_distance_m", 0.10)
        )
        self.last_expert_to_global_distance_m = self.expert_interlayer_distance_m
        self.global_to_detector_distance_m = float(
            _nested(raw, "optical.physics.global_to_detector_distance_m", 0.10)
        )
        self.phase_parameterization = str(
            _nested(raw, "optical.phase.parameterization", "sigmoid")
        )
        self.phase_init = str(_nested(raw, "optical.phase.init", "zeros"))
        self.phase_init_std = float(_nested(raw, "optical.phase.init_std", 0.02))
        self.k_space_constraint_enabled = bool(
            _nested(raw, "optical.k_space.enabled", False)
        )
        self.theta_max_deg = float(_nested(raw, "optical.k_space.theta_max_deg", 1.0))
        self.interlayer_enabled = bool(_nested(raw, "optical.oeo.enabled", True))
        self.interlayer_per_expert_enabled = bool(
            _nested(raw, "optical.oeo.per_expert_enabled", True)
        )
        self.interlayer_elementwise_affine = bool(
            _nested(raw, "optical.oeo.elementwise_affine", False)
        )
        self.interlayer_hard_route_mask = bool(
            _nested(raw, "optical.oeo.hard_route_mask", True)
        )
        self.interlayer_reapply_routing_weights = bool(
            _nested(raw, "optical.oeo.reapply_routing_weights", True)
        )
        self.interlayer_layernorm_eps = float(
            _nested(raw, "optical.oeo.layernorm_eps", 1e-5)
        )
        self.interlayer_nonlinearity = str(
            _nested(raw, "optical.oeo.nonlinearity", "relu")
        )
        self.detector_output_size = int(
            _nested(raw, "optical.detector.output_size", 224)
        )
        self.detector_layernorm_eps = float(
            _nested(raw, "optical.detector.layernorm_eps", 1e-5)
        )
        self.detector_layernorm_affine = bool(
            _nested(raw, "optical.detector.layernorm_affine", False)
        )
        self.detector_layernorm_scope = str(
            _nested(raw, "optical.detector.layernorm_scope", "per_token")
        )
        self.detector_nonlinearity = str(
            _nested(raw, "optical.detector.nonlinearity", "relu")
        )
        self.phase_dropout_mode = str(
            _nested(raw, "optical.phase_dropout.mode", "none")
        )
        self.phase_dropout_p = float(_nested(raw, "optical.phase_dropout.p", 0.0))
        self.phase_dropout_block_size = int(
            _nested(raw, "optical.phase_dropout.block_size", 8)
        )
        self.phase_dropout_batch_shared = bool(
            _nested(raw, "optical.phase_dropout.batch_shared", True)
        )

        self.vision_hidden_size: int | None = None
        self.vision_depth: int | None = None
        self.validate()

    @property
    def pca_path(self) -> Path:
        return self.artifact_cache_dir / "pca" / "bdd_vision_pca_1024_to_224.pt"

    @property
    def bdd_index_path(self) -> Path:
        return self.artifact_cache_dir / "manifests" / "bdd100k_index.json"

    @property
    def bench_index_path(self) -> Path:
        return self.artifact_cache_dir / "manifests" / "bench2drive_index.json"

    def resolve_architecture(self, model: Any) -> None:
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.vision_depth = int(model.config.vision_config.depth)

    def validate(self) -> None:
        if self.image_size != 224:
            raise ValueError("This optical experiment fixes input images at 224x224")
        if self.input_adapter_dim != 224 or self.detector_output_size != 224:
            raise ValueError("Optical latent and CCD readout must both be 224")
        if (
            self.num_experts,
            self.expert_grid_rows,
            self.expert_grid_cols,
            self.top_k,
        ) != (16, 4, 4, 4):
            raise ValueError("Optical topology is fixed to MoE16 (4x4), Top-4")
        if self.expert_layers != 1:
            raise ValueError("The requested driving backbone has one expert phase layer")
        if (self.canvas_size, self.active_size, self.expert_size) != (
            1026,
            986,
            224,
        ):
            raise ValueError("Validated geometry is canvas=1026, active=986, expert=224")
        if self.pca_rank != 224:
            raise ValueError("Teacher spatial PCA target is fixed to 224 dimensions")
        if not 0 < self.bench_train_fraction < 1:
            raise ValueError("bench2drive.train_fraction must be in (0,1)")
        if self.bench_frame_stride <= 0:
            raise ValueError("bench2drive.frame_stride must be positive")
        if self.num_commands != 6:
            raise ValueError("Bench2Drive uses six high-level navigation commands")
        if self.sac_unfreeze_phase_step is not None and not self.sac_store_images:
            raise ValueError(
                "SAC phase unfreezing needs replay_store_images=true so gradients "
                "can be recomputed through the Optical Backbone"
            )
        if self.sac_unfreeze_linear_step is not None and not self.sac_store_images:
            raise ValueError(
                "SAC Linear unfreezing needs replay_store_images=true so gradients "
                "can be recomputed through the Optical Backbone"
            )
        for name, step in (
            ("linear_step", self.sac_unfreeze_linear_step),
            ("phase_step", self.sac_unfreeze_phase_step),
        ):
            if step is not None and step < self.sac_freeze_backbone_steps:
                raise ValueError(
                    f"sac.unfreeze.{name}={step} precedes the mandatory frozen "
                    f"warm-up of {self.sac_freeze_backbone_steps} steps"
                )
        for name, value in self.reward_weights.items():
            if value < 0:
                raise ValueError(f"SAC reward weight {name} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = copy.deepcopy(self.raw)
        value["resolved_paths"] = {
            "project_dir": str(PROJECT_DIR),
            "output_dir": str(self.output_dir),
            "artifact_cache_dir": str(self.artifact_cache_dir),
            "bdd100k_root": str(self.bdd_root),
            "bench2drive_root": str(self.bench2drive_root),
            "pretrained_backbone_checkpoint": str(
                self.pretrained_backbone_checkpoint
            ),
        }
        value["resolved_architecture"] = {
            "vision_hidden_size": self.vision_hidden_size,
            "vision_depth": self.vision_depth,
        }
        return value


def _required_path(value: Path | None) -> Path:
    if value is None:
        raise ValueError("A required path is missing from configuration")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    return Settings(_read_yaml(config_path), config_path)


def save_resolved_config(settings: Settings) -> None:
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
