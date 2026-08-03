from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .io_utils import write_json


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config involving {path}")
    seen.add(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config root must be a mapping")
    parent = value.pop("base_config", None)
    if parent is None:
        return value
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), value)


def _get(raw: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = raw
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def _output_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else EXPERIMENT_DIR / path).resolve()


class Settings:
    def __init__(self, raw: dict[str, Any], config_path: Path) -> None:
        self.raw = copy.deepcopy(raw)
        self.config_path = config_path.resolve()

        self.data_root = _project_path(_get(raw, "dataset.data_root", "data/SALICON"))
        self.download = bool(_get(raw, "dataset.download", True))
        self.train_images_url = str(
            _get(
                raw,
                "dataset.train_images_url",
                "https://s3.amazonaws.com/salicon-dataset/2015r1/train.zip",
            )
        )
        self.validation_images_url = str(
            _get(
                raw,
                "dataset.validation_images_url",
                "https://s3.amazonaws.com/salicon-dataset/2015r1/val.zip",
            )
        )
        self.train_annotations_url = str(
            _get(
                raw,
                "dataset.train_annotations_url",
                "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_train2014.json",
            )
        )
        self.validation_annotations_url = str(
            _get(
                raw,
                "dataset.validation_annotations_url",
                "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_val2014.json",
            )
        )
        self.train_limit = _optional_int(_get(raw, "dataset.train_limit"))
        self.validation_limit = _optional_int(
            _get(raw, "dataset.validation_limit")
        )
        self.materialize_density_maps = bool(
            _get(raw, "dataset.materialize_density_maps", True)
        )
        self.density_sigma_px = float(_get(raw, "dataset.density_sigma_px", 19.0))

        self.output_dir = _output_path(
            _get(raw, "output_dir", "runs/salicon_vision_optical_saliency")
        )
        self.artifact_cache_dir = _project_path(
            _get(
                raw,
                "artifact_cache_dir",
                "cache/qwen3_vl_embedding_2b_salicon_vision_optical_saliency",
            )
        )
        self.model_id = str(
            _get(raw, "qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")
        )
        self.cache_dir = _project_path(_get(raw, "qwen.cache_dir"))
        self.local_files_only = bool(_get(raw, "qwen.local_files_only", False))
        self.processor_min_pixels = int(
            _get(raw, "qwen.processor_min_pixels", 50_176)
        )
        self.processor_max_pixels = int(
            _get(raw, "qwen.processor_max_pixels", 50_176)
        )
        self.dtype = str(_get(raw, "qwen.dtype", "bfloat16"))
        self.attn_implementation = str(
            _get(raw, "qwen.attn_implementation", "sdpa")
        )
        self.device = str(_get(raw, "qwen.device", "cuda"))
        self.image_size = int(_get(raw, "data.image_size", 224))

        self.teacher_batch_size = int(_get(raw, "batching.teacher_batch_size", 16))
        self.student_batch_size = int(_get(raw, "batching.student_batch_size", 8))
        self.inference_batch_size = int(
            _get(raw, "batching.inference_batch_size", 8)
        )
        self.num_workers = int(_get(raw, "batching.num_workers", 6))

        self.teacher_epochs = int(_get(raw, "training.teacher_epochs", 30))
        self.student_epochs = int(_get(raw, "training.student_epochs", 60))
        self.teacher_learning_rate = float(
            _get(raw, "training.teacher_learning_rate", 1.0e-3)
        )
        self.student_learning_rate = float(
            _get(raw, "training.student_learning_rate", 1.0e-3)
        )
        self.phase_learning_rate = float(
            _get(raw, "training.phase_learning_rate", 1.0e-3)
        )
        self.router_learning_rate = float(
            _get(raw, "training.router_learning_rate", 5.0e-4)
        )
        self.weight_decay = float(_get(raw, "training.weight_decay", 0.0))
        self.random_seed = int(_get(raw, "training.random_seed", 42))
        self.amp_enabled = bool(_get(raw, "training.amp_enabled", True))
        self.log_interval_batches = int(
            _get(raw, "training.log_interval_batches", 100)
        )
        self.checkpoint_metric = str(
            _get(raw, "training.checkpoint_metric", "validation_cc")
        )
        self.gradient_clip_norm = float(
            _get(raw, "training.gradient_clip_norm", 1.0)
        )

        self.kl_weight = float(_get(raw, "loss.kl_weight", 1.0))
        self.cc_weight = float(_get(raw, "loss.cc_weight", 0.5))
        self.sim_weight = float(_get(raw, "loss.sim_weight", 0.25))
        self.nss_weight = float(_get(raw, "loss.nss_weight", 0.1))
        self.router_balance_weight = float(
            _get(raw, "loss.router_balance_weight", 0.03)
        )
        self.router_importance_weight = float(
            _get(raw, "loss.router_importance_weight", 0.005)
        )
        self.phase_dc_weight = float(_get(raw, "loss.phase_dc_weight", 0.0))
        self.map_kd_weight = float(_get(raw, "loss.map_kd_weight", 0.0))
        self.map_kd_temperature = float(
            _get(raw, "loss.map_kd_temperature", 1.0)
        )
        self.teacher_checkpoint = _project_path(
            _get(raw, "map_kd.teacher_checkpoint")
        )

        self.augmentation_enabled = bool(
            _get(raw, "augmentation.enabled", True)
        )
        self.crop_scale_min = float(
            _get(raw, "augmentation.crop_scale_min", 0.90)
        )
        self.horizontal_flip_probability = float(
            _get(raw, "augmentation.horizontal_flip_probability", 0.5)
        )
        self.brightness_jitter = float(
            _get(raw, "augmentation.brightness_jitter", 0.10)
        )
        self.contrast_jitter = float(
            _get(raw, "augmentation.contrast_jitter", 0.10)
        )

        self.segmentation_projection_dim = int(
            _get(raw, "saliency_head.projection_dim", 128)
        )
        self.segmentation_channels = tuple(
            int(value)
            for value in _get(raw, "saliency_head.decoder_channels", [64, 32, 16])
        )
        self.segmentation_groupnorm_groups = int(
            _get(raw, "saliency_head.groupnorm_groups", 8)
        )
        self.student_segmentation_refinement_enabled = False
        self.student_segmentation_progressive_refinement_enabled = False
        self.student_detector_residual_enabled = False
        self.student_detector_residual_source = "nonnegative_input_field"
        self.student_detector_identity_scale_init = 1.0
        self.student_detector_input_scale_init = 0.1
        self.student_detector_identity_scale_trainable = False
        self.student_detector_input_scale_trainable = True

        self.input_adapter_dim = int(
            _get(raw, "optical.input_adapter_dim", 224)
        )
        self.max_visual_tokens = int(
            _get(raw, "optical.max_visual_tokens", 224)
        )
        self.max_language_tokens = 224
        self.vision_tap_stages = tuple(
            int(value) for value in _get(raw, "optical.vision_tap_stages", [1])
        )
        self.canvas_size = int(
            _get(raw, "optical.geometry.canvas_size", 1026)
        )
        self.active_size = int(
            _get(raw, "optical.geometry.active_size", 986)
        )
        self.expert_size = int(
            _get(raw, "optical.geometry.expert_size", 224)
        )
        self.expert_pitch = int(
            _get(raw, "optical.geometry.expert_pitch", 254)
        )
        self.num_experts = int(
            _get(raw, "optical.geometry.num_experts", 16)
        )
        self.expert_grid_rows = int(
            _get(raw, "optical.geometry.grid_rows", 4)
        )
        self.expert_grid_cols = int(
            _get(raw, "optical.geometry.grid_cols", 4)
        )
        self.expert_layers = int(
            _get(raw, "optical.geometry.layers_per_expert", 1)
        )
        self.top_k = int(_get(raw, "optical.router.top_k", 4))
        self.router_pool_size = int(_get(raw, "optical.router.pool_size", 14))
        self.router_temperature = float(
            _get(raw, "optical.router.temperature", 1.0)
        )
        self.router_input_layernorm_enabled = bool(
            _get(raw, "optical.router.input_layernorm_enabled", True)
        )
        self.router_input_layernorm_eps = float(
            _get(raw, "optical.router.input_layernorm_eps", 1.0e-5)
        )
        self.amplitude_slm_weight_domain = str(
            _get(raw, "optical.router.amplitude_weight_domain", "amplitude")
        )
        self.amplitude_slm_input_normalization = str(
            _get(raw, "optical.router.input_normalization", "none")
        )
        self.amplitude_phase_relay = "ideal_4f_identity"
        self.wavelength_nm = float(
            _get(raw, "optical.physics.wavelength_nm", 532.0)
        )
        self.pixel_pitch_um = float(
            _get(raw, "optical.physics.pixel_pitch_um", 8.0)
        )
        self.expert_interlayer_distance_m = float(
            _get(raw, "optical.physics.inter_layer_distance_m", 0.10)
        )
        self.phase_parameterization = str(
            _get(raw, "optical.phase.parameterization", "sigmoid")
        )
        self.phase_init = str(_get(raw, "optical.phase.init", "zeros"))
        self.phase_init_std = float(_get(raw, "optical.phase.init_std", 0.02))
        self.k_space_constraint_enabled = bool(
            _get(raw, "optical.k_space.enabled", False)
        )
        self.theta_max_deg = float(
            _get(raw, "optical.k_space.theta_max_deg", 1.0)
        )
        self.interlayer_enabled = bool(_get(raw, "optical.oeo.enabled", True))
        self.interlayer_per_expert_enabled = bool(
            _get(raw, "optical.oeo.per_expert_enabled", True)
        )
        self.interlayer_elementwise_affine = bool(
            _get(raw, "optical.oeo.elementwise_affine", False)
        )
        self.interlayer_hard_route_mask = bool(
            _get(raw, "optical.oeo.hard_route_mask", True)
        )
        self.interlayer_reapply_routing_weights = bool(
            _get(raw, "optical.oeo.reapply_routing_weights", True)
        )
        self.interlayer_layernorm_eps = float(
            _get(raw, "optical.oeo.layernorm_eps", 1.0e-5)
        )
        self.interlayer_nonlinearity = str(
            _get(raw, "optical.oeo.nonlinearity", "relu")
        )
        self.detector_output_size = int(
            _get(raw, "optical.detector.output_size", 224)
        )
        self.detector_layernorm_eps = float(
            _get(raw, "optical.detector.layernorm_eps", 1.0e-5)
        )
        self.detector_layernorm_affine = bool(
            _get(raw, "optical.detector.layernorm_affine", False)
        )
        self.detector_layernorm_scope = str(
            _get(raw, "optical.detector.layernorm_scope", "per_token")
        )
        self.detector_nonlinearity = str(
            _get(raw, "optical.detector.nonlinearity", "relu")
        )
        self.phase_dropout_mode = "none"
        self.phase_dropout_p = 0.0
        self.phase_dropout_block_size = 8
        self.phase_dropout_batch_shared = True

        self.visualization_sample_count = int(
            _get(raw, "visualization.sample_count", 12)
        )
        self.visualization_optical_sample_count = int(
            _get(raw, "visualization.optical_sample_count", 4)
        )
        self.vision_hidden_size: int | None = None
        self.vision_depth: int | None = None

    def resolve_architecture(self, model: Any) -> None:
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.vision_depth = int(model.config.vision_config.depth)

    def validate(self) -> None:
        if self.data_root is None or self.artifact_cache_dir is None:
            raise ValueError("dataset.data_root and artifact_cache_dir are required")
        if self.image_size != 224:
            raise ValueError("This experiment fixes image size to 224")
        if self.checkpoint_metric not in {"validation_cc", "train_loss"}:
            raise ValueError("checkpoint_metric must be validation_cc or train_loss")
        if self.density_sigma_px <= 0:
            raise ValueError("density_sigma_px must be positive")
        if not 0 < self.crop_scale_min <= 1:
            raise ValueError("crop_scale_min must be in (0,1]")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        if self.map_kd_temperature <= 0:
            raise ValueError("map_kd_temperature must be positive")
        if self.map_kd_weight > 0 and self.augmentation_enabled:
            raise ValueError(
                "Mask KD cache is stored in the unaugmented image coordinate "
                "system. Set augmentation.enabled=false when map_kd_weight>0 "
                "so teacher/student maps remain spatially aligned."
            )
        if min(
            self.kl_weight,
            self.cc_weight,
            self.sim_weight,
            self.nss_weight,
            self.router_balance_weight,
            self.router_importance_weight,
            self.phase_dc_weight,
            self.map_kd_weight,
        ) < 0:
            raise ValueError("Loss weights cannot be negative")
        geometry = (
            self.canvas_size,
            self.active_size,
            self.expert_size,
            self.expert_pitch,
            self.num_experts,
            self.expert_grid_rows,
            self.expert_grid_cols,
            self.expert_layers,
            self.top_k,
        )
        expected = (1026, 986, 224, 254, 16, 4, 4, 1, 4)
        if geometry != expected:
            raise ValueError(f"Optical geometry must be {expected}, got {geometry}")
        if self.max_visual_tokens > self.expert_size:
            raise ValueError("max_visual_tokens exceeds optical row count")
        if self.phase_dropout_mode != "none" or self.phase_dropout_p:
            raise ValueError("Phase dropout is disabled in this baseline")

    def to_dict(self) -> dict[str, Any]:
        value = copy.deepcopy(self.raw)
        value["resolved_paths"] = {
            "project_dir": str(PROJECT_DIR),
            "experiment_dir": str(EXPERIMENT_DIR),
            "data_root": str(self.data_root),
            "artifact_cache_dir": str(self.artifact_cache_dir),
            "output_dir": str(self.output_dir),
        }
        value["resolved_architecture"] = {
            "vision_hidden_size": self.vision_hidden_size,
            "vision_depth": self.vision_depth,
        }
        return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    settings = Settings(_read_config(config_path), config_path)
    settings.validate()
    return settings


def save_resolved_config(settings: Settings) -> None:
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
