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
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    parent = raw.pop("base_config", None)
    if parent is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), raw)


def _project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


class Settings:
    """Resolved configuration with the flat attributes required by Optical MoE16."""

    def __init__(self, raw: dict[str, Any], config_path: Path) -> None:
        self.raw = copy.deepcopy(raw)
        self.config_path = config_path.resolve()

        # Dataset and immutable split.
        self.dataset_root = _project_path(_nested(raw, "dataset.root", "data/abo"))
        self.output_dir = _project_path(
            _nested(
                raw,
                "output_dir",
                "runs/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16",
            )
        )
        self.artifact_cache_dir = _project_path(
            _nested(
                raw,
                "artifact_cache_dir",
                "cache/qwen3_vl_embedding_2b_abo_product_retrieval_optical_moe16",
            )
        )
        self.download = bool(_nested(raw, "dataset.download", True))
        self.listings_url = str(
            _nested(
                raw,
                "dataset.listings_url",
                "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar",
            )
        )
        self.images_url = str(
            _nested(
                raw,
                "dataset.images_url",
                "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar",
            )
        )
        self.stage1_item_count = int(_nested(raw, "dataset.stage1_item_count", 20_000))
        self.stage1_target_image_count = int(
            _nested(raw, "dataset.stage1_target_image_count", 60_000)
        )
        self.stage1_min_images_per_item = int(
            _nested(raw, "dataset.stage1_min_images_per_item", 2)
        )
        self.stage1_max_images_per_item = int(
            _nested(raw, "dataset.stage1_max_images_per_item", 6)
        )
        self.stage2_item_count = int(_nested(raw, "dataset.stage2_item_count", 3_000))
        self.stage2_min_images_per_item = int(
            _nested(raw, "dataset.stage2_min_images_per_item", 4)
        )
        self.stage2_max_images_per_item = int(
            _nested(raw, "dataset.stage2_max_images_per_item", 10)
        )
        self.stage2_train_fraction = float(
            _nested(raw, "dataset.stage2_train_fraction", 0.60)
        )
        self.stage2_gallery_fraction = float(
            _nested(raw, "dataset.stage2_gallery_fraction", 0.20)
        )
        self.stage2_query_fraction = float(
            _nested(raw, "dataset.stage2_query_fraction", 0.20)
        )
        self.preferred_product_types = tuple(
            str(value)
            for value in _nested(raw, "dataset.preferred_product_types", [])
        )
        self.stage2_product_type_count = int(
            _nested(raw, "dataset.stage2_product_type_count", 30)
        )
        self.quality_scan_enabled = bool(
            _nested(raw, "dataset.quality_scan_enabled", True)
        )
        self.quality_candidate_multiplier = int(
            _nested(raw, "dataset.quality_candidate_multiplier", 4)
        )
        self.minimum_original_short_side = int(
            _nested(raw, "dataset.minimum_original_short_side", 224)
        )
        self.rebuild_manifest = bool(_nested(raw, "dataset.rebuild_manifest", False))

        # Qwen.
        self.model_id = str(
            _nested(raw, "qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")
        )
        self.cache_dir = _project_path(_nested(raw, "qwen.cache_dir", None))
        self.local_files_only = bool(_nested(raw, "qwen.local_files_only", False))
        self.instruction = str(
            _nested(
                raw,
                "qwen.instruction",
                "Represent this product image for image-to-image product retrieval.",
            )
        )
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

        # Retrieval and batching.
        self.image_size = int(_nested(raw, "retrieval.image_size", 224))
        self.embedding_dim = int(_nested(raw, "retrieval.embedding_dim", 224))
        self.gallery_aggregation = str(
            _nested(raw, "retrieval.gallery_aggregation", "mean_prototype")
        )
        self.teacher_batch_size = int(_nested(raw, "batching.teacher_batch_size", 4))
        self.stage1_batch_size = int(_nested(raw, "batching.stage1_batch_size", 8))
        self.stage1_pk_items = int(_nested(raw, "batching.stage1_pk_items", 4))
        self.stage1_pk_images = int(_nested(raw, "batching.stage1_pk_images", 2))
        self.stage2_batch_size = int(_nested(raw, "batching.stage2_batch_size", 8))
        self.stage2_pk_items = int(_nested(raw, "batching.stage2_pk_items", 4))
        self.stage2_pk_images = int(_nested(raw, "batching.stage2_pk_images", 2))
        self.inference_batch_size = int(
            _nested(raw, "batching.inference_batch_size", 8)
        )
        self.num_workers = int(_nested(raw, "batching.num_workers", 6))

        # Training.
        self.stage1_epochs = int(_nested(raw, "training.stage1_epochs", 50))
        self.stage2_epochs = int(_nested(raw, "training.stage2_epochs", 50))
        self.stage1_learning_rate = float(
            _nested(raw, "training.stage1_learning_rate", 0.002)
        )
        self.stage2_learning_rate = float(
            _nested(raw, "training.stage2_learning_rate", 0.001)
        )
        self.weight_decay = float(_nested(raw, "training.weight_decay", 0.0))
        self.random_seed = int(_nested(raw, "training.random_seed", 42))
        self.amp_enabled = bool(_nested(raw, "training.amp_enabled", True))
        self.log_interval_batches = int(
            _nested(raw, "training.log_interval_batches", 500)
        )
        self.lambda_stage1_kd = float(_nested(raw, "loss.stage1_kd", 1.0))
        self.lambda_stage1_supcon = float(_nested(raw, "loss.stage1_supcon", 1.0))
        self.lambda_stage2_supcon = float(_nested(raw, "loss.stage2_supcon", 1.0))
        self.lambda_stage2_id = float(_nested(raw, "loss.stage2_id", 0.5))
        self.lambda_stage2_kd = float(_nested(raw, "loss.stage2_kd", 0.2))
        self.lambda_router_balance = float(_nested(raw, "loss.router_balance", 0.03))
        self.lambda_router_importance = float(
            _nested(raw, "loss.router_importance", 0.0)
        )
        self.temperature = float(_nested(raw, "loss.temperature", 0.07))

        # Optical core. Names intentionally match the validated reusable core.
        self.input_adapter_dim = int(_nested(raw, "optical.input_adapter_dim", 224))
        self.max_visual_tokens = int(_nested(raw, "optical.max_visual_tokens", 224))
        self.max_language_tokens = int(
            _nested(raw, "optical.max_language_tokens", 224)
        )
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
            _nested(raw, "optical.router.input_layernorm_eps", 1.0e-5)
        )
        self.amplitude_slm_weight_domain = str(
            _nested(raw, "optical.router.amplitude_weight_domain", "amplitude")
        )
        self.amplitude_slm_input_normalization = str(
            _nested(raw, "optical.router.input_normalization", "none")
        )
        self.amplitude_phase_relay = "ideal_4f_coplanar"
        self.wavelength_nm = float(_nested(raw, "optical.physics.wavelength_nm", 532.0))
        self.pixel_pitch_um = float(_nested(raw, "optical.physics.pixel_pitch_um", 8.0))
        self.expert_interlayer_distance_m = float(
            _nested(raw, "optical.physics.inter_layer_distance_m", 0.10)
        )
        self.last_expert_to_global_distance_m = float(
            _nested(raw, "optical.physics.last_expert_to_global_distance_m", 0.10)
        )
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
            _nested(raw, "optical.oeo.layernorm_eps", 1.0e-5)
        )
        self.interlayer_nonlinearity = str(
            _nested(raw, "optical.oeo.nonlinearity", "relu")
        )
        self.detector_output_size = int(
            _nested(raw, "optical.detector.output_size", 224)
        )
        self.detector_layernorm_eps = float(
            _nested(raw, "optical.detector.layernorm_eps", 1.0e-5)
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
    def manifest_csv(self) -> Path:
        return self.artifact_cache_dir / "manifests" / "abo_fixed_split.csv"

    @property
    def manifest_metadata_json(self) -> Path:
        return self.artifact_cache_dir / "manifests" / "abo_fixed_split.json"

    @property
    def teacher_cache_path(self) -> Path:
        return self.artifact_cache_dir / "teacher" / "teacher_embeddings.pt"

    def resolve_architecture(self, model: Any) -> None:
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.vision_depth = int(model.config.vision_config.depth)

    def validate(self) -> None:
        if self.dataset_root is None or self.output_dir is None or self.artifact_cache_dir is None:
            raise ValueError("dataset_root, output_dir, and artifact_cache_dir are required")
        if self.embedding_dim != 224:
            raise ValueError("This experiment fixes the retrieval embedding to 224 dimensions")
        if self.image_size != 224:
            raise ValueError("The validated optical input size is fixed at 224x224")
        if self.stage1_min_images_per_item < 2:
            raise ValueError("Stage 1 needs at least two views per item")
        if self.stage2_min_images_per_item < 4:
            raise ValueError("Stage 2 needs at least four views per item")
        fractions = (
            self.stage2_train_fraction
            + self.stage2_gallery_fraction
            + self.stage2_query_fraction
        )
        if abs(fractions - 1.0) > 1.0e-8:
            raise ValueError("Stage-2 train/gallery/query fractions must sum to 1")
        if min(
            self.stage2_train_fraction,
            self.stage2_gallery_fraction,
            self.stage2_query_fraction,
        ) <= 0:
            raise ValueError("Every stage-2 split fraction must be positive")
        if self.stage1_target_image_count < 2 * self.stage1_item_count:
            raise ValueError("stage1_target_image_count must allow at least two images/item")
        if self.stage1_batch_size != self.stage1_pk_items * self.stage1_pk_images:
            raise ValueError("stage1_batch_size must equal stage1_pk_items*stage1_pk_images")
        if self.stage2_batch_size != self.stage2_pk_items * self.stage2_pk_images:
            raise ValueError("stage2_batch_size must equal stage2_pk_items*stage2_pk_images")
        if self.stage1_pk_images < 2 or self.stage2_pk_images < 2:
            raise ValueError("PK batches need at least two views per item")
        if self.input_adapter_dim != 224 or self.expert_size != 224:
            raise ValueError("Optical token/channel field must remain 224x224")
        if self.max_visual_tokens > self.expert_size:
            raise ValueError("max_visual_tokens cannot exceed expert_size")
        if (
            self.num_experts != 16
            or self.expert_grid_rows != 4
            or self.expert_grid_cols != 4
            or self.top_k != 4
        ):
            raise ValueError("This experiment fixes MoE16 on a 4x4 grid with Top-4 routing")
        if self.expert_layers != 1:
            raise ValueError("The requested ABO backbone has exactly one expert phase layer")
        if self.canvas_size != 1026 or self.active_size != 986:
            raise ValueError("The validated MoE16 geometry is 986 active / 1026 FFT canvas")
        if self.detector_output_size != 224:
            raise ValueError("CCD readout must be 224x224")
        if self.phase_dropout_mode != "none" or self.phase_dropout_p != 0:
            raise ValueError("Phase dropout is disabled in the initial ABO experiment")
        if self.temperature <= 0:
            raise ValueError("Contrastive temperature must be positive")
        for name in (
            "lambda_stage1_kd",
            "lambda_stage1_supcon",
            "lambda_stage2_supcon",
            "lambda_stage2_id",
            "lambda_stage2_kd",
            "lambda_router_balance",
            "lambda_router_importance",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def split_identity(self) -> dict[str, Any]:
        return {
            "version": "abo_fixed_split_v1",
            "seed": self.random_seed,
            "stage1_item_count": self.stage1_item_count,
            "stage1_target_image_count": self.stage1_target_image_count,
            "stage1_min_images_per_item": self.stage1_min_images_per_item,
            "stage1_max_images_per_item": self.stage1_max_images_per_item,
            "stage2_item_count": self.stage2_item_count,
            "stage2_min_images_per_item": self.stage2_min_images_per_item,
            "stage2_max_images_per_item": self.stage2_max_images_per_item,
            "stage2_fractions": [
                self.stage2_train_fraction,
                self.stage2_gallery_fraction,
                self.stage2_query_fraction,
            ],
            "preferred_product_types": list(self.preferred_product_types),
            "stage2_product_type_count": self.stage2_product_type_count,
            "minimum_original_short_side": self.minimum_original_short_side,
            "quality_scan_enabled": self.quality_scan_enabled,
            "quality_candidate_multiplier": self.quality_candidate_multiplier,
        }

    def to_dict(self) -> dict[str, Any]:
        value = copy.deepcopy(self.raw)
        value["resolved_paths"] = {
            "project_dir": str(PROJECT_DIR),
            "dataset_root": str(self.dataset_root),
            "output_dir": str(self.output_dir),
            "artifact_cache_dir": str(self.artifact_cache_dir),
        }
        value["resolved_architecture"] = {
            "vision_hidden_size": self.vision_hidden_size,
            "vision_depth": self.vision_depth,
        }
        return value


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    return Settings(_read_config(config_path), config_path)


def save_resolved_config(settings: Settings) -> None:
    write_json(settings.output_dir / "resolved_config.json", settings.to_dict())
