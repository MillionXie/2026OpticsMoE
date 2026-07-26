from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import EXPERIMENT_NAME, MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
PATH_FIELDS = {
    "manifest_path",
    "validation_manifest_path",
    "output_dir",
    "precompute_cache_dir",
    "cache_dir",
    "dataset_archive_dir",
    "dataset_image_dir",
}


@dataclass
class Settings:
    config_version: int = 1
    experiment_name: str = EXPERIMENT_NAME
    output_dir: Path = PROJECT_DIR / "runs" / EXPERIMENT_NAME
    seed: int = 42

    dataset: str = "cc3m_jsonl"
    manifest_path: Path = PROJECT_DIR.parent.parent / "data" / "cc3m" / "cc3m.jsonl"
    validation_manifest_path: Path | None = None
    validation_fraction: float = 0.01
    sample_limit: int | None = None
    calibration_sample_count: int = 2048
    caption_prompt_template: str = "{caption}"
    dataset_auto_prepare: bool = True
    dataset_source_repo_id: str = "chaocq/cc3m-wds"
    dataset_source_revision: str = "28cde01364d7e3b180681f8c448935edf47e2fd5"
    dataset_source_endpoint: str | None = None
    dataset_source_split: str = "train"
    dataset_archive_dir: Path = PROJECT_DIR.parent.parent / "data" / "cc3m" / "webdataset"
    dataset_image_dir: Path = PROJECT_DIR.parent.parent / "data" / "cc3m" / "images"
    dataset_download_workers: int = 4
    dataset_download_max_shards: int | None = None
    dataset_keep_archives: bool = False

    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int = 25600
    processor_max_pixels: int = 25600
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    text_depth: int | None = None
    text_hidden_size: int | None = None
    deepstack_visual_indexes: tuple[int, ...] | None = None
    language_tap_indexes: tuple[int, ...] | None = None

    latent_dim: int = 224
    max_visual_tokens: int = 224
    max_language_tokens: int = 224
    pca_vision_calibration_tokens: int = 200_000
    pca_language_calibration_tokens: int = 200_000
    pca_eigenvalue_eps: float = 1e-8
    pca_oracle_samples: int = 16
    pca_oracle_warn_cosine_below: float = 0.90

    feature_batch_size: int = 1
    student_batch_size: int = 1
    validation_batch_size: int = 1
    num_workers: int = 8
    cpu_threads: int = 4
    cpu_interop_threads: int = 1

    precompute_cache_dir: Path = PROJECT_DIR / "cache"
    cache_dtype: str = "float16"
    teacher_cache_shard_size: int = 128
    teacher_cache_lru_shards: int = 8
    log_interval_batches: int = 100

    canvas_size: int = 1026
    active_size: int = 986
    expert_size: int = 224
    expert_pitch: int = 254
    num_experts: int = 16
    expert_grid_rows: int = 4
    expert_grid_cols: int = 4
    expert_layers: int = 4
    top_k: int = 4
    router_pool_size: int = 14
    router_temperature: float = 1.0
    router_input_layernorm_enabled: bool = True
    router_input_layernorm_eps: float = 1e-5
    amplitude_slm_weight_domain: str = "amplitude"
    amplitude_slm_input_normalization: str = "none"
    amplitude_phase_relay: str = "ideal_4f_identity"

    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 8.0
    expert_interlayer_distance_m: float = 0.10
    last_expert_to_global_distance_m: float = 0.10
    global_to_detector_distance_m: float = 0.10
    phase_parameterization: str = "sigmoid"
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    phase_dropout_mode: str = "none"
    phase_dropout_p: float = 0.0
    phase_dropout_block_size: int = 8
    phase_dropout_batch_shared: bool = True
    k_space_constraint_enabled: bool = False
    theta_max_deg: float = 1.0

    detector_output_size: int = 224
    detector_layernorm_eps: float = 1e-5
    detector_layernorm_affine: bool = False
    detector_layernorm_scope: str = "per_token"

    vision_epochs: int = 30
    language_epochs: int = 30
    joint_epochs: int = 30
    learning_rate: float = 2e-3
    router_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    router_balance_weight: float = 0.03
    router_importance_weight: float = 0.0
    checkpoint_interval_epochs: int = 1
    progress: bool = True

    manifest_digest: str | None = None

    def validate(self) -> None:
        if self.config_version != 1:
            raise ValueError("config_version must be 1")
        if self.dataset != "cc3m_jsonl":
            raise ValueError("dataset.name must be 'cc3m_jsonl'")
        if not self.caption_prompt_template or "{caption}" not in self.caption_prompt_template:
            raise ValueError("caption_prompt_template must contain {caption}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0,1)")
        if self.sample_limit is not None and self.sample_limit <= 1:
            raise ValueError("sample_limit must be greater than one")
        if self.calibration_sample_count <= 0:
            raise ValueError("calibration_sample_count must be positive")
        if not self.dataset_source_repo_id or not self.dataset_source_revision:
            raise ValueError("CC3M source repo_id and revision must be non-empty")
        if self.dataset_source_split not in {"train", "validation"}:
            raise ValueError("dataset.prepare.source_split must be train or validation")
        if self.dataset_download_workers <= 0:
            raise ValueError("dataset.prepare.download_workers must be positive")
        if (
            self.dataset_download_max_shards is not None
            and self.dataset_download_max_shards <= 0
        ):
            raise ValueError("dataset.prepare.max_shards must be positive or null")
        if self.model_id != MODEL_ID and not Path(self.model_id).is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or an existing local directory")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        if self.latent_dim != 224 or self.max_visual_tokens != 224 or self.max_language_tokens != 224:
            raise ValueError("PCA latent dimension and both maximum token counts are fixed at 224")
        if self.detector_output_size != self.latent_dim or self.expert_size != self.latent_dim:
            raise ValueError("expert_size, detector_output_size, and latent_dim must match")
        geometry = (
            self.canvas_size,
            self.active_size,
            self.expert_size,
            self.expert_pitch,
            self.num_experts,
            self.expert_grid_rows,
            self.expert_grid_cols,
        )
        if geometry != (1026, 986, 224, 254, 16, 4, 4):
            raise ValueError("The formal MoE16-224 geometry is fixed at 1026/986/224/254/16/4x4")
        if self.expert_layers != 4 or self.top_k != 4:
            raise ValueError("This experiment requires four optical stages and top_k=4")
        if self.detector_layernorm_affine:
            raise ValueError("Signed detector readout must use non-affine LayerNorm")
        if self.detector_layernorm_scope not in {"per_token", "full_field"}:
            raise ValueError("detector_layernorm_scope must be per_token or full_field")
        if self.phase_dropout_mode not in {"none", "phase_bypass", "block_phase_bypass"}:
            raise ValueError("Unsupported phase dropout mode")
        if not 0.0 <= self.phase_dropout_p < 1.0:
            raise ValueError("phase_dropout_p must be in [0,1)")
        if self.amplitude_slm_weight_domain not in {"amplitude", "power"}:
            raise ValueError("amplitude_slm_weight_domain must be amplitude or power")
        if self.amplitude_slm_input_normalization not in {"none", "per_sample_max"}:
            raise ValueError("Unsupported amplitude SLM input normalization")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported Qwen dtype")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        positive = (
            "feature_batch_size", "student_batch_size", "validation_batch_size",
            "cpu_threads", "cpu_interop_threads", "teacher_cache_shard_size",
            "teacher_cache_lru_shards", "log_interval_batches",
            "pca_vision_calibration_tokens", "pca_language_calibration_tokens",
            "pca_oracle_samples", "vision_epochs", "language_epochs", "joint_epochs",
            "checkpoint_interval_epochs",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        for name in (
            "learning_rate", "router_learning_rate", "pca_eigenvalue_eps",
            "wavelength_nm", "pixel_pitch_um", "expert_interlayer_distance_m",
            "last_expert_to_global_distance_m", "global_to_detector_distance_m",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or self.router_balance_weight < 0 or self.router_importance_weight < 0:
            raise ValueError("loss/regularization weights must be non-negative")

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.text_depth = int(model.config.text_config.num_hidden_layers)
        self.text_hidden_size = int(model.config.text_config.hidden_size)
        visual = getattr(getattr(model, "model", model), "visual", None)
        indexes = tuple(int(value) for value in getattr(visual, "deepstack_visual_indexes", ()))
        if len(indexes) != 3:
            raise RuntimeError(f"Expected three native Qwen3-VL DeepStack taps, got {indexes}")
        self.deepstack_visual_indexes = indexes
        self.language_tap_indexes = (0, 1, 2, self.text_depth - 1)

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for path, attribute in NESTED_FIELDS.items():
            cursor = output
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            value = getattr(self, attribute)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            cursor[path[-1]] = value
        return output


NESTED_FIELDS: dict[tuple[str, ...], str] = {
    ("config_version",): "config_version",
    ("experiment", "name"): "experiment_name",
    ("experiment", "output_dir"): "output_dir",
    ("experiment", "seed"): "seed",
    ("dataset", "name"): "dataset",
    ("dataset", "manifest_path"): "manifest_path",
    ("dataset", "validation_manifest_path"): "validation_manifest_path",
    ("dataset", "validation_fraction"): "validation_fraction",
    ("dataset", "sample_limit"): "sample_limit",
    ("dataset", "calibration_sample_count"): "calibration_sample_count",
    ("dataset", "caption_prompt_template"): "caption_prompt_template",
    ("dataset", "manifest_digest"): "manifest_digest",
    ("dataset", "prepare", "auto_if_missing"): "dataset_auto_prepare",
    ("dataset", "prepare", "source_repo_id"): "dataset_source_repo_id",
    ("dataset", "prepare", "source_revision"): "dataset_source_revision",
    ("dataset", "prepare", "endpoint"): "dataset_source_endpoint",
    ("dataset", "prepare", "source_split"): "dataset_source_split",
    ("dataset", "prepare", "archive_dir"): "dataset_archive_dir",
    ("dataset", "prepare", "image_dir"): "dataset_image_dir",
    ("dataset", "prepare", "download_workers"): "dataset_download_workers",
    ("dataset", "prepare", "max_shards"): "dataset_download_max_shards",
    ("dataset", "prepare", "keep_archives"): "dataset_keep_archives",
    ("qwen", "model_id"): "model_id",
    ("qwen", "cache_dir"): "cache_dir",
    ("qwen", "local_files_only"): "local_files_only",
    ("qwen", "processor", "min_pixels"): "processor_min_pixels",
    ("qwen", "processor", "max_pixels"): "processor_max_pixels",
    ("qwen", "runtime", "dtype"): "dtype",
    ("qwen", "runtime", "attn_implementation"): "attn_implementation",
    ("qwen", "runtime", "device"): "device",
    ("qwen", "architecture", "vision_depth"): "vision_depth",
    ("qwen", "architecture", "vision_hidden_size"): "vision_hidden_size",
    ("qwen", "architecture", "text_depth"): "text_depth",
    ("qwen", "architecture", "text_hidden_size"): "text_hidden_size",
    ("qwen", "architecture", "deepstack_visual_indexes"): "deepstack_visual_indexes",
    ("qwen", "architecture", "language_tap_indexes"): "language_tap_indexes",
    ("pca", "latent_dim"): "latent_dim",
    ("pca", "vision_calibration_tokens"): "pca_vision_calibration_tokens",
    ("pca", "language_calibration_tokens"): "pca_language_calibration_tokens",
    ("pca", "eigenvalue_eps"): "pca_eigenvalue_eps",
    ("pca", "oracle_samples"): "pca_oracle_samples",
    ("pca", "oracle_warn_cosine_below"): "pca_oracle_warn_cosine_below",
    ("token_limits", "visual"): "max_visual_tokens",
    ("token_limits", "language"): "max_language_tokens",
    ("batching", "feature_batch_size"): "feature_batch_size",
    ("batching", "student_batch_size"): "student_batch_size",
    ("batching", "validation_batch_size"): "validation_batch_size",
    ("batching", "num_workers"): "num_workers",
    ("batching", "cpu_threads"): "cpu_threads",
    ("batching", "cpu_interop_threads"): "cpu_interop_threads",
    ("cache", "root"): "precompute_cache_dir",
    ("cache", "dtype"): "cache_dtype",
    ("cache", "shard_size"): "teacher_cache_shard_size",
    ("cache", "lru_shards"): "teacher_cache_lru_shards",
    ("cache", "log_interval_batches"): "log_interval_batches",
    ("moe", "geometry", "canvas_size"): "canvas_size",
    ("moe", "geometry", "active_size"): "active_size",
    ("moe", "geometry", "expert_size"): "expert_size",
    ("moe", "geometry", "expert_pitch"): "expert_pitch",
    ("moe", "geometry", "num_experts"): "num_experts",
    ("moe", "geometry", "grid_rows"): "expert_grid_rows",
    ("moe", "geometry", "grid_cols"): "expert_grid_cols",
    ("moe", "geometry", "stages"): "expert_layers",
    ("moe", "router", "top_k"): "top_k",
    ("moe", "router", "pool_size"): "router_pool_size",
    ("moe", "router", "temperature"): "router_temperature",
    ("moe", "router", "input_layernorm_enabled"): "router_input_layernorm_enabled",
    ("moe", "router", "input_layernorm_eps"): "router_input_layernorm_eps",
    ("moe", "router", "amplitude_slm_weight_domain"): "amplitude_slm_weight_domain",
    ("moe", "router", "amplitude_slm_input_normalization"): "amplitude_slm_input_normalization",
    ("moe", "router", "amplitude_phase_relay"): "amplitude_phase_relay",
    ("moe", "optics", "wavelength_nm"): "wavelength_nm",
    ("moe", "optics", "pixel_pitch_um"): "pixel_pitch_um",
    ("moe", "optics", "inter_layer_distance_m"): "expert_interlayer_distance_m",
    ("moe", "optics", "last_expert_to_global_distance_m"): "last_expert_to_global_distance_m",
    ("moe", "optics", "global_to_detector_distance_m"): "global_to_detector_distance_m",
    ("moe", "optics", "phase_parameterization"): "phase_parameterization",
    ("moe", "optics", "phase_init"): "phase_init",
    ("moe", "optics", "phase_init_std"): "phase_init_std",
    ("moe", "optics", "phase_dropout_mode"): "phase_dropout_mode",
    ("moe", "optics", "phase_dropout_p"): "phase_dropout_p",
    ("moe", "optics", "phase_dropout_block_size"): "phase_dropout_block_size",
    ("moe", "optics", "phase_dropout_batch_shared"): "phase_dropout_batch_shared",
    ("moe", "optics", "k_space_constraint_enabled"): "k_space_constraint_enabled",
    ("moe", "optics", "theta_max_deg"): "theta_max_deg",
    ("moe", "detector", "output_size"): "detector_output_size",
    ("moe", "detector", "layernorm_eps"): "detector_layernorm_eps",
    ("moe", "detector", "layernorm_affine"): "detector_layernorm_affine",
    ("moe", "detector", "layernorm_scope"): "detector_layernorm_scope",
    ("training", "vision_epochs"): "vision_epochs",
    ("training", "language_epochs"): "language_epochs",
    ("training", "joint_epochs"): "joint_epochs",
    ("training", "learning_rate"): "learning_rate",
    ("training", "router_learning_rate"): "router_learning_rate",
    ("training", "weight_decay"): "weight_decay",
    ("training", "scheduler"): "scheduler",
    ("training", "checkpoint_interval_epochs"): "checkpoint_interval_epochs",
    ("training", "progress"): "progress",
    ("loss", "router_balance_weight"): "router_balance_weight",
    ("loss", "router_importance_weight"): "router_importance_weight",
}


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_name = raw.pop("base_config", None)
    if base_name is not None:
        base_path = resolve_path(base_name, config_path.parent, "base_config")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base.pop("base_config", None)
        raw = _deep_merge(base, raw)
    reverse = {path: attribute for path, attribute in NESTED_FIELDS.items()}
    allowed = {field.name for field in fields(Settings)}
    values: dict[str, Any] = {}

    def visit(value: Any, path_parts: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, (*path_parts, key))
            return
        if path_parts in reverse:
            values[reverse[path_parts]] = value
        elif len(path_parts) == 1 and path_parts[0] in allowed:
            values[path_parts[0]] = value
        else:
            raise ValueError(f"Unknown config key: {'.'.join(path_parts)}")

    for key, value in raw.items():
        visit(value, (key,))
    if values.get("model_id") and values["model_id"] != MODEL_ID:
        values["model_id"] = str(resolve_path(values["model_id"], config_path.parent, "model_id"))
    for name in PATH_FIELDS:
        if values.get(name) is not None:
            values[name] = resolve_path(values[name], config_path.parent, name)
    for name in ("deepstack_visual_indexes", "language_tap_indexes"):
        if isinstance(values.get(name), list):
            values[name] = tuple(values[name])
    settings = Settings(**values)
    settings.validate()
    return settings


def resolve_path(value: str | Path, base: Path, field_name: str) -> Path:
    raw = os.path.expanduser(str(value))
    missing = sorted({a or b for a, b in ENV_REFERENCE.findall(raw) if not os.environ.get(a or b)})
    if missing:
        raise ValueError(f"{field_name} references unset environment variables: {', '.join(missing)}")
    expanded = os.path.expandvars(raw)
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
