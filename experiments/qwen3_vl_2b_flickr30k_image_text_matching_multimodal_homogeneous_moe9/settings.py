from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import DATASET_REPO_ID, MODEL_ID, PROMPT_TEMPLATE


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {"data_root", "output_dir", "cache_dir"}
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class Settings:
    config_version: int = 2
    experiment_name: str = "qwen3_vl_2b_flickr30k_itm_vision_language_homogeneous_moe9"
    dataset: str = "flickr30k_image_text_matching"
    data_root: Path = PROJECT_DIR.parent.parent / "data" / "flickr30k"
    dataset_repo_id: str = DATASET_REPO_ID
    dataset_revision: str | None = None
    hf_endpoint: str | None = None
    download: bool = True
    validate_standard_counts: bool = True
    captions_per_image: int = 1
    negatives_per_positive: int = 1
    negative_sampling_algorithm: str = "stable_caption_sattolo_derangement_v1"
    train_image_limit: int | None = None
    test_image_limit: int | None = None
    train_samples_per_epoch: int | None = None
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_flickr30k_itm_vision_language_homogeneous_moe9"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int = 25600
    processor_max_pixels: int = 25600
    prompt_template: str = PROMPT_TEMPLATE
    feature_batch_size: int = 1
    student_batch_size: int = 1
    inference_batch_size: int = 1
    head_batch_size: int = 512
    teacher_cache_shard_size: int = 128
    teacher_cache_lru_shards: int = 8
    teacher_cache_log_interval_batches: int = 100
    num_workers: int = 8
    cache_dtype: str = "float16"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    head_type: str = "normalized_binary_classification"
    student_head_learning_rate: float = 1e-3
    initialize_student_head_from_teacher: bool = True
    input_adapter_dim: int = 120
    max_visual_tokens: int = 120
    max_language_tokens: int = 120
    student_language_mode: str = "optical_moe"
    vision_tap_stages: tuple[int, int, int] = (1, 3, 4)
    canvas_size: int = 480
    active_size: int = 450
    expert_size: int = 120
    expert_pitch: int = 150
    num_experts: int = 9
    top_k: int = 3
    router_pool_size: int = 10
    router_temperature: float = 1.0
    router_learning_rate: float = 1e-3
    router_input_layernorm_enabled: bool = True
    router_input_layernorm_eps: float = 1e-5
    expert_layers: int = 5
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 16.0
    prompt_focal_length_m: float = 0.0722
    prompt_to_expert_distance_m: float = 0.1444
    expert_interlayer_distance_m: float = 0.20
    last_expert_to_global_distance_m: float = 0.20
    global_to_detector_distance_m: float = 0.20
    phase_parameterization: str = "sigmoid"
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    k_space_constraint_enabled: bool = False
    theta_max_deg: float = 1.0
    interlayer_layernorm_eps: float = 1e-5
    interlayer_nonlinearity: str = "relu"
    interlayer_enabled: bool = True
    interlayer_per_expert_enabled: bool = True
    interlayer_elementwise_affine: bool = False
    interlayer_hard_route_mask: bool = True
    interlayer_reapply_routing_weights: bool = True
    detector_pool_kernel: int = 4
    detector_layernorm_eps: float = 1e-5
    detector_layernorm_affine: bool = False
    detector_nonlinearity: str = "relu"
    loss_hidden_weight: float = 1.0
    loss_answer_weight: float = 1.0
    loss_logit_weight: float = 0.5
    loss_classification_weight: float = 1.0
    smooth_l1_beta: float = 0.1
    router_balance_weight: float = 0.03
    router_importance_weight: float = 0.0
    classification_threshold: float = 0.5
    log_interval_batches: int = 100
    checkpoint_interval_epochs: int = 1
    student_selection_split: str = "test"
    student_selection_metric: str = "auroc"
    phase_dropout_enabled: bool = False
    phase_dropout_mode: str = "none"
    phase_dropout_p: float = 0.0
    phase_dropout_block_size: int = 8
    phase_dropout_batch_shared: bool = True
    phase_dropout_start_epoch: int = 0
    visualization_enabled: bool = True
    visualization_interval_epochs: int = 10
    save_phase_masks: bool = True
    save_training_curves: bool = True
    save_confusion_matrix: bool = True
    save_predictions: bool = True
    seed: int = 42
    progress: bool = True
    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    text_depth: int | None = None
    text_hidden_size: int | None = None
    deepstack_visual_indexes: tuple[int, ...] | None = None
    resolved_dataset_fingerprints: dict[str, str] | None = None
    pair_manifest_digests: dict[str, str] | None = None

    def validate(self) -> None:
        if self.config_version != 2:
            raise ValueError("config_version must be 2")
        if self.dataset != "flickr30k_image_text_matching":
            raise ValueError("This experiment requires dataset=flickr30k_image_text_matching")
        if self.dataset_repo_id != DATASET_REPO_ID:
            raise ValueError(f"dataset.repo_id must be {DATASET_REPO_ID}")
        if self.model_id != MODEL_ID and not Path(self.model_id).is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or an existing local directory")
        if self.prompt_template.count("{caption}") != 1:
            raise ValueError("prompt_template must contain exactly one {caption} placeholder")
        if self.negative_sampling_algorithm != "stable_caption_sattolo_derangement_v1":
            raise ValueError("Unsupported negative sampling algorithm")
        if not 1 <= self.captions_per_image <= 5:
            raise ValueError("captions_per_image must be in [1,5]")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be positive")
        if self.head_type != "normalized_binary_classification":
            raise ValueError("Teacher and student must use normalized_binary_classification")
        if self.student_language_mode not in {"electronic", "optical_moe"}:
            raise ValueError("student.language_stack_mode must be electronic or optical_moe")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        if (self.canvas_size, self.active_size, self.expert_size, self.expert_pitch, self.num_experts) != (480, 450, 120, 150, 9):
            raise ValueError("Verified geometry is fixed at canvas480/active450/expert120/pitch150/9 experts")
        if self.input_adapter_dim != 120 or self.max_visual_tokens != 120 or self.max_language_tokens != 120:
            raise ValueError("Direct token-row mapping requires optical channels and token limits of 120")
        if self.expert_layers != 5 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("Five expert layers and a valid top_k are required")
        if len(self.vision_tap_stages) != 3 or tuple(sorted(self.vision_tap_stages)) != tuple(self.vision_tap_stages):
            raise ValueError("vision tap stages must be three ordered stages")
        if any(stage < 1 or stage > self.expert_layers for stage in self.vision_tap_stages):
            raise ValueError("vision tap stage is outside the optical stack")
        if abs(self.prompt_to_expert_distance_m - 2.0 * self.prompt_focal_length_m) > 1e-6:
            raise ValueError("prompt_to_expert must equal 2 * prompt_focal_length")
        if self.detector_pool_kernel != 4 or self.canvas_size // self.detector_pool_kernel != 120:
            raise ValueError("Detector pooling must map 480x480 to 120x120")
        if self.detector_layernorm_affine:
            raise ValueError("Post-detector LayerNorm must be non-affine")
        if self.interlayer_hard_route_mask and not self.interlayer_per_expert_enabled:
            raise ValueError("hard routing requires per-expert normalization")
        if self.interlayer_reapply_routing_weights and not self.interlayer_per_expert_enabled:
            raise ValueError("routing-weight restoration requires per-expert normalization")
        if self.interlayer_nonlinearity not in {"relu", "softplus"} or self.detector_nonlinearity not in {"relu", "softplus"}:
            raise ValueError("Optical nonlinearities must be relu or softplus")
        if self.optimizer_type not in {"adam", "adamw"} or self.scheduler_type not in {"cosine", "none"}:
            raise ValueError("Unsupported optimizer or scheduler")
        if self.student_selection_split != "test" or self.student_selection_metric != "auroc":
            raise ValueError("This user-requested experiment selects best checkpoints by test AUROC")
        if not 0.0 < self.classification_threshold < 1.0:
            raise ValueError("classification_threshold must be in (0,1)")
        if self.phase_dropout_mode not in {"none", "phase_bypass", "block_phase_bypass"}:
            raise ValueError("Unsupported phase dropout mode")
        if not 0.0 <= self.phase_dropout_p < 1.0:
            raise ValueError("phase dropout p must be in [0,1)")
        if self.phase_dropout_enabled and (self.phase_dropout_mode == "none" or self.phase_dropout_p <= 0):
            raise ValueError("Enabled phase dropout requires a non-none mode and positive p")
        if self.cache_dtype not in {"float16", "float32"} or self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported dtype")
        positive = ("feature_batch_size", "student_batch_size", "inference_batch_size", "head_batch_size",
                    "teacher_cache_shard_size", "teacher_cache_lru_shards", "teacher_cache_log_interval_batches",
                    "epochs", "log_interval_batches", "checkpoint_interval_epochs", "visualization_interval_epochs")
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        for name in ("train_image_limit", "test_image_limit", "train_samples_per_epoch"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        for name in ("loss_hidden_weight", "loss_answer_weight", "loss_logit_weight", "loss_classification_weight",
                     "router_balance_weight", "router_importance_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.smooth_l1_beta <= 0 or self.learning_rate <= 0 or self.student_head_learning_rate <= 0:
            raise ValueError("Loss beta and learning rates must be positive")

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.text_depth = int(model.config.text_config.num_hidden_layers)
        self.text_hidden_size = int(model.config.text_config.hidden_size)
        visual = getattr(getattr(model, "model", model), "visual", None)
        self.deepstack_visual_indexes = tuple(int(value) for value in getattr(visual, "deepstack_visual_indexes", ()))
        if len(self.deepstack_visual_indexes) != 3:
            raise RuntimeError(f"Expected three native Qwen3-VL DeepStack taps, got {self.deepstack_visual_indexes}")

    def to_dict(self) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for path, attribute in NESTED_FIELDS.items():
            cursor = grouped
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = getattr(self, attribute)
        return grouped


NESTED_FIELDS: dict[tuple[str, ...], str] = {
    ("config_version",): "config_version",
    ("experiment", "name"): "experiment_name", ("experiment", "output_dir"): "output_dir", ("experiment", "seed"): "seed",
    ("dataset", "name"): "dataset", ("dataset", "data_root"): "data_root",
    ("dataset", "repo_id"): "dataset_repo_id", ("dataset", "revision"): "dataset_revision",
    ("dataset", "hf_endpoint"): "hf_endpoint", ("dataset", "download"): "download",
    ("dataset", "validate_standard_counts"): "validate_standard_counts",
    ("dataset", "captions_per_image"): "captions_per_image",
    ("dataset", "negatives_per_positive"): "negatives_per_positive",
    ("dataset", "negative_sampling_algorithm"): "negative_sampling_algorithm",
    ("dataset", "train_image_limit"): "train_image_limit", ("dataset", "test_image_limit"): "test_image_limit",
    ("dataset", "train_samples_per_epoch"): "train_samples_per_epoch",
    ("dataset", "resolved_fingerprints"): "resolved_dataset_fingerprints",
    ("dataset", "pair_manifest_digests"): "pair_manifest_digests",
    ("qwen", "model_id"): "model_id", ("qwen", "cache_dir"): "cache_dir",
    ("qwen", "local_files_only"): "local_files_only", ("qwen", "prompt_template"): "prompt_template",
    ("qwen", "processor", "min_pixels"): "processor_min_pixels", ("qwen", "processor", "max_pixels"): "processor_max_pixels",
    ("qwen", "runtime", "dtype"): "dtype", ("qwen", "runtime", "attn_implementation"): "attn_implementation",
    ("qwen", "runtime", "device"): "device", ("qwen", "architecture", "vision_depth"): "vision_depth",
    ("qwen", "architecture", "vision_hidden_size"): "vision_hidden_size",
    ("qwen", "architecture", "text_depth"): "text_depth", ("qwen", "architecture", "text_hidden_size"): "text_hidden_size",
    ("qwen", "architecture", "deepstack_visual_indexes"): "deepstack_visual_indexes",
    ("student", "language_stack_mode"): "student_language_mode",
    ("student", "initialize_head_from_teacher"): "initialize_student_head_from_teacher",
    ("batching", "feature_batch_size"): "feature_batch_size", ("batching", "student_batch_size"): "student_batch_size",
    ("batching", "inference_batch_size"): "inference_batch_size", ("batching", "head_batch_size"): "head_batch_size",
    ("batching", "num_workers"): "num_workers",
    ("teacher_cache", "dtype"): "cache_dtype", ("teacher_cache", "shard_size"): "teacher_cache_shard_size",
    ("teacher_cache", "lru_shards"): "teacher_cache_lru_shards",
    ("teacher_cache", "log_interval_batches"): "teacher_cache_log_interval_batches",
    ("vision_adapter", "optical_channels"): "input_adapter_dim", ("vision_adapter", "max_visual_tokens"): "max_visual_tokens",
    ("vision_adapter", "tap_stages"): "vision_tap_stages", ("language_adapter", "max_language_tokens"): "max_language_tokens",
    ("moe", "geometry", "canvas_size"): "canvas_size", ("moe", "geometry", "active_size"): "active_size",
    ("moe", "geometry", "expert_size"): "expert_size", ("moe", "geometry", "expert_pitch"): "expert_pitch",
    ("moe", "geometry", "num_experts"): "num_experts", ("moe", "geometry", "layers_per_expert"): "expert_layers",
    ("moe", "router", "top_k"): "top_k", ("moe", "router", "pool_size"): "router_pool_size",
    ("moe", "router", "temperature"): "router_temperature", ("moe", "router", "learning_rate"): "router_learning_rate",
    ("moe", "router", "input_layernorm_enabled"): "router_input_layernorm_enabled",
    ("moe", "router", "input_layernorm_eps"): "router_input_layernorm_eps",
    ("moe", "optics", "wavelength_nm"): "wavelength_nm", ("moe", "optics", "pixel_pitch_um"): "pixel_pitch_um",
    ("moe", "optics", "prompt_focal_length_m"): "prompt_focal_length_m",
    ("moe", "optics", "distances_m", "prompt_to_expert"): "prompt_to_expert_distance_m",
    ("moe", "optics", "distances_m", "inter_layer"): "expert_interlayer_distance_m",
    ("moe", "optics", "distances_m", "last_expert_to_global"): "last_expert_to_global_distance_m",
    ("moe", "optics", "distances_m", "global_to_detector"): "global_to_detector_distance_m",
    ("moe", "optics", "phase", "parameterization"): "phase_parameterization",
    ("moe", "optics", "phase", "init"): "phase_init", ("moe", "optics", "phase", "init_std"): "phase_init_std",
    ("moe", "optics", "k_space", "enabled"): "k_space_constraint_enabled",
    ("moe", "optics", "k_space", "theta_max_deg"): "theta_max_deg",
    ("moe", "optoelectronic_interlayers", "enabled"): "interlayer_enabled",
    ("moe", "optoelectronic_interlayers", "per_expert_enabled"): "interlayer_per_expert_enabled",
    ("moe", "optoelectronic_interlayers", "elementwise_affine"): "interlayer_elementwise_affine",
    ("moe", "optoelectronic_interlayers", "hard_route_mask"): "interlayer_hard_route_mask",
    ("moe", "optoelectronic_interlayers", "reapply_routing_weights"): "interlayer_reapply_routing_weights",
    ("moe", "optoelectronic_interlayers", "layernorm_eps"): "interlayer_layernorm_eps",
    ("moe", "optoelectronic_interlayers", "nonlinearity"): "interlayer_nonlinearity",
    ("moe", "final_detector_readout", "pool_kernel"): "detector_pool_kernel",
    ("moe", "final_detector_readout", "layernorm_eps"): "detector_layernorm_eps",
    ("moe", "final_detector_readout", "layernorm_affine"): "detector_layernorm_affine",
    ("moe", "final_detector_readout", "nonlinearity"): "detector_nonlinearity",
    ("classification_head", "type"): "head_type", ("classification_head", "threshold"): "classification_threshold",
    ("loss", "vision_hidden_weight"): "loss_hidden_weight", ("loss", "answer_hidden_weight"): "loss_answer_weight",
    ("loss", "logit_distill_weight"): "loss_logit_weight", ("loss", "classification_weight"): "loss_classification_weight",
    ("loss", "smooth_l1_beta"): "smooth_l1_beta", ("loss", "router_balance_weight"): "router_balance_weight",
    ("loss", "router_importance_weight"): "router_importance_weight",
    ("optimizer", "type"): "optimizer_type", ("optimizer", "learning_rate"): "learning_rate",
    ("optimizer", "student_head_learning_rate"): "student_head_learning_rate",
    ("optimizer", "weight_decay"): "weight_decay", ("optimizer", "scheduler"): "scheduler_type",
    ("training", "epochs"): "epochs", ("training", "logging", "interval_batches"): "log_interval_batches",
    ("training", "progress"): "progress", ("training", "checkpoint_interval_epochs"): "checkpoint_interval_epochs",
    ("training", "student_selection_split"): "student_selection_split",
    ("training", "student_selection_metric"): "student_selection_metric",
    ("regularization", "phase_dropout", "enabled"): "phase_dropout_enabled",
    ("regularization", "phase_dropout", "mode"): "phase_dropout_mode",
    ("regularization", "phase_dropout", "p"): "phase_dropout_p",
    ("regularization", "phase_dropout", "block_size"): "phase_dropout_block_size",
    ("regularization", "phase_dropout", "batch_shared"): "phase_dropout_batch_shared",
    ("regularization", "phase_dropout", "start_epoch"): "phase_dropout_start_epoch",
    ("visualization", "enabled"): "visualization_enabled", ("visualization", "interval_epochs"): "visualization_interval_epochs",
    ("visualization", "save_phase_masks"): "save_phase_masks", ("visualization", "save_training_curves"): "save_training_curves",
    ("visualization", "save_confusion_matrix"): "save_confusion_matrix", ("visualization", "save_predictions"): "save_predictions",
}


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_name = raw.pop("base_config", None)
    if base_name is not None:
        base_path = resolve_path(base_name, config_path.parent, "base_config")
        base_raw = json.loads(base_path.read_text(encoding="utf-8")); base_raw.pop("base_config", None)
        raw = _deep_merge(base_raw, raw)
    allowed = {item.name for item in fields(Settings)}; reverse = {path: attr for path, attr in NESTED_FIELDS.items()}
    values: dict[str, Any] = {}

    def visit(value: Any, path_parts: tuple[str, ...]) -> None:
        if isinstance(value, dict) and path_parts not in reverse:
            for key, nested in value.items(): visit(nested, (*path_parts, key))
        elif path_parts in reverse:
            values[reverse[path_parts]] = value
        elif len(path_parts) == 1 and path_parts[0] in allowed:
            values[path_parts[0]] = value
        else:
            raise ValueError(f"Unknown config key: {'.'.join(path_parts)}")

    for key, value in raw.items(): visit(value, (key,))
    if values.get("model_id") and values["model_id"] != MODEL_ID:
        values["model_id"] = str(resolve_path(values["model_id"], config_path.parent, "model_id"))
    for name in PATH_FIELDS:
        if values.get(name) is not None: values[name] = resolve_path(values[name], config_path.parent, name)
    settings = Settings(**values); settings.validate(); return settings


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolve_path(value: str | Path, base: Path, field_name: str) -> Path:
    raw = os.path.expanduser(str(value))
    missing = sorted({a or b for a, b in ENV_REFERENCE.findall(raw) if not os.environ.get(a or b)})
    if missing: raise ValueError(f"{field_name} references unset environment variables: {', '.join(missing)}")
    path = Path(os.path.expandvars(raw))
    return (path if path.is_absolute() else base / path).resolve()
