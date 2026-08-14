from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
SUPPORTED_GALLERY_AGGREGATIONS = {"mean_prototype", "max_similarity"}


def _nested(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (expanded if expanded.is_absolute() else base / expanded).resolve()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic base_config reference involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    parent_value = raw.pop("base_config", None)
    if parent_value is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent_value))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), raw)


@dataclass
class Settings:
    config_path: Path
    dataset_root: Path
    output_dir: Path
    selected_classes: tuple[str, ...]
    use_all_classes: bool
    download: bool
    download_url: str
    download_timeout_sec: int
    delete_archive_after_extract: bool
    test_fraction: float
    train_limit_per_class: int | None
    test_limit_per_class: int | None
    gallery_images_per_class: int
    minimum_images_per_class: int

    model_id: str
    cache_dir: Path | None
    teacher_cache_source_path: Path | None
    local_files_only: bool
    instruction: str
    processor_min_pixels: int
    processor_max_pixels: int
    dtype: str
    attn_implementation: str
    device: str

    image_size: int
    embedding_dim: int
    gallery_aggregation: str
    teacher_batch_size: int
    batch_size: int
    pk_classes_per_batch: int
    pk_images_per_class: int
    inference_batch_size: int
    num_workers: int
    optimizer_steps_per_epoch: int | None

    epochs: int
    learning_rate: float
    adapter_learning_rate: float | None
    readout_learning_rate: float | None
    weight_decay: float
    lambda_kd: float
    lambda_relational_kd: float
    lambda_ret: float
    lambda_gallery: float
    lambda_teacher_gallery: float
    lambda_router_balance: float
    lambda_router_importance: float
    lambda_router_hard_load_balance: float
    router_load_bias_update_rate: float
    router_load_bias_update_start_epoch: int
    lambda_router_response_consistency: float
    phase_dc_enabled: bool
    lambda_phase_dc: float
    phase_dc_start_epoch: int
    temperature: float
    gallery_temperature: float
    gallery_prototype_stop_gradient: bool
    router_learning_rate: float | None
    phase_learning_rate: float | None
    gradient_clip_norm: float | None
    phase_focus_enabled: bool
    phase_focus_warmup_epochs: int
    phase_focus_interval_epochs: int
    phase_gradient_measure_interval_batches: int
    phase_motion_warning_epoch: int
    phase_motion_warning_threshold_rad: float
    phase_preview_interval_epochs: int
    ema_decay: float | None
    resume_optimizer_state: bool
    random_seed: int
    amp_enabled: bool
    evaluate_test_each_epoch: bool
    log_interval_batches: int

    augmentation_enabled: bool
    crop_scale_min: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float
    horizontal_flip_probability: float
    visualization_sample_count: int

    # Qwen replacement / optical core settings.
    input_adapter_dim: int
    max_visual_tokens: int
    max_language_tokens: int
    vision_tap_stages: tuple[int, ...]
    student_language_mode: str
    native_pre_attention_enabled: bool
    native_pre_attention_initialize_from_teacher: bool
    native_pre_attention_trainable: bool
    transformer_residual_enabled: bool
    vision_attention_source_layer: int
    language_attention_source_layer: int

    canvas_size: int
    active_size: int
    expert_size: int
    expert_pitch: int
    num_experts: int
    expert_grid_rows: int
    expert_grid_cols: int
    expert_layers: int
    top_k: int
    router_pool_size: int
    router_temperature: float
    router_input_layernorm_enabled: bool
    router_input_layernorm_eps: float
    amplitude_slm_weight_domain: str
    amplitude_slm_input_normalization: str
    amplitude_phase_relay: str

    wavelength_nm: float
    pixel_pitch_um: float
    expert_interlayer_distance_m: float
    last_expert_to_global_distance_m: float
    global_to_detector_distance_m: float
    phase_parameterization: str
    phase_init: str
    phase_init_std: float
    k_space_constraint_enabled: bool
    theta_max_deg: float

    interlayer_enabled: bool
    interlayer_per_expert_enabled: bool
    interlayer_elementwise_affine: bool
    interlayer_hard_route_mask: bool
    interlayer_reapply_routing_weights: bool
    oeo_preserve_response_amplitude: bool
    oeo_response_gain_min: float
    oeo_response_gain_max: float
    interlayer_detector_integration_factor: int
    interlayer_layernorm_eps: float
    interlayer_nonlinearity: str

    detector_output_size: int
    detector_layernorm_eps: float
    detector_layernorm_affine: bool
    detector_layernorm_scope: str
    detector_nonlinearity: str
    phase_dropout_mode: str
    phase_dropout_p: float
    phase_dropout_block_size: int
    phase_dropout_batch_shared: bool

    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    text_depth: int | None = None
    text_hidden_size: int | None = None
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def teacher_cache_path(self) -> Path:
        return self.output_dir / "teacher_cache" / "teacher_embeddings.pt"

    @property
    def dataset_variant(self) -> str:
        return "caltech101_all101" if self.use_all_classes else f"caltech101_{len(self.selected_classes)}class"

    @property
    def subset_manifest_path(self) -> Path:
        return self.output_dir / "manifests" / f"{self.dataset_variant}_subset.csv"

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.text_depth = int(model.config.text_config.num_hidden_layers)
        self.text_hidden_size = int(model.config.text_config.hidden_size)
        visual = getattr(getattr(model, "model", model), "visual", None)
        indexes = getattr(visual, "deepstack_visual_indexes", ())
        self.deepstack_visual_indexes = tuple(int(value) for value in indexes)
        if len(self.deepstack_visual_indexes) != 3:
            raise RuntimeError(
                "The reused Qwen3-VL DeepStack replacement expects exactly three native "
                f"visual taps, got {self.deepstack_visual_indexes}"
            )

    def validate(self) -> None:
        if not self.use_all_classes and len(self.selected_classes) < 2:
            raise ValueError("selected_classes must contain at least two class names")
        if len(set(self.selected_classes)) != len(self.selected_classes):
            raise ValueError("selected_classes must not contain duplicate class names")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if self.embedding_dim != 64:
            raise ValueError("This experiment fixes the retrieval embedding dimension to 64")
        if self.gallery_aggregation not in SUPPORTED_GALLERY_AGGREGATIONS:
            raise ValueError(
                f"gallery_aggregation must be one of {sorted(SUPPORTED_GALLERY_AGGREGATIONS)}"
            )
        if self.batch_size != self.pk_classes_per_batch * self.pk_images_per_class:
            raise ValueError("batch_size must equal P*K for the PK sampler")
        if not self.use_all_classes and self.pk_classes_per_batch > len(self.selected_classes):
            raise ValueError("PK sampler P cannot exceed the selected class count")
        if self.pk_images_per_class < 2:
            raise ValueError("PK sampler K must be at least 2 for supervised contrastive loss")
        if self.gallery_images_per_class < 1:
            raise ValueError("gallery_images_per_class must be positive")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("dataset.test_fraction must be strictly between 0 and 1")
        if self.minimum_images_per_class < self.gallery_images_per_class + 2:
            raise ValueError(
                "minimum_images_per_class must leave at least one train and one query image"
            )
        if self.download_timeout_sec <= 0:
            raise ValueError("dataset.download_timeout_sec must be positive")
        for name in (
            "image_size",
            "teacher_batch_size",
            "batch_size",
            "inference_batch_size",
            "epochs",
            "log_interval_batches",
            "input_adapter_dim",
            "max_visual_tokens",
            "max_language_tokens",
            "canvas_size",
            "active_size",
            "expert_size",
            "expert_pitch",
            "num_experts",
            "expert_grid_rows",
            "expert_grid_cols",
            "expert_layers",
            "top_k",
            "router_pool_size",
            "detector_output_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("augmentation.horizontal_flip_probability must be in [0,1]")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
        distances = (
            self.expert_interlayer_distance_m,
            self.last_expert_to_global_distance_m,
            self.global_to_detector_distance_m,
        )
        if any(value <= 0 for value in distances):
            raise ValueError("All optical propagation distances must be positive")
        if max(distances) - min(distances) > 1.0e-12:
            raise ValueError(
                "This optical core reuses one transfer function; inter-layer, "
                "expert-to-global and global-to-detector distances must match"
            )
        if self.input_adapter_dim != self.expert_size:
            raise ValueError("input_adapter_dim must equal expert_size for token-row optical mapping")
        if self.detector_output_size != self.input_adapter_dim:
            raise ValueError("detector_output_size must equal input_adapter_dim")
        if self.max_visual_tokens > self.expert_size or self.max_language_tokens > self.expert_size:
            raise ValueError("token limits cannot exceed expert_size")
        if self.top_k > self.num_experts:
            raise ValueError("router top_k cannot exceed num_experts")
        if self.student_language_mode != "optical_moe":
            raise ValueError("This retrieval experiment requires optical vision and language stacks")
        if self.native_pre_attention_enabled:
            raise ValueError("The first retrieval baseline keeps native electronic attention disabled")
        if self.expert_layers != 1 or self.vision_tap_stages != (1,):
            raise ValueError(
                "The retrieval baseline requires one expert stage and one "
                "student DeepStack tap at stage 1"
            )
        if self.phase_dropout_mode != "none" or self.phase_dropout_p != 0.0:
            raise ValueError("Phase dropout is disabled for the initial retrieval experiment")
        loss_weights = (
            self.lambda_kd,
            self.lambda_relational_kd,
            self.lambda_ret,
            self.lambda_gallery,
            self.lambda_teacher_gallery,
            self.lambda_router_balance,
            self.lambda_router_importance,
            self.lambda_router_hard_load_balance,
            self.lambda_router_response_consistency,
            self.lambda_phase_dc,
        )
        if any(value < 0 for value in loss_weights):
            raise ValueError("All training loss weights must be non-negative")
        if self.lambda_kd + self.lambda_ret + self.lambda_gallery <= 0:
            raise ValueError(
                "At least one of lambda_kd, lambda_ret, or lambda_gallery must be positive"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.gallery_temperature <= 0:
            raise ValueError("gallery_temperature must be positive")
        # A per-group learning rate of exactly zero is an intentional freeze.
        # This is useful for hardware-compatible generalization continuations:
        # the deployed phase/router tensors stay bit-identical while the small
        # electronic adapters/readout are calibrated.  Negative rates remain
        # invalid and the global learning rate is still required to be positive.
        if self.router_learning_rate is not None and self.router_learning_rate < 0:
            raise ValueError("router_learning_rate must be non-negative when configured")
        if self.phase_learning_rate is not None and self.phase_learning_rate < 0:
            raise ValueError("phase_learning_rate must be non-negative when configured")
        if self.adapter_learning_rate is not None and self.adapter_learning_rate < 0:
            raise ValueError("adapter_learning_rate must be non-negative when configured")
        if self.readout_learning_rate is not None and self.readout_learning_rate < 0:
            raise ValueError("readout_learning_rate must be non-negative when configured")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        if self.phase_focus_warmup_epochs < 0:
            raise ValueError("phase_focus_warmup_epochs cannot be negative")
        if self.phase_focus_interval_epochs <= 0:
            raise ValueError("phase_focus_interval_epochs must be positive")
        if self.phase_gradient_measure_interval_batches <= 0:
            raise ValueError("phase_gradient_measure_interval_batches must be positive")
        if self.phase_motion_warning_epoch <= 0:
            raise ValueError("phase_motion_warning_epoch must be positive")
        if self.phase_motion_warning_threshold_rad < 0:
            raise ValueError("phase_motion_warning_threshold_rad cannot be negative")
        if self.phase_preview_interval_epochs <= 0:
            raise ValueError("phase_preview_interval_epochs must be positive")
        if self.phase_dc_start_epoch <= 0:
            raise ValueError("phase_dc_start_epoch must be positive")
        if self.interlayer_detector_integration_factor <= 0:
            raise ValueError("optical.oeo.detector_integration_factor must be positive")
        if self.canvas_size % self.interlayer_detector_integration_factor:
            raise ValueError(
                "canvas_size must be divisible by optical.oeo.detector_integration_factor"
            )
        if self.router_load_bias_update_rate < 0:
            raise ValueError("router_load_bias_update_rate must be non-negative")
        if self.router_load_bias_update_start_epoch < 1:
            raise ValueError("router_load_bias_update_start_epoch must be at least 1")
        if self.expert_size % self.interlayer_detector_integration_factor:
            raise ValueError(
                "expert_size must be divisible by optical.oeo.detector_integration_factor"
            )
        if self.oeo_response_gain_min <= 0:
            raise ValueError("optical.oeo.response_gain_min must be positive")
        if self.oeo_response_gain_max < self.oeo_response_gain_min:
            raise ValueError(
                "optical.oeo.response_gain_max must be >= response_gain_min"
            )
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be strictly between 0 and 1 when configured")
        if self.gallery_aggregation == "mean_prototype" and self.gallery_images_per_class < 1:
            raise ValueError("mean_prototype needs at least one gallery image")
        if self.optimizer_steps_per_epoch is not None and self.optimizer_steps_per_epoch <= 0:
            raise ValueError("batching.optimizer_steps_per_epoch must be positive or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "dataset_root": str(self.dataset_root),
                "download": self.download,
                "download_url": self.download_url,
                "selected_classes": list(self.selected_classes),
                "use_all_classes": self.use_all_classes,
                "download_timeout_sec": self.download_timeout_sec,
                "delete_archive_after_extract": self.delete_archive_after_extract,
                "test_fraction": self.test_fraction,
                "train_limit_per_class": self.train_limit_per_class,
                "test_limit_per_class": self.test_limit_per_class,
                "gallery_images_per_class": self.gallery_images_per_class,
                "minimum_images_per_class": self.minimum_images_per_class,
            },
            "qwen": {
                "model_id": self.model_id,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "teacher_cache_source": (
                    str(self.teacher_cache_source_path)
                    if self.teacher_cache_source_path
                    else None
                ),
                "local_files_only": self.local_files_only,
                "instruction": self.instruction,
                "processor_min_pixels": self.processor_min_pixels,
                "processor_max_pixels": self.processor_max_pixels,
                "dtype": self.dtype,
                "attn_implementation": self.attn_implementation,
                "device": self.device,
                "resolved_architecture": {
                    "vision_depth": self.vision_depth,
                    "vision_hidden_size": self.vision_hidden_size,
                    "text_depth": self.text_depth,
                    "text_hidden_size": self.text_hidden_size,
                    "deepstack_visual_indexes": list(self.deepstack_visual_indexes),
                },
            },
            "retrieval": {
                "image_size": self.image_size,
                "embedding_dim": self.embedding_dim,
                "gallery_aggregation": self.gallery_aggregation,
            },
            "batching": {
                "teacher_batch_size": self.teacher_batch_size,
                "batch_size": self.batch_size,
                "pk_classes_per_batch": self.pk_classes_per_batch,
                "pk_images_per_class": self.pk_images_per_class,
                "inference_batch_size": self.inference_batch_size,
                "num_workers": self.num_workers,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            },
            "training": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "adapter_learning_rate": self.adapter_learning_rate,
                "readout_learning_rate": self.readout_learning_rate,
                "weight_decay": self.weight_decay,
                "lambda_kd": self.lambda_kd,
                "lambda_relational_kd": self.lambda_relational_kd,
                "lambda_ret": self.lambda_ret,
                "lambda_gallery": self.lambda_gallery,
                "lambda_teacher_gallery": self.lambda_teacher_gallery,
                "lambda_router_balance": self.lambda_router_balance,
                "lambda_router_importance": self.lambda_router_importance,
                "lambda_router_hard_load_balance": (
                    self.lambda_router_hard_load_balance
                ),
                "router_load_bias_update_rate": self.router_load_bias_update_rate,
                "router_load_bias_update_start_epoch": (
                    self.router_load_bias_update_start_epoch
                ),
                "lambda_router_response_consistency": (
                    self.lambda_router_response_consistency
                ),
                "phase_dc": {
                    "enabled": self.phase_dc_enabled,
                    "weight": self.lambda_phase_dc,
                    "start_epoch": self.phase_dc_start_epoch,
                },
                # Retain the flat fields in resolved configs/checkpoints so old
                # analysis scripts continue to work.
                "lambda_phase_dc": self.lambda_phase_dc,
                "phase_dc_start_epoch": self.phase_dc_start_epoch,
                "temperature": self.temperature,
                "gallery_temperature": self.gallery_temperature,
                "gallery_prototype_stop_gradient": (
                    self.gallery_prototype_stop_gradient
                ),
                "router_learning_rate": self.router_learning_rate,
                "phase_learning_rate": self.phase_learning_rate,
                "gradient_clip_norm": self.gradient_clip_norm,
                "phase_focus": {
                    "enabled": self.phase_focus_enabled,
                    "warmup_epochs": self.phase_focus_warmup_epochs,
                    "interval_epochs": self.phase_focus_interval_epochs,
                },
                "phase_diagnostics": {
                    "gradient_measure_interval_batches": (
                        self.phase_gradient_measure_interval_batches
                    ),
                    "motion_warning_epoch": self.phase_motion_warning_epoch,
                    "motion_warning_threshold_rad": (
                        self.phase_motion_warning_threshold_rad
                    ),
                    "preview_interval_epochs": self.phase_preview_interval_epochs,
                },
                "ema_decay": self.ema_decay,
                "resume_optimizer_state": self.resume_optimizer_state,
                "random_seed": self.random_seed,
                "amp_enabled": self.amp_enabled,
                "evaluate_test_each_epoch": self.evaluate_test_each_epoch,
                "log_interval_batches": self.log_interval_batches,
            },
            "augmentation": {
                "enabled": self.augmentation_enabled,
                "crop_scale_min": self.crop_scale_min,
                "brightness_jitter": self.brightness_jitter,
                "contrast_jitter": self.contrast_jitter,
                "rotation_degrees": self.rotation_degrees,
                "horizontal_flip_probability": self.horizontal_flip_probability,
            },
            "optical": {
                "input_adapter_dim": self.input_adapter_dim,
                "max_visual_tokens": self.max_visual_tokens,
                "max_language_tokens": self.max_language_tokens,
                "vision_tap_stages": list(self.vision_tap_stages),
                "native_pre_attention_enabled": self.native_pre_attention_enabled,
                "initialize_attention_from_teacher": (
                    self.native_pre_attention_initialize_from_teacher
                ),
                "native_pre_attention_trainable": self.native_pre_attention_trainable,
                "residual_enabled": self.transformer_residual_enabled,
                "geometry": {
                    "canvas_size": self.canvas_size,
                    "active_size": self.active_size,
                    "expert_size": self.expert_size,
                    "expert_pitch": self.expert_pitch,
                    "num_experts": self.num_experts,
                    "grid_rows": self.expert_grid_rows,
                    "grid_cols": self.expert_grid_cols,
                    "layers_per_expert": self.expert_layers,
                },
                "router": {
                    "top_k": self.top_k,
                    "pool_size": self.router_pool_size,
                    "temperature": self.router_temperature,
                    "input_layernorm_enabled": self.router_input_layernorm_enabled,
                    "input_layernorm_eps": self.router_input_layernorm_eps,
                    "amplitude_weight_domain": self.amplitude_slm_weight_domain,
                    "input_normalization": self.amplitude_slm_input_normalization,
                },
                "physics": {
                    "wavelength_nm": self.wavelength_nm,
                    "pixel_pitch_um": self.pixel_pitch_um,
                    "inter_layer_distance_m": self.expert_interlayer_distance_m,
                    "last_expert_to_global_distance_m": (
                        self.last_expert_to_global_distance_m
                    ),
                    "global_to_detector_distance_m": (
                        self.global_to_detector_distance_m
                    ),
                },
                "phase": {
                    "parameterization": self.phase_parameterization,
                    "init": self.phase_init,
                    "init_std": self.phase_init_std,
                    "dropout_mode": self.phase_dropout_mode,
                    "dropout_p": self.phase_dropout_p,
                },
                "k_space": {
                    "enabled": self.k_space_constraint_enabled,
                    "theta_max_deg": self.theta_max_deg,
                },
                "oeo": {
                    "enabled": self.interlayer_enabled,
                    "per_expert_enabled": self.interlayer_per_expert_enabled,
                    "elementwise_affine": self.interlayer_elementwise_affine,
                    "hard_route_mask": self.interlayer_hard_route_mask,
                    "reapply_routing_weights": self.interlayer_reapply_routing_weights,
                    "preserve_response_amplitude": (
                        self.oeo_preserve_response_amplitude
                    ),
                    "response_gain_min": self.oeo_response_gain_min,
                    "response_gain_max": self.oeo_response_gain_max,
                    "detector_integration_factor": (
                        self.interlayer_detector_integration_factor
                    ),
                    "layernorm_eps": self.interlayer_layernorm_eps,
                    "nonlinearity": self.interlayer_nonlinearity,
                },
                "detector": {
                    "output_size": self.detector_output_size,
                    "layernorm_eps": self.detector_layernorm_eps,
                    "layernorm_affine": self.detector_layernorm_affine,
                    "layernorm_scope": self.detector_layernorm_scope,
                    "nonlinearity": self.detector_nonlinearity,
                },
            },
            "visualization": {"sample_count": self.visualization_sample_count},
            "output_dir": str(self.output_dir),
        }


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_config(config_path)
    base = config_path.parent
    d = lambda key, default=None: _nested(raw, key, default)
    settings = Settings(
        config_path=config_path,
        dataset_root=_resolve(d("dataset.dataset_root"), base),
        output_dir=_resolve(d("output_dir"), base),
        selected_classes=tuple(str(value) for value in d("dataset.selected_classes", [])),
        use_all_classes=bool(d("dataset.use_all_classes", False)),
        download=bool(d("dataset.download", True)),
        download_url=str(d("dataset.download_url")),
        download_timeout_sec=int(d("dataset.download_timeout_sec", 300)),
        delete_archive_after_extract=bool(
            d("dataset.delete_archive_after_extract", True)
        ),
        test_fraction=float(d("dataset.test_fraction", 0.20)),
        train_limit_per_class=d("dataset.train_limit_per_class"),
        test_limit_per_class=d("dataset.test_limit_per_class"),
        gallery_images_per_class=int(d("dataset.gallery_images_per_class", 3)),
        minimum_images_per_class=int(d("dataset.minimum_images_per_class", 10)),
        model_id=str(d("qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")),
        cache_dir=_resolve(d("qwen.cache_dir"), base),
        teacher_cache_source_path=_resolve(d("qwen.teacher_cache_source"), base),
        local_files_only=bool(d("qwen.local_files_only", False)),
        instruction=str(d("qwen.instruction")),
        processor_min_pixels=int(d("qwen.processor_min_pixels", 50176)),
        processor_max_pixels=int(d("qwen.processor_max_pixels", 50176)),
        dtype=str(d("qwen.dtype", "bfloat16")),
        attn_implementation=str(d("qwen.attn_implementation", "sdpa")),
        device=str(d("qwen.device", "cuda")),
        image_size=int(d("retrieval.image_size", 224)),
        embedding_dim=int(d("retrieval.embedding_dim", 64)),
        gallery_aggregation=str(d("retrieval.gallery_aggregation", "mean_prototype")),
        teacher_batch_size=int(d("batching.teacher_batch_size", 4)),
        batch_size=int(d("batching.batch_size", 40)),
        pk_classes_per_batch=int(d("batching.pk_classes_per_batch", 10)),
        pk_images_per_class=int(d("batching.pk_images_per_class", 4)),
        inference_batch_size=int(d("batching.inference_batch_size", 4)),
        num_workers=int(d("batching.num_workers", 4)),
        optimizer_steps_per_epoch=(
            None
            if d("batching.optimizer_steps_per_epoch") is None
            else int(d("batching.optimizer_steps_per_epoch"))
        ),
        epochs=int(d("training.epochs", 50)),
        learning_rate=float(d("training.learning_rate", 0.002)),
        adapter_learning_rate=(
            None
            if d("training.adapter_learning_rate") is None
            else float(d("training.adapter_learning_rate"))
        ),
        readout_learning_rate=(
            None
            if d("training.readout_learning_rate") is None
            else float(d("training.readout_learning_rate"))
        ),
        weight_decay=float(d("training.weight_decay", 0.0)),
        lambda_kd=float(d("training.lambda_kd", 1.0)),
        lambda_relational_kd=float(d("training.lambda_relational_kd", 0.0)),
        lambda_ret=float(d("training.lambda_ret", 1.0)),
        lambda_gallery=float(d("training.lambda_gallery", 0.0)),
        lambda_teacher_gallery=float(d("training.lambda_teacher_gallery", 0.0)),
        lambda_router_balance=float(d("training.lambda_router_balance", 0.0)),
        lambda_router_importance=float(d("training.lambda_router_importance", 0.0)),
        lambda_router_hard_load_balance=float(
            d("training.lambda_router_hard_load_balance", 0.0)
        ),
        router_load_bias_update_rate=float(
            d("training.router_load_bias_update_rate", 0.0)
        ),
        router_load_bias_update_start_epoch=int(
            d("training.router_load_bias_update_start_epoch", 1)
        ),
        lambda_router_response_consistency=float(
            d("training.lambda_router_response_consistency", 0.0)
        ),
        phase_dc_enabled=bool(
            d(
                "training.phase_dc.enabled",
                float(d("training.lambda_phase_dc", 0.0)) > 0.0,
            )
        ),
        lambda_phase_dc=float(
            d("training.phase_dc.weight", d("training.lambda_phase_dc", 0.0))
        ),
        phase_dc_start_epoch=int(
            d("training.phase_dc.start_epoch", d("training.phase_dc_start_epoch", 1))
        ),
        temperature=float(d("training.temperature", 0.07)),
        gallery_temperature=float(
            d("training.gallery_temperature", d("training.temperature", 0.07))
        ),
        gallery_prototype_stop_gradient=bool(
            d("training.gallery_prototype_stop_gradient", False)
        ),
        router_learning_rate=(
            None
            if d("training.router_learning_rate") is None
            else float(d("training.router_learning_rate"))
        ),
        phase_learning_rate=(
            None
            if d("training.phase_learning_rate") is None
            else float(d("training.phase_learning_rate"))
        ),
        gradient_clip_norm=(
            None
            if d("training.gradient_clip_norm") is None
            else float(d("training.gradient_clip_norm"))
        ),
        phase_focus_enabled=bool(d("training.phase_focus.enabled", False)),
        phase_focus_warmup_epochs=int(
            d("training.phase_focus.warmup_epochs", 5)
        ),
        phase_focus_interval_epochs=int(
            d("training.phase_focus.interval_epochs", 3)
        ),
        phase_gradient_measure_interval_batches=int(
            d("training.phase_diagnostics.gradient_measure_interval_batches", 10)
        ),
        phase_motion_warning_epoch=int(
            d("training.phase_diagnostics.motion_warning_epoch", 5)
        ),
        phase_motion_warning_threshold_rad=float(
            d("training.phase_diagnostics.motion_warning_threshold_rad", 0.01)
        ),
        phase_preview_interval_epochs=int(
            d("training.phase_diagnostics.preview_interval_epochs", 5)
        ),
        ema_decay=(
            None
            if d("training.ema_decay") is None
            else float(d("training.ema_decay"))
        ),
        resume_optimizer_state=bool(d("training.resume_optimizer_state", True)),
        random_seed=int(d("training.random_seed", 42)),
        amp_enabled=bool(d("training.amp_enabled", True)),
        evaluate_test_each_epoch=bool(d("training.evaluate_test_each_epoch", True)),
        log_interval_batches=int(d("training.log_interval_batches", 10)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.9)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.1)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.1)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        horizontal_flip_probability=float(
            d("augmentation.horizontal_flip_probability", 0.5)
        ),
        visualization_sample_count=int(d("visualization.sample_count", 8)),
        input_adapter_dim=int(d("optical.input_adapter_dim", 224)),
        max_visual_tokens=int(d("optical.max_visual_tokens", 224)),
        max_language_tokens=int(d("optical.max_language_tokens", 224)),
        vision_tap_stages=tuple(int(v) for v in d("optical.vision_tap_stages", [1])),
        student_language_mode="optical_moe",
        native_pre_attention_enabled=bool(d("optical.native_pre_attention_enabled", False)),
        native_pre_attention_initialize_from_teacher=bool(
            d("optical.initialize_attention_from_teacher", False)
        ),
        native_pre_attention_trainable=bool(d("optical.native_pre_attention_trainable", False)),
        transformer_residual_enabled=bool(d("optical.residual_enabled", True)),
        vision_attention_source_layer=0,
        language_attention_source_layer=0,
        canvas_size=int(d("optical.geometry.canvas_size", 1026)),
        active_size=int(d("optical.geometry.active_size", 986)),
        expert_size=int(d("optical.geometry.expert_size", 224)),
        expert_pitch=int(d("optical.geometry.expert_pitch", 254)),
        num_experts=int(d("optical.geometry.num_experts", 16)),
        expert_grid_rows=int(d("optical.geometry.grid_rows", 4)),
        expert_grid_cols=int(d("optical.geometry.grid_cols", 4)),
        expert_layers=int(d("optical.geometry.layers_per_expert", 1)),
        top_k=int(d("optical.router.top_k", 4)),
        router_pool_size=int(d("optical.router.pool_size", 14)),
        router_temperature=float(d("optical.router.temperature", 1.0)),
        router_input_layernorm_enabled=bool(d("optical.router.input_layernorm_enabled", True)),
        router_input_layernorm_eps=float(d("optical.router.input_layernorm_eps", 1e-5)),
        amplitude_slm_weight_domain=str(d("optical.router.amplitude_weight_domain", "amplitude")),
        amplitude_slm_input_normalization=str(d("optical.router.input_normalization", "none")),
        amplitude_phase_relay="ideal_4f_identity",
        wavelength_nm=float(d("optical.physics.wavelength_nm", 532.0)),
        pixel_pitch_um=float(d("optical.physics.pixel_pitch_um", 8.0)),
        expert_interlayer_distance_m=float(d("optical.physics.inter_layer_distance_m", 0.1)),
        last_expert_to_global_distance_m=float(
            d("optical.physics.last_expert_to_global_distance_m", 0.1)
        ),
        global_to_detector_distance_m=float(
            d("optical.physics.global_to_detector_distance_m", 0.1)
        ),
        phase_parameterization=str(d("optical.phase.parameterization", "sigmoid")),
        phase_init=str(d("optical.phase.init", "zeros")),
        phase_init_std=float(d("optical.phase.init_std", 0.02)),
        k_space_constraint_enabled=bool(d("optical.k_space.enabled", False)),
        theta_max_deg=float(d("optical.k_space.theta_max_deg", 1.0)),
        interlayer_enabled=bool(d("optical.oeo.enabled", True)),
        interlayer_per_expert_enabled=bool(d("optical.oeo.per_expert_enabled", True)),
        interlayer_elementwise_affine=bool(d("optical.oeo.elementwise_affine", False)),
        interlayer_hard_route_mask=bool(d("optical.oeo.hard_route_mask", True)),
        interlayer_reapply_routing_weights=bool(
            d("optical.oeo.reapply_routing_weights", True)
        ),
        oeo_preserve_response_amplitude=bool(
            d("optical.oeo.preserve_response_amplitude", False)
        ),
        oeo_response_gain_min=float(
            d("optical.oeo.response_gain_min", 0.25)
        ),
        oeo_response_gain_max=float(
            d("optical.oeo.response_gain_max", 4.0)
        ),
        interlayer_detector_integration_factor=int(
            d("optical.oeo.detector_integration_factor", 1)
        ),
        interlayer_layernorm_eps=float(d("optical.oeo.layernorm_eps", 1e-5)),
        interlayer_nonlinearity=str(d("optical.oeo.nonlinearity", "relu")),
        detector_output_size=int(d("optical.detector.output_size", 224)),
        detector_layernorm_eps=float(d("optical.detector.layernorm_eps", 1e-5)),
        detector_layernorm_affine=bool(d("optical.detector.layernorm_affine", False)),
        detector_layernorm_scope=str(d("optical.detector.layernorm_scope", "per_token")),
        detector_nonlinearity=str(d("optical.detector.nonlinearity", "relu")),
        phase_dropout_mode="none",
        phase_dropout_p=0.0,
        phase_dropout_block_size=4,
        phase_dropout_batch_shared=True,
    )
    if settings.dataset_root is None or settings.output_dir is None:
        raise ValueError("dataset.dataset_root and output_dir are required")
    settings.validate()
    return settings


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
