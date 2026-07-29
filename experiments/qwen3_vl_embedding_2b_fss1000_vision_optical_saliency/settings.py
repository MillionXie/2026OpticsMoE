from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]


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
    parent = raw.pop("base_config", None)
    if parent is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), raw)


@dataclass
class Settings:
    config_path: Path
    data_root: Path
    output_dir: Path
    download: bool
    download_source: str
    download_file_id: str
    huggingface_dataset_id: str
    huggingface_endpoint: str
    official_test_list_url: str
    merge_official_validation_into_train: bool
    train_class_limit: int | None
    test_class_limit: int | None
    images_per_class_limit: int | None

    model_id: str
    cache_dir: Path | None
    local_files_only: bool
    processor_min_pixels: int
    processor_max_pixels: int
    dtype: str
    attn_implementation: str
    device: str

    image_size: int
    teacher_batch_size: int
    student_batch_size: int
    inference_batch_size: int
    num_workers: int
    teacher_epochs: int
    student_epochs: int
    teacher_learning_rate: float
    student_learning_rate: float
    router_learning_rate: float | None
    phase_learning_rate: float | None
    student_lr_schedule: str
    student_lr_min_ratio: float
    checkpoint_interval_epochs: int
    weight_decay: float
    bce_weight: float
    dice_weight: float
    soft_iou_weight: float
    boundary_weight: float
    router_balance_weight: float
    router_importance_weight: float
    mask_kd_weight: float
    mask_kd_temperature: float
    teacher_checkpoint: Path | None
    teacher_mask_cache: Path | None
    mask_kd_align_augmentation: bool
    student_initial_checkpoint: Path | None
    student_expand_expert_layers: bool
    random_seed: int
    amp_enabled: bool
    evaluate_test_each_epoch: bool
    log_interval_batches: int

    augmentation_enabled: bool
    crop_scale_min: float
    horizontal_flip_probability: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float
    visualization_sample_count: int

    segmentation_projection_dim: int
    segmentation_channels: tuple[int, ...]
    segmentation_groupnorm_groups: int
    student_segmentation_refinement_enabled: bool
    student_segmentation_progressive_refinement_enabled: bool
    student_detector_residual_enabled: bool
    student_detector_residual_source: str
    student_detector_identity_scale_init: float
    student_detector_input_scale_init: float
    student_detector_identity_scale_trainable: bool
    student_detector_input_scale_trainable: bool
    freeze_student_optical_core: bool
    freeze_student_base_head: bool

    # Exact one-stage MoE16 interface expected by the reused optical core.
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

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)

    def validate(self) -> None:
        if self.image_size != 224:
            raise ValueError("This experiment fixes RGB images and masks at 224x224")
        if self.download_source not in {"auto", "huggingface", "google_drive"}:
            raise ValueError("dataset.download_source must be auto, huggingface, or google_drive")
        if not self.merge_official_validation_into_train:
            raise ValueError(
                "This no-validation experiment requires merge_official_validation_into_train=true"
            )
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be explicit positive integers")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
        for name in (
            "teacher_batch_size", "student_batch_size", "inference_batch_size",
            "teacher_epochs", "student_epochs", "log_interval_batches",
            "segmentation_projection_dim", "input_adapter_dim", "max_visual_tokens",
            "canvas_size", "active_size", "expert_size", "expert_pitch", "num_experts",
            "expert_grid_rows", "expert_grid_cols", "expert_layers", "top_k",
            "router_pool_size", "detector_output_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.student_lr_schedule not in {"constant", "cosine"}:
            raise ValueError("student_lr_schedule must be constant or cosine")
        if not 0.0 <= self.student_lr_min_ratio <= 1.0:
            raise ValueError("student_lr_min_ratio must be in [0, 1]")
        if self.checkpoint_interval_epochs < 0:
            raise ValueError("checkpoint_interval_epochs cannot be negative")
        if self.input_adapter_dim != 224 or self.expert_size != 224:
            raise ValueError("The reused Optical MoE16 requires 224 optical rows/channels")
        if self.max_visual_tokens > self.expert_size:
            raise ValueError("max_visual_tokens cannot exceed expert_size")
        if self.detector_output_size != 224:
            raise ValueError("The physical CCD readout is fixed at 224x224")
        expected = (1026, 986, 16, 4, 4, 4)
        actual = (
            self.canvas_size, self.active_size, self.num_experts,
            self.expert_grid_rows, self.expert_grid_cols, self.top_k,
        )
        if actual != expected:
            raise ValueError(
                "This experiment deliberately reuses the validated MoE16 geometry; "
                f"expected {expected}, got {actual}"
            )
        if self.vision_tap_stages != (1,):
            raise ValueError(
                "The saliency student exposes one final optical vision output, "
                "so vision_tap_stages must remain [1]"
            )
        if self.student_expand_expert_layers:
            if self.student_initial_checkpoint is None:
                raise ValueError(
                    "Expert-layer expansion requires student_initialization.checkpoint"
                )
            if self.expert_layers <= 1:
                raise ValueError(
                    "Expert-layer expansion requires layers_per_expert > 1"
                )
        if self.phase_dropout_mode != "none" or self.phase_dropout_p != 0.0:
            raise ValueError("Phase dropout is disabled in the first saliency baseline")
        if not self.segmentation_channels or any(v <= 0 for v in self.segmentation_channels):
            raise ValueError("segmentation_channels must contain positive values")
        if self.segmentation_groupnorm_groups <= 0:
            raise ValueError("segmentation_groupnorm_groups must be positive")
        if (
            self.student_segmentation_refinement_enabled
            and self.student_segmentation_progressive_refinement_enabled
        ):
            raise ValueError(
                "Enable either the local or progressive student refinement branch, "
                "not both"
            )
        for name in (
            "student_detector_identity_scale_init",
            "student_detector_input_scale_init",
        ):
            value = float(getattr(self, name))
            if not (-1.0e6 < value < 1.0e6):
                raise ValueError(f"{name} must be finite")
        if self.student_detector_residual_source not in {
            "nonnegative_input_field",
            "signed_adapter_latent",
        }:
            raise ValueError(
                "segmentation_head.student_detector_residual.source must be "
                "nonnegative_input_field or signed_adapter_latent"
            )
        if min(
            self.bce_weight, self.dice_weight, self.router_balance_weight,
            self.soft_iou_weight, self.boundary_weight,
            self.router_importance_weight, self.mask_kd_weight,
        ) < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.bce_weight + self.dice_weight <= 0:
            raise ValueError("At least one ground-truth segmentation loss must be enabled")
        if self.mask_kd_temperature <= 0:
            raise ValueError("mask_kd_temperature must be positive")
        if (
            self.mask_kd_weight > 0
            and self.augmentation_enabled
            and not self.mask_kd_align_augmentation
        ):
            raise ValueError(
                "Cached teacher mask KD with augmentation requires "
                "mask_kd.align_augmentation=true so teacher logits and ground-truth "
                "pixels remain geometrically aligned"
            )
        if (
            self.mask_kd_weight > 0
            and self.augmentation_enabled
            and self.mask_kd_align_augmentation
            and self.rotation_degrees != 0.0
        ):
            raise ValueError(
                "Aligned cached mask KD currently supports normalized crop and "
                "horizontal flip; set augmentation.rotation_degrees=0"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)

        def convert(item: Any) -> Any:
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, tuple):
                return [convert(v) for v in item]
            if isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            return item

        return convert(value)


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_config(config_path)
    base = config_path.parent
    d = lambda key, default=None: _nested(raw, key, default)
    settings = Settings(
        config_path=config_path,
        data_root=_resolve(d("dataset.data_root"), base),
        output_dir=_resolve(d("output_dir"), base),
        download=bool(d("dataset.download", True)),
        download_source=str(d("dataset.download_source", "auto")),
        download_file_id=str(d("dataset.download_file_id", "16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI")),
        huggingface_dataset_id=str(d("dataset.huggingface_dataset_id", "nobg/FSS-1000")),
        huggingface_endpoint=str(d("dataset.huggingface_endpoint", "https://hf-mirror.com")),
        official_test_list_url=str(d(
            "dataset.official_test_list_url",
            "https://raw.githubusercontent.com/HKUSTCV/FSS-1000/master/fss_test_set.txt",
        )),
        merge_official_validation_into_train=bool(
            d("dataset.merge_official_validation_into_train", True)
        ),
        train_class_limit=d("dataset.train_class_limit"),
        test_class_limit=d("dataset.test_class_limit"),
        images_per_class_limit=d("dataset.images_per_class_limit"),
        model_id=str(d("qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")),
        cache_dir=_resolve(d("qwen.cache_dir"), base),
        local_files_only=bool(d("qwen.local_files_only", False)),
        processor_min_pixels=int(d("qwen.processor_min_pixels", 50176)),
        processor_max_pixels=int(d("qwen.processor_max_pixels", 50176)),
        dtype=str(d("qwen.dtype", "bfloat16")),
        attn_implementation=str(d("qwen.attn_implementation", "sdpa")),
        device=str(d("qwen.device", "cuda")),
        image_size=int(d("data.image_size", 224)),
        teacher_batch_size=int(d("batching.teacher_batch_size", 4)),
        student_batch_size=int(d("batching.student_batch_size", 2)),
        inference_batch_size=int(d("batching.inference_batch_size", 4)),
        num_workers=int(d("batching.num_workers", 4)),
        teacher_epochs=int(d("training.teacher_epochs", 30)),
        student_epochs=int(d("training.student_epochs", 50)),
        teacher_learning_rate=float(d("training.teacher_learning_rate", 1e-3)),
        student_learning_rate=float(d("training.student_learning_rate", 2e-3)),
        router_learning_rate=(
            None if d("training.router_learning_rate") is None
            else float(d("training.router_learning_rate"))
        ),
        phase_learning_rate=(
            None if d("training.phase_learning_rate") is None
            else float(d("training.phase_learning_rate"))
        ),
        student_lr_schedule=str(d("training.student_lr_schedule", "constant")),
        student_lr_min_ratio=float(d("training.student_lr_min_ratio", 0.0)),
        checkpoint_interval_epochs=int(
            d("training.checkpoint_interval_epochs", 0)
        ),
        weight_decay=float(d("training.weight_decay", 0.0)),
        bce_weight=float(d("loss.bce_weight", 1.0)),
        dice_weight=float(d("loss.dice_weight", 1.0)),
        soft_iou_weight=float(d("loss.soft_iou_weight", 0.0)),
        boundary_weight=float(d("loss.boundary_weight", 0.0)),
        router_balance_weight=float(d("loss.router_balance_weight", 0.03)),
        router_importance_weight=float(d("loss.router_importance_weight", 0.0)),
        mask_kd_weight=float(d("loss.mask_kd_weight", 0.0)),
        mask_kd_temperature=float(d("loss.mask_kd_temperature", 1.0)),
        teacher_checkpoint=_resolve(d("mask_kd.teacher_checkpoint"), base),
        teacher_mask_cache=_resolve(d("mask_kd.teacher_mask_cache"), base),
        mask_kd_align_augmentation=bool(
            d("mask_kd.align_augmentation", False)
        ),
        student_initial_checkpoint=_resolve(
            d("student_initialization.checkpoint"), base
        ),
        student_expand_expert_layers=bool(
            d("student_initialization.expand_expert_layers", False)
        ),
        random_seed=int(d("training.random_seed", 42)),
        amp_enabled=bool(d("training.amp_enabled", True)),
        evaluate_test_each_epoch=bool(d("training.evaluate_test_each_epoch", True)),
        log_interval_batches=int(d("training.log_interval_batches", 25)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.9)),
        horizontal_flip_probability=float(d("augmentation.horizontal_flip_probability", 0.5)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.1)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.1)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        visualization_sample_count=int(d("visualization.sample_count", 12)),
        segmentation_projection_dim=int(d("segmentation_head.projection_dim", 128)),
        segmentation_channels=tuple(int(v) for v in d(
            "segmentation_head.decoder_channels", [64, 32, 16]
        )),
        segmentation_groupnorm_groups=int(d("segmentation_head.groupnorm_groups", 8)),
        student_segmentation_refinement_enabled=bool(
            d("segmentation_head.student_refinement_enabled", False)
        ),
        student_segmentation_progressive_refinement_enabled=bool(
            d("segmentation_head.student_progressive_refinement_enabled", False)
        ),
        student_detector_residual_enabled=bool(
            d("segmentation_head.student_detector_residual.enabled", False)
        ),
        student_detector_residual_source=str(
            d(
                "segmentation_head.student_detector_residual.source",
                "nonnegative_input_field",
            )
        ),
        student_detector_identity_scale_init=float(
            d("segmentation_head.student_detector_residual.identity_scale_init", 1.0)
        ),
        student_detector_input_scale_init=float(
            d("segmentation_head.student_detector_residual.input_scale_init", 0.1)
        ),
        student_detector_identity_scale_trainable=bool(
            d(
                "segmentation_head.student_detector_residual."
                "identity_scale_trainable",
                False,
            )
        ),
        student_detector_input_scale_trainable=bool(
            d(
                "segmentation_head.student_detector_residual.input_scale_trainable",
                True,
            )
        ),
        freeze_student_optical_core=bool(
            d("training.freeze_student_optical_core", False)
        ),
        freeze_student_base_head=bool(
            d("training.freeze_student_base_head", False)
        ),
        input_adapter_dim=int(d("optical.input_adapter_dim", 224)),
        max_visual_tokens=int(d("optical.max_visual_tokens", 224)),
        max_language_tokens=224,
        vision_tap_stages=tuple(int(v) for v in d("optical.vision_tap_stages", [1])),
        student_language_mode="disabled",
        native_pre_attention_enabled=False,
        native_pre_attention_initialize_from_teacher=False,
        native_pre_attention_trainable=False,
        transformer_residual_enabled=False,
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
        last_expert_to_global_distance_m=float(d(
            "optical.physics.last_expert_to_global_distance_m", 0.1
        )),
        global_to_detector_distance_m=float(d(
            "optical.physics.global_to_detector_distance_m", 0.1
        )),
        phase_parameterization=str(d("optical.phase.parameterization", "sigmoid")),
        phase_init=str(d("optical.phase.init", "zeros")),
        phase_init_std=float(d("optical.phase.init_std", 0.02)),
        k_space_constraint_enabled=bool(d("optical.k_space.enabled", False)),
        theta_max_deg=float(d("optical.k_space.theta_max_deg", 1.0)),
        interlayer_enabled=bool(d("optical.oeo.enabled", True)),
        interlayer_per_expert_enabled=bool(d("optical.oeo.per_expert_enabled", True)),
        interlayer_elementwise_affine=bool(d("optical.oeo.elementwise_affine", False)),
        interlayer_hard_route_mask=bool(d("optical.oeo.hard_route_mask", True)),
        interlayer_reapply_routing_weights=bool(d("optical.oeo.reapply_routing_weights", True)),
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
    if settings.data_root is None or settings.output_dir is None:
        raise ValueError("dataset.data_root and output_dir are required")
    settings.validate()
    return settings


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
