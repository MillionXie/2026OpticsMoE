from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    value = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (value if value.is_absolute() else base / value).resolve()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
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
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge(_read(parent_path, seen), raw)


@dataclass
class Settings:
    config_path: Path
    data_root: Path
    output_dir: Path
    download: bool
    lsp_urls: tuple[str, ...]
    lspet_urls: tuple[str, ...]
    lspet_expected_count: int
    visibility_policy: str
    strict_dataset_counts: bool
    train_limit: int | None
    test_limit: int | None

    model_id: str
    cache_dir: Path | None
    local_files_only: bool
    processor_min_pixels: int
    processor_max_pixels: int
    dtype: str
    attn_implementation: str
    device: str

    image_size: int
    heatmap_size: int
    heatmap_sigma: float
    teacher_batch_size: int
    student_batch_size: int
    inference_batch_size: int
    num_workers: int
    teacher_epochs: int
    student_epochs: int
    teacher_learning_rate: float
    student_learning_rate: float
    router_learning_rate: float
    phase_learning_rate: float
    weight_decay: float
    heatmap_loss_weight: float
    coordinate_loss_weight: float
    router_balance_weight: float
    router_importance_weight: float
    phase_dc_weight: float
    teacher_distill_weight: float
    teacher_distill_temperature: float
    teacher_distill_checkpoint: Path | None
    teacher_cache_path: Path | None
    teacher_cache_batch_size: int
    router_response_consistency_weight: float
    ema_decay: float
    tta_enabled: bool
    random_seed: int
    amp_enabled: bool
    log_interval_batches: int
    resume_student_checkpoint: Path | None
    reinit_router: bool
    reinit_head: bool

    augmentation_enabled: bool
    crop_margin: float
    crop_scale_jitter: float
    crop_center_jitter: float
    horizontal_flip_probability: float
    brightness_jitter: float
    contrast_jitter: float

    pose_projection_dim: int
    pose_decoder_channels: tuple[int, ...]
    pose_groupnorm_groups: int
    pose_head_mode: str
    visualization_sample_count: int

    # Interface consumed by the reused, validated Optical MoE16 core.
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
    router_noise_std: float
    router_gate_init_std: float
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
    interlayer_detector_integration_factor: int
    oeo_preserve_amplitude: bool
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
            raise ValueError("This experiment fixes the Qwen input at 224x224")
        if self.heatmap_size <= 0 or self.image_size % self.heatmap_size:
            raise ValueError("heatmap_size must be a positive divisor of image_size")
        if self.heatmap_sigma <= 0:
            raise ValueError("heatmap_sigma must be positive")
        if self.visibility_policy not in {
            "coordinates_in_image", "third_channel_zero_visible", "third_channel_one_visible"
        }:
            raise ValueError("Unsupported visibility_policy")
        if not self.lsp_urls or not self.lspet_urls:
            raise ValueError("Both LSP and LSPET download URL lists must be non-empty")
        if self.lspet_expected_count <= 0:
            raise ValueError("lspet_expected_count must be positive")
        for name in (
            "teacher_batch_size", "student_batch_size", "inference_batch_size",
            "teacher_epochs", "student_epochs", "log_interval_batches",
            "pose_projection_dim", "input_adapter_dim", "max_visual_tokens",
            "canvas_size", "active_size", "expert_size", "expert_pitch",
            "num_experts", "expert_grid_rows", "expert_grid_cols", "expert_layers",
            "top_k", "router_pool_size", "detector_output_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        if self.crop_margin < 1 or self.crop_scale_jitter < 0 or self.crop_center_jitter < 0:
            raise ValueError("Invalid person crop augmentation")
        if min(
            self.heatmap_loss_weight, self.coordinate_loss_weight,
            self.router_balance_weight, self.router_importance_weight,
            self.phase_dc_weight,
        ) < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.input_adapter_dim != 224 or self.expert_size != 224:
            raise ValueError("The reused Optical MoE16 requires 224 optical rows/channels")
        if self.max_visual_tokens > self.expert_size:
            raise ValueError("max_visual_tokens cannot exceed expert_size")
        expected = (1026, 986, 16, 4, 4, 1, 4, 224)
        actual = (
            self.canvas_size, self.active_size, self.num_experts,
            self.expert_grid_rows, self.expert_grid_cols, self.expert_layers,
            self.top_k, self.detector_output_size,
        )
        if actual != expected:
            raise ValueError(f"Validated MoE16 geometry must be {expected}, got {actual}")
        if self.vision_tap_stages != (1,):
            raise ValueError("One-layer optical pose baseline requires vision_tap_stages=[1]")
        if self.phase_dropout_mode != "none" or self.phase_dropout_p != 0:
            raise ValueError("Phase dropout is disabled in the reproducible baseline")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value
        return convert(asdict(self))


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _read(config_path)
    base = config_path.parent
    d = lambda key, default=None: _get(raw, key, default)
    settings = Settings(
        config_path=config_path,
        data_root=_resolve(d("dataset.data_root", "../../../data/lsp_pose"), base),
        output_dir=_resolve(d("output_dir", "../runs/lsp_pose_optical_moe16"), base),
        download=bool(d("dataset.download", True)),
        lsp_urls=tuple(str(v) for v in d("dataset.lsp_urls", [])),
        lspet_urls=tuple(str(v) for v in d("dataset.lspet_urls", [])),
        lspet_expected_count=int(d("dataset.lspet_expected_count", 9428)),
        visibility_policy=str(d("dataset.visibility_policy", "coordinates_in_image")),
        strict_dataset_counts=bool(d("dataset.strict_dataset_counts", True)),
        train_limit=d("dataset.train_limit"),
        test_limit=d("dataset.test_limit"),
        model_id=str(d("qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")),
        cache_dir=_resolve(d("qwen.cache_dir"), base),
        local_files_only=bool(d("qwen.local_files_only", False)),
        processor_min_pixels=int(d("qwen.processor_min_pixels", 50176)),
        processor_max_pixels=int(d("qwen.processor_max_pixels", 50176)),
        dtype=str(d("qwen.dtype", "bfloat16")),
        attn_implementation=str(d("qwen.attn_implementation", "sdpa")),
        device=str(d("qwen.device", "cuda")),
        image_size=int(d("data.image_size", 224)),
        heatmap_size=int(d("data.heatmap_size", 56)),
        heatmap_sigma=float(d("data.heatmap_sigma", 2.0)),
        teacher_batch_size=int(d("batching.teacher_batch_size", 8)),
        student_batch_size=int(d("batching.student_batch_size", 4)),
        inference_batch_size=int(d("batching.inference_batch_size", 8)),
        num_workers=int(d("batching.num_workers", 8)),
        teacher_epochs=int(d("training.teacher_epochs", 40)),
        student_epochs=int(d("training.student_epochs", 100)),
        teacher_learning_rate=float(d("training.teacher_learning_rate", 1e-3)),
        student_learning_rate=float(d("training.student_learning_rate", 1e-3)),
        router_learning_rate=float(d("training.router_learning_rate", 5e-4)),
        phase_learning_rate=float(d("training.phase_learning_rate", 1e-3)),
        weight_decay=float(d("training.weight_decay", 0.0)),
        heatmap_loss_weight=float(d("loss.heatmap_weight", 1.0)),
        coordinate_loss_weight=float(d("loss.coordinate_weight", 0.1)),
        router_balance_weight=float(d("loss.router_balance_weight", 0.03)),
        router_importance_weight=float(d("loss.router_importance_weight", 0.005)),
        phase_dc_weight=float(d("loss.phase_dc_weight", 0.0)),
        teacher_distill_weight=float(d("loss.teacher_distill_weight", 0.0)),
        teacher_distill_temperature=float(d("loss.teacher_distill_temperature", 4.0)),
        teacher_distill_checkpoint=_resolve(d("loss.teacher_distill_checkpoint"), base),
        teacher_cache_path=_resolve(d("loss.teacher_cache_path"), base),
        teacher_cache_batch_size=int(d("batching.teacher_cache_batch_size", 8)),
        router_response_consistency_weight=float(d("loss.router_response_consistency_weight", 0.0)),
        ema_decay=float(d("training.ema_decay", 0.0)),
        tta_enabled=bool(d("inference.tta_enabled", False)),
        random_seed=int(d("training.random_seed", 42)),
        amp_enabled=bool(d("training.amp_enabled", True)),
        log_interval_batches=int(d("training.log_interval_batches", 100)),
        resume_student_checkpoint=_resolve(d("training.resume_student_checkpoint"), base),
        reinit_router=bool(d("training.reinit_router", False)),
        reinit_head=bool(d("training.reinit_head", False)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_margin=float(d("augmentation.crop_margin", 1.25)),
        crop_scale_jitter=float(d("augmentation.scale_jitter", 0.15)),
        crop_center_jitter=float(d("augmentation.center_jitter", 0.05)),
        horizontal_flip_probability=float(d("augmentation.horizontal_flip_probability", 0.5)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.1)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.1)),
        pose_projection_dim=int(d("pose_head.projection_dim", 128)),
        pose_decoder_channels=tuple(int(v) for v in d("pose_head.decoder_channels", [128, 64])),
        pose_groupnorm_groups=int(d("pose_head.groupnorm_groups", 8)),
        pose_head_mode=str(d("pose_head.mode", "bilinear")),
        visualization_sample_count=int(d("visualization.sample_count", 16)),
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
        router_noise_std=float(d("optical.router.noise_std", 0.0)),
        router_gate_init_std=float(d("optical.router.gate_init_std", 0.01)),
        router_input_layernorm_enabled=bool(d("optical.router.input_layernorm_enabled", True)),
        router_input_layernorm_eps=float(d("optical.router.input_layernorm_eps", 1e-5)),
        amplitude_slm_weight_domain=str(d("optical.router.amplitude_weight_domain", "amplitude")),
        amplitude_slm_input_normalization=str(d("optical.router.input_normalization", "none")),
        amplitude_phase_relay="ideal_4f_identity",
        wavelength_nm=float(d("optical.physics.wavelength_nm", 532.0)),
        pixel_pitch_um=float(d("optical.physics.pixel_pitch_um", 8.0)),
        expert_interlayer_distance_m=float(d("optical.physics.inter_layer_distance_m", 0.1)),
        last_expert_to_global_distance_m=float(d("optical.physics.last_expert_to_global_distance_m", 0.1)),
        global_to_detector_distance_m=float(d("optical.physics.global_to_detector_distance_m", 0.1)),
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
        interlayer_detector_integration_factor=int(d("optical.oeo.detector_integration_factor", 1)),
        oeo_preserve_amplitude=bool(d("optical.oeo.preserve_amplitude", False)),
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
    from experiments.vision2_hybrid_dense.settings import (
        apply_vision2_hybrid_settings,
    )

    apply_vision2_hybrid_settings(settings, raw)
    return settings


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
