from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    config_path: Path
    dataset_root: Path
    dataset_variant: str
    download: bool
    download_url: str
    selected_skus: tuple[str, ...]
    merge_official_validation_into_train: bool
    train_limit_per_sku: int | None
    test_limit_per_sku: int | None
    gallery_images_per_sku: int
    model_id: str
    cache_dir: Path | None
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
    max_visual_tokens: int
    max_language_tokens: int
    teacher_batch_size: int
    pk_skus_per_batch: int
    pk_images_per_sku: int
    inference_batch_size: int
    num_workers: int
    optimizer_steps_per_epoch: int | None
    epochs: int
    learning_rate: float
    router_learning_rate: float
    phase_learning_rate: float
    weight_decay: float
    lambda_kd: float
    lambda_ret: float
    lambda_router_balance: float
    lambda_router_importance: float
    temperature: float
    gradient_clip_norm: float
    amp_enabled: bool
    evaluate_test_each_epoch: bool
    log_interval_batches: int
    random_seed: int
    augmentation_enabled: bool
    crop_scale_min: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float
    token_grid_rows: int
    token_grid_cols: int
    token_feature_side: int
    expert_grid_rows: int
    expert_grid_cols: int
    expert_gap: int
    token_group_gap: int
    propagation_padding: int
    share_expert_phase_across_tokens: bool
    top_k: int
    router_temperature: float
    router_layernorm_enabled: bool
    router_layernorm_affine: bool
    router_noise_std: float
    router_gate_init_std: float
    amplitude_weight_domain: str
    input_normalization: str
    input_nonlinearity: str
    input_amplitude_normalization: str
    wavelength_nm: float
    pixel_pitch_um: float
    propagation_distance_m: float
    second_plane_mode: str
    phase_parameterization: str
    phase_init: str
    phase_init_std: float
    phase_dropout_mode: str
    phase_dropout_p: float
    phase_dropout_block_size: int
    phase_dropout_batch_shared: bool
    k_space_enabled: bool
    theta_max_deg: float
    oeo_enabled: bool
    oeo_layernorm_eps: float
    oeo_elementwise_affine: bool
    oeo_nonlinearity: str
    oeo_reapply_routing_weights: bool
    oeo_hard_route_mask: bool
    final_layernorm_eps: float
    final_layernorm_affine: bool
    final_aggregation: str
    residual_enabled: bool
    residual_scale: float
    residual_scale_trainable: bool
    phase_dc_enabled: bool
    phase_dc_weight: float
    capture_intermediate_fields: bool
    visualization_sample_count: int
    output_dir: Path
    teacher_cache_path: Path

    @property
    def hidden_size(self) -> int:
        return self.token_feature_side * self.token_feature_side

    @property
    def num_experts(self) -> int:
        return self.expert_grid_rows * self.expert_grid_cols

    @property
    def max_tokens(self) -> int:
        return self.token_grid_rows * self.token_grid_cols

    @property
    def expert_group_height(self) -> int:
        return self.expert_grid_rows * self.token_feature_side + (self.expert_grid_rows - 1) * self.expert_gap

    @property
    def expert_group_width(self) -> int:
        return self.expert_grid_cols * self.token_feature_side + (self.expert_grid_cols - 1) * self.expert_gap

    @property
    def active_height(self) -> int:
        return self.token_grid_rows * self.expert_group_height + (self.token_grid_rows - 1) * self.token_group_gap

    @property
    def active_width(self) -> int:
        return self.token_grid_cols * self.expert_group_width + (self.token_grid_cols - 1) * self.token_group_gap

    @property
    def canvas_size(self) -> int:
        if self.active_height != self.active_width:
            raise ValueError("Angular-spectrum implementation requires a square active panel")
        return self.active_height + 2 * self.propagation_padding

    @property
    def subset_manifest_path(self) -> Path:
        return self.output_dir / "data" / "subset_manifest.csv"

    def validate(self) -> None:
        if self.hidden_size != 1024:
            raise ValueError(
                f"token_feature_side^2 must equal Qwen vision hidden size 1024, got {self.hidden_size}"
            )
        if self.max_visual_tokens != self.max_tokens:
            raise ValueError("max_visual_tokens must equal token_grid_rows * token_grid_cols")
        if self.num_experts < 1 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if min(
            self.token_grid_rows, self.token_grid_cols, self.token_feature_side,
            self.expert_grid_rows, self.expert_grid_cols,
        ) <= 0:
            raise ValueError("All token/expert layout dimensions must be positive")
        if min(self.expert_gap, self.token_group_gap, self.propagation_padding) < 0:
            raise ValueError("Layout gaps and propagation padding cannot be negative")
        if self.active_height != self.active_width:
            raise ValueError(
                f"Active panel is {self.active_height}x{self.active_width}; configure a square layout"
            )
        if self.second_plane_mode not in {"global", "expert"}:
            raise ValueError("second_plane_mode must be 'global' or 'expert'")
        if self.amplitude_weight_domain not in {"amplitude", "power"}:
            raise ValueError("amplitude_weight_domain must be amplitude or power")
        if self.input_normalization not in {"none", "layernorm"}:
            raise ValueError("input_normalization must be none or layernorm")
        if self.input_nonlinearity not in {"softplus", "relu", "absolute"}:
            raise ValueError("input_nonlinearity must be softplus, relu, or absolute")
        if self.input_amplitude_normalization not in {"none", "per_token_max", "per_token_rms"}:
            raise ValueError("Unsupported input_amplitude_normalization")
        if self.oeo_nonlinearity not in {"relu", "softplus"}:
            raise ValueError("oeo_nonlinearity must be relu or softplus")
        if self.final_aggregation not in {"routing_weighted_sum", "selected_mean"}:
            raise ValueError("Unsupported final_aggregation")
        if self.phase_parameterization not in {"sigmoid", "unconstrained"}:
            raise ValueError("Unsupported phase_parameterization")
        if self.phase_init not in {"zeros", "uniform", "normal"}:
            raise ValueError("phase_init must be zeros, uniform, or normal")
        if self.phase_dropout_mode not in {"none", "phase_bypass", "block_phase_bypass"}:
            raise ValueError("Unsupported phase_dropout_mode")
        if not 0.0 <= self.phase_dropout_p < 1.0:
            raise ValueError("phase_dropout_p must be in [0,1)")
        if self.propagation_distance_m < 0.0:
            raise ValueError("propagation_distance_m cannot be negative")
        if self.gallery_aggregation not in {"mean_prototype", "max_similarity"}:
            raise ValueError("Unsupported gallery_aggregation")
        if self.pk_images_per_sku < 2:
            raise ValueError("PK training needs at least two images per SKU")
        if not self.oeo_enabled:
            raise ValueError("The two-plane token-wise experiment requires OEO between the planes")

    def resolve_architecture(self, model: Any) -> None:
        visual = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
        if visual is None or not hasattr(visual, "blocks"):
            raise RuntimeError("Unable to locate Qwen visual.blocks")
        candidates = (
            getattr(getattr(visual, "config", None), "hidden_size", None),
            getattr(getattr(model, "config", None), "vision_config", None),
        )
        hidden = candidates[0]
        if hidden is None and candidates[1] is not None:
            hidden = getattr(candidates[1], "hidden_size", None)
        if hidden is not None and int(hidden) != self.hidden_size:
            raise RuntimeError(
                f"Qwen vision hidden size is {hidden}, but token field is {self.hidden_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update({
            "num_experts": self.num_experts,
            "max_tokens": self.max_tokens,
            "hidden_size": self.hidden_size,
            "expert_group_size": [self.expert_group_height, self.expert_group_width],
            "active_panel_size": [self.active_height, self.active_width],
            "canvas_size": self.canvas_size,
        })
        return _jsonable(values)


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml(config_path)
    base = config_path.parent
    d = lambda key, default=None: _get(raw, key, default)
    values = Settings(
        config_path=config_path,
        dataset_root=_path(d("dataset.dataset_root"), base),
        dataset_variant=str(d("dataset.variant", "selected10")),
        download=bool(d("dataset.download", True)),
        download_url=str(d("dataset.download_url")),
        selected_skus=tuple(str(v) for v in d("dataset.selected_skus", [])),
        merge_official_validation_into_train=bool(d("dataset.merge_official_validation_into_train", True)),
        train_limit_per_sku=d("dataset.train_limit_per_sku"),
        test_limit_per_sku=d("dataset.test_limit_per_sku"),
        gallery_images_per_sku=int(d("dataset.gallery_images_per_sku", 1)),
        model_id=str(d("qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")),
        cache_dir=_path(d("qwen.cache_dir"), base),
        local_files_only=bool(d("qwen.local_files_only", False)),
        instruction=str(d("qwen.instruction")),
        processor_min_pixels=int(d("qwen.processor_min_pixels", 38416)),
        processor_max_pixels=int(d("qwen.processor_max_pixels", 38416)),
        dtype=str(d("qwen.dtype", "bfloat16")),
        attn_implementation=str(d("qwen.attn_implementation", "sdpa")),
        device=str(d("qwen.device", "cuda")),
        image_size=int(d("retrieval.image_size", 196)),
        embedding_dim=int(d("retrieval.embedding_dim", 64)),
        gallery_aggregation=str(d("retrieval.gallery_aggregation", "mean_prototype")),
        max_visual_tokens=int(d("optical.layout.max_tokens", 196)),
        max_language_tokens=int(d("qwen.max_language_tokens", 512)),
        teacher_batch_size=int(d("batching.teacher_batch_size", 4)),
        pk_skus_per_batch=int(d("batching.pk_skus_per_batch", 4)),
        pk_images_per_sku=int(d("batching.pk_images_per_sku", 2)),
        inference_batch_size=int(d("batching.inference_batch_size", 1)),
        num_workers=int(d("batching.num_workers", 4)),
        optimizer_steps_per_epoch=d("batching.optimizer_steps_per_epoch"),
        epochs=int(d("training.epochs", 100)),
        learning_rate=float(d("training.learning_rate", 2e-4)),
        router_learning_rate=float(d("training.router_learning_rate", 2e-4)),
        phase_learning_rate=float(d("training.phase_learning_rate", 4e-3)),
        weight_decay=float(d("training.weight_decay", 0.0)),
        lambda_kd=float(d("training.lambda_kd", 1.0)),
        lambda_ret=float(d("training.lambda_ret", 1.0)),
        lambda_router_balance=float(d("training.lambda_router_balance", 0.03)),
        lambda_router_importance=float(d("training.lambda_router_importance", 0.0)),
        temperature=float(d("training.temperature", 0.07)),
        gradient_clip_norm=float(d("training.gradient_clip_norm", 1.0)),
        amp_enabled=bool(d("training.amp_enabled", True)),
        evaluate_test_each_epoch=bool(d("training.evaluate_test_each_epoch", True)),
        log_interval_batches=int(d("training.log_interval_batches", 5)),
        random_seed=int(d("training.random_seed", 42)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.9)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.1)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.1)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        token_grid_rows=int(d("optical.layout.token_grid_rows", 14)),
        token_grid_cols=int(d("optical.layout.token_grid_cols", 14)),
        token_feature_side=int(d("optical.layout.token_feature_side", 32)),
        expert_grid_rows=int(d("optical.layout.expert_grid_rows", 2)),
        expert_grid_cols=int(d("optical.layout.expert_grid_cols", 2)),
        expert_gap=int(d("optical.layout.expert_gap", 2)),
        token_group_gap=int(d("optical.layout.token_group_gap", 2)),
        propagation_padding=int(d("optical.layout.propagation_padding", 20)),
        share_expert_phase_across_tokens=bool(d("optical.layout.share_expert_phase_across_tokens", True)),
        top_k=int(d("optical.router.top_k", 2)),
        router_temperature=float(d("optical.router.temperature", 1.0)),
        router_layernorm_enabled=bool(d("optical.router.layernorm_enabled", True)),
        router_layernorm_affine=bool(d("optical.router.layernorm_affine", False)),
        router_noise_std=float(d("optical.router.noise_std", 0.0)),
        router_gate_init_std=float(d("optical.router.gate_init_std", 0.01)),
        amplitude_weight_domain=str(d("optical.router.amplitude_weight_domain", "amplitude")),
        input_normalization=str(d("optical.input.normalization", "layernorm")),
        input_nonlinearity=str(d("optical.input.nonlinearity", "softplus")),
        input_amplitude_normalization=str(d("optical.input.amplitude_normalization", "none")),
        wavelength_nm=float(d("optical.physics.wavelength_nm", 532.0)),
        pixel_pitch_um=float(d("optical.physics.pixel_pitch_um", 16.0)),
        propagation_distance_m=float(d("optical.physics.propagation_distance_m", 0.05)),
        second_plane_mode=str(d("optical.second_plane.mode", "global")),
        phase_parameterization=str(d("optical.phase.parameterization", "sigmoid")),
        phase_init=str(d("optical.phase.init", "zeros")),
        phase_init_std=float(d("optical.phase.init_std", 0.0)),
        phase_dropout_mode=str(d("optical.phase.dropout_mode", "none")),
        phase_dropout_p=float(d("optical.phase.dropout_p", 0.0)),
        phase_dropout_block_size=int(d("optical.phase.dropout_block_size", 4)),
        phase_dropout_batch_shared=bool(d("optical.phase.dropout_batch_shared", True)),
        k_space_enabled=bool(d("optical.k_space.enabled", False)),
        theta_max_deg=float(d("optical.k_space.theta_max_deg", 0.65)),
        oeo_enabled=bool(d("optical.oeo.enabled", True)),
        oeo_layernorm_eps=float(d("optical.oeo.layernorm_eps", 1e-5)),
        oeo_elementwise_affine=bool(d("optical.oeo.elementwise_affine", False)),
        oeo_nonlinearity=str(d("optical.oeo.nonlinearity", "relu")),
        oeo_reapply_routing_weights=bool(d("optical.oeo.reapply_routing_weights", True)),
        oeo_hard_route_mask=bool(d("optical.oeo.hard_route_mask", True)),
        final_layernorm_eps=float(d("optical.detector.layernorm_eps", 1e-5)),
        final_layernorm_affine=bool(d("optical.detector.layernorm_affine", False)),
        final_aggregation=str(d("optical.detector.aggregation", "routing_weighted_sum")),
        residual_enabled=bool(d("optical.residual.enabled", False)),
        residual_scale=float(d("optical.residual.initial_scale", 1.0)),
        residual_scale_trainable=bool(d("optical.residual.trainable_scale", False)),
        phase_dc_enabled=bool(d("regularization.phase_dc.enabled", False)),
        phase_dc_weight=float(d("regularization.phase_dc.weight", 0.0)),
        capture_intermediate_fields=bool(d("visualization.capture_intermediate_fields", False)),
        visualization_sample_count=int(d("visualization.sample_count", 4)),
        output_dir=_path(d("output_dir", "../runs/tokenwise_moe4"), base),
        teacher_cache_path=_path(d("teacher_cache_path", "../cache/teacher_embeddings.pt"), base),
    )
    if values.dataset_root is None or values.output_dir is None or values.teacher_cache_path is None:
        raise ValueError("dataset_root, output_dir, and teacher_cache_path are required")
    values.validate()
    return values


def _load_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic base_config reference at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = raw.pop("base_config", None)
    if base_name is None:
        return raw
    base_path = Path(base_name).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _deep_merge(_load_yaml(base_path, seen), raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "resolved_config.json").write_text(
        json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value
