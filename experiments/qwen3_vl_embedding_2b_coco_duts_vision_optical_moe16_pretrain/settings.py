from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    config_path: Path

    # Data and persistent caches.
    coco_root: Path
    duts_root: Path
    cache_root: Path
    output_dir: Path
    auto_download: bool
    remove_archives_after_extract: bool
    coco_train_url: str
    coco_val_url: str
    duts_train_url: str
    duts_test_url: str
    coco_train_limit: int | None
    coco_val_limit: int | None
    duts_train_limit: int | None
    duts_test_limit: int | None
    image_size: int
    coco_resize_mode: str

    # Frozen Qwen Vision.
    model_id: str
    cache_dir: Path | None
    local_files_only: bool
    processor_min_pixels: int
    processor_max_pixels: int
    dtype: str
    attn_implementation: str
    device: str

    # PCA and teacher target cache.
    pca_rank: int
    pca_calibration_images: int
    pca_tokens_per_image: int
    pca_max_tokens: int
    pca_oversample: int
    pca_niter: int
    pca_device: str
    pca_path: Path
    teacher_cache_root: Path
    teacher_cache_shard_size: int
    teacher_cache_lru_shards: int
    teacher_cache_dtype: str
    rebuild_teacher_cache: bool

    # Batching/runtime.
    pca_batch_size: int
    teacher_cache_batch_size: int
    coco_batch_size: int
    duts_batch_size: int
    inference_batch_size: int
    num_workers: int
    log_interval_batches: int
    amp_enabled: bool
    random_seed: int

    # COCO feature distillation.
    coco_epochs: int
    coco_optical_learning_rate: float
    coco_phase_learning_rate: float
    coco_router_learning_rate: float
    coco_recombiner_learning_rate: float
    coco_weight_decay: float
    cosine_loss_weight: float
    smooth_l1_loss_weight: float
    smooth_l1_beta: float
    router_balance_weight: float
    router_importance_weight: float
    phase_dc_weight: float
    coco_checkpoint: Path

    # DUTS segmentation pretraining.
    duts_head_warmup_epochs: int
    duts_finetune_epochs: int
    duts_optical_learning_rate: float
    duts_phase_learning_rate: float
    duts_recombiner_learning_rate: float
    duts_head_learning_rate: float
    duts_weight_decay: float
    bce_weight: float
    dice_weight: float
    soft_iou_weight: float
    boundary_weight: float
    evaluate_duts_test_each_epoch: bool
    checkpoint_interval_epochs: int

    # Paired segmentation augmentation.
    augmentation_enabled: bool
    crop_scale_min: float
    horizontal_flip_probability: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float

    # Shared CCD feature recombination and downstream head.
    recombiner_alpha_init: float
    recombiner_alpha_trainable: bool
    recombiner_layernorm_affine: bool
    recombiner_layernorm_eps: float
    segmentation_projection_dim: int
    segmentation_channels: tuple[int, ...]
    segmentation_groupnorm_groups: int
    visualization_interval_epochs: int
    visualization_sample_count: int

    # Fields consumed by the validated MoE16 optical core.
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

    @property
    def duts_total_epochs(self) -> int:
        return self.duts_head_warmup_epochs + self.duts_finetune_epochs

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        if self.vision_hidden_size != 1024:
            raise RuntimeError(
                "This experiment was designed for Qwen3-VL-Embedding-2B "
                f"Vision hidden size 1024, got {self.vision_hidden_size}"
            )

    def validate(self) -> None:
        if self.image_size != 224:
            raise ValueError("This experiment deliberately keeps 224x224 inputs")
        if self.processor_min_pixels != 50176 or self.processor_max_pixels != 50176:
            raise ValueError("Qwen processor pixel budget must remain 224*224=50176")
        if self.pca_rank != 224:
            raise ValueError("Teacher PCA rank and optical feature width must both be 224")
        if self.input_adapter_dim != 224 or self.detector_output_size != 224:
            raise ValueError("Optical adapter and CCD readout must both be 224")
        if self.max_visual_tokens > 224:
            raise ValueError("max_visual_tokens cannot exceed 224")
        geometry = (
            self.canvas_size,
            self.active_size,
            self.expert_size,
            self.expert_pitch,
            self.num_experts,
            self.expert_grid_rows,
            self.expert_grid_cols,
        )
        expected_geometry = (1026, 986, 224, 254, 16, 4, 4)
        if geometry != expected_geometry:
            raise ValueError(
                f"Expected validated MoE16 geometry {expected_geometry}, got {geometry}"
            )
        if self.expert_layers != 3:
            raise ValueError("The shared optical backbone must contain exactly 3 expert stages")
        if self.top_k != 4:
            raise ValueError("The shared electronic router must select Top-4 experts")
        if not self.interlayer_enabled:
            raise ValueError("Three optical stages require OEO reload between stage planes")
        if not self.interlayer_per_expert_enabled:
            raise ValueError("OEO normalization must be independent per selected expert")
        if not self.interlayer_hard_route_mask:
            raise ValueError("Unselected expert fields must remain zero at OEO reload")
        if not self.interlayer_reapply_routing_weights:
            raise ValueError("The original input-dependent routing weights must be reapplied")
        if self.phase_dropout_mode != "none" or self.phase_dropout_p != 0.0:
            raise ValueError("Phase dropout is disabled for the initial pretraining baseline")
        if self.coco_resize_mode not in {"center_crop", "stretch"}:
            raise ValueError("dataset.coco_resize_mode must be center_crop or stretch")
        if self.teacher_cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache.teacher_dtype must be float16 or float32")
        positive_ints = (
            "pca_calibration_images",
            "pca_tokens_per_image",
            "pca_max_tokens",
            "pca_batch_size",
            "teacher_cache_batch_size",
            "teacher_cache_shard_size",
            "teacher_cache_lru_shards",
            "coco_batch_size",
            "duts_batch_size",
            "inference_batch_size",
            "coco_epochs",
            "duts_head_warmup_epochs",
            "duts_finetune_epochs",
            "log_interval_batches",
            "visualization_interval_epochs",
            "visualization_sample_count",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.pca_max_tokens < self.pca_rank:
            raise ValueError("pca.max_tokens must be at least pca.rank")
        if self.pca_tokens_per_image > self.max_visual_tokens:
            raise ValueError("pca.tokens_per_image cannot exceed max_visual_tokens")
        for name in (
            "coco_optical_learning_rate",
            "coco_phase_learning_rate",
            "coco_router_learning_rate",
            "coco_recombiner_learning_rate",
            "duts_optical_learning_rate",
            "duts_phase_learning_rate",
            "duts_recombiner_learning_rate",
            "duts_head_learning_rate",
            "smooth_l1_beta",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "cosine_loss_weight",
            "smooth_l1_loss_weight",
            "router_balance_weight",
            "router_importance_weight",
            "phase_dc_weight",
            "bce_weight",
            "dice_weight",
            "soft_iou_weight",
            "boundary_weight",
            "coco_weight_decay",
            "duts_weight_decay",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cosine_loss_weight + self.smooth_l1_loss_weight <= 0:
            raise ValueError("At least one COCO feature loss must be enabled")
        if self.bce_weight + self.dice_weight <= 0:
            raise ValueError("At least one DUTS segmentation loss must be enabled")
        if not self.segmentation_channels or min(self.segmentation_channels) <= 0:
            raise ValueError("segmentation decoder channels must be positive")
        if self.recombiner_layernorm_eps <= 0:
            raise ValueError("recombiner LayerNorm epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = config_path.parent

    def d(key: str, default: Any = None) -> Any:
        value: Any = raw
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def resolve(value: str | Path | None, default: str | None = None) -> Path | None:
        selected = value if value is not None else default
        if selected is None:
            return None
        result = Path(selected).expanduser()
        return (base / result).resolve() if not result.is_absolute() else result.resolve()

    cache_root = resolve(
        d("cache.root"),
        "../../../data/cache/qwen3_vl_embedding_2b_coco_duts_moe16",
    )
    output_dir = resolve(
        d("output_dir"),
        "../runs/coco_duts_vision_optical_moe16_pretrain",
    )
    assert cache_root is not None and output_dir is not None
    pca_path = resolve(d("pca.path"), str(cache_root / "pca_vision_1024_to_224.pt"))
    teacher_cache_root = resolve(
        d("cache.teacher_target_root"),
        str(cache_root / "coco_teacher_pca224"),
    )
    coco_checkpoint = resolve(
        d("duts.initial_backbone_checkpoint"),
        str(output_dir / "checkpoints" / "coco_backbone_best_train_loss.pt"),
    )
    assert pca_path is not None and teacher_cache_root is not None
    assert coco_checkpoint is not None

    settings = Settings(
        config_path=config_path,
        coco_root=resolve(d("dataset.coco_root"), "../../../data/COCO2017"),  # type: ignore[arg-type]
        duts_root=resolve(d("dataset.duts_root"), "../../../data/DUTS"),  # type: ignore[arg-type]
        cache_root=cache_root,
        output_dir=output_dir,
        auto_download=bool(d("dataset.auto_download", True)),
        remove_archives_after_extract=bool(
            d("dataset.remove_archives_after_extract", True)
        ),
        coco_train_url=str(
            d("dataset.coco_train_url", "http://images.cocodataset.org/zips/train2017.zip")
        ),
        coco_val_url=str(
            d("dataset.coco_val_url", "http://images.cocodataset.org/zips/val2017.zip")
        ),
        duts_train_url=str(
            d("dataset.duts_train_url", "https://saliencydetection.net/duts/download/DUTS-TR.zip")
        ),
        duts_test_url=str(
            d("dataset.duts_test_url", "https://saliencydetection.net/duts/download/DUTS-TE.zip")
        ),
        coco_train_limit=_optional_int(d("dataset.coco_train_limit")),
        coco_val_limit=_optional_int(d("dataset.coco_val_limit")),
        duts_train_limit=_optional_int(d("dataset.duts_train_limit")),
        duts_test_limit=_optional_int(d("dataset.duts_test_limit")),
        image_size=int(d("dataset.image_size", 224)),
        coco_resize_mode=str(d("dataset.coco_resize_mode", "center_crop")),
        model_id=str(d("qwen.model_id", "Qwen/Qwen3-VL-Embedding-2B")),
        cache_dir=resolve(d("qwen.cache_dir")),
        local_files_only=bool(d("qwen.local_files_only", False)),
        processor_min_pixels=int(d("qwen.processor_min_pixels", 50176)),
        processor_max_pixels=int(d("qwen.processor_max_pixels", 50176)),
        dtype=str(d("qwen.dtype", "bfloat16")),
        attn_implementation=str(d("qwen.attn_implementation", "sdpa")),
        device=str(d("qwen.device", "cuda")),
        pca_rank=int(d("pca.rank", 224)),
        pca_calibration_images=int(d("pca.calibration_images", 2048)),
        pca_tokens_per_image=int(d("pca.tokens_per_image", 32)),
        pca_max_tokens=int(d("pca.max_tokens", 65536)),
        pca_oversample=int(d("pca.oversample", 32)),
        pca_niter=int(d("pca.niter", 4)),
        pca_device=str(d("pca.device", "cuda")),
        pca_path=pca_path,
        teacher_cache_root=teacher_cache_root,
        teacher_cache_shard_size=int(d("cache.teacher_shard_size", 128)),
        teacher_cache_lru_shards=int(d("cache.teacher_lru_shards", 4)),
        teacher_cache_dtype=str(d("cache.teacher_dtype", "float16")),
        rebuild_teacher_cache=bool(d("cache.rebuild", False)),
        pca_batch_size=int(d("batching.pca_batch_size", 4)),
        teacher_cache_batch_size=int(d("batching.teacher_cache_batch_size", 4)),
        coco_batch_size=int(d("batching.coco_batch_size", 4)),
        duts_batch_size=int(d("batching.duts_batch_size", 4)),
        inference_batch_size=int(d("batching.inference_batch_size", 4)),
        num_workers=int(d("batching.num_workers", 8)),
        log_interval_batches=int(d("runtime.log_interval_batches", 250)),
        amp_enabled=bool(d("runtime.amp_enabled", True)),
        random_seed=int(d("runtime.random_seed", 42)),
        coco_epochs=int(d("coco_training.epochs", 10)),
        coco_optical_learning_rate=float(
            d("coco_training.optical_learning_rate", 2e-4)
        ),
        coco_phase_learning_rate=float(
            d("coco_training.phase_learning_rate", 4e-3)
        ),
        coco_router_learning_rate=float(
            d("coco_training.router_learning_rate", 5e-4)
        ),
        coco_recombiner_learning_rate=float(
            d("coco_training.recombiner_learning_rate", 5e-4)
        ),
        coco_weight_decay=float(d("coco_training.weight_decay", 0.0)),
        cosine_loss_weight=float(d("coco_loss.cosine_weight", 1.0)),
        smooth_l1_loss_weight=float(d("coco_loss.smooth_l1_weight", 0.5)),
        smooth_l1_beta=float(d("coco_loss.smooth_l1_beta", 0.1)),
        router_balance_weight=float(d("coco_loss.router_balance_weight", 0.03)),
        router_importance_weight=float(
            d("coco_loss.router_importance_weight", 0.0)
        ),
        phase_dc_weight=float(d("optical.phase.dc_loss_weight", 0.0)),
        coco_checkpoint=coco_checkpoint,
        duts_head_warmup_epochs=int(d("duts_training.head_warmup_epochs", 5)),
        duts_finetune_epochs=int(d("duts_training.finetune_epochs", 50)),
        duts_optical_learning_rate=float(
            d("duts_training.optical_learning_rate", 1e-4)
        ),
        duts_phase_learning_rate=float(
            d("duts_training.phase_learning_rate", 2e-3)
        ),
        duts_recombiner_learning_rate=float(
            d("duts_training.recombiner_learning_rate", 2e-4)
        ),
        duts_head_learning_rate=float(
            d("duts_training.head_learning_rate", 1e-3)
        ),
        duts_weight_decay=float(d("duts_training.weight_decay", 0.0)),
        bce_weight=float(d("duts_loss.bce_weight", 1.0)),
        dice_weight=float(d("duts_loss.dice_weight", 1.0)),
        soft_iou_weight=float(d("duts_loss.soft_iou_weight", 0.75)),
        boundary_weight=float(d("duts_loss.boundary_weight", 0.25)),
        evaluate_duts_test_each_epoch=bool(
            d("duts_training.evaluate_test_each_epoch", True)
        ),
        checkpoint_interval_epochs=int(
            d("duts_training.checkpoint_interval_epochs", 10)
        ),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.9)),
        horizontal_flip_probability=float(
            d("augmentation.horizontal_flip_probability", 0.5)
        ),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.1)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.1)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        recombiner_alpha_init=float(d("recombiner.alpha_init", 0.1)),
        recombiner_alpha_trainable=bool(d("recombiner.alpha_trainable", True)),
        recombiner_layernorm_affine=bool(
            d("recombiner.layernorm_affine", True)
        ),
        recombiner_layernorm_eps=float(d("recombiner.layernorm_eps", 1e-5)),
        segmentation_projection_dim=int(
            d("segmentation_head.projection_dim", 128)
        ),
        segmentation_channels=tuple(
            int(value)
            for value in d("segmentation_head.decoder_channels", [64, 32, 16])
        ),
        segmentation_groupnorm_groups=int(
            d("segmentation_head.groupnorm_groups", 8)
        ),
        visualization_interval_epochs=int(
            d("visualization.interval_epochs", 10)
        ),
        visualization_sample_count=int(d("visualization.sample_count", 8)),
        input_adapter_dim=int(d("optical.input_adapter_dim", 224)),
        max_visual_tokens=int(d("optical.max_visual_tokens", 224)),
        max_language_tokens=224,
        vision_tap_stages=(1, 2, 3),
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
        expert_layers=int(d("optical.geometry.layers_per_expert", 3)),
        top_k=int(d("optical.router.top_k", 4)),
        router_pool_size=int(d("optical.router.pool_size", 14)),
        router_temperature=float(d("optical.router.temperature", 1.0)),
        router_input_layernorm_enabled=bool(
            d("optical.router.input_layernorm_enabled", True)
        ),
        router_input_layernorm_eps=float(
            d("optical.router.input_layernorm_eps", 1e-5)
        ),
        amplitude_slm_weight_domain=str(
            d("optical.router.amplitude_weight_domain", "amplitude")
        ),
        amplitude_slm_input_normalization=str(
            d("optical.router.input_normalization", "none")
        ),
        amplitude_phase_relay="ideal_4f_identity",
        wavelength_nm=float(d("optical.physics.wavelength_nm", 532.0)),
        pixel_pitch_um=float(d("optical.physics.pixel_pitch_um", 8.0)),
        expert_interlayer_distance_m=float(
            d("optical.physics.inter_layer_distance_m", 0.1)
        ),
        last_expert_to_global_distance_m=float(
            d("optical.physics.last_expert_to_global_distance_m", 0.1)
        ),
        global_to_detector_distance_m=float(
            d("optical.physics.global_to_detector_distance_m", 0.1)
        ),
        phase_parameterization=str(
            d("optical.phase.parameterization", "sigmoid")
        ),
        phase_init=str(d("optical.phase.init", "zeros")),
        phase_init_std=float(d("optical.phase.init_std", 0.02)),
        k_space_constraint_enabled=bool(d("optical.k_space.enabled", False)),
        theta_max_deg=float(d("optical.k_space.theta_max_deg", 1.0)),
        interlayer_enabled=bool(d("optical.oeo.enabled", True)),
        interlayer_per_expert_enabled=bool(
            d("optical.oeo.per_expert_enabled", True)
        ),
        interlayer_elementwise_affine=bool(
            d("optical.oeo.elementwise_affine", False)
        ),
        interlayer_hard_route_mask=bool(
            d("optical.oeo.hard_route_mask", True)
        ),
        interlayer_reapply_routing_weights=bool(
            d("optical.oeo.reapply_routing_weights", True)
        ),
        interlayer_layernorm_eps=float(
            d("optical.oeo.layernorm_eps", 1e-5)
        ),
        interlayer_nonlinearity=str(d("optical.oeo.nonlinearity", "relu")),
        detector_output_size=int(d("optical.detector.output_size", 224)),
        detector_layernorm_eps=float(
            d("optical.detector.layernorm_eps", 1e-5)
        ),
        detector_layernorm_affine=bool(
            d("optical.detector.layernorm_affine", False)
        ),
        detector_layernorm_scope=str(
            d("optical.detector.layernorm_scope", "per_token")
        ),
        detector_nonlinearity=str(
            d("optical.detector.nonlinearity", "relu")
        ),
        phase_dropout_mode="none",
        phase_dropout_p=0.0,
        phase_dropout_block_size=4,
        phase_dropout_batch_shared=True,
    )
    if settings.coco_root is None or settings.duts_root is None:
        raise ValueError("dataset.coco_root and dataset.duts_root are required")
    settings.validate()
    return settings


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    with (settings.output_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            settings.to_dict(),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
