"""SALICON task settings layered on the audited T02 optical contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    _nested,
    _read_config,
)
from LightGenV2.tasks.t02_keypoint_detection.settings import (
    load_settings as load_t02_settings,
    save_resolved_config as save_t02_resolved_config,
)


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(path: str | Path) -> Any:
    config = Path(path).expanduser().resolve()
    settings = load_t02_settings(config)
    raw = _read_config(config)
    d = lambda key, default=None: _nested(raw, key, default)

    # The shared server keeps Hugging Face snapshots beside the repository,
    # while portable configs point at a repository-local cache.  Resolve the
    # former only when the configured cache is absent; this remains offline.
    adjacent_hf_cache = config.parents[5] / ".cache" / "huggingface" / "hub"
    if settings.cache_dir is not None and not settings.cache_dir.exists() and adjacent_hf_cache.is_dir():
        settings.cache_dir = adjacent_hf_cache.resolve()

    settings.validation_limit = d("dataset.validation_limit")
    if settings.validation_limit is not None:
        settings.validation_limit = int(settings.validation_limit)
    settings.materialize_density_maps = bool(
        d("dataset.materialize_density_maps", True)
    )
    settings.density_sigma_px = float(d("dataset.density_sigma_px", 19.0))
    settings.train_images_url = str(
        d("dataset.train_images_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/train.zip")
    )
    settings.validation_images_url = str(
        d("dataset.validation_images_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/val.zip")
    )
    settings.train_annotations_url = str(
        d("dataset.train_annotations_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_train2014.json")
    )
    settings.validation_annotations_url = str(
        d("dataset.validation_annotations_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_val2014.json")
    )
    settings.artifact_cache_dir = _resolve(
        d("artifact_cache_dir", "../../../../cache/qwen3_vl_embedding_2b_salicon_lightgen"),
        config.parent,
    )
    settings.augmentation_enabled = bool(d("augmentation.enabled", True))
    settings.augmentation_mode = str(d("augmentation.mode", "legacy"))
    if settings.augmentation_mode not in {"legacy", "aligned_flip", "aligned_weak"}:
        raise ValueError("Unknown T03 augmentation mode")
    settings.crop_scale_min = float(d("augmentation.crop_scale_min", 0.90))
    settings.horizontal_flip_probability = float(
        d("augmentation.horizontal_flip_probability", 0.5)
    )
    settings.brightness_jitter = float(d("augmentation.brightness_jitter", 0.10))
    settings.contrast_jitter = float(d("augmentation.contrast_jitter", 0.10))
    settings.augmentation_end_epoch = int(d("augmentation.end_epoch", 0))
    settings.augmentation_apply_probability = float(d("augmentation.apply_probability", 1.0))
    if not 0 <= settings.augmentation_apply_probability <= 1:
        raise ValueError("augmentation.apply_probability must be in [0,1]")
    if settings.augmentation_apply_probability != 1 and settings.augmentation_mode != "aligned_weak":
        raise ValueError("Partial augmentation is supported only for aligned_weak")
    if settings.augmentation_mode == "aligned_weak" and (
        not 0.8 <= settings.crop_scale_min <= 1 or not 0 <= settings.brightness_jitter <= .15
        or not 0 <= settings.contrast_jitter <= .15 or settings.augmentation_end_epoch < 0
    ):
        raise ValueError("aligned_weak requires bounded crop/jitter and nonnegative end epoch")
    if not 0 <= settings.horizontal_flip_probability <= 1:
        raise ValueError("Invalid horizontal flip probability")
    if settings.augmentation_mode == "aligned_flip" and (
        settings.crop_scale_min != 1.0 or settings.brightness_jitter != 0.0 or settings.contrast_jitter != 0.0
    ):
        raise ValueError("aligned_flip forbids crop/photometric jitter; teacher maps must remain aligned")
    settings.kl_weight = float(d("loss.kl_weight", 1.0))
    settings.cc_weight = float(d("loss.cc_weight", 0.5))
    settings.sim_weight = float(d("loss.sim_weight", 0.25))
    settings.nss_weight = float(d("loss.nss_weight", 0.1))
    settings.map_kd_weight = 0.0
    settings.sam_rho = float(d("training.sam_rho", 0.0))
    if not 0 <= settings.sam_rho <= .1:
        raise ValueError("training.sam_rho must be in [0,.1]")
    settings.map_kd_temperature = 1.0
    settings.distillation_initial_weight = float(d("distillation.initial_weight", 0.0))
    settings.distillation_final_weight = float(d("distillation.final_weight", 0.0))
    settings.distillation_end_epoch = int(d("distillation.end_epoch", 50))
    cache = d("distillation.cache_file")
    settings.distillation_cache = _resolve(cache, config.parent) if cache else None
    settings.distillation_teacher_sha256 = d("distillation.teacher_sha256")
    settings.feature_hint_initial_weight = float(d("feature_hint.initial_weight",0.0))
    settings.feature_hint_final_weight = float(d("feature_hint.final_weight",0.0))
    settings.feature_hint_end_epoch = int(d("feature_hint.end_epoch",30))
    settings.feature_hint_learning_rate = float(d("feature_hint.learning_rate",0.0002))
    settings.feature_hint_loss_mode = str(d('feature_hint.loss_mode','cosine'))
    if settings.feature_hint_loss_mode not in {'cosine','spatial_centered_cosine'}:
        raise ValueError('Unknown feature hint loss mode')
    hint_cache = d("feature_hint.cache_file")
    settings.feature_hint_cache = _resolve(hint_cache,config.parent) if hint_cache else None
    if not 0 <= settings.feature_hint_final_weight <= settings.feature_hint_initial_weight < float('inf'):
        raise ValueError("Invalid feature hint weights")
    if settings.feature_hint_initial_weight > 0 and (
        settings.augmentation_enabled or not settings.feature_hint_cache or not settings.distillation_teacher_sha256
        or settings.feature_hint_end_epoch < 2 or not 0 < settings.feature_hint_learning_rate < float('inf')
    ):
        raise ValueError("Feature hints require unaugmented train cache, teacher SHA, and valid schedule")
    settings.ema_decay = float(d("training.ema_decay", 0.0))
    settings.phase_weight_decay = float(d("training.phase_weight_decay", settings.weight_decay))
    if not 0 <= settings.ema_decay < 1 or not 0 <= settings.distillation_final_weight <= settings.distillation_initial_weight:
        raise ValueError("Invalid EMA/KD coefficient")
    if settings.distillation_initial_weight > 0 and (
        (settings.augmentation_enabled and settings.augmentation_mode not in {"aligned_flip", "aligned_weak"}) or settings.distillation_cache is None
        or not settings.distillation_teacher_sha256 or settings.distillation_end_epoch < 2
    ):
        raise ValueError("KD requires aligned inputs (none/aligned_flip/aligned_weak), cache, teacher SHA and valid end epoch")
    settings.teacher_checkpoint = None
    settings.ccd_normalization = str(d("lightgen.ccd_normalization", "historical_log1p"))
    if settings.ccd_normalization not in {"historical_log1p", "mean_only"}:
        raise ValueError("Unknown T03 CCD normalization")
    warmstart = d("training.initialization_checkpoint")
    settings.initialization_checkpoint = _resolve(warmstart, config.parent) if warmstart else None
    settings.initialization_checkpoint_sha256 = d("training.initialization_checkpoint_sha256")
    settings.reset_fusion_on_warmstart = bool(d("training.reset_fusion_on_warmstart", False))
    settings.electronic_spatial_kernel_size = int(d("lightgen.electronic_spatial_kernel_size", 3))
    settings.expand_kernel_on_warmstart = bool(d("training.expand_kernel_on_warmstart", False))
    settings.electronic_grn = bool(d("lightgen.electronic_grn", False))
    settings.initialize_grn_on_warmstart = bool(d("training.initialize_grn_on_warmstart", False))
    settings.electronic_ffn_spatial_dilation = int(d("lightgen.electronic_ffn_spatial_dilation", 0))
    settings.initialize_ffn_on_warmstart = bool(d("training.initialize_ffn_on_warmstart", False))
    settings.ffn_spatial_learning_rate = float(d("training.ffn_spatial_learning_rate", 0.0001))
    settings.electronic_global_rank = int(d("lightgen.electronic_global_rank", 0))
    settings.initialize_global_on_warmstart = bool(d("training.initialize_global_on_warmstart", False))
    settings.electronic_ffn_hidden_width = int(d("lightgen.electronic_ffn_hidden_width", 384))
    settings.widen_ffn_on_warmstart = bool(d("training.widen_ffn_on_warmstart", False))
    if settings.electronic_ffn_hidden_width not in (384,576):
        raise ValueError("Only original384 or widened576 FFN hidden width is audited")
    if settings.widen_ffn_on_warmstart and settings.electronic_ffn_hidden_width != 576:
        raise ValueError("FFN widening transfer requires target576")
    if settings.electronic_ffn_hidden_width == 576:
        if (settings.electronic_ffn_spatial_dilation != 1 or settings.electronic_global_rank
                or settings.electronic_grn or settings.electronic_spatial_kernel_size != 3
                or settings.initialize_ffn_on_warmstart or settings.reset_fusion_on_warmstart
                or settings.initialize_global_on_warmstart or settings.expand_kernel_on_warmstart
                or settings.initialize_grn_on_warmstart):
            raise ValueError("Isolate widening to an existing CFFN1; no other architecture/fusion transfer")
        if not settings.initialization_checkpoint or not settings.initialization_checkpoint_sha256:
            raise ValueError("Widened FFN requires SHA-pinned warmstart")
    if settings.sam_rho and settings.feature_hint_initial_weight:
        raise ValueError("Isolate SAM from training-only feature hints")
    settings.global_spatial_learning_rate = float(d("training.global_spatial_learning_rate", 0.0002))
    if settings.electronic_global_rank not in (0,16):
        raise ValueError("Global spatial rank must be 0(off) or audited rank16")
    if settings.initialize_global_on_warmstart and not settings.electronic_global_rank:
        raise ValueError("Global transfer requires enabled module")
    if settings.electronic_global_rank:
        if settings.electronic_grn or settings.electronic_spatial_kernel_size != 3:
            raise ValueError("Isolate global spatial mixing from GRN/large kernels")
        if not settings.initialization_checkpoint or settings.electronic_width != 192:
            raise ValueError("Global spatial mixing requires width192 warmstarted SALICON")
        if not 0 < settings.global_spatial_learning_rate < float('inf'):
            raise ValueError("Global spatial learning rate must be finite and positive")
    if settings.electronic_ffn_spatial_dilation not in (0, 1, 2):
        raise ValueError("Spatial FFN dilation must be 0(off), 1, or 2")
    if settings.initialize_ffn_on_warmstart and not settings.electronic_ffn_spatial_dilation:
        raise ValueError("Spatial FFN transfer requires enabled module")
    if settings.electronic_ffn_spatial_dilation:
        if settings.electronic_grn or settings.electronic_spatial_kernel_size != 3:
            raise ValueError("Isolate spatial FFN from GRN/expanded token kernels")
        if not settings.initialization_checkpoint or settings.electronic_width != 192 or settings.electronic_expansion != 2:
            raise ValueError("Spatial FFN requires audited 192->384 warmstarted SALICON")
        if not 0 < settings.ffn_spatial_learning_rate < float('inf'):
            raise ValueError("Spatial FFN learning rate must be finite and positive")
    if settings.initialize_grn_on_warmstart and not settings.electronic_grn:
        raise ValueError("GRN identity transfer requires electronic_grn")
    if settings.electronic_spatial_kernel_size not in {3, 5, 13}:
        raise ValueError("T03 supports audited electronic spatial kernels 3, 5, or 13")
    if settings.expand_kernel_on_warmstart and settings.electronic_spatial_kernel_size not in (5,13):
        raise ValueError("Kernel expansion requires target kernel 5 or 13")
    # Legacy Vision2 overwrites training.* rates with optimization.* defaults.
    # Keep historical profiles reproducible; new profiles explicitly opt in.
    settings.learning_rate_source = str(d("training.learning_rate_source", "legacy_optimization"))
    if settings.learning_rate_source not in {"legacy_optimization", "task_training"}:
        raise ValueError("Unknown learning_rate_source")
    if settings.learning_rate_source == "task_training":
        for name in ("student_learning_rate", "phase_learning_rate", "router_learning_rate"):
            value = d(f"training.{name}")
            if value is None or not 0 < float(value) < float("inf"):
                raise ValueError(f"Explicit positive training.{name} required")
            setattr(settings, name, float(value))
    settings.staged_training = bool(d("training.staged.enabled", False))
    settings.staged_warmup_epochs = int(d("training.staged.warmup_epochs", 10))
    settings.staged_polish_start = int(d("training.staged.polish_start", 71))
    settings.staged_final_hard_balance = float(d("training.staged.final_hard_balance", 0.10))
    settings.staged_freeze_electronic_gradients = bool(d("training.staged.freeze_electronic_gradients", False))
    settings.dense_readout_learning_rate = float(d("training.dense_readout_learning_rate", settings.dense_readout_learning_rate))
    settings.dense_head_learning_rate = float(d("training.dense_head_learning_rate", settings.dense_head_learning_rate))
    settings.gradient_clip_norm = float(d("training.gradient_clip_norm", 1.0))
    settings.test_interval_epochs = int(d("protocol.test_interval_epochs", 5))
    settings.adaptive_plateau_enabled = bool(d("training.adaptive_plateau.enabled", False))
    settings.adaptive_plateau_options = {
        "patience": int(d("training.adaptive_plateau.patience", 3)),
        "min_delta": float(d("training.adaptive_plateau.min_delta", 0.0001)),
        "factor": float(d("training.adaptive_plateau.factor", 0.5)),
        "max_reductions": int(d("training.adaptive_plateau.max_reductions", 2)),
        "min_epoch": int(d("training.adaptive_plateau.min_epoch", 6)),
    }
    if settings.adaptive_plateau_enabled:
        from .plateau import PlateauController
        PlateauController(**settings.adaptive_plateau_options)
        if not settings.staged_training or settings.test_interval_epochs < 1:
            raise ValueError("Adaptive plateau requires staged training and periodic tests")
    settings.router_hard_load_balance_weight = float(
        d("loss.router_hard_load_balance_weight", 0.50)
    )
    settings.segmentation_projection_dim = int(d("saliency_head.projection_dim", 128))
    settings.segmentation_channels = tuple(
        int(value) for value in d("saliency_head.decoder_channels", [96, 64, 32, 16])
    )
    settings.segmentation_groupnorm_groups = int(
        d("saliency_head.groupnorm_groups", 8)
    )
    settings.visualization_sample_count = int(d("visualization.sample_count", 16))
    settings.visualization_optical_sample_count = int(
        d("visualization.optical_sample_count", 4)
    )
    if settings.router_backend != "optical" or settings.top_k != 2:
        raise ValueError("T03 main contract is fixed to optical Router Top-2")
    if settings.lightgen_model_variant == "optical_router_scale_matched_moe":
        if settings.router_hard_load_balance_weight <= 0:
            raise ValueError("T03 optical Router requires a positive hard-load loss")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_t02_resolved_config(settings)
    import yaml
    path = settings.output_dir / "resolved_config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["lightgen"].update(task="t03_saliency", ccd_normalization=settings.ccd_normalization,
                            electronic_grn=settings.electronic_grn,
                            electronic_ffn_spatial_dilation=settings.electronic_ffn_spatial_dilation,
                            electronic_global_rank=settings.electronic_global_rank,
                            electronic_ffn_hidden_width=settings.electronic_ffn_hidden_width,
                            electronic_spatial_kernel_size=settings.electronic_spatial_kernel_size)
    values.setdefault("augmentation", {}).update(enabled=settings.augmentation_enabled, mode=settings.augmentation_mode,
        crop_scale_min=settings.crop_scale_min, horizontal_flip_probability=settings.horizontal_flip_probability,
        brightness_jitter=settings.brightness_jitter, contrast_jitter=settings.contrast_jitter,
        end_epoch=settings.augmentation_end_epoch, apply_probability=settings.augmentation_apply_probability)
    values.setdefault("training", {}).update(
        learning_rate_source=settings.learning_rate_source,
        ema_decay=settings.ema_decay,
        phase_weight_decay=settings.phase_weight_decay,
        sam_rho=settings.sam_rho,
        initialization_checkpoint=str(settings.initialization_checkpoint) if settings.initialization_checkpoint else None,
        initialization_checkpoint_sha256=settings.initialization_checkpoint_sha256,
        reset_fusion_on_warmstart=settings.reset_fusion_on_warmstart,
        expand_kernel_on_warmstart=settings.expand_kernel_on_warmstart,
        initialize_grn_on_warmstart=settings.initialize_grn_on_warmstart,
        initialize_ffn_on_warmstart=settings.initialize_ffn_on_warmstart,
        ffn_spatial_learning_rate=settings.ffn_spatial_learning_rate,
        initialize_global_on_warmstart=settings.initialize_global_on_warmstart,
        widen_ffn_on_warmstart=settings.widen_ffn_on_warmstart,
        global_spatial_learning_rate=settings.global_spatial_learning_rate,
        adaptive_plateau={"enabled": settings.adaptive_plateau_enabled,
                          **settings.adaptive_plateau_options},
        staged={"enabled": settings.staged_training, "warmup_epochs": settings.staged_warmup_epochs,
                "polish_start": settings.staged_polish_start, "final_hard_balance": settings.staged_final_hard_balance,
                "freeze_electronic_gradients": settings.staged_freeze_electronic_gradients},
    )
    values["effective_optimizer_learning_rates"] = {
        name: getattr(settings, name) for name in (
            "student_learning_rate", "phase_learning_rate", "router_learning_rate",
            "dense_readout_learning_rate", "dense_head_learning_rate")
    }
    values["distillation"] = {"initial_weight": settings.distillation_initial_weight,
        "final_weight": settings.distillation_final_weight,
        "end_epoch": settings.distillation_end_epoch,
        "cache_file": str(settings.distillation_cache) if settings.distillation_cache else None,
        "teacher_sha256": settings.distillation_teacher_sha256}
    values['feature_hint'] = {'initial_weight':settings.feature_hint_initial_weight,
        'loss_mode':settings.feature_hint_loss_mode,
        'final_weight':settings.feature_hint_final_weight,'end_epoch':settings.feature_hint_end_epoch,
        'learning_rate':settings.feature_hint_learning_rate,
        'cache_file':str(settings.feature_hint_cache) if settings.feature_hint_cache else None,
        'inference_parameters_added':0,'training_only_projection_parameters':36864 if settings.feature_hint_initial_weight else 0}
    path.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")


__all__ = ["load_settings", "save_resolved_config"]
