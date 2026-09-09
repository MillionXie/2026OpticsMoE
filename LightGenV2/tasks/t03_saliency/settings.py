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
    settings.crop_scale_min = float(d("augmentation.crop_scale_min", 0.90))
    settings.horizontal_flip_probability = float(
        d("augmentation.horizontal_flip_probability", 0.5)
    )
    settings.brightness_jitter = float(d("augmentation.brightness_jitter", 0.10))
    settings.contrast_jitter = float(d("augmentation.contrast_jitter", 0.10))
    settings.kl_weight = float(d("loss.kl_weight", 1.0))
    settings.cc_weight = float(d("loss.cc_weight", 0.5))
    settings.sim_weight = float(d("loss.sim_weight", 0.25))
    settings.nss_weight = float(d("loss.nss_weight", 0.1))
    settings.map_kd_weight = 0.0
    settings.map_kd_temperature = 1.0
    settings.distillation_initial_weight = float(d("distillation.initial_weight", 0.0))
    settings.distillation_end_epoch = int(d("distillation.end_epoch", 50))
    cache = d("distillation.cache_file")
    settings.distillation_cache = _resolve(cache, config.parent) if cache else None
    settings.distillation_teacher_sha256 = d("distillation.teacher_sha256")
    settings.ema_decay = float(d("training.ema_decay", 0.0))
    settings.phase_weight_decay = float(d("training.phase_weight_decay", settings.weight_decay))
    if not 0 <= settings.ema_decay < 1 or settings.distillation_initial_weight < 0:
        raise ValueError("Invalid EMA/KD coefficient")
    if settings.distillation_initial_weight > 0 and (
        settings.augmentation_enabled or settings.distillation_cache is None
        or not settings.distillation_teacher_sha256 or settings.distillation_end_epoch < 2
    ):
        raise ValueError("KD requires aligned nonaugmented inputs, cache, teacher SHA and valid end epoch")
    settings.teacher_checkpoint = None
    settings.ccd_normalization = str(d("lightgen.ccd_normalization", "historical_log1p"))
    if settings.ccd_normalization not in {"historical_log1p", "mean_only"}:
        raise ValueError("Unknown T03 CCD normalization")
    warmstart = d("training.initialization_checkpoint")
    settings.initialization_checkpoint = _resolve(warmstart, config.parent) if warmstart else None
    settings.initialization_checkpoint_sha256 = d("training.initialization_checkpoint_sha256")
    settings.reset_fusion_on_warmstart = bool(d("training.reset_fusion_on_warmstart", False))
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
    settings.dense_readout_learning_rate = float(d("training.dense_readout_learning_rate", settings.dense_readout_learning_rate))
    settings.dense_head_learning_rate = float(d("training.dense_head_learning_rate", settings.dense_head_learning_rate))
    settings.gradient_clip_norm = float(d("training.gradient_clip_norm", 1.0))
    settings.test_interval_epochs = int(d("protocol.test_interval_epochs", 5))
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
    values["lightgen"].update(task="t03_saliency", ccd_normalization=settings.ccd_normalization)
    values.setdefault("training", {}).update(
        learning_rate_source=settings.learning_rate_source,
        ema_decay=settings.ema_decay,
        phase_weight_decay=settings.phase_weight_decay,
        initialization_checkpoint=str(settings.initialization_checkpoint) if settings.initialization_checkpoint else None,
        initialization_checkpoint_sha256=settings.initialization_checkpoint_sha256,
        reset_fusion_on_warmstart=settings.reset_fusion_on_warmstart,
        staged={"enabled": settings.staged_training, "warmup_epochs": settings.staged_warmup_epochs,
                "polish_start": settings.staged_polish_start, "final_hard_balance": settings.staged_final_hard_balance},
    )
    values["effective_optimizer_learning_rates"] = {
        name: getattr(settings, name) for name in (
            "student_learning_rate", "phase_learning_rate", "router_learning_rate",
            "dense_readout_learning_rate", "dense_head_learning_rate")
    }
    values["distillation"] = {"initial_weight": settings.distillation_initial_weight,
        "end_epoch": settings.distillation_end_epoch,
        "cache_file": str(settings.distillation_cache) if settings.distillation_cache else None,
        "teacher_sha256": settings.distillation_teacher_sha256}
    path.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")


__all__ = ["load_settings", "save_resolved_config"]
