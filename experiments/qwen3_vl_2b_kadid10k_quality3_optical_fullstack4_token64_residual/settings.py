from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {"data_root", "output_dir", "cache_dir"}
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class Settings:
    dataset: str = "kadid10k_quality3"
    data_root: Path = PROJECT_DIR / "data" / "kadid10k"
    download: bool = True
    dataset_download_url: str = (
        "https://files.osf.io/v1/resources/xkqjh/providers/osfstorage/"
        "5eafe5bf0ffc0500ec6f6c94/?zip="
    )
    metadata_csv: str = "dmos.csv"
    image_dir: str = "images"
    quality_label_mode: str = "score_tertile"
    quality_score_higher_is_better: bool | None = None
    train_reference_fraction: float = 0.8
    test_reference_fraction: float = 0.2
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int | None = 16384
    processor_max_pixels: int | None = 16384
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    train_samples_per_class_per_epoch: int | None = 1000
    feature_batch_size: int = 1
    inference_batch_size: int = 1
    student_batch_size: int = 1
    head_batch_size: int = 512
    teacher_cache_shard_size: int = 128
    teacher_cache_lru_shards: int = 8
    num_workers: int = 8
    cache_dtype: str = "float16"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    epochs: int = 30
    validation_fraction: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    hidden_dim: int = 1024
    dropout: float = 0.1
    head_type: str = "mlp"
    head_hidden_dim: int | None = None
    head_bottleneck_dim: int = 128
    head_use_layernorm: bool = False
    classification_prompt: str = (
        "Rate image quality: high_quality, medium_quality, or low_quality. Answer:"
    )
    replace_vision_stack: bool = True
    replace_language_stack: bool = True
    optical_conversions_per_stack: int = 4
    optical_dim: int = 64
    optical_field_size: int = 64
    optical_padding_size: int = 128
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 8.0
    mask_distance_cm: float = 5.0
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    amplitude_mask_enabled: bool = False
    optical_residual_enabled: bool = True
    optical_identity_scale_init: float = 1.0
    optical_modulated_scale_init: float = 0.1
    optical_identity_scale_trainable: bool = False
    optical_modulated_scale_trainable: bool = True
    loss_vision_weight: float = 0.4
    loss_answer_weight: float = 0.4
    loss_kd_weight: float = 1.0
    loss_ce_weight: float = 1.0
    distill_temperature: float = 2.0
    initialize_student_mlp_from_teacher: bool = True
    freeze_qwen_backbone_except_optical: bool = True
    log_interval_batches: int = 100
    save_predictions_interval_epochs: int = 1
    save_visualization_interval_epochs: int = 10
    save_debug_visualizations: bool = True
    debug_visualization_sample_count: int = 8
    debug_visualization_interval_epochs: int = 1
    debug_visualization_max_tokens: int = 64
    debug_visualization_save_raw_tensors: bool = True
    debug_visualization_percentile_clip: float = 99.0
    debug_visualization_include_correct: bool = True
    debug_visualization_include_incorrect: bool = True
    benchmark_batches: int | None = None
    seed: int = 42
    progress: bool = True
    # Runtime-resolved architecture fields. They remain null in source config.
    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    text_depth: int | None = None
    text_hidden_size: int | None = None
    quality_score_column: str | None = None

    def validate(self) -> None:
        if self.dataset != "kadid10k_quality3":
            raise ValueError("This experiment requires dataset='kadid10k_quality3'")
        if self.head_type not in {"mlp", "linear", "bottleneck", "normalized_linear"}:
            raise ValueError("head_type must be mlp, linear, bottleneck, or normalized_linear")
        if self.head_bottleneck_dim <= 0:
            raise ValueError("head_bottleneck_dim must be positive")
        if self.head_hidden_dim is not None and self.head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive when set")
        if self.quality_label_mode not in {"score_tertile", "distortion_level_3class"}:
            raise ValueError("quality_label_mode must be score_tertile or distortion_level_3class")
        if not self.metadata_csv.strip():
            raise ValueError("metadata_csv must be non-empty")
        if not self.image_dir.strip():
            raise ValueError("image_dir must be non-empty")
        if self.download and not self.dataset_download_url.strip():
            raise ValueError("dataset_download_url must be non-empty when download=true")
        if self.quality_score_higher_is_better is not None and not isinstance(self.quality_score_higher_is_better,bool):
            raise ValueError("quality_score_higher_is_better must be true, false, or null")
        if not 0.0 < self.train_reference_fraction < 1.0:
            raise ValueError("train_reference_fraction must be between 0 and 1")
        if not 0.0 < self.test_reference_fraction < 1.0:
            raise ValueError("test_reference_fraction must be between 0 and 1")
        if not math.isclose(self.train_reference_fraction + self.test_reference_fraction, 1.0):
            raise ValueError("train_reference_fraction + test_reference_fraction must equal 1")
        model_path = Path(self.model_id)
        if self.model_id != MODEL_ID and not model_path.is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or an existing local directory")
        if not self.replace_vision_stack or not self.replace_language_stack:
            raise ValueError("Only the both-optical4 experiment is implemented; both replacement flags must be true")
        if self.optical_conversions_per_stack != 4:
            raise ValueError("fullstack4 requires optical_conversions_per_stack=4")
        if self.optical_padding_size < self.optical_field_size:
            raise ValueError("optical_padding_size must be >= optical_field_size")
        if self.optical_dim != self.optical_field_size:
            raise ValueError(
                "token64 direct row mapping requires optical_dim == optical_field_size"
            )
        if self.processor_min_pixels is None or self.processor_max_pixels is None:
            raise ValueError("token64 requires explicit processor_min_pixels and processor_max_pixels")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        for name in ("optical_identity_scale_init", "optical_modulated_scale_init"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.phase_init not in {"zeros", "identity", "uniform", "uniform_0_2pi", "normal", "small_normal"}:
            raise ValueError(
                "phase_init must be one of: zeros, identity, uniform, "
                "uniform_0_2pi, normal, small_normal"
            )
        if self.phase_init_std <= 0:
            raise ValueError("phase_init_std must be positive")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        positive = (
            "feature_batch_size", "inference_batch_size", "student_batch_size", "head_batch_size",
            "teacher_cache_shard_size", "teacher_cache_lru_shards", "epochs", "optical_dim", "optical_field_size",
            "optical_padding_size", "log_interval_batches", "save_predictions_interval_epochs",
            "save_visualization_interval_epochs",
            "debug_visualization_sample_count", "debug_visualization_interval_epochs",
            "debug_visualization_max_tokens",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.debug_visualization_percentile_clip <= 100.0:
            raise ValueError("debug_visualization_percentile_clip must be in (0, 100]")
        for name in ("train_limit", "test_limit", "train_limit_per_class", "test_limit_per_class", "train_samples_per_class_per_epoch", "benchmark_batches"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set")
        for name in ("loss_vision_weight", "loss_answer_weight", "loss_kd_weight", "loss_ce_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if not self.initialize_student_mlp_from_teacher or not self.freeze_qwen_backbone_except_optical:
            raise ValueError("Teacher initialization and frozen non-optical Qwen parameters are required")

    def resolve_architecture(self, model: Any) -> None:
        self.vision_depth = int(model.config.vision_config.depth)
        self.vision_hidden_size = int(model.config.vision_config.hidden_size)
        self.text_depth = int(model.config.text_config.num_hidden_layers)
        self.text_hidden_size = int(model.config.text_config.hidden_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(raw)
    if values.get("model_id") and values["model_id"] != MODEL_ID:
        values["model_id"] = str(resolve_path(values["model_id"], config_path.parent, "model_id"))
    for name in PATH_FIELDS:
        if values.get(name) is not None:
            values[name] = resolve_path(values[name], config_path.parent, name)
    settings = Settings(**values)
    settings.validate()
    return settings


def resolve_path(value: str | Path, base: Path, field_name: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    unresolved = sorted({a or b for a, b in ENV_REFERENCE.findall(expanded)})
    if unresolved:
        raise ValueError(f"{field_name} references unset environment variables: {', '.join(unresolved)}")
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()
