from __future__ import annotations

import json
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
    dataset: str = "bdd100k_timeofday3"
    data_root: Path = PROJECT_DIR / "data" / "bdd100k_timeofday3"
    download: bool = True
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int | None = 50176
    processor_max_pixels: int | None = 50176
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = 5000
    test_limit_per_class: int | None = None
    feature_batch_size: int = 4
    inference_batch_size: int = 4
    student_batch_size: int = 4
    head_batch_size: int = 512
    teacher_cache_shard_size: int = 128
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
    classification_prompt: str = (
        "Classify this driving scene into one of the following time-of-day conditions: "
        "daytime, night, dawn_dusk. Answer:"
    )
    replace_vision_stack: bool = True
    replace_language_stack: bool = True
    optical_conversions_per_stack: int = 4
    optical_dim: int = 256
    optical_field_size: int = 256
    optical_padding_size: int = 400
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    mask_distance_cm: float = 5.0
    amplitude_mask_enabled: bool = True
    loss_vision_weight: float = 1.0
    loss_answer_weight: float = 1.0
    loss_kd_weight: float = 0.5
    loss_ce_weight: float = 0.5
    distill_temperature: float = 2.0
    initialize_student_mlp_from_teacher: bool = True
    freeze_qwen_backbone_except_optical: bool = True
    log_interval_batches: int = 20
    save_predictions_interval_epochs: int = 1
    save_visualization_interval_epochs: int = 10
    benchmark_batches: int | None = None
    seed: int = 42
    progress: bool = True
    # Runtime-resolved architecture fields. They remain null in source config.
    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    text_depth: int | None = None
    text_hidden_size: int | None = None

    def validate(self) -> None:
        if self.dataset != "bdd100k_timeofday3":
            raise ValueError("This experiment requires dataset='bdd100k_timeofday3'")
        model_path = Path(self.model_id)
        if self.model_id != MODEL_ID and not model_path.is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or an existing local directory")
        if not self.replace_vision_stack or not self.replace_language_stack:
            raise ValueError("Only the both-optical4 experiment is implemented; both replacement flags must be true")
        if self.optical_conversions_per_stack != 4:
            raise ValueError("fullstack4 requires optical_conversions_per_stack=4")
        if self.optical_padding_size < self.optical_field_size:
            raise ValueError("optical_padding_size must be >= optical_field_size")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        positive = (
            "feature_batch_size", "inference_batch_size", "student_batch_size", "head_batch_size",
            "teacher_cache_shard_size", "epochs", "optical_dim", "optical_field_size",
            "optical_padding_size", "log_interval_batches", "save_predictions_interval_epochs",
            "save_visualization_interval_epochs",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("train_limit", "test_limit", "train_limit_per_class", "test_limit_per_class", "benchmark_batches"):
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
