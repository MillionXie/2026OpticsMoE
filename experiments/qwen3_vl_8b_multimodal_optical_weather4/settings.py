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
    dataset: str = "imagefolder"
    data_root: Path = PROJECT_DIR / "data"
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_8b_multimodal_optical_bdd100k_weather4"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    download: bool = True
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    resize_to: int | None = None
    processor_min_pixels: int | None = 224 * 224
    processor_max_pixels: int | None = 224 * 224
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    feature_batch_size: int = 4
    inference_batch_size: int = 1
    head_batch_size: int = 512
    num_workers: int = 4
    cache_features: bool = True
    cache_dtype: str = "float16"
    hidden_dim: int = 1024
    dropout: float = 0.1
    epochs: int = 30
    validation_fraction: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dtype: str = "float32"
    device: str = "cuda"
    attn_implementation: str = "eager"
    warmup_batches: int = 5
    benchmark_batches: int | None = None
    seed: int = 42
    progress: bool = True
    classification_prompt: str = (
        "Classify this driving scene into one of the following weather conditions: "
        "clear, rainy, snowy, foggy. Answer:"
    )
    optical_enabled: bool = True
    replace_vision_block_start: int = 26
    replace_vision_block_end: int = 26
    optical_dim: int = 256
    optical_layers: int = 4
    optical_field_size: int = 256
    optical_padding_size: int = 400
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    mask_distance_cm: float = 5.0
    distill_temperature: float = 2.0
    loss_hidden_weight: float = 1.0
    loss_kd_weight: float = 0.5
    loss_ce_weight: float = 0.5
    train_mlp: bool = True
    initialize_student_mlp_from_teacher: bool = True
    freeze_qwen_backbone_except_optical: bool = True

    def validate(self) -> None:
        if self.dataset != "imagefolder":
            raise ValueError("This experiment requires dataset='imagefolder'")
        model_path = Path(self.model_id)
        if self.model_id != MODEL_ID and not model_path.is_dir():
            raise ValueError(
                f"This project is fixed to {MODEL_ID}; model_id may only differ when it is an "
                "existing local checkpoint directory."
            )
        for name in ("feature_batch_size", "inference_batch_size", "head_batch_size", "epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "train_limit",
            "test_limit",
            "train_limit_per_class",
            "test_limit_per_class",
            "benchmark_batches",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        # if self.dtype != "float32":
        #     raise ValueError("This conservative baseline requires dtype='float32'")
        # if self.attn_implementation != "eager":
        #     raise ValueError("This conservative baseline requires attn_implementation='eager'")
        if self.processor_min_pixels and self.processor_max_pixels:
            if self.processor_min_pixels > self.processor_max_pixels:
                raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
        if not self.classification_prompt.strip():
            raise ValueError("classification_prompt must be non-empty")
        if not self.optical_enabled:
            raise ValueError("The optical replacement experiment requires optical_enabled=true")
        if not self.train_mlp:
            raise ValueError("The optical student experiment requires train_mlp=true")
        if not self.initialize_student_mlp_from_teacher:
            raise ValueError(
                "The first version requires initialize_student_mlp_from_teacher=true"
            )
        if not self.freeze_qwen_backbone_except_optical:
            raise ValueError("The Qwen backbone must remain frozen except for the optical surrogate")
        if self.replace_vision_block_start != self.replace_vision_block_end:
            raise ValueError("The first version supports replacement of exactly one vision block")
        if self.replace_vision_block_start < 0:
            raise ValueError("replace_vision_block_start must be non-negative")
        for name in ("optical_dim", "optical_layers", "optical_field_size", "optical_padding_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.optical_padding_size < self.optical_field_size:
            raise ValueError("optical_padding_size must be >= optical_field_size")
        for name in ("wavelength_nm", "pixel_pitch_um", "mask_distance_cm", "distill_temperature"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("loss_hidden_weight", "loss_kd_weight", "loss_ce_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: Path) -> Settings:
    path = resolve_path(path, Path.cwd(), "config")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(raw)
    model_value = values.get("model_id")
    if model_value:
        values["model_id"] = resolve_model_id(model_value, path.parent)
    for name in PATH_FIELDS:
        value = values.get(name)
        if value is None:
            continue
        values[name] = resolve_path(value, path.parent, name)
    settings = Settings(**values)
    settings.data_root = resolve_path(settings.data_root, Path.cwd(), "data_root")
    settings.output_dir = resolve_path(settings.output_dir, Path.cwd(), "output_dir")
    if settings.cache_dir is not None:
        settings.cache_dir = resolve_path(settings.cache_dir, Path.cwd(), "cache_dir")
    settings.validate()
    return settings


def resolve_model_id(value: str | Path, base_dir: Path) -> str:
    raw = str(value)
    if raw == MODEL_ID:
        return raw
    candidate = resolve_path(raw, base_dir, "model_id")
    if not candidate.is_dir():
        raise ValueError(f"Local model_id directory does not exist: {candidate}")
    return str(candidate)


def resolve_path(value: str | Path, base_dir: Path, field_name: str) -> Path:
    raw = str(value)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    unresolved = sorted({left or right for left, right in ENV_REFERENCE.findall(expanded)})
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(
            f"Config/CLI field '{field_name}' references unset environment variable(s): {names}"
        )
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def normalize_hub_cache_dir(cache_dir: Path | None, model_id: str) -> Path | None:
    """Accept either a Transformers cache directory or an HF_HOME root."""

    if cache_dir is None or Path(model_id).is_dir():
        return cache_dir
    repo_cache_name = "models--" + model_id.replace("/", "--")
    direct_repo = cache_dir / repo_cache_name
    nested_hub = cache_dir / "hub"
    nested_repo = nested_hub / repo_cache_name
    if direct_repo.is_dir():
        return cache_dir
    if nested_repo.is_dir():
        return nested_hub.resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        resolved_home = resolve_path(hf_home, Path.cwd(), "HF_HOME")
        if cache_dir.resolve() == resolved_home:
            return nested_hub.resolve()
    return cache_dir
