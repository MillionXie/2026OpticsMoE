from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {"data_root", "output_dir", "cache_dir"}


@dataclass
class Settings:
    dataset: str = "cifar100"
    data_root: Path = PROJECT_DIR / "data"
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_8b_cifar100"
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
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    warmup_batches: int = 5
    benchmark_batches: int | None = None
    seed: int = 42
    progress: bool = True

    def validate(self) -> None:
        supported = {"cifar10", "cifar100", "stl10", "svhn", "fashionmnist", "imagefolder"}
        if self.dataset not in supported:
            raise ValueError(f"dataset must be one of {sorted(supported)}, got {self.dataset!r}")
        model_path = Path(self.model_id).expanduser()
        if self.model_id != MODEL_ID and not model_path.exists():
            raise ValueError(
                f"This project is fixed to {MODEL_ID}; model_id may only differ when it is an "
                "existing local checkpoint directory."
            )
        for name in ("feature_batch_size", "inference_batch_size", "head_batch_size", "epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("train_limit", "test_limit", "benchmark_batches"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.processor_min_pixels and self.processor_max_pixels:
            if self.processor_min_pixels > self.processor_max_pixels:
                raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: Path) -> Settings:
    path = path.expanduser().resolve()
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
    if model_value and model_value != MODEL_ID:
        model_candidate = Path(model_value).expanduser()
        if not model_candidate.is_absolute():
            model_candidate = path.parent / model_candidate
        values["model_id"] = str(model_candidate.resolve())
    for name in PATH_FIELDS:
        value = values.get(name)
        if value is None:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        values[name] = candidate.resolve()
    settings = Settings(**values)
    settings.data_root = Path(settings.data_root).expanduser().resolve()
    settings.output_dir = Path(settings.output_dir).expanduser().resolve()
    if settings.cache_dir is not None:
        settings.cache_dir = Path(settings.cache_dir).expanduser().resolve()
    settings.validate()
    return settings
