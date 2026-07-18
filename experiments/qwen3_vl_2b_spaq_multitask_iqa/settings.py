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

DEFAULT_PROMPTS = {
    "MOS": "Predict the human-rated overall perceptual quality of this image on a 0-100 scale. Score:",
    "Brightness": "Predict the human-rated brightness and exposure quality of this image on a 0-100 scale. Score:",
    "Colorfulness": "Predict the human-rated colorfulness of this image on a 0-100 scale. Score:",
    "Contrast": "Predict the human-rated contrast of this image on a 0-100 scale. Score:",
}


@dataclass
class Settings:
    dataset: str = "spaq"
    data_root: Path = PROJECT_DIR / "data" / "SPAQ"
    annotations_file: Path | None = None
    image_dir: Path | None = None
    download: bool = True
    download_source: str = "huggingface"
    download_repo_id: str = "chaofengc/IQA-PyTorch-Datasets"
    download_filename: str = "spaq.tgz"
    download_endpoint: str | None = "https://hf-mirror.com"
    download_url: str | None = None
    keep_download_archive: bool = False
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_spaq_multitask_iqa"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int | None = 50176
    processor_max_pixels: int | None = 50176
    train_fraction: float = 0.9
    train_image_limit: int | None = None
    test_image_limit: int | None = None
    feature_batch_size: int = 1
    head_batch_size: int = 512
    num_workers: int = 4
    cache_features: bool = True
    cache_dtype: str = "float16"
    expected_feature_dim: int = 2048
    head_hidden_dim: int = 64
    head_output_activation: str = "none"
    dropout: float = 0.1
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    smooth_l1_beta: float = 0.1
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    seed: int = 42
    progress: bool = True
    task_prompts: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.task_prompts is None:
            self.task_prompts = dict(DEFAULT_PROMPTS)

    def validate(self) -> None:
        if self.dataset != "spaq":
            raise ValueError("dataset must be 'spaq'")
        if self.download_source not in {"huggingface", "google_drive"}:
            raise ValueError("download_source must be 'huggingface' or 'google_drive'")
        if self.download and self.download_source == "huggingface":
            if not self.download_repo_id.strip() or not self.download_filename.strip():
                raise ValueError("download_repo_id and download_filename must be non-empty")
            if self.download_endpoint is not None and not self.download_endpoint.startswith(("http://", "https://")):
                raise ValueError("download_endpoint must be an HTTP(S) URL or null")
        if self.download and self.download_source == "google_drive" and not self.download_url:
            raise ValueError("download_url is required for google_drive downloads")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.seed != 42:
            raise ValueError("This experiment requires the fixed split seed 42")
        if self.expected_feature_dim != 2048:
            raise ValueError("Qwen3-VL-2B answer hidden dimension must be 2048")
        if self.head_output_activation not in {"none", "sigmoid"}:
            raise ValueError("head_output_activation must be 'none' or 'sigmoid'")
        for name in ("feature_batch_size", "head_batch_size", "epochs", "head_hidden_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("train_image_limit", "test_image_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.processor_min_pixels is not None and self.processor_min_pixels <= 0:
            raise ValueError("processor_min_pixels must be positive")
        if self.processor_max_pixels is not None and self.processor_max_pixels <= 0:
            raise ValueError("processor_max_pixels must be positive")
        if self.processor_min_pixels and self.processor_max_pixels:
            if self.processor_min_pixels > self.processor_max_pixels:
                raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
        expected = set(DEFAULT_PROMPTS)
        if set(self.task_prompts or {}) != expected:
            raise ValueError(f"task_prompts must contain exactly {sorted(expected)}")
        for task, prompt in (self.task_prompts or {}).items():
            if not str(prompt).strip():
                raise ValueError(f"Prompt for {task} is empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: Path) -> Settings:
    path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(raw)
    if values.get("model_id"):
        values["model_id"] = resolve_model_id(values["model_id"], path.parent)
    for name in PATH_FIELDS:
        value = values.get(name)
        if value is not None:
            values[name] = resolve_path(value, path.parent, name)
    data_root = values.get("data_root", Settings.data_root)
    data_root = Path(data_root)
    for name in ("annotations_file", "image_dir"):
        value = values.get(name)
        if value is not None:
            values[name] = resolve_path(value, data_root, name)
    settings = Settings(**values)
    settings.data_root = resolve_path(settings.data_root, Path.cwd(), "data_root")
    settings.output_dir = resolve_path(settings.output_dir, Path.cwd(), "output_dir")
    settings.validate()
    return settings


def resolve_model_id(value: str | Path, base_dir: Path) -> str:
    raw = str(value)
    if raw == MODEL_ID or ("/" in raw and not raw.startswith((".", "/", "~", "$"))):
        return raw
    candidate = resolve_path(raw, base_dir, "model_id")
    if not candidate.is_dir():
        raise ValueError(f"Local model_id directory does not exist: {candidate}")
    return str(candidate)


def resolve_path(value: str | Path, base_dir: Path, field_name: str) -> Path:
    raw = str(value)
    variables = {left or right for left, right in ENV_REFERENCE.findall(raw)}
    missing = sorted(name for name in variables if name not in os.environ)
    if missing:
        raise ValueError(
            f"Field '{field_name}' references unset environment variable(s): {', '.join(missing)}"
        )
    expanded = os.path.expandvars(os.path.expanduser(raw))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def normalize_hub_cache_dir(cache_dir: Path | None, model_id: str) -> Path | None:
    if cache_dir is None or Path(model_id).is_dir():
        return cache_dir
    repo_name = "models--" + model_id.replace("/", "--")
    if (cache_dir / repo_name).is_dir():
        return cache_dir
    if (cache_dir / "hub" / repo_name).is_dir():
        return (cache_dir / "hub").resolve()
    return cache_dir
