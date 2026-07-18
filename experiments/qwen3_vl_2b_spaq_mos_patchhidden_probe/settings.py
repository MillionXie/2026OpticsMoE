from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
PATH_FIELDS = {"data_root", "annotations_file", "image_dir", "output_dir", "source_output_dir", "cache_dir"}


@dataclass
class Settings:
    dataset: str = "spaq_mos"
    task_name: str = "MOS"
    data_root: Path = PROJECT_DIR.parent.parent / "data" / "SPAQ"
    annotations_file: Path | None = None
    image_dir: Path | None = None
    download: bool = True
    train_fraction: float = 0.9
    train_image_limit: int | None = None
    test_image_limit: int | None = None
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_spaq_mos_patchhidden_probe"
    source_output_dir: Path = (PROJECT_DIR.parent / "qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9" /
                               "runs" / "qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9")
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int = 25600
    processor_max_pixels: int = 25600
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    feature_batch_size: int = 16
    head_batch_size: int = 512
    inference_batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    smooth_l1_beta: float = 0.1
    output_activation: str = "linear"
    seed: int = 42
    log_interval_epochs: int = 1

    def validate(self) -> None:
        if self.dataset != "spaq_mos" or self.task_name != "MOS":
            raise ValueError("This probe supports only SPAQ MOS")
        if self.model_id != MODEL_ID and not Path(self.model_id).is_dir():
            raise ValueError(f"model_id must be {MODEL_ID} or a local model directory")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported dtype")
        if self.output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be linear or sigmoid")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")
        if not 0.0 < self.train_fraction < 1.0 or self.smooth_l1_beta <= 0:
            raise ValueError("Invalid split fraction or SmoothL1 beta")
        for name in ("feature_batch_size", "head_batch_size", "inference_batch_size", "epochs", "log_interval_epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate",):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        for name in ("train_image_limit", "test_image_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for name in PATH_FIELDS:
            value = values.get(name)
            values[name] = str(value) if value is not None else None
        return values


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = set(Settings.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    for name in PATH_FIELDS:
        if raw.get(name) is not None:
            raw[name] = resolve_path(raw[name], config_path.parent, name)
    settings = Settings(**raw)
    settings.validate()
    return settings


def resolve_path(value: str | Path, base: Path, field_name: str) -> Path:
    raw = os.path.expanduser(str(value))
    missing = sorted({a or b for a, b in ENV_REFERENCE.findall(raw) if not os.environ.get(a or b)})
    if missing:
        raise ValueError(f"{field_name} references unset environment variables: {', '.join(missing)}")
    path = Path(os.path.expandvars(raw))
    return (path if path.is_absolute() else base / path).resolve()

