from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_2b_spaq_multitask_iqa.settings import (
    normalize_hub_cache_dir,
    resolve_model_id,
    resolve_path,
)

from . import MODEL_ID, TASK_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {"data_root", "output_dir", "cache_dir", "supervised_reference_metrics"}

DEFAULT_PROMPTS = {
    "MOS": (
        "Evaluate the overall perceptual quality of this image. Return only one numeric score "
        "from 0 (bad) to 100 (excellent), with no explanation. Score:"
    ),
    "Brightness": (
        "Evaluate the brightness and exposure quality of this image. Return only one numeric "
        "score from 0 (worst exposure) to 100 (best exposure), with no explanation. Score:"
    ),
    "Colorfulness": (
        "Evaluate the perceived colorfulness of this image. Return only one numeric score from "
        "0 (least colorful) to 100 (most colorful), with no explanation. Score:"
    ),
    "Contrast": (
        "Evaluate the perceived contrast of this image. Return only one numeric score from "
        "0 (lowest contrast) to 100 (highest contrast), with no explanation. Score:"
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a numeric image-rating API. Your entire response must contain exactly one number "
    "and nothing else. Do not explain, describe, justify, or add units."
)


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
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_spaq_zeroshot_iqa"
    supervised_reference_metrics: Path | None = (
        PROJECT_DIR.parent / "qwen3_vl_2b_spaq_multitask_iqa" / "runs"
        / "qwen3_vl_2b_spaq_multitask_iqa_sigmoid" / "test_metrics.json"
    )
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int | None = 50176
    processor_max_pixels: int | None = 50176
    train_fraction: float = 0.9
    train_image_limit: int | None = None
    test_image_limit: int | None = None
    generation_batch_size: int = 4
    num_workers: int = 4
    max_new_tokens: int = 32
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    seed: int = 42
    log_interval_batches: int = 20
    save_interval_batches: int = 20
    progress: bool = True
    task_prompts: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.task_prompts is None:
            self.task_prompts = dict(DEFAULT_PROMPTS)

    def validate(self) -> None:
        if self.dataset != "spaq":
            raise ValueError("dataset must be 'spaq'")
        if self.seed != 42:
            raise ValueError("This comparison requires split seed 42")
        if self.download_source not in {"huggingface", "google_drive"}:
            raise ValueError("download_source must be 'huggingface' or 'google_drive'")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        for name in (
            "generation_batch_size", "num_workers", "max_new_tokens",
            "log_interval_batches", "save_interval_batches",
        ):
            if int(getattr(self, name)) <= 0 and name != "num_workers":
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        for name in ("train_image_limit", "test_image_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.processor_min_pixels is not None and self.processor_min_pixels <= 0:
            raise ValueError("processor_min_pixels must be positive")
        if self.processor_max_pixels is not None and self.processor_max_pixels <= 0:
            raise ValueError("processor_max_pixels must be positive")
        if self.processor_min_pixels and self.processor_max_pixels:
            if self.processor_min_pixels > self.processor_max_pixels:
                raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
        if set(self.task_prompts or {}) != set(TASK_NAMES):
            raise ValueError(f"task_prompts must contain exactly {list(TASK_NAMES)}")
        if any(not str(prompt).strip() for prompt in (self.task_prompts or {}).values()):
            raise ValueError("task prompts must be non-empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")

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
        if values.get(name) is not None:
            values[name] = resolve_path(values[name], path.parent, name)
    data_root = Path(values.get("data_root", Settings.data_root))
    for name in ("annotations_file", "image_dir"):
        if values.get(name) is not None:
            values[name] = resolve_path(values[name], data_root, name)
    settings = Settings(**values)
    settings.data_root = resolve_path(settings.data_root, Path.cwd(), "data_root")
    settings.output_dir = resolve_path(settings.output_dir, Path.cwd(), "output_dir")
    settings.validate()
    return settings


__all__ = ["Settings", "load_settings", "normalize_hub_cache_dir"]
