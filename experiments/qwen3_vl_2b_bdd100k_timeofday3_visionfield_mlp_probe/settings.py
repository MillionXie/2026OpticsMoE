from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import MODEL_ID


PROJECT_DIR = Path(__file__).resolve().parent
PATH_FIELDS = {
    "data_root", "output_dir", "cache_dir", "source_experiment_dir",
    "source_vision_checkpoint",
}
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class Settings:
    dataset: str = "bdd100k_timeofday3"
    data_root: Path = PROJECT_DIR / "data" / "bdd100k_timeofday3"
    download: bool = True
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    output_dir: Path = PROJECT_DIR / "runs" / "qwen3_vl_2b_bdd100k_timeofday3_visionfield_mlp_probe"
    source_experiment_dir: Path = PROJECT_DIR.parent / "qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual" / "runs" / "qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual"
    source_vision_checkpoint: Path = source_experiment_dir / "checkpoints" / "vision_optical_stack_best.pt"
    model_id: str = MODEL_ID
    cache_dir: Path | None = None
    local_files_only: bool = False
    processor_min_pixels: int = 16384
    processor_max_pixels: int = 16384
    classification_prompt: str = "Classify this driving scene: daytime, night, or dawn_dusk. Answer:"
    feature_batch_size: int = 1
    head_batch_size: int = 512
    inference_batch_size: int = 1
    num_workers: int = 8
    cache_dtype: str = "float16"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    epochs: int = 50
    validation_fraction: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    probe_head_type: str = "mlp"
    probe_hidden_dim: int = 512
    probe_dropout: float = 0.1
    optical_dim: int = 64
    optical_field_size: int = 64
    optical_padding_size: int = 128
    optical_conversions_per_stack: int = 4
    amplitude_mask_enabled: bool = False
    phase_init: str = "zeros"
    phase_init_std: float = 0.02
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 8.0
    mask_distance_cm: float = 5.0
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    train_samples_per_class_per_epoch: int | None = None
    save_feature_visualizations: bool = True
    feature_visualization_sample_count: int = 16
    finetune_vision_input_adapter: bool = False
    seed: int = 42
    progress: bool = True

    def validate(self) -> None:
        if self.dataset != "bdd100k_timeofday3":
            raise ValueError("dataset must be 'bdd100k_timeofday3'")
        if self.probe_head_type not in {"mlp", "linear", "bottleneck"}:
            raise ValueError("probe_head_type must be mlp, linear, or bottleneck")
        if self.probe_hidden_dim <= 0:
            raise ValueError("probe_hidden_dim must be positive")
        if not 0 <= self.probe_dropout < 1:
            raise ValueError("probe_dropout must be in [0, 1)")
        if self.optical_dim != 64 or self.optical_field_size != 64:
            raise ValueError("This probe requires optical_dim=optical_field_size=64")
        if self.optical_conversions_per_stack != 4:
            raise ValueError("Source fullstack4 checkpoint requires four conversions")
        if self.processor_min_pixels <= 0 or self.processor_max_pixels <= 0:
            raise ValueError("processor pixel budgets must be positive")
        if self.processor_min_pixels > self.processor_max_pixels:
            raise ValueError("processor_min_pixels must be <= processor_max_pixels")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        for name in ("feature_batch_size", "head_batch_size", "inference_batch_size", "epochs", "num_workers", "feature_visualization_sample_count"):
            if int(getattr(self, name)) < (0 if name == "num_workers" else 1):
                raise ValueError(f"{name} has an invalid value")
        for name in ("train_limit", "test_limit", "train_limit_per_class", "test_limit_per_class"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.finetune_vision_input_adapter:
            raise NotImplementedError(
                "finetune_vision_input_adapter is reserved for an online-feature extension. "
                "The fixed-feature probe requires false."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
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
    original = str(value)
    missing = sorted({a or b for a, b in ENV_REFERENCE.findall(original) if not os.environ.get(a or b)})
    if missing:
        raise ValueError(f"{field_name} references unset environment variables: {', '.join(missing)}")
    expanded = os.path.expandvars(os.path.expanduser(original))
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()
