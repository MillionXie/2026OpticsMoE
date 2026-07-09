from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_EXPERIMENT_DIR = (
    PROJECT_DIR.parent / "qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual"
)
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
PATH_FIELDS = {"data_root", "output_dir"}
CLASS_NAMES = ["daytime", "night", "dawn_dusk"]
MODEL_TYPES = {"standard_d2nn", "lenet5", "resnet18", "vgg11_bn", "mobilenet_v2"}


@dataclass
class Settings:
    dataset: str = "bdd100k_timeofday3"
    data_root: Path = REFERENCE_EXPERIMENT_DIR / "data" / "bdd100k_timeofday3"
    download: bool = True
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    output_dir: Path = PROJECT_DIR / "runs" / "standard_d2nn64"
    model_type: str = "standard_d2nn"
    num_classes: int = 3
    class_names: list[str] | None = None
    image_size: int = 224
    image_normalization: str = "none"
    validation_fraction: float = 0.1
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    train_samples_per_class_per_epoch: int | None = 1000
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 30
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    device: str = "cuda"
    seed: int = 42
    progress: bool = True
    log_interval_batches: int = 50
    save_interval_epochs: int = 10
    save_predictions_interval_epochs: int = 1
    pretrained: bool = False
    optical_layers: int = 5
    optical_field_size: int = 64
    optical_padding_size: int = 128
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 8.0
    mask_distance_cm: float = 5.0
    phase_init: str = "uniform"
    phase_init_std: float = 0.02
    detector_region_size: int = 12
    detector_region_temperature: float = 1.0
    detector_concentration_loss_weight: float = 0.0
    input_energy_normalization: str = "rms"
    reference_experiment: str = "qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual"
    reference_processor_min_pixels: int = 16384
    reference_processor_max_pixels: int = 16384

    def __post_init__(self) -> None:
        if self.class_names is None:
            self.class_names = list(CLASS_NAMES)

    def validate(self) -> None:
        if self.dataset != "bdd100k_timeofday3":
            raise ValueError("dataset must be bdd100k_timeofday3")
        if self.class_names != CLASS_NAMES or self.num_classes != 3:
            raise ValueError("TimeOfDay-3 class order must be daytime, night, dawn_dusk")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}")
        if self.optimizer != "adamw" or self.scheduler != "cosine":
            raise ValueError("Only AdamW + cosine are supported")
        if self.image_normalization not in {"none", "imagenet"}:
            raise ValueError("image_normalization must be none or imagenet")
        if self.model_type in {"standard_d2nn", "lenet5"} and self.image_normalization == "imagenet":
            raise ValueError("imagenet normalization is only for RGB torchvision baselines")
        if self.model_type == "lenet5" and self.image_size != 32:
            raise ValueError("standard LeNet-5 config requires image_size=32")
        if self.model_type == "standard_d2nn":
            if self.optical_layers <= 0:
                raise ValueError("optical_layers must be positive")
            if self.optical_padding_size < self.optical_field_size:
                raise ValueError("optical_padding_size must be >= optical_field_size")
            max_nonoverlap_size = self.optical_field_size // (self.num_classes + 1)
            if not 0 < self.detector_region_size <= max_nonoverlap_size:
                raise ValueError(f"detector_region_size must be between 1 and {max_nonoverlap_size}")
            if self.detector_region_temperature <= 0:
                raise ValueError("detector_region_temperature must be positive")
            if self.detector_concentration_loss_weight < 0:
                raise ValueError("detector_concentration_loss_weight must be non-negative")
            if self.phase_init not in {"zeros", "uniform", "normal", "small_normal"}:
                raise ValueError("phase_init must be zeros, uniform, normal, or small_normal")
            if self.phase_init_std <= 0:
                raise ValueError("phase_init_std must be positive")
            if self.input_energy_normalization not in {"none", "rms", "mean"}:
                raise ValueError("input_energy_normalization must be none, rms, or mean")
        if self.pretrained and self.model_type in {"standard_d2nn", "lenet5"}:
            raise ValueError("pretrained only applies to torchvision baselines")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        for name in (
            "image_size",
            "batch_size",
            "epochs",
            "log_interval_batches",
            "save_interval_epochs",
            "save_predictions_interval_epochs",
            "reference_processor_min_pixels",
            "reference_processor_max_pixels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "train_limit",
            "test_limit",
            "train_limit_per_class",
            "test_limit_per_class",
            "train_samples_per_class_per_epoch",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set")
        for name in ("learning_rate", "weight_decay"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path) -> Settings:
    config_path = resolve_path(path, Path.cwd(), "config")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "On Linux/macOS shells, use forward slashes in paths, for example "
            "experiments/bdd100k_timeofday3_standard_baselines/configs/"
            "bdd100k_timeofday3_standard_d2nn64.json."
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(raw)
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
