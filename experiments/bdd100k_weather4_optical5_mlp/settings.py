from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhaseDropoutSettings:
    enabled: bool = True
    p: float = 0.05
    mode: str = "block_phase_bypass"
    block_size: int = 8
    batch_shared: bool = True
    start_epoch: int = 10


@dataclass
class Settings:
    dataset: str = "imagefolder"
    data_root: str = "../data/bdd100k_weather4"
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    output_dir: str = "../runs/bdd100k_weather4_optical5_mlp"
    input_size: int = 224
    optical_layers: int = 5
    optical_field_size: int = 256
    optical_padding_size: int = 400
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    mask_distance_cm: float = 5.0
    phase_init: str = "uniform"
    amplitude_mask_enabled: bool = True
    detector_pool_size: int = 16
    mlp_hidden_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 4
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    validation_fraction: float = 0.1
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    seed: int = 42
    device: str = "cuda"
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    save_interval_epochs: int = 10
    progress: bool = True
    regularization: dict[str, Any] = field(
        default_factory=lambda: {"phase_dropout": asdict(PhaseDropoutSettings())}
    )

    @property
    def phase_dropout(self) -> PhaseDropoutSettings:
        value = self.regularization.get("phase_dropout", {}) or {}
        return PhaseDropoutSettings(**value)

    def validate(self) -> None:
        if self.dataset != "imagefolder":
            raise ValueError("Only dataset='imagefolder' is supported")
        if self.optical_layers != 5:
            raise ValueError("This baseline requires exactly optical_layers=5")
        if self.num_classes != 4:
            raise ValueError("BDD100K Weather-4 requires num_classes=4")
        if self.optical_padding_size < self.optical_field_size:
            raise ValueError("optical_padding_size must be >= optical_field_size")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.epochs < 1 or self.batch_size < 1 or self.save_interval_epochs < 1:
            raise ValueError("epochs, batch_size, and save_interval_epochs must be positive")
        if self.optimizer.lower() != "adamw":
            raise ValueError("Only optimizer='adamw' is supported")
        if self.scheduler.lower() != "cosine":
            raise ValueError("Only scheduler='cosine' is supported")
        dropout = self.phase_dropout
        if dropout.mode not in {"none", "phase_bypass", "block_phase_bypass"}:
            raise ValueError(f"Unsupported phase dropout mode: {dropout.mode}")
        if not 0.0 <= dropout.p < 1.0:
            raise ValueError("phase dropout p must satisfy 0 <= p < 1")


def load_settings(config_path: str | Path) -> tuple[Settings, Path]:
    path = Path(config_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = set(Settings.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    settings = Settings(**raw)
    settings.data_root = str(_resolve_config_path(settings.data_root, path))
    settings.output_dir = str(_resolve_config_path(settings.output_dir, path))
    settings.validate()
    return settings, path


def _resolve_config_path(value: str, config_path: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def resolved_dict(settings: Settings) -> dict[str, Any]:
    return asdict(settings)
