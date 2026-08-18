from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class OpticalConfig:
    canvas_size: int
    input_channels: int
    num_stages: int
    wavelength_m: float
    pixel_size_m: float
    propagation_distance_m: float
    phase_init_std: float
    layernorm_eps: float
    residual_mode: str
    residual_main_init: float
    residual_main_min: float
    normalize_branch_rms: bool
    electronic_skip_mode: str
    electronic_skip_hidden_channels: int
    electronic_skip_scale_init: float
    electronic_skip_scale_max: float
    long_skip_enabled: bool
    long_skip_weight_init: float
    long_skip_weight_max: float
    readout_mode: str
    pool_size: int
    hidden_dim: int
    conv_channels: int
    dropout: float


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    root: Path
    validation_per_class: int
    split_seed: int
    num_workers: int
    pin_memory: bool
    crop_padding: int
    horizontal_flip: bool


@dataclass(frozen=True)
class OptimizerConfig:
    phase_learning_rate: float
    residual_learning_rate: float
    electronic_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_epochs: int
    min_learning_rate_ratio: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    seeds: tuple[int, ...]
    batch_size: int
    evaluation_batch_size: int
    label_smoothing: float
    use_amp: bool
    log_interval_batches: int
    checkpoint_interval_epochs: int
    gradient_clip_norm: float
    max_train_batches: int | None
    max_evaluation_batches: int | None
    init_checkpoint: Path | None
    load_backbone_only: bool


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    optical: OpticalConfig
    data: DataConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    @property
    def num_classes(self) -> int:
        return 10 if self.data.dataset == "cifar10" else 100

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    optical = OpticalConfig(
        canvas_size=int(_get(raw, "optical.canvas_size", 256)),
        input_channels=int(_get(raw, "optical.input_channels", 3)),
        num_stages=int(_get(raw, "optical.num_stages", 8)),
        wavelength_m=float(_get(raw, "optical.wavelength_m", 5.32e-7)),
        pixel_size_m=float(_get(raw, "optical.pixel_size_m", 1.6e-5)),
        propagation_distance_m=float(_get(raw, "optical.propagation_distance_m", 0.05)),
        phase_init_std=float(_get(raw, "optical.phase_init_std", 0.05)),
        layernorm_eps=float(_get(raw, "optical.layernorm_eps", 1e-5)),
        residual_mode=str(_get(raw, "optical.residual.mode", "constrained")),
        residual_main_init=float(_get(raw, "optical.residual.main_init", 0.5)),
        residual_main_min=float(_get(raw, "optical.residual.main_min", 0.35)),
        normalize_branch_rms=bool(_get(raw, "optical.normalize_branch_rms", True)),
        electronic_skip_mode=str(_get(raw, "optical.residual.electronic.mode", "identity")),
        electronic_skip_hidden_channels=int(
            _get(raw, "optical.residual.electronic.hidden_channels", 12)
        ),
        electronic_skip_scale_init=float(
            _get(raw, "optical.residual.electronic.scale_init", 0.10)
        ),
        electronic_skip_scale_max=float(
            _get(raw, "optical.residual.electronic.scale_max", 0.25)
        ),
        long_skip_enabled=bool(_get(raw, "optical.residual.long_skip.enabled", False)),
        long_skip_weight_init=float(
            _get(raw, "optical.residual.long_skip.weight_init", 0.10)
        ),
        long_skip_weight_max=float(
            _get(raw, "optical.residual.long_skip.weight_max", 0.25)
        ),
        readout_mode=str(_get(raw, "optical.readout.mode", "mlp")),
        pool_size=int(_get(raw, "optical.readout.pool_size", 16)),
        hidden_dim=int(_get(raw, "optical.readout.hidden_dim", 512)),
        conv_channels=int(_get(raw, "optical.readout.conv_channels", 32)),
        dropout=float(_get(raw, "optical.readout.dropout", 0.1)),
    )
    data = DataConfig(
        dataset=str(_get(raw, "data.dataset", "cifar10")).lower(),
        root=_resolve(_get(raw, "data.root", "data/cifar_performance_optical")),
        validation_per_class=int(_get(raw, "data.validation_per_class", 500)),
        split_seed=int(_get(raw, "data.split_seed", 42)),
        num_workers=int(_get(raw, "data.num_workers", 8)),
        pin_memory=bool(_get(raw, "data.pin_memory", True)),
        crop_padding=int(_get(raw, "data.augmentation.crop_padding", 4)),
        horizontal_flip=bool(_get(raw, "data.augmentation.horizontal_flip", True)),
    )
    optimizer = OptimizerConfig(
        phase_learning_rate=float(_get(raw, "optimizer.phase_learning_rate", 0.003)),
        residual_learning_rate=float(_get(raw, "optimizer.residual_learning_rate", 0.001)),
        electronic_learning_rate=float(_get(raw, "optimizer.electronic_learning_rate", 0.001)),
        weight_decay=float(_get(raw, "optimizer.weight_decay", 1e-4)),
        betas=tuple(float(x) for x in _get(raw, "optimizer.betas", [0.9, 0.999])),
        eps=float(_get(raw, "optimizer.eps", 1e-8)),
        warmup_epochs=int(_get(raw, "optimizer.warmup_epochs", 5)),
        min_learning_rate_ratio=float(_get(raw, "optimizer.min_learning_rate_ratio", 0.05)),
    )
    init_value = _get(raw, "training.init_checkpoint", "")
    training = TrainingConfig(
        epochs=int(_get(raw, "training.epochs", 80)),
        seeds=tuple(int(x) for x in _get(raw, "training.seeds", [1234])),
        batch_size=int(_get(raw, "training.batch_size", 32)),
        evaluation_batch_size=int(_get(raw, "training.evaluation_batch_size", 64)),
        label_smoothing=float(_get(raw, "training.label_smoothing", 0.1)),
        use_amp=bool(_get(raw, "training.use_amp", True)),
        log_interval_batches=int(_get(raw, "training.log_interval_batches", 100)),
        checkpoint_interval_epochs=int(_get(raw, "training.checkpoint_interval_epochs", 10)),
        gradient_clip_norm=float(_get(raw, "training.gradient_clip_norm", 5.0)),
        max_train_batches=(
            int(_get(raw, "training.max_train_batches"))
            if _get(raw, "training.max_train_batches") is not None
            else None
        ),
        max_evaluation_batches=(
            int(_get(raw, "training.max_evaluation_batches"))
            if _get(raw, "training.max_evaluation_batches") is not None
            else None
        ),
        init_checkpoint=_resolve(init_value) if init_value else None,
        load_backbone_only=bool(_get(raw, "training.load_backbone_only", False)),
    )

    if data.dataset not in {"cifar10", "cifar100"}:
        raise ValueError("data.dataset must be cifar10 or cifar100")
    if optical.input_channels not in {1, 3}:
        raise ValueError("optical.input_channels must be 1 or 3")
    if optical.residual_mode not in {"fixed", "learned", "constrained", "none"}:
        raise ValueError("Unsupported residual mode")
    if optical.electronic_skip_mode not in {"identity", "pointwise", "depthwise"}:
        raise ValueError("Unsupported electronic residual mode")
    if optical.readout_mode not in {"mlp", "conv"}:
        raise ValueError("Unsupported readout mode")
    if not (0.0 <= optical.residual_main_min <= optical.residual_main_init <= 1.0):
        raise ValueError("Residual weights must satisfy 0 <= min <= init <= 1")
    if not (
        0.0 <= optical.electronic_skip_scale_init <= optical.electronic_skip_scale_max <= 1.0
    ):
        raise ValueError("Electronic residual scales must satisfy 0 <= init <= max <= 1")
    if not (0.0 <= optical.long_skip_weight_init <= optical.long_skip_weight_max <= 1.0):
        raise ValueError("Long-skip weights must satisfy 0 <= init <= max <= 1")
    if optical.electronic_skip_hidden_channels < optical.input_channels:
        raise ValueError("electronic hidden_channels must be at least input_channels")
    if optical.conv_channels < 1:
        raise ValueError("readout conv_channels must be positive")
    if len(optimizer.betas) != 2:
        raise ValueError("optimizer.betas must contain two values")

    return Settings(
        output_dir=_resolve(_get(raw, "output_dir", "experiments/d2nn_cifar10_high_performance_optical_backbone/runs/main")),
        optical=optical,
        data=data,
        optimizer=optimizer,
        training=training,
    )
