from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _resolve(value: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (expanded if expanded.is_absolute() else base / expanded).resolve()


@dataclass(frozen=True)
class OpticalConfig:
    canvas_size: int
    num_stages: int
    wavelength_m: float
    pixel_size_m: float
    propagation_distance_m: float
    phase_parameterization: str
    phase_init: str
    layernorm_eps: float
    residual_main_init: float
    residual_skip_init: float
    readout_pool_size: int
    readout_hidden_dim: int
    num_output_classes: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    torchvision_root: Path
    cifar100c_root: Path
    cifar100c_url: str
    selected_classes: tuple[str, ...]
    corruptions: tuple[str, ...]
    severity: int
    train_per_class: int
    validation_per_class: int
    test_per_class: int
    split_seed: int
    num_workers: int
    pin_memory: bool
    pretrain_samples_per_epoch: int | None
    pretrain_crop_padding: int
    pretrain_horizontal_flip: bool
    finetune_crop_padding: int
    finetune_horizontal_flip: bool


@dataclass(frozen=True)
class OptimizerConfig:
    phase_learning_rate: float
    electronic_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    gradient_clip_norm: float | None


@dataclass(frozen=True)
class TrainingConfig:
    pretrain_epochs: int
    finetune_epochs: int
    pretrain_batch_size: int
    finetune_batch_size: int
    pretrain_seed: int
    finetune_seeds: tuple[int, ...]
    log_interval_batches: int
    evaluation_interval_epochs: int
    checkpoint_interval_epochs: int
    diagnostic_epochs: tuple[int, ...]
    use_amp: bool


@dataclass(frozen=True)
class Settings:
    config_path: Path
    output_dir: Path
    optical: OpticalConfig
    data: DataConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    def validate(self) -> None:
        o, d, t, p = self.optical, self.data, self.training, self.optimizer
        if o.canvas_size != 400:
            raise ValueError("The formal experiment requires a 400x400 optical/CCD canvas")
        if o.num_stages != 20:
            raise ValueError("The formal experiment requires exactly 20 OEO optical stages")
        if o.phase_parameterization != "sigmoid":
            raise ValueError("Only sigmoid raw-phase parameterization is supported")
        if o.phase_init != "zeros":
            raise ValueError("The controlled experiment requires raw_phase=0 initialization")
        if not 0.0 < o.residual_main_init < 1.0 or not 0.0 < o.residual_skip_init < 1.0:
            raise ValueError("Residual branch initial weights must be in (0,1)")
        if abs(o.residual_main_init + o.residual_skip_init - 1.0) > 1e-6:
            raise ValueError("Residual branch initial weights must sum to one")
        if len(d.selected_classes) != 10 or len(set(d.selected_classes)) != 10:
            raise ValueError("Exactly ten distinct CIFAR-100 downstream classes are required")
        if not 1 <= d.severity <= 5:
            raise ValueError("CIFAR-100-C severity must be in [1,5]")
        if d.train_per_class + d.validation_per_class + d.test_per_class > 100:
            raise ValueError("CIFAR-100-C has only 100 base test images per class")
        if t.pretrain_epochs < 20 or t.finetune_epochs < 20:
            raise ValueError("This experiment intentionally disallows very short formal training")
        if t.use_amp:
            raise ValueError("AMP is disabled for complex64 FFT reproducibility")
        if p.weight_decay != 0.0:
            raise ValueError("Weight decay must remain zero for the controlled comparison")
        if p.gradient_clip_norm is not None:
            raise ValueError("Gradient clipping is disabled so gradient directions remain observable")

    def resolved_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        payload["output_dir"] = str(self.output_dir)
        for key in ("root", "torchvision_root", "cifar100c_root"):
            payload["data"][key] = str(payload["data"][key])
        return payload

    def digest(self) -> str:
        encoded = json.dumps(self.resolved_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    optical = OpticalConfig(
        canvas_size=int(_get(raw, "optical.canvas_size", 400)),
        num_stages=int(_get(raw, "optical.num_stages", 20)),
        wavelength_m=float(_get(raw, "optical.wavelength_m", 532e-9)),
        pixel_size_m=float(_get(raw, "optical.pixel_size_m", 16e-6)),
        propagation_distance_m=float(_get(raw, "optical.propagation_distance_m", 0.05)),
        phase_parameterization=str(_get(raw, "optical.phase_parameterization", "sigmoid")),
        phase_init=str(_get(raw, "optical.phase_init", "zeros")),
        layernorm_eps=float(_get(raw, "optical.layernorm_eps", 1e-5)),
        residual_main_init=float(_get(raw, "optical.residual.main_init", 0.9)),
        residual_skip_init=float(_get(raw, "optical.residual.skip_init", 0.1)),
        readout_pool_size=int(_get(raw, "optical.readout.pool_size", 20)),
        readout_hidden_dim=int(_get(raw, "optical.readout.hidden_dim", 128)),
        num_output_classes=int(_get(raw, "optical.readout.num_output_classes", 100)),
    )
    data_root = _resolve(_get(raw, "data.root", "data/cifar100_fixed_feedback"))
    data = DataConfig(
        root=data_root,
        torchvision_root=_resolve(_get(raw, "data.torchvision_root", str(data_root / "torchvision"))),
        cifar100c_root=_resolve(_get(raw, "data.cifar100c_root", str(data_root / "CIFAR-100-C"))),
        cifar100c_url=str(
            _get(
                raw,
                "data.cifar100c_url",
                "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
            )
        ),
        selected_classes=tuple(str(v) for v in _get(raw, "data.selected_classes", [])),
        corruptions=tuple(str(v) for v in _get(raw, "data.corruptions", ["gaussian_noise"])),
        severity=int(_get(raw, "data.severity", 3)),
        train_per_class=int(_get(raw, "data.split.train_per_class", 60)),
        validation_per_class=int(_get(raw, "data.split.validation_per_class", 20)),
        test_per_class=int(_get(raw, "data.split.test_per_class", 20)),
        split_seed=int(_get(raw, "data.split.seed", 42)),
        num_workers=int(_get(raw, "data.num_workers", 8)),
        pin_memory=bool(_get(raw, "data.pin_memory", True)),
        pretrain_samples_per_epoch=(
            None
            if _get(raw, "data.pretrain_samples_per_epoch", None) is None
            else int(_get(raw, "data.pretrain_samples_per_epoch"))
        ),
        pretrain_crop_padding=int(_get(raw, "data.augmentation.pretrain_crop_padding", 4)),
        pretrain_horizontal_flip=bool(_get(raw, "data.augmentation.pretrain_horizontal_flip", True)),
        finetune_crop_padding=int(_get(raw, "data.augmentation.finetune_crop_padding", 2)),
        finetune_horizontal_flip=bool(_get(raw, "data.augmentation.finetune_horizontal_flip", True)),
    )
    optimizer = OptimizerConfig(
        phase_learning_rate=float(_get(raw, "optimizer.phase_learning_rate", 1e-2)),
        electronic_learning_rate=float(_get(raw, "optimizer.electronic_learning_rate", 1e-3)),
        weight_decay=float(_get(raw, "optimizer.weight_decay", 0.0)),
        betas=tuple(float(v) for v in _get(raw, "optimizer.betas", [0.9, 0.999])),
        eps=float(_get(raw, "optimizer.eps", 1e-8)),
        gradient_clip_norm=(
            None
            if _get(raw, "optimizer.gradient_clip_norm", None) is None
            else float(_get(raw, "optimizer.gradient_clip_norm"))
        ),
    )
    training = TrainingConfig(
        pretrain_epochs=int(_get(raw, "training.pretrain_epochs", 80)),
        finetune_epochs=int(_get(raw, "training.finetune_epochs", 50)),
        pretrain_batch_size=int(_get(raw, "training.pretrain_batch_size", 8)),
        finetune_batch_size=int(_get(raw, "training.finetune_batch_size", 8)),
        pretrain_seed=int(_get(raw, "training.pretrain_seed", 42)),
        finetune_seeds=tuple(int(v) for v in _get(raw, "training.finetune_seeds", [1234, 2345, 3456])),
        log_interval_batches=int(_get(raw, "training.log_interval_batches", 100)),
        evaluation_interval_epochs=int(_get(raw, "training.evaluation_interval_epochs", 1)),
        checkpoint_interval_epochs=int(_get(raw, "training.checkpoint_interval_epochs", 10)),
        diagnostic_epochs=tuple(int(v) for v in _get(raw, "training.diagnostic_epochs", [0, 1, 5, 10, 20, 30, 40, 50])),
        use_amp=bool(_get(raw, "training.use_amp", False)),
    )
    settings = Settings(
        config_path=config_path,
        output_dir=_resolve(_get(raw, "output_dir", str(EXPERIMENT_DIR / "runs" / "main"))),
        optical=optical,
        data=data,
        optimizer=optimizer,
        training=training,
    )
    settings.validate()
    return settings
