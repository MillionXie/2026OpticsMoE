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


def _resolve(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (expanded if expanded.is_absolute() else REPOSITORY_ROOT / expanded).resolve()


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
    embedding_dim: int
    embedding_dropout: float


@dataclass(frozen=True)
class BalancedBatchConfig:
    classes_per_batch: int
    images_per_class: int
    views_per_image: int
    batches_per_epoch: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    torchvision_root: Path
    num_workers: int
    pin_memory: bool
    cifar100_validation_per_class: int
    cifar10_support_per_class: int
    cifar10_validation_per_class: int
    split_seed: int
    crop_padding: int
    horizontal_flip: bool


@dataclass(frozen=True)
class LossConfig:
    contrastive_temperature: float
    prototype_temperature: float
    pretrain_supcon_weight: float
    finetune_supcon_weight: float
    finetune_prototype_weight: float


@dataclass(frozen=True)
class OptimizerConfig:
    pretrain_phase_learning_rate: float
    finetune_phase_learning_rate: float
    electronic_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


@dataclass(frozen=True)
class TrainingConfig:
    pretrain_epochs: int
    finetune_epochs: int
    pretrain_seed: int
    finetune_seeds: tuple[int, ...]
    pretrain_batch: BalancedBatchConfig
    finetune_batch: BalancedBatchConfig
    validation_batches: int
    evaluation_batch_size: int
    log_interval_batches: int
    checkpoint_interval_epochs: int
    diagnostic_epochs: tuple[int, ...]
    use_amp: bool


@dataclass(frozen=True)
class Settings:
    config_path: Path
    output_dir: Path
    optical: OpticalConfig
    data: DataConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    def validate(self) -> None:
        o, d, loss, opt, train = self.optical, self.data, self.loss, self.optimizer, self.training
        if o.canvas_size != 400 or o.num_stages != 20:
            raise ValueError("The formal experiment requires 20 stages on a 400x400 canvas")
        if o.phase_parameterization != "sigmoid" or o.phase_init != "zeros":
            raise ValueError("The controlled experiment requires sigmoid phase with raw_phase=0")
        if min(o.residual_main_init, o.residual_skip_init) <= 0 or abs(
            o.residual_main_init + o.residual_skip_init - 1.0
        ) > 1e-6:
            raise ValueError("Residual initialization must be positive and sum to one")
        if not 0.0 <= o.embedding_dropout < 1.0:
            raise ValueError("embedding_dropout must be in [0,1)")
        if o.embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")
        if d.cifar100_validation_per_class >= 500:
            raise ValueError("CIFAR-100 has only 500 train images per class")
        if d.cifar10_support_per_class + d.cifar10_validation_per_class >= 5000:
            raise ValueError("CIFAR-10 has only 5000 train images per class")
        for batch, classes in ((train.pretrain_batch, 100), (train.finetune_batch, 10)):
            if not 2 <= batch.classes_per_batch <= classes:
                raise ValueError("classes_per_batch is outside the dataset class range")
            if batch.images_per_class < 2 or batch.views_per_image < 2 or batch.batches_per_epoch < 1:
                raise ValueError("Contrastive batches require K>=2, views>=2 and at least one batch")
        if min(loss.contrastive_temperature, loss.prototype_temperature) <= 0:
            raise ValueError("Contrastive temperatures must be positive")
        if min(loss.pretrain_supcon_weight, loss.finetune_supcon_weight, loss.finetune_prototype_weight) < 0:
            raise ValueError("Loss weights cannot be negative")
        if train.pretrain_epochs < 20 or train.finetune_epochs < 10:
            raise ValueError("Formal training schedules are intentionally not short")
        if train.use_amp:
            raise ValueError("AMP remains disabled for complex64 FFT reproducibility")
        if opt.weight_decay != 0.0:
            raise ValueError("Weight decay is zero for the controlled comparison")

    def resolved_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        payload["output_dir"] = str(self.output_dir)
        payload["data"]["root"] = str(payload["data"]["root"])
        payload["data"]["torchvision_root"] = str(payload["data"]["torchvision_root"])
        return payload

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.resolved_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def _batch(raw: dict[str, Any], prefix: str, defaults: tuple[int, int, int, int]) -> BalancedBatchConfig:
    p, k, views, batches = defaults
    return BalancedBatchConfig(
        classes_per_batch=int(_get(raw, f"training.{prefix}.classes_per_batch", p)),
        images_per_class=int(_get(raw, f"training.{prefix}.images_per_class", k)),
        views_per_image=int(_get(raw, f"training.{prefix}.views_per_image", views)),
        batches_per_epoch=int(_get(raw, f"training.{prefix}.batches_per_epoch", batches)),
    )


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    data_root = _resolve(_get(raw, "data.root", "data/cifar_contrastive_fixed_feedback"))
    settings = Settings(
        config_path=config_path,
        output_dir=_resolve(
            _get(
                raw,
                "output_dir",
                "experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/runs/main",
            )
        ),
        optical=OpticalConfig(
            canvas_size=int(_get(raw, "optical.canvas_size", 400)),
            num_stages=int(_get(raw, "optical.num_stages", 20)),
            wavelength_m=float(_get(raw, "optical.wavelength_m", 532e-9)),
            pixel_size_m=float(_get(raw, "optical.pixel_size_m", 16e-6)),
            propagation_distance_m=float(_get(raw, "optical.propagation_distance_m", 0.05)),
            phase_parameterization=str(_get(raw, "optical.phase_parameterization", "sigmoid")),
            phase_init=str(_get(raw, "optical.phase_init", "zeros")),
            layernorm_eps=float(_get(raw, "optical.layernorm_eps", 1e-5)),
            residual_main_init=float(_get(raw, "optical.residual.main_init", 0.35)),
            residual_skip_init=float(_get(raw, "optical.residual.skip_init", 0.65)),
            readout_pool_size=int(_get(raw, "optical.readout.pool_size", 20)),
            embedding_dim=int(_get(raw, "optical.readout.embedding_dim", 128)),
            embedding_dropout=float(_get(raw, "optical.readout.dropout", 0.1)),
        ),
        data=DataConfig(
            root=data_root,
            torchvision_root=_resolve(_get(raw, "data.torchvision_root", str(data_root / "torchvision"))),
            num_workers=int(_get(raw, "data.num_workers", 8)),
            pin_memory=bool(_get(raw, "data.pin_memory", True)),
            cifar100_validation_per_class=int(_get(raw, "data.cifar100_validation_per_class", 50)),
            cifar10_support_per_class=int(_get(raw, "data.cifar10.support_per_class", 100)),
            cifar10_validation_per_class=int(_get(raw, "data.cifar10.validation_per_class", 200)),
            split_seed=int(_get(raw, "data.split_seed", 42)),
            crop_padding=int(_get(raw, "data.augmentation.crop_padding", 4)),
            horizontal_flip=bool(_get(raw, "data.augmentation.horizontal_flip", True)),
        ),
        loss=LossConfig(
            contrastive_temperature=float(_get(raw, "loss.contrastive_temperature", 0.1)),
            prototype_temperature=float(_get(raw, "loss.prototype_temperature", 0.1)),
            pretrain_supcon_weight=float(_get(raw, "loss.pretrain_supcon_weight", 1.0)),
            finetune_supcon_weight=float(_get(raw, "loss.finetune_supcon_weight", 1.0)),
            finetune_prototype_weight=float(_get(raw, "loss.finetune_prototype_weight", 0.5)),
        ),
        optimizer=OptimizerConfig(
            pretrain_phase_learning_rate=float(_get(raw, "optimizer.pretrain_phase_learning_rate", 0.01)),
            finetune_phase_learning_rate=float(_get(raw, "optimizer.finetune_phase_learning_rate", 0.003)),
            electronic_learning_rate=float(_get(raw, "optimizer.electronic_learning_rate", 0.001)),
            weight_decay=float(_get(raw, "optimizer.weight_decay", 0.0)),
            betas=tuple(float(v) for v in _get(raw, "optimizer.betas", [0.9, 0.999])),
            eps=float(_get(raw, "optimizer.eps", 1e-8)),
        ),
        training=TrainingConfig(
            pretrain_epochs=int(_get(raw, "training.pretrain_epochs", 120)),
            finetune_epochs=int(_get(raw, "training.finetune_epochs", 30)),
            pretrain_seed=int(_get(raw, "training.pretrain_seed", 42)),
            finetune_seeds=tuple(int(v) for v in _get(raw, "training.finetune_seeds", [1234, 2345, 3456])),
            pretrain_batch=_batch(raw, "pretrain_batch", (16, 4, 2, 125)),
            finetune_batch=_batch(raw, "finetune_batch", (10, 6, 2, 50)),
            validation_batches=int(_get(raw, "training.validation_batches", 20)),
            evaluation_batch_size=int(_get(raw, "training.evaluation_batch_size", 128)),
            log_interval_batches=int(_get(raw, "training.log_interval_batches", 25)),
            checkpoint_interval_epochs=int(_get(raw, "training.checkpoint_interval_epochs", 10)),
            diagnostic_epochs=tuple(int(v) for v in _get(raw, "training.diagnostic_epochs", [0, 1, 5, 10, 20, 30])),
            use_amp=bool(_get(raw, "training.use_amp", False)),
        ),
    )
    settings.validate()
    return settings
