from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
SUPPORTED_MODELS = {"resnet18", "efficientnet_b0", "mobilenet_v3_small"}
SUPPORTED_WEIGHTS = {"imagenet1k", "none"}


def _nested(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic base_config reference involving {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    parent = raw.pop("base_config", None)
    if parent is None:
        return raw
    parent_path = Path(os.path.expandvars(os.path.expanduser(str(parent))))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_update(_read_config(parent_path, seen), raw)


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


@dataclass
class Settings:
    config_path: Path
    dataset_root: Path
    output_dir: Path
    selected_skus: tuple[str, ...]
    download: bool
    download_url: str
    merge_official_validation_into_train: bool
    train_limit_per_sku: int | None
    test_limit_per_sku: int | None
    gallery_images_per_sku: int

    model_name: str
    weights: str
    embedding_dim: int
    image_size: int
    train_backbone: bool

    batch_size: int
    pk_skus_per_batch: int
    pk_images_per_sku: int
    inference_batch_size: int
    num_workers: int

    epochs: int
    backbone_learning_rate: float
    projection_learning_rate: float
    weight_decay: float
    lambda_ret: float
    lambda_gallery: float
    temperature: float
    gallery_temperature: float
    gallery_prototype_stop_gradient: bool
    scheduler: str
    min_learning_rate: float
    random_seed: int
    amp_enabled: bool
    evaluate_test_each_epoch: bool
    log_interval_batches: int

    augmentation_enabled: bool
    crop_scale_min: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float
    gallery_aggregation: str
    visualization_sample_count: int
    device: str
    optical_metrics_path: Path | None

    @property
    def dataset_variant(self) -> str:
        return f"grocery{len(self.selected_skus)}"

    @property
    def subset_manifest_path(self) -> Path:
        return self.output_dir / "manifests" / f"{self.dataset_variant}_subset.csv"

    def validate(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"model.name must be one of {sorted(SUPPORTED_MODELS)}")
        if self.weights not in SUPPORTED_WEIGHTS:
            raise ValueError(f"model.weights must be one of {sorted(SUPPORTED_WEIGHTS)}")
        if len(self.selected_skus) < 2 or len(set(self.selected_skus)) != len(
            self.selected_skus
        ):
            raise ValueError("selected_skus must contain at least two unique names")
        if self.embedding_dim != 64:
            raise ValueError("The comparison protocol fixes embedding_dim=64")
        if self.batch_size != self.pk_skus_per_batch * self.pk_images_per_sku:
            raise ValueError("batch_size must equal P*K")
        if self.pk_skus_per_batch > len(self.selected_skus):
            raise ValueError("P cannot exceed selected SKU count")
        if self.pk_images_per_sku < 2:
            raise ValueError("K must be at least two for supervised contrastive loss")
        for name in (
            "image_size",
            "batch_size",
            "inference_batch_size",
            "epochs",
            "log_interval_batches",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "backbone_learning_rate",
            "projection_learning_rate",
            "temperature",
            "gallery_temperature",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lambda_ret < 0 or self.lambda_gallery < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")
        if self.gallery_aggregation not in {"mean_prototype", "max_similarity"}:
            raise ValueError("Unsupported gallery aggregation")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
            elif isinstance(item, tuple):
                value[key] = list(item)
        return value


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = _read_config(config_path)
    base = config_path.parent
    d = lambda key, default=None: _nested(raw, key, default)
    settings = Settings(
        config_path=config_path,
        dataset_root=_resolve(d("dataset.dataset_root"), base),
        output_dir=_resolve(d("output_dir"), base),
        selected_skus=tuple(str(value) for value in d("dataset.selected_skus", [])),
        download=bool(d("dataset.download", True)),
        download_url=str(d("dataset.download_url")),
        merge_official_validation_into_train=bool(
            d("dataset.merge_official_validation_into_train", True)
        ),
        train_limit_per_sku=d("dataset.train_limit_per_sku"),
        test_limit_per_sku=d("dataset.test_limit_per_sku"),
        gallery_images_per_sku=int(d("dataset.gallery_images_per_sku", 1)),
        model_name=str(d("model.name")),
        weights=str(d("model.weights", "imagenet1k")),
        embedding_dim=int(d("model.embedding_dim", 64)),
        image_size=int(d("model.image_size", 224)),
        train_backbone=bool(d("model.train_backbone", True)),
        batch_size=int(d("batching.batch_size", 30)),
        pk_skus_per_batch=int(d("batching.pk_skus_per_batch", 10)),
        pk_images_per_sku=int(d("batching.pk_images_per_sku", 3)),
        inference_batch_size=int(d("batching.inference_batch_size", 64)),
        num_workers=int(d("batching.num_workers", 4)),
        epochs=int(d("training.epochs", 100)),
        backbone_learning_rate=float(d("training.backbone_learning_rate", 1e-5)),
        projection_learning_rate=float(
            d("training.projection_learning_rate", 3e-4)
        ),
        weight_decay=float(d("training.weight_decay", 1e-4)),
        lambda_ret=float(d("training.lambda_ret", 1.0)),
        lambda_gallery=float(d("training.lambda_gallery", 0.25)),
        temperature=float(d("training.temperature", 0.07)),
        gallery_temperature=float(d("training.gallery_temperature", 0.15)),
        gallery_prototype_stop_gradient=bool(
            d("training.gallery_prototype_stop_gradient", True)
        ),
        scheduler=str(d("training.scheduler", "cosine")),
        min_learning_rate=float(d("training.min_learning_rate", 1e-6)),
        random_seed=int(d("training.random_seed", 42)),
        amp_enabled=bool(d("training.amp_enabled", True)),
        evaluate_test_each_epoch=bool(
            d("training.evaluate_test_each_epoch", True)
        ),
        log_interval_batches=int(d("training.log_interval_batches", 5)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.90)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.10)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.10)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        gallery_aggregation=str(
            d("retrieval.gallery_aggregation", "mean_prototype")
        ),
        visualization_sample_count=int(d("visualization.sample_count", 8)),
        device=str(d("device", "cuda")),
        optical_metrics_path=_resolve(d("comparison.optical_metrics_path"), base),
    )
    settings.validate()
    return settings
