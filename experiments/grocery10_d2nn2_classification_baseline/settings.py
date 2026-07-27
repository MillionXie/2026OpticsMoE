from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent


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


def _resolve(value: str | Path, base: Path) -> Path:
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

    image_size: int
    input_encoding: str
    canvas_size: int
    active_size: int
    first_phase_size: int
    wavelength_nm: float
    pixel_pitch_um: float
    input_to_first_phase_distance_m: float
    first_to_second_phase_distance_m: float
    second_phase_to_detector_distance_m: float
    phase_parameterization: str
    phase_init: str
    phase_init_std: float
    k_space_constraint_enabled: bool
    theta_max_deg: float

    detector_row_layout: tuple[int, ...]
    detector_size: int
    detector_horizontal_gap: int
    detector_vertical_gap: int
    detector_normalize_total_energy: bool
    detector_eps: float

    loss_type: str
    loss_eps: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    min_learning_rate: float
    epochs: int
    batch_size: int
    inference_batch_size: int
    num_workers: int
    class_balanced_sampling: bool
    random_seed: int
    evaluate_test_each_epoch: bool
    log_interval_batches: int

    augmentation_enabled: bool
    crop_scale_min: float
    brightness_jitter: float
    contrast_jitter: float
    rotation_degrees: float
    save_visualization_interval_epochs: int
    visualization_sample_count: int
    device: str

    @property
    def dataset_variant(self) -> str:
        return f"grocery{len(self.selected_skus)}"

    @property
    def subset_manifest_path(self) -> Path:
        return self.output_dir / "manifests" / f"{self.dataset_variant}_subset.csv"

    @property
    def active_guard_pixels(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    def validate(self) -> None:
        if len(self.selected_skus) != 10 or len(set(self.selected_skus)) != 10:
            raise ValueError("This baseline requires exactly 10 unique selected_skus")
        if self.gallery_images_per_sku != 1:
            raise ValueError("Grocery Store supplies exactly one iconic gallery image per SKU")
        if self.input_encoding != "grayscale_amplitude":
            raise ValueError("input.encoding must be grayscale_amplitude for the one-shot scalar field")
        for name in (
            "image_size",
            "canvas_size",
            "active_size",
            "first_phase_size",
            "detector_size",
            "epochs",
            "batch_size",
            "inference_batch_size",
            "log_interval_batches",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.canvas_size < self.active_size or (self.canvas_size - self.active_size) % 2:
            raise ValueError("canvas_size-active_size must be nonnegative and even")
        if self.active_size < self.first_phase_size or (
            self.canvas_size - self.first_phase_size
        ) % 2:
            raise ValueError("first_phase_size must be centered inside the canvas")
        if self.image_size != self.first_phase_size:
            raise ValueError(
                "image_size must equal first_phase_size so no spatial resampling occurs inside the model"
            )
        if self.canvas_size != 1026 or self.active_size != 986 or self.first_phase_size != 224:
            raise ValueError(
                "The formal experimental-path baseline fixes canvas/active/first phase to 1026/986/224"
            )
        if self.input_to_first_phase_distance_m < 0:
            raise ValueError("input_to_first_phase_distance_m cannot be negative")
        for name in (
            "first_to_second_phase_distance_m",
            "second_phase_to_detector_distance_m",
            "wavelength_nm",
            "pixel_pitch_um",
            "detector_eps",
            "loss_eps",
            "learning_rate",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.phase_parameterization not in {"sigmoid", "unconstrained"}:
            raise ValueError("Unsupported phase parameterization")
        if self.phase_init not in {"zeros", "uniform", "normal"}:
            raise ValueError("phase_init must be zeros, uniform, or normal")
        if sum(self.detector_row_layout) != 10 or tuple(self.detector_row_layout) != (
            3,
            4,
            3,
        ):
            raise ValueError("The Grocery-10 detector layout must be [3,4,3]")
        if self.loss_type != "detector_region_cross_entropy":
            raise ValueError("Only detector_region_cross_entropy is supported")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("augmentation.crop_scale_min must be in (0,1]")

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
        image_size=int(d("input.image_size", 224)),
        input_encoding=str(d("input.encoding", "grayscale_amplitude")),
        canvas_size=int(d("optics.canvas_size", 1026)),
        active_size=int(d("optics.active_size", 986)),
        first_phase_size=int(d("optics.first_phase_size", 224)),
        wavelength_nm=float(d("optics.wavelength_nm", 532.0)),
        pixel_pitch_um=float(d("optics.pixel_pitch_um", 8.0)),
        input_to_first_phase_distance_m=float(
            d("optics.input_to_first_phase_distance_m", 0.0)
        ),
        first_to_second_phase_distance_m=float(
            d("optics.first_to_second_phase_distance_m", 0.10)
        ),
        second_phase_to_detector_distance_m=float(
            d("optics.second_phase_to_detector_distance_m", 0.10)
        ),
        phase_parameterization=str(d("optics.phase.parameterization", "sigmoid")),
        phase_init=str(d("optics.phase.init", "zeros")),
        phase_init_std=float(d("optics.phase.init_std", 0.02)),
        k_space_constraint_enabled=bool(d("optics.k_space.enabled", False)),
        theta_max_deg=float(d("optics.k_space.theta_max_deg", 1.0)),
        detector_row_layout=tuple(int(value) for value in d("detector.row_layout", [3, 4, 3])),
        detector_size=int(d("detector.region_size", 120)),
        detector_horizontal_gap=int(d("detector.horizontal_gap", 60)),
        detector_vertical_gap=int(d("detector.vertical_gap", 100)),
        detector_normalize_total_energy=bool(
            d("detector.normalize_total_energy", True)
        ),
        detector_eps=float(d("detector.eps", 1e-8)),
        loss_type=str(d("loss.type", "detector_region_cross_entropy")),
        loss_eps=float(d("loss.eps", 1e-8)),
        optimizer=str(d("training.optimizer", "adam")),
        learning_rate=float(d("training.learning_rate", 0.01)),
        weight_decay=float(d("training.weight_decay", 0.0)),
        scheduler=str(d("training.scheduler", "cosine")),
        min_learning_rate=float(d("training.min_learning_rate", 1e-5)),
        epochs=int(d("training.epochs", 100)),
        batch_size=int(d("training.batch_size", 4)),
        inference_batch_size=int(d("training.inference_batch_size", 4)),
        num_workers=int(d("training.num_workers", 4)),
        class_balanced_sampling=bool(d("training.class_balanced_sampling", True)),
        random_seed=int(d("training.random_seed", 42)),
        evaluate_test_each_epoch=bool(d("training.evaluate_test_each_epoch", True)),
        log_interval_batches=int(d("training.log_interval_batches", 20)),
        augmentation_enabled=bool(d("augmentation.enabled", True)),
        crop_scale_min=float(d("augmentation.crop_scale_min", 0.90)),
        brightness_jitter=float(d("augmentation.brightness_jitter", 0.10)),
        contrast_jitter=float(d("augmentation.contrast_jitter", 0.10)),
        rotation_degrees=float(d("augmentation.rotation_degrees", 5.0)),
        save_visualization_interval_epochs=int(
            d("visualization.save_interval_epochs", 10)
        ),
        visualization_sample_count=int(d("visualization.sample_count", 8)),
        device=str(d("device", "cuda")),
    )
    settings.validate()
    return settings
