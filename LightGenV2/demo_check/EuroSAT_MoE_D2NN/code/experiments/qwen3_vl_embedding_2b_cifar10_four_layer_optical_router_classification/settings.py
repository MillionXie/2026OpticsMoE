from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


SOURCE_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval"
)
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _source_settings() -> Any:
    errors: list[BaseException] = []
    for name in (
        f"experiments.{SOURCE_PACKAGE}.settings",
        f"{SOURCE_PACKAGE}.settings",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            errors.append(error)
    raise ModuleNotFoundError("Could not import router experiment settings") from errors[-1]


_source = _source_settings()


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = _source.load_settings(config_path)
    raw = _source._read_config(config_path)
    d = lambda key, default=None: _source._nested(raw, key, default)

    settings.task = "cifar10_classification"
    settings.num_classes = int(d("classification.num_classes", 10))
    settings.class_names = tuple(d("classification.class_names", CLASS_NAMES))
    settings.split_seed = int(d("classification.split_seed", settings.random_seed))
    settings.validation_samples_per_class = int(
        d("classification.validation_samples_per_class", 500)
    )
    test_per_class = d("classification.test_samples_per_class", None)
    settings.test_samples_per_class = (
        None if test_per_class is None else int(test_per_class)
    )
    settings.data_download = bool(
        d("classification.download", d("dataset.download", True))
    )
    settings.train_augmentation = bool(
        d("classification.train_augmentation", True)
    )
    settings.num_workers = int(d("classification.num_workers", 4))
    settings.pin_memory = bool(d("classification.pin_memory", True))
    settings.label_smoothing = float(d("classification.label_smoothing", 0.0))
    settings.classification_head_seed = int(
        d("classification.head_seed", settings.random_seed + 1000)
    )
    settings.classification_head_learning_rate = float(
        d("classification.head_learning_rate", 0.001)
    )
    settings.classification_initialization = str(
        d("classification.initialization", "warmstart_body")
    )
    configured_steps = d(
        "classification.max_train_steps_per_epoch",
        settings.optimizer_steps_per_epoch,
    )
    settings.max_train_steps_per_epoch = (
        None if configured_steps is None else int(configured_steps)
    )
    settings.eval_every_epochs = int(d("classification.eval_every_epochs", 1))
    settings.log_every_steps = int(d("classification.log_every_steps", 1))
    settings.early_stopping_patience = int(
        d("classification.early_stopping_patience", 0)
    )
    settings.run_final_test = bool(d("classification.run_final_test", True))

    if settings.num_classes != 10 or len(settings.class_names) != 10:
        raise ValueError("CIFAR-10 classification requires exactly ten class names")
    if tuple(settings.class_names) != CLASS_NAMES:
        raise ValueError("class_names must use torchvision CIFAR-10 canonical order")
    if not 1 <= settings.validation_samples_per_class < 5000:
        raise ValueError("validation_samples_per_class must be in [1,4999]")
    if settings.test_samples_per_class is not None and not (
        1 <= settings.test_samples_per_class <= 1000
    ):
        raise ValueError("test_samples_per_class must be null or in [1,1000]")
    if settings.num_workers < 0:
        raise ValueError("classification.num_workers must be nonnegative")
    if not 0.0 <= settings.label_smoothing < 1.0:
        raise ValueError("classification.label_smoothing must be in [0,1)")
    if settings.classification_head_learning_rate <= 0.0:
        raise ValueError("classification.head_learning_rate must be positive")
    if settings.classification_initialization not in {"random", "warmstart_body"}:
        raise ValueError("classification.initialization must be random or warmstart_body")
    if settings.max_train_steps_per_epoch is not None and (
        settings.max_train_steps_per_epoch <= 0
    ):
        raise ValueError("max_train_steps_per_epoch must be positive or null")
    if settings.eval_every_epochs <= 0 or settings.log_every_steps <= 0:
        raise ValueError("evaluation and logging intervals must be positive")
    return settings


def save_resolved_config(settings: Any) -> None:
    _source.save_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["classification"] = {
        "task": settings.task,
        "num_classes": settings.num_classes,
        "class_names": list(settings.class_names),
        "dataset_root": str(settings.dataset_root),
        "split_seed": settings.split_seed,
        "validation_samples_per_class": settings.validation_samples_per_class,
        "test_samples_per_class": settings.test_samples_per_class,
        "download": settings.data_download,
        "train_augmentation": settings.train_augmentation,
        "num_workers": settings.num_workers,
        "label_smoothing": settings.label_smoothing,
        "head_seed": settings.classification_head_seed,
        "head_learning_rate": settings.classification_head_learning_rate,
        "initialization": settings.classification_initialization,
        "max_train_steps_per_epoch": settings.max_train_steps_per_epoch,
        "eval_every_epochs": settings.eval_every_epochs,
        "log_every_steps": settings.log_every_steps,
        "early_stopping_patience": settings.early_stopping_patience,
        "run_final_test": settings.run_final_test,
        "selection_criterion": "maximum_validation_accuracy_then_minimum_val_loss",
        "test_metrics_used_for_selection": False,
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["CLASS_NAMES", "load_settings", "save_resolved_config"]
