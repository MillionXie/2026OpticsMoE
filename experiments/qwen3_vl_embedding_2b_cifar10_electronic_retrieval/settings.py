from __future__ import annotations

from pathlib import Path

import yaml

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.settings import (
    load_settings as load_electronic_settings,
    save_resolved_config as save_electronic_resolved_config,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    Caltech101Settings,
)


CIFAR10_CLASSES = (
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


class CIFAR10Settings(Caltech101Settings):
    @property
    def dataset_variant(self) -> str:
        return "cifar10_retrieval"

    @property
    def materialized_image_root(self) -> Path:
        return self.dataset_root / "retrieval_png"


def load_settings(path: str | Path) -> CIFAR10Settings:
    settings = load_electronic_settings(path)
    if tuple(settings.selected_skus) != CIFAR10_CLASSES:
        raise ValueError("CIFAR-10 class order must match the official label order")
    if settings.teacher_enabled:
        raise ValueError("The CIFAR-10 electronic experiment does not use a teacher")
    settings.__class__ = CIFAR10Settings
    return settings


def save_resolved_config(settings: CIFAR10Settings) -> None:
    save_electronic_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["dataset"]["name"] = "CIFAR-10"
    values["dataset"]["official_train_used_for"] = "gallery_then_train"
    values["dataset"]["official_test_used_for"] = "test_only"
    values["dataset"]["materialized_image_root"] = str(
        settings.materialized_image_root
    )
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
