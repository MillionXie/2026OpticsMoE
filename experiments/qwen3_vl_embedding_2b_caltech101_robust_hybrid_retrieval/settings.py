from __future__ import annotations

from pathlib import Path

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    Settings,
    _nested,
    _read_config,
)
from experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval.settings import (
    load_settings as load_robust_settings,
    save_resolved_config as save_robust_resolved_config,
)

from .categories import CALTECH101_CATEGORIES


class Caltech101Settings(Settings):
    @property
    def dataset_variant(self) -> str:
        return f"caltech101_{len(self.selected_skus)}class"


def load_settings(path: str | Path) -> Caltech101Settings:
    config_path = Path(path).expanduser().resolve()
    settings = load_robust_settings(config_path)
    raw = _read_config(config_path)
    use_all = bool(_nested(raw, "dataset.use_all_categories", True))
    if use_all:
        settings.selected_skus = CALTECH101_CATEGORIES
    settings.__class__ = Caltech101Settings
    if len(settings.selected_skus) < 2:
        raise ValueError("Caltech101 retrieval needs at least two categories")
    if len(set(settings.selected_skus)) != len(settings.selected_skus):
        raise ValueError("Caltech101 category selection contains duplicates")
    if settings.pk_skus_per_batch > len(settings.selected_skus):
        raise ValueError("PK sampler P cannot exceed the selected category count")
    return settings


def save_resolved_config(settings: Caltech101Settings) -> None:
    save_robust_resolved_config(settings)
