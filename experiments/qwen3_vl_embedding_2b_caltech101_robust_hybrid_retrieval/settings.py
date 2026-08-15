from __future__ import annotations

from pathlib import Path

import yaml

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
    shared_teacher_cache_dir: Path
    teacher_cache_source_path: Path | None
    use_all_categories: bool

    @property
    def teacher_cache_path(self) -> Path:
        return self.shared_teacher_cache_dir / "teacher_embeddings.pt"

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
    settings.use_all_categories = use_all
    settings.shared_teacher_cache_dir = _resolve_from_config(
        _nested(
            raw,
            "teacher_cache.directory",
            "../../cache/caltech101_all101_seed42_g3_train30_test20",
        ),
        config_path,
    )
    source = _nested(raw, "teacher_cache.derive_from", None)
    settings.teacher_cache_source_path = (
        _resolve_from_config(source, config_path) / "teacher_embeddings.pt"
        if source is not None
        else None
    )
    if len(settings.selected_skus) < 2:
        raise ValueError("Caltech101 retrieval needs at least two categories")
    if len(set(settings.selected_skus)) != len(settings.selected_skus):
        raise ValueError("Caltech101 category selection contains duplicates")
    if settings.pk_skus_per_batch > len(settings.selected_skus):
        raise ValueError("PK sampler P cannot exceed the selected category count")
    return settings


def save_resolved_config(settings: Caltech101Settings) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["dataset"]["use_all_categories"] = settings.use_all_categories
    values["teacher_cache"] = {
        "directory": str(settings.shared_teacher_cache_dir),
        "path": str(settings.teacher_cache_path),
        "derive_from": (
            str(settings.teacher_cache_source_path.parent)
            if settings.teacher_cache_source_path is not None
            else None
        ),
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _resolve_from_config(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.parent / path).resolve()
