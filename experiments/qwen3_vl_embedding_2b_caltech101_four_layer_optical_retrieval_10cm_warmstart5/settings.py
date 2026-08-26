from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.settings import (
    load_settings as load_robust_settings,
    save_resolved_config as save_robust_resolved_config,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)


STAGES = {"optical_calibration", "joint"}


def _optional_path(value: Any, config_path: Path) -> Path | None:
    if value is None:
        return None
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def _optional_digest(value: Any) -> str:
    """Normalize an optional SHA without turning YAML null into the text 'none'."""
    return "" if value is None else str(value).strip().lower()


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_robust_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.warmstart_stage = str(d("warmstart.stage", "joint"))
    settings.warmstart_electronic_checkpoint = _optional_path(
        d("warmstart.electronic_checkpoint"), config_path
    )
    settings.warmstart_electronic_sha256 = _optional_digest(
        d("warmstart.electronic_sha256", "")
    )
    settings.warmstart_optical_checkpoint = _optional_path(
        d("warmstart.optical_checkpoint"), config_path
    )
    settings.warmstart_optical_sha256 = _optional_digest(
        d("warmstart.optical_sha256", "")
    )
    settings.warmstart_stage_a_checkpoint = _optional_path(
        d("warmstart.stage_a_checkpoint"), config_path
    )

    if settings.warmstart_stage not in STAGES:
        raise ValueError(f"warmstart.stage must be one of {sorted(STAGES)}")
    if settings.evaluate_test_each_epoch:
        raise ValueError(
            "The warmstart5 sealed-test protocol requires "
            "training.evaluate_test_each_epoch=false"
        )
    if abs(settings.optical_fusion_minimum - 0.05) > 1.0e-12:
        raise ValueError("warmstart5 requires hybrid.minimum_optical_fusion=0.05")
    if abs(settings.optical_fusion_initial - 0.055) > 1.0e-12:
        raise ValueError("warmstart5 requires hybrid.initial_fusion=0.055")
    if settings.warmstart_stage == "optical_calibration":
        if settings.warmstart_electronic_checkpoint is None:
            raise ValueError("Stage A requires warmstart.electronic_checkpoint")
        if settings.warmstart_optical_checkpoint is None:
            raise ValueError("Stage A requires warmstart.optical_checkpoint")
        for label, digest in (
            ("electronic", settings.warmstart_electronic_sha256),
            ("optical", settings.warmstart_optical_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Stage A requires an exact SHA-256 for {label}")
    elif settings.warmstart_stage_a_checkpoint is None:
        raise ValueError("Stage B requires warmstart.stage_a_checkpoint")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["warmstart"] = {
        "stage": settings.warmstart_stage,
        "electronic_checkpoint": (
            None
            if settings.warmstart_electronic_checkpoint is None
            else str(settings.warmstart_electronic_checkpoint)
        ),
        "electronic_sha256": settings.warmstart_electronic_sha256 or None,
        "optical_checkpoint": (
            None
            if settings.warmstart_optical_checkpoint is None
            else str(settings.warmstart_optical_checkpoint)
        ),
        "optical_sha256": settings.warmstart_optical_sha256 or None,
        "stage_a_checkpoint": (
            None
            if settings.warmstart_stage_a_checkpoint is None
            else str(settings.warmstart_stage_a_checkpoint)
        ),
        "test_policy": (
            "no per-epoch test; one explicit evaluate of the predeclared "
            "Stage-B EMA train-loss checkpoint"
        ),
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = ["STAGES", "load_settings", "save_resolved_config"]
