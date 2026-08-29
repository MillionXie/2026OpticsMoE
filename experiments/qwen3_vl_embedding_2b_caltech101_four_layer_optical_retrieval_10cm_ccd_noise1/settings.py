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


def _required_path(value: Any, config_path: Path, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_robust_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.continuation_checkpoint = _required_path(
        d("continuation.checkpoint"), config_path, "continuation.checkpoint"
    )
    settings.continuation_sha256 = str(d("continuation.sha256", "")).lower()
    settings.noise_variant = str(d("continuation.variant", "unnamed"))

    if len(settings.continuation_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in settings.continuation_sha256
    ):
        raise ValueError("continuation.sha256 must be an exact lowercase SHA-256")
    if settings.evaluate_test_each_epoch:
        raise ValueError("Noise runs must keep training.evaluate_test_each_epoch=false")
    if settings.resume_optimizer_state:
        raise ValueError("Noise runs must reset optimizer state at the warm-start boundary")
    if abs(settings.optical_fusion_minimum - 0.01) > 1.0e-12:
        raise ValueError("Noise study requires hybrid.minimum_optical_fusion=0.01")
    if abs(settings.optical_fusion_initial - 0.015) > 1.0e-12:
        raise ValueError("Noise study requires hybrid.initial_fusion=0.015")
    if settings.language_optical_ccd_noise_distribution != "truncated_biased_gaussian":
        raise ValueError("Noise study requires a truncated_biased_gaussian CCD model")
    if not settings.phase_dc_enabled or settings.lambda_phase_dc <= 0.0:
        raise ValueError("Noise study requires a positive phase_dc constraint")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["continuation"] = {
        "variant": settings.noise_variant,
        "checkpoint": str(settings.continuation_checkpoint),
        "sha256": settings.continuation_sha256,
        "optimizer_state": "reset",
        "selection_policy": (
            "minimum training loss inside each run; no per-epoch test access"
        ),
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["load_settings", "save_resolved_config"]

