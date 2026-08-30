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


ALLOWED_FUSION_FLOORS = {0.001, 0.005}


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
    settings.tradeoff_variant = str(d("continuation.variant", "unnamed"))
    if len(settings.continuation_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in settings.continuation_sha256
    ):
        raise ValueError("continuation.sha256 must be an exact lowercase SHA-256")
    if settings.evaluate_test_each_epoch:
        raise ValueError("Trade-off runs must keep the test split sealed")
    if settings.resume_optimizer_state:
        raise ValueError("Trade-off runs must reset optimizer state")
    if not any(
        abs(settings.optical_fusion_minimum - value) < 1.0e-12
        for value in ALLOWED_FUSION_FLOORS
    ):
        raise ValueError(
            "Trade-off fusion floor must be exactly 0.001 or 0.005"
        )
    if not settings.language_optical_zero_order_enabled:
        raise ValueError("Early robust runs require coherent zero-order training")
    if settings.language_optical_ccd_noise_distribution != (
        "truncated_biased_gaussian"
    ):
        raise ValueError("Early robust runs require truncated biased Gaussian CCD noise")
    if not settings.phase_dc_enabled or settings.lambda_phase_dc <= 0.0:
        raise ValueError("Early robust runs require positive phase-DC suppression")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["continuation"] = {
        "variant": settings.tradeoff_variant,
        "checkpoint": str(settings.continuation_checkpoint),
        "sha256": settings.continuation_sha256,
        "optimizer_state": "reset",
        "noise_start": "first continuation epoch after Stage-A epoch 4",
        "selection_policy": "minimum training loss; sealed test evaluated once",
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["load_settings", "save_resolved_config"]
