from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    Settings,
    _nested,
    _read_config,
    load_settings as load_baseline_settings,
)


def _probability(name: str, value: Any) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be in [0,1)")
    return result


def load_settings(path: str | Path) -> Settings:
    """Load the baseline schema and attach robust-hybrid experiment controls.

    The baseline loader deliberately pins phase dropout to ``none``.  This
    experiment opts into it only after the baseline configuration has passed
    all of its compatibility checks, then performs its own stricter checks.
    """

    config_path = Path(path).expanduser().resolve()
    settings = load_baseline_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.phase_dropout_mode = str(
        d("optical.phase.dropout_mode", "block_phase_bypass")
    )
    settings.phase_dropout_p = _probability(
        "optical.phase.dropout_p", d("optical.phase.dropout_p", 0.05)
    )
    settings.phase_dropout_block_size = int(
        d("optical.phase.dropout_block_size", 8)
    )
    settings.phase_dropout_batch_shared = bool(
        d("optical.phase.dropout_batch_shared", True)
    )

    settings.hybrid_residual_initial_weight = float(
        d("robust_hybrid.residual.initial_input_weight", 0.8)
    )
    settings.hybrid_refiner_width = int(
        d("robust_hybrid.electronic_refiner.width", 16)
    )
    settings.hybrid_refiner_dilation = int(
        d("robust_hybrid.electronic_refiner.second_dilation", 2)
    )
    settings.hybrid_refiner_dropout = _probability(
        "robust_hybrid.electronic_refiner.dropout",
        d("robust_hybrid.electronic_refiner.dropout", 0.05),
    )
    settings.readout_bottleneck_dim = int(
        d("robust_hybrid.embedding_head.bottleneck_dim", 96)
    )
    settings.readout_dropout = _probability(
        "robust_hybrid.embedding_head.dropout",
        d("robust_hybrid.embedding_head.dropout", 0.10),
    )

    settings.alignment_augmentation_enabled = bool(
        d("robust_hybrid.alignment.enabled", True)
    )
    settings.alignment_apply_during_eval = bool(
        d("robust_hybrid.alignment.apply_during_eval", False)
    )
    settings.input_shift_max_px = int(
        d("robust_hybrid.alignment.input_shift_max_logical_px", 12)
    )
    settings.phase_shift_max_px = int(
        d("robust_hybrid.alignment.phase_shift_max_logical_px", 12)
    )
    settings.ccd_shift_max_px = int(
        d("robust_hybrid.alignment.ccd_shift_max_logical_px", 12)
    )
    settings.alignment_batch_shared = bool(
        d("robust_hybrid.alignment.batch_shared", True)
    )

    if settings.phase_dropout_mode not in {
        "none",
        "phase_bypass",
        "block_phase_bypass",
    }:
        raise ValueError("Unsupported optical.phase.dropout_mode")
    if settings.phase_dropout_mode == "none" and settings.phase_dropout_p != 0.0:
        raise ValueError("phase dropout probability must be zero when mode=none")
    if settings.phase_dropout_block_size <= 0:
        raise ValueError("phase dropout block size must be positive")
    if not 0.0 < settings.hybrid_residual_initial_weight < 1.0:
        raise ValueError("initial_input_weight must be strictly between zero and one")
    if settings.hybrid_refiner_width <= 0 or settings.hybrid_refiner_dilation <= 0:
        raise ValueError("electronic refiner width/dilation must be positive")
    if settings.readout_bottleneck_dim <= 0:
        raise ValueError("embedding-head bottleneck_dim must be positive")
    for name in ("input_shift_max_px", "phase_shift_max_px", "ccd_shift_max_px"):
        if getattr(settings, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if not settings.alignment_batch_shared:
        raise ValueError(
            "This experiment models one physical registration per optical setup; "
            "robust_hybrid.alignment.batch_shared must be true"
        )
    return settings


def save_resolved_config(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    values = settings.to_dict()
    values["optical"]["phase"].update(
        {
            "dropout_mode": settings.phase_dropout_mode,
            "dropout_p": settings.phase_dropout_p,
            "dropout_block_size": settings.phase_dropout_block_size,
            "dropout_batch_shared": settings.phase_dropout_batch_shared,
        }
    )
    values["robust_hybrid"] = {
        "residual": {
            "initial_input_weight": settings.hybrid_residual_initial_weight,
            "parameterization": "sigmoid_convex_mix",
        },
        "electronic_refiner": {
            "type": "depthwise_separable_local_residual",
            "width": settings.hybrid_refiner_width,
            "second_dilation": settings.hybrid_refiner_dilation,
            "dropout": settings.hybrid_refiner_dropout,
        },
        "embedding_head": {
            "bottleneck_dim": settings.readout_bottleneck_dim,
            "dropout": settings.readout_dropout,
        },
        "alignment": {
            "enabled": settings.alignment_augmentation_enabled,
            "apply_during_eval": settings.alignment_apply_during_eval,
            "input_shift_max_logical_px": settings.input_shift_max_px,
            "phase_shift_max_logical_px": settings.phase_shift_max_px,
            "ccd_shift_max_logical_px": settings.ccd_shift_max_px,
            "batch_shared": settings.alignment_batch_shared,
        },
    }
    (settings.output_dir / "config.yaml").write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def residual_logit(probability: float) -> float:
    """Public helper used by architecture tests and checkpoint diagnostics."""

    return math.log(float(probability) / (1.0 - float(probability)))
