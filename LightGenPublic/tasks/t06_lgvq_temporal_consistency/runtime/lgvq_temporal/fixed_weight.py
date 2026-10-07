"""Clean facade over the frozen checkpoint-compatibility implementation.

The historical modules stay untouched so state-dict names and numerical output
remain stable.  New entry points import this module instead of reaching into
the legacy ``LightGenV2`` and ``experiments`` namespaces directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as _legacy

from .contracts import load_reference_contract


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = PACKAGE_ROOT / "configs" / "temporal_16x4_s163.yaml"
PINS = _legacy.PINS


def load_model(target: str, checkpoint: str | Path, device: str = "cpu") -> tuple[Any, Any]:
    model, settings = _legacy.load_model(target, checkpoint, device)
    if target == "temporal":
        load_reference_contract(REFERENCE_CONFIG).require_settings(settings)
    return model, settings


def forward(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return _legacy.forward(model, batch)


def regression_metrics(prediction: Any, target: Any, target_name: str) -> dict[str, float]:
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import (
        regression_metrics as _regression_metrics,
    )

    return _regression_metrics(prediction, target, target_name)


__all__ = ["PINS", "forward", "load_model", "regression_metrics"]
