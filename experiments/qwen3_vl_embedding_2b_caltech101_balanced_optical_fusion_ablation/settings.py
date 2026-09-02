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


FUSION_MODES = {"scale_matched_convex", "electronic_only"}


def _optional_path(value: Any, config_path: Path) -> Path | None:
    if value is None:
        return None
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_robust_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.fusion_mode = str(d("balanced_fusion.mode", "scale_matched_convex"))
    settings.fusion_alpha_min = float(d("balanced_fusion.alpha_min", 0.05))
    settings.fusion_alpha_max = float(d("balanced_fusion.alpha_max", 0.49))
    settings.fusion_alpha_initial = float(
        d("balanced_fusion.alpha_initial", settings.optical_fusion_initial)
    )
    settings.fusion_rms_epsilon = float(d("balanced_fusion.rms_epsilon", 1.0e-6))
    settings.fusion_detach_scale_statistics = bool(
        d("balanced_fusion.detach_scale_statistics", True)
    )
    settings.initialization_checkpoint = _optional_path(
        d("initialization.checkpoint"), config_path
    )
    digest = d("initialization.sha256", "")
    settings.initialization_sha256 = "" if digest is None else str(digest).lower()
    settings.test_evaluation_interval_epochs = int(
        d("training.test_evaluation_interval_epochs", 1)
    )

    if settings.fusion_mode not in FUSION_MODES:
        raise ValueError(f"balanced_fusion.mode must be one of {sorted(FUSION_MODES)}")
    if not 0.0 <= settings.fusion_alpha_min < settings.fusion_alpha_max <= 1.0:
        raise ValueError("balanced_fusion alpha range must satisfy 0<=min<max<=1")
    if not settings.fusion_alpha_min < settings.fusion_alpha_initial < settings.fusion_alpha_max:
        raise ValueError("balanced_fusion.alpha_initial must lie strictly inside its range")
    if settings.fusion_rms_epsilon <= 0.0:
        raise ValueError("balanced_fusion.rms_epsilon must be positive")
    if not settings.fusion_detach_scale_statistics:
        raise ValueError(
            "Formal balanced fusion requires detach_scale_statistics=true so a branch "
            "cannot game its contribution coefficient by changing only its RMS"
        )
    if settings.initialization_checkpoint is None:
        raise ValueError("initialization.checkpoint is required for a fair ablation")
    if len(settings.initialization_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in settings.initialization_sha256
    ):
        raise ValueError("initialization.sha256 must be an exact lowercase SHA-256")
    if not settings.evaluate_test_each_epoch:
        raise ValueError(
            "This user-requested ablation selects by periodically observed test; "
            "set training.evaluate_test_each_epoch=true"
        )
    if settings.test_evaluation_interval_epochs <= 0:
        raise ValueError("training.test_evaluation_interval_epochs must be positive")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    electronic_only = settings.fusion_mode == "electronic_only"
    values["balanced_fusion"] = {
        "mode": settings.fusion_mode,
        "equation": (
            "F=E; optical propagation and learned gates are bypassed"
            if electronic_only
            else (
                "rE*((1-alpha)*E/rE + alpha*O/rO)/rms(mixture); all RMS "
                "statistics are detached and computed per sample over all valid "
                "tokens and channels"
            )
        ),
        "alpha_min": settings.fusion_alpha_min,
        "alpha_max": settings.fusion_alpha_max,
        "alpha_initial": settings.fusion_alpha_initial,
        "electronic_coefficient": 1.0 if electronic_only else "1-alpha",
        "optical_coefficient": 0.0 if electronic_only else "alpha",
        "configured_alpha_is_unused": electronic_only,
        "rms_epsilon": settings.fusion_rms_epsilon,
        "detach_scale_statistics": settings.fusion_detach_scale_statistics,
        "alpha_zero_identity": "fused(E,O,alpha=0) == E algebraically",
        "token_relative_magnitudes_preserved": True,
        "post_fusion_rms_matches_electronic": True,
    }
    values["initialization"] = {
        "checkpoint": str(settings.initialization_checkpoint),
        "sha256": settings.initialization_sha256,
        "gate_policy": "load all common tensors strictly, then reset four alpha logits",
    }
    values.setdefault("training", {})["test_evaluation_interval_epochs"] = (
        settings.test_evaluation_interval_epochs
    )
    values["test_selection_policy"] = {
        "evaluate_every_n_epochs": settings.test_evaluation_interval_epochs,
        "primary_checkpoint": "ema_best_observed_test_checkpoint.pt",
        "criterion": "maximum periodically observed EMA test Top-1",
        "test_leakage_accepted_by_user": True,
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = ["FUSION_MODES", "load_settings", "save_resolved_config"]
