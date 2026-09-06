"""Audited settings for the Caltech101 LightGenV2 comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.settings import (
    load_settings as load_router_settings,
    save_resolved_config as save_router_resolved_config,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)


MODEL_VARIANTS = {
    "optical_router_scale_matched_moe",
    "d2nn_active_expert_matched",
    "frozen_qwen_embedding",
}


def load_settings(path: str | Path) -> Any:
    """Load the verified 10 cm Router contract and add the LightGen contract.

    The historical Router loader deliberately enforces sealed-test evaluation.
    Its validation therefore runs first with the inherited ``false`` value;
    only then do we install the explicitly requested periodic-test selection
    policy under a separate LightGen namespace.
    """

    config_path = Path(path).expanduser().resolve()
    settings = load_router_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.lightgen_model_variant = str(
        d("lightgen.model_variant", "optical_router_scale_matched_moe")
    )
    settings.fusion_mode = str(
        d("balanced_fusion.mode", "scale_matched_convex")
    )
    settings.fusion_alpha_min = float(d("balanced_fusion.alpha_min", 0.01))
    settings.fusion_alpha_max = float(d("balanced_fusion.alpha_max", 0.95))
    settings.fusion_alpha_initial = float(
        d("balanced_fusion.alpha_initial", 0.055)
    )
    settings.fusion_rms_epsilon = float(
        d("balanced_fusion.rms_epsilon", 1.0e-6)
    )
    settings.fusion_detach_scale_statistics = bool(
        d("balanced_fusion.detach_scale_statistics", True)
    )
    settings.test_evaluation_interval_epochs = int(
        d("lightgen.selection.test_interval_epochs", 5)
    )
    settings.evaluate_test_each_epoch = bool(
        d("lightgen.selection.use_periodic_test", True)
    )
    settings.lightgen_primary_checkpoint = str(
        d(
            "lightgen.selection.primary_checkpoint",
            "ema_best_observed_test_checkpoint.pt",
        )
    )
    settings.lightgen_test_selected = bool(settings.evaluate_test_each_epoch)
    settings.d2nn_phase_size = int(d("d2nn.phase_size", settings.expert_size))
    settings.d2nn_phase_layers_per_modality = int(
        d("d2nn.phase_layers_per_modality", 2)
    )

    if settings.lightgen_model_variant not in MODEL_VARIANTS:
        raise ValueError(
            "lightgen.model_variant must be one of "
            f"{sorted(MODEL_VARIANTS)}"
        )
    if settings.router_backend != "optical" or settings.top_k != 2:
        raise ValueError("The LightGen main contract requires optical Top-2 routing")
    if settings.fusion_mode != "scale_matched_convex":
        raise ValueError("LightGen T01 requires scale_matched_convex fusion")
    if not 0.0 <= settings.fusion_alpha_min < settings.fusion_alpha_max <= 1.0:
        raise ValueError("balanced_fusion alpha range must satisfy 0<=min<max<=1")
    if not (
        settings.fusion_alpha_min
        < settings.fusion_alpha_initial
        < settings.fusion_alpha_max
    ):
        raise ValueError("balanced_fusion alpha_initial must be inside its range")
    if settings.fusion_rms_epsilon <= 0.0:
        raise ValueError("balanced_fusion.rms_epsilon must be positive")
    if not settings.fusion_detach_scale_statistics:
        raise ValueError("Formal scale matching requires detached RMS statistics")
    if not settings.evaluate_test_each_epoch:
        raise ValueError("This comparison explicitly selects by periodic test Top-1")
    if settings.test_evaluation_interval_epochs <= 0:
        raise ValueError("test_interval_epochs must be positive")
    if settings.d2nn_phase_size != settings.expert_size:
        raise ValueError("The requested D2NN match requires 224x224 phase planes")
    if settings.d2nn_phase_layers_per_modality != settings.top_k:
        raise ValueError(
            "D2NN phase layers per modality must equal the Top-2 active experts"
        )
    return settings


def save_resolved_config(settings: Any) -> None:
    save_router_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["lightgen"] = {
        "task": "t01_object_retrieval",
        "model_variant": settings.lightgen_model_variant,
        "selection": {
            "use_periodic_test": settings.evaluate_test_each_epoch,
            "test_interval_epochs": settings.test_evaluation_interval_epochs,
            "primary_checkpoint": settings.lightgen_primary_checkpoint,
            "criterion": "maximum periodically observed EMA test Top-1",
            "test_metrics_used_for_selection": True,
        },
    }
    values["balanced_fusion"] = {
        "mode": settings.fusion_mode,
        "equation": (
            "rE*((1-alpha)*E/rE + alpha*O/rO)/rms(mixture); RMS statistics "
            "are detached per sample over all valid tokens and channels"
        ),
        "alpha_min": settings.fusion_alpha_min,
        "alpha_max": settings.fusion_alpha_max,
        "alpha_initial": settings.fusion_alpha_initial,
        "rms_epsilon": settings.fusion_rms_epsilon,
        "detach_scale_statistics": True,
    }
    values["d2nn"] = {
        "phase_size": settings.d2nn_phase_size,
        "phase_layers_per_modality": settings.d2nn_phase_layers_per_modality,
        "trainable_phase_parameters_per_modality": (
            settings.d2nn_phase_layers_per_modality
            * settings.d2nn_phase_size
            * settings.d2nn_phase_size
        ),
        "matching_target": "top_k * expert_size^2 per modality",
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["MODEL_VARIANTS", "load_settings", "save_resolved_config"]
