"""Audited settings for the LSP optical-Router/D2NN comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    _nested,
    _read_config,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.settings import (
    load_settings as load_router_settings,
    save_resolved_config as save_router_resolved_config,
)


MODEL_VARIANTS = {
    "optical_router_scale_matched_moe",
    "d2nn_active_expert_matched",
}


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_router_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.lightgen_model_variant = str(
        d("lightgen.model_variant", "optical_router_scale_matched_moe")
    )
    settings.lightgen_primary_checkpoint = str(
        d("lightgen.selection.primary_checkpoint", "best_checkpoint.pt")
    )
    settings.fusion_mode = str(d("balanced_fusion.mode", "scale_matched_convex"))
    settings.fusion_alpha_min = float(d("balanced_fusion.alpha_min", 0.01))
    settings.fusion_alpha_max = float(d("balanced_fusion.alpha_max", 0.95))
    settings.fusion_alpha_initial = float(d("balanced_fusion.alpha_initial", 0.055))
    settings.fusion_rms_epsilon = float(d("balanced_fusion.rms_epsilon", 1.0e-6))
    settings.fusion_detach_scale_statistics = bool(
        d("balanced_fusion.detach_scale_statistics", True)
    )
    settings.phase_dc_weight = float(d("loss.phase_dc_weight", 0.0))
    settings.d2nn_phase_size = int(d("d2nn.phase_size", settings.expert_size))
    settings.d2nn_phase_layers = int(d("d2nn.phase_layers", settings.top_k))

    if settings.lightgen_model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown LightGen T02 model variant: {settings.lightgen_model_variant}")
    if settings.router_backend != "optical" or settings.top_k != 2:
        raise ValueError("T02 formal comparison is fixed to optical Top-2 routing")
    if settings.fusion_mode != "scale_matched_convex":
        raise ValueError("T02 requires scale_matched_convex fusion")
    if not 0.0 <= settings.fusion_alpha_min < settings.fusion_alpha_max <= 1.0:
        raise ValueError("Invalid balanced-fusion alpha interval")
    if not settings.fusion_alpha_min < settings.fusion_alpha_initial < settings.fusion_alpha_max:
        raise ValueError("Initial alpha must lie strictly inside the configured interval")
    if settings.fusion_rms_epsilon <= 0.0 or not settings.fusion_detach_scale_statistics:
        raise ValueError("Formal scale matching requires positive epsilon and detached RMS")
    if settings.d2nn_phase_size != settings.expert_size or settings.d2nn_phase_layers != settings.top_k:
        raise ValueError("D2NN must exactly match Top-2 activated 224x224 expert phase budget")
    if not settings.language_optical_zero_order_enabled:
        raise ValueError("The formal T02 rerun requires coherent zero-order augmentation")
    zero_order_values = (
        settings.language_optical_amplitude_zero_order_intensity_min,
        settings.language_optical_amplitude_zero_order_intensity_max,
        settings.language_optical_phase_zero_order_intensity_min,
        settings.language_optical_phase_zero_order_intensity_max,
    )
    if min(zero_order_values) < 0.20:
        raise ValueError("The formal T02 zero-order intensity fraction must be at least 20%")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_router_resolved_config(settings)
    path = settings.output_dir / "resolved_config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["lightgen"] = {
        "task": "t02_keypoint_detection",
        "model_variant": settings.lightgen_model_variant,
        "primary_checkpoint": settings.lightgen_primary_checkpoint,
        "checkpoint_retention": ["best_checkpoint.pt", "last_checkpoint.pt"],
    }
    values["balanced_fusion"] = {
        "mode": settings.fusion_mode,
        "equation": "rE*((1-alpha)*E/rE+alpha*O/rO)/rms(mixture)",
        "alpha_min": settings.fusion_alpha_min,
        "alpha_max": settings.fusion_alpha_max,
        "alpha_initial": settings.fusion_alpha_initial,
        "rms_epsilon": settings.fusion_rms_epsilon,
        "detach_scale_statistics": True,
    }
    values["d2nn"] = {
        "phase_size": settings.d2nn_phase_size,
        "phase_layers": settings.d2nn_phase_layers,
        "phase_parameters": settings.d2nn_phase_layers * settings.d2nn_phase_size**2,
        "matching_target": "top_k * expert_size^2",
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = ["MODEL_VARIANTS", "load_settings", "save_resolved_config"]
