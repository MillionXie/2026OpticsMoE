from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.settings import (
    load_settings as load_reference_settings,
    save_resolved_config as save_reference_resolved_config,
)


VARIANTS = {
    "moe_continuous_fixed_router",
    "d2nn_continuous",
    "d2nn_oeo_sigmoid",
    "moe_oeo_fixed_router",
    "moe_oeo_dynamic_router",
}
SAMPLING_MODES = {"epoch_random", "cyclic_balanced"}


def load_settings(path: str | Path) -> Any:
    """Load the reference recipe and replace only the optical architecture.

    The reference loader is intentionally reused so dataset splitting, Qwen
    processor budgets, augmentation, optimizer policy and retrieval losses are
    identical to the published Caltech-101 run.
    """

    config_path = Path(path).expanduser().resolve()
    settings = load_reference_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.multiplane_variant = str(
        d("multiplane.variant", "moe_continuous_fixed_router")
    )
    if settings.multiplane_variant not in VARIANTS:
        raise ValueError(
            f"multiplane.variant must be one of {sorted(VARIANTS)}, got "
            f"{settings.multiplane_variant!r}"
        )

    settings.multiplane_expert_planes = int(d("multiplane.expert_planes", 4))
    settings.multiplane_d2nn_planes = int(d("multiplane.d2nn_planes", 5))
    settings.multiplane_global_planes = int(d("multiplane.global_planes", 1))
    settings.multiplane_interplane_distance_m = float(
        d("multiplane.interplane_distance_m", settings.language_optical_distance_m)
    )
    settings.multiplane_detector_distance_m = float(
        d("multiplane.detector_distance_m", settings.language_optical_distance_m)
    )
    settings.multiplane_static_expert_apertures = bool(
        d("multiplane.static_expert_apertures", True)
    )
    settings.multiplane_oeo_eps = float(d("multiplane.oeo.eps", 1.0e-5))
    settings.multiplane_oeo_gain = float(d("multiplane.oeo.sigmoid_gain", 1.0))
    settings.multiplane_oeo_affine = bool(
        d("multiplane.oeo.layernorm_affine", False)
    )
    settings.multiplane_save_stage_diagnostics = bool(
        d("multiplane.save_stage_diagnostics", True)
    )
    settings.multiplane_sampling_mode = str(
        d("batching.sampling_mode", "cyclic_balanced")
    )
    if settings.multiplane_sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"batching.sampling_mode must be one of {sorted(SAMPLING_MODES)}, got "
            f"{settings.multiplane_sampling_mode!r}"
        )

    if settings.multiplane_expert_planes != 4:
        raise ValueError("The controlled MoE comparison requires exactly 4 expert planes")
    if settings.multiplane_d2nn_planes != 5:
        raise ValueError("The controlled D2NN comparison requires exactly 5 phase planes")
    if settings.multiplane_global_planes != 1:
        raise ValueError("The controlled MoE comparison requires exactly 1 global plane")
    if min(
        settings.multiplane_interplane_distance_m,
        settings.multiplane_detector_distance_m,
        settings.multiplane_oeo_eps,
        settings.multiplane_oeo_gain,
    ) <= 0.0:
        raise ValueError("Propagation distances, OEO eps and sigmoid gain must be positive")

    is_d2nn = settings.multiplane_variant.startswith("d2nn")
    settings.expert_layers = (
        settings.multiplane_d2nn_planes
        if is_d2nn
        else settings.multiplane_expert_planes
    )
    # Retrieval consumes mean+max pooling of 224 detector-token channels.
    settings.detector_output_size = 2 * int(settings.input_adapter_dim)
    settings.electronic_deepstack_enabled = False
    settings.vision_tap_stages = ()
    settings.native_pre_attention_enabled = False
    settings.transformer_residual_enabled = False
    settings.student_language_mode = "optical_moe"
    settings.phase_dc_enabled = False
    settings.lambda_phase_dc = 0.0
    settings.lambda_ccd_operating_point = 0.0

    if is_d2nn:
        # Trainer diagnostics treat D2NN as a deterministic one-path router.
        settings.num_experts = 1
        settings.expert_grid_rows = 1
        settings.expert_grid_cols = 1
        settings.top_k = 1
        settings.active_size = int(settings.expert_size)
        settings.canvas_size = int(settings.expert_size)
        settings.expert_pitch = int(settings.expert_size)
        settings.router_learning_rate = float(settings.learning_rate)
    else:
        # Restore the verified 2x2 MoE4 geometry after any inherited settings.
        settings.num_experts = 4
        settings.expert_grid_rows = 2
        settings.expert_grid_cols = 2
        settings.top_k = 2
        settings.active_size = 478
        settings.canvas_size = 518
        settings.expert_size = 224
        settings.expert_pitch = 254

    settings.expert_interlayer_distance_m = (
        settings.multiplane_interplane_distance_m
    )
    settings.global_to_detector_distance_m = settings.multiplane_detector_distance_m
    return settings


def save_resolved_config(settings: Any) -> None:
    save_reference_resolved_config(settings)
    destination = settings.output_dir / "config.yaml"
    values = yaml.safe_load(destination.read_text(encoding="utf-8"))
    values["multiplane"] = {
        "variant": settings.multiplane_variant,
        "expert_planes": settings.multiplane_expert_planes,
        "d2nn_planes": settings.multiplane_d2nn_planes,
        "global_planes": settings.multiplane_global_planes,
        "interplane_distance_m": settings.multiplane_interplane_distance_m,
        "detector_distance_m": settings.multiplane_detector_distance_m,
        "intermediate_electronics": (
            "full-aperture square-law + non-affine normalization + sigmoid + zero-phase reload"
            if settings.multiplane_variant == "d2nn_oeo_sigmoid"
            else "per-expert square-law + non-affine normalization + sigmoid + routing-weighted zero-phase reload"
            if "oeo" in settings.multiplane_variant
            else "none"
        ),
        "router_policy": (
            "independent router at every expert plane"
            if settings.multiplane_variant == "moe_oeo_dynamic_router"
            else "one input router reused across every expert plane"
            if not settings.multiplane_variant.startswith("d2nn")
            else "none"
        ),
        "oeo": {
            "eps": settings.multiplane_oeo_eps,
            "sigmoid_gain": settings.multiplane_oeo_gain,
            "layernorm_affine": settings.multiplane_oeo_affine,
        },
        "retrieval_detector_dim": settings.detector_output_size,
    }
    values.setdefault("hardware", {}).setdefault("ccd", {})["target_size"] = int(
        settings.active_size
    )
    values["hardware"]["ccd"]["transport_size"] = int(settings.active_size)
    values.setdefault("batching", {})["sampling_mode"] = (
        settings.multiplane_sampling_mode
    )
    values["batching"]["samples_per_epoch"] = int(
        settings.pk_skus_per_batch
        * settings.pk_images_per_sku
        * (
            settings.optimizer_steps_per_epoch
            if settings.optimizer_steps_per_epoch is not None
            else 0
        )
    )
    destination.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["SAMPLING_MODES", "VARIANTS", "load_settings", "save_resolved_config"]
