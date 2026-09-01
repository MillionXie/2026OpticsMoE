from __future__ import annotations

import hashlib
import json
import math
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

from .router import ROUTING_NORMALIZATIONS


ROUTER_BACKENDS = {"electronic", "optical"}
SCORE_NORMALIZATIONS = {"standardized_region_energy", "log_energy_fraction"}


def router_contract_payload(settings: Any) -> dict[str, Any]:
    """Return every setting that changes router numerics or hardware meaning."""

    common: dict[str, Any] = {
        "schema_version": 1,
        "backend": settings.router_backend,
        "top_k": settings.top_k,
        "temperature": settings.router_temperature,
        "weight_normalization": settings.router_weight_normalization,
        "straight_through": settings.router_straight_through,
        "amplitude_weight_domain": settings.amplitude_slm_weight_domain,
        "geometry": {
            "canvas_size": settings.canvas_size,
            "active_size": settings.active_size,
            "expert_size": settings.expert_size,
            "expert_pitch": settings.expert_pitch,
            "num_experts": settings.num_experts,
            "grid_rows": settings.expert_grid_rows,
            "grid_cols": settings.expert_grid_cols,
        },
    }
    if settings.router_backend == "electronic":
        common["electronic"] = {
            "pool_size": settings.router_pool_size,
            "input_layernorm_enabled": settings.router_input_layernorm_enabled,
            "input_layernorm_eps": settings.router_input_layernorm_eps,
            "gate_init_std": getattr(settings, "router_gate_init_std", 0.01),
            "train_logit_noise_std": settings.router_noise_std,
        }
    else:
        common["optical"] = {
            "detector_intervals": [
                list(value) for value in settings.optical_router_detector_intervals
            ],
            "score_normalization": settings.optical_router_score_normalization,
            "energy_eps": settings.optical_router_energy_eps,
            "capture_loss_scale": settings.optical_router_capture_loss_scale,
            "phase_dropout_p": settings.optical_router_phase_dropout_p,
            "phase_dropout_block_size": settings.optical_router_phase_dropout_block_size,
            "input_shift_pixels": settings.optical_router_input_shift_pixels,
            "phase_shift_pixels": settings.optical_router_phase_shift_pixels,
            "ccd_shift_pixels": settings.optical_router_ccd_shift_pixels,
            "wavelength_nm": settings.language_optical_wavelength_nm,
            "logical_pixel_pitch_um": settings.language_optical_pixel_pitch_um,
            "distance_m": settings.language_optical_distance_m,
            "k_space_enabled": settings.language_optical_k_space_enabled,
            "theta_max_deg": settings.language_optical_theta_max_deg,
            "phase_parameterization": "2pi_sigmoid",
            "phase_slm": {
                "width": settings.hardware_phase_slm_width,
                "height": settings.hardware_phase_slm_height,
                "pixel_pitch_um": settings.hardware_phase_slm_pixel_pitch_um,
                "center_xy": [
                    settings.hardware_phase_slm_center_x,
                    settings.hardware_phase_slm_center_y,
                ],
                "flip_vertical": settings.hardware_phase_flip_vertical,
                "flip_horizontal": settings.hardware_phase_flip_horizontal,
            },
        }
    return common


def router_contract_sha256(settings: Any) -> str:
    encoded = json.dumps(
        router_contract_payload(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_path(value: Any, config_path: Path, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required")
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def _required_sha256(value: Any, label: str) -> str:
    digest = "" if value is None else str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be an exact lowercase/uppercase SHA-256")
    return digest


def load_settings(path: str | Path) -> Any:
    """Load canonical warmstart geometry, then apply only router-ablation knobs.

    The robust parent deliberately locks its native router to top-k=2.  The
    separate ``router_experiment`` namespace is parsed only after that audited
    contract has passed, so no existing experiment silently changes meaning.
    """

    config_path = Path(path).expanduser().resolve()
    settings = load_robust_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.router_backend = str(d("router_experiment.backend", "electronic"))
    settings.router_optimization_seed = int(
        d("router_experiment.optimization_seed", settings.random_seed)
    )
    settings.top_k = int(d("router_experiment.top_k", 2))
    settings.router_weight_normalization = str(
        d("router_experiment.weight_normalization", "power_l2")
    )
    settings.router_straight_through = bool(
        d("router_experiment.straight_through", True)
    )
    settings.router_source_checkpoint = _required_path(
        d("router_experiment.source_checkpoint"),
        config_path,
        "router_experiment.source_checkpoint",
    )
    settings.router_source_sha256 = _required_sha256(
        d("router_experiment.source_sha256"),
        "router_experiment.source_sha256",
    )
    settings.router_protocol = str(
        d("router_experiment.protocol", "equal_budget_adaptation")
    )
    # Re-evaluate the default now that protocol is known.  An explicit YAML
    # value remains authoritative.
    settings.router_reset_parameters = bool(
        d(
            "router_experiment.reset_router_parameters",
            settings.router_protocol != "legacy_anchor",
        )
    )

    settings.optical_router_detector_intervals = tuple(
        tuple(int(value) for value in interval)
        for interval in d("router_experiment.optical.detector_intervals", [[162, 221], [257, 316]])
    )
    settings.optical_router_energy_eps = float(
        d("router_experiment.optical.energy_eps", 1.0e-8)
    )
    settings.optical_router_score_normalization = str(
        d(
            "router_experiment.optical.score_normalization",
            "standardized_region_energy",
        )
    )
    settings.optical_router_capture_loss_scale = float(
        d("router_experiment.optical.capture_loss_scale", 0.10)
    )
    settings.optical_router_maximum_saturated_pixel_fraction = float(
        d(
            "router_experiment.optical.hardware_quality.maximum_saturated_pixel_fraction",
            0.02,
        )
    )
    settings.optical_router_minimum_p99_uint8 = float(
        d("router_experiment.optical.hardware_quality.minimum_p99_uint8", 8.0)
    )
    settings.optical_router_minimum_dynamic_range_uint8 = float(
        d(
            "router_experiment.optical.hardware_quality.minimum_dynamic_range_uint8",
            4.0,
        )
    )
    settings.optical_router_minimum_topk_probability_margin = float(
        d(
            "router_experiment.optical.hardware_quality.minimum_topk_probability_margin",
            0.01,
        )
    )
    settings.optical_router_phase_dropout_p = float(
        d("router_experiment.optical.phase_dropout_p", 0.05)
    )
    settings.optical_router_phase_dropout_block_size = int(
        d("router_experiment.optical.phase_dropout_block_size", 8)
    )
    settings.optical_router_input_shift_pixels = int(
        d("router_experiment.optical.input_shift_pixels", 16)
    )
    settings.optical_router_phase_shift_pixels = int(
        d("router_experiment.optical.phase_shift_pixels", 16)
    )
    settings.optical_router_ccd_shift_pixels = int(
        d("router_experiment.optical.ccd_shift_pixels", 16)
    )

    if settings.router_backend not in ROUTER_BACKENDS:
        raise ValueError(f"router backend must be one of {sorted(ROUTER_BACKENDS)}")
    if settings.router_optimization_seed < 0:
        raise ValueError("router_experiment.optimization_seed must be nonnegative")
    if settings.top_k not in {1, 2, 4}:
        raise ValueError("router_experiment.top_k must be one of 1, 2, 4")
    if settings.router_weight_normalization not in ROUTING_NORMALIZATIONS:
        raise ValueError(
            "router_experiment.weight_normalization must be one of "
            f"{sorted(ROUTING_NORMALIZATIONS)}"
        )
    if settings.optical_router_score_normalization not in SCORE_NORMALIZATIONS:
        raise ValueError(
            "router optical score_normalization must be one of "
            f"{sorted(SCORE_NORMALIZATIONS)}"
        )
    if settings.router_protocol not in {
        "legacy_anchor",
        "equal_budget_adaptation",
    }:
        raise ValueError("router_experiment.protocol is not recognized")
    if settings.router_protocol == "legacy_anchor" and (
        settings.top_k != 2
        or settings.router_backend != "electronic"
        or settings.router_weight_normalization != "legacy_l1"
        or settings.router_straight_through
        or settings.router_reset_parameters
    ):
        raise ValueError(
            "legacy_anchor must be electronic/top-k2/legacy_l1/no-STE"
        )
    if settings.router_protocol == "equal_budget_adaptation" and (
        settings.router_weight_normalization != "power_l2"
        or not settings.router_straight_through
        or not settings.router_reset_parameters
    ):
        raise ValueError(
            "equal-budget adaptation requires power_l2, straight-through and "
            "fresh router initialization"
        )
    if settings.amplitude_slm_weight_domain != "amplitude":
        raise ValueError(
            "This experiment emits amplitude weights directly and therefore "
            "requires optical.amplitude_slm.weight_domain=amplitude"
        )
    if settings.evaluate_test_each_epoch:
        raise ValueError("Sealed test cannot be evaluated during training")
    if settings.phase_focus_enabled:
        raise ValueError(
            "Router ablation disables phase-focus alternation so every matched "
            "optimizer step can update the router"
        )
    if settings.optical_router_energy_eps <= 0.0:
        raise ValueError("optical router energy_eps must be positive")
    if settings.optical_router_capture_loss_scale < 0.0:
        raise ValueError("optical router capture_loss_scale must be nonnegative")
    if not 0.0 <= settings.optical_router_maximum_saturated_pixel_fraction <= 1.0:
        raise ValueError(
            "optical router maximum_saturated_pixel_fraction must be in [0,1]"
        )
    if not 0.0 <= settings.optical_router_minimum_p99_uint8 <= 255.0:
        raise ValueError("optical router minimum_p99_uint8 must be in [0,255]")
    if not 0.0 <= settings.optical_router_minimum_dynamic_range_uint8 <= 255.0:
        raise ValueError(
            "optical router minimum_dynamic_range_uint8 must be in [0,255]"
        )
    if not 0.0 <= settings.optical_router_minimum_topk_probability_margin < 1.0:
        raise ValueError(
            "optical router minimum_topk_probability_margin must be in [0,1)"
        )
    if not 0.0 <= settings.optical_router_phase_dropout_p < 1.0:
        raise ValueError("optical router phase_dropout_p must be in [0,1)")
    if settings.optical_router_phase_dropout_block_size <= 0:
        raise ValueError("optical router phase_dropout_block_size must be positive")
    for label, maximum in (
        ("input", settings.optical_router_input_shift_pixels),
        ("phase", settings.optical_router_phase_shift_pixels),
        ("ccd", settings.optical_router_ccd_shift_pixels),
    ):
        if not 0 <= maximum <= 20:
            raise ValueError(f"optical router {label} shift must be in [0,20]")

    intervals = settings.optical_router_detector_intervals
    if len(intervals) != 2:
        raise ValueError("optical router requires two detector intervals")
    previous_end = -1
    for start, end in intervals:
        if not 0 <= start < end <= settings.active_size:
            raise ValueError("optical router detector intervals exceed the 478 ROI")
        if start < previous_end:
            raise ValueError("optical router detector intervals overlap")
        previous_end = end
    if (intervals[0][1] - intervals[0][0]) != (intervals[1][1] - intervals[1][0]):
        raise ValueError("all optical router detector windows must have equal area")
    detector_center = 0.5 * (settings.active_size - 1)
    interval_centers = [0.5 * (start + end - 1) for start, end in intervals]
    radial_pixels = max(
        math.hypot(x - detector_center, y - detector_center)
        for x in interval_centers
        for y in interval_centers
    )
    transverse_ratio = (
        radial_pixels
        * settings.language_optical_pixel_pitch_um
        * 1.0e-6
        / settings.language_optical_distance_m
    )
    if transverse_ratio >= 1.0:
        raise ValueError("optical router detector target is not physically reachable")
    settings.optical_router_required_center_angle_deg = math.degrees(
        math.asin(transverse_ratio)
    )
    if (
        settings.router_backend == "optical"
        and settings.language_optical_k_space_enabled
        and settings.optical_router_required_center_angle_deg
        > settings.language_optical_theta_max_deg
    ):
        raise ValueError(
            "Optical-router detector centres require "
            f"{settings.optical_router_required_center_angle_deg:.6f} deg but the "
            f"radial k-space cutoff is {settings.language_optical_theta_max_deg:.6f} deg"
        )
    settings.router_contract = router_contract_payload(settings)
    settings.router_contract_sha256 = router_contract_sha256(settings)
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["router_experiment"] = {
        "backend": settings.router_backend,
        "optimization_seed": settings.router_optimization_seed,
        "dataset_and_batch_seed": settings.random_seed,
        "top_k": settings.top_k,
        "weight_normalization": settings.router_weight_normalization,
        "straight_through": settings.router_straight_through,
        "reset_router_parameters": settings.router_reset_parameters,
        "protocol": settings.router_protocol,
        "source_checkpoint": str(settings.router_source_checkpoint),
        "source_sha256": settings.router_source_sha256,
        "source_test_used_for_selection": False,
        "contract": settings.router_contract,
        "contract_sha256": settings.router_contract_sha256,
        "global_reuses_expert_routing": True,
        "optical": {
            "extra_router_exposures_per_sample": (
                2 if settings.router_backend == "optical" else 0
            ),
            "detector_intervals": [
                list(interval) for interval in settings.optical_router_detector_intervals
            ],
            "detector_window_order": [
                "expert_0_top_left",
                "expert_1_top_right",
                "expert_2_bottom_left",
                "expert_3_bottom_right",
            ],
            "score_normalization": settings.optical_router_score_normalization,
            "capture_loss_scale_inside_balance_loss": (
                settings.optical_router_capture_loss_scale
            ),
            "phase_dropout_p": settings.optical_router_phase_dropout_p,
            "phase_dropout_block_size": (
                settings.optical_router_phase_dropout_block_size
            ),
            "input_shift_pixels": settings.optical_router_input_shift_pixels,
            "phase_shift_pixels": settings.optical_router_phase_shift_pixels,
            "ccd_shift_pixels": settings.optical_router_ccd_shift_pixels,
            "canvas_size": settings.canvas_size,
            "ccd_roi_size": settings.active_size,
            "input_size": settings.expert_size,
            "distance_m": settings.language_optical_distance_m,
            "wavelength_nm": settings.language_optical_wavelength_nm,
            "logical_pixel_pitch_um": settings.language_optical_pixel_pitch_um,
            "learned_electronic_head_after_ccd": False,
            "required_detector_center_angle_deg": (
                settings.optical_router_required_center_angle_deg
            ),
        },
    }
    values["language_optical"]["layout"] = (
        f"MoE4_2x2_topk{settings.top_k}_{settings.router_backend}_router"
    )
    # These are acquisition acceptance thresholds, not trainable-model
    # numerics. Keep them outside the checkpoint Router contract so changing a
    # laboratory quality gate never changes architecture_label/checkpoint SHA
    # compatibility; the six-stage session seals the full resolved config.
    values["router_hardware_measurement"] = {
        "background_subtraction": False,
        "uncalibrated_capture_metric": "raw_capture_fraction",
        "quality_gates": {
            "maximum_saturated_pixel_fraction": (
                settings.optical_router_maximum_saturated_pixel_fraction
            ),
            "minimum_p99_uint8": settings.optical_router_minimum_p99_uint8,
            "minimum_dynamic_range_uint8": (
                settings.optical_router_minimum_dynamic_range_uint8
            ),
            "minimum_topk_probability_margin": (
                settings.optical_router_minimum_topk_probability_margin
            ),
        },
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = [
    "ROUTER_BACKENDS",
    "SCORE_NORMALIZATIONS",
    "load_settings",
    "router_contract_payload",
    "router_contract_sha256",
    "save_resolved_config",
]
