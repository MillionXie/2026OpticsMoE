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

from .router import ROUTING_NORMALIZATIONS


ROUTER_BACKENDS = {"electronic", "optical"}
SCORE_NORMALIZATIONS = {"standardized_region_energy", "log_energy_fraction"}


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
    return settings


def save_resolved_config(settings: Any) -> None:
    save_robust_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["router_experiment"] = {
        "backend": settings.router_backend,
        "top_k": settings.top_k,
        "weight_normalization": settings.router_weight_normalization,
        "straight_through": settings.router_straight_through,
        "reset_router_parameters": settings.router_reset_parameters,
        "protocol": settings.router_protocol,
        "source_checkpoint": str(settings.router_source_checkpoint),
        "source_sha256": settings.router_source_sha256,
        "source_test_used_for_selection": False,
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
        },
    }
    values["language_optical"]["layout"] = (
        f"MoE4_2x2_topk{settings.top_k}_{settings.router_backend}_router"
    )
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


__all__ = [
    "ROUTER_BACKENDS",
    "SCORE_NORMALIZATIONS",
    "load_settings",
    "save_resolved_config",
]
