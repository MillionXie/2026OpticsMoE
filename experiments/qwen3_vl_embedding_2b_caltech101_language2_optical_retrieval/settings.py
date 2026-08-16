from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.settings import (
    load_settings as load_electronic_settings,
    save_resolved_config as save_electronic_resolved_config,
)


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_electronic_settings(config_path)
    raw = _read_config(config_path)
    d = lambda key, default=None: _nested(raw, key, default)

    settings.initial_electronic_checkpoint = Path(
        str(d("hybrid.initial_electronic_checkpoint"))
    ).expanduser()
    if not settings.initial_electronic_checkpoint.is_absolute():
        settings.initial_electronic_checkpoint = (
            config_path.parent / settings.initial_electronic_checkpoint
        ).resolve()
    settings.hybrid_freeze_electronic = bool(d("hybrid.freeze_electronic", True))
    settings.optical_fusion_initial = float(d("hybrid.initial_fusion", 0.05))
    settings.lambda_ccd_operating_point = float(
        d("training.lambda_ccd_operating_point", 0.02)
    )

    settings.language_optical_grid_size = int(d("language_optical.grid_size", 224))
    settings.language_optical_canvas_size = int(d("language_optical.canvas_size", 518))
    settings.language_optical_wavelength_nm = float(
        d("language_optical.wavelength_nm", 532.0)
    )
    settings.language_optical_pixel_pitch_um = float(
        d("language_optical.pixel_pitch_um", 16.0)
    )
    settings.language_optical_distance_m = float(
        d("language_optical.distance_m", 0.10)
    )
    settings.language_optical_k_space_enabled = bool(
        d("language_optical.k_space.enabled", True)
    )
    settings.language_optical_theta_max_deg = float(
        d("language_optical.k_space.theta_max_deg", 0.65)
    )
    settings.language_optical_phase_parameterization = str(
        d("language_optical.phase.parameterization", "sigmoid")
    )
    settings.language_optical_phase_init = str(
        d("language_optical.phase.init", "small_normal")
    )
    settings.language_optical_phase_init_std = float(
        d("language_optical.phase.init_std", 0.02)
    )
    settings.language_optical_phase_dropout_mode = str(
        d("language_optical.phase.dropout_mode", "block_phase_bypass")
    )
    settings.language_optical_phase_dropout_p = float(
        d("language_optical.phase.dropout_p", 0.05)
    )
    settings.language_optical_phase_dropout_block_size = int(
        d("language_optical.phase.dropout_block_size", 8)
    )
    settings.language_optical_input_rms = float(
        d("language_optical.input_rms", 0.50)
    )
    settings.language_optical_ccd_target_mean = float(
        d("language_optical.ccd_target_mean", 0.25)
    )
    settings.language_optical_background_quantile = float(
        d("language_optical.normalization.background_quantile", 0.01)
    )
    settings.language_optical_normalization_clip = float(
        d("language_optical.normalization.relative_clip", 12.0)
    )
    settings.language_optical_log_compression = float(
        d("language_optical.normalization.log_compression", 1.0)
    )
    settings.language_optical_max_shift_pixels = int(
        d("language_optical.perturbation.max_shift_pixels", 4)
    )
    settings.language_optical_ccd_shift_pixels = int(
        d("language_optical.perturbation.ccd_shift_pixels", 4)
    )
    settings.language_optical_gain_min = float(
        d("language_optical.perturbation.gain_min", 0.5)
    )
    settings.language_optical_gain_max = float(
        d("language_optical.perturbation.gain_max", 2.0)
    )
    settings.language_optical_offset_fraction = float(
        d("language_optical.perturbation.offset_fraction", 0.03)
    )
    settings.language_optical_read_noise_fraction = float(
        d("language_optical.perturbation.read_noise_fraction", 0.01)
    )
    settings.hardware_phase_flip_vertical = bool(
        d("hardware.phase_mask.flip_vertical", True)
    )
    settings.hardware_phase_flip_horizontal = bool(
        d("hardware.phase_mask.flip_horizontal", False)
    )
    roi = d("hardware.ccd.roi_xywh", None)
    settings.hardware_ccd_roi_xywh = (
        None if roi is None else tuple(int(value) for value in roi)
    )
    settings.hardware_ccd_flip_vertical = bool(
        d("hardware.ccd.flip_vertical", False)
    )
    settings.hardware_ccd_flip_horizontal = bool(
        d("hardware.ccd.flip_horizontal", False)
    )
    settings.hardware_ccd_registration_mode = str(
        d("hardware.ccd.registration_mode", "center_crop_resize")
    )
    settings.hardware_ccd_physical_binning_factor = int(
        d("hardware.ccd.physical_binning_factor", 2)
    )
    settings.hardware_ccd_target_size = int(
        d("hardware.ccd.target_size", settings.language_optical_grid_size)
    )

    if settings.electronic_layers != 2:
        raise ValueError("Language-layer-2 optics requires exactly two electronic blocks")
    if settings.language_optical_grid_size != settings.max_language_tokens:
        raise ValueError("Optical grid_size must equal max_language_tokens")
    if settings.language_optical_canvas_size < settings.language_optical_grid_size:
        raise ValueError("Optical canvas_size must not be smaller than grid_size")
    if (settings.language_optical_canvas_size - settings.language_optical_grid_size) % 2:
        raise ValueError("Optical grid must be centered exactly in the propagation canvas")
    if not 0.0 < settings.optical_fusion_initial < 1.0:
        raise ValueError("hybrid.initial_fusion must be in (0,1)")
    if settings.lambda_ccd_operating_point < 0.0:
        raise ValueError("lambda_ccd_operating_point must be nonnegative")
    if (
        settings.language_optical_max_shift_pixels < 0
        or settings.language_optical_ccd_shift_pixels < 0
    ):
        raise ValueError("Optical input/CCD shift bounds must be nonnegative")
    if not 0.0 <= settings.language_optical_background_quantile < 0.5:
        raise ValueError("background_quantile must be in [0,0.5)")
    if not 0.0 < settings.language_optical_phase_dropout_p < 1.0:
        raise ValueError("phase dropout probability must be in (0,1)")
    if settings.lambda_router_balance or settings.lambda_router_importance:
        raise ValueError("This single-path optical experiment has no router loss")
    if settings.hardware_ccd_roi_xywh is not None:
        if len(settings.hardware_ccd_roi_xywh) != 4:
            raise ValueError("hardware.ccd.roi_xywh must be null or [x,y,width,height]")
        if any(value < 0 for value in settings.hardware_ccd_roi_xywh[:2]) or any(
            value <= 0 for value in settings.hardware_ccd_roi_xywh[2:]
        ):
            raise ValueError("hardware.ccd.roi_xywh contains invalid coordinates")
    if settings.hardware_ccd_registration_mode not in {
        "strict",
        "resize",
        "center_crop_resize",
    }:
        raise ValueError(
            "hardware.ccd.registration_mode must be strict, resize, or center_crop_resize"
        )
    if settings.hardware_ccd_physical_binning_factor <= 0:
        raise ValueError("hardware.ccd.physical_binning_factor must be positive")
    if settings.hardware_ccd_target_size != settings.language_optical_grid_size:
        raise ValueError("hardware.ccd.target_size must equal the optical grid size")
    return settings


def save_resolved_config(settings: Any) -> None:
    save_electronic_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["hybrid"] = {
        "type": "language_block2_parallel_optical_residual",
        "initial_electronic_checkpoint": str(settings.initial_electronic_checkpoint),
        "freeze_electronic": settings.hybrid_freeze_electronic,
        "initial_fusion": settings.optical_fusion_initial,
        "router_enabled": False,
    }
    values["language_optical"] = {
        "grid_size": settings.language_optical_grid_size,
        "canvas_size": settings.language_optical_canvas_size,
        "wavelength_nm": settings.language_optical_wavelength_nm,
        "pixel_pitch_um": settings.language_optical_pixel_pitch_um,
        "distance_m": settings.language_optical_distance_m,
        "normalization": {
            "type": "dark_quantile_then_frame_mean_then_log_row_layernorm",
            "background_quantile": settings.language_optical_background_quantile,
            "relative_clip": settings.language_optical_normalization_clip,
        },
        "ccd_operating_point_loss_weight": settings.lambda_ccd_operating_point,
    }
    values["hardware"] = {
        "phase_mask": {
            "flip_vertical": settings.hardware_phase_flip_vertical,
            "flip_horizontal": settings.hardware_phase_flip_horizontal,
        },
        "ccd": {
            "roi_xywh": (
                None
                if settings.hardware_ccd_roi_xywh is None
                else list(settings.hardware_ccd_roi_xywh)
            ),
            "flip_vertical": settings.hardware_ccd_flip_vertical,
            "flip_horizontal": settings.hardware_ccd_flip_horizontal,
            "registration_mode": settings.hardware_ccd_registration_mode,
            "physical_binning_factor": settings.hardware_ccd_physical_binning_factor,
            "target_size": settings.hardware_ccd_target_size,
        },
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
