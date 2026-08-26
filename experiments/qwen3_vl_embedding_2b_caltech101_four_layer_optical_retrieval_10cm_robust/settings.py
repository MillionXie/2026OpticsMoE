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

    checkpoint_value = d("hybrid.initial_electronic_checkpoint")
    settings.initial_electronic_checkpoint = (
        None if checkpoint_value is None else Path(str(checkpoint_value)).expanduser()
    )
    if (
        settings.initial_electronic_checkpoint is not None
        and not settings.initial_electronic_checkpoint.is_absolute()
    ):
        settings.initial_electronic_checkpoint = (
            config_path.parent / settings.initial_electronic_checkpoint
        ).resolve()
    settings.hybrid_freeze_electronic = bool(d("hybrid.freeze_electronic", True))
    settings.optical_fusion_initial = float(d("hybrid.initial_fusion", 0.05))
    settings.optical_fusion_minimum = float(
        d("hybrid.minimum_optical_fusion", 0.0)
    )
    settings.lambda_ccd_operating_point = float(
        d("training.lambda_ccd_operating_point", 0.02)
    )
    # The electronic parent intentionally zeros every optical/router setting.
    # Restore the experiment-local values after loading the electronic base so
    # YAML phase/router controls actually reach the shared optimizer.
    settings.phase_learning_rate = float(
        d("training.phase_learning_rate", 5.0e-4)
    )
    settings.phase_focus_enabled = bool(
        d("training.phase_focus.enabled", False)
    )
    settings.phase_focus_warmup_epochs = int(
        d("training.phase_focus.warmup_epochs", 5)
    )
    settings.phase_focus_interval_epochs = int(
        d("training.phase_focus.interval_epochs", 3)
    )
    settings.lambda_router_balance = float(
        d("training.lambda_router_balance", 0.0)
    )
    settings.lambda_router_importance = float(
        d("training.lambda_router_importance", 0.0)
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
    settings.language_optical_normalization_clip = float(
        d("language_optical.normalization.relative_clip", 12.0)
    )
    settings.language_optical_log_compression = float(
        d("language_optical.normalization.log_compression", 1.0)
    )
    settings.language_optical_max_shift_pixels = int(
        d("language_optical.perturbation.max_shift_pixels", 4)
    )
    settings.language_optical_phase_shift_pixels = int(
        d("language_optical.perturbation.phase_shift_pixels", 4)
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
    settings.hardware_amplitude_slm_width = int(
        d("hardware.amplitude_slm.width", 1920)
    )
    settings.hardware_amplitude_slm_height = int(
        d("hardware.amplitude_slm.height", 1080)
    )
    settings.hardware_amplitude_slm_pixel_pitch_um = float(
        d("hardware.amplitude_slm.pixel_pitch_um", 8.0)
    )
    settings.hardware_amplitude_slm_center_x = float(
        d(
            "hardware.amplitude_slm.center_x",
            settings.hardware_amplitude_slm_width / 2.0,
        )
    )
    settings.hardware_amplitude_slm_center_y = float(
        d(
            "hardware.amplitude_slm.center_y",
            settings.hardware_amplitude_slm_height / 2.0,
        )
    )
    settings.hardware_amplitude_bright_value_uint8 = int(
        d("hardware.amplitude_slm.bright_value_uint8", 255)
    )
    settings.hardware_amplitude_dark_value_uint8 = int(
        d("hardware.amplitude_slm.dark_value_uint8", 0)
    )
    settings.hardware_amplitude_invert_before_export = bool(
        d("hardware.amplitude_slm.invert_before_export", False)
    )
    settings.hardware_phase_slm_width = int(
        d("hardware.phase_mask.width", 1920)
    )
    settings.hardware_phase_slm_height = int(
        d("hardware.phase_mask.height", 1200)
    )
    settings.hardware_phase_slm_pixel_pitch_um = float(
        d("hardware.phase_mask.pixel_pitch_um", 8.0)
    )
    settings.hardware_phase_slm_center_x = float(
        d(
            "hardware.phase_mask.center_x",
            settings.hardware_phase_slm_width / 2.0,
        )
    )
    settings.hardware_phase_slm_center_y = float(
        d(
            "hardware.phase_mask.center_y",
            settings.hardware_phase_slm_height / 2.0,
        )
    )
    settings.hardware_ccd_flip_vertical = bool(
        d("hardware.ccd.flip_vertical", False)
    )
    settings.hardware_ccd_flip_horizontal = bool(
        d("hardware.ccd.flip_horizontal", False)
    )
    settings.hardware_ccd_physical_binning_factor = int(
        d("hardware.ccd.physical_binning_factor", 2)
    )
    settings.hardware_ccd_target_size = int(
        d("hardware.ccd.target_size", settings.active_size)
    )
    settings.hardware_ccd_transport_size = int(
        d("hardware.ccd.transport_size", settings.hardware_ccd_target_size)
    )
    # The electronic experiment intentionally forces a one-path compatibility
    # router. Restore the canonical Grocery MoE4 geometry for this hybrid.
    settings.num_experts = int(d("optical.geometry.num_experts", 4))
    settings.expert_grid_rows = int(d("optical.geometry.grid_rows", 2))
    settings.expert_grid_cols = int(d("optical.geometry.grid_cols", 2))
    settings.top_k = int(d("optical.router.top_k", 2))
    settings.router_learning_rate = float(d("training.router_learning_rate", 5.0e-5))
    # HomogeneousMoEOpticalCore consumes the canonical setting names.  Make the
    # experiment-local phase controls authoritative for all four expert masks
    # and the global mask.
    settings.phase_parameterization = settings.language_optical_phase_parameterization
    settings.phase_init = settings.language_optical_phase_init
    settings.phase_init_std = settings.language_optical_phase_init_std
    settings.phase_dropout_mode = settings.language_optical_phase_dropout_mode
    settings.phase_dropout_p = settings.language_optical_phase_dropout_p
    settings.phase_dropout_block_size = (
        settings.language_optical_phase_dropout_block_size
    )
    settings.wavelength_nm = settings.language_optical_wavelength_nm
    settings.pixel_pitch_um = settings.language_optical_pixel_pitch_um
    settings.expert_interlayer_distance_m = settings.language_optical_distance_m
    settings.k_space_constraint_enabled = settings.language_optical_k_space_enabled
    settings.theta_max_deg = settings.language_optical_theta_max_deg

    if settings.electronic_layers != 2:
        raise ValueError("Language-layer-2 optics requires exactly two electronic blocks")
    if settings.language_optical_grid_size != settings.max_language_tokens:
        raise ValueError("Optical grid_size must equal max_language_tokens")
    if settings.language_optical_canvas_size < settings.language_optical_grid_size:
        raise ValueError("Optical canvas_size must not be smaller than grid_size")
    if (settings.language_optical_canvas_size - settings.language_optical_grid_size) % 2:
        raise ValueError("Optical grid must be centered exactly in the propagation canvas")
    if not 0.0 <= settings.optical_fusion_minimum < 1.0:
        raise ValueError("hybrid.minimum_optical_fusion must be in [0,1)")
    if not settings.optical_fusion_minimum < settings.optical_fusion_initial < 1.0:
        raise ValueError(
            "hybrid.initial_fusion must be greater than the optical floor and below 1"
        )
    if abs(settings.language_optical_distance_m - 0.10) > 1.0e-12:
        raise ValueError("The robust project requires exactly 10 cm propagation")
    if settings.lambda_ccd_operating_point < 0.0:
        raise ValueError("lambda_ccd_operating_point must be nonnegative")
    if (
        settings.language_optical_max_shift_pixels < 0
        or settings.language_optical_phase_shift_pixels < 0
        or settings.language_optical_ccd_shift_pixels < 0
    ):
        raise ValueError("Optical input/CCD shift bounds must be nonnegative")
    if not 0.0 < settings.language_optical_phase_dropout_p < 1.0:
        raise ValueError("phase dropout probability must be in (0,1)")
    if settings.phase_learning_rate <= 0.0:
        raise ValueError("Four-layer joint training requires a positive phase learning rate")
    if settings.phase_focus_warmup_epochs < 0:
        raise ValueError("training.phase_focus.warmup_epochs must be nonnegative")
    if settings.phase_focus_interval_epochs <= 0:
        raise ValueError("training.phase_focus.interval_epochs must be positive")
    if settings.lambda_router_balance < 0.0 or settings.lambda_router_importance < 0.0:
        raise ValueError("Router auxiliary-loss weights must be nonnegative")
    if settings.hardware_ccd_physical_binning_factor <= 0:
        raise ValueError("hardware.ccd.physical_binning_factor must be positive")
    if min(
        settings.hardware_amplitude_slm_width,
        settings.hardware_amplitude_slm_height,
        settings.hardware_phase_slm_width,
        settings.hardware_phase_slm_height,
    ) <= 0:
        raise ValueError("Hardware SLM dimensions must be positive")
    if min(
        settings.hardware_amplitude_slm_pixel_pitch_um,
        settings.hardware_phase_slm_pixel_pitch_um,
    ) <= 0.0:
        raise ValueError("Hardware SLM pixel pitches must be positive")
    if (
        settings.hardware_amplitude_bright_value_uint8 != 255
        or settings.hardware_amplitude_dark_value_uint8 != 0
        or settings.hardware_amplitude_invert_before_export
    ):
        raise ValueError(
            "Corrected amplitude contract requires 255=bright, 0=dark, no inversion"
        )
    if settings.hardware_ccd_target_size != settings.active_size:
        raise ValueError("hardware.ccd.target_size must equal MoE active_size=478")
    if settings.hardware_ccd_transport_size not in {
        settings.active_size,
        settings.active_size * settings.hardware_ccd_physical_binning_factor,
    }:
        raise ValueError(
            "hardware.ccd.transport_size must be active_size or "
            "active_size*physical_binning_factor"
        )
    if settings.hybrid_freeze_electronic:
        raise ValueError("Four-layer joint training requires hybrid.freeze_electronic=false")
    if (
        settings.canvas_size != 518
        or settings.active_size != 478
        or settings.expert_size != 224
        or settings.expert_pitch != 254
        or settings.num_experts != 4
        or settings.expert_grid_rows != 2
        or settings.expert_grid_cols != 2
        or settings.top_k != 2
    ):
        raise ValueError(
            "Language layer-2 hardware experiment requires canonical "
            "MoE4: canvas518/active478/expert224/grid2x2/top_k2"
        )
    return settings


def save_resolved_config(settings: Any) -> None:
    save_electronic_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["hybrid"] = {
        "type": "vision2_language2_moe4_10cm_robust_bounded_fusion",
        "initial_electronic_checkpoint": (
            None
            if settings.initial_electronic_checkpoint is None
            else str(settings.initial_electronic_checkpoint)
        ),
        "freeze_electronic": settings.hybrid_freeze_electronic,
        "initial_fusion": {
            "block1": settings.optical_fusion_initial,
            "block2": settings.optical_fusion_initial,
        },
        "minimum_optical_fusion": settings.optical_fusion_minimum,
        "fusion_parameterization": (
            "minimum + (1-minimum)*sigmoid(raw_gate)"
        ),
        "router_enabled": True,
        "router_loss_enabled": bool(
            settings.lambda_router_balance or settings.lambda_router_importance
        ),
    }
    values["language_optical"] = {
        "layout": "MoE4_2x2_topk2",
        "expert_size": settings.expert_size,
        "active_size": settings.active_size,
        "canvas_size": settings.language_optical_canvas_size,
        "wavelength_nm": settings.language_optical_wavelength_nm,
        "pixel_pitch_um": settings.language_optical_pixel_pitch_um,
        "distance_m": settings.language_optical_distance_m,
        "k_space": {
            "enabled": settings.language_optical_k_space_enabled,
            "theta_max_deg": settings.language_optical_theta_max_deg,
        },
        "phase": {
            "parameterization": settings.language_optical_phase_parameterization,
            "init": settings.language_optical_phase_init,
            "init_std": settings.language_optical_phase_init_std,
            "dropout_mode": settings.language_optical_phase_dropout_mode,
            "dropout_p": settings.language_optical_phase_dropout_p,
            "dropout_block_size": settings.language_optical_phase_dropout_block_size,
            "activated_during_joint_training": True,
        },
        "perturbation": {
            "input_shift_pixels": settings.language_optical_max_shift_pixels,
            "phase_relative_shift_pixels": settings.language_optical_phase_shift_pixels,
            "ccd_shift_pixels": settings.language_optical_ccd_shift_pixels,
            "gain_min": settings.language_optical_gain_min,
            "gain_max": settings.language_optical_gain_max,
            "offset_fraction": settings.language_optical_offset_fraction,
            "read_noise_fraction": settings.language_optical_read_noise_fraction,
        },
        "normalization": {
            "type": "frame_mean_then_log_then_pool_row_layernorm",
            "background_subtraction": False,
            "relative_clip": settings.language_optical_normalization_clip,
        },
        "ccd_operating_point_loss_weight": settings.lambda_ccd_operating_point,
    }
    values["hardware"] = {
        "amplitude_slm": {
            "width": settings.hardware_amplitude_slm_width,
            "height": settings.hardware_amplitude_slm_height,
            "pixel_pitch_um": settings.hardware_amplitude_slm_pixel_pitch_um,
            "center_x": settings.hardware_amplitude_slm_center_x,
            "center_y": settings.hardware_amplitude_slm_center_y,
            "bright_value_uint8": settings.hardware_amplitude_bright_value_uint8,
            "dark_value_uint8": settings.hardware_amplitude_dark_value_uint8,
            "invert_before_export": settings.hardware_amplitude_invert_before_export,
            "mapping": "one_to_one" if (
                settings.hardware_amplitude_slm_pixel_pitch_um
                == settings.language_optical_pixel_pitch_um
            ) else "physical_pitch_nearest",
        },
        "phase_mask": {
            "flip_vertical": settings.hardware_phase_flip_vertical,
            "flip_horizontal": settings.hardware_phase_flip_horizontal,
            "width": settings.hardware_phase_slm_width,
            "height": settings.hardware_phase_slm_height,
            "pixel_pitch_um": settings.hardware_phase_slm_pixel_pitch_um,
            "center_x": settings.hardware_phase_slm_center_x,
            "center_y": settings.hardware_phase_slm_center_y,
            "mapping": "physical_pitch_nearest",
        },
        "ccd": {
            "flip_vertical": settings.hardware_ccd_flip_vertical,
            "flip_horizontal": settings.hardware_ccd_flip_horizontal,
            "physical_binning_factor": settings.hardware_ccd_physical_binning_factor,
            "target_size": settings.hardware_ccd_target_size,
            "transport_size": settings.hardware_ccd_transport_size,
        },
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
