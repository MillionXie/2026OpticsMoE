from __future__ import annotations

from typing import Any


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def apply_vision2_hybrid_settings(settings: Any, raw: dict[str, Any]) -> Any:
    """Attach the canonical MoE4 Vision2 contract to an existing task setting.

    Existing SALICON/ISIC/LSP configuration loaders deliberately retain their
    legacy fields. Only an explicitly enabled hybrid configuration is changed.
    """

    enabled = bool(_get(raw, "vision2_hybrid.enabled", False))
    settings.vision2_hybrid_enabled = enabled
    if not enabled:
        settings.ccd_operating_point_weight = 0.0
        return settings

    d = lambda path, default=None: _get(raw, f"vision2_hybrid.{path}", default)
    settings.electronic_width = int(d("electronic.width", 192))
    settings.electronic_layers = int(d("electronic.layers", 2))
    settings.electronic_expansion = float(d("electronic.expansion", 2.0))
    settings.electronic_dropout = float(d("electronic.dropout", 0.10))
    settings.electronic_initial_residual_weight = float(
        d("electronic.initial_residual_weight", 0.10)
    )
    settings.electronic_token_mixer_enabled = True
    settings.electronic_token_mixer_kernel_size = int(
        d("electronic.kernel_size", 3)
    )
    settings.electronic_vision_token_mixer_type = "depthwise_conv2d"
    settings.electronic_vision_token_mixer_kernel_size = int(
        d("electronic.kernel_size", 3)
    )
    # Required only by the shared optical class; Language is never executed.
    settings.electronic_language_token_mixer_type = "depthwise_conv1d"
    settings.electronic_language_token_mixer_kernel_size = 5

    # Each token fills one 224-pixel SLM row. A shared per-token output
    # adapter maps the 224-wide CCD readout back to the 192-D electronic latent.
    settings.input_adapter_dim = 224
    settings.detector_output_size = 224
    settings.max_visual_tokens = int(d("max_visual_tokens", 224))
    settings.max_language_tokens = settings.max_visual_tokens
    settings.optical_fusion_initial = float(d("initial_fusion", 0.05))
    settings.ccd_operating_point_weight = float(
        d("loss.ccd_operating_point_weight", 0.02)
    )
    settings.student_learning_rate = float(
        d("optimization.electronic_learning_rate", 1.0e-4)
    )
    settings.phase_learning_rate = float(
        d("optimization.phase_learning_rate", 1.0e-4)
    )
    settings.router_learning_rate = float(
        d("optimization.router_learning_rate", 5.0e-5)
    )
    settings.dense_readout_learning_rate = float(
        d("optimization.readout_learning_rate", 5.0e-5)
    )
    settings.dense_head_learning_rate = float(
        d("optimization.head_learning_rate", 3.0e-4)
    )

    settings.canvas_size = 518
    settings.active_size = 478
    settings.expert_size = 224
    settings.expert_pitch = 254
    settings.num_experts = 4
    settings.expert_grid_rows = 2
    settings.expert_grid_cols = 2
    settings.expert_layers = 1
    settings.top_k = 2
    settings.router_pool_size = int(d("router.pool_size", 14))
    settings.router_temperature = float(d("router.temperature", 1.0))
    settings.router_input_layernorm_enabled = True
    settings.router_input_layernorm_eps = 1.0e-5
    settings.amplitude_slm_weight_domain = "amplitude"
    settings.amplitude_slm_input_normalization = "none"
    settings.amplitude_phase_relay = "ideal_4f_identity"

    settings.wavelength_nm = float(d("physics.wavelength_nm", 532.0))
    settings.pixel_pitch_um = float(d("physics.pixel_pitch_um", 16.0))
    distance = float(d("physics.distance_m", 0.10))
    settings.expert_interlayer_distance_m = distance
    settings.last_expert_to_global_distance_m = distance
    settings.global_to_detector_distance_m = distance
    settings.k_space_constraint_enabled = bool(d("k_space.enabled", True))
    settings.theta_max_deg = float(d("k_space.theta_max_deg", 0.65))

    settings.phase_parameterization = str(d("phase.parameterization", "sigmoid"))
    settings.phase_init = str(d("phase.init", "small_normal"))
    settings.phase_init_std = float(d("phase.init_std", 0.02))
    settings.phase_dropout_mode = str(
        d("phase.dropout_mode", "block_phase_bypass")
    )
    settings.phase_dropout_p = float(d("phase.dropout_p", 0.05))
    settings.phase_dropout_block_size = int(d("phase.dropout_block_size", 8))
    settings.phase_dropout_batch_shared = True

    settings.interlayer_enabled = True
    settings.interlayer_per_expert_enabled = True
    settings.interlayer_elementwise_affine = False
    settings.interlayer_hard_route_mask = True
    settings.interlayer_reapply_routing_weights = True
    settings.interlayer_layernorm_eps = 1.0e-5
    settings.interlayer_nonlinearity = "relu"
    settings.interlayer_detector_integration_factor = 2
    settings.oeo_preserve_response_amplitude = False
    # Compatibility with older configuration objects.
    settings.oeo_preserve_amplitude = False
    settings.oeo_response_gain_min = 0.25
    settings.oeo_response_gain_max = 4.0

    settings.detector_layernorm_eps = 1.0e-5
    settings.detector_layernorm_affine = False
    settings.detector_layernorm_scope = "per_token"
    settings.detector_nonlinearity = "relu"
    settings.vision_tap_stages = ()

    settings.language_optical_input_rms = float(d("normalization.input_rms", 0.50))
    settings.language_optical_ccd_target_mean = float(
        d("normalization.ccd_target_mean", 0.25)
    )
    settings.language_optical_normalization_clip = float(
        d("normalization.relative_clip", 12.0)
    )
    settings.language_optical_log_compression = float(
        d("normalization.log_compression", 1.0)
    )
    settings.language_optical_max_shift_pixels = int(
        d("perturbation.max_shift_pixels", 8)
    )
    settings.language_optical_phase_shift_pixels = int(
        d("perturbation.phase_shift_pixels", 8)
    )
    settings.language_optical_ccd_shift_pixels = int(
        d("perturbation.ccd_shift_pixels", 8)
    )
    settings.language_optical_gain_min = float(d("perturbation.gain_min", 0.5))
    settings.language_optical_gain_max = float(d("perturbation.gain_max", 2.0))
    settings.language_optical_offset_fraction = float(
        d("perturbation.offset_fraction", 0.03)
    )
    settings.language_optical_read_noise_fraction = float(
        d("perturbation.read_noise_fraction", 0.01)
    )

    settings.hardware_phase_flip_vertical = bool(
        d("hardware.phase_flip_vertical", True)
    )
    settings.hardware_phase_flip_horizontal = bool(
        d("hardware.phase_flip_horizontal", False)
    )
    settings.hardware_ccd_flip_vertical = bool(
        d("hardware.ccd_flip_vertical", True)
    )
    settings.hardware_ccd_flip_horizontal = bool(
        d("hardware.ccd_flip_horizontal", True)
    )
    settings.hardware_ccd_target_size = 478
    settings.hardware_ccd_transport_size = 478
    settings.hardware_ccd_physical_binning_factor = 2
    settings.hardware_capture_train_limit = int(
        d("hardware.capture_train_limit", 256)
    )
    settings.hardware_capture_eval_limit = int(
        d("hardware.capture_eval_limit", 64)
    )
    settings.hardware_finetune_batch_size = int(
        d("hardware.finetune_batch_size", 8)
    )
    settings.hardware_finetune_learning_rate = float(
        d("hardware.finetune_learning_rate", 5.0e-5)
    )

    if settings.electronic_layers != 2:
        raise ValueError("Vision2 hybrid dense backbone requires exactly two blocks")
    if settings.electronic_width != 192:
        raise ValueError("The hardware-compatible dense backbone fixes width=192")
    if settings.max_visual_tokens > settings.expert_size:
        raise ValueError("Visual token count exceeds the 224-row optical interface")
    if not 0.0 < settings.optical_fusion_initial < 1.0:
        raise ValueError("vision2_hybrid.initial_fusion must be in (0,1)")
    if not 0.0 <= settings.phase_dropout_p < 1.0:
        raise ValueError("phase dropout probability must be in [0,1)")
    if settings.ccd_operating_point_weight < 0.0:
        raise ValueError("CCD operating-point loss weight cannot be negative")
    if min(
        settings.hardware_capture_train_limit,
        settings.hardware_capture_eval_limit,
        settings.hardware_finetune_batch_size,
    ) <= 0:
        raise ValueError("Hardware capture limits and batch size must be positive")
    return settings


__all__ = ["apply_vision2_hybrid_settings"]
