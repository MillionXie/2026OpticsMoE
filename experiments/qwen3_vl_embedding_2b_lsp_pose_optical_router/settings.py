from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    ROUTING_NORMALIZATIONS,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import (
    _get,
    _read,
    load_settings as load_lsp_settings,
)


ROUTER_BACKENDS = {"electronic", "optical"}


def _path(value: Any, base: Path, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required")
    result = Path(str(value)).expanduser()
    return (result if result.is_absolute() else base / result).resolve()


def _router_contract(settings: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "LSP_pose_vision_only",
        "backend": settings.router_backend,
        "top_k": settings.top_k,
        "temperature": settings.router_temperature,
        "weight_normalization": settings.router_weight_normalization,
        "straight_through": settings.router_straight_through,
        "geometry": {
            "canvas_size": settings.canvas_size,
            "active_size": settings.active_size,
            "expert_size": settings.expert_size,
            "expert_pitch": settings.expert_pitch,
            "num_experts": settings.num_experts,
            "grid": [settings.expert_grid_rows, settings.expert_grid_cols],
            "pixel_pitch_um": settings.pixel_pitch_um,
            "distance_m": settings.global_to_detector_distance_m,
        },
    }
    if settings.router_backend == "electronic":
        result["electronic"] = {
            "pool_size": settings.router_pool_size,
            "input_layernorm": settings.router_input_layernorm_enabled,
            "input_layernorm_eps": settings.router_input_layernorm_eps,
            "gate_init_std": settings.router_gate_init_std,
            "noise_std": settings.router_noise_std,
        }
    else:
        result["optical"] = {
            "detector_intervals": [
                list(interval) for interval in settings.optical_router_detector_intervals
            ],
            "score_normalization": settings.optical_router_score_normalization,
            "energy_eps": settings.optical_router_energy_eps,
            "capture_loss_scale": settings.optical_router_capture_loss_scale,
            "phase_dropout_p": settings.optical_router_phase_dropout_p,
            "phase_dropout_block_size": settings.optical_router_phase_dropout_block_size,
            "input_shift_pixels": settings.optical_router_input_shift_pixels,
            "phase_shift_pixels": settings.optical_router_phase_shift_pixels,
            "ccd_shift_pixels": settings.optical_router_ccd_shift_pixels,
        }
    return result


def _contract_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_settings(path: str | Path) -> Any:
    config_path = Path(path).expanduser().resolve()
    settings = load_lsp_settings(config_path)
    raw = _read(config_path)
    d = lambda key, default=None: _get(raw, key, default)

    if not settings.vision2_hybrid_enabled:
        raise ValueError("This project requires vision2_hybrid.enabled=true")
    settings.router_backend = str(d("router_experiment.backend", "electronic"))
    settings.top_k = int(d("router_experiment.top_k", 2))
    settings.router_weight_normalization = str(
        d("router_experiment.weight_normalization", "power_l2")
    )
    settings.router_straight_through = bool(
        d("router_experiment.straight_through", True)
    )
    settings.router_gate_init_std = float(
        d("router_experiment.electronic.gate_init_std", 0.01)
    )
    settings.router_noise_std = float(
        d("router_experiment.train_logit_noise_std", 0.10)
    )

    settings.initialization_seed = int(d("protocol.initialization_seed", 42))
    settings.periodic_test_interval_epochs = int(
        d("protocol.test_interval_epochs", 5)
    )
    settings.periodic_test_at_epoch_one = bool(
        d("protocol.test_at_epoch_one", True)
    )
    settings.periodic_test_at_final_epoch = bool(
        d("protocol.test_at_final_epoch", True)
    )
    settings.common_initialization_checkpoint = _path(
        d("protocol.common_initialization_checkpoint"),
        config_path.parent,
        "protocol.common_initialization_checkpoint",
    )
    settings.evaluate_test_each_epoch = False

    # Exact robust Caltech Vision2 physical/body contract.  Do not route
    # through vision2_hybrid_dense's older 16-um non-robust core.
    settings.canvas_size = 518
    settings.active_size = 478
    settings.expert_size = 224
    settings.expert_pitch = 254
    settings.num_experts = 4
    settings.expert_grid_rows = 2
    settings.expert_grid_cols = 2
    settings.expert_layers = 1
    settings.electronic_width = 192
    settings.electronic_layers = 2
    settings.electronic_vision_token_mixer_type = "depthwise_conv2d"
    settings.electronic_vision_token_mixer_kernel_size = 3
    settings.max_visual_tokens = 224
    settings.max_language_tokens = 224
    settings.input_adapter_dim = 224
    settings.detector_output_size = 224
    settings.optical_fusion_minimum = float(
        d("robust_vision.minimum_optical_fusion", 0.05)
    )
    settings.optical_fusion_initial = float(
        d("robust_vision.initial_optical_fusion", 0.055)
    )

    settings.language_optical_wavelength_nm = 532.0
    settings.language_optical_pixel_pitch_um = float(
        d("robust_vision.pixel_pitch_um", 17.0)
    )
    settings.language_optical_distance_m = float(
        d("robust_vision.distance_m", 0.10)
    )
    settings.language_optical_k_space_enabled = bool(
        d("robust_vision.k_space.enabled", True)
    )
    settings.language_optical_theta_max_deg = float(
        d("robust_vision.k_space.theta_max_deg", 0.65)
    )
    settings.language_optical_input_rms = float(
        d("robust_vision.normalization.input_rms", 0.50)
    )
    settings.language_optical_ccd_target_mean = float(
        d("robust_vision.normalization.ccd_target_mean", 0.25)
    )
    settings.language_optical_normalization_clip = float(
        d("robust_vision.normalization.relative_clip", 12.0)
    )
    settings.language_optical_log_compression = float(
        d("robust_vision.normalization.log_compression", 1.0)
    )
    settings.language_optical_max_shift_pixels = int(
        d("robust_vision.perturbation.input_shift_pixels", 16)
    )
    settings.language_optical_phase_shift_pixels = int(
        d("robust_vision.perturbation.phase_shift_pixels", 16)
    )
    settings.language_optical_ccd_shift_pixels = int(
        d("robust_vision.perturbation.ccd_shift_pixels", 16)
    )
    settings.language_optical_gain_min = float(
        d("robust_vision.perturbation.gain_min", 0.4)
    )
    settings.language_optical_gain_max = float(
        d("robust_vision.perturbation.gain_max", 2.5)
    )
    settings.language_optical_offset_fraction = float(
        d("robust_vision.perturbation.offset_fraction", 0.05)
    )
    settings.language_optical_read_noise_fraction = float(
        d("robust_vision.perturbation.read_noise_fraction", 0.015)
    )
    settings.language_optical_ccd_noise_distribution = str(
        d(
            "robust_vision.perturbation.ccd_noise.distribution",
            "legacy_uniform_offset_gaussian",
        )
    )
    settings.language_optical_ccd_noise_mean_fraction = float(
        d("robust_vision.perturbation.ccd_noise.mean_fraction", 0.0)
    )
    settings.language_optical_ccd_noise_std_fraction = float(
        d("robust_vision.perturbation.ccd_noise.std_fraction", 0.0)
    )
    settings.language_optical_ccd_noise_min_fraction = float(
        d("robust_vision.perturbation.ccd_noise.min_fraction", -0.10)
    )
    settings.language_optical_ccd_noise_max_fraction = float(
        d("robust_vision.perturbation.ccd_noise.max_fraction", 0.20)
    )
    settings.language_optical_zero_order_enabled = bool(
        d("robust_vision.perturbation.zero_order.enabled", False)
    )
    settings.language_optical_amplitude_zero_order_intensity_min = float(
        d(
            "robust_vision.perturbation.zero_order.amplitude_intensity_fraction_min",
            0.0,
        )
    )
    settings.language_optical_amplitude_zero_order_intensity_max = float(
        d(
            "robust_vision.perturbation.zero_order.amplitude_intensity_fraction_max",
            0.0,
        )
    )
    settings.language_optical_phase_zero_order_intensity_min = float(
        d(
            "robust_vision.perturbation.zero_order.phase_intensity_fraction_min",
            0.0,
        )
    )
    settings.language_optical_phase_zero_order_intensity_max = float(
        d(
            "robust_vision.perturbation.zero_order.phase_intensity_fraction_max",
            0.0,
        )
    )
    settings.language_optical_zero_order_random_relative_phase = bool(
        d("robust_vision.perturbation.zero_order.random_relative_phase", True)
    )

    settings.language_optical_phase_parameterization = "sigmoid"
    settings.language_optical_phase_init = "small_normal"
    settings.language_optical_phase_init_std = 0.02
    settings.language_optical_phase_dropout_mode = "block_phase_bypass"
    settings.language_optical_phase_dropout_p = float(
        d("robust_vision.phase.dropout_p", 0.08)
    )
    settings.language_optical_phase_dropout_block_size = int(
        d("robust_vision.phase.dropout_block_size", 8)
    )
    settings.phase_parameterization = settings.language_optical_phase_parameterization
    settings.phase_init = settings.language_optical_phase_init
    settings.phase_init_std = settings.language_optical_phase_init_std
    settings.phase_dropout_mode = settings.language_optical_phase_dropout_mode
    settings.phase_dropout_p = settings.language_optical_phase_dropout_p
    settings.phase_dropout_block_size = settings.language_optical_phase_dropout_block_size
    settings.wavelength_nm = settings.language_optical_wavelength_nm
    settings.pixel_pitch_um = settings.language_optical_pixel_pitch_um
    settings.expert_interlayer_distance_m = settings.language_optical_distance_m
    settings.last_expert_to_global_distance_m = settings.language_optical_distance_m
    settings.global_to_detector_distance_m = settings.language_optical_distance_m
    settings.k_space_constraint_enabled = settings.language_optical_k_space_enabled
    settings.theta_max_deg = settings.language_optical_theta_max_deg

    settings.ccd_operating_point_weight = float(
        d("loss.ccd_operating_point_weight", 0.02)
    )
    settings.lambda_ccd_operating_point = settings.ccd_operating_point_weight
    settings.router_balance_weight = float(d("loss.router_balance_weight", 0.05))
    settings.router_importance_weight = float(
        d("loss.router_importance_weight", 0.005)
    )
    settings.teacher_distill_weight = 0.0
    settings.router_response_consistency_weight = 0.0
    settings.phase_dc_weight = 0.0

    settings.optical_router_detector_intervals = tuple(
        tuple(int(value) for value in interval)
        for interval in d(
            "router_experiment.optical.detector_intervals",
            [[164, 223], [255, 314]],
        )
    )
    settings.optical_router_score_normalization = str(
        d(
            "router_experiment.optical.score_normalization",
            "standardized_region_energy",
        )
    )
    settings.optical_router_energy_eps = float(
        d("router_experiment.optical.energy_eps", 1.0e-8)
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
        raise ValueError(f"Router backend must be one of {sorted(ROUTER_BACKENDS)}")
    if settings.top_k not in {1, 2, 4}:
        raise ValueError("Router top_k must be one of 1, 2, 4")
    if settings.router_backend == "optical" and settings.top_k != 2:
        raise ValueError("The requested optical-Router comparison is fixed at top-k=2")
    if settings.router_weight_normalization not in ROUTING_NORMALIZATIONS:
        raise ValueError("Unsupported Router amplitude normalization")
    if settings.router_weight_normalization != "power_l2":
        raise ValueError("Fair top-k comparison requires power_l2 normalization")
    if not settings.router_straight_through:
        raise ValueError("Fair top-k comparison requires the corrected STE")
    if settings.periodic_test_interval_epochs < 1:
        raise ValueError("protocol.test_interval_epochs must be >= 1")
    if not settings.periodic_test_at_epoch_one:
        raise ValueError("Formal periodic-test protocol evaluates epoch 1")
    if not settings.periodic_test_at_final_epoch:
        raise ValueError("Formal periodic-test protocol evaluates the final epoch")
    if settings.initialization_seed != 42 or settings.random_seed != 42:
        raise ValueError("Formal single-seed comparison fixes initialization/data seed 42")
    expected = (518, 478, 224, 254, 4, 2, 2, 1, 192, 2)
    actual = (
        settings.canvas_size,
        settings.active_size,
        settings.expert_size,
        settings.expert_pitch,
        settings.num_experts,
        settings.expert_grid_rows,
        settings.expert_grid_cols,
        settings.expert_layers,
        settings.electronic_width,
        settings.electronic_layers,
    )
    if actual != expected:
        raise ValueError(f"Robust Vision2 contract must be {expected}, got {actual}")
    if settings.pixel_pitch_um != 17.0 or settings.global_to_detector_distance_m != 0.10:
        raise ValueError("Formal LSP Router physics is fixed to 17 um and 10 cm")
    if not 0.0 <= settings.optical_fusion_minimum < settings.optical_fusion_initial < 1.0:
        raise ValueError("Optical fusion must start strictly above its hard minimum")
    if max(
        settings.language_optical_max_shift_pixels,
        settings.language_optical_phase_shift_pixels,
        settings.language_optical_ccd_shift_pixels,
    ) > 16:
        raise ValueError("Formal robust displacement bounds cannot exceed 16 pixels")
    if len(settings.optical_router_detector_intervals) != 2:
        raise ValueError("Optical Router needs two detector intervals per axis")
    detector_center = 0.5 * (settings.active_size - 1)
    interval_centers: list[float] = []
    for start, end in settings.optical_router_detector_intervals:
        if not 0 <= start < end <= settings.active_size:
            raise ValueError("Optical Router interval exceeds canonical CCD")
        interval_centers.append(0.5 * (start + end - 1))
    radial = max(
        math.hypot(x - detector_center, y - detector_center)
        for x in interval_centers
        for y in interval_centers
    )
    required_angle = math.degrees(
        math.asin(
            radial
            * settings.pixel_pitch_um
            * 1.0e-6
            / settings.global_to_detector_distance_m
        )
    )
    settings.optical_router_required_center_angle_deg = required_angle
    if settings.language_optical_k_space_enabled and required_angle > settings.theta_max_deg:
        raise ValueError("Optical Router detector centres exceed the k-space cutoff")
    settings.router_contract = _router_contract(settings)
    settings.router_contract_sha256 = _contract_sha256(settings.router_contract)
    return settings


def save_resolved_config(settings: Any) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    values = settings.to_dict()
    values.update(
        {
            "protocol": {
                "initialization_seed": settings.initialization_seed,
                "validation_count": 0,
                "test_interval_epochs": settings.periodic_test_interval_epochs,
                "test_at_epoch_one": settings.periodic_test_at_epoch_one,
                "test_at_final_epoch": settings.periodic_test_at_final_epoch,
                "common_initialization_checkpoint": str(
                    settings.common_initialization_checkpoint
                ),
                "checkpoint_selection": (
                    "max periodic-test PCK@0.2 torso; then min torso NME; "
                    "then min periodic-test loss; then earliest epoch"
                ),
                "test_evaluated_during_training": True,
                "test_used_for_checkpoint_selection": True,
            },
            "router_experiment": {
                "contract": settings.router_contract,
                "contract_sha256": settings.router_contract_sha256,
            },
            "robust_vision": {
                "source_core": (
                    "qwen3_vl_embedding_2b_caltech101_four_layer_"
                    "optical_retrieval_10cm_robust"
                ),
                "minimum_optical_fusion": settings.optical_fusion_minimum,
                "initial_optical_fusion": settings.optical_fusion_initial,
                "pixel_pitch_um": settings.pixel_pitch_um,
                "distance_m": settings.global_to_detector_distance_m,
                "input_phase_ccd_shift_pixels": [
                    settings.language_optical_max_shift_pixels,
                    settings.language_optical_phase_shift_pixels,
                    settings.language_optical_ccd_shift_pixels,
                ],
            },
        }
    )
    (settings.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


__all__ = ["ROUTER_BACKENDS", "load_settings", "save_resolved_config"]
