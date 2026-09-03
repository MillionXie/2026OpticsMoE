from __future__ import annotations

from typing import Any

import torch

from .data import cache_report, load_single_metric_cache, read_manifest, split_counts
from .settings import ExperimentSettings


QUALITY_CHANNELS = (
    "R",
    "G",
    "B",
    "luminance",
    "sobel_x",
    "sobel_y",
    "gradient_magnitude",
    "absolute_laplacian",
    "local_std_5x5",
    "saturation",
    "previous_frame_luminance_abs_difference",
    "x_coordinate",
    "y_coordinate",
    "time_coordinate",
)


def run_preflight(
    settings: ExperimentSettings, *, require_cache: bool = True
) -> dict[str, Any]:
    settings.validate()
    report: dict[str, Any] = {
        "status": "ready",
        "architecture": settings.architecture_label,
        "target": settings.target_name,
        "output": "one continuous MOS scalar",
        "prompt": settings.prompt,
        "separate_spatial_temporal_models": True,
        "validation_used": False,
        "test_selection_interval_epochs": settings.test_interval_epochs,
        "qwen_front": {
            "checkpoint_family": "Qwen3-VL-2B-Instruct",
            "official_processor": True,
            "vision": "frozen patch_embed + official interpolated position embedding",
            "vision_blocks_executed": 0,
            "vision_merger_executed": False,
            "text": "chat template + tokenizer + frozen embed_tokens",
            "language_blocks_executed": 0,
            "lm_head_executed": False,
            "cached_before_student_training": True,
        },
        "student": {
            "attention_modules": 0,
            "transformer_blocks": 0,
            "frame_count": settings.frame_count,
            "qwen_vision_token_shape_per_video": [settings.frame_count, 49, 1024],
            "quality_side_shape_per_video": [settings.frame_count, 49, 14],
            "quality_channels": list(QUALITY_CHANNELS),
            "internal_width": settings.model_width,
        },
        "geometry": {
            "canvas": settings.geometry.canvas_size,
            "active": settings.geometry.active_size,
            "parallel_frame_grid": [
                settings.geometry.lane_grid,
                settings.geometry.lane_grid,
            ],
            "parallel_lane_size": settings.geometry.lane_size,
            "parallel_lane_pitch": settings.geometry.lane_pitch,
            "parallel_lane_origins_active_yx": [
                list(value) for value in settings.geometry.lane_origins
            ],
            "parallel_expert_grid_per_frame": [2, 2],
            "parallel_expert_size": settings.geometry.parallel_expert_size,
            "parallel_expert_pitch": settings.geometry.parallel_expert_pitch,
            "serial_expert_size": settings.geometry.serial_expert_size,
            "serial_expert_pitch": settings.geometry.serial_expert_pitch,
        },
        "router": {
            "implementation": "optical region-energy router",
            "top_k": settings.top_k,
            "parallel_decisions_per_video": settings.frame_count,
            "serial_decisions_per_video": 1,
            "parallel_detector_intervals": [
                list(value) for value in settings.parallel_router_intervals
            ],
            "serial_detector_intervals": [
                list(value) for value in settings.serial_router_intervals
            ],
        },
        "fusion": {
            "equation": "F=rE*((1-alpha)*E/rE + alpha*O/rO)/rms(mixture)",
            "independent_alpha_per_stage": True,
            "alpha_range": [settings.alpha_min, settings.alpha_max],
            "alpha_initial": settings.alpha_initial,
            "same_checkpoint_optical_bypass_only": True,
        },
        "unmodulated_leakage": {
            "model": "coherent field mixture",
            "nominal_power_train_range": [
                settings.unmodulated_power_fraction_min,
                settings.unmodulated_power_fraction_max,
            ],
            "nominal_power_evaluation": settings.unmodulated_power_fraction_eval,
        },
        "phase_snapshots": {
            "interval_epochs": settings.phase_snapshot_interval_epochs,
            "format": "optical_phase_evolution_snapshot_v1",
        },
    }

    if settings.manifest_path is None or not settings.manifest_path.is_file():
        if not settings.synthetic:
            report["status"] = "blocked"
            report["manifest_error"] = f"Missing manifest: {settings.manifest_path}"
    else:
        counts = split_counts(read_manifest(settings.manifest_path))
        report["manifest_counts"] = counts
        if not settings.synthetic and counts != {
            "train": 2250,
            "validation": 0,
            "test": 558,
        }:
            report["status"] = "blocked"
            report["manifest_error"] = (
                "Formal LGVQ split must be 2250 train / 0 validation / 558 test; "
                f"got {counts}"
            )

    cache_paths = {
        "vision": settings.vision_cache_path,
        "language": settings.language_cache_path,
    }
    missing = [name for name, path in cache_paths.items() if path is None or not path.is_file()]
    if missing and require_cache and not settings.synthetic:
        report["status"] = "blocked"
        report["cache_error"] = f"Missing target cache(s): {missing}"
    elif not missing:
        payload = load_single_metric_cache(settings)
        report["cache"] = cache_report(payload)
        if report["cache"]["split_counts"] != {
            "train": 2250,
            "validation": 0,
            "test": 558,
        } and not settings.synthetic:
            report["status"] = "blocked"
            report["cache_error"] = "Cached split counts do not match the formal manifest"

    if settings.training_soft_targets_path is not None:
        report["training_soft_targets"] = {
            "path": str(settings.training_soft_targets_path),
            "exists": settings.training_soft_targets_path.is_file(),
            "training_only": True,
        }
        if settings.soft_target_weight > 0 and not settings.training_soft_targets_path.is_file():
            report["status"] = "blocked"
            report["soft_target_error"] = "Configured training soft-target file is missing"
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
    return report


__all__ = ["QUALITY_CHANNELS", "run_preflight"]
