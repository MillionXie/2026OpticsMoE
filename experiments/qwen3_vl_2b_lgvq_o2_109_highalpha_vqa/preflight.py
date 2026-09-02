from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import cache_report, file_sha256, load_canonical_cache, read_manifest, split_counts
from .settings import ExperimentSettings


def run_preflight(settings: ExperimentSettings, *, require_cache: bool = True) -> dict[str, Any]:
    settings.validate()
    report: dict[str, Any] = {
        "status": "ready",
        "architecture": settings.architecture_label,
        "targets": ["spatial", "temporal"],
        "alignment": "forbidden",
        "validation_used": False,
        "test_selection_interval_epochs": settings.test_interval_epochs,
        "geometry": {
            "canvas": settings.geometry.canvas_size,
            "active": settings.geometry.active_size,
            "frame_lane": settings.geometry.quadrant_size,
            "frame_lane_pitch": (
                settings.geometry.quadrant_size + settings.geometry.lane_gap
            ),
            "frame_lane_gap": settings.geometry.lane_gap,
            "frame_lane_origins_active_yx": [
                list(value) for value in settings.geometry.lane_origins
            ],
            "expert": settings.geometry.expert_size,
            "expert_pitch": settings.geometry.expert_pitch,
            "expert_gap": (
                settings.geometry.expert_pitch - settings.geometry.expert_size
            ),
            "expert_grid_starts_active": sorted(
                {
                    origin + local
                    for origin in {
                        coordinate
                        for point in settings.geometry.lane_origins
                        for coordinate in point
                    }
                    for local in (0, settings.geometry.expert_pitch)
                }
            ),
            "frame_lanes": 4,
            "experts_per_lane": 4,
        },
        "router": {
            "backend": settings.router_backend,
            "top_k": settings.top_k,
            "weight_normalization": settings.router_weight_normalization,
            "corrected_ste": settings.router_straight_through,
            "vision_detector_intervals_per_232_lane": [
                list(value) for value in settings.router_detector_intervals
            ],
            "vision_detector_window_size": [
                settings.router_detector_intervals[0][1]
                - settings.router_detector_intervals[0][0],
                settings.router_detector_intervals[1][1]
                - settings.router_detector_intervals[1][0],
            ],
            "language_detector_intervals_per_478_roi": [
                list(value)
                for value in settings.language_router_detector_intervals
            ],
            "language_detector_window_size": [
                settings.language_router_detector_intervals[0][1]
                - settings.language_router_detector_intervals[0][0],
                settings.language_router_detector_intervals[1][1]
                - settings.language_router_detector_intervals[1][0],
            ],
            "phase_initialization": "deterministic_caltech_four_spot_hologram",
            "robustness": {
                "input_shift_pixels": settings.router_input_shift_pixels,
                "phase_shift_pixels": settings.router_phase_shift_pixels,
                "ccd_shift_pixels": settings.router_ccd_shift_pixels,
                "translation_wraparound": False,
                "phase_dropout": "block_phase_bypass",
                "phase_dropout_block_size": settings.router_phase_dropout_block_size,
                "energy_eps": settings.router_energy_eps,
                "score_normalization": settings.router_score_normalization,
                "capture_loss_scale": settings.router_capture_loss_scale,
            },
        },
        "prompt": settings.prompt,
        "spaq_required": False,
        "fusion": {
            "equation": "F=rE*((1-alpha)*E/rE+alpha*O/rO)/rms(mixture)",
            "per_layer_alpha_range": [
                settings.fusion_alpha_min,
                settings.fusion_alpha_max,
            ],
            "initial_alpha": settings.fusion_alpha_initial,
            "all_four_layers_have_hard_high_alpha_floor": (
                settings.fusion_alpha_min >= 0.20
            ),
        },
        "readout": {
            "type": "attention_free_dual_task_post_optical",
            "outputs": ["spatial", "temporal"],
            "width": settings.quality_head_width,
            "alignment_output": False,
        },
        "optimizer_learning_rates": {
            "electronic": settings.learning_rate,
            "feature_phase": settings.phase_learning_rate,
            "optical_router_phase": settings.optical_router_phase_learning_rate,
        },
    }
    initialization = settings.initialization_checkpoint
    if initialization is not None:
        report["initialization_checkpoint"] = {
            "path": str(initialization),
            "exists": initialization.exists(),
            "migration": (
                "exact compatible tensors; wrapped phase cos/sin resize; "
                "adapter row/column resize; alpha reset"
            ),
        }
        if not initialization.exists() and not settings.synthetic:
            report["status"] = "blocked"
            report["initialization_error"] = (
                f"Configured O2 initialization checkpoint is missing: {initialization}"
            )
    if settings.manifest_path is not None and settings.manifest_path.exists():
        counts = split_counts(read_manifest(settings.manifest_path))
        report["manifest_counts"] = counts
        report["fixed_test_count_is_558"] = counts["test"] == 558
        report["validation_count_is_zero"] = counts["validation"] == 0
        if (
            counts["train"] != 2250
            or counts["test"] != 558
            or counts["validation"] != 0
        ) and not settings.synthetic:
            report["status"] = "blocked"
            report["manifest_error"] = (
                "Formal split must contain exactly 2250 train, zero validation, "
                f"and 558 test videos; found {counts}"
            )
    elif not settings.synthetic:
        report["status"] = "blocked"
        report["manifest_error"] = f"Missing manifest: {settings.manifest_path}"
    if settings.cache_path is not None and settings.cache_path.exists():
        payload = load_canonical_cache(settings.cache_path)
        report["cache"] = cache_report(payload, settings.cache_path)
        expected_contract = settings.feature_contract
        if payload.get("feature_contract") != expected_contract:
            raise ValueError(
                f"Formal cache feature_contract must be {expected_contract!r}; "
                f"got {payload.get('feature_contract')!r}"
            )
        expected_model_id = "Qwen/Qwen3-VL-2B-Instruct"
        if payload.get("qwen_model_id") != expected_model_id:
            raise ValueError(
                "Formal cache must come from the pure Qwen3-VL-2B-Instruct "
                f"checkpoint, got {payload.get('qwen_model_id')!r}"
            )
        if payload.get("vision_main_merger_used") is not True:
            raise ValueError("Formal cache must include the learned Vision main merger")
        if payload.get("deepstack_used") is not False:
            raise ValueError("Formal cache must explicitly disable DeepStack")
        if payload.get("qwen_prompt") != settings.prompt:
            raise ValueError("Canonical cache prompt differs from the fixed formal prompt")
        if settings.manifest_path is not None and settings.manifest_path.exists():
            expected_manifest_sha = file_sha256(settings.manifest_path)
            if payload.get("manifest_sha256") != expected_manifest_sha:
                raise ValueError("Canonical cache was built from a different manifest SHA256")
        shape = payload["frame_tokens"].shape
        language_shape = payload["language_tokens"].shape
        expected = (settings.frame_count, settings.token_count, settings.input_width)
        if tuple(shape[1:]) != expected:
            raise ValueError(f"Vision cache shape {tuple(shape[1:])} != {expected}")
        if language_shape[-1] != settings.language_input_width:
            raise ValueError("Language cache width does not match model.language_input_width")
        if language_shape[1] > settings.language_token_count:
            raise ValueError("Language cache sequence exceeds configured maximum")
        cached_counts = report["cache"]["split_counts"]
        if cached_counts != {"train": 2250, "validation": 0, "test": 558}:
            raise ValueError(f"Formal cache split counts are wrong: {cached_counts}")
    elif require_cache and not settings.synthetic:
        report["status"] = "blocked"
        report["cache_error"] = f"Missing canonical cache: {settings.cache_path}"
    if settings.device.startswith("cuda"):
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
    return report


__all__ = ["run_preflight"]

