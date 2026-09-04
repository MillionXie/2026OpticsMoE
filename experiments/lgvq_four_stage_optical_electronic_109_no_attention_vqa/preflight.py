from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from torch import nn

from .data import load_frame_cache, load_training_soft_targets, read_manifest
from .modeling import build_model
from .settings import ExperimentSettings
from .training import inspect_training_initialization


FORBIDDEN_CLASS_FRAGMENTS = ("transform", "attention", "multihead", "qwen", "mixer", "block")


def _architecture_audit(
    settings: ExperimentSettings,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    model = build_model(settings) if model is None else model
    violations = []
    module_types = []
    for module in model.modules():
        qualified = f"{module.__class__.__module__}.{module.__class__.__name__}"
        module_types.append(qualified)
        lowered = module.__class__.__name__.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_CLASS_FRAGMENTS):
            violations.append(qualified)
    source_path = Path(inspect.getsourcefile(build_model) or "").resolve()
    source = source_path.read_text(encoding="utf-8").lower()
    source_hits = [fragment for fragment in FORBIDDEN_CLASS_FRAGMENTS if fragment in source]
    front_conv_count = sum(isinstance(module, nn.Conv2d) for module in model.frame_stem.modules())
    frozen_front_parameters = sum(
        parameter.numel() for parameter in model.frame_stem.parameters() if not parameter.requires_grad
    )
    if violations or source_hits or model.frame_stem.input_channels != 14 or front_conv_count != 5 or frozen_front_parameters:
        raise RuntimeError(f"Architecture audit failed: module_types={violations}, source_hits={source_hits}")
    return {
        "status": "passed",
        "architecture_label": settings.architecture_label,
        "formal_model_source": str(source_path),
        "forbidden_class_fragments": list(FORBIDDEN_CLASS_FRAGMENTS),
        "module_type_count": len(set(module_types)),
        "external_model_package_modules": [name for name in sorted(set(module_types)) if name.startswith("transformers.")],
        "parameter_breakdown": model.parameter_breakdown(),
        "raw_frame_optical_stage1": True,
        "raw_frame_electronic_stage1": True,
        "optical_router_top_k": settings.top_k,
        "front_quality_channel_count": model.frame_stem.input_channels,
        "front_trainable_conv2d_count": front_conv_count,
        "front_pretrained_network": False,
        "front_frozen_parameter_count": frozen_front_parameters,
        "bridge_and_readout_contract": "v4stats_mean_std_max" if settings.spatial_statistics_pooling else "v1_pooled_tokens_then_frame_means",
        "spatial_statistics_pooling": settings.spatial_statistics_pooling,
        "temporal_readout_contract": "v1_frame_mean_depthwise_k3",
        "electronic_route_contract": "v1_direct_convolution_output",
    }


def run_preflight(settings: ExperimentSettings, *, require_cache: bool) -> dict[str, Any]:
    audit_model = build_model(settings)
    audit = _architecture_audit(settings, audit_model)
    problems = []
    manifest_count = None
    manifest_rows = None
    cache_shape = None
    soft_target_provenance = None
    initialization_provenance = None
    if settings.manifest_path is None or not settings.manifest_path.is_file():
        problems.append(f"manifest missing: {settings.manifest_path}")
    else:
        rows = read_manifest(settings.manifest_path)
        manifest_rows = rows
        manifest_count = len(rows)
        if not settings.synthetic and len(rows) != 2808:
            problems.append(f"formal manifest has {len(rows)} rows, expected 2808")
    if require_cache:
        if settings.frame_cache_path is None or not settings.frame_cache_path.is_file():
            problems.append(f"frame cache missing: {settings.frame_cache_path}")
        else:
            payload = load_frame_cache(settings.frame_cache_path)
            cache_shape = list(payload["frames"].shape)
            if payload["frames"].shape[-1] != settings.frame_size:
                problems.append("frame cache resolution differs from config")
            if manifest_count is not None and payload["frames"].shape[0] != manifest_count:
                problems.append("frame cache sample count differs from manifest")
    if settings.training_soft_targets_path is not None:
        if not settings.training_soft_targets_path.is_file():
            problems.append(f"training soft targets missing: {settings.training_soft_targets_path}")
        elif manifest_rows is None:
            problems.append("training soft targets cannot be aligned without a valid manifest")
        else:
            train_ids = [row.sample_id for row in manifest_rows if row.split == "train"]
            try:
                _, soft_target_provenance = load_training_soft_targets(settings.training_soft_targets_path, train_ids)
            except (RuntimeError, ValueError, OSError) as error:
                problems.append(f"training soft targets invalid: {error}")
    if settings.initialization_checkpoint is not None:
        try:
            initialization_provenance = inspect_training_initialization(audit_model, settings)
        except (FileNotFoundError, RuntimeError, ValueError, OSError, TypeError) as error:
            problems.append(f"training initialization invalid: {error}")
    return {
        "status": "ready" if not problems else "blocked",
        "problems": problems,
        "architecture_audit": audit,
        "manifest_count": manifest_count,
        "frame_cache_shape": cache_shape,
        "training_soft_target_weight": settings.soft_target_weight,
        "training_soft_target_provenance": soft_target_provenance,
        "training_initialization": initialization_provenance,
        "geometry": {
            "canvas": settings.geometry.canvas_size,
            "active": settings.geometry.active_size,
            "lane": settings.geometry.quadrant_size,
            "expert": settings.geometry.expert_size,
            "pitch": settings.geometry.expert_pitch,
        },
        "comparison_contract": "one optical-on checkpoint evaluated normally and with all optical computation bypassed",
        "separate_electronic_training": False,
    }


__all__ = ["run_preflight"]
