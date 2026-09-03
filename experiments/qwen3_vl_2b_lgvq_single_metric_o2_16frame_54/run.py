from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import load_single_metric_cache
from .modeling import LGVQSingleMetricOEO16, build_model
from .preflight import run_preflight
from .settings import ExperimentSettings, Geometry, load_settings, resolved_dict
from .training import evaluate_checkpoint_modes, train


PHASES = {"preflight", "train", "evaluate", "smoke"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(settings: ExperimentSettings) -> torch.device:
    if settings.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(settings.device)


def _load_compatible_initialization(
    model: LGVQSingleMetricOEO16, settings: ExperimentSettings
) -> dict[str, Any]:
    path = settings.initialization_checkpoint
    if path is None:
        return {"used": False, "reason": "not configured"}
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint is missing: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    source = raw.get("state_dict", raw.get("model", raw))
    if not isinstance(source, dict):
        raise ValueError("Initialization checkpoint has no state_dict mapping")
    destination = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in destination
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(destination[name].shape)
    }
    if not compatible:
        raise RuntimeError("Initialization checkpoint has no shape-compatible tensors")
    result = model.load_state_dict(compatible, strict=False)
    return {
        "used": True,
        "path": str(path),
        "loaded_tensors": len(compatible),
        "loaded_parameters": sum(value.numel() for value in compatible.values()),
        "missing_tensors": len(result.missing_keys),
        "unexpected_tensors": list(result.unexpected_keys),
        "policy": "exact-name and exact-shape only; no silent resizing",
    }


def synthetic_smoke(settings: ExperimentSettings) -> dict[str, Any]:
    if settings.frame_count == 4:
        geometry = Geometry(
            canvas_size=96,
            active_size=88,
            lane_grid=2,
            lane_size=40,
            lane_pitch=48,
            lane_offset=0,
            parallel_expert_size=18,
            parallel_expert_pitch=22,
            serial_expert_size=40,
            serial_expert_pitch=20,
        )
        parallel_intervals = ((8, 16), (24, 32))
    else:
        geometry = Geometry(
            canvas_size=96,
            active_size=88,
            lane_grid=4,
            lane_size=18,
            lane_pitch=22,
            lane_offset=2,
            parallel_expert_size=8,
            parallel_expert_pitch=10,
            serial_expert_size=40,
            serial_expert_pitch=20,
        )
        parallel_intervals = ((4, 8), (10, 14))
    small = replace(
        settings,
        geometry=geometry,
        maximum_language_tokens=32,
        detector_projection_size=16,
        parallel_router_intervals=parallel_intervals,
        serial_router_intervals=((20, 30), (50, 60)),
        input_shift_pixels=1,
        phase_shift_pixels=1,
        ccd_shift_pixels=1,
        phase_dropout_cell_size=2,
        k_space_enabled=False,
        batch_size=2,
        num_workers=0,
        initialization_checkpoint=None,
        synthetic=True,
        device="cpu",
    )
    small.validate()
    model = build_model(small).train()
    generator = torch.Generator().manual_seed(small.random_seed)
    vision = torch.randn(2, small.frame_count, 49, 1024, generator=generator)
    quality = torch.randn(2, small.frame_count, 49, 14, generator=generator)
    language = torch.randn(2, 12, 2048, generator=generator)
    mask = torch.ones(2, 12, dtype=torch.bool)
    result = model(vision, quality, language, mask, optical_enabled=True)
    loss = (
        result["normalized_prediction"].square().mean()
        + 0.01 * result["optical_alignment_loss"]
        + 0.01 * result["router_balance_loss"]
    )
    loss.backward()
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    required_gradients = (
        "vision_adapter.1.weight",
        "language_adapter.1.weight",
        "prompt_to_visual.1.weight",
        "parallel_router.raw_router_phase",
        "parallel_optics.raw_expert_phase",
        "serial_router.raw_router_phase",
        "serial_optics.raw_global_phase",
    )
    missing_gradients = [
        name
        for name in required_gradients
        if dict(model.named_parameters())[name].grad is None
    ]
    attention = [
        module.__class__.__name__
        for module in model.modules()
        if "attention" in module.__class__.__name__.lower()
        or "transformer" in module.__class__.__name__.lower()
    ]
    if bad or missing_gradients or attention or result["prediction"].shape != (2,):
        raise RuntimeError(
            f"Smoke failed: bad={bad}, missing_gradients={missing_gradients}, "
            f"forbidden_modules={attention}, prediction={tuple(result['prediction'].shape)}"
        )
    return {
        "status": "passed",
        "target": small.target_name,
        "prediction_shape": list(result["prediction"].shape),
        "quality_gate": float(result["quality_gate"].detach()),
        "vision_router_shape": list(result["routing"]["vision"]["weights"].shape),
        "language_router_shape": list(result["routing"]["language"]["weights"].shape),
        "all_required_gradients_present": True,
        "forbidden_attention_or_transformer_modules": attention,
        "parameters": model.parameter_breakdown(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Text-conditioned 4/16-frame LGVQ optical-electronic model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config, synthetic=args.phase == "smoke")
    _seed(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _json(settings.output_dir / "resolved_config.json", resolved_dict(settings))
    if args.phase == "smoke":
        report = synthetic_smoke(settings)
        _json(settings.output_dir / "synthetic_smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    preflight = run_preflight(settings, require_cache=args.phase != "preflight")
    _json(settings.output_dir / "preflight.json", preflight)
    if args.phase == "preflight":
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight["status"] == "ready" else 2
    if preflight["status"] != "ready":
        raise RuntimeError(f"Preflight is blocked: {preflight}")

    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    initialization = _load_compatible_initialization(model, settings)
    _json(settings.output_dir / "initialization_report.json", initialization)
    _json(settings.output_dir / "parameter_breakdown.json", model.parameter_breakdown())
    device = _device(settings)
    if args.phase == "train":
        report = train(model, payload, settings, device)
    else:
        if args.checkpoint is None:
            parser.error("--phase evaluate requires --checkpoint")
        report = evaluate_checkpoint_modes(
            model,
            payload,
            settings,
            device,
            Path(args.checkpoint).expanduser().resolve(),
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


__all__ = ["main", "synthetic_smoke"]
