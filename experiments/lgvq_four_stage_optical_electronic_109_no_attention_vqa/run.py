from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import attach_training_soft_targets, load_frame_cache
from .modeling import build_model
from .preflight import run_preflight
from .settings import ExperimentSettings, OpticalGeometry, load_settings, resolved_dict
from .training import (
    _optimizer,
    apply_training_initialization,
    evaluate_checkpoint_modes,
    pairwise_ranking_loss,
    train,
)


PHASES = {"preflight", "train", "evaluate", "smoke"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(settings: ExperimentSettings) -> torch.device:
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(settings.device)


def synthetic_smoke(settings: ExperimentSettings) -> dict[str, Any]:
    variant = replace(
        settings,
        output_dir=settings.output_dir / "smoke",
        geometry=OpticalGeometry(canvas_size=96, active_size=88, quadrant_size=40, expert_size=16, expert_pitch=24),
        frame_size=64,
        token_grid=4,
        width=192,
        bridge_pool=1,
        parallel_detector_intervals=((6, 12), (28, 34)),
        serial_detector_intervals=((20, 30), (58, 68)),
        input_shift_pixels=2,
        phase_shift_pixels=2,
        ccd_shift_pixels=2,
        detector_projection_size=16,
        head_width=32,
        batch_size=2,
        num_workers=0,
        k_space_enabled=False,
        synthetic=True,
        device="cpu",
    )
    variant.validate()
    model = build_model(variant)
    frames = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8)
    targets = torch.tensor([[40.0, 45.0], [60.0, 55.0]])
    model.set_target_statistics(targets.mean(0), targets.std(0, unbiased=False).clamp_min(1.0))
    optimizer = _optimizer(model, variant)
    model.train()
    result = model(frames, optical_enabled=True)
    normalized_target = (targets - model.target_mean) / model.target_std
    loss = torch.nn.functional.smooth_l1_loss(result["normalized_prediction"], normalized_target)
    loss = loss + 0.1 * pairwise_ranking_loss(result["normalized_prediction"], normalized_target)
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Smoke loss is non-finite")
    phase_gradients = {
        name: float(parameter.grad.detach().abs().mean())
        for name, parameter in model.named_parameters()
        if "raw_" in name and "phase" in name and parameter.grad is not None
    }
    expected_phase_names = {
        name for name, _ in model.named_parameters() if "raw_" in name and "phase" in name
    }
    if set(phase_gradients) != expected_phase_names or any(not math.isfinite(value) or value <= 0.0 for value in phase_gradients.values()):
        raise RuntimeError(f"Optical phase gradient contract failed: {phase_gradients}")
    if result["prediction"].shape != (2, 2):
        raise RuntimeError("Output must be [B,2]")
    for name, routing in result["routing"].items():
        if not bool((routing["selected_mask"].sum(-1) == 2).all()):
            raise RuntimeError(f"{name} did not select exactly two optical experts")
    model.eval()
    off = model(frames, optical_enabled=False)
    if off["routing"] or not bool(torch.isfinite(off["prediction"]).all()):
        raise RuntimeError("Optical-off inference contract failed")
    return {
        "status": "passed",
        "loss": float(loss.detach()),
        "optical_on_shape": list(result["prediction"].shape),
        "optical_off_shape": list(off["prediction"].shape),
        "same_model_instance": True,
        "optical_on_router_names": sorted(result["routing"]),
        "optical_off_router_count": len(off["routing"]),
        "phase_gradient_mean_abs": phase_gradients,
        "parameter_breakdown": model.parameter_breakdown(),
        "optimizer_groups": {
            str(group["name"]): {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LGVQ four-stage optical-electronic quality model")
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
    if args.phase == "preflight":
        report = run_preflight(settings, require_cache=False)
        _json(settings.output_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "ready" else 2
    preflight = run_preflight(settings, require_cache=True)
    _json(settings.output_dir / "preflight.json", preflight)
    if preflight["status"] != "ready":
        raise RuntimeError(f"Preflight blocked: {preflight['problems']}")
    assert settings.frame_cache_path is not None
    payload = load_frame_cache(settings.frame_cache_path)
    if args.phase == "train" and settings.training_soft_targets_path is not None:
        payload = attach_training_soft_targets(payload, settings.training_soft_targets_path)
    model = build_model(settings)
    _json(settings.output_dir / "parameter_breakdown.json", model.parameter_breakdown())
    device = _device(settings)
    if args.phase == "train":
        initialization = apply_training_initialization(model, settings)
        _json(settings.output_dir / "training_initialization.json", initialization)
        report = train(
            model,
            payload,
            settings,
            device,
            initialization_provenance=initialization,
        )
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        report = evaluate_checkpoint_modes(model, payload, settings, device, Path(args.checkpoint).expanduser().resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "synthetic_smoke"]
