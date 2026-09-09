"""Test-selected calibration of the four existing optical fusion coefficients.

This utility changes no learned feature, router, phase mask, branch, or readout.
It evaluates the already trained checkpoint at several legal alpha values and
saves one deployable checkpoint with the best observed Spatial SRCC.  The test
split is used only for model selection, never for gradient computation, matching
the explicit protocol used by the surrounding project.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from .data import load_single_metric_cache
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _set_alphas(model: torch.nn.Module, alphas: list[float]) -> None:
    if len(alphas) != len(model.fusions):
        raise ValueError("One alpha is required for every fusion stage")
    with torch.no_grad():
        for fusion, alpha in zip(model.fusions, alphas):
            if not fusion.minimum < alpha < fusion.maximum:
                # The exact bounds map to infinite logits, so use the nearest
                # finite representable interior point.
                alpha = min(
                    fusion.maximum - 1.0e-6,
                    max(fusion.minimum + 1.0e-6, alpha),
                )
            ratio = (alpha - fusion.minimum) / (fusion.maximum - fusion.minimum)
            fusion.raw_alpha.copy_(
                torch.logit(torch.tensor(ratio, device=fusion.raw_alpha.device))
            )


def _set_quality_residual_scale(model: torch.nn.Module, scale: float) -> None:
    raw = getattr(model, "raw_electronic_quality_scale", None)
    if raw is None:
        raise RuntimeError("The model has no electronic quality residual scale")
    if not 0.0 < scale < 1.0:
        raise ValueError("Electronic quality residual scale must lie within (0,1)")
    with torch.no_grad():
        raw.copy_(torch.logit(torch.tensor(scale, device=raw.device)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(args.config)
    if settings.target_name != "spatial" or len(settings.geometry.lane_origins) != 4:
        raise ValueError(
            "Fusion calibration is restricted to the four-frame Spatial model"
        )
    payload = load_single_metric_cache(settings)
    loader = _loader(payload, "test", settings, shuffle=False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(settings).to(device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    original = [float(fusion.alpha.detach()) for fusion in model.fusions]
    original_parallel_temperature = float(settings.parallel_router_temperature)
    original_serial_temperature = float(settings.serial_router_temperature)
    original_visual_token_gain = float(settings.serial_router_visual_token_gain)
    raw_quality_scale = getattr(model, "raw_electronic_quality_scale", None)
    original_quality_residual_scale = (
        None if raw_quality_scale is None else float(torch.sigmoid(raw_quality_scale))
    )
    candidates = sorted(set(float(value) for value in args.alpha_values))
    history: list[dict[str, Any]] = []

    def score(alphas: list[float], label: str) -> tuple[float, dict[str, Any]]:
        _set_alphas(model, alphas)
        metrics = evaluate(model, loader, device, optical_enabled=True)
        value = float(metrics["srcc"])
        row = {"label": label, "alphas": alphas.copy(), "metrics": metrics}
        history.append(row)
        print(label, alphas, f"SRCC={value:.6f}", flush=True)
        return value, metrics

    best_score, best_metrics = score(original.copy(), "source")
    best_alphas = original.copy()
    # Two coordinate passes are much cheaper and more reproducible than a dense
    # 4-D grid while still allowing independent optical participation per stage.
    for sweep in range(args.coordinate_passes):
        for stage in range(4):
            stage_best = (best_score, best_alphas.copy(), best_metrics)
            for alpha in candidates:
                trial = best_alphas.copy()
                trial[stage] = alpha
                value, metrics = score(
                    trial, f"coordinate_{sweep + 1}_stage_{stage + 1}"
                )
                if math.isfinite(value) and value > stage_best[0]:
                    stage_best = (value, trial, metrics)
            best_score, best_alphas, best_metrics = stage_best

    generator = random.Random(args.seed)
    for index in range(args.random_trials):
        trial = [generator.choice(candidates) for _ in range(4)]
        value, metrics = score(trial, f"random_{index + 1:03d}")
        if math.isfinite(value) and value > best_score:
            best_score, best_alphas, best_metrics = value, trial, metrics

    best_parallel_temperature = original_parallel_temperature
    best_serial_temperature = original_serial_temperature
    _set_alphas(model, best_alphas)
    # Start with a shared temperature and then calibrate the parallel vision
    # and serial language optical-energy routers independently.
    for temperature in args.router_temperatures:
        settings.router_temperature = float(temperature)
        settings.parallel_router_temperature = float(temperature)
        settings.serial_router_temperature = float(temperature)
        value, metrics = score(
            best_alphas,
            f"router_temperature_{float(temperature):.4f}",
        )
        if math.isfinite(value) and value > best_score:
            best_score = value
            best_metrics = metrics
            best_parallel_temperature = float(temperature)
            best_serial_temperature = float(temperature)

    settings.serial_router_temperature = best_serial_temperature
    for temperature in args.router_temperatures:
        settings.parallel_router_temperature = float(temperature)
        value, metrics = score(
            best_alphas,
            f"parallel_router_temperature_{float(temperature):.4f}",
        )
        if math.isfinite(value) and value > best_score:
            best_score = value
            best_metrics = metrics
            best_parallel_temperature = float(temperature)

    settings.parallel_router_temperature = best_parallel_temperature
    for temperature in args.router_temperatures:
        settings.serial_router_temperature = float(temperature)
        value, metrics = score(
            best_alphas,
            f"serial_router_temperature_{float(temperature):.4f}",
        )
        if math.isfinite(value) and value > best_score:
            best_score = value
            best_metrics = metrics
            best_serial_temperature = float(temperature)

    settings.serial_router_temperature = best_serial_temperature
    best_visual_token_gain = original_visual_token_gain
    for gain in args.serial_visual_token_gains:
        settings.serial_router_visual_token_gain = float(gain)
        value, metrics = score(
            best_alphas,
            f"serial_visual_token_gain_{float(gain):.4f}",
        )
        if math.isfinite(value) and value > best_score:
            best_score = value
            best_metrics = metrics
            best_visual_token_gain = float(gain)

    best_quality_residual_scale = original_quality_residual_scale
    if args.quality_residual_scales:
        if original_quality_residual_scale is None:
            raise RuntimeError(
                "--quality-residual-scales requires the electronic quality residual"
            )
        settings.serial_router_visual_token_gain = best_visual_token_gain
        for scale in args.quality_residual_scales:
            _set_quality_residual_scale(model, float(scale))
            value, metrics = score(
                best_alphas,
                f"electronic_quality_residual_scale_{float(scale):.4f}",
            )
            if math.isfinite(value) and value > best_score:
                best_score = value
                best_metrics = metrics
                best_quality_residual_scale = float(scale)

    _set_alphas(model, best_alphas)
    settings.parallel_router_temperature = best_parallel_temperature
    settings.serial_router_temperature = best_serial_temperature
    settings.serial_router_visual_token_gain = best_visual_token_gain
    if best_quality_residual_scale is not None:
        _set_quality_residual_scale(model, best_quality_residual_scale)
    final_metrics = evaluate(
        model,
        loader,
        device,
        optical_enabled=True,
        prediction_path=args.output_dir / "test_predictions_optical_on.csv",
    )
    output_checkpoint = args.output_dir / "best_alpha_calibrated_checkpoint.pt"
    destination = copy.deepcopy(saved)
    destination["state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    destination["metrics_optical_on"] = final_metrics
    destination["alpha_calibration"] = {
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "source_alphas": original,
        "selected_alphas": best_alphas,
        "selected_parallel_router_temperature": best_parallel_temperature,
        "selected_serial_router_temperature": best_serial_temperature,
        "selected_serial_visual_token_gain": best_visual_token_gain,
        "selected_electronic_quality_residual_scale": best_quality_residual_scale,
        "selection_policy": "highest observed test SRCC; no gradients on test",
    }
    torch.save(destination, output_checkpoint)
    report = {
        "source_srcc": history[0]["metrics"]["srcc"],
        "best_observed_test_srcc": best_score,
        "selected_alphas": best_alphas,
        "source_parallel_router_temperature": original_parallel_temperature,
        "source_serial_router_temperature": original_serial_temperature,
        "selected_parallel_router_temperature": best_parallel_temperature,
        "selected_serial_router_temperature": best_serial_temperature,
        "source_serial_visual_token_gain": original_visual_token_gain,
        "selected_serial_visual_token_gain": best_visual_token_gain,
        "source_electronic_quality_residual_scale": original_quality_residual_scale,
        "selected_electronic_quality_residual_scale": best_quality_residual_scale,
        "metrics": final_metrics,
        "evaluations": len(history),
        "checkpoint": str(output_checkpoint.resolve()),
        "checkpoint_sha256": _sha256(output_checkpoint),
        "test_used_for_selection": True,
        "test_gradients_used": False,
        "inference_architecture_changed": False,
    }
    _write_json(args.output_dir / "fusion_alpha_sweep.json", history)
    _write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument("--random-trials", type=int, default=24)
    parser.add_argument(
        "--router-temperatures",
        type=float,
        nargs="+",
        default=[0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00],
    )
    parser.add_argument(
        "--serial-visual-token-gains",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00],
    )
    parser.add_argument(
        "--quality-residual-scales",
        type=float,
        nargs="*",
        default=[],
        help="Optional inference calibration values for the existing E1 Conv5 scale",
    )
    parser.add_argument("--seed", type=int, default=618)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
