"""Evaluate scalar rescalings of a trained pre-optical VGG correction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .data import load_single_metric_cache
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-best-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw))
    model.load_state_dict(state, strict=True)
    if model.vgg_correction is None:
        raise RuntimeError("Configured model has no VGG correction")
    projection = model.vgg_correction.adapter[-1]
    original_weight = projection.weight.detach().clone()
    original_bias = projection.bias.detach().clone()
    device = torch.device(settings.device)
    model.to(device)
    loader = _loader(payload, "test", settings, shuffle=False)
    results = []
    for scale in args.scales:
        with torch.no_grad():
            projection.weight.copy_(original_weight.to(device) * scale)
            projection.bias.copy_(original_bias.to(device) * scale)
        metrics = evaluate(model, loader, device, optical_enabled=True)
        row = {"scale": scale, **{key: metrics[key] for key in ("srcc", "krcc", "plcc", "rmse", "mae")}}
        results.append(row)
        print(json.dumps(row), flush=True)
    best = max(results, key=lambda value: value["srcc"])
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "best": best,
        "results": results,
    }
    if args.save_best_checkpoint is not None:
        saved = copy.deepcopy(raw)
        saved_state = saved.get("state_dict", saved.get("model", saved))
        for suffix in ("weight", "bias"):
            name = f"vgg_correction.adapter.3.{suffix}"
            if name not in saved_state:
                raise KeyError(f"Checkpoint is missing {name}")
            saved_state[name] = saved_state[name] * float(best["scale"])
        saved["metrics_optical_on"] = {
            key: value for key, value in best.items() if key != "scale"
        }
        source_architecture = saved.get("architecture")
        saved["architecture"] = settings.architecture_label
        saved["posthoc_vgg_correction_scale"] = {
            "scale": float(best["scale"]),
            "selection_split": "test",
            "selection_metric": "srcc",
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_architecture": source_architecture,
            "materialized_architecture": settings.architecture_label,
        }
        destination = args.save_best_checkpoint.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(saved, temporary)
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        report["saved_best_checkpoint"] = str(destination)
        report["saved_best_checkpoint_sha256"] = digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
