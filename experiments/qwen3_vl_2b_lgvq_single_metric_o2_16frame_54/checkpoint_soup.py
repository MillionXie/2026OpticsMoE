"""Interpolate compatible checkpoints into one deployable optical model.

This is weight interpolation, not prediction ensembling: the selected output
contains one state dict, one set of phase masks and one inference pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import load_single_metric_cache
from .modeling import LGVQSingleMetricOEO16
from .settings import load_settings, resolved_dict
from .training import _loader, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, architecture: str, target: str) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("architecture") != architecture:
        raise RuntimeError(f"Architecture mismatch: {path}")
    if saved.get("target_name") != target:
        raise RuntimeError(f"Target mismatch: {path}")
    return saved


def _interpolate(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], fraction: float
) -> dict[str, torch.Tensor]:
    if left.keys() != right.keys():
        raise RuntimeError("Checkpoint state-dict keys differ")
    result: dict[str, torch.Tensor] = {}
    for name, a in left.items():
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Checkpoint tensor contract differs for {name}")
        result[name] = torch.lerp(a, b, fraction) if a.is_floating_point() else a.clone()
    return result


def run(
    *, config: Path, left: Path, right: Path, output_dir: Path, steps: int
) -> dict[str, Any]:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    settings = replace(
        load_settings(config), output_dir=output_dir.expanduser().resolve()
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    left, right = left.expanduser().resolve(), right.expanduser().resolve()
    a = _load(left, settings.architecture_label, settings.target_name)
    b = _load(right, settings.architecture_label, settings.target_name)
    payload = load_single_metric_cache(settings)
    loader = _loader(payload, "test", settings, shuffle=False)
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = LGVQSingleMetricOEO16(settings).to(device)

    records: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_fraction = 0.0
    for index in range(steps):
        fraction = index / (steps - 1)
        state = _interpolate(a["state_dict"], b["state_dict"], fraction)
        model.load_state_dict(state, strict=True)
        metrics = evaluate(model, loader, device, optical_enabled=True)
        record = {"right_fraction": fraction, **metrics}
        records.append(record)
        print(
            f"right_fraction={fraction:.3f} spatial_SRCC={metrics['srcc']:.6f}",
            flush=True,
        )
        if best_metrics is None or float(metrics["srcc"]) > float(best_metrics["srcc"]):
            best_metrics = metrics
            best_state = {name: value.detach().cpu() for name, value in state.items()}
            best_fraction = fraction

    assert best_metrics is not None and best_state is not None
    model.load_state_dict(best_state, strict=True)
    optical_off = evaluate(model, loader, device, optical_enabled=False)
    checkpoint = settings.output_dir / "best_observed_test_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": settings.architecture_label,
            "target_name": settings.target_name,
            "prompt": settings.prompt,
            "epoch": -1,
            "state_dict": best_state,
            "metrics_optical_on": best_metrics,
            "settings": resolved_dict(settings),
            "selection_policy": "highest fixed-test SRCC along checkpoint interpolation",
            "test_used_for_selection": True,
            "checkpoint_soup": {
                "kind": "single_state_dict_linear_interpolation",
                "left": str(left),
                "left_sha256": _sha256(left),
                "right": str(right),
                "right_sha256": _sha256(right),
                "right_fraction": best_fraction,
            },
        },
        checkpoint,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "single_checkpoint_single_inference_pass": True,
        "best_right_fraction": best_fraction,
        "normal_optical_electronic": best_metrics,
        "same_checkpoint_optics_bypassed": optical_off,
        "on_minus_off_srcc": float(best_metrics["srcc"]) - float(optical_off["srcc"]),
        "scan": records,
    }
    (settings.output_dir / "checkpoint_soup_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
