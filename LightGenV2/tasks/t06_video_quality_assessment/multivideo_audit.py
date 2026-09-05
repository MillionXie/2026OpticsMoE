"""Audit physical-slot invariance and optical contribution for MultiVideo-9x4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import (
    load_single_metric_cache,
)
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import (
    regression_metrics,
)

from .models.multivideo9x4 import build_model
from .multivideo_settings import load_settings
from .multivideo_training import evaluate


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: (float(row["target"]), float(row["prediction"]))
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint.parent / "slot_cycle_audit"
    )
    output.mkdir(parents=True, exist_ok=True)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device(settings.device)
    model.to(device)
    payload = load_single_metric_cache(settings)

    cycles: list[dict[str, object]] = []
    predictions: list[dict[str, tuple[float, float]]] = []
    for shift in range(settings.videos_per_field):
        prediction_path = output / f"predictions_shift_{shift}.csv"
        metrics = evaluate(
            model,
            payload,
            settings,
            device,
            optical_enabled=True,
            prediction_path=prediction_path,
            slot_shift=shift,
        )
        predictions.append(_load_predictions(prediction_path))
        cycles.append(
            {
                "shift": shift,
                "srcc": metrics["srcc"],
                "krcc": metrics["krcc"],
                "plcc": metrics["plcc"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "router_diagnostics": metrics.get("router_diagnostics", {}),
            }
        )

    sample_ids = sorted(predictions[0])
    target = torch.tensor([predictions[0][key][0] for key in sample_ids])
    cycle_prediction = torch.tensor(
        [[values[key][1] for key in sample_ids] for values in predictions]
    )
    ensemble = regression_metrics(
        cycle_prediction.mean(0), target, settings.target_name
    )
    per_video_slot_std = cycle_prediction.std(0, unbiased=False)
    optical_off = evaluate(
        model, payload, settings, device, optical_enabled=False
    )
    srcc = torch.tensor([float(row["srcc"]) for row in cycles])
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved.get("epoch"),
        "architecture": saved.get("architecture"),
        "contract": "nine unrelated videos x four frames; one MOS per video",
        "cycles": cycles,
        "cycle_srcc": {
            "mean": float(srcc.mean()),
            "minimum": float(srcc.min()),
            "maximum": float(srcc.max()),
        },
        "prediction_slot_std": {
            "mean_mos": float(per_video_slot_std.mean()),
            "p95_mos": float(torch.quantile(per_video_slot_std, 0.95)),
        },
        "nine_cycle_ensemble": ensemble,
        "optical_off_without_retraining": optical_off,
    }
    _json(output / "slot_cycle_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
