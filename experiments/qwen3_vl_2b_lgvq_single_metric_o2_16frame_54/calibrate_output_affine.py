"""Fit a train-only positive affine calibration and fold it into a checkpoint.

The calibration changes only the final score unit and offset.  A positive
slope preserves every sample ordering, so SRCC/KRCC and the optical inference
graph are unchanged.  Test labels are never used by this utility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import torch

from .data import load_single_metric_cache
from .metrics import regression_metrics
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_checkpoint = args.output_checkpoint.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    settings = load_settings(config)
    device = torch.device(settings.device)
    payload = load_single_metric_cache(settings)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint architecture does not match the config")
    if saved.get("target_name") != settings.target_name:
        raise RuntimeError("Checkpoint target does not match the config")

    model = build_model(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device)
    prediction_csv = report_path.with_name(report_path.stem + "_train_predictions.csv")
    loader = _loader(payload, "train", settings, shuffle=False)
    pre_metrics = evaluate(
        model,
        loader,
        device,
        optical_enabled=True,
        prediction_path=prediction_csv,
    )

    predictions: list[float] = []
    targets: list[float] = []
    with prediction_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            predictions.append(float(row["prediction"]))
            targets.append(float(row["target"]))
    prediction = torch.tensor(predictions, dtype=torch.float64)
    target = torch.tensor(targets, dtype=torch.float64)
    centered_prediction = prediction - prediction.mean()
    denominator = centered_prediction.square().sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise RuntimeError("Train predictions have zero or non-finite variance")
    slope = float((centered_prediction * (target - target.mean())).sum() / denominator)
    if not (slope > 0.0):
        raise RuntimeError(f"Positive affine calibration required; fitted slope={slope}")
    intercept = float(target.mean() - slope * prediction.mean())
    calibrated_prediction = prediction * slope + intercept
    post_metrics = regression_metrics(
        calibrated_prediction.float(), target.float(), settings.target_name
    )

    calibrated = copy.deepcopy(saved)
    state = calibrated["state_dict"]
    old_mean = torch.as_tensor(state["target_mean"]).detach().clone()
    old_std = torch.as_tensor(state["target_std"]).detach().clone()
    state["target_mean"] = old_mean * slope + intercept
    state["target_std"] = old_std * slope
    calibrated["train_only_positive_affine_calibration"] = {
        "schema_version": 1,
        "selection_split": "train",
        "test_labels_used": False,
        "optical_enabled_during_fit": True,
        "slope": slope,
        "intercept": intercept,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "old_target_mean": float(old_mean),
        "old_target_std": float(old_std),
        "new_target_mean": float(state["target_mean"]),
        "new_target_std": float(state["target_std"]),
        "ordering_preserved": True,
    }
    calibrated.pop("metrics_optical_on", None)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_suffix(output_checkpoint.suffix + ".tmp")
    torch.save(calibrated, temporary)
    temporary.replace(output_checkpoint)

    report = {
        "schema_version": 1,
        "config": str(config),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": _sha256(output_checkpoint),
        "fit_split": "train",
        "test_labels_used": False,
        "sample_count": len(predictions),
        "slope": slope,
        "intercept": intercept,
        "train_metrics_before": {
            key: pre_metrics[key]
            for key in ("srcc", "krcc", "plcc", "rmse", "mae")
        },
        "train_metrics_after": {
            key: post_metrics[key]
            for key in ("srcc", "krcc", "plcc", "rmse", "mae")
        },
        "prediction_csv": str(prediction_csv),
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
