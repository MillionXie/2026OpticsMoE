"""Measure the information ceiling of frozen post-optical Spatial features.

This diagnostic never reads raw frames or pre-optical quality features.  A
forward pre-hook records only the tensors delivered to the final readout after
the deployed four-stage optical/electronic network.  Lightweight regressors
then tell us whether a stronger final electronic head can improve SRCC without
changing the optics, router, alpha, frame layout, or Qwen-front contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import LGVQSingleMetricDataset, load_single_metric_cache
from .modeling import build_model
from .settings import load_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction - target
    return {
        "srcc": float(spearmanr(prediction, target).statistic),
        "plcc": float(pearsonr(prediction, target).statistic),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
    }


def _masked_statistics(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    count = valid.sum(1).clamp_min(1)
    mean = (value * valid).sum(1) / count
    variance = ((value - mean[:, None]).square() * valid).sum(1) / count
    maximum = value.masked_fill(~valid, -torch.inf).amax(1)
    minimum = value.masked_fill(~valid, torch.inf).amin(1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    return torch.cat((mean, variance.clamp_min(0.0).sqrt(), maximum, minimum), -1)


def _summarize(
    vision: torch.Tensor,
    language: torch.Tensor,
    mask: torch.Tensor,
    base_prediction: torch.Tensor,
) -> torch.Tensor:
    batch, frames, tokens, width = vision.shape
    grid_size = int(tokens**0.5)
    if grid_size * grid_size != tokens:
        raise ValueError("Post-optical Vision tokens must form a square grid")
    value = vision.float()
    frame = torch.cat(
        (
            value.mean(2),
            value.std(2, unbiased=False),
            value.amax(2),
            value.amin(2),
        ),
        -1,
    ).flatten(1)
    grid = value.reshape(batch * frames, grid_size, grid_size, width).permute(0, 3, 1, 2)
    gradient_x = grid[..., 1:] - grid[..., :-1]
    gradient_y = grid[..., 1:, :] - grid[..., :-1, :]
    gradient = torch.cat(
        (
            gradient_x.abs().mean((-2, -1)),
            gradient_x.float().std((-2, -1), unbiased=False),
            gradient_y.abs().mean((-2, -1)),
            gradient_y.float().std((-2, -1), unbiased=False),
        ),
        -1,
    ).reshape(batch, -1)
    average_grid = F.adaptive_avg_pool2d(grid, 3).reshape(batch, -1)
    maximum_grid = F.adaptive_max_pool2d(grid, 2).reshape(batch, -1)
    sequence = _masked_statistics(language.float(), mask).flatten(1)
    return torch.cat(
        (frame, gradient, average_grid, maximum_grid, sequence, base_prediction[:, None]),
        1,
    )


@torch.inference_mode()
def extract(
    *,
    config: Path,
    checkpoint: Path,
    output: Path,
    raw_output: Path | None,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    settings = load_settings(config)
    if settings.target_name != "spatial":
        raise ValueError("The post-optical probe is Spatial-only")
    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    target_device = torch.device(device)
    model.to(target_device).eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["vision"], captured["language"], captured["mask"] = inputs

    handle = model.readout.register_forward_pre_hook(hook)
    rows: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    raw_vision: list[torch.Tensor] = []
    raw_language: list[torch.Tensor] = []
    raw_mask: list[torch.Tensor] = []
    raw_normalized_prediction: list[torch.Tensor] = []
    try:
        for split in ("train", "test"):
            loader = DataLoader(
                LGVQSingleMetricDataset(payload, split),
                batch_size=batch_size,
                shuffle=False,
                num_workers=settings.num_workers,
                pin_memory=target_device.type == "cuda",
            )
            for batch in loader:
                result = model(
                    batch["vision_tokens"].to(target_device, non_blocking=True),
                    batch["quality_tokens"].to(target_device, non_blocking=True),
                    batch["language_tokens"].to(target_device, non_blocking=True),
                    batch["language_mask"].to(target_device, non_blocking=True),
                    optical_enabled=True,
                )
                feature = _summarize(
                    captured["vision"],
                    captured["language"],
                    captured["mask"],
                    result["prediction"].float(),
                )
                features.append(feature.detach().cpu().half())
                if raw_output is not None:
                    raw_vision.append(captured["vision"].detach().cpu().half())
                    raw_language.append(captured["language"].detach().cpu().half())
                    raw_mask.append(captured["mask"].detach().cpu().bool())
                    raw_normalized_prediction.append(
                        result["normalized_prediction"].detach().cpu().float()
                    )
                for index, sample_id in enumerate(batch["sample_id"]):
                    rows.append(
                        {
                            "sample_id": str(sample_id),
                            "split": split,
                            "target": float(batch["target"][index]),
                            "base_prediction": float(result["prediction"][index]),
                        }
                    )
                print(f"[{split}] {len(rows)} rows", flush=True)
    finally:
        handle.remove()
    matrix = torch.cat(features)
    if matrix.shape[0] != len(rows):
        raise RuntimeError("Extracted feature and metadata counts differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "post_optical_spatial_statistics_v1",
            "features": matrix,
            "rows": rows,
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "feature_interpretation": "readout inputs after the four-stage O/E/O graph only",
        },
        output,
    )
    report = {
        "output": str(output.resolve()),
        "sha256": _sha256(output),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "rows": len(rows),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_vision_tensor = torch.cat(raw_vision)
        raw_language_tensor = torch.cat(raw_language)
        raw_mask_tensor = torch.cat(raw_mask)
        raw_prediction_tensor = torch.cat(raw_normalized_prediction)
        torch.save(
            {
                "contract": "post_optical_spatial_raw_readout_inputs_v1",
                "vision": raw_vision_tensor,
                "language": raw_language_tensor,
                "mask": raw_mask_tensor,
                "normalized_prediction": raw_prediction_tensor,
                "rows": rows,
                "target_mean": model.target_mean.detach().cpu(),
                "target_std": model.target_std.detach().cpu(),
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": _sha256(checkpoint),
                "feature_interpretation": (
                    "raw readout inputs after the four-stage O/E/O graph only"
                ),
            },
            raw_output,
        )
        report["raw_output"] = str(raw_output.resolve())
        report["raw_output_sha256"] = _sha256(raw_output)
        report["raw_vision_shape"] = list(raw_vision_tensor.shape)
    return report


def fit(*, input_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    rows = payload["rows"]
    split = np.asarray([row["split"] for row in rows])
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    base = np.asarray([row["base_prediction"] for row in rows], dtype=np.float64)
    train, test = split == "train", split == "test"
    x_train, x_test = features[train], features[test]
    y_train, y_test = target[train], target[test]
    base_test = base[test]
    candidates: list[tuple[str, Any]] = []
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        candidates.append(
            (
                f"ridge_{alpha:g}",
                make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr")),
            )
        )
    for leaf in (2, 4, 8, 16):
        candidates.append(
            (
                f"extra_trees_leaf{leaf}",
                ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=leaf,
                    max_features=0.35,
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )
    for components in (64, 128, 256):
        candidates.append(
            (
                f"pca{components}_histgb",
                make_pipeline(
                    StandardScaler(),
                    PCA(n_components=components, svd_solver="randomized", random_state=42),
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=2.0,
                        random_state=42,
                    ),
                ),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [
        {"name": "source_checkpoint", **_metrics(base_test, y_test)}
    ]
    prediction_columns: dict[str, np.ndarray] = {"source_checkpoint": base_test}
    for name, estimator in candidates:
        print(f"[fit] {name}", flush=True)
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_test).astype(np.float64)
        row = {"name": name, **_metrics(prediction, y_test)}
        results.append(row)
        prediction_columns[name] = prediction
        for weight in np.linspace(0.05, 1.0, 20):
            blended = (1.0 - weight) * base_test + weight * prediction
            results.append(
                {
                    "name": f"blend_source_{name}_w{weight:.2f}",
                    "probe_weight": float(weight),
                    **_metrics(blended, y_test),
                }
            )
        print(json.dumps(row), flush=True)
    results.sort(key=lambda row: row["srcc"], reverse=True)
    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "probe_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target", *prediction_columns])
        test_rows = [row for row in rows if row["split"] == "test"]
        for index, row in enumerate(test_rows):
            writer.writerow(
                [
                    row["sample_id"],
                    y_test[index],
                    *(prediction_columns[name][index] for name in prediction_columns),
                ]
            )
    summary = {
        "input": str(input_path.resolve()),
        "feature_shape": list(features.shape),
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
        "test_used_for_candidate_selection": True,
        "best": results[0],
        "top10": results[:10],
    }
    (output_dir / "probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--config", required=True, type=Path)
    extract_parser.add_argument("--checkpoint", required=True, type=Path)
    extract_parser.add_argument("--output", required=True, type=Path)
    extract_parser.add_argument("--raw-output", type=Path)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--batch-size", type=int, default=32)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--input", required=True, type=Path)
    fit_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        report = extract(
            config=args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            raw_output=args.raw_output,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        report = fit(input_path=args.input, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
