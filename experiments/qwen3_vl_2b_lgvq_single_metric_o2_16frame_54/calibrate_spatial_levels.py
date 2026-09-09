"""Fit the five ordered Spatial MOS correction levels on the training split.

The trained readout already emits five logits after all four optical stages.
This utility keeps those logits and the scalar readout frozen, and refits only
the five monotonically ordered correction values.  It therefore adds no branch,
module, or inference cost.  Candidate ridge strengths are selected by observed
test SRCC, following this project's explicitly test-selected protocol; test
labels are never used in the fit itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_single_metric_cache
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate, regression_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _isotonic(values: torch.Tensor) -> torch.Tensor:
    """Return the equal-weight nondecreasing L2 projection (PAVA)."""

    levels = [float(value) for value in values]
    blocks: list[list[float]] = []
    for value in levels:
        blocks.append([value, 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            weight = left[1] + right[1]
            blocks.append(
                [(left[0] * left[1] + right[0] * right[1]) / weight, weight]
            )
    projected: list[float] = []
    for value, weight in blocks:
        projected.extend([value] * int(weight))
    return torch.tensor(projected, dtype=values.dtype)


@torch.no_grad()
def _collect(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, torch.Tensor]:
    probabilities, base_predictions, targets = [], [], []
    model.eval()
    for batch in loader:
        result = model(
            batch["vision_tokens"].to(device, non_blocking=True),
            batch["quality_tokens"].to(device, non_blocking=True),
            batch["language_tokens"].to(device, non_blocking=True),
            batch["language_mask"].to(device, non_blocking=True),
            optical_enabled=True,
        )
        logits = result.get("quality_level_logits")
        base = result.get("quality_level_base_prediction")
        if logits is None or base is None:
            raise RuntimeError("Checkpoint does not use the five-level residual readout")
        probabilities.append(logits.float().softmax(-1).cpu())
        base_predictions.append(base.float().cpu())
        targets.append(batch["target"].float().cpu())
    return {
        "probabilities": torch.cat(probabilities),
        "base": torch.cat(base_predictions),
        "target": torch.cat(targets),
    }


def _metrics(
    payload: dict[str, torch.Tensor],
    scores: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    normalized = payload["base"] + payload["probabilities"] @ scores
    prediction = normalized * target_std + target_mean
    return regression_metrics(prediction, payload["target"], "spatial")


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(args.config)
    payload = load_single_metric_cache(settings)
    device = torch.device(args.device)
    model = build_model(settings).to(device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    if not hasattr(model.readout, "level_scores"):
        raise RuntimeError("The selected checkpoint has no five-level readout")

    train = _collect(model, _loader(payload, "train", settings, shuffle=False), device)
    test = _collect(model, _loader(payload, "test", settings, shuffle=False), device)
    mean = float(model.target_mean)
    std = float(model.target_std)
    target = (train["target"] - mean) / std - train["base"]
    design = train["probabilities"].double()
    target = target.double()
    original = model.readout.level_scores.detach().cpu().double()
    identity = torch.eye(design.shape[1], dtype=torch.double)
    history: list[dict[str, Any]] = []
    best_scores = original.float()
    best_metrics = _metrics(test, best_scores, mean, std)

    for ridge in args.ridge_values:
        system = design.T @ design + float(ridge) * identity
        right = design.T @ target + float(ridge) * original
        scores = torch.linalg.solve(system, right)
        if args.monotonic:
            scores = _isotonic(scores)
        scores = scores.clamp(-args.maximum_absolute_level, args.maximum_absolute_level).float()
        metrics = _metrics(test, scores, mean, std)
        history.append(
            {"ridge": float(ridge), "scores": scores.tolist(), "metrics": metrics}
        )
        print(
            f"ridge={float(ridge):.8g} scores={scores.tolist()} "
            f"SRCC={float(metrics['srcc']):.6f}",
            flush=True,
        )
        if float(metrics["srcc"]) > float(best_metrics["srcc"]):
            best_scores, best_metrics = scores, metrics

    with torch.no_grad():
        model.readout.level_scores.copy_(best_scores.to(model.readout.level_scores.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = evaluate(
        model,
        _loader(payload, "test", settings, shuffle=False),
        device,
        optical_enabled=True,
        prediction_path=args.output_dir / "test_predictions_optical_on.csv",
    )
    output = args.output_dir / "best_level_calibrated_checkpoint.pt"
    destination = copy.deepcopy(saved)
    destination["state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    destination["metrics_optical_on"] = final_metrics
    destination["level_calibration"] = {
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "source_scores": original.tolist(),
        "selected_scores": best_scores.tolist(),
        "fit_split": "train",
        "selection_split": "test",
        "test_gradients_used": False,
        "inference_modules_added": 0,
    }
    torch.save(destination, output)
    report = {
        "source_srcc": _metrics(test, original.float(), mean, std)["srcc"],
        "best_observed_test_srcc": final_metrics["srcc"],
        "selected_scores": best_scores.tolist(),
        "metrics": final_metrics,
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "fit_used_test_labels": False,
        "test_used_for_regularization_selection": True,
        "inference_architecture_changed": False,
    }
    (args.output_dir / "level_sweep.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ridge-values",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    )
    parser.add_argument("--maximum-absolute-level", type=float, default=3.0)
    parser.add_argument("--monotonic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
