from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.nn import functional as F

from .settings import Settings
from .visualization import save_comparison_plots


METHODS = ("bp", "fa_pretrained", "fa_random")


def _flatten_delta(parameters: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            (parameters[name].detach().float().cpu() - initial[name].detach().float().cpu()).reshape(-1)
            for name in sorted(initial)
        ]
    )


def _history_order_hashes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["batch_order_sha256"] for row in csv.DictReader(handle)]


def _aggregate(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def compare_methods(settings: Settings) -> Path:
    initial_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    if not initial_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint is missing: {initial_path}")
    initial_checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
    initial_parameters = initial_checkpoint["parameters"]
    per_seed: list[dict[str, Any]] = []
    bp_deltas: dict[int, torch.Tensor] = {}
    checkpoints: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in settings.training.finetune_seeds:
        reference_hashes: list[str] | None = None
        for method in METHODS:
            run_dir = settings.output_dir / "finetune" / method / f"seed_{seed}"
            checkpoint_path = run_dir / "checkpoints" / "last.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Matched endpoint checkpoint is missing: {checkpoint_path}")
            hashes = _history_order_hashes(run_dir / "training_history.csv")
            if reference_hashes is None:
                reference_hashes = hashes
            elif hashes != reference_hashes:
                raise RuntimeError(f"Batch-order mismatch for seed {seed}: {method} differs from BP")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["initial_parameter_digest"] != initial_checkpoint["parameter_digest"]:
                raise RuntimeError(f"Initialization mismatch for {method}, seed {seed}")
            checkpoints[(method, seed)] = checkpoint
            delta = _flatten_delta(checkpoint["parameters"], initial_parameters)
            if method == "bp":
                bp_deltas[seed] = delta
    for seed in settings.training.finetune_seeds:
        bp_delta = bp_deltas[seed]
        for method in METHODS:
            checkpoint = checkpoints[(method, seed)]
            delta = _flatten_delta(checkpoint["parameters"], initial_parameters)
            initial_flat = torch.cat(
                [initial_parameters[name].detach().float().reshape(-1) for name in sorted(initial_parameters)]
            )
            cosine = float(F.cosine_similarity(delta, bp_delta, dim=0, eps=1e-12))
            metrics = checkpoint["metrics"]
            per_seed.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_accuracy": float(metrics["test_accuracy"]),
                    "validation_accuracy": float(metrics["validation_accuracy"]),
                    "relative_parameter_drift": float(delta.norm() / initial_flat.norm().clamp_min(1e-12)),
                    "endpoint_update_norm": float(delta.norm()),
                    "endpoint_cosine_to_bp": cosine,
                    "phase_circular_rms_rad": float(metrics["phase_circular_rms_rad"]),
                    "phase_phasor_drift": float(metrics["phase_phasor_drift"]),
                    "phase_operator_coherence": float(metrics["phase_operator_coherence"]),
                }
            )
    noft_path = settings.output_dir / "finetune" / "no_finetune" / "summary.json"
    if not noft_path.exists():
        raise FileNotFoundError(f"No-fine-tuning result is missing: {noft_path}")
    noft = json.loads(noft_path.read_text(encoding="utf-8"))
    per_seed.append(
        {
            "method": "no_finetune",
            "seed": None,
            "test_accuracy": float(noft["test_accuracy"]),
            "validation_accuracy": float("nan"),
            "relative_parameter_drift": 0.0,
            "endpoint_update_norm": 0.0,
            "endpoint_cosine_to_bp": 0.0,
            "phase_circular_rms_rad": 0.0,
            "phase_phasor_drift": 0.0,
            "phase_operator_coherence": 1.0,
        }
    )
    aggregate: list[dict[str, Any]] = []
    for method in (*METHODS, "no_finetune"):
        rows = [row for row in per_seed if row["method"] == method]
        test_mean, test_std = _aggregate([float(row["test_accuracy"]) for row in rows])
        drift_mean, drift_std = _aggregate([float(row["relative_parameter_drift"]) for row in rows])
        cosine_mean, cosine_std = _aggregate([float(row["endpoint_cosine_to_bp"]) for row in rows])
        aggregate.append(
            {
                "method": method,
                "test_accuracy_mean": test_mean,
                "test_accuracy_std": test_std,
                "relative_parameter_drift_mean": drift_mean,
                "relative_parameter_drift_std": drift_std,
                "endpoint_cosine_to_bp_mean": cosine_mean,
                "endpoint_cosine_to_bp_std": cosine_std,
            }
        )
    comparison_dir = settings.output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    with (comparison_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    output = comparison_dir / "comparison.json"
    output.write_text(
        json.dumps(
            {
                "pretrained_checkpoint": str(initial_path),
                "settings_digest": settings.digest(),
                "control_check": "batch-order hashes match BP for every method and seed",
                "per_seed": per_seed,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_comparison_plots(aggregate, comparison_dir)
    print(json.dumps(aggregate, indent=2), flush=True)
    return output
