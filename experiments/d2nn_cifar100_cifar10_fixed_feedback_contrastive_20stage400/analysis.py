from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.nn import functional as F

from .settings import Settings


METHODS = ("bp", "fa_pretrained", "fa_random")


def _flatten_delta(parameters: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            (parameters[name].detach().float().cpu() - initial[name].detach().float().cpu()).reshape(-1)
            for name in sorted(initial)
        ]
    )


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def compare_methods(settings: Settings) -> Path:
    initial_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    initial_checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
    initial_parameters = initial_checkpoint["parameters"]
    initial_flat = torch.cat([initial_parameters[name].float().reshape(-1) for name in sorted(initial_parameters)])
    task_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in settings.training.finetune_seeds:
            summary_path = settings.output_dir / "finetune" / method / f"seed_{seed}" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for policy in ("fixed_endpoint", "validation_selected"):
                task_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "policy": policy,
                        "selected_epoch": summary[policy]["epoch"],
                        "test_accuracy": summary[policy]["test_accuracy"],
                    }
                )
    noft = json.loads(
        (settings.output_dir / "finetune" / "no_finetune" / "summary.json").read_text(encoding="utf-8")
    )
    task_rows.append(
        {
            "method": "no_finetune",
            "seed": "",
            "policy": "fixed_model",
            "selected_epoch": 0,
            "test_accuracy": noft["test_accuracy"],
        }
    )

    geometry_rows: list[dict[str, Any]] = []
    matched_epochs = sorted(set(settings.training.diagnostic_epochs) & set(range(1, settings.training.finetune_epochs + 1)))
    matched_epochs = [epoch for epoch in matched_epochs if epoch % settings.training.checkpoint_interval_epochs == 0]
    if settings.training.finetune_epochs not in matched_epochs:
        matched_epochs.append(settings.training.finetune_epochs)
    for epoch in matched_epochs:
        for seed in settings.training.finetune_seeds:
            deltas: dict[str, torch.Tensor] = {}
            for method in METHODS:
                run_dir = settings.output_dir / "finetune" / method / f"seed_{seed}" / "checkpoints"
                path = run_dir / ("last.pt" if epoch == settings.training.finetune_epochs else f"epoch_{epoch:03d}.pt")
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint["initial_parameter_digest"] != initial_checkpoint["parameter_digest"]:
                    raise RuntimeError(f"Initialization mismatch for {method}, seed {seed}")
                deltas[method] = _flatten_delta(checkpoint["parameters"], initial_parameters)
            bp_norm = deltas["bp"].norm().clamp_min(1e-12)
            for method in METHODS:
                delta = deltas[method]
                geometry_rows.append(
                    {
                        "matched_epoch": epoch,
                        "method": method,
                        "seed": seed,
                        "relative_parameter_drift": float(delta.norm() / initial_flat.norm().clamp_min(1e-12)),
                        "drift_ratio_to_bp": float(delta.norm() / bp_norm),
                        "endpoint_cosine_to_bp": float(
                            F.cosine_similarity(delta, deltas["bp"], dim=0, eps=1e-12).clamp(-1.0, 1.0)
                        ),
                    }
                )

    aggregate: list[dict[str, Any]] = []
    for policy in ("fixed_endpoint", "validation_selected"):
        for method in METHODS:
            rows = [row for row in task_rows if row["method"] == method and row["policy"] == policy]
            center, spread = _summary([float(row["test_accuracy"]) for row in rows])
            aggregate.append(
                {
                    "policy": policy,
                    "method": method,
                    "test_accuracy_mean": center,
                    "test_accuracy_sample_sd": spread,
                    "selected_epoch_mean": mean(float(row["selected_epoch"]) for row in rows),
                }
            )
    aggregate.append(
        {
            "policy": "fixed_model",
            "method": "no_finetune",
            "test_accuracy_mean": noft["test_accuracy"],
            "test_accuracy_sample_sd": float("nan"),
            "selected_epoch_mean": 0.0,
        }
    )
    output_dir = settings.output_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("task_metrics.csv", task_rows), ("endpoint_geometry.csv", geometry_rows), ("aggregate.csv", aggregate)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output = output_dir / "comparison.json"
    output.write_text(
        json.dumps(
            {
                "task_definition": "CIFAR-100 SupCon pretraining -> actual CIFAR-10 prototype transfer",
                "test_used_for_checkpoint_selection": False,
                "task_rows": task_rows,
                "geometry_rows": geometry_rows,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return output
