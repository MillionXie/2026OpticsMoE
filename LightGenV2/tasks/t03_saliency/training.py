"""Periodic-test SALICON trainer retaining only selected-best and last."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import (
    training as legacy,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.visualization import (
    save_examples,
)

from .modeling import build_student, initialize_student, optimizer
from .visualize import render


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(
    path: Path,
    model: Any,
    epoch: int,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "architecture": model.checkpoint_architecture,
            "epoch": int(epoch),
            "core": model.core.state_dict(),
            "saliency_head": model.head.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "selection_biased": True,
        },
        path,
    )


def train(loaded: Any, bundle: Any, settings: Any) -> dict[str, Any]:
    model = build_student(loaded, settings)
    initialization = initialize_student(model, settings)
    _write_json(settings.output_dir / "initialization_report.json", initialization)
    train_loader, test_loader = legacy.build_loaders(bundle, settings, training=True)
    optim = optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, int(settings.student_epochs))
    )
    history: list[dict[str, Any]] = []
    best_cc = -math.inf
    best_epoch = -1
    started = time.perf_counter()
    try:
        for epoch in range(1, int(settings.student_epochs) + 1):
            model.core.set_phase_dropout_active(True)
            train_metrics = legacy._train_epoch(
                "student", model, train_loader, loaded, settings, optim
            )
            model.core.set_phase_dropout_active(False)
            scheduled_test = (
                epoch == 1
                or epoch % int(settings.test_interval_epochs) == 0
                or epoch == int(settings.student_epochs)
            )
            test_metrics = None
            if scheduled_test:
                test_metrics, _ = legacy.evaluate_model(
                    model, test_loader, loaded, settings
                )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **(
                    {f"test_{key}": value for key, value in test_metrics.items()}
                    if test_metrics is not None
                    else {}
                ),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(row)
            _write_csv(settings.output_dir / "metrics" / "training_history.csv", history)
            _checkpoint(
                settings.output_dir / "last_checkpoint.pt",
                model,
                epoch,
                train_metrics,
                test_metrics,
            )
            if test_metrics is not None and float(test_metrics["cc"]) > best_cc:
                best_cc = float(test_metrics["cc"])
                best_epoch = epoch
                _checkpoint(
                    settings.output_dir / "best_checkpoint.pt",
                    model,
                    epoch,
                    train_metrics,
                    test_metrics,
                )
            scheduler.step()
            suffix = "" if test_metrics is None else f" test_CC={test_metrics['cc']:.4f}"
            print(
                f"[T03] epoch={epoch:03d}/{settings.student_epochs:03d} "
                f"train_loss={train_metrics['loss']:.5f}{suffix} best_CC={best_cc:.4f}",
                flush=True,
            )
    finally:
        model.core.set_phase_dropout_active(False)
        model.restore_native()
    render(settings.output_dir / "best_checkpoint.pt", settings.output_dir / "best_visualization")
    report = {
        "elapsed_seconds": time.perf_counter() - started,
        "selected_epoch": best_epoch,
        "selected_periodic_test_cc": best_cc,
        "selection": "maximum public-test CC at epoch 1, every 5 epochs, and final",
        "checkpoint_retention": ["best_checkpoint.pt", "last_checkpoint.pt"],
    }
    _write_json(settings.output_dir / "training_report.json", report)
    return report


@torch.inference_mode()
def evaluate_selected_checkpoint(
    loaded: Any, bundle: Any, settings: Any, checkpoint: Path
) -> dict[str, Any]:
    model = build_student(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != model.checkpoint_architecture:
        raise RuntimeError("T03 checkpoint architecture mismatch")
    model.core.load_state_dict(payload["core"], strict=True)
    model.head.load_state_dict(payload["saliency_head"], strict=True)
    _, loader = legacy.build_loaders(bundle, settings, training=False)
    try:
        model.core.set_phase_dropout_active(False)
        metrics, examples = legacy.evaluate_model(
            model,
            loader,
            loaded,
            settings,
            collect_examples=settings.visualization_sample_count,
        )
        result = {
            "system": settings.lightgen_model_variant,
            "split": "SALICON official val2014 used as public test/selection",
            "test_samples": len(bundle.validation_records),
            "selected_epoch": int(payload["epoch"]),
            "checkpoint": str(checkpoint),
            "selection_biased": True,
            "metrics": metrics,
        }
        _write_json(settings.output_dir / "selected_checkpoint_test_evaluation.json", result)
        save_examples(
            settings.output_dir / "best_visualization" / "saliency_examples",
            examples,
            kind="student",
        )
        return result
    finally:
        model.restore_native()


__all__ = ["evaluate_selected_checkpoint", "train"]
