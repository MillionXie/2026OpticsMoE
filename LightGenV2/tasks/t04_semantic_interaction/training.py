"""OpenMoji trainer with periodic-test selection and compact artifacts."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing import (
    training as legacy,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import (
    editing_objective,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.training import (
    EMA,
    seed_everything,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .modeling import build_model, initialize_from_legacy
from .settings import Settings
from .visualize import render


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    ema: EMA | None = None,
) -> None:
    value = {
        "schema_version": 1,
        "architecture": model.checkpoint_architecture,
        "epoch": int(epoch),
        "model": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "selection_biased": True,
    }
    if optimizer is not None:
        value["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        value["scheduler"] = scheduler.state_dict()
    if ema is not None:
        value["ema_model"] = ema.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def _set_phase_dropout(model: Any, enabled: bool) -> None:
    for path in model._optical_paths():
        path.set_phase_dropout_active(enabled)


def train(settings: Settings, device: torch.device) -> dict[str, Any]:
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = legacy.build_loaders(settings)
    model = build_model(settings, device)
    initialization = initialize_from_legacy(model, settings)
    _json(settings.output_dir / "initialization_report.json", initialization)
    _json(settings.output_dir / "student_architecture.json", model.architecture_report())
    optim = AdamW(legacy._parameter_groups(model, settings), weight_decay=settings.weight_decay)
    scheduler = LambdaLR(
        optim,
        legacy._schedule(settings.epochs * len(train_loader)),
    )
    ema = EMA(model, settings.ema_decay)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.set_phase_trainable(True)
        _set_phase_dropout(model, True)
        model.train()
        model.vision_stem.eval()
        totals: dict[str, float] = defaultdict(float)
        count = 0
        epoch_started = time.perf_counter()
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = legacy._move(raw, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=settings.amp_enabled and device.type == "cuda",
            ):
                outputs = model(batch["source_image"], batch["prompt_hidden"])
                losses = editing_objective(outputs, batch, settings)
                importance = model.router_importance_loss()
                hard_load = model.router_hard_load_balance_loss()
                dc = phase_dc_loss(model)
                losses["total"] = (
                    losses["total"]
                    + settings.router_importance_weight * importance
                    + settings.router_hard_load_balance_weight * hard_load
                    + settings.phase_dc_weight * dc
                )
                losses["router_importance"] = importance
                losses["router_hard_load"] = hard_load
                losses["phase_dc"] = dc
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                settings.gradient_clip_norm,
            )
            optim.step()
            scheduler.step()
            ema.update(model)
            batch_count = len(batch["task"])
            count += batch_count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_count
            totals["grad_norm"] += float(grad_norm) * batch_count
            if batch_index % settings.log_interval == 0 or batch_index == len(train_loader):
                print(
                    f"[T04] epoch={epoch:03d}/{settings.epochs:03d} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"loss={totals['total']/count:.5f}",
                    flush=True,
                )
        _set_phase_dropout(model, False)
        train_metrics = {name: value / count for name, value in totals.items()}
        train_metrics["samples"] = count
        train_metrics["epoch_seconds"] = time.perf_counter() - epoch_started
        scheduled_test = (
            epoch == 1
            or epoch % settings.test_interval_epochs == 0
            or epoch == settings.epochs
        )
        test_metrics = None
        if scheduled_test:
            backup = ema.copy_to(model)
            model.eval()
            try:
                test_metrics, _, _ = legacy._evaluate(model, test_loader, settings, device)
                score = float(test_metrics["overall"]["changed_cell_accuracy"])
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    _checkpoint(
                        settings.output_dir / "best_checkpoint.pt",
                        model,
                        epoch,
                        train_metrics,
                        test_metrics,
                    )
            finally:
                EMA.restore(model, backup)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **(
                {
                    f"test_{key}": value
                    for key, value in test_metrics["overall"].items()
                }
                if test_metrics is not None
                else {}
            ),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        _csv(settings.output_dir / "metrics" / "training_history.csv", history)
        _checkpoint(
            settings.output_dir / "last_checkpoint.pt",
            model,
            epoch,
            train_metrics,
            test_metrics,
            optimizer=optim,
            scheduler=scheduler,
            ema=ema,
        )
        suffix = "" if test_metrics is None else f" test_changed={test_metrics['overall']['changed_cell_accuracy']:.4f}"
        print(
            f"[T04] epoch={epoch:03d}/{settings.epochs:03d}{suffix} "
            f"best={best_score:.4f}@{best_epoch}",
            flush=True,
        )
    render(settings.output_dir / "best_checkpoint.pt", settings.output_dir / "best_visualization")
    report = {
        "elapsed_seconds": time.perf_counter() - started,
        "selected_epoch": best_epoch,
        "selected_test_changed_cell_accuracy": best_score,
        "selection": "maximum periodic-test changed-cell accuracy",
        "checkpoint_retention": ["best_checkpoint.pt", "last_checkpoint.pt"],
    }
    _json(settings.output_dir / "training_report.json", report)
    return report


@torch.inference_mode()
def evaluate_selected(settings: Settings, device: torch.device, checkpoint: Path) -> dict[str, Any]:
    _, loader = legacy.build_loaders(settings)
    model = build_model(settings, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != model.checkpoint_architecture:
        raise RuntimeError("T04 checkpoint architecture mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    _set_phase_dropout(model, False)
    metrics, predictions, galleries = legacy._evaluate(model, loader, settings, device)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "selected_epoch": int(payload["epoch"]),
        "split": "deterministic disjoint test",
        "test_samples": settings.test_samples,
        "selection_biased": True,
        "metrics": metrics,
    }
    _json(settings.output_dir / "selected_checkpoint_test_evaluation.json", result)
    with (settings.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    legacy._save_gallery(settings.output_dir / "best_visualization" / "test_examples", galleries, settings)
    return result


__all__ = ["evaluate_selected", "train"]
