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


def _selection_report(counts: torch.Tensor) -> dict[str, Any]:
    total = float(counts.sum())
    share = counts.float() / max(total, 1.0)
    return {
        "selected_slots": [int(value) for value in counts.tolist()],
        "selection_share": [float(value) for value in share.tolist()],
        "effective_experts_inverse_simpson": float(
            1.0 / share.square().sum().clamp_min(1.0e-12)
        ),
        "unused_experts": [
            index for index, value in enumerate(counts.tolist()) if value == 0
        ],
    }


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
            "fusion_contract": {"minimum": model.core.hybrid.fusion_alpha_min,
                                "maximum": model.core.hybrid.fusion_alpha_max},
            "test_metrics": test_metrics,
            "selection_biased": True,
        },
        path,
    )


def staged_epoch(optim: Any, settings: Any, epoch: int) -> dict[str, Any]:
    """Closed-form LR schedule, without recursively compounding multipliers."""
    warmup = settings.staged_warmup_epochs
    if epoch <= warmup:
        stage, factor = "optics_readout_adaptation", 1.0
    elif epoch < settings.staged_polish_start:
        stage = "joint"
        progress = (epoch - warmup - 1) / max(1, settings.staged_polish_start - warmup - 1)
        factor = 0.2 + 0.8 * (1 + math.cos(math.pi * progress)) / 2
    else:
        stage = "polish"
        progress = (epoch - settings.staged_polish_start) / max(1, settings.student_epochs - settings.staged_polish_start)
        factor = 0.02 + 0.08 * (1 + math.cos(math.pi * progress)) / 2
    for group in optim.param_groups:
        group.setdefault("schedule_base_lr", group["lr"])
        group["lr"] = group["schedule_base_lr"] * (0.0 if stage == "optics_readout_adaptation" and group["name"] == "electronic" else factor)
    progress = max(0., min(1., (epoch - warmup) / max(1, settings.staged_polish_start - warmup)))
    hard = settings.router_hard_load_balance_weight * (1-progress) + settings.staged_final_hard_balance * progress
    return {"stage": stage, "hard_balance_weight": hard,
            **{f"lr_{g['name']}": g["lr"] for g in optim.param_groups}}


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
        if getattr(settings, "initialization_checkpoint", None) is not None:
            model.core.set_phase_dropout_active(False)
            initial_metrics, _ = legacy.evaluate_model(model, test_loader, loaded, settings)
            best_cc, best_epoch = float(initial_metrics["cc"]), 0
            _checkpoint(settings.output_dir / "best_checkpoint.pt", model, 0, {}, initial_metrics)
            _write_json(settings.output_dir / "warmstart_evaluation.json", initial_metrics)
            print(f"[T03] warmstart CC={best_cc:.6f}", flush=True)
        for epoch in range(1, int(settings.student_epochs) + 1):
            stage_report = {}
            if settings.staged_training:
                stage_report = staged_epoch(optim, settings, epoch)
                model._router_hard_weight = stage_report["hard_balance_weight"]
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
                "learning_rate": optim.param_groups[0]["lr"],
                **stage_report,
                "alpha1": float(model.core.hybrid.block1_optical_fusion.detach()),
                "alpha2": float(model.core.hybrid.block2_optical_fusion.detach()),
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
            if not settings.staged_training:
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
    router_counts = torch.zeros(4, dtype=torch.long)
    router_energy_sum = torch.zeros(4, dtype=torch.float64)
    router_probability_sum = torch.zeros(4, dtype=torch.float64)
    router_samples = 0
    handle = None
    if settings.lightgen_model_variant != "d2nn_active_expert_matched":
        router = model.core.optical_branch.core.router

        def collect_selection(
            _module: Any, _inputs: Any, output: dict[str, Any]
        ) -> None:
            nonlocal router_samples
            router_counts.add_(
                output["selected_mask"].detach().sum(dim=0).cpu()
            )
            router_energy_sum.add_(
                output["detector_energy_fraction"]
                .detach()
                .double()
                .sum(dim=0)
                .cpu()
            )
            router_probability_sum.add_(
                output["probabilities"].detach().double().sum(dim=0).cpu()
            )
            router_samples += int(output["selected_mask"].shape[0])

        handle = router.register_forward_hook(collect_selection)
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
            "alpha": [float(model.core.hybrid.block1_optical_fusion.detach()),
                      float(model.core.hybrid.block2_optical_fusion.detach())],
            "ccd_normalization": settings.ccd_normalization,
            "phase_change_from_initialization": phase_change_report(payload, settings),
            "router_audit": None if handle is None else {
                **_selection_report(router_counts),
                "samples": router_samples,
                "mean_detector_energy_fraction": [
                    float(value) for value in (router_energy_sum / router_samples).tolist()
                ],
                "mean_router_probability": [
                    float(value)
                    for value in (router_probability_sum / router_samples).tolist()
                ],
            },
        }
        _write_json(settings.output_dir / "selected_checkpoint_test_evaluation.json", result)
        save_examples(
            settings.output_dir / "best_visualization" / "saliency_examples",
            examples,
            kind="student",
        )
        return result
    finally:
        if handle is not None:
            handle.remove()
        model.restore_native()


__all__ = ["evaluate_selected_checkpoint", "train"]


def phase_change_report(payload: dict, settings: Any) -> dict:
    """Report real sigmoid-phase motion, not merely raw parameter updates."""
    source = getattr(settings, "initialization_checkpoint", None)
    if source is None:
        return {}
    previous = torch.load(source, map_location="cpu", weights_only=False)["core"]
    result = {}
    for name, value in payload["core"].items():
        if not any(key in name for key in ("raw_phase", "raw_router_phase")) or name not in previous:
            continue
        before, after = previous[name].float(), value.float()
        radians = 2 * math.pi * (after.sigmoid() - before.sigmoid())
        circular = torch.atan2(radians.sin(), radians.cos())
        result[name] = {"elements": value.numel(),
                        "raw_rms_change": float((after-before).square().mean().sqrt()),
                        "circular_phase_rms_rad": float(circular.square().mean().sqrt()),
                        "fraction_above_001_rad": float((circular.abs() > .01).float().mean())}
    return result
