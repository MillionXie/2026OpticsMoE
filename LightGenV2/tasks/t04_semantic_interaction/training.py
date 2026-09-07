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


_SEMANTIC_TOP2_CODES = {
    "add": (0, 1),
    "replace": (0, 2),
    "move": (1, 3),
    "remove": (2, 3),
}


def _semantic_router_code_loss(
    model: Any,
    tasks: list[str],
    source_image: torch.Tensor,
) -> torch.Tensor:
    """Supervise physical Router energies without changing inference.

    The standardized four-way softmax is useful for deterministic Top-2
    selection, but its normalization makes phase gradients very small when a
    single detector initially owns nearly all captured power.  Training the
    pre-normalization detector-energy fractions both matches the quantity
    measured on the CCD and supplies a usable gradient to the phase mask.
    The language Router receives balanced operation codes.  The vision Router
    receives a content code: its two target detectors are the two image
    quadrants with the highest input energy.  Both signals are training-only;
    inference receives only the same optical detector measurements.
    """
    if model.router_backend != "optical":
        return next(model.parameters()).new_zeros(())
    language_energy = model.language_core.optical_branch.core.last_routing[
        "detector_energy_fraction"
    ]
    language_target = torch.zeros_like(language_energy)
    for row, task in enumerate(tasks):
        pair = _SEMANTIC_TOP2_CODES[str(task)]
        language_target[row, pair[0]] = 0.5
        language_target[row, pair[1]] = 0.5
    language_loss = -(
        language_target * language_energy.clamp_min(1.0e-8).log()
    ).sum(dim=-1).mean()

    vision_energy = model.vision_core.optical_branch.core.last_routing[
        "detector_energy_fraction"
    ]
    image_energy = source_image.float().square().mean(dim=1)
    height_mid = image_energy.shape[-2] // 2
    width_mid = image_energy.shape[-1] // 2
    quadrant_energy = torch.stack(
        (
            image_energy[:, :height_mid, :width_mid].mean(dim=(-2, -1)),
            image_energy[:, :height_mid, width_mid:].mean(dim=(-2, -1)),
            image_energy[:, height_mid:, :width_mid].mean(dim=(-2, -1)),
            image_energy[:, height_mid:, width_mid:].mean(dim=(-2, -1)),
        ),
        dim=-1,
    )
    vision_indices = torch.topk(quadrant_energy, k=2, dim=-1).indices
    vision_target = torch.zeros_like(vision_energy).scatter(
        1, vision_indices, 0.5
    )
    vision_loss = -(
        vision_target * vision_energy.clamp_min(1.0e-8).log()
    ).sum(dim=-1).mean()
    return 0.5 * (language_loss + vision_loss)


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
                semantic_code = _semantic_router_code_loss(
                    model,
                    batch["task"],
                    batch["source_image"],
                )
                dc = phase_dc_loss(model)
                losses["total"] = (
                    losses["total"]
                    + settings.router_importance_weight * importance
                    + settings.router_hard_load_balance_weight * hard_load
                    + settings.router_semantic_code_weight * semantic_code
                    + settings.phase_dc_weight * dc
                )
                losses["router_importance"] = importance
                losses["router_hard_load"] = hard_load
                losses["router_semantic_code"] = semantic_code
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
def evaluate_selected(
    settings: Settings,
    device: torch.device,
    checkpoint: Path,
    fusion_ablation: str = "none",
) -> dict[str, Any]:
    _, loader = legacy.build_loaders(settings)
    model = build_model(settings, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != model.checkpoint_architecture:
        raise RuntimeError("T04 checkpoint architecture mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for core in (model.language_core, model.vision_core):
        core.set_fusion_ablation(fusion_ablation)
    _set_phase_dropout(model, False)
    _json(
        settings.output_dir / "student_architecture.json",
        model.architecture_report(),
    )
    router_counts: dict[str, torch.Tensor] = {}
    router_energy_sums: dict[str, torch.Tensor] = {}
    router_probability_sums: dict[str, torch.Tensor] = {}
    router_sample_counts: dict[str, int] = {}
    handles = []
    if model.router_backend == "optical" and fusion_ablation != "remove_optical":
        for label, path in zip(("language", "vision"), model._optical_paths()):
            counts = torch.zeros(4, dtype=torch.long)
            router_counts[label] = counts
            router_energy_sums[label] = torch.zeros(4, dtype=torch.float64)
            router_probability_sums[label] = torch.zeros(4, dtype=torch.float64)
            router_sample_counts[label] = 0

            def collect_selection(
                _module: Any,
                _inputs: Any,
                output: dict[str, Any],
                *,
                destination: torch.Tensor = counts,
                route_label: str = label,
            ) -> None:
                destination.add_(
                    output["selected_mask"].detach().sum(dim=0).cpu()
                )
                router_energy_sums[route_label].add_(
                    output["detector_energy_fraction"]
                    .detach()
                    .double()
                    .sum(dim=0)
                    .cpu()
                )
                router_probability_sums[route_label].add_(
                    output["probabilities"].detach().double().sum(dim=0).cpu()
                )
                router_sample_counts[route_label] += int(
                    output["selected_mask"].shape[0]
                )

            handles.append(
                path.core.router.register_forward_hook(collect_selection)
            )
    try:
        metrics, predictions, galleries = legacy._evaluate(
            model, loader, settings, device
        )
    finally:
        for handle in handles:
            handle.remove()
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "selected_epoch": int(payload["epoch"]),
        "split": "deterministic disjoint test",
        "test_samples": settings.test_samples,
        "selection_biased": True,
        "fusion_ablation": fusion_ablation,
        "metrics": metrics,
        "router_audit": (
            {
                label: {
                    **_selection_report(counts),
                    "samples": router_sample_counts[label],
                    "mean_detector_energy_fraction": [
                        float(value)
                        for value in (
                            router_energy_sums[label] / router_sample_counts[label]
                        ).tolist()
                    ],
                    "mean_router_probability": [
                        float(value)
                        for value in (
                            router_probability_sums[label] / router_sample_counts[label]
                        ).tolist()
                    ],
                }
                for label, counts in router_counts.items()
            }
            if router_counts
            else None
        ),
    }
    suffix = "" if fusion_ablation == "none" else f"_{fusion_ablation}"
    _json(
        settings.output_dir / f"selected_checkpoint_test_evaluation{suffix}.json",
        result,
    )
    with (settings.output_dir / f"test_predictions{suffix}.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    legacy._save_gallery(settings.output_dir / "best_visualization" / "test_examples", galleries, settings)
    return result


__all__ = ["evaluate_selected", "train"]
