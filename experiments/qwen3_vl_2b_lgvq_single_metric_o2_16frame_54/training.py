from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import LGVQSingleMetricDataset
from .metrics import regression_metrics
from .modeling import LGVQSingleMetricOEO16
from .phase_snapshots import save_phase_snapshot
from .settings import ExperimentSettings, resolved_dict


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _loader(
    payload: Mapping[str, Any],
    split: str,
    settings: ExperimentSettings,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        LGVQSingleMetricDataset(payload, split),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
        persistent_workers=settings.num_workers > 0,
        drop_last=False,
    )


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_difference: float = 0.05,
) -> torch.Tensor:
    prediction, target = prediction.flatten(), target.flatten()
    difference = target[:, None] - target[None, :]
    predicted = prediction[:, None] - prediction[None, :]
    valid = torch.triu(
        torch.ones_like(difference, dtype=torch.bool), diagonal=1
    ) & (difference.abs() >= minimum_difference)
    if not bool(valid.any()):
        return prediction.new_zeros(())
    return F.softplus(-difference[valid].sign() * predicted[valid]).mean()


def batch_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0e-6
) -> torch.Tensor:
    prediction, target = prediction.float().flatten(), target.float().flatten()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    target_energy = target.square().sum()
    if float(target_energy.detach()) <= epsilon:
        return prediction.new_zeros(())
    prediction_energy = prediction.square().sum()
    # Clamp before sqrt: sqrt(0) has an infinite derivative even if its output
    # is clamped afterwards, which can poison an otherwise zero-weight loss.
    denominator = (
        prediction_energy.clamp_min(epsilon) * target_energy.clamp_min(epsilon)
    ).sqrt()
    return 1.0 - ((prediction * target).sum() / denominator).clamp(-1.0, 1.0)


def soft_spearman_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Approximate batch SRCC with differentiable pairwise soft ranks.

    The target ranks are exact and constant. Prediction ranks use a sigmoid
    relaxation, so this objective changes training only and adds no inference
    module or parameter.
    """

    prediction, target = prediction.float().flatten(), target.float().flatten()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    soft_ranks = torch.sigmoid(
        (prediction[:, None] - prediction[None, :]) / temperature
    ).sum(dim=1)
    target_ranks = torch.argsort(torch.argsort(target)).to(dtype=prediction.dtype)
    return batch_correlation_loss(soft_ranks, target_ranks)


def _optimizer(
    model: nn.Module, settings: ExperimentSettings
) -> torch.optim.Optimizer:
    electronic: list[nn.Parameter] = []
    feature_phase: list[nn.Parameter] = []
    router_phase: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_router_phase" in name:
            router_phase.append(parameter)
        elif "raw_" in name and "phase" in name:
            feature_phase.append(parameter)
        else:
            electronic.append(parameter)
    groups = [
        {
            "params": electronic,
            "lr": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "name": "electronic",
        },
        {
            "params": feature_phase,
            "lr": settings.phase_learning_rate,
            "weight_decay": 0.0,
            "name": "feature_phase",
        },
        {
            "params": router_phase,
            "lr": settings.router_phase_learning_rate,
            "weight_decay": 0.0,
            "name": "router_phase",
        },
    ]
    groups = [group for group in groups if group["params"]]
    assigned = [id(value) for group in groups for value in group["params"]]
    expected = [id(value) for value in model.parameters() if value.requires_grad]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("Optimizer groups overlap or omit trainable parameters")
    return torch.optim.AdamW(groups)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(
    path: Path,
    model: LGVQSingleMetricOEO16,
    optimizer: torch.optim.Optimizer,
    settings: ExperimentSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "architecture": settings.architecture_label,
        "target_name": settings.target_name,
        "prompt": settings.prompt,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics_optical_on": dict(metrics or {}),
        "settings": resolved_dict(settings),
        "selection_policy": (
            f"highest periodically observed {settings.target_name} test SRCC; "
            "no validation split"
        ),
        "test_used_for_selection": True,
        "teacher_or_qwen_loaded_during_student_inference": False,
        "qwen_front_contract": (
            "frozen processor+vision patch/position embedding and "
            "tokenizer+text embedding cached before student training"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    model: LGVQSingleMetricOEO16,
    path: Path,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint architecture does not match this experiment")
    if saved.get("target_name") != settings.target_name:
        raise RuntimeError("Spatial and temporal checkpoints cannot be interchanged")
    model.load_state_dict(saved["state_dict"], strict=True)
    return saved


@torch.no_grad()
def evaluate(
    model: LGVQSingleMetricOEO16,
    loader: DataLoader,
    device: torch.device,
    *,
    optical_enabled: bool,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    sample_ids: list[str] = []
    video_paths: list[str] = []
    fusion_sums: dict[str, dict[str, float]] = {}
    router_sums: dict[str, dict[str, Any]] = {}
    sample_count = 0
    for batch in loader:
        result = model(
            batch["vision_tokens"].to(device, non_blocking=True),
            batch["quality_tokens"].to(device, non_blocking=True),
            batch["language_tokens"].to(device, non_blocking=True),
            batch["language_mask"].to(device, non_blocking=True),
            None
            if "raw_frames" not in batch
            else batch["raw_frames"].to(device, non_blocking=True),
            vgg_tokens=None
            if "vgg_tokens" not in batch
            else batch["vgg_tokens"].to(device, non_blocking=True),
            optical_enabled=optical_enabled,
        )
        prediction = result["prediction"]
        predictions.append(prediction.detach().cpu())
        targets.append(batch["target"].detach().cpu())
        sample_ids.extend(batch["sample_id"])
        video_paths.extend(batch["video_path"])
        count = int(prediction.shape[0])
        sample_count += count
        if optical_enabled:
            for stage, diagnostics in model.fusion_diagnostics().items():
                accumulator = fusion_sums.setdefault(stage, {})
                for name, value in diagnostics.items():
                    accumulator[name] = accumulator.get(name, 0.0) + float(value) * count
            for stage, routing in result["routing"].items():
                probability = routing["probabilities"].detach().float().reshape(-1, 4).cpu()
                selected = routing["selected_mask"].detach().float().reshape(-1, 4).cpu()
                accumulator = router_sums.setdefault(
                    stage,
                    {
                        "count": 0,
                        "probability": torch.zeros(4),
                        "selected": torch.zeros(4),
                        "capture_sum": 0.0,
                        "capture_count": 0,
                        "implementation": routing["router_implementation"],
                    },
                )
                accumulator["count"] += probability.shape[0]
                accumulator["probability"] += probability.sum(0)
                accumulator["selected"] += selected.sum(0)
                capture = routing["capture_fraction"].detach().float().reshape(-1).cpu()
                accumulator["capture_sum"] += float(capture.sum())
                accumulator["capture_count"] += capture.numel()
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = regression_metrics(prediction, target, model.settings.target_name)
    metrics["optical_enabled"] = bool(optical_enabled)
    metrics["fusion_diagnostics"] = {
        stage: {
            name: value / max(1, sample_count) for name, value in diagnostics.items()
        }
        for stage, diagnostics in fusion_sums.items()
    }
    metrics["router_diagnostics"] = {}
    for stage, values in router_sums.items():
        count = max(1, int(values["count"]))
        selected_total = max(1.0, float(values["selected"].sum()))
        metrics["router_diagnostics"][stage] = {
            "implementation": values["implementation"],
            "decision_count": int(values["count"]),
            "mean_probability": (values["probability"] / count).tolist(),
            "selected_share": (values["selected"] / selected_total).tolist(),
            "capture_fraction_mean": values["capture_sum"]
            / max(1, int(values["capture_count"])),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("sample_id", "video_path", "target_name", "target", "prediction", "absolute_error")
            )
            for index, sample_id in enumerate(sample_ids):
                writer.writerow(
                    (
                        sample_id,
                        video_paths[index],
                        model.settings.target_name,
                        float(target[index]),
                        float(prediction[index]),
                        abs(float(prediction[index]) - float(target[index])),
                    )
                )
    return metrics


@torch.no_grad()
def _phase_diagnostics(
    model: LGVQSingleMetricOEO16, initial: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    planes: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "raw_" not in name or "phase" not in name:
            continue
        start = 2.0 * math.pi * torch.sigmoid(initial[name].float())
        final = 2.0 * math.pi * torch.sigmoid(parameter.detach().cpu().float())
        difference = final - start
        wrapped = torch.atan2(torch.sin(difference), torch.cos(difference))
        planes[name] = {
            "parameters": int(parameter.numel()),
            "phase_rad_std_initial": float(start.std(unbiased=False)),
            "phase_rad_std_final": float(final.std(unbiased=False)),
            "wrapped_delta_rad_rms": float(wrapped.square().mean().sqrt()),
            "fraction_changed_over_0p05_rad": float((wrapped.abs() > 0.05).float().mean()),
        }
    return {
        "planes": planes,
        "plane_count": len(planes),
        "mean_wrapped_delta_rad_rms": sum(
            value["wrapped_delta_rad_rms"] for value in planes.values()
        )
        / max(1, len(planes)),
    }


def evaluate_checkpoint_modes(
    model: LGVQSingleMetricOEO16,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    saved = _load_checkpoint(model, checkpoint, settings)
    model.to(device)
    loader = _loader(payload, "test", settings, shuffle=False)
    optical_on = evaluate(
        model,
        loader,
        device,
        optical_enabled=True,
        prediction_path=settings.output_dir / "test_predictions_optical_on.csv",
    )
    optical_off = evaluate(
        model,
        loader,
        device,
        optical_enabled=False,
        prediction_path=settings.output_dir / "test_predictions_optical_off.csv",
    )
    delta = {
        name: float(optical_on[name]) - float(optical_off[name])
        for name in ("srcc", "krcc", "plcc", "rmse", "mae")
    }
    report = {
        "target": settings.target_name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_epoch": int(saved["epoch"]),
        "normal_optical_electronic": optical_on,
        "same_checkpoint_optics_bypassed": optical_off,
        "on_minus_off": delta,
        "separately_trained_electronic_baseline": False,
    }
    _json(settings.output_dir / "test_metrics_optical_on.json", optical_on)
    _json(settings.output_dir / "test_metrics_optical_off.json", optical_off)
    _json(settings.output_dir / "optical_contribution_same_checkpoint.json", report)
    return report


def train(
    model: LGVQSingleMetricOEO16,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(payload, "train", settings, shuffle=True)
    test_loader = _loader(payload, "test", settings, shuffle=False)
    train_indices = [
        index for index, split in enumerate(payload["splits"]) if split == "train"
    ]
    all_targets = torch.as_tensor(payload["targets"], dtype=torch.float32)
    if all_targets.ndim != 1:
        raise ValueError("A single-target run requires one scalar target per video")
    train_targets = all_targets[train_indices]
    model.set_target_statistics(
        train_targets.mean(), train_targets.std(unbiased=False).clamp_min(1.0e-6)
    )
    model.to(device)
    initial_phase = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "raw_" in name and "phase" in name
    }
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs
    )
    # Measure and preserve the exact warm-start before any optimizer update.
    # With test-driven selection requested for these experiments, epoch 0 is a
    # valid candidate and guarantees that a new readout cannot silently replace
    # a stronger source checkpoint.
    initial_metrics = evaluate(model, test_loader, device, optical_enabled=True)
    best_srcc = float(initial_metrics["srcc"])
    best_epoch = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "test_evaluated": True,
            "test_optical_on": initial_metrics,
            "warm_start_before_optimizer_update": True,
        }
    ]
    _checkpoint(
        settings.output_dir / "best_observed_test_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=0,
        metrics=initial_metrics,
    )
    _json(
        settings.output_dir / "metrics_best_observed_test_optical_on.json",
        initial_metrics,
    )
    _json(settings.output_dir / "train_history.json", history)
    print(
        f"epoch 000 warm-start {settings.target_name}_SRCC={best_srcc:.4f}",
        flush=True,
    )
    for epoch in range(1, settings.epochs + 1):
        model.train()
        totals = {
            name: 0.0
            for name in (
                "loss",
                "regression",
                "ranking",
                "correlation",
                "soft_spearman",
                "soft_target",
                "optical_alignment",
                "router_balance",
                "router_importance",
                "serial_router_balance",
                "serial_router_importance",
                "router_capture",
            )
        }
        batches = 0
        for batch in train_loader:
            vision = batch["vision_tokens"].to(device, non_blocking=True)
            quality = batch["quality_tokens"].to(device, non_blocking=True)
            language = batch["language_tokens"].to(device, non_blocking=True)
            language_mask = batch["language_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = model(
                vision,
                quality,
                language,
                language_mask,
                None
                if "raw_frames" not in batch
                else batch["raw_frames"].to(device, non_blocking=True),
                vgg_tokens=None
                if "vgg_tokens" not in batch
                else batch["vgg_tokens"].to(device, non_blocking=True),
                optical_enabled=True,
            )
            regression = F.smooth_l1_loss(
                result["normalized_prediction"], normalized_target
            )
            ranking = pairwise_ranking_loss(
                result["normalized_prediction"], normalized_target
            )
            correlation = batch_correlation_loss(
                result["normalized_prediction"], normalized_target
            )
            soft_spearman = result["normalized_prediction"].new_zeros(())
            if settings.soft_spearman_weight > 0.0:
                soft_spearman = soft_spearman_loss(
                    result["normalized_prediction"],
                    normalized_target,
                    settings.soft_rank_temperature,
                )
            soft_target = result["normalized_prediction"].new_zeros(())
            if "soft_target" in batch:
                teacher = batch["soft_target"].to(device, non_blocking=True)
                normalized_teacher = (teacher - model.target_mean) / model.target_std
                soft_target = F.smooth_l1_loss(
                    result["normalized_prediction"], normalized_teacher
                )
            language_routing = result["routing"]["language"]
            serial_router_balance = language_routing["balance_loss"]
            serial_router_importance = language_routing["importance_loss"]
            loss = (
                regression
                + settings.ranking_weight * ranking
                + settings.correlation_weight * correlation
                + settings.soft_spearman_weight * soft_spearman
                + settings.soft_target_weight * soft_target
                + settings.optical_alignment_weight * result["optical_alignment_loss"]
                + settings.router_balance_weight * result["router_balance_loss"]
                + settings.router_importance_weight * result["router_importance_loss"]
                + settings.serial_router_balance_weight * serial_router_balance
                + settings.serial_router_importance_weight * serial_router_importance
                + settings.router_capture_weight * result["router_capture_loss"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            bad = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad:
                raise RuntimeError(f"Non-finite gradients in {bad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss,
                "regression": regression,
                "ranking": ranking,
                "correlation": correlation,
                "soft_spearman": soft_spearman,
                "soft_target": soft_target,
                "optical_alignment": result["optical_alignment_loss"],
                "router_balance": result["router_balance_loss"],
                "router_importance": result["router_importance_loss"],
                "serial_router_balance": serial_router_balance,
                "serial_router_importance": serial_router_importance,
                "router_capture": result["router_capture_loss"],
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "test_evaluated": False,
        }
        if epoch == 1 or epoch % settings.test_interval_epochs == 0 or epoch == settings.epochs:
            metrics = evaluate(
                model, test_loader, device, optical_enabled=True
            )
            row["test_evaluated"] = True
            row["test_optical_on"] = metrics
            score = float(metrics["srcc"])
            if math.isfinite(score) and score > best_srcc:
                best_srcc, best_epoch = score, epoch
                _checkpoint(
                    settings.output_dir / "best_observed_test_checkpoint.pt",
                    model,
                    optimizer,
                    settings,
                    epoch=epoch,
                    metrics=metrics,
                )
                _json(
                    settings.output_dir / "metrics_best_observed_test_optical_on.json",
                    metrics,
                )
        history.append(row)
        _json(settings.output_dir / "train_history.json", history)
        if epoch % settings.phase_snapshot_interval_epochs == 0:
            save_phase_snapshot(
                model,
                settings,
                epoch=epoch,
                metrics=row.get("test_optical_on"),
            )
        if row["test_evaluated"]:
            print(
                f"epoch {epoch:03d} loss={row['loss']:.6f} "
                f"{settings.target_name}_SRCC={row['test_optical_on']['srcc']:.4f}",
                flush=True,
            )
        else:
            print(f"epoch {epoch:03d} loss={row['loss']:.6f} test=skipped", flush=True)
    _checkpoint(
        settings.output_dir / "last_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=settings.epochs,
        metrics=history[-1].get("test_optical_on"),
    )
    checkpoint = settings.output_dir / "best_observed_test_checkpoint.pt"
    comparison = evaluate_checkpoint_modes(model, payload, settings, device, checkpoint)
    phase = _phase_diagnostics(model, initial_phase)
    _json(settings.output_dir / "phase_training_diagnostics.json", phase)
    summary = {
        "target": settings.target_name,
        "prompt": settings.prompt,
        "best_epoch": best_epoch,
        "best_observed_test_srcc": best_srcc,
        "checkpoint": str(checkpoint),
        "validation_used": False,
        "test_used_for_selection": True,
        "periodic_test_interval": settings.test_interval_epochs,
        "same_checkpoint_optical_ablation": comparison,
        "phase_training_diagnostics": phase,
    }
    _json(settings.output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "batch_correlation_loss",
    "evaluate",
    "evaluate_checkpoint_modes",
    "pairwise_ranking_loss",
    "soft_spearman_loss",
    "train",
]
