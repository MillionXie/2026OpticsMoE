from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import LGVQFeatureDataset
from .metrics import regression_metrics
from .modeling import LGVQSpatiotemporalModel
from .settings import ExperimentSettings, resolved_dict


def _write_json(path: Path, value: Any) -> None:
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
        LGVQFeatureDataset(payload, split),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
        drop_last=False,
    )


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_difference: float = 0.05,
) -> torch.Tensor:
    losses = []
    for column in range(2):
        difference = target[:, None, column] - target[None, :, column]
        predicted_difference = (
            prediction[:, None, column] - prediction[None, :, column]
        )
        upper = torch.triu(
            torch.ones_like(difference, dtype=torch.bool), diagonal=1
        )
        valid = upper & (difference.abs() >= minimum_difference)
        if bool(valid.any()):
            sign = difference[valid].sign()
            losses.append(F.softplus(-sign * predicted_difference[valid]).mean())
    return torch.stack(losses).mean() if losses else prediction.new_zeros(())


def batch_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Differentiable batch-wise Pearson loss for spatial and temporal MOS.

    A target column with no within-batch variation contains no ranking signal
    and is omitted.  A constant prediction is deliberately kept: the clamped
    denominator leaves a useful gradient that can move it away from collapse.
    """

    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Correlation loss inputs must have matching [B,C] shapes")
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    predicted = prediction.float() - prediction.float().mean(0, keepdim=True)
    truth = target.float() - target.float().mean(0, keepdim=True)
    predicted_energy = predicted.square().sum(0)
    target_energy = truth.square().sum(0)
    denominator = (predicted_energy * target_energy).sqrt().clamp_min(epsilon)
    correlation = (predicted * truth).sum(0) / denominator
    valid = target_energy > epsilon
    if not bool(valid.any()):
        return prediction.new_zeros(())
    return (1.0 - correlation[valid].clamp(-1.0, 1.0)).mean()


@torch.no_grad()
def evaluate(
    model: LGVQSpatiotemporalModel,
    loader: DataLoader,
    device: torch.device,
    *,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions, targets, sample_ids, video_paths = [], [], [], []
    fusion_sums: dict[str, dict[str, float]] = {}
    fusion_samples = 0
    router_sums: dict[str, dict[str, Any]] = {}
    for batch in loader:
        output = model(
            batch["features"].to(device, non_blocking=True),
            batch["language_tokens"].to(device, non_blocking=True),
            batch["language_mask"].to(device, non_blocking=True),
        )
        batch_size = int(output["prediction"].shape[0])
        predictions.append(output["prediction"].detach().cpu())
        targets.append(batch["target"].detach().cpu())
        sample_ids.extend(batch["sample_id"])
        video_paths.extend(batch["video_path"])
        fusion_samples += batch_size
        for stage, values in model.fusion_diagnostics().items():
            accumulator = fusion_sums.setdefault(stage, {})
            for key, value in values.items():
                accumulator[key] = accumulator.get(key, 0.0) + float(value) * batch_size
        for branch, routing in output["routing"].items():
            probabilities = routing["probabilities"].detach().float().reshape(-1, 4).cpu()
            selected = routing["selected_mask"].detach().float().reshape(-1, 4).cpu()
            count = int(probabilities.shape[0])
            accumulator = router_sums.setdefault(
                branch,
                {
                    "decision_count": 0,
                    "probability_sum": torch.zeros(4),
                    "selected_sum": torch.zeros(4),
                    "entropy_sum": 0.0,
                    "capture_sum": 0.0,
                    "capture_count": 0,
                    "implementation": str(routing["router_implementation"]),
                },
            )
            accumulator["decision_count"] += count
            accumulator["probability_sum"] += probabilities.sum(0)
            accumulator["selected_sum"] += selected.sum(0)
            entropy = -(
                probabilities.clamp_min(1.0e-8).log() * probabilities
            ).sum(-1) / math.log(4.0)
            accumulator["entropy_sum"] += float(entropy.sum())
            if "capture_fraction" in routing:
                capture = routing["capture_fraction"].detach().float().reshape(-1).cpu()
                accumulator["capture_sum"] += float(capture.sum())
                accumulator["capture_count"] += int(capture.numel())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = regression_metrics(prediction, target)
    metrics["fusion_diagnostics"] = {
        stage: {
            key: value / max(1, fusion_samples)
            for key, value in values.items()
        }
        for stage, values in fusion_sums.items()
    }
    metrics["router_diagnostics"] = {}
    for branch, values in router_sums.items():
        count = max(1, int(values["decision_count"]))
        selected_total = float(values["selected_sum"].sum())
        metrics["router_diagnostics"][branch] = {
            "implementation": values["implementation"],
            "decision_count": int(values["decision_count"]),
            "mean_probability": (values["probability_sum"] / count).tolist(),
            "selected_fraction_per_decision": (values["selected_sum"] / count).tolist(),
            "selected_share_among_active": (
                values["selected_sum"] / max(1.0, selected_total)
            ).tolist(),
            "normalized_entropy": float(values["entropy_sum"]) / count,
            "capture_fraction_mean": (
                float(values["capture_sum"]) / int(values["capture_count"])
                if int(values["capture_count"]) > 0
                else None
            ),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "video_path",
                    "spatial_target",
                    "spatial_prediction",
                    "temporal_target",
                    "temporal_prediction",
                ]
            )
            for index, sample_id in enumerate(sample_ids):
                writer.writerow(
                    [
                        sample_id,
                        video_paths[index],
                        float(target[index, 0]),
                        float(prediction[index, 0]),
                        float(target[index, 1]),
                        float(prediction[index, 1]),
                    ]
                )
    return metrics


def _optimizer(model: nn.Module, settings: ExperimentSettings) -> torch.optim.Optimizer:
    phase, router, router_phase, base = [], [], [], []
    router_prefixes = ("vision_router.", "language_router.")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(router_prefixes):
            if "raw_router_phase" in name:
                router_phase.append(parameter)
            else:
                router.append(parameter)
        elif "raw_" in name and "phase" in name:
            phase.append(parameter)
        else:
            base.append(parameter)
    groups = [{"params": base, "lr": settings.learning_rate, "name": "electronic"}]
    if router:
        groups.append(
            {"params": router, "lr": settings.router_learning_rate, "name": "router"}
        )
    if router_phase:
        groups.append(
            {
                "params": router_phase,
                "lr": settings.optical_router_phase_learning_rate,
                "name": "optical_router_phase",
            }
        )
    if phase:
        groups.append(
            {"params": phase, "lr": settings.phase_learning_rate, "name": "phase"}
        )
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected or sum(len(group["params"]) for group in groups) != len(assigned):
        raise RuntimeError("Optimizer parameter groups overlap or omit trainable tensors")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _checkpoint(
    path: Path,
    model: LGVQSpatiotemporalModel,
    optimizer: torch.optim.Optimizer,
    settings: ExperimentSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
    initialization: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "architecture": settings.architecture_label,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": dict(metrics or {}),
        "settings": resolved_dict(settings),
        "initialization": dict(initialization),
        "selection_policy": (
            "highest periodically observed test mean(SRCC_spatial,SRCC_temporal); "
            "test leakage explicitly accepted; no validation split"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    model: LGVQSpatiotemporalModel,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(payload, "train", settings, shuffle=True)
    test_loader = _loader(payload, "test", settings, shuffle=False)
    train_indices = [
        index for index, split in enumerate(payload["splits"]) if split == "train"
    ]
    train_targets = payload["targets"][train_indices].float()
    model.set_target_statistics(
        train_targets.mean(0), train_targets.std(0, unbiased=False).clamp_min(1.0e-6)
    )
    model.to(device)
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs
    )
    best_score = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "ranking": 0.0,
            "correlation": 0.0,
            "optical_alignment": 0.0,
            "router_balance": 0.0,
            "router_importance": 0.0,
            "router_capture": 0.0,
            "batches": 0,
        }
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            language = batch["language_tokens"].to(device, non_blocking=True)
            language_mask = batch["language_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = model(features, language, language_mask)
            regression = F.smooth_l1_loss(
                result["normalized_prediction"], normalized_target
            )
            ranking = pairwise_ranking_loss(
                result["normalized_prediction"], normalized_target
            )
            correlation = batch_correlation_loss(
                result["normalized_prediction"], normalized_target
            )
            optical_alignment = result["optical_alignment_loss"]
            loss = (
                regression
                + settings.ranking_loss_weight * ranking
                + settings.correlation_loss_weight * correlation
                + settings.optical_alignment_loss_weight * optical_alignment
                + settings.router_balance_weight * result["router_balance_loss"]
                + settings.router_importance_weight
                * result["router_importance_loss"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite LGVQ training loss")
            loss.backward()
            bad = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad:
                raise RuntimeError(f"Non-finite gradients: {bad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["regression"] += float(regression.detach())
            totals["ranking"] += float(ranking.detach())
            totals["correlation"] += float(correlation.detach())
            totals["optical_alignment"] += float(optical_alignment.detach())
            totals["router_balance"] += float(result["router_balance_loss"].detach())
            totals["router_importance"] += float(
                result["router_importance_loss"].detach()
            )
            totals["router_capture"] += float(result["router_capture_loss"].detach())
            totals["batches"] += 1
        scheduler.step()
        count = max(1, totals.pop("batches"))
        row: dict[str, Any] = {
            "epoch": epoch,
            **{key: value / count for key, value in totals.items()},
            "test_evaluated": False,
        }
        should_test = (
            epoch == 1
            or epoch % settings.test_interval_epochs == 0
            or epoch == settings.epochs
        )
        if should_test:
            test_metrics = evaluate(model, test_loader, device)
            row["test_evaluated"] = True
            row["test"] = test_metrics
            score = float(test_metrics["selection_mean_srcc"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                _checkpoint(
                    output / "best_observed_test_checkpoint.pt",
                    model,
                    optimizer,
                    settings,
                    epoch=epoch,
                    metrics=test_metrics,
                    initialization=initialization,
                )
                _write_json(output / "metrics_best_observed_test.json", test_metrics)
        history.append(row)
        _write_json(output / "train_history.json", history)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.6f} "
            f"corr={row['correlation']:.5f} "
            f"opt_align={row['optical_alignment']:.5f} "
            f"router_capture={row['router_capture']:.5f} "
            + (
                "test "
                f"spatial[SRCC={row['test']['spatial']['srcc']:.4f} "
                f"KRCC={row['test']['spatial']['krcc']:.4f} "
                f"PLCC={row['test']['spatial']['plcc']:.4f} "
                f"RMSE={row['test']['spatial']['rmse']:.4f} "
                f"MAE={row['test']['spatial']['mae']:.4f}] "
                f"temporal[SRCC={row['test']['temporal']['srcc']:.4f} "
                f"KRCC={row['test']['temporal']['krcc']:.4f} "
                f"PLCC={row['test']['temporal']['plcc']:.4f} "
                f"RMSE={row['test']['temporal']['rmse']:.4f} "
                f"MAE={row['test']['temporal']['mae']:.4f}] "
                f"mean_SRCC={row['test']['selection_mean_srcc']:.4f}"
                if row["test_evaluated"]
                else "test=skipped"
            ),
            flush=True,
        )
    _checkpoint(
        output / "last_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=settings.epochs,
        metrics=history[-1].get("test"),
        initialization=initialization,
    )
    report = {
        "best_epoch": best_epoch,
        "best_test_mean_srcc": best_score,
        "test_interval_epochs": settings.test_interval_epochs,
        "validation_used": False,
        "test_used_for_selection": True,
        "checkpoint": str(output / "best_observed_test_checkpoint.pt"),
    }
    _write_json(output / "training_summary.json", report)
    return report


def evaluate_checkpoint(
    model: LGVQSpatiotemporalModel,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: {saved.get('architecture')!r} != "
            f"{settings.architecture_label!r}"
        )
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device)
    metrics = evaluate(
        model,
        _loader(payload, "test", settings, shuffle=False),
        device,
        prediction_path=settings.output_dir / "test_predictions.csv",
    )
    _write_json(settings.output_dir / "test_metrics.json", metrics)
    _write_json(
        settings.output_dir / "fusion_diagnostics.json",
        metrics["fusion_diagnostics"],
    )
    _write_json(
        settings.output_dir / "router_diagnostics.json",
        metrics["router_diagnostics"],
    )
    return metrics


__all__ = [
    "batch_correlation_loss",
    "evaluate",
    "evaluate_checkpoint",
    "pairwise_ranking_loss",
    "train",
]


