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

from .data import LGVQFrameDataset
from .metrics import regression_metrics
from .modeling import LGVQFourStageOEO
from .settings import ExperimentSettings, resolved_dict


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _loader(payload: Mapping[str, Any], split: str, settings: ExperimentSettings, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        LGVQFrameDataset(payload, split), batch_size=settings.batch_size, shuffle=shuffle,
        num_workers=settings.num_workers, pin_memory=settings.device.startswith("cuda"), drop_last=False,
        persistent_workers=settings.num_workers > 0,
    )


def _target_weights(reference: torch.Tensor, weights: tuple[float, float] | None) -> torch.Tensor:
    """Return positive target weights with mean one to preserve the loss scale."""
    if weights is None:
        return reference.new_ones((reference.shape[-1],))
    result = reference.new_tensor(weights)
    if result.numel() != reference.shape[-1] or not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError("Target loss weights must be finite, positive, and match the output width")
    return result * (result.numel() / result.sum())


def weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_weights: tuple[float, float] | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Regression inputs must have equal [B,C] shape")
    per_target = F.smooth_l1_loss(prediction, target, reduction="none").mean(0)
    weights = _target_weights(prediction, target_weights)
    return (per_target * weights).mean()


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_difference: float = 0.05,
    target_weights: tuple[float, float] | None = None,
) -> torch.Tensor:
    losses, columns = [], []
    for column in range(2):
        difference = target[:, None, column] - target[None, :, column]
        predicted = prediction[:, None, column] - prediction[None, :, column]
        valid = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1) & (difference.abs() >= minimum_difference)
        if bool(valid.any()):
            losses.append(F.softplus(-difference[valid].sign() * predicted[valid]).mean())
            columns.append(column)
    if not losses:
        return prediction.new_zeros(())
    weights = _target_weights(prediction, target_weights)[columns]
    return (torch.stack(losses) * weights).sum() / weights.sum()


def batch_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-6,
    target_weights: tuple[float, float] | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Correlation inputs must have equal [B,C] shape")
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    left = prediction.float() - prediction.float().mean(0, keepdim=True)
    right = target.float() - target.float().mean(0, keepdim=True)
    target_energy = right.square().sum(0)
    denominator = (left.square().sum(0) * target_energy).sqrt().clamp_min(epsilon)
    correlation = (left * right).sum(0) / denominator
    valid = target_energy > epsilon
    if not bool(valid.any()):
        return prediction.new_zeros(())
    losses = 1.0 - correlation[valid].clamp(-1.0, 1.0)
    weights = _target_weights(prediction, target_weights)[valid]
    return (losses * weights).sum() / weights.sum()


def normalized_soft_target_loss(
    normalized_prediction: torch.Tensor,
    soft_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_weights: tuple[float, float] | None = None,
) -> torch.Tensor:
    if normalized_prediction.shape != soft_target.shape or normalized_prediction.ndim != 2:
        raise ValueError("Soft-target inputs must have equal [B,2] shape")
    normalized_teacher = (soft_target - target_mean) / target_std
    return weighted_smooth_l1_loss(normalized_prediction, normalized_teacher, target_weights)


@torch.no_grad()
def evaluate(
    model: LGVQFourStageOEO,
    loader: DataLoader,
    device: torch.device,
    *,
    optical_enabled: bool,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions, targets, sample_ids, video_paths = [], [], [], []
    fusion_sums: dict[str, dict[str, float]] = {}
    fusion_count = 0
    routing_sums: dict[str, dict[str, Any]] = {}
    for batch in loader:
        result = model(batch["frames"].to(device, non_blocking=True), optical_enabled=optical_enabled)
        prediction = result["prediction"]
        predictions.append(prediction.detach().cpu())
        targets.append(batch["target"].detach().cpu())
        sample_ids.extend(batch["sample_id"])
        video_paths.extend(batch["video_path"])
        if optical_enabled:
            count = prediction.shape[0]
            fusion_count += count
            for stage, values in model.fusion_diagnostics().items():
                accumulator = fusion_sums.setdefault(stage, {})
                for key, value in values.items():
                    accumulator[key] = accumulator.get(key, 0.0) + float(value) * count
            for stage, routing in result["routing"].items():
                probabilities = routing["probabilities"].detach().float().reshape(-1, 4).cpu()
                selected = routing["selected_mask"].detach().float().reshape(-1, 4).cpu()
                accumulator = routing_sums.setdefault(stage, {"count": 0, "probability": torch.zeros(4), "selected": torch.zeros(4), "capture_sum": 0.0, "capture_count": 0, "implementation": routing["router_implementation"]})
                accumulator["count"] += probabilities.shape[0]
                accumulator["probability"] += probabilities.sum(0)
                accumulator["selected"] += selected.sum(0)
                capture = routing["capture_fraction"].detach().float().reshape(-1).cpu()
                accumulator["capture_sum"] += float(capture.sum())
                accumulator["capture_count"] += capture.numel()
    prediction, target = torch.cat(predictions), torch.cat(targets)
    metrics = regression_metrics(prediction, target)
    metrics["optical_enabled"] = optical_enabled
    metrics["checkpoint_training_mode"] = "optical_on"
    metrics["fusion_diagnostics"] = {
        stage: {key: value / max(1, fusion_count) for key, value in values.items()}
        for stage, values in fusion_sums.items()
    }
    metrics["router_diagnostics"] = {}
    for stage, values in routing_sums.items():
        count = max(1, int(values["count"]))
        selected_total = max(1.0, float(values["selected"].sum()))
        metrics["router_diagnostics"][stage] = {
            "implementation": values["implementation"],
            "decision_count": int(values["count"]),
            "mean_probability": (values["probability"] / count).tolist(),
            "selected_share": (values["selected"] / selected_total).tolist(),
            "capture_fraction_mean": values["capture_sum"] / max(1, values["capture_count"]),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("sample_id", "video_path", "spatial_target", "spatial_prediction", "temporal_target", "temporal_prediction"))
            for index, sample_id in enumerate(sample_ids):
                writer.writerow((sample_id, video_paths[index], float(target[index, 0]), float(prediction[index, 0]), float(target[index, 1]), float(prediction[index, 1])))
    return metrics


def _optimizer(model: nn.Module, settings: ExperimentSettings) -> torch.optim.Optimizer:
    electronic, feature_phase, router_phase = [], [], []
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
        {"params": electronic, "lr": settings.learning_rate, "weight_decay": settings.weight_decay, "name": "electronic"},
        {"params": feature_phase, "lr": settings.phase_learning_rate, "weight_decay": 0.0, "name": "feature_phase"},
        {"params": router_phase, "lr": settings.router_phase_learning_rate, "weight_decay": 0.0, "name": "router_phase"},
    ]
    groups = [group for group in groups if group["params"]]
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("Optimizer groups overlap or omit trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_initialization(
    model: LGVQFourStageOEO,
    settings: ExperimentSettings,
    *,
    apply: bool,
) -> dict[str, Any] | None:
    checkpoint = settings.initialization_checkpoint
    if checkpoint is None:
        return None
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training initialization checkpoint is missing: {checkpoint}")
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Training initialization checkpoint cannot be loaded: {checkpoint}: {error}") from error
    if not isinstance(saved, Mapping):
        raise RuntimeError("Training initialization checkpoint root must be a mapping")
    source_architecture = saved.get("architecture")
    if not isinstance(source_architecture, str) or not source_architecture:
        raise RuntimeError("Training initialization checkpoint has no valid source architecture")
    if not settings.synthetic and not source_architecture.endswith("_v1"):
        raise RuntimeError(
            "Formal partial warm-start requires a source checkpoint whose architecture ends with _v1"
        )
    source_state = saved.get("state_dict")
    if not isinstance(source_state, Mapping) or not source_state:
        raise RuntimeError("Training initialization checkpoint has no valid state_dict")
    target_state = model.state_dict()
    parameter_names = {name for name, _ in model.named_parameters()}
    compatible: dict[str, torch.Tensor] = {}
    skipped_source: list[dict[str, Any]] = []
    for name, value in source_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("Training initialization state_dict must map string names to tensors")
        if name not in target_state:
            skipped_source.append({"key": name, "reason": "not_present_in_v3"})
            continue
        if tuple(value.shape) != tuple(target_state[name].shape):
            skipped_source.append({
                "key": name,
                "reason": "shape_mismatch",
                "source_shape": list(value.shape),
                "target_shape": list(target_state[name].shape),
            })
            continue
        compatible[name] = value
    if not compatible:
        raise RuntimeError("Training initialization checkpoint has no name-and-shape compatible tensors")
    if apply:
        model.load_state_dict(compatible, strict=False)
    source_epoch = saved.get("epoch")
    if source_epoch is not None:
        try:
            source_epoch = int(source_epoch)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Training initialization checkpoint epoch must be an integer") from error
    matched_parameter_count = sum(
        int(target_state[name].numel()) for name in compatible if name in parameter_names
    )
    fresh_target = sorted(name for name in target_state if name not in compatible)
    return {
        "mode": "partial_v1_to_v3_name_and_shape",
        "source_path": str(checkpoint),
        "source_sha256": _file_sha256(checkpoint),
        "source_architecture": source_architecture,
        "source_epoch": source_epoch,
        "source_schema_version": saved.get("schema_version"),
        "target_architecture": settings.architecture_label,
        "matched_tensor_count": len(compatible),
        "matched_parameter_count": matched_parameter_count,
        "matched_parameter_fraction": matched_parameter_count / max(
            1, sum(parameter.numel() for parameter in model.parameters())
        ),
        "skipped_keys": {
            "source": skipped_source,
            "target_fresh": fresh_target,
        },
        "optimizer_restored": False,
        "applied_to_model": bool(apply),
    }


def inspect_training_initialization(
    model: LGVQFourStageOEO,
    settings: ExperimentSettings,
) -> dict[str, Any] | None:
    return _training_initialization(model, settings, apply=False)


def apply_training_initialization(
    model: LGVQFourStageOEO,
    settings: ExperimentSettings,
) -> dict[str, Any] | None:
    return _training_initialization(model, settings, apply=True)


def _checkpoint(
    path: Path,
    model: LGVQFourStageOEO,
    optimizer: torch.optim.Optimizer,
    settings: ExperimentSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
    soft_target_provenance: Mapping[str, Any] | None = None,
    initialization_provenance: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "architecture": settings.architecture_label,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics_optical_on": dict(metrics or {}),
        "settings": resolved_dict(settings),
        "selection_policy": "highest periodically observed optical-on test mean SRCC; no validation split",
        "electronic_only_was_trained": False,
        "training_soft_targets": None if soft_target_provenance is None else dict(soft_target_provenance),
        "soft_target_weight": float(settings.soft_target_weight),
        "deployed_inference_uses_teacher": False,
        "training_initialization": None if initialization_provenance is None else dict(initialization_provenance),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_state(model: LGVQFourStageOEO, checkpoint: Path, settings: ExperimentSettings) -> dict[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint architecture does not match this project")
    model.load_state_dict(saved["state_dict"], strict=True)
    return saved


@torch.no_grad()
def _phase_diagnostics(model: LGVQFourStageOEO, initial: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "raw_" not in name or "phase" not in name:
            continue
        start_raw = initial[name].to(parameter.device, dtype=parameter.dtype)
        final_raw = parameter.detach()
        start_phase = 2.0 * math.pi * torch.sigmoid(start_raw.float())
        final_phase = 2.0 * math.pi * torch.sigmoid(final_raw.float())
        difference = final_phase - start_phase
        wrapped = torch.atan2(torch.sin(difference), torch.cos(difference))
        report[name] = {
            "parameters": int(parameter.numel()),
            "raw_std_initial": float(start_raw.float().std(unbiased=False)),
            "raw_std_final": float(final_raw.float().std(unbiased=False)),
            "phase_rad_std_initial": float(start_phase.std(unbiased=False)),
            "phase_rad_std_final": float(final_phase.std(unbiased=False)),
            "wrapped_delta_rad_rms": float(wrapped.square().mean().sqrt()),
            "wrapped_delta_rad_mean_abs": float(wrapped.abs().mean()),
            "fraction_changed_over_0p05_rad": float((wrapped.abs() > 0.05).float().mean()),
        }
    aggregate = [value["wrapped_delta_rad_rms"] for value in report.values()]
    return {
        "planes": report,
        "plane_count": len(report),
        "mean_wrapped_delta_rad_rms": sum(aggregate) / max(1, len(aggregate)),
        "checkpoint_state": "best observed optical-on test checkpoint",
    }


def evaluate_checkpoint_modes(model: LGVQFourStageOEO, payload: Mapping[str, Any], settings: ExperimentSettings, device: torch.device, checkpoint: Path) -> dict[str, Any]:
    saved = _load_state(model, checkpoint, settings)
    model.to(device)
    loader = _loader(payload, "test", settings, shuffle=False)
    optical_on = evaluate(model, loader, device, optical_enabled=True, prediction_path=settings.output_dir / "test_predictions_optical_on.csv")
    optical_off = evaluate(model, loader, device, optical_enabled=False, prediction_path=settings.output_dir / "test_predictions_optical_off.csv")
    delta: dict[str, Any] = {}
    for target in ("spatial", "temporal"):
        delta[target] = {
            name: float(optical_on[target][name]) - float(optical_off[target][name])
            for name in ("srcc", "krcc", "plcc", "rmse", "mae")
        }
    report = {
        "checkpoint": str(checkpoint.resolve()),
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
    model: LGVQFourStageOEO,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
    *,
    initialization_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(payload, "train", settings, shuffle=True)
    test_loader = _loader(payload, "test", settings, shuffle=False)
    train_indices = [index for index, split in enumerate(payload["splits"]) if split == "train"]
    train_targets = payload["targets"][train_indices].float()
    model.set_target_statistics(train_targets.mean(0), train_targets.std(0, unbiased=False).clamp_min(1.0e-6))
    model.to(device)
    initial_phase = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "raw_" in name and "phase" in name
    }
    optimizer = _optimizer(model, settings)
    soft_target_provenance = payload.get("training_soft_target_provenance")
    if settings.soft_target_weight > 0.0 and soft_target_provenance is None:
        raise RuntimeError("Soft-target loss is enabled but aligned training soft targets are absent")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    target_loss_weights = (settings.spatial_target_weight, settings.temporal_target_weight)
    best_score, best_epoch = float("-inf"), 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "regression", "ranking", "correlation", "soft_target", "optical_alignment", "router_balance", "router_importance", "router_capture")}
        batches = 0
        for batch in train_loader:
            frames = batch["frames"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = model(frames, optical_enabled=True)
            regression = weighted_smooth_l1_loss(
                result["normalized_prediction"], normalized_target, target_loss_weights
            )
            ranking = pairwise_ranking_loss(
                result["normalized_prediction"], normalized_target, target_weights=target_loss_weights
            )
            correlation = batch_correlation_loss(
                result["normalized_prediction"], normalized_target, target_weights=target_loss_weights
            )
            soft_target = result["normalized_prediction"].new_zeros(())
            if "soft_target" in batch:
                teacher = batch["soft_target"].to(device, non_blocking=True)
                soft_target = normalized_soft_target_loss(
                    result["normalized_prediction"], teacher, model.target_mean, model.target_std,
                    target_loss_weights,
                )
            loss = (
                regression + settings.ranking_weight * ranking + settings.correlation_weight * correlation
                + settings.soft_target_weight * soft_target
                + settings.optical_alignment_weight * result["optical_alignment_loss"]
                + settings.router_balance_weight * result["router_balance_loss"]
                + settings.router_importance_weight * result["router_importance_loss"]
                + settings.router_capture_weight * result["router_capture_loss"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            bad = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
            if bad:
                raise RuntimeError(f"Non-finite gradients: {bad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss, "regression": regression, "ranking": ranking, "correlation": correlation, "soft_target": soft_target,
                "optical_alignment": result["optical_alignment_loss"], "router_balance": result["router_balance_loss"],
                "router_importance": result["router_importance_loss"], "router_capture": result["router_capture_loss"],
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}, "test_evaluated": False}
        if epoch == 1 or epoch % settings.test_interval_epochs == 0 or epoch == settings.epochs:
            metrics = evaluate(model, test_loader, device, optical_enabled=True)
            row["test_evaluated"] = True
            row["test_optical_on"] = metrics
            score = float(metrics["selection_mean_srcc"])
            if score > best_score:
                best_score, best_epoch = score, epoch
                _checkpoint(
                    settings.output_dir / "best_observed_test_checkpoint.pt",
                    model,
                    optimizer,
                    settings,
                    epoch=epoch,
                    metrics=metrics,
                    soft_target_provenance=soft_target_provenance,
                    initialization_provenance=initialization_provenance,
                )
                _json(settings.output_dir / "metrics_best_observed_test_optical_on.json", metrics)
        history.append(row)
        _json(settings.output_dir / "train_history.json", history)
        if row["test_evaluated"]:
            test = row["test_optical_on"]
            print(f"epoch {epoch:03d} loss={row['loss']:.6f} spatial_SRCC={test['spatial']['srcc']:.4f} temporal_SRCC={test['temporal']['srcc']:.4f} mean_SRCC={test['selection_mean_srcc']:.4f}", flush=True)
        else:
            print(f"epoch {epoch:03d} loss={row['loss']:.6f} test=skipped", flush=True)
    _checkpoint(
        settings.output_dir / "last_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=settings.epochs,
        metrics=history[-1].get("test_optical_on"),
        soft_target_provenance=soft_target_provenance,
        initialization_provenance=initialization_provenance,
    )
    best_checkpoint = settings.output_dir / "best_observed_test_checkpoint.pt"
    comparison = evaluate_checkpoint_modes(model, payload, settings, device, best_checkpoint)
    phase_report = _phase_diagnostics(model, initial_phase)
    _json(settings.output_dir / "phase_training_diagnostics.json", phase_report)
    report = {
        "best_epoch": best_epoch,
        "best_optical_on_test_mean_srcc": best_score,
        "checkpoint": str(best_checkpoint),
        "periodic_test_interval": settings.test_interval_epochs,
        "validation_used": False,
        "test_used_for_selection": True,
        "training_soft_targets": soft_target_provenance,
        "soft_target_weight": float(settings.soft_target_weight),
        "supervised_target_weights": {
            "spatial": float(settings.spatial_target_weight),
            "temporal": float(settings.temporal_target_weight),
            "normalization": "mean_one",
            "applied_to": ["regression", "ranking", "correlation", "soft_target"],
        },
        "deployed_inference_uses_teacher": False,
        "training_initialization": None if initialization_provenance is None else dict(initialization_provenance),
        "same_checkpoint_optical_ablation": comparison,
        "phase_training_diagnostics": phase_report,
    }
    _json(settings.output_dir / "training_summary.json", report)
    return report


__all__ = [
    "apply_training_initialization",
    "batch_correlation_loss",
    "evaluate",
    "evaluate_checkpoint_modes",
    "normalized_soft_target_loss",
    "weighted_smooth_l1_loss",
    "pairwise_ranking_loss",
    "inspect_training_initialization",
    "train",
]
