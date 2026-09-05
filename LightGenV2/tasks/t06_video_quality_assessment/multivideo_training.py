"""Training/evaluation for the true full-field 9x4 Temporal model."""

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

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import regression_metrics
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.training import (
    batch_correlation_loss,
    pairwise_ranking_loss,
)

from .models.multivideo9x4 import MultiVideo9x4OpticalVQA
from .multivideo_data import NineVideoFieldDataset, permute_video_slots
from .multivideo_settings import MultiVideoSettings, resolved_dict


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loader(
    payload: Mapping[str, Any],
    split: str,
    settings: MultiVideoSettings,
    *,
    grouping_seed: int,
    shuffle_membership: bool,
    shuffle_groups: bool,
) -> DataLoader:
    dataset = NineVideoFieldDataset(
        payload,
        split,
        videos_per_field=settings.videos_per_field,
        grouping_seed=grouping_seed,
        shuffle_membership=shuffle_membership,
    )
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle_groups,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
        persistent_workers=False,
        drop_last=False,
    )


def compatible_warm_start(
    model: MultiVideo9x4OpticalVQA, settings: MultiVideoSettings
) -> dict[str, Any]:
    path = settings.initialization_checkpoint
    if path is None:
        return {"used": False, "reason": "not configured"}
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint is missing: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    source = saved.get("state_dict", saved.get("model", saved))
    destination = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in destination
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(destination[name].shape)
    }
    loaded = model.load_state_dict(compatible, strict=False)
    return {
        "used": True,
        "path": str(path),
        "sha256": _sha256(path),
        "source_architecture": saved.get("architecture"),
        "source_epoch": saved.get("epoch"),
        "loaded_tensors": len(compatible),
        "loaded_parameters": sum(value.numel() for value in compatible.values()),
        "missing_tensors": list(loaded.missing_keys),
        "policy": "exact-name and exact-shape only; new 9x4 optical masks are never resized",
    }


def _optimizer(
    model: nn.Module, settings: MultiVideoSettings
) -> torch.optim.Optimizer:
    electronic, feature_phase, router_phase = [], [], []
    for name, parameter in model.named_parameters():
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
    return torch.optim.AdamW([group for group in groups if group["params"]])


def _checkpoint(
    path: Path,
    model: MultiVideo9x4OpticalVQA,
    optimizer: torch.optim.Optimizer,
    settings: MultiVideoSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "architecture": settings.architecture_label,
        "frame_semantics": "nine_independent_videos_each_four_frames",
        "output_contract": "prediction[B,9], one continuous Temporal MOS per video",
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics_optical_on": dict(metrics or {}),
        "settings": resolved_dict(settings),
        "selection_policy": "highest periodically observed Temporal test SRCC; no validation split",
        "test_used_for_selection": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _phase_snapshot(
    model: MultiVideo9x4OpticalVQA,
    settings: MultiVideoSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
) -> None:
    phases = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "raw_" in name and "phase" in name
    }
    path = settings.output_dir / "phase_snapshots" / f"epoch_{epoch:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "architecture": settings.architecture_label,
            "epoch": epoch,
            "parameterization": "phase_rad = 2*pi*sigmoid(raw_phase)",
            "raw_phase_state": phases,
            "metrics_optical_on": dict(metrics or {}),
            "layout": resolved_dict(settings)["geometry"],
        },
        path,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: MultiVideo9x4OpticalVQA,
    payload: Mapping[str, Any],
    settings: MultiVideoSettings,
    device: torch.device,
    *,
    optical_enabled: bool,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = _loader(
        payload,
        "test",
        settings,
        grouping_seed=0,
        shuffle_membership=False,
        shuffle_groups=False,
    )
    predictions, targets, indices, slots = [], [], [], []
    router = {}
    fusion = {}
    guard_sum = 0.0
    valid_count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        result = model(
            batch["vision_tokens"],
            batch["quality_tokens"],
            batch["language_tokens"],
            batch["language_mask"],
            optical_enabled=optical_enabled,
        )
        valid = batch["valid"]
        predictions.append(result["prediction"][valid].cpu())
        targets.append(batch["target"][valid].cpu())
        indices.append(batch["source_indices"][valid].cpu())
        slot_grid = torch.arange(settings.videos_per_field, device=device)[None].expand_as(valid)
        slots.append(slot_grid[valid].cpu())
        count = int(valid.sum())
        valid_count += count
        if optical_enabled:
            guard_sum += float(result["guard_energy_loss"]) * count
            for name, diagnostics in model.fusion_diagnostics().items():
                row = fusion.setdefault(name, {})
                for key, value in diagnostics.items():
                    row[key] = row.get(key, 0.0) + float(value) * count
            for name, diagnostics in result["routing"].items():
                probability = diagnostics["probabilities"].detach().reshape(-1, 4)
                selected = diagnostics["selected_mask"].detach().float().reshape(-1, 4)
                row = router.setdefault(
                    name,
                    {"probability": torch.zeros(4, device=device), "selected": torch.zeros(4, device=device), "count": 0, "implementation": diagnostics["router_implementation"]},
                )
                row["probability"] += probability.sum(0)
                row["selected"] += selected.sum(0)
                row["count"] += probability.shape[0]
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    source = torch.cat(indices)
    slot = torch.cat(slots)
    metrics = regression_metrics(prediction, target, settings.target_name)
    metrics["optical_enabled"] = optical_enabled
    metrics["video_count"] = int(prediction.numel())
    metrics["physical_field_count"] = math.ceil(prediction.numel() / settings.videos_per_field)
    metrics["output_contract"] = "one scalar Temporal MOS per video"
    metrics["slot_diagnostics"] = {
        str(index): {
            "count": int((slot == index).sum()),
            "prediction_mean": float(prediction[slot == index].mean()),
            "prediction_std": float(prediction[slot == index].std(unbiased=False)),
        }
        for index in range(settings.videos_per_field)
    }
    if optical_enabled:
        metrics["guard_energy_fraction_mean"] = guard_sum / max(1, valid_count)
        metrics["fusion_diagnostics"] = {
            name: {key: value / max(1, valid_count) for key, value in values.items()}
            for name, values in fusion.items()
        }
        metrics["router_diagnostics"] = {}
        for name, values in router.items():
            probability = (values["probability"] / max(1, values["count"])).cpu()
            selected_share = (values["selected"] / values["selected"].sum().clamp_min(1)).cpu()
            metrics["router_diagnostics"][name] = {
                "implementation": values["implementation"],
                "mean_probability": probability.tolist(),
                "selected_share": selected_share.tolist(),
                "maximum_selected_share": float(selected_share.max()),
                "effective_experts_probability": float(torch.exp(-(probability * probability.clamp_min(1e-8).log()).sum())),
            }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("sample_id", "video_path", "slot", "target", "prediction", "absolute_error"))
            for i, source_index in enumerate(source.tolist()):
                writer.writerow((payload["sample_ids"][source_index], payload["video_paths"][source_index], int(slot[i]), float(target[i]), float(prediction[i]), abs(float(target[i]) - float(prediction[i]))))
    return metrics


def train(
    model: MultiVideo9x4OpticalVQA,
    payload: Mapping[str, Any],
    settings: MultiVideoSettings,
    device: torch.device,
) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_indices = [index for index, split in enumerate(payload["splits"]) if split == "train"]
    train_targets = torch.as_tensor(payload["targets"])[train_indices].float()
    model.set_target_statistics(train_targets.mean(), train_targets.std(unbiased=False))
    model.to(device)
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    initial = evaluate(model, payload, settings, device, optical_enabled=True)
    best_srcc, best_epoch = float(initial["srcc"]), 0
    history = [{"epoch": 0, "test_evaluated": True, "test_optical_on": initial}]
    best_path = settings.output_dir / "best_observed_test_checkpoint.pt"
    _checkpoint(best_path, model, optimizer, settings, epoch=0, metrics=initial)
    _json(settings.output_dir / "metrics_best_observed_test_optical_on.json", initial)
    print(f"epoch 000 warm-start temporal_SRCC={best_srcc:.4f}", flush=True)
    slot_generator = torch.Generator().manual_seed(settings.random_seed + 100000)
    for epoch in range(1, settings.epochs + 1):
        loader = _loader(
            payload,
            "train",
            settings,
            grouping_seed=settings.random_seed + epoch,
            shuffle_membership=True,
            shuffle_groups=True,
        )
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch_index, raw_batch in enumerate(loader):
            raw_batch, _ = permute_video_slots(raw_batch, generator=slot_generator)
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = model(
                batch["vision_tokens"], batch["quality_tokens"], batch["language_tokens"], batch["language_mask"], optical_enabled=True
            )
            valid = batch["valid"]
            prediction = result["normalized_prediction"][valid]
            target = ((batch["target"] - model.target_mean) / model.target_std)[valid]
            regression = F.smooth_l1_loss(prediction, target)
            ranking = pairwise_ranking_loss(prediction, target)
            correlation = batch_correlation_loss(prediction, target)
            soft_target = prediction.new_zeros(())
            if "soft_target" in batch:
                present = valid & batch["soft_target_present"]
                teacher = ((batch["soft_target"] - model.target_mean) / model.target_std)[present]
                soft_target = F.smooth_l1_loss(result["normalized_prediction"][present], teacher)
            slot_consistency = prediction.new_zeros(())
            if settings.slot_consistency_weight > 0 and batch_index % settings.slot_consistency_interval == 0:
                second_raw, inverse = permute_video_slots(raw_batch, generator=slot_generator)
                second = _to_device(second_raw, device)
                second_result = model(
                    second["vision_tokens"], second["quality_tokens"], second["language_tokens"], second["language_mask"], optical_enabled=True
                )
                aligned = torch.gather(
                    second_result["normalized_prediction"], 1, inverse.to(device)
                )
                slot_consistency = F.smooth_l1_loss(
                    aligned[batch["valid"]], result["normalized_prediction"].detach()[batch["valid"]]
                )
            loss = (
                regression
                + settings.ranking_weight * ranking
                + settings.correlation_weight * correlation
                + settings.soft_target_weight * soft_target
                + settings.optical_alignment_weight * result["optical_alignment_loss"]
                + settings.router_balance_weight * result["router_balance_loss"]
                + settings.router_importance_weight * result["router_importance_loss"]
                + settings.router_capture_weight * result["router_capture_loss"]
                + settings.guard_energy_weight * result["guard_energy_loss"]
                + settings.slot_consistency_weight * slot_consistency
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss,
                "regression": regression,
                "ranking": ranking,
                "correlation": correlation,
                "soft_target": soft_target,
                "optical_alignment": result["optical_alignment_loss"],
                "router_balance": result["router_balance_loss"],
                "router_importance": result["router_importance_loss"],
                "router_capture": result["router_capture_loss"],
                "guard_energy": result["guard_energy_loss"],
                "slot_consistency": slot_consistency,
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}, "test_evaluated": False}
        if epoch == 1 or epoch % settings.test_interval_epochs == 0 or epoch == settings.epochs:
            metrics = evaluate(model, payload, settings, device, optical_enabled=True)
            row["test_evaluated"] = True
            row["test_optical_on"] = metrics
            score = float(metrics["srcc"])
            if math.isfinite(score) and score > best_srcc:
                best_srcc, best_epoch = score, epoch
                _checkpoint(best_path, model, optimizer, settings, epoch=epoch, metrics=metrics)
                _json(settings.output_dir / "metrics_best_observed_test_optical_on.json", metrics)
        history.append(row)
        _json(settings.output_dir / "train_history.json", history)
        if epoch % settings.phase_snapshot_interval_epochs == 0:
            _phase_snapshot(model, settings, epoch=epoch, metrics=row.get("test_optical_on"))
        if row["test_evaluated"]:
            print(f"epoch {epoch:03d} loss={row['loss']:.6f} temporal_SRCC={row['test_optical_on']['srcc']:.4f}", flush=True)
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
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device)
    optical_on = evaluate(
        model, payload, settings, device, optical_enabled=True,
        prediction_path=settings.output_dir / "test_predictions_optical_on.csv",
    )
    optical_off = evaluate(
        model, payload, settings, device, optical_enabled=False,
        prediction_path=settings.output_dir / "test_predictions_optical_off.csv",
    )
    comparison = {
        "same_checkpoint": str(best_path),
        "normal_optical_electronic": optical_on,
        "same_checkpoint_optics_bypassed": optical_off,
        "separately_trained_electronic_baseline": False,
    }
    _json(settings.output_dir / "optical_contribution_same_checkpoint.json", comparison)
    summary = {
        "architecture": settings.architecture_label,
        "best_epoch": best_epoch,
        "best_observed_test_srcc": best_srcc,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "test_used_for_selection": True,
        "validation_used": False,
        "output_contract": "[physical_batch,9], one scalar Temporal MOS per video",
        "comparison": comparison,
    }
    _json(settings.output_dir / "training_summary.json", summary)
    return summary


__all__ = ["compatible_warm_start", "evaluate", "train"]
