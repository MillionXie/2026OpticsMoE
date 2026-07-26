from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .features import move_inputs, run_multimodal_forward
from .io_utils import write_json
from .teacher_cache import (
    ProjectedTeacherCacheStore,
    collate_cached_rows,
)


TRAINING_MODES = ("vision", "language", "joint")


def masked_tokenwise_normalized_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student/teacher PCA tap shapes differ: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )
    if mask is not None:
        if mask.shape != student.shape[:-1]:
            raise RuntimeError("Token mask shape does not match PCA token tensor")
        student = student[mask]
        teacher = teacher[mask]
    if student.numel() == 0:
        raise RuntimeError("Masked PCA distillation loss received no valid tokens")
    student = F.layer_norm(student.float(), (student.shape[-1],))
    teacher = F.layer_norm(teacher.float(), (teacher.shape[-1],))
    return F.mse_loss(student, teacher)


def compute_stage_losses(
    replacement: Any,
    batch: dict[str, Any],
    mode: str,
) -> dict[str, torch.Tensor]:
    if mode not in TRAINING_MODES:
        raise ValueError(mode)
    zero = next(replacement.vision_surrogate.parameters()).new_zeros(())
    vision_losses: list[torch.Tensor] = []
    language_losses: list[torch.Tensor] = []
    if mode in {"vision", "joint"}:
        students = replacement.vision_surrogate.stage_latents
        teachers = batch["vision_targets"]
        if len(students) != 4 or len(teachers) != 4:
            raise RuntimeError("Vision distillation requires four stage taps")
        vision_losses = [
            masked_tokenwise_normalized_mse(student, target.to(student.device))
            for student, target in zip(students, teachers)
        ]
    if mode in {"language", "joint"}:
        students = replacement.language_surrogate.stage_latents
        mask = batch["language_mask"]
        teachers = [target[mask] for target in batch["language_targets"]]
        if len(students) != 4 or len(teachers) != 4:
            raise RuntimeError("Language distillation requires four stage taps")
        language_losses = [
            masked_tokenwise_normalized_mse(student, target.to(student.device))
            for student, target in zip(students, teachers)
        ]
    vision = torch.stack(vision_losses).mean() if vision_losses else zero
    language = torch.stack(language_losses).mean() if language_losses else zero
    router = replacement.router_losses()
    balances = []
    importances = []
    if mode in {"vision", "joint"}:
        balances.append(router["vision_balance"])
        importances.append(router["vision_importance"])
    if mode in {"language", "joint"}:
        balances.append(router["language_balance"])
        importances.append(router["language_importance"])
    balance = torch.stack(balances).mean()
    importance = torch.stack(importances).mean()
    return {
        "vision": vision,
        "language": language,
        "balance": balance,
        "importance": importance,
    }


def train_phase(
    mode: str,
    model: torch.nn.Module,
    replacement: Any,
    train_store: ProjectedTeacherCacheStore,
    validation_store: ProjectedTeacherCacheStore,
    settings: Any,
    device: torch.device,
    *,
    pad_token_id: int = 0,
    padding_side: str = "left",
) -> dict[str, Any]:
    if mode not in TRAINING_MODES:
        raise ValueError(mode)
    if mode == "joint":
        load_checkpoint(
            settings.output_dir / "checkpoints" / "vision_best.pt",
            replacement,
            "vision",
        )
        load_checkpoint(
            settings.output_dir / "checkpoints" / "language_best.pt",
            replacement,
            "language",
        )
    _configure_mode(replacement, mode)
    parameters = [
        parameter
        for parameter in replacement.trainable_parameters(mode)
        if parameter.requires_grad
    ]
    unique = {id(parameter): parameter for parameter in parameters}
    router_ids = {
        id(parameter)
        for module in _active_surrogates(replacement, mode)
        for parameter in module.core.router.parameters()
    }
    router_parameters = [
        parameter for identity, parameter in unique.items() if identity in router_ids
    ]
    optical_parameters = [
        parameter for identity, parameter in unique.items() if identity not in router_ids
    ]
    groups = [{"params": optical_parameters, "lr": settings.learning_rate}]
    if router_parameters:
        groups.append({"params": router_parameters, "lr": settings.router_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    epochs = {
        "vision": settings.vision_epochs,
        "language": settings.language_epochs,
        "joint": settings.joint_epochs,
    }[mode]
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if settings.scheduler == "cosine"
        else None
    )
    history_path = settings.output_dir / "metrics" / f"{mode}_training_history.csv"
    best_loss = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_report = _run_epoch(
            mode,
            model,
            replacement,
            train_store,
            settings.student_batch_size,
            settings,
            device,
            optimizer,
            epoch,
            pad_token_id,
            padding_side,
        )
        validation_report = _run_epoch(
            mode,
            model,
            replacement,
            validation_store,
            settings.validation_batch_size,
            settings,
            device,
            None,
            epoch,
            pad_token_id,
            padding_side,
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_total_loss": train_report["total_loss"],
            "train_vision_loss": train_report["vision_loss"],
            "train_language_loss": train_report["language_loss"],
            "train_router_balance_loss": train_report["router_balance_loss"],
            "validation_total_loss": validation_report["total_loss"],
            "validation_vision_loss": validation_report["vision_loss"],
            "validation_language_loss": validation_report["language_loss"],
            "validation_router_balance_loss": validation_report["router_balance_loss"],
            "epoch_time_sec": time.perf_counter() - epoch_started,
        }
        _append_csv(history_path, row)
        write_json(
            settings.output_dir / "metrics" / f"{mode}_training_latest.json", row
        )
        save_checkpoint(
            settings.output_dir / "checkpoints" / f"{mode}_last.pt",
            replacement,
            mode,
            epoch,
            validation_report["total_loss"],
            settings,
        )
        if validation_report["total_loss"] < best_loss:
            best_loss = validation_report["total_loss"]
            best_epoch = epoch
            save_checkpoint(
                settings.output_dir / "checkpoints" / f"{mode}_best.pt",
                replacement,
                mode,
                epoch,
                best_loss,
                settings,
            )
            write_json(
                settings.output_dir / "metrics" / f"{mode}_best_validation.json",
                {
                    "epoch": epoch,
                    "validation_total_loss": best_loss,
                    **validation_report,
                },
            )
        if scheduler is not None:
            scheduler.step()
        print(
            f"[{mode}] epoch {epoch:03d}/{epochs} "
            f"train={train_report['total_loss']:.6f} "
            f"validation={validation_report['total_loss']:.6f} "
            f"best={best_loss:.6f}",
            flush=True,
        )
    report = {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs": epochs,
        "training_time_sec": time.perf_counter() - started,
    }
    write_json(settings.output_dir / "metrics" / f"{mode}_training.json", report)
    return report


def _run_epoch(
    mode: str,
    model: torch.nn.Module,
    replacement: Any,
    store: ProjectedTeacherCacheStore,
    batch_size: int,
    settings: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    pad_token_id: int,
    padding_side: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.eval()
    for surrogate in _active_surrogates(replacement, mode):
        surrogate.train(training)
    replacement.use_student(
        vision=mode in {"vision", "joint"},
        language=mode in {"language", "joint"},
    )
    loader = DataLoader(
        range(len(store)),
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        generator=torch.Generator().manual_seed(settings.seed + epoch),
    )
    totals = {name: 0.0 for name in ("total", "vision", "language", "balance", "importance")}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, indexes in enumerate(loader, start=1):
            rows = store.get_many(indexes.tolist())
            batch = collate_cached_rows(rows, pad_token_id, padding_side)
            batch["inputs"] = move_inputs(batch["inputs"], device)
            batch["language_targets"] = [
                target.to(device, non_blocking=True)
                for target in batch["language_targets"]
            ]
            batch["language_mask"] = batch["language_mask"].to(device, non_blocking=True)
            batch["vision_targets"] = [
                target.to(device, non_blocking=True)
                for target in batch["vision_targets"]
            ]
            replacement.prepare_student_batch(batch["inputs"]["attention_mask"])
            if training:
                optimizer.zero_grad(set_to_none=True)
            run_multimodal_forward(model, batch["inputs"])
            if mode in {"vision", "joint"} and (
                replacement.vision_surrogate.last_token_counts
                != batch["visual_token_counts"]
            ):
                raise RuntimeError("Student visual token counts differ from teacher cache")
            losses = compute_stage_losses(replacement, batch, mode)
            total = (
                losses["vision"]
                + losses["language"]
                + settings.router_balance_weight * losses["balance"]
                + settings.router_importance_weight * losses["importance"]
            )
            if training:
                total.backward()
                optimizer.step()
            count = len(indexes)
            samples += count
            for key, value in losses.items():
                totals[key] += float(value.detach()) * count
            totals["total"] += float(total.detach()) * count
            if batch_index % settings.log_interval_batches == 0:
                print(
                    f"[{mode}] epoch={epoch} batch={batch_index}/{len(loader)} "
                    f"loss={totals['total']/samples:.6f} "
                    f"vision={totals['vision']/samples:.6f} "
                    f"language={totals['language']/samples:.6f} "
                    f"balance={totals['balance']/samples:.6f}",
                    flush=True,
                )
    return {
        "total_loss": totals["total"] / samples,
        "vision_loss": totals["vision"] / samples,
        "language_loss": totals["language"] / samples,
        "router_balance_loss": totals["balance"] / samples,
        "router_importance_loss": totals["importance"] / samples,
        "samples": samples,
    }


def _configure_mode(replacement: Any, mode: str) -> None:
    replacement.vision_surrogate.requires_grad_(mode in {"vision", "joint"})
    replacement.language_surrogate.requires_grad_(mode in {"language", "joint"})
    replacement.vision_surrogate.pca.requires_grad_(False)
    replacement.language_surrogate.pca.requires_grad_(False)


def _active_surrogates(replacement: Any, mode: str) -> list[torch.nn.Module]:
    output = []
    if mode in {"vision", "joint"}:
        output.append(replacement.vision_surrogate)
    if mode in {"language", "joint"}:
        output.append(replacement.language_surrogate)
    return output


def save_checkpoint(
    path: Path,
    replacement: Any,
    mode: str,
    epoch: int,
    validation_loss: float,
    settings: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "manifest_digest": settings.manifest_digest,
        "latent_dim": settings.latent_dim,
        "contains_task_head": False,
    }
    if mode in {"vision", "joint"}:
        payload["vision"] = replacement.vision_surrogate.state_dict()
    if mode in {"language", "joint"}:
        payload["language"] = replacement.language_surrogate.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, replacement: Any, mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mode in {"vision", "joint"}:
        replacement.vision_surrogate.load_state_dict(payload["vision"])
    if mode in {"language", "joint"}:
        replacement.language_surrogate.load_state_dict(payload["language"])
    return {key: payload[key] for key in ("mode", "epoch", "validation_loss")}


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
