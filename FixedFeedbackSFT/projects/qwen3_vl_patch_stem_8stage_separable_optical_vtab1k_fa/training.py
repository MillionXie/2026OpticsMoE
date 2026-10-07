from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .datasets import TASK_SPECS, build_loaders
from .model import P11VtabModel
from .settings import Settings


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def save_torch(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, target)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_model(settings: Settings) -> P11VtabModel:
    if sha256_file(settings.source_backbone) != settings.source_sha256:
        raise RuntimeError("P11 source checkpoint SHA-256 mismatch")
    model = P11VtabModel(
        stem_checkpoint=settings.stem_checkpoint,
        source_checkpoint=settings.source_backbone,
        p11_config=settings.p11_config,
        task_name=settings.task,
        num_classes=settings.num_classes,
        head_hidden_dim=settings.head_hidden_dim,
    )
    report = model.parameter_report()
    if report["optical_phase_parameters"] != 1_204_224:
        raise RuntimeError("VTAB transfer did not retain the 1,204,224-phase P11 body")
    if report["optical_fraction_of_reusable_backbone"] < 0.5:
        raise RuntimeError("Reusable VTAB backbone fell below 50% optical parameters")
    return model


def configure_feedback(model: P11VtabModel, settings: Settings) -> None:
    if settings.method in {"noft", "bp"}:
        model.configure_feedback("bp")
    elif settings.method == "fa_pretrained":
        model.configure_feedback("fa_pretrained")
    else:
        model.configure_feedback("fa_random", random_seed=8_000_003 + settings.seed)


def _optimizer_groups(model: P11VtabModel, settings: Settings) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if settings.method != "noft":
        groups.extend(
            [
                {
                    "params": list(model.phase_parameters()),
                    "lr": settings.phase_learning_rate,
                    "weight_decay": 0.0,
                    "name": "phase",
                },
                {
                    "params": list(model.adapter_parameters()),
                    "lr": settings.adapter_learning_rate,
                    "weight_decay": settings.electronic_weight_decay,
                    "name": "adapter",
                },
                {
                    "params": list(model.residual_parameters()),
                    "lr": settings.residual_learning_rate,
                    "weight_decay": settings.electronic_weight_decay,
                    "name": "residual",
                },
            ]
        )
    groups.append(
        {
            "params": list(model.head_parameters()),
            "lr": settings.head_learning_rate,
            "weight_decay": settings.electronic_weight_decay,
            "name": "head",
        }
    )
    if any(not group["params"] for group in groups):
        raise RuntimeError("An optimizer parameter group is empty")
    return groups


def build_optimizer_scheduler(
    model: P11VtabModel, settings: Settings, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, settings), betas=(0.9, 0.999), eps=1.0e-8
    )
    epochs = settings.head_only_epochs if settings.method == "noft" else settings.adaptation_epochs
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = min(settings.warmup_epochs * steps_per_epoch, total_steps - 1)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        floor = settings.minimum_learning_rate_ratio
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: P11VtabModel,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    use_amp: bool,
    ablation: str = "normal",
) -> dict[str, float | int]:
    model.eval()
    classes = int(model.head.classifier.out_features)  # type: ignore[attr-defined]
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    loss_sum = 0.0
    correct = 0
    samples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(batch["image"], ablation=ablation)["logits"]
        labels = batch["label"]
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum())
        samples += int(labels.numel())
        indices = (labels * classes + predictions).detach().cpu()
        confusion += torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)
    recalls = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
    return {
        "loss": loss_sum / max(samples, 1),
        "top1": correct / max(samples, 1),
        "balanced_accuracy": float(recalls.mean()),
        "samples": samples,
    }


def train_epoch(
    model: P11VtabModel,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(batch["image"])["logits"]
            loss = F.cross_entropy(logits.float(), batch["label"], label_smoothing=0.10)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if list(model.phase_parameters())[0].requires_grad:
            torch.nn.utils.clip_grad_norm_(list(model.phase_parameters()), 2.0)
        non_phase = [
            parameter
            for group in optimizer.param_groups
            if group.get("name") != "phase"
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(non_phase, 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        count = int(batch["label"].numel())
        total_loss += float(loss.detach()) * count
        correct += int((logits.argmax(dim=1) == batch["label"]).sum())
        samples += count
    return {"loss": total_loss / max(samples, 1), "top1": correct / max(samples, 1)}


def _checkpoint_payload(
    model: P11VtabModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    settings: Settings,
    epoch: int,
    history: list[dict[str, Any]],
    data_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": "p14-vtab1k-checkpoint-v1",
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "rng_state": capture_rng_state(),
        "settings_digest": settings.digest(),
        "data_manifest": dict(data_manifest),
    }


def run_experiment(settings: Settings, *, resume: bool = True) -> dict[str, Any]:
    result_path = settings.run_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[P14] already complete: {settings.run_dir}", flush=True)
            return existing

    seed_everything(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_manifest = build_loaders(
        settings.data_root,
        settings.task,
        train_batch_size=settings.train_batch_size,
        evaluation_batch_size=settings.evaluation_batch_size,
        num_workers=settings.num_workers,
        seed=settings.seed,
        smoke_samples=settings.smoke_samples,
    )
    model = build_model(settings)
    if settings.method == "noft":
        model.set_backbone_trainable(False)
    else:
        if not settings.common_checkpoint.is_file():
            raise FileNotFoundError(
                f"{settings.method} requires completed common NoFT checkpoint: "
                f"{settings.common_checkpoint}"
            )
        common = torch.load(settings.common_checkpoint, map_location="cpu", weights_only=False)
        if common.get("task") != settings.task or common.get("seed") != settings.seed:
            raise RuntimeError("Common NoFT checkpoint identity mismatch")
        model.load_state_dict(common["model"], strict=True)
        model.set_backbone_trainable(True)
    configure_feedback(model, settings)
    model.to(device)

    train_loader = loaders["train800val200"]
    optimizer, scheduler = build_optimizer_scheduler(model, settings, len(train_loader))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=settings.use_amp and device.type == "cuda", init_scale=256
    )
    epochs = settings.head_only_epochs if settings.method == "noft" else settings.adaptation_epochs
    history: list[dict[str, Any]] = []
    start_epoch = 0
    checkpoint_path = settings.run_dir / "checkpoints" / "last.pt"
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("settings_digest") != settings.digest():
            raise RuntimeError("Resume checkpoint settings digest mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        configure_feedback(model, settings)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])

    settings.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.run_dir / "resolved_settings.json", settings.identity_payload())
    write_json(settings.run_dir / "dataset_manifest.json", data_manifest)
    write_json(settings.run_dir / "model_report.json", model.parameter_report())
    started = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            use_amp=settings.use_amp,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "phase": model.phase_report(),
            "learning_rates": {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        history.append(row)
        save_torch(
            checkpoint_path,
            _checkpoint_payload(
                model, optimizer, scheduler, scaler, settings, epoch, history, data_manifest
            ),
        )
        write_json(settings.run_dir / "history.json", history)
        print(
            f"[P14] task={settings.task} method={settings.method} seed={settings.seed} "
            f"epoch={epoch}/{epochs} train_top1={train_metrics['top1']:.4f} "
            f"phase_mae={model.phase_report()['mean_absolute_rad']:.5f}",
            flush=True,
        )

    test = evaluate(
        model, loaders["test"], device,
        use_amp=settings.use_amp,
    )
    result = {
        "format": "p14-vtab1k-result-v1",
        "status": "complete",
        "benchmark": "VTAB-1k",
        "protocol": "fixed recipe on train800val200; sealed test evaluated once",
        "task": settings.task,
        "category": TASK_SPECS[settings.task].category,
        "method": settings.method,
        "seed": settings.seed,
        "head_only_epochs": settings.head_only_epochs,
        "adaptation_epochs": 0 if settings.method == "noft" else settings.adaptation_epochs,
        "test": test,
        "phase": model.phase_report(),
        "feedback": model.feedback_manifest(),
        "settings_digest": settings.digest(),
        "source_checkpoint_sha256": settings.source_sha256,
        "data_manifest": data_manifest,
        "git_commit": git_commit(),
        "physical_gpu": os.environ.get("P14_PHYSICAL_GPU", "unrecorded"),
        "physical_gpu_uuid": os.environ.get("P14_GPU_UUID", "unrecorded"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unrestricted"),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "wall_seconds": time.time() - started,
    }
    write_json(result_path, result)
    print(
        f"[P14] complete task={settings.task} method={settings.method} "
        f"seed={settings.seed} test_top1={float(test['top1']):.5f}",
        flush=True,
    )
    return result


__all__ = ["build_model", "evaluate", "run_experiment"]
