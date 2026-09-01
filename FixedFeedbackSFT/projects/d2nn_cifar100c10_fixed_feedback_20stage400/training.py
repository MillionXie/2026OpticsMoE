from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .datasets import DatasetBundle, epoch_order, order_digest
from .model import OpticalClassifier
from .optics import phasor_operator_coherence, phasor_operator_distance
from .settings import Settings
from .visualization import (
    save_confusion_matrix,
    save_history_plot,
    save_optical_example,
    save_phase_overview,
)


Method = Literal["bp", "fa_pretrained", "fa_random", "no_finetune"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _loader(
    dataset: Dataset,
    indices: list[int] | None,
    *,
    batch_size: int,
    settings: Settings,
) -> DataLoader:
    selected = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        selected,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=settings.data.num_workers,
        pin_memory=settings.data.pin_memory,
        persistent_workers=False,
        drop_last=False,
    )


def build_model(settings: Settings, device: torch.device) -> OpticalClassifier:
    model = OpticalClassifier(settings.optical).to(device)
    return model


def build_optimizer(model: OpticalClassifier, settings: Settings) -> torch.optim.Optimizer:
    phase_parameters = list(model.phase_parameters())
    electronic_parameters = list(model.electronic_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": phase_parameters, "lr": settings.optimizer.phase_learning_rate, "name": "phase"},
            {
                "params": electronic_parameters,
                "lr": settings.optimizer.electronic_learning_rate,
                "name": "electronic",
            },
        ],
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
        weight_decay=settings.optimizer.weight_decay,
    )
    return optimizer


def _selected_logits(
    logits: torch.Tensor,
    batch: dict[str, Any],
    selected_class_indices: tuple[int, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if selected_class_indices is None:
        return logits, batch["target"].to(logits.device, non_blocking=True).long()
    index = torch.tensor(selected_class_indices, device=logits.device, dtype=torch.long)
    return logits.index_select(1, index), batch["local_target"].to(logits.device, non_blocking=True).long()


def train_epoch(
    model: OpticalClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    selected_class_indices: tuple[int, ...] | None,
    *,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    prefix: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        task_logits, targets = _selected_logits(logits, batch, selected_class_indices)
        loss = F.cross_entropy(task_logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {prefix} epoch={epoch} batch={batch_index}")
        loss.backward()
        optimizer.step()
        count = int(targets.numel())
        total_loss += float(loss.detach()) * count
        total_correct += int((task_logits.argmax(dim=1) == targets).sum())
        total_samples += count
        if batch_index % log_interval == 0 or batch_index == len(loader):
            print(
                f"[{prefix}] epoch={epoch}/{total_epochs} batch={batch_index}/{len(loader)} "
                f"loss={total_loss/max(total_samples,1):.5f} "
                f"acc={total_correct/max(total_samples,1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate(
    model: OpticalClassifier,
    loader: DataLoader,
    device: torch.device,
    selected_class_indices: tuple[int, ...] | None,
    *,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        task_logits, targets = _selected_logits(logits, batch, selected_class_indices)
        loss = F.cross_entropy(task_logits, targets, reduction="sum")
        predictions = task_logits.argmax(dim=1)
        total_loss += float(loss)
        total_correct += int((predictions == targets).sum())
        total_samples += int(targets.numel())
        np.add.at(confusion, (targets.cpu().numpy(), predictions.cpu().numpy()), 1)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": total_samples,
        "confusion": confusion,
    }


def _parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _parameter_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _flatten_delta(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    names = sorted(reference)
    delta = torch.cat([(current[name].float() - reference[name].float()).reshape(-1) for name in names])
    initial = torch.cat([reference[name].float().reshape(-1) for name in names])
    return delta, initial


def endpoint_metrics(model: OpticalClassifier, initial_parameters: dict[str, torch.Tensor], initial_phases: torch.Tensor) -> dict[str, float]:
    current_parameters = _parameter_state(model)
    delta, initial = _flatten_delta(current_parameters, initial_parameters)
    phases = model.snapshot_feedback_phases()
    circular = torch.angle(torch.exp(1j * (phases - initial_phases)))
    return {
        "relative_parameter_drift": float(delta.norm() / initial.norm().clamp_min(1e-12)),
        "endpoint_update_norm": float(delta.norm()),
        "phase_circular_rms_rad": float(circular.square().mean().sqrt()),
        "phase_phasor_drift": float(phasor_operator_distance(phases, initial_phases)),
        "phase_operator_coherence": float(phasor_operator_coherence(phases, initial_phases)),
    }


def _gradient_diagnostic(
    model: OpticalClassifier,
    batch: dict[str, Any],
    device: torch.device,
    selected_class_indices: tuple[int, ...],
    *,
    method: Method,
    pretrained_phases: torch.Tensor,
    random_seed: int,
) -> list[dict[str, float]]:
    if method == "no_finetune":
        return []
    images = batch["image"].to(device)

    def gradients(mode: str) -> list[torch.Tensor]:
        model.configure_feedback(
            mode, pretrained_phases=pretrained_phases if mode == "fa_pretrained" else None, random_seed=random_seed
        )
        model.zero_grad(set_to_none=True)
        logits = model(images)
        task_logits, targets = _selected_logits(logits, batch, selected_class_indices)
        F.cross_entropy(task_logits, targets).backward()
        result = [stage.raw_phase.grad.detach().float().cpu().clone() for stage in model.stages]
        model.zero_grad(set_to_none=True)
        return result

    method_gradients = gradients(method)
    bp_gradients = gradients("bp")
    model.configure_feedback(
        method, pretrained_phases=pretrained_phases if method == "fa_pretrained" else None, random_seed=random_seed
    )
    rows: list[dict[str, float]] = []
    for index, (approximate, exact) in enumerate(zip(method_gradients, bp_gradients, strict=True)):
        approximate_flat = approximate.flatten()
        exact_flat = exact.flatten()
        rows.append(
            {
                "stage": float(index + 1),
                "gradient_cosine_to_bp": float(
                    F.cosine_similarity(approximate_flat, exact_flat, dim=0, eps=1e-12)
                ),
                "gradient_norm": float(approximate_flat.norm()),
                "bp_gradient_norm": float(exact_flat.norm()),
                "gradient_norm_ratio": float(
                    approximate_flat.norm() / exact_flat.norm().clamp_min(1e-12)
                ),
                "relative_gradient_error": float(
                    (approximate_flat - exact_flat).norm() / exact_flat.norm().clamp_min(1e-12)
                ),
            }
        )
    return rows


def pretrain(settings: Settings, bundle: DatasetBundle, device: torch.device, *, force: bool = False) -> Path:
    run_dir = settings.output_dir / "pretrain"
    best_path = run_dir / "checkpoints" / "pretrained_best_validation.pt"
    complete_path = run_dir / "complete.json"
    if best_path.exists() and complete_path.exists() and not force:
        print(f"[pretrain] reusing {best_path}", flush=True)
        return best_path
    set_seed(settings.training.pretrain_seed)
    model = build_model(settings, device)
    model.configure_feedback("bp")
    optimizer = build_optimizer(model, settings)
    history_path = run_dir / "training_history.csv"
    last_path = run_dir / "checkpoints" / "pretrained_last.pt"
    if force and history_path.exists():
        history_path.unlink()
    validation_loader = _loader(
        bundle.pretrain_validation,
        None,
        batch_size=settings.training.pretrain_batch_size,
        settings=settings,
    )
    initial_parameters = _parameter_state(model)
    start_epoch = 1
    best_accuracy = -1.0
    if last_path.exists() and not force:
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if resume.get("settings_digest") != settings.digest():
            raise RuntimeError("Existing pretraining checkpoint uses a different resolved configuration")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"]) + 1
        if best_path.exists():
            best_accuracy = float(
                torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["validation_accuracy"]
            )
        print(f"[pretrain] resuming at epoch {start_epoch}", flush=True)
    if start_epoch > settings.training.pretrain_epochs and best_path.exists():
        complete_path.write_text(json.dumps({"epochs": settings.training.pretrain_epochs}), encoding="utf-8")
        return best_path
    for epoch in range(start_epoch, settings.training.pretrain_epochs + 1):
        bundle.pretrain_train.set_epoch(epoch)
        order = epoch_order(
            len(bundle.pretrain_train),
            epoch=epoch,
            seed=settings.training.pretrain_seed,
            limit=settings.data.pretrain_samples_per_epoch,
        )
        loader = _loader(
            bundle.pretrain_train,
            order,
            batch_size=settings.training.pretrain_batch_size,
            settings=settings,
        )
        train_metrics = train_epoch(
            model,
            loader,
            optimizer,
            device,
            None,
            epoch=epoch,
            total_epochs=settings.training.pretrain_epochs,
            log_interval=settings.training.log_interval_batches,
            prefix="pretrain-bp",
        )
        validation = evaluate(model, validation_loader, device, None, num_classes=100)
        endpoint = endpoint_metrics(model, initial_parameters, torch.full_like(model.snapshot_feedback_phases(), torch.pi))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "samples_this_epoch": int(train_metrics["samples"]),
            "epoch_seconds": train_metrics["seconds"],
            "batch_order_sha256": order_digest(order),
            **endpoint,
        }
        _write_csv_row(history_path, row)
        checkpoint = {
            "kind": "pretrained",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "feedback_phases": model.snapshot_feedback_phases(),
            "parameters": _parameter_state(model),
            "parameter_digest": _parameter_digest(_parameter_state(model)),
            "settings_digest": settings.digest(),
            "metrics": {key: value for key, value in row.items() if key != "batch_order_sha256"},
        }
        _atomic_torch_save(checkpoint, last_path)
        if epoch % settings.training.checkpoint_interval_epochs == 0:
            _atomic_torch_save(checkpoint, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            _atomic_torch_save(checkpoint, best_path)
        print(
            f"[pretrain-bp] epoch={epoch} complete train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={validation['accuracy']:.4f} best={best_accuracy:.4f} "
            f"phase_drift={endpoint['phase_circular_rms_rad']:.4f}rad",
            flush=True,
        )
    save_history_plot(history_path, run_dir / "figures" / "training_curves.png", "CIFAR-100 BP pretraining")
    save_phase_overview(
        model.snapshot_feedback_phases(),
        model.residual_weights(),
        run_dir / "figures" / "pretrained_phase_masks.png",
        title="Pretrained phase masks and residual weights",
    )
    complete_path.write_text(
        json.dumps({"epochs": settings.training.pretrain_epochs, "best_validation_accuracy": best_accuracy}, indent=2),
        encoding="utf-8",
    )
    return best_path


def load_pretrained(model: OpticalClassifier, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def finetune(
    settings: Settings,
    bundle: DatasetBundle,
    device: torch.device,
    *,
    method: Method,
    seed: int,
    force: bool = False,
) -> Path:
    if method not in {"bp", "fa_pretrained", "fa_random"}:
        raise ValueError(f"Cannot fine-tune method {method}")
    pretrained_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    run_dir = settings.output_dir / "finetune" / method / f"seed_{seed}"
    best_path = run_dir / "checkpoints" / "best_validation.pt"
    complete_path = run_dir / "complete.json"
    if best_path.exists() and complete_path.exists() and not force:
        print(f"[{method}] reusing {best_path}", flush=True)
        return best_path
    set_seed(seed)
    model = build_model(settings, device)
    checkpoint = load_pretrained(model, pretrained_path, device)
    pretrained_phases = checkpoint["feedback_phases"].detach().cpu()
    initial_parameters = _parameter_state(model)
    initial_digest = _parameter_digest(initial_parameters)
    model.configure_feedback(
        method,
        pretrained_phases=pretrained_phases if method == "fa_pretrained" else None,
        random_seed=seed,
    )
    optimizer = build_optimizer(model, settings)
    history_path = run_dir / "training_history.csv"
    diagnostics_path = run_dir / "gradient_diagnostics.csv"
    for path in (history_path, diagnostics_path):
        if force and path.exists():
            path.unlink()
    validation_loader = _loader(
        bundle.finetune_validation,
        None,
        batch_size=settings.training.finetune_batch_size,
        settings=settings,
    )
    test_loader = _loader(
        bundle.finetune_test,
        None,
        batch_size=settings.training.finetune_batch_size,
        settings=settings,
    )
    diagnostic_loader = _loader(
        bundle.finetune_train,
        list(range(min(settings.training.finetune_batch_size, len(bundle.finetune_train)))),
        batch_size=settings.training.finetune_batch_size,
        settings=settings,
    )
    diagnostic_batch = next(iter(diagnostic_loader))
    best_accuracy = -1.0
    start_epoch = 1
    last_path = run_dir / "checkpoints" / "last.pt"
    if last_path.exists() and not force:
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if resume.get("settings_digest") != settings.digest():
            raise RuntimeError(f"Existing {method} checkpoint uses a different resolved configuration")
        if resume.get("initial_parameter_digest") != initial_digest:
            raise RuntimeError(f"Existing {method} checkpoint uses a different pretrained initialization")
        model.load_state_dict(resume["model_state"], strict=True)
        model.configure_feedback(
            method,
            pretrained_phases=pretrained_phases if method == "fa_pretrained" else None,
            random_seed=seed,
        )
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"]) + 1
        if best_path.exists():
            best_accuracy = float(
                torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["validation_accuracy"]
            )
        print(f"[{method} seed={seed}] resuming at epoch {start_epoch}", flush=True)
    if start_epoch == 1 and 0 in settings.training.diagnostic_epochs:
        for row in _gradient_diagnostic(
            model,
            diagnostic_batch,
            device,
            bundle.selected_class_indices,
            method=method,
            pretrained_phases=pretrained_phases,
            random_seed=seed,
        ):
            _write_csv_row(diagnostics_path, {"epoch": 0, "method": method, **row})
    if start_epoch > settings.training.finetune_epochs and best_path.exists():
        complete_path.write_text(json.dumps({"epochs": settings.training.finetune_epochs}), encoding="utf-8")
        return best_path
    for epoch in range(start_epoch, settings.training.finetune_epochs + 1):
        bundle.finetune_train.set_epoch(epoch)
        order = epoch_order(len(bundle.finetune_train), epoch=epoch, seed=seed, limit=None)
        loader = _loader(
            bundle.finetune_train,
            order,
            batch_size=settings.training.finetune_batch_size,
            settings=settings,
        )
        train_metrics = train_epoch(
            model,
            loader,
            optimizer,
            device,
            bundle.selected_class_indices,
            epoch=epoch,
            total_epochs=settings.training.finetune_epochs,
            log_interval=settings.training.log_interval_batches,
            prefix=f"finetune-{method}-seed{seed}",
        )
        validation = evaluate(model, validation_loader, device, bundle.selected_class_indices, num_classes=10)
        test = evaluate(model, test_loader, device, bundle.selected_class_indices, num_classes=10)
        endpoint = endpoint_metrics(model, initial_parameters, pretrained_phases)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "samples_this_epoch": int(train_metrics["samples"]),
            "epoch_seconds": train_metrics["seconds"],
            "batch_order_sha256": order_digest(order),
            **endpoint,
        }
        _write_csv_row(history_path, row)
        if epoch in settings.training.diagnostic_epochs:
            for diagnostic in _gradient_diagnostic(
                model,
                diagnostic_batch,
                device,
                bundle.selected_class_indices,
                method=method,
                pretrained_phases=pretrained_phases,
                random_seed=seed,
            ):
                _write_csv_row(diagnostics_path, {"epoch": epoch, "method": method, **diagnostic})
        payload = {
            "kind": "finetuned",
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "parameters": _parameter_state(model),
            "feedback_phases": pretrained_phases,
            "initial_parameter_digest": initial_digest,
            "settings_digest": settings.digest(),
            "metrics": row,
        }
        _atomic_torch_save(payload, last_path)
        if epoch % settings.training.checkpoint_interval_epochs == 0:
            _atomic_torch_save(payload, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            _atomic_torch_save(payload, best_path)
            save_confusion_matrix(
                test["confusion"],
                list(settings.data.selected_classes),
                run_dir / "figures" / "best_validation_test_confusion.png",
                f"{method} seed {seed}: test at best validation",
            )
        print(
            f"[finetune-{method}-seed{seed}] epoch={epoch} complete "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={validation['accuracy']:.4f} "
            f"test_acc={test['accuracy']:.4f} best_val={best_accuracy:.4f} "
            f"phase_drift={endpoint['phase_circular_rms_rad']:.4f}rad",
            flush=True,
        )
    save_history_plot(history_path, run_dir / "figures" / "training_curves.png", f"{method}, seed {seed}")
    save_phase_overview(
        model.snapshot_feedback_phases(),
        model.residual_weights(),
        run_dir / "figures" / "final_phase_masks.png",
        title=f"{method} final phase masks, seed {seed}",
    )
    example = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        _, intermediates = model(example["image"][:1].to(device), return_intermediates=True)
    save_optical_example(
        intermediates,
        run_dir / "figures" / "optical_stage_examples.png",
        title=f"{method} seed {seed}: CCD/LN/reload examples",
    )
    summary = {
        "method": method,
        "seed": seed,
        "best_validation_accuracy": best_accuracy,
        "last_metrics": row,
        "pretrained_checkpoint": str(pretrained_path),
        "initial_parameter_digest": initial_digest,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete_path.write_text(
        json.dumps({"epochs": settings.training.finetune_epochs, "best_validation_accuracy": best_accuracy}, indent=2),
        encoding="utf-8",
    )
    return best_path


def evaluate_no_finetuning(settings: Settings, bundle: DatasetBundle, device: torch.device) -> Path:
    run_dir = settings.output_dir / "finetune" / "no_finetune"
    checkpoint_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    model = build_model(settings, device)
    checkpoint = load_pretrained(model, checkpoint_path, device)
    model.configure_feedback("bp")
    loader = _loader(
        bundle.finetune_test,
        None,
        batch_size=settings.training.finetune_batch_size,
        settings=settings,
    )
    metrics = evaluate(model, loader, device, bundle.selected_class_indices, num_classes=10)
    payload = {
        "method": "no_finetune",
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_samples": metrics["samples"],
        "relative_parameter_drift": 0.0,
        "endpoint_update_norm": 0.0,
        "phase_circular_rms_rad": 0.0,
        "phase_phasor_drift": 0.0,
        "phase_operator_coherence": 1.0,
        "pretrained_checkpoint": str(checkpoint_path),
        "parameter_digest": checkpoint["parameter_digest"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_confusion_matrix(
        metrics["confusion"],
        list(settings.data.selected_classes),
        run_dir / "figures" / "test_confusion.png",
        "No fine-tuning: CIFAR-100-C ten-class test",
    )
    print(
        f"[no-finetune] test_loss={metrics['loss']:.5f} test_accuracy={metrics['accuracy']:.4f}",
        flush=True,
    )
    return output
