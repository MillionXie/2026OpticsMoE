from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .datasets import BalancedClassBatchSampler, DatasetBundle, batch_order_digest
from .losses import contrastive_transfer_loss
from .model import OpticalEmbeddingNetwork
from .optics import phasor_operator_coherence, phasor_operator_distance
from .settings import BalancedBatchConfig, Settings


Method = Literal["bp", "fa_pretrained", "fa_random", "no_finetune"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_model(settings: Settings, device: torch.device) -> OpticalEmbeddingNetwork:
    return OpticalEmbeddingNetwork(settings.optical).to(device)


def build_optimizer(model: OpticalEmbeddingNetwork, settings: Settings, *, finetune: bool) -> torch.optim.Optimizer:
    phase_lr = (
        settings.optimizer.finetune_phase_learning_rate
        if finetune
        else settings.optimizer.pretrain_phase_learning_rate
    )
    return torch.optim.AdamW(
        [
            {"params": list(model.phase_parameters()), "lr": phase_lr, "name": "phase"},
            {
                "params": list(model.electronic_parameters()),
                "lr": settings.optimizer.electronic_learning_rate,
                "name": "electronic",
            },
        ],
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
        weight_decay=settings.optimizer.weight_decay,
    )


def _balanced_loader(
    dataset: Any,
    config: BalancedBatchConfig,
    settings: Settings,
    *,
    seed: int,
    epoch: int,
    batches_override: int | None = None,
) -> tuple[DataLoader, BalancedClassBatchSampler]:
    batch_config = config
    if batches_override is not None:
        batch_config = BalancedBatchConfig(
            classes_per_batch=config.classes_per_batch,
            images_per_class=config.images_per_class,
            views_per_image=config.views_per_image,
            batches_per_epoch=int(batches_override),
        )
    sampler = BalancedClassBatchSampler(dataset.targets, batch_config, seed=seed, epoch=epoch)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=settings.data.num_workers,
        pin_memory=settings.data.pin_memory,
        persistent_workers=False,
    )
    return loader, sampler


def _plain_loader(dataset: Any, settings: Settings) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings.training.evaluation_batch_size,
        shuffle=False,
        num_workers=settings.data.num_workers,
        pin_memory=settings.data.pin_memory,
        persistent_workers=False,
    )


def _forward_views(model: OpticalEmbeddingNetwork, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    views = batch["views"].to(device, non_blocking=True)
    if views.ndim != 5:
        raise ValueError(f"Expected views [B,V,1,32,32], got {tuple(views.shape)}")
    batch_size, view_count = views.shape[:2]
    embeddings = model(views.reshape(batch_size * view_count, *views.shape[2:]))
    embeddings = embeddings.reshape(batch_size, view_count, -1)
    labels = batch["target"].to(device, non_blocking=True).long()
    return embeddings, labels


def _loss(
    settings: Settings,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    pretrain: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return contrastive_transfer_loss(
        embeddings,
        labels,
        contrastive_temperature=settings.loss.contrastive_temperature,
        prototype_temperature=settings.loss.prototype_temperature,
        supcon_weight=(settings.loss.pretrain_supcon_weight if pretrain else settings.loss.finetune_supcon_weight),
        prototype_weight=(0.0 if pretrain else settings.loss.finetune_prototype_weight),
    )


def train_epoch(
    model: OpticalEmbeddingNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    settings: Settings,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    prefix: str,
    pretrain: bool,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "supcon": 0.0, "prototype": 0.0, "accuracy": 0.0}
    samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        embeddings, labels = _forward_views(model, batch, device)
        loss, parts = _loss(settings, embeddings, labels, pretrain=pretrain)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {prefix} epoch={epoch} batch={batch_index}")
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        totals["loss"] += float(loss.detach()) * count
        totals["supcon"] += float(parts["supcon"].detach()) * count
        totals["prototype"] += float(parts["prototype"].detach()) * count
        totals["accuracy"] += float(parts["batch_prototype_accuracy"].detach()) * count
        samples += count
        if batch_index % settings.training.log_interval_batches == 0 or batch_index == len(loader):
            residual = model.residual_weights().detach().mean(dim=0)
            print(
                f"[{prefix}] epoch={epoch}/{total_epochs} batch={batch_index}/{len(loader)} "
                f"loss={totals['loss']/samples:.5f} supcon={totals['supcon']/samples:.5f} "
                f"proto={totals['prototype']/samples:.5f} batch_proto_acc={totals['accuracy']/samples:.4f} "
                f"residual_optical/skip={float(residual[0]):.3f}/{float(residual[1]):.3f}",
                flush=True,
            )
    return {key: value / max(samples, 1) for key, value in totals.items()} | {
        "samples": float(samples),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_contrastive(
    model: OpticalEmbeddingNetwork,
    loader: DataLoader,
    settings: Settings,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    samples = 0
    for batch in loader:
        embeddings, labels = _forward_views(model, batch, device)
        loss, parts = _loss(settings, embeddings, labels, pretrain=True)
        count = int(labels.numel())
        total_loss += float(loss) * count
        total_accuracy += float(parts["batch_prototype_accuracy"]) * count
        samples += count
    return {"loss": total_loss / max(samples, 1), "batch_prototype_accuracy": total_accuracy / max(samples, 1)}


@torch.no_grad()
def compute_prototypes(
    model: OpticalEmbeddingNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    sums = torch.zeros(num_classes, model.config.embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for batch in loader:
        images = batch["views"][:, 0].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True).long()
        embeddings = model(images)
        sums.index_add_(0, labels, embeddings)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
    if torch.any(counts == 0):
        raise RuntimeError("At least one class has no prototype support")
    return F.normalize(sums / counts[:, None], dim=-1, eps=1e-12)


@torch.no_grad()
def evaluate_prototypes(
    model: OpticalEmbeddingNetwork,
    prototypes: torch.Tensor,
    loader: DataLoader,
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((len(prototypes), len(prototypes)), dtype=np.int64)
    total_loss = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        images = batch["views"][:, 0].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True).long()
        embeddings = model(images)
        logits = embeddings @ prototypes.T / settings.loss.prototype_temperature
        predictions = logits.argmax(dim=1)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((predictions == labels).sum())
        samples += int(labels.numel())
        np.add.at(confusion, (labels.cpu().numpy(), predictions.cpu().numpy()), 1)
    return {
        "loss": total_loss / max(samples, 1),
        "accuracy": correct / max(samples, 1),
        "samples": samples,
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


def endpoint_metrics(
    model: OpticalEmbeddingNetwork,
    initial_parameters: dict[str, torch.Tensor],
    initial_phases: torch.Tensor,
) -> dict[str, float]:
    current = _parameter_state(model)
    names = sorted(initial_parameters)
    delta = torch.cat([(current[name].float() - initial_parameters[name].float()).reshape(-1) for name in names])
    initial = torch.cat([initial_parameters[name].float().reshape(-1) for name in names])
    phases = model.snapshot_feedback_phases()
    circular = torch.angle(torch.exp(1j * (phases - initial_phases)))
    weights = model.residual_weights().detach().cpu()
    return {
        "relative_parameter_drift": float(delta.norm() / initial.norm().clamp_min(1e-12)),
        "endpoint_update_norm": float(delta.norm()),
        "phase_circular_rms_rad": float(circular.square().mean().sqrt()),
        "phase_phasor_drift": float(phasor_operator_distance(phases, initial_phases)),
        "phase_operator_coherence": float(phasor_operator_coherence(phases, initial_phases)),
        "residual_optical_weight_mean": float(weights[:, 0].mean()),
        "residual_optical_weight_min": float(weights[:, 0].min()),
        "residual_optical_weight_max": float(weights[:, 0].max()),
    }


def _checkpoint_payload(
    model: OpticalEmbeddingNetwork,
    optimizer: torch.optim.Optimizer,
    settings: Settings,
    *,
    kind: str,
    method: str,
    seed: int,
    epoch: int,
    metrics: dict[str, Any],
    initial_digest: str,
    feedback_phases: torch.Tensor,
) -> dict[str, Any]:
    parameters = _parameter_state(model)
    return {
        "kind": kind,
        "method": method,
        "seed": seed,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "parameters": parameters,
        "parameter_digest": _parameter_digest(parameters),
        "initial_parameter_digest": initial_digest,
        "feedback_phases": feedback_phases,
        "settings_digest": settings.digest(),
        "metrics": metrics,
    }


def pretrain(settings: Settings, bundle: DatasetBundle, device: torch.device, *, force: bool = False) -> Path:
    run_dir = settings.output_dir / "pretrain"
    best_path = run_dir / "checkpoints" / "pretrained_best_validation.pt"
    last_path = run_dir / "checkpoints" / "pretrained_last.pt"
    complete_path = run_dir / "complete.json"
    if best_path.exists() and complete_path.exists() and not force:
        print(f"[pretrain] reusing {best_path}", flush=True)
        return best_path
    set_seed(settings.training.pretrain_seed)
    model = build_model(settings, device)
    model.configure_feedback("bp")
    optimizer = build_optimizer(model, settings, finetune=False)
    initial_parameters = _parameter_state(model)
    initial_digest = _parameter_digest(initial_parameters)
    initial_phases = model.snapshot_feedback_phases()
    history_path = run_dir / "training_history.csv"
    if force and history_path.exists():
        history_path.unlink()
    start_epoch = 1
    best_loss = float("inf")
    if last_path.exists() and not force:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["settings_digest"] != settings.digest():
            raise RuntimeError("Pretraining checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best_loss = float(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["validation_loss"])
    for epoch in range(start_epoch, settings.training.pretrain_epochs + 1):
        bundle.pretrain_train.set_epoch(epoch)
        loader, sampler = _balanced_loader(
            bundle.pretrain_train,
            settings.training.pretrain_batch,
            settings,
            seed=settings.training.pretrain_seed,
            epoch=epoch,
        )
        metrics = train_epoch(
            model,
            loader,
            optimizer,
            settings,
            device,
            epoch=epoch,
            total_epochs=settings.training.pretrain_epochs,
            prefix="pretrain-bp",
            pretrain=True,
        )
        bundle.pretrain_validation.set_epoch(0)
        validation_loader, _ = _balanced_loader(
            bundle.pretrain_validation,
            settings.training.pretrain_batch,
            settings,
            seed=settings.training.pretrain_seed + 99,
            epoch=0,
            batches_override=settings.training.validation_batches,
        )
        validation = evaluate_contrastive(model, validation_loader, settings, device)
        endpoint = endpoint_metrics(model, initial_parameters, initial_phases)
        row = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "train_supcon": metrics["supcon"],
            "train_batch_prototype_accuracy": metrics["accuracy"],
            "validation_loss": validation["loss"],
            "validation_batch_prototype_accuracy": validation["batch_prototype_accuracy"],
            "samples_this_epoch": int(metrics["samples"]),
            "epoch_seconds": metrics["seconds"],
            "batch_order_sha256": batch_order_digest(sampler),
            **endpoint,
        }
        _write_row(history_path, row)
        payload = _checkpoint_payload(
            model,
            optimizer,
            settings,
            kind="contrastive_pretrained",
            method="bp",
            seed=settings.training.pretrain_seed,
            epoch=epoch,
            metrics=row,
            initial_digest=initial_digest,
            feedback_phases=model.snapshot_feedback_phases(),
        )
        _atomic_save(payload, last_path)
        if epoch % settings.training.checkpoint_interval_epochs == 0:
            _atomic_save(payload, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation["loss"] < best_loss:
            best_loss = float(validation["loss"])
            _atomic_save(payload, best_path)
        print(
            f"[pretrain-bp] epoch={epoch} complete val_supcon={validation['loss']:.5f} "
            f"val_batch_proto_acc={validation['batch_prototype_accuracy']:.4f} best_loss={best_loss:.5f}",
            flush=True,
        )
    complete_path.write_text(json.dumps({"epochs": settings.training.pretrain_epochs, "best_validation_loss": best_loss}, indent=2), encoding="utf-8")
    return best_path


def _load_pretrained(model: OpticalEmbeddingNetwork, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def _evaluate_downstream(model: OpticalEmbeddingNetwork, bundle: DatasetBundle, settings: Settings, device: torch.device, split: str) -> dict[str, Any]:
    prototypes = compute_prototypes(model, _plain_loader(bundle.prototype_support, settings), device, num_classes=10)
    dataset = bundle.finetune_validation if split == "validation" else bundle.finetune_test
    return evaluate_prototypes(model, prototypes, _plain_loader(dataset, settings), settings, device)


def _gradient_diagnostic(
    model: OpticalEmbeddingNetwork,
    batch: dict[str, Any],
    settings: Settings,
    device: torch.device,
    *,
    method: str,
    pretrained_phases: torch.Tensor,
    random_seed: int,
) -> list[dict[str, float]]:
    def gradients(mode: str) -> list[torch.Tensor]:
        model.configure_feedback(
            mode, pretrained_phases=pretrained_phases if mode == "fa_pretrained" else None, random_seed=random_seed
        )
        model.zero_grad(set_to_none=True)
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20260815)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(20260815)
            embeddings, labels = _forward_views(model, batch, device)
            loss, _ = _loss(settings, embeddings, labels, pretrain=False)
            loss.backward()
        values = [stage.raw_phase.grad.detach().float().cpu().clone() for stage in model.stages]
        model.zero_grad(set_to_none=True)
        return values

    approximate = gradients(method)
    exact = gradients("bp")
    model.configure_feedback(
        method, pretrained_phases=pretrained_phases if method == "fa_pretrained" else None, random_seed=random_seed
    )
    rows = []
    for stage, (current, reference) in enumerate(zip(approximate, exact, strict=True), start=1):
        current_flat, reference_flat = current.flatten(), reference.flatten()
        rows.append(
            {
                "stage": float(stage),
                "gradient_cosine_to_bp": float(F.cosine_similarity(current_flat, reference_flat, dim=0, eps=1e-12)),
                "gradient_norm_ratio": float(current_flat.norm() / reference_flat.norm().clamp_min(1e-12)),
                "relative_gradient_error": float(
                    (current_flat - reference_flat).norm() / reference_flat.norm().clamp_min(1e-12)
                ),
            }
        )
    return rows


def finetune(
    settings: Settings,
    bundle: DatasetBundle,
    device: torch.device,
    *,
    method: Literal["bp", "fa_pretrained", "fa_random"],
    seed: int,
    force: bool = False,
) -> Path:
    pretrained_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    run_dir = settings.output_dir / "finetune" / method / f"seed_{seed}"
    best_path = run_dir / "checkpoints" / "best_validation.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    complete_path = run_dir / "complete.json"
    if best_path.exists() and complete_path.exists() and not force:
        print(f"[{method} seed={seed}] reusing completed run", flush=True)
        return best_path
    set_seed(seed)
    model = build_model(settings, device)
    pretrained = _load_pretrained(model, pretrained_path, device)
    pretrained_phases = pretrained["feedback_phases"].detach().cpu()
    initial_parameters = _parameter_state(model)
    initial_digest = _parameter_digest(initial_parameters)
    model.configure_feedback(
        method, pretrained_phases=pretrained_phases if method == "fa_pretrained" else None, random_seed=seed
    )
    optimizer = build_optimizer(model, settings, finetune=True)
    history_path = run_dir / "training_history.csv"
    diagnostics_path = run_dir / "gradient_diagnostics.csv"
    if force:
        for path in (history_path, diagnostics_path):
            if path.exists():
                path.unlink()
    best_accuracy = -1.0
    start_epoch = 1
    if last_path.exists() and not force:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["settings_digest"] != settings.digest() or checkpoint["initial_parameter_digest"] != initial_digest:
            raise RuntimeError("Fine-tuning checkpoint control mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.configure_feedback(
            method, pretrained_phases=pretrained_phases if method == "fa_pretrained" else None, random_seed=seed
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best_accuracy = float(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["validation_accuracy"])
    diagnostic_loader, _ = _balanced_loader(
        bundle.finetune_train,
        settings.training.finetune_batch,
        settings,
        seed=seed,
        epoch=0,
        batches_override=1,
    )
    diagnostic_batch = next(iter(diagnostic_loader))
    if start_epoch == 1 and 0 in settings.training.diagnostic_epochs:
        for item in _gradient_diagnostic(
            model,
            diagnostic_batch,
            settings,
            device,
            method=method,
            pretrained_phases=pretrained_phases,
            random_seed=seed,
        ):
            _write_row(diagnostics_path, {"epoch": 0, "method": method, **item})
    for epoch in range(start_epoch, settings.training.finetune_epochs + 1):
        bundle.finetune_train.set_epoch(epoch)
        loader, sampler = _balanced_loader(
            bundle.finetune_train,
            settings.training.finetune_batch,
            settings,
            seed=seed,
            epoch=epoch,
        )
        train = train_epoch(
            model,
            loader,
            optimizer,
            settings,
            device,
            epoch=epoch,
            total_epochs=settings.training.finetune_epochs,
            prefix=f"finetune-{method}-seed{seed}",
            pretrain=False,
        )
        validation = _evaluate_downstream(model, bundle, settings, device, "validation")
        endpoint = endpoint_metrics(model, initial_parameters, pretrained_phases)
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_supcon": train["supcon"],
            "train_prototype_loss": train["prototype"],
            "train_batch_prototype_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "samples_this_epoch": int(train["samples"]),
            "epoch_seconds": train["seconds"],
            "batch_order_sha256": batch_order_digest(sampler),
            **endpoint,
        }
        _write_row(history_path, row)
        if epoch in settings.training.diagnostic_epochs:
            for item in _gradient_diagnostic(
                model,
                diagnostic_batch,
                settings,
                device,
                method=method,
                pretrained_phases=pretrained_phases,
                random_seed=seed,
            ):
                _write_row(diagnostics_path, {"epoch": epoch, "method": method, **item})
        payload = _checkpoint_payload(
            model,
            optimizer,
            settings,
            kind="contrastive_finetuned",
            method=method,
            seed=seed,
            epoch=epoch,
            metrics=row,
            initial_digest=initial_digest,
            feedback_phases=pretrained_phases,
        )
        _atomic_save(payload, last_path)
        if epoch % settings.training.checkpoint_interval_epochs == 0:
            _atomic_save(payload, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            _atomic_save(payload, best_path)
        print(
            f"[finetune-{method}-seed{seed}] epoch={epoch} complete val_acc={validation['accuracy']:.4f} "
            f"best={best_accuracy:.4f} drift={endpoint['relative_parameter_drift']:.4f} "
            f"residual_optical={endpoint['residual_optical_weight_mean']:.3f}",
            flush=True,
        )

    def test_checkpoint(path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        result = _evaluate_downstream(model, bundle, settings, device, "test")
        return {"epoch": int(checkpoint["epoch"]), "test_accuracy": result["accuracy"], "test_loss": result["loss"]}

    fixed_endpoint = test_checkpoint(last_path)
    validation_selected = test_checkpoint(best_path)
    summary = {
        "method": method,
        "seed": seed,
        "best_validation_accuracy": best_accuracy,
        "fixed_endpoint": fixed_endpoint,
        "validation_selected": validation_selected,
        "test_used_for_selection": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_path


def evaluate_no_finetuning(settings: Settings, bundle: DatasetBundle, device: torch.device) -> Path:
    run_dir = settings.output_dir / "finetune" / "no_finetune"
    model = build_model(settings, device)
    pretrained_path = settings.output_dir / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    checkpoint = _load_pretrained(model, pretrained_path, device)
    model.configure_feedback("bp")
    validation = _evaluate_downstream(model, bundle, settings, device, "validation")
    test = _evaluate_downstream(model, bundle, settings, device, "test")
    payload = {
        "method": "no_finetune",
        "pretrained_epoch": int(checkpoint["epoch"]),
        "validation_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "test_used_for_selection": False,
        "prototype_support_images": len(bundle.prototype_support),
        "relative_parameter_drift": 0.0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
