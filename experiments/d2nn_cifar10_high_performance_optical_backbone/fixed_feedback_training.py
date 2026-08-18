from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from .datasets import DatasetBundle, make_loader
from .formal_settings import FormalSettings
from .model import OpticalClassifier
from .training import build_model, evaluate, set_seed


Method = Literal["noft", "bp", "fa_pretrained", "fa_random"]
METHODS: tuple[Method, ...] = ("noft", "bp", "fa_pretrained", "fa_random")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_source_backbone(model: OpticalClassifier, settings: FormalSettings) -> dict[str, Any]:
    actual = sha256_file(settings.formal.source_checkpoint)
    if actual != settings.formal.source_checkpoint_sha256:
        raise RuntimeError(f"Source checkpoint SHA-256 mismatch: expected {settings.formal.source_checkpoint_sha256}, got {actual}")
    payload = torch.load(settings.formal.source_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    backbone = {name: value for name, value in state.items() if name.startswith("stages.")}
    result = model.load_state_dict(backbone, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected source backbone keys: {result.unexpected_keys}")
    return payload


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    minimum: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(epochs - warmup_epochs - 1, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _train_epoch(
    model: OpticalClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: FormalSettings,
    *,
    epoch: int,
    total_epochs: int,
    prefix: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    use_amp = settings.base.training.use_amp and device.type == "cuda"
    maximum = settings.base.training.max_train_batches
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if maximum is not None and batch_index > maximum:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=settings.base.training.label_smoothing,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {prefix} epoch={epoch} batch={batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.base.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        count = int(targets.numel())
        total_loss += float(loss.detach()) * count
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
        total_samples += count
        if batch_index % settings.base.training.log_interval_batches == 0 or batch_index == len(loader):
            print(
                f"[{prefix}] epoch={epoch}/{total_epochs} batch={batch_index}/{len(loader)} "
                f"loss={total_loss/max(total_samples,1):.5f} acc={total_correct/max(total_samples,1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "seconds": time.perf_counter() - started,
    }


def prepare_common_checkpoint(
    settings: FormalSettings,
    datasets: DatasetBundle,
    device: torch.device,
    *,
    force: bool = False,
) -> dict[str, object]:
    output = settings.formal.common_checkpoint
    if output.exists() and not force:
        payload = torch.load(output, map_location="cpu", weights_only=False)
        print(f"Reusing common checkpoint {output} sha256={sha256_file(output)}", flush=True)
        return payload
    set_seed(settings.formal.head_warmup_seed)
    model = build_model(settings.base, device)
    _load_source_backbone(model, settings)
    source_phases = model.snapshot_phases()
    for parameter in model.stages.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=settings.formal.head_learning_rate,
        betas=settings.base.optimizer.betas,
        eps=settings.base.optimizer.eps,
        weight_decay=settings.base.optimizer.weight_decay,
    )
    scheduler = _cosine_scheduler(
        optimizer,
        epochs=settings.formal.head_warmup_epochs,
        warmup_epochs=min(1, settings.formal.head_warmup_epochs),
        minimum=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=settings.base.training.use_amp and device.type == "cuda")
    train_loader = make_loader(datasets.train, settings.base, train=True, seed=settings.formal.head_warmup_seed)
    validation_loader = make_loader(
        datasets.validation,
        settings.base,
        train=False,
        seed=settings.formal.head_warmup_seed + 1,
    )
    best = -1.0
    history: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    for epoch in range(1, settings.formal.head_warmup_epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            settings,
            epoch=epoch,
            total_epochs=settings.formal.head_warmup_epochs,
            prefix="head_warmup",
        )
        validation = evaluate(model, validation_loader, device, settings.base)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "seconds": train_metrics["seconds"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if float(validation["accuracy"]) > best:
            best = float(validation["accuracy"])
            best_payload = {
                "model": model.state_dict(),
                "source_phases": source_phases,
                "selected_epoch": epoch,
                "best_validation_accuracy": best,
                "source_checkpoint": str(settings.formal.source_checkpoint),
                "source_checkpoint_sha256": settings.formal.source_checkpoint_sha256,
                "head_warmup_seed": settings.formal.head_warmup_seed,
            }
            _atomic_save(best_payload, output)
        scheduler.step()
        _write_history(output.parent / "head_warmup_history.csv", history)
        print(f"[head_warmup_epoch] epoch={epoch} val={validation['accuracy']:.4f} best={best:.4f}", flush=True)
    if best_payload is None:
        raise RuntimeError("Head warm-up produced no checkpoint")
    metadata = {
        "checkpoint": str(output),
        "checkpoint_sha256": sha256_file(output),
        "source_checkpoint_sha256": settings.formal.source_checkpoint_sha256,
        "selected_epoch": best_payload["selected_epoch"],
        "best_validation_accuracy": best,
    }
    _write_json(output.parent / "common_checkpoint.json", metadata)
    return best_payload


def _configure_method(
    model: OpticalClassifier,
    method: Method,
    *,
    source_phases: torch.Tensor,
    seed: int,
) -> None:
    if method in {"noft", "bp"}:
        model.configure_feedback("bp")
    elif method == "fa_pretrained":
        model.configure_feedback("fa_pretrained", pretrained_phases=source_phases)
    else:
        model.configure_feedback("fa_random", random_seed=seed + 99173)


def _optimizer(model: OpticalClassifier, settings: FormalSettings) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": list(model.phase_parameters()), "lr": settings.formal.phase_learning_rate, "name": "phase"},
            {
                "params": list(model.electronic_parameters()),
                "lr": settings.formal.electronic_learning_rate,
                "name": "electronic",
            },
            {
                "params": list(model.residual_parameters()),
                "lr": settings.formal.residual_learning_rate,
                "name": "residual",
            },
        ],
        betas=settings.base.optimizer.betas,
        eps=settings.base.optimizer.eps,
        weight_decay=settings.base.optimizer.weight_decay,
    )


def _phase_gradients(
    model: OpticalClassifier,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: Method,
    source_phases: torch.Tensor,
    seed: int,
) -> list[torch.Tensor]:
    _configure_method(model, mode, source_phases=source_phases, seed=seed)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), targets).backward()
    gradients = [stage.raw_phase.grad.detach().float().cpu().clone() for stage in model.stages]
    model.zero_grad(set_to_none=True)
    return gradients


def gradient_diagnostic(
    model: OpticalClassifier,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    method: Method,
    source_phases: torch.Tensor,
    seed: int,
) -> list[dict[str, float]]:
    if method in {"noft", "bp"}:
        return []
    images, targets = batch
    images = images.to(device)
    targets = targets.to(device)
    approximate = _phase_gradients(
        model,
        images,
        targets,
        mode=method,
        source_phases=source_phases,
        seed=seed,
    )
    exact = _phase_gradients(
        model,
        images,
        targets,
        mode="bp",
        source_phases=source_phases,
        seed=seed,
    )
    _configure_method(model, method, source_phases=source_phases, seed=seed)
    rows = []
    for index, (candidate, reference) in enumerate(zip(approximate, exact, strict=True)):
        candidate_flat = candidate.flatten()
        reference_flat = reference.flatten()
        rows.append(
            {
                "stage": float(index),
                "cosine": float(F.cosine_similarity(candidate_flat, reference_flat, dim=0)),
                "candidate_norm": float(candidate_flat.norm()),
                "bp_norm": float(reference_flat.norm()),
            }
        )
    return rows


def _phase_geometry(current: torch.Tensor, initial: torch.Tensor) -> dict[str, float]:
    circular = torch.angle(torch.exp(1j * (current - initial)))
    return {
        "phase_circular_rms_rad": float(circular.square().mean().sqrt()),
        "phase_phasor_distance": float(
            (torch.exp(1j * current) - torch.exp(1j * initial)).abs().square().mean().sqrt()
        ),
        "phase_operator_coherence": float(torch.cos(current - initial).mean()),
    }


def _test_diagnostics(
    model: OpticalClassifier,
    datasets: DatasetBundle,
    device: torch.device,
    settings: FormalSettings,
    seed: int,
) -> dict[str, object]:
    loader = make_loader(datasets.test, settings.base, train=False, seed=seed + 2)
    results = {
        name: evaluate(model, loader, device, settings.base, ablation=name)
        for name in ("normal", "optical_off", "phase_random", "phase_shuffle")
    }
    full = float(results["normal"]["accuracy"])
    off = float(results["optical_off"]["accuracy"])
    chance = 1.0 / settings.base.num_classes
    return {
        "ablations": results,
        "absolute_full_minus_off": full - off,
        "normalized_optical_dependence": (full - off) / max(full - chance, 1e-12),
    }


def run_method(
    settings: FormalSettings,
    datasets: DatasetBundle,
    device: torch.device,
    *,
    method: Method,
    seed: int,
    force: bool = False,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    common_sha = sha256_file(settings.formal.common_checkpoint)
    common = torch.load(settings.formal.common_checkpoint, map_location="cpu", weights_only=False)
    set_seed(seed)
    run_dir = settings.base.output_dir / method / f"seed_{seed}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    model = build_model(settings.base, device)
    model.load_state_dict(common["model"], strict=True)
    source_phases = common["source_phases"].float()
    initial_phases = model.snapshot_phases()
    _configure_method(model, method, source_phases=source_phases, seed=seed)
    train_loader = make_loader(datasets.train, settings.base, train=True, seed=seed)
    validation_loader = make_loader(datasets.validation, settings.base, train=False, seed=seed + 1)
    initial_validation = evaluate(model, validation_loader, device, settings.base)
    first_batch = next(iter(train_loader))
    gradient_rows = gradient_diagnostic(
        model,
        first_batch,
        device,
        method=method,
        source_phases=source_phases,
        seed=seed,
    )
    best_accuracy = float(initial_validation["accuracy"])
    history: list[dict[str, object]] = []
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_epoch = 0
    if method != "noft":
        optimizer = _optimizer(model, settings)
        scheduler = _cosine_scheduler(
            optimizer,
            epochs=settings.formal.finetune_epochs,
            warmup_epochs=settings.formal.warmup_epochs,
            minimum=settings.formal.min_learning_rate_ratio,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=settings.base.training.use_amp and device.type == "cuda")
        for epoch in range(1, settings.formal.finetune_epochs + 1):
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                settings,
                epoch=epoch,
                total_epochs=settings.formal.finetune_epochs,
                prefix=method,
            )
            validation = evaluate(model, validation_loader, device, settings.base)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "seconds": train_metrics["seconds"],
                "phase_lr": optimizer.param_groups[0]["lr"],
                "mean_optical_weight": float(np.mean(model.optical_weights())),
            }
            history.append(row)
            if float(validation["accuracy"]) > best_accuracy:
                best_accuracy = float(validation["accuracy"])
                selected_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            scheduler.step()
            _write_history(run_dir / "history.csv", history)
            if epoch % settings.formal.checkpoint_interval_epochs == 0:
                _atomic_save(
                    {"model": model.state_dict(), "epoch": epoch, "common_checkpoint_sha256": common_sha},
                    run_dir / f"epoch_{epoch:03d}.pt",
                )
            print(
                f"[formal_epoch] method={method} epoch={epoch} val={validation['accuracy']:.4f} "
                f"best={best_accuracy:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state, strict=True)
    endpoint_phases = model.snapshot_phases()
    test = _test_diagnostics(model, datasets, device, settings, seed)
    checkpoint_payload = {
        "model": best_state,
        "method": method,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "best_validation_accuracy": best_accuracy,
        "common_checkpoint_sha256": common_sha,
        "source_checkpoint_sha256": settings.formal.source_checkpoint_sha256,
    }
    _atomic_save(checkpoint_payload, run_dir / "best.pt")
    result: dict[str, object] = {
        "method": method,
        "seed": seed,
        "common_checkpoint": str(settings.formal.common_checkpoint),
        "common_checkpoint_sha256": common_sha,
        "source_checkpoint_sha256": settings.formal.source_checkpoint_sha256,
        "initial_validation": initial_validation,
        "selected_epoch": selected_epoch,
        "best_validation_accuracy": best_accuracy,
        "test": test,
        "gradient_diagnostic_epoch_zero": gradient_rows,
        "endpoint_geometry": _phase_geometry(endpoint_phases, initial_phases),
        "optical_weights": model.optical_weights(),
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def compare(settings: FormalSettings) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    common_digests: set[str] = set()
    missing: list[str] = []
    for method in METHODS:
        for seed in settings.formal.finetune_seeds:
            path = settings.base.output_dir / method / f"seed_{seed}" / "result.json"
            if not path.exists():
                missing.append(f"{method}/seed_{seed}")
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            common_digests.add(result["common_checkpoint_sha256"])
            normal = result["test"]["ablations"]["normal"]["accuracy"]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_epoch": result["selected_epoch"],
                    "validation_accuracy": result["best_validation_accuracy"],
                    "test_accuracy": normal,
                    "optical_off_accuracy": result["test"]["ablations"]["optical_off"]["accuracy"],
                    "phase_random_accuracy": result["test"]["ablations"]["phase_random"]["accuracy"],
                    "phase_shuffle_accuracy": result["test"]["ablations"]["phase_shuffle"]["accuracy"],
                    "normalized_optical_dependence": result["test"]["normalized_optical_dependence"],
                    **result["endpoint_geometry"],
                }
            )
    if missing:
        raise FileNotFoundError(f"Formal comparison is incomplete; missing: {', '.join(missing)}")
    if len(common_digests) > 1:
        raise RuntimeError(f"Methods used different common checkpoints: {sorted(common_digests)}")
    if not rows:
        raise FileNotFoundError("No formal result.json files found")
    summary = {}
    for method in METHODS:
        values = [float(row["test_accuracy"]) for row in rows if row["method"] == method]
        if values:
            summary[method] = {
                "n": len(values),
                "test_accuracy_mean": float(np.mean(values)),
                "test_accuracy_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    output = {"common_checkpoint_sha256": next(iter(common_digests)), "summary": summary, "runs": rows}
    comparison_dir = settings.base.output_dir / "comparison"
    _write_json(comparison_dir / "comparison.json", output)
    _write_history(comparison_dir / "runs.csv", rows)
    return output
