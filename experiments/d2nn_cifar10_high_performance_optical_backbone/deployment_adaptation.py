from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .datasets import DatasetBundle, load_datasets, make_loader
from .deployment_robustness import (
    DeploymentCondition,
    build_differentiable_deployment_state,
)
from .fixed_feedback_training import METHODS, _configure_method, sha256_file
from .formal_settings import FormalSettings, load_formal_settings
from .model import OpticalClassifier
from .optics import OpticalDeploymentState
from .training import build_model, set_seed


@dataclass(frozen=True)
class AdaptationSettings:
    config_path: Path
    formal_config: Path
    output_dir: Path
    source_method: str
    source_checkpoint: str | None
    methods: tuple[str, ...]
    training_seeds: tuple[int, ...]
    deployment_seeds: tuple[int, ...]
    conditions: tuple[DeploymentCondition, ...]
    evaluation_split: str
    epochs: int
    phase_learning_rate: float
    electronic_learning_rate: float
    residual_learning_rate: float
    warmup_epochs: int
    min_learning_rate_ratio: float
    max_train_batches: int | None


def load_adaptation_settings(path: Path) -> AdaptationSettings:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    methods = tuple(str(value) for value in payload["methods"])
    if set(methods) != set(METHODS):
        raise ValueError(f"Deployment adaptation must use exactly the four methods: {METHODS}")
    source_method = str(payload.get("source_method", "bp"))
    if source_method not in METHODS or source_method == "noft":
        raise ValueError("source_method must name a trained formal method")
    evaluation_split = str(payload.get("evaluation_split", "validation"))
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    conditions = tuple(DeploymentCondition(**item) for item in payload["conditions"])
    for condition in conditions:
        condition.validate()
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Adaptation condition names must be unique")
    adaptation = payload["adaptation"]
    settings = AdaptationSettings(
        config_path=path,
        formal_config=Path(payload["formal_config"]),
        output_dir=Path(payload["output_dir"]),
        source_method=source_method,
        source_checkpoint=(
            None
            if not payload.get("source_checkpoint")
            else str(payload["source_checkpoint"])
        ),
        methods=methods,
        training_seeds=tuple(int(value) for value in payload["training_seeds"]),
        deployment_seeds=tuple(int(value) for value in payload["deployment_seeds"]),
        conditions=conditions,
        evaluation_split=evaluation_split,
        epochs=int(adaptation["epochs"]),
        phase_learning_rate=float(adaptation["phase_learning_rate"]),
        electronic_learning_rate=float(adaptation["electronic_learning_rate"]),
        residual_learning_rate=float(adaptation["residual_learning_rate"]),
        warmup_epochs=int(adaptation.get("warmup_epochs", 1)),
        min_learning_rate_ratio=float(adaptation.get("min_learning_rate_ratio", 0.1)),
        max_train_batches=(
            None
            if adaptation.get("max_train_batches") is None
            else int(adaptation["max_train_batches"])
        ),
    )
    if settings.epochs < 1:
        raise ValueError("adaptation.epochs must be positive")
    if settings.warmup_epochs < 0 or settings.warmup_epochs > settings.epochs:
        raise ValueError("adaptation.warmup_epochs must be between zero and epochs")
    return settings


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _optimizer(
    model: OpticalClassifier,
    settings: AdaptationSettings,
    formal: FormalSettings,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": list(model.phase_parameters()),
                "lr": settings.phase_learning_rate,
                "name": "phase",
            },
            {
                "params": list(model.electronic_parameters()),
                "lr": settings.electronic_learning_rate,
                "name": "electronic",
            },
            {
                "params": list(model.residual_parameters()),
                "lr": settings.residual_learning_rate,
                "name": "residual",
            },
        ],
        betas=formal.base.optimizer.betas,
        eps=formal.base.optimizer.eps,
        weight_decay=formal.base.optimizer.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    settings: AdaptationSettings,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(epoch: int) -> float:
        if settings.warmup_epochs > 0 and epoch < settings.warmup_epochs:
            return float(epoch + 1) / float(settings.warmup_epochs)
        progress = float(epoch - settings.warmup_epochs) / float(
            max(settings.epochs - settings.warmup_epochs - 1, 1)
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        return settings.min_learning_rate_ratio + (1.0 - settings.min_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


@torch.inference_mode()
def evaluate(
    model: OpticalClassifier,
    loader,
    device: torch.device,
    *,
    deployment: OpticalDeploymentState | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, deployment=deployment)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_samples += int(targets.numel())
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


def _train_epoch(
    model: OpticalClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    formal: FormalSettings,
    settings: AdaptationSettings,
    deployment: OpticalDeploymentState,
    *,
    epoch: int,
    method: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if settings.max_train_batches is not None and batch_index > settings.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=formal.base.training.use_amp and device.type == "cuda",
        ):
            logits = model(images, deployment=deployment)
            loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=formal.base.training.label_smoothing,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), formal.base.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * targets.numel()
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
        total_samples += int(targets.numel())
        if batch_index % formal.base.training.log_interval_batches == 0:
            print(
                f"[adapt_batch] method={method} epoch={epoch}/{settings.epochs} "
                f"batch={batch_index}/{len(loader)} loss={total_loss / max(total_samples, 1):.4f} "
                f"accuracy={total_correct / max(total_samples, 1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


def _phase_gradients(
    model: OpticalClassifier,
    images: torch.Tensor,
    targets: torch.Tensor,
    deployment: OpticalDeploymentState,
    *,
    mode: str,
    source_phases: torch.Tensor,
    seed: int,
) -> list[torch.Tensor]:
    _configure_method(model, mode, source_phases=source_phases, seed=seed)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(images, deployment=deployment), targets).backward()
    gradients = [stage.raw_phase.grad.detach().float().cpu().clone() for stage in model.stages]
    model.zero_grad(set_to_none=True)
    return gradients


def gradient_diagnostic(
    model: OpticalClassifier,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    deployment: OpticalDeploymentState,
    *,
    method: str,
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
        deployment,
        mode=method,
        source_phases=source_phases,
        seed=seed,
    )
    exact = _phase_gradients(
        model,
        images,
        targets,
        deployment,
        mode="bp",
        source_phases=source_phases,
        seed=seed,
    )
    _configure_method(model, method, source_phases=source_phases, seed=seed)
    rows: list[dict[str, float]] = []
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


def _load_source(
    settings: AdaptationSettings,
    formal: FormalSettings,
    model: OpticalClassifier,
    training_seed: int,
) -> tuple[Path, str, torch.Tensor]:
    if settings.source_checkpoint is None:
        checkpoint = (
            formal.base.output_dir
            / settings.source_method
            / f"seed_{training_seed}"
            / "best.pt"
        )
    else:
        checkpoint = Path(
            settings.source_checkpoint.format(training_seed=training_seed)
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Adaptation source checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return checkpoint, sha256_file(checkpoint), model.snapshot_phases()


def run_one(
    settings: AdaptationSettings,
    formal: FormalSettings,
    datasets: DatasetBundle,
    *,
    method: str,
    condition: DeploymentCondition,
    training_seed: int,
    deployment_seed: int,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    run_dir = (
        settings.output_dir
        / method
        / f"seed_{training_seed}"
        / f"deployment_seed_{deployment_seed}"
        / condition.name
    )
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    adaptation_seed = int(training_seed) * 100_000 + int(deployment_seed)
    set_seed(adaptation_seed)
    model = build_model(formal.base, device)
    source_checkpoint, source_sha, source_phases = _load_source(
        settings, formal, model, training_seed
    )
    source_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    _configure_method(model, method, source_phases=source_phases, seed=training_seed)
    deployment, deployment_metadata = build_differentiable_deployment_state(
        model,
        condition,
        deployment_seed=deployment_seed,
        device=device,
    )

    train_loader = make_loader(datasets.train, formal.base, train=True, seed=adaptation_seed)
    validation_loader = make_loader(
        datasets.validation, formal.base, train=False, seed=adaptation_seed + 1
    )
    final_dataset = datasets.validation if settings.evaluation_split == "validation" else datasets.test
    final_loader = make_loader(final_dataset, formal.base, train=False, seed=adaptation_seed + 2)
    max_evaluation_batches = formal.base.training.max_evaluation_batches

    source_ideal = evaluate(
        model,
        validation_loader,
        device,
        deployment=None,
        max_batches=max_evaluation_batches,
    )
    initial_deployed = evaluate(
        model,
        validation_loader,
        device,
        deployment=deployment,
        max_batches=max_evaluation_batches,
    )
    first_batch = next(iter(train_loader))
    gradient_rows = gradient_diagnostic(
        model,
        first_batch,
        device,
        deployment,
        method=method,
        source_phases=source_phases,
        seed=training_seed,
    )

    best_accuracy = float(initial_deployed["accuracy"])
    best_state = source_state
    selected_epoch = 0
    history: list[dict[str, object]] = []
    if method != "noft":
        optimizer = _optimizer(model, settings, formal)
        scheduler = _scheduler(optimizer, settings)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=formal.base.training.use_amp and device.type == "cuda"
        )
        for epoch in range(1, settings.epochs + 1):
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                formal,
                settings,
                deployment,
                epoch=epoch,
                method=method,
            )
            validation = evaluate(
                model,
                validation_loader,
                device,
                deployment=deployment,
                max_batches=max_evaluation_batches,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "recovery_pp": 100.0
                * (float(validation["accuracy"]) - float(initial_deployed["accuracy"])),
                "seconds": train_metrics["seconds"],
                "phase_lr": optimizer.param_groups[0]["lr"],
                "mean_optical_weight": float(np.mean(model.optical_weights())),
            }
            history.append(row)
            if float(validation["accuracy"]) > best_accuracy:
                best_accuracy = float(validation["accuracy"])
                selected_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            scheduler.step()
            _write_history(run_dir / "history.csv", history)
            print(
                f"[adapt_epoch] condition={condition.name} method={method} "
                f"epoch={epoch}/{settings.epochs} validation={validation['accuracy']:.4f} "
                f"best={best_accuracy:.4f}",
                flush=True,
            )

    model.load_state_dict(best_state, strict=True)
    _configure_method(model, method, source_phases=source_phases, seed=training_seed)
    final_metrics = evaluate(
        model,
        final_loader,
        device,
        deployment=deployment,
        max_batches=max_evaluation_batches,
    )
    recovery = best_accuracy - float(initial_deployed["accuracy"])
    lost_accuracy = float(source_ideal["accuracy"]) - float(initial_deployed["accuracy"])
    result: dict[str, Any] = {
        "config": str(settings.config_path),
        "config_sha256": sha256_file(settings.config_path),
        "formal_config": str(settings.formal_config),
        "source_method": settings.source_method,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "method": method,
        "training_seed": training_seed,
        "deployment_seed": deployment_seed,
        "adaptation_seed": adaptation_seed,
        "condition": asdict(condition),
        "deployment": deployment_metadata,
        "source_ideal_validation": source_ideal,
        "initial_deployed_validation": initial_deployed,
        "selected_epoch": selected_epoch,
        "best_deployed_validation_accuracy": best_accuracy,
        "recovery_accuracy": recovery,
        "recovered_fraction_of_shift_loss": (
            recovery / lost_accuracy if lost_accuracy > 0.0 else None
        ),
        "evaluation_split": settings.evaluation_split,
        "final": final_metrics,
        "gradient_diagnostic_epoch_zero": gradient_rows,
        "optical_weights": model.optical_weights(),
    }
    _atomic_save(
        {
            "model": best_state,
            "method": method,
            "training_seed": training_seed,
            "deployment_seed": deployment_seed,
            "condition": asdict(condition),
            "selected_epoch": selected_epoch,
            "source_checkpoint_sha256": source_sha,
        },
        run_dir / "best.pt",
    )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compare(settings: AdaptationSettings) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for method in settings.methods:
        for training_seed in settings.training_seeds:
            for deployment_seed in settings.deployment_seeds:
                for condition in settings.conditions:
                    path = (
                        settings.output_dir
                        / method
                        / f"seed_{training_seed}"
                        / f"deployment_seed_{deployment_seed}"
                        / condition.name
                        / "result.json"
                    )
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    result = json.loads(path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "method": method,
                            "training_seed": training_seed,
                            "deployment_seed": deployment_seed,
                            "condition": condition.name,
                            "initial_accuracy": result["initial_deployed_validation"]["accuracy"],
                            "best_validation_accuracy": result["best_deployed_validation_accuracy"],
                            "final_accuracy": result["final"]["accuracy"],
                            "recovery_accuracy": result["recovery_accuracy"],
                            "recovered_fraction_of_shift_loss": result[
                                "recovered_fraction_of_shift_loss"
                            ],
                            "selected_epoch": result["selected_epoch"],
                        }
                    )
    if missing:
        raise FileNotFoundError("Missing adaptation results:\n" + "\n".join(missing))

    summaries: dict[str, object] = {}
    for condition in settings.conditions:
        condition_summary: dict[str, object] = {}
        for method in settings.methods:
            selected = [
                row
                for row in rows
                if row["condition"] == condition.name and row["method"] == method
            ]
            condition_summary[method] = {
                "initial_accuracy": _summary([float(row["initial_accuracy"]) for row in selected]),
                "final_accuracy": _summary([float(row["final_accuracy"]) for row in selected]),
                "recovery_accuracy": _summary([float(row["recovery_accuracy"]) for row in selected]),
            }
        summaries[condition.name] = condition_summary

    comparison = {
        "config": str(settings.config_path),
        "config_sha256": sha256_file(settings.config_path),
        "source_method": settings.source_method,
        "evaluation_split": settings.evaluation_split,
        "rows": rows,
        "summaries": summaries,
    }
    comparison_dir = settings.output_dir / "comparison"
    _write_json(comparison_dir / "comparison.json", comparison)
    _write_history(comparison_dir / "runs.csv", rows)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt one shared ideal checkpoint after a fixed optical deployment shift"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "compare"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_adaptation_settings(args.config)
    if args.phase == "compare":
        print(json.dumps(compare(settings), indent=2), flush=True)
        return
    requested_methods = tuple(args.methods) if args.methods else settings.methods
    condition_lookup = {condition.name: condition for condition in settings.conditions}
    requested_condition_names = (
        tuple(args.conditions) if args.conditions else tuple(condition_lookup)
    )
    unknown = sorted(set(requested_condition_names) - set(condition_lookup))
    if unknown:
        raise ValueError(f"Unknown deployment conditions: {unknown}")
    formal = load_formal_settings(settings.formal_config)
    datasets = load_datasets(formal.base, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    for condition_name in requested_condition_names:
        condition = condition_lookup[condition_name]
        for method in requested_methods:
            for training_seed in settings.training_seeds:
                for deployment_seed in settings.deployment_seeds:
                    run_one(
                        settings,
                        formal,
                        datasets,
                        method=method,
                        condition=condition,
                        training_seed=training_seed,
                        deployment_seed=deployment_seed,
                        device=device,
                        force=args.force,
                    )


if __name__ == "__main__":
    main()
