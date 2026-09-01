from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .datasets import DatasetBundle, load_datasets, make_loader
from .deployment_adaptation import evaluate
from .deployment_robustness import (
    DeploymentCondition,
    build_differentiable_deployment_state,
)
from .fixed_feedback_training import _configure_method, sha256_file
from .formal_settings import FormalSettings, load_formal_settings
from .model import OpticalClassifier
from .optics import OpticalDeploymentState
from .training import build_model, set_seed


@dataclass(frozen=True)
class VaccinationSettings:
    config_path: Path
    formal_config: Path
    output_dir: Path
    source_method: str
    source_checkpoint: str | None
    training_seeds: tuple[int, ...]
    evaluation_deployment_seeds: tuple[int, ...]
    evaluation_conditions: tuple[DeploymentCondition, ...]
    epochs: int
    phase_learning_rate: float
    electronic_learning_rate: float
    residual_learning_rate: float
    warmup_epochs: int
    min_learning_rate_ratio: float
    starting_shift_pixels: float
    target_shift_pixels: float
    curriculum_epochs: int
    global_geometry_probability: float
    ideal_loss_weight: float
    shifted_loss_weight: float
    consistency_weight: float
    consistency_temperature: float
    max_train_batches: int | None


def load_vaccination_settings(path: Path) -> VaccinationSettings:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = payload["vaccination"]
    conditions = tuple(DeploymentCondition(**item) for item in payload["evaluation_conditions"])
    for condition in conditions:
        condition.validate()
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Vaccination evaluation condition names must be unique")
    if "ideal" not in names:
        raise ValueError("Vaccination evaluation conditions must include ideal")
    settings = VaccinationSettings(
        config_path=path,
        formal_config=Path(payload["formal_config"]),
        output_dir=Path(payload["output_dir"]),
        source_method=str(payload.get("source_method", "bp")),
        source_checkpoint=(
            None
            if not payload.get("source_checkpoint")
            else str(payload["source_checkpoint"])
        ),
        training_seeds=tuple(int(value) for value in payload["training_seeds"]),
        evaluation_deployment_seeds=tuple(
            int(value) for value in payload["evaluation_deployment_seeds"]
        ),
        evaluation_conditions=conditions,
        epochs=int(training["epochs"]),
        phase_learning_rate=float(training["phase_learning_rate"]),
        electronic_learning_rate=float(training["electronic_learning_rate"]),
        residual_learning_rate=float(training["residual_learning_rate"]),
        warmup_epochs=int(training.get("warmup_epochs", 1)),
        min_learning_rate_ratio=float(training.get("min_learning_rate_ratio", 0.1)),
        starting_shift_pixels=float(training.get("starting_shift_pixels", 0.0)),
        target_shift_pixels=float(training["target_shift_pixels"]),
        curriculum_epochs=int(training["curriculum_epochs"]),
        global_geometry_probability=float(training.get("global_geometry_probability", 0.5)),
        ideal_loss_weight=float(training["ideal_loss_weight"]),
        shifted_loss_weight=float(training["shifted_loss_weight"]),
        consistency_weight=float(training.get("consistency_weight", 0.0)),
        consistency_temperature=float(training.get("consistency_temperature", 1.0)),
        max_train_batches=(
            None
            if training.get("max_train_batches") is None
            else int(training["max_train_batches"])
        ),
    )
    if settings.source_method != "bp":
        raise ValueError("Misalignment vaccination must start from the shared BP source")
    if not settings.training_seeds:
        raise ValueError("At least one training seed is required")
    if not settings.evaluation_deployment_seeds:
        raise ValueError("At least one held-out deployment seed is required")
    if settings.epochs < 1 or not 1 <= settings.curriculum_epochs <= settings.epochs:
        raise ValueError("epochs must be positive and curriculum_epochs must lie within training")
    if not 0.0 <= settings.starting_shift_pixels <= settings.target_shift_pixels:
        raise ValueError("Shift curriculum endpoints are invalid")
    if not 0.0 <= settings.global_geometry_probability <= 1.0:
        raise ValueError("global_geometry_probability must be in [0, 1]")
    if settings.ideal_loss_weight < 0.0 or settings.shifted_loss_weight <= 0.0:
        raise ValueError("The ideal/shifted loss weights are invalid")
    if settings.consistency_weight < 0.0 or settings.consistency_temperature <= 0.0:
        raise ValueError("The consistency loss settings are invalid")
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _source_path(
    settings: VaccinationSettings,
    formal: FormalSettings,
    training_seed: int,
) -> Path:
    if settings.source_checkpoint is None:
        return (
            formal.base.output_dir
            / settings.source_method
            / f"seed_{training_seed}"
            / "best.pt"
        )
    return Path(settings.source_checkpoint.format(training_seed=training_seed))


def _load_source(
    model: OpticalClassifier,
    settings: VaccinationSettings,
    formal: FormalSettings,
    training_seed: int,
) -> tuple[Path, str]:
    checkpoint = _source_path(settings, formal, training_seed)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Vaccination source checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    _configure_method(
        model,
        "bp",
        source_phases=model.snapshot_phases(),
        seed=training_seed,
    )
    return checkpoint, sha256_file(checkpoint)


def _optimizer(
    model: OpticalClassifier,
    settings: VaccinationSettings,
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
    settings: VaccinationSettings,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(epoch: int) -> float:
        if settings.warmup_epochs > 0 and epoch < settings.warmup_epochs:
            return float(epoch + 1) / float(settings.warmup_epochs)
        progress = float(epoch - settings.warmup_epochs) / float(
            max(settings.epochs - settings.warmup_epochs - 1, 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return settings.min_learning_rate_ratio + (
            1.0 - settings.min_learning_rate_ratio
        ) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def curriculum_shift_pixels(settings: VaccinationSettings, epoch: int) -> float:
    progress = min(max(float(epoch) / float(settings.curriculum_epochs), 0.0), 1.0)
    return settings.starting_shift_pixels + progress * (
        settings.target_shift_pixels - settings.starting_shift_pixels
    )


def sample_training_deployment(
    num_stages: int,
    *,
    max_shift_pixels: float,
    global_geometry_probability: float,
    generator: torch.Generator,
) -> tuple[OpticalDeploymentState, dict[str, object]]:
    """Sample a continuous, batch-wise physical alignment state."""

    use_global = float(torch.rand((), generator=generator)) < global_geometry_probability

    def pair() -> tuple[float, float]:
        values = (2.0 * torch.rand(2, generator=generator) - 1.0) * max_shift_pixels
        return float(values[0]), float(values[1])

    if use_global:
        sampled = pair()
        shifts = tuple(sampled for _ in range(num_stages))
        geometry = "global"
    else:
        shifts = tuple(pair() for _ in range(num_stages))
        geometry = "layerwise"
    return (
        OpticalDeploymentState(phase_shifts_dy_dx=shifts),
        {
            "geometry": geometry,
            "max_shift_pixels": max_shift_pixels,
            "phase_shifts_dy_dx": [list(value) for value in shifts],
        },
    )


def _train_epoch(
    model: OpticalClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    formal: FormalSettings,
    settings: VaccinationSettings,
    *,
    epoch: int,
    shift_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    maximum_shift = curriculum_shift_pixels(settings, epoch)
    total_loss = 0.0
    total_ideal_loss = 0.0
    total_shifted_loss = 0.0
    total_consistency = 0.0
    total_ideal_correct = 0
    total_shifted_correct = 0
    total_global_batches = 0
    total_batches = 0
    total_samples = 0
    started = time.perf_counter()
    use_amp = formal.base.training.use_amp and device.type == "cuda"
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if settings.max_train_batches is not None and batch_index > settings.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        deployment, deployment_metadata = sample_training_deployment(
            len(model.stages),
            max_shift_pixels=maximum_shift,
            global_geometry_probability=settings.global_geometry_probability,
            generator=shift_generator,
        )
        total_batches += 1
        total_global_batches += int(deployment_metadata["geometry"] == "global")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            ideal_logits = model(images)
            shifted_logits = model(images, deployment=deployment)
            ideal_loss = F.cross_entropy(
                ideal_logits,
                targets,
                label_smoothing=formal.base.training.label_smoothing,
            )
            shifted_loss = F.cross_entropy(
                shifted_logits,
                targets,
                label_smoothing=formal.base.training.label_smoothing,
            )
            temperature = settings.consistency_temperature
            teacher = F.softmax(ideal_logits.detach().float() / temperature, dim=1)
            consistency = F.kl_div(
                F.log_softmax(shifted_logits.float() / temperature, dim=1),
                teacher,
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = (
                settings.ideal_loss_weight * ideal_loss
                + settings.shifted_loss_weight * shifted_loss
                + settings.consistency_weight * consistency
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite vaccination loss at epoch={epoch}, batch={batch_index}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), formal.base.training.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        count = int(targets.numel())
        total_loss += float(loss.detach()) * count
        total_ideal_loss += float(ideal_loss.detach()) * count
        total_shifted_loss += float(shifted_loss.detach()) * count
        total_consistency += float(consistency.detach()) * count
        total_ideal_correct += int((ideal_logits.detach().argmax(dim=1) == targets).sum())
        total_shifted_correct += int((shifted_logits.detach().argmax(dim=1) == targets).sum())
        total_samples += count
        if (
            batch_index % formal.base.training.log_interval_batches == 0
            or batch_index == len(loader)
        ):
            print(
                f"[vaccinate_batch] epoch={epoch}/{settings.epochs} "
                f"batch={batch_index}/{len(loader)} max_shift={maximum_shift:.3f} "
                f"loss={total_loss / max(total_samples, 1):.4f} "
                f"ideal_acc={total_ideal_correct / max(total_samples, 1):.4f} "
                f"shifted_acc={total_shifted_correct / max(total_samples, 1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "ideal_loss": total_ideal_loss / max(total_samples, 1),
        "shifted_loss": total_shifted_loss / max(total_samples, 1),
        "consistency_loss": total_consistency / max(total_samples, 1),
        "ideal_accuracy": total_ideal_correct / max(total_samples, 1),
        "shifted_accuracy": total_shifted_correct / max(total_samples, 1),
        "global_batch_fraction": total_global_batches / max(total_batches, 1),
        "max_shift_pixels": maximum_shift,
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


def _evaluation_grid(
    model: OpticalClassifier,
    loader,
    device: torch.device,
    formal: FormalSettings,
    settings: VaccinationSettings,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    for deployment_seed in settings.evaluation_deployment_seeds:
        for condition in settings.evaluation_conditions:
            if condition.name == "ideal":
                deployment = None
                metadata: dict[str, object] = {"phase_shifts_dy_dx": []}
            else:
                deployment, metadata = build_differentiable_deployment_state(
                    model,
                    condition,
                    deployment_seed=deployment_seed,
                    device=device,
                )
            metrics = evaluate(
                model,
                loader,
                device,
                deployment=deployment,
                max_batches=formal.base.training.max_evaluation_batches,
            )
            rows.append(
                {
                    "condition": condition.name,
                    "deployment_seed": deployment_seed,
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                    "samples": metrics["samples"],
                    "deployment": metadata,
                }
            )
    accuracies = [float(row["accuracy"]) for row in rows]
    ideal = [float(row["accuracy"]) for row in rows if row["condition"] == "ideal"]
    summary = {
        "mean_accuracy": float(np.mean(accuracies)),
        "worst_accuracy": float(np.min(accuracies)),
        "ideal_accuracy": float(np.mean(ideal)),
    }
    return rows, summary


def run_seed(
    settings: VaccinationSettings,
    formal: FormalSettings,
    datasets: DatasetBundle,
    *,
    training_seed: int,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    run_dir = settings.output_dir / f"seed_{training_seed}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    vaccination_seed = int(training_seed) * 100_000 + 50_005
    set_seed(vaccination_seed)
    model = build_model(formal.base, device)
    source_checkpoint, source_sha = _load_source(
        model, settings, formal, training_seed
    )
    validation_loader = make_loader(
        datasets.validation,
        formal.base,
        train=False,
        seed=vaccination_seed + 1,
    )
    source_grid, source_summary = _evaluation_grid(
        model, validation_loader, device, formal, settings
    )
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    best_score = float(source_summary["mean_accuracy"])
    best_grid = source_grid
    best_summary = source_summary
    selected_epoch = 0
    history: list[dict[str, object]] = []

    optimizer = _optimizer(model, settings, formal)
    scheduler = _scheduler(optimizer, settings)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=formal.base.training.use_amp and device.type == "cuda"
    )
    start_epoch = 1
    last_path = run_dir / "last.pt"
    if last_path.exists() and not force:
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("config_sha256") != sha256_file(settings.config_path):
            raise RuntimeError("Refusing to resume vaccination from a different config")
        model.load_state_dict(resume["model"], strict=True)
        _configure_method(
            model,
            "bp",
            source_phases=model.snapshot_phases(),
            seed=training_seed,
        )
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = list(resume["history"])
        best_state = resume["best_state"]
        best_score = float(resume["best_score"])
        best_grid = resume["best_grid"]
        best_summary = resume["best_summary"]
        selected_epoch = int(resume["selected_epoch"])
        start_epoch = int(resume["epoch"]) + 1
        print(f"[vaccinate_resume] seed={training_seed} epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, settings.epochs + 1):
        train_loader = make_loader(
            datasets.train,
            formal.base,
            train=True,
            seed=vaccination_seed + epoch,
        )
        shift_generator = torch.Generator().manual_seed(
            vaccination_seed + epoch * 10_009
        )
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            formal,
            settings,
            epoch=epoch,
            shift_generator=shift_generator,
        )
        validation_grid, validation_summary = _evaluation_grid(
            model, validation_loader, device, formal, settings
        )
        score = float(validation_summary["mean_accuracy"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ideal_loss": train_metrics["ideal_loss"],
            "train_shifted_loss": train_metrics["shifted_loss"],
            "train_consistency_loss": train_metrics["consistency_loss"],
            "train_ideal_accuracy": train_metrics["ideal_accuracy"],
            "train_shifted_accuracy": train_metrics["shifted_accuracy"],
            "train_max_shift_pixels": train_metrics["max_shift_pixels"],
            "train_global_batch_fraction": train_metrics["global_batch_fraction"],
            "validation_mean_accuracy": validation_summary["mean_accuracy"],
            "validation_worst_accuracy": validation_summary["worst_accuracy"],
            "validation_ideal_accuracy": validation_summary["ideal_accuracy"],
            "phase_lr": optimizer.param_groups[0]["lr"],
            "mean_optical_weight": float(np.mean(model.optical_weights())),
            "seconds": train_metrics["seconds"],
        }
        history.append(row)
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_grid = validation_grid
            best_summary = validation_summary
        scheduler.step()
        _write_history(run_dir / "history.csv", history)
        _write_json(
            run_dir / f"validation_epoch_{epoch:03d}.json",
            {"rows": validation_grid, "summary": validation_summary},
        )
        _atomic_save(
            {
                "config_sha256": sha256_file(settings.config_path),
                "epoch": epoch,
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_state": best_state,
                "best_score": best_score,
                "best_grid": best_grid,
                "best_summary": best_summary,
                "selected_epoch": selected_epoch,
            },
            last_path,
        )
        print(
            f"[vaccinate_epoch] epoch={epoch}/{settings.epochs} "
            f"ideal={validation_summary['ideal_accuracy']:.4f} "
            f"mean={score:.4f} worst={validation_summary['worst_accuracy']:.4f} "
            f"best={best_score:.4f}@{selected_epoch}",
            flush=True,
        )

    model.load_state_dict(best_state, strict=True)
    _configure_method(
        model,
        "bp",
        source_phases=model.snapshot_phases(),
        seed=training_seed,
    )
    result: dict[str, Any] = {
        "config": str(settings.config_path),
        "config_sha256": sha256_file(settings.config_path),
        "formal_config": str(settings.formal_config),
        "source_method": settings.source_method,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "training_seed": training_seed,
        "vaccination_seed": vaccination_seed,
        "selected_epoch": selected_epoch,
        "selection_metric": "held_out_deployment_mean_accuracy",
        "source_evaluation": {"rows": source_grid, "summary": source_summary},
        "best_evaluation": {"rows": best_grid, "summary": best_summary},
        "improvement": {
            "mean_accuracy": best_summary["mean_accuracy"] - source_summary["mean_accuracy"],
            "worst_accuracy": best_summary["worst_accuracy"] - source_summary["worst_accuracy"],
            "ideal_accuracy": best_summary["ideal_accuracy"] - source_summary["ideal_accuracy"],
        },
        "optical_weights": model.optical_weights(),
        "phase_parameters": sum(parameter.numel() for parameter in model.phase_parameters()),
        "residual_parameters": sum(parameter.numel() for parameter in model.residual_parameters()),
        "electronic_parameters": sum(parameter.numel() for parameter in model.electronic_parameters()),
    }
    _atomic_save(
        {
            "model": best_state,
            "training_seed": training_seed,
            "selected_epoch": selected_epoch,
            "source_checkpoint_sha256": source_sha,
            "vaccination": asdict(settings),
        },
        run_dir / "best.pt",
    )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-deployment curriculum training against continuous optical misalignment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_vaccination_settings(args.config)
    requested_seeds = (args.seed,) if args.seed is not None else settings.training_seeds
    unknown = sorted(set(requested_seeds) - set(settings.training_seeds))
    if unknown:
        raise ValueError(f"Seeds are not registered in the config: {unknown}")
    formal = load_formal_settings(settings.formal_config)
    datasets = load_datasets(formal.base, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    for training_seed in requested_seeds:
        run_seed(
            settings,
            formal,
            datasets,
            training_seed=training_seed,
            device=device,
            force=args.force,
        )


if __name__ == "__main__":
    main()
