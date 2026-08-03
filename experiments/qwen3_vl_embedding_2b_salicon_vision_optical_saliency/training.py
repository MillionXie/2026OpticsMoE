from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .datasets import (
    SALICONBundle,
    SALICONSaliencyDataset,
    collate_salicon,
)
from .io_utils import write_json
from .modeling import (
    LoadedVisionBackbone,
    VisionOpticalSaliencyStudent,
    assert_student_trainability,
    build_student,
    build_teacher,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import SaliencyAccumulator, saliency_loss
from .visualization import save_examples, save_optical_parameters, save_training_curves


def build_loaders(
    bundle: SALICONBundle,
    settings: Any,
    *,
    training: bool,
    train_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train = SALICONSaliencyDataset(
        bundle.train_records, settings, training=training
    )
    validation = SALICONSaliencyDataset(
        bundle.validation_records, settings, training=False
    )
    batch_size = (
        int(train_batch_size or settings.student_batch_size)
        if training
        else settings.inference_batch_size
    )
    common = {
        "num_workers": settings.num_workers,
        "pin_memory": settings.device == "cuda",
        "persistent_workers": settings.num_workers > 0,
        "collate_fn": collate_salicon,
    }
    generator = torch.Generator().manual_seed(settings.random_seed)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=training,
            generator=generator if training else None,
            **common,
        ),
        DataLoader(validation, batch_size=settings.inference_batch_size, **common),
    )


def train_teacher(
    loaded: LoadedVisionBackbone,
    bundle: SALICONBundle,
    settings: Any,
) -> Path:
    model = build_teacher(loaded, settings)
    train_loader, validation_loader = build_loaders(
        bundle,
        settings,
        training=True,
        train_batch_size=settings.teacher_batch_size,
    )
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=settings.teacher_learning_rate,
        weight_decay=settings.weight_decay,
    )
    report = trainable_parameter_report(model, prefix="teacher")
    write_json(settings.output_dir / "teacher_model.json", report)
    _print_report(report)
    best_cc = -math.inf
    best_path = settings.output_dir / "checkpoints" / "teacher_best.pt"
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(1, settings.teacher_epochs + 1):
            train_metrics = _train_epoch(
                "teacher",
                model,
                train_loader,
                loaded,
                settings,
                optimizer,
            )
            validation_metrics, _ = evaluate_model(
                model, validation_loader, loaded, settings
            )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            history.append(row)
            _write_history(
                settings.output_dir / "metrics" / "teacher_history.csv", history
            )
            payload = {
                "format_version": 1,
                "epoch": epoch,
                "head": model.head.state_dict(),
                "validation_metrics": validation_metrics,
                "train_metrics": train_metrics,
            }
            _save_checkpoint(
                settings.output_dir / "checkpoints" / "teacher_last.pt", payload
            )
            if validation_metrics["cc"] > best_cc:
                best_cc = float(validation_metrics["cc"])
                _save_checkpoint(best_path, payload)
            print(
                f"[teacher] epoch={epoch:03d}/{settings.teacher_epochs:03d} "
                f"train_loss={train_metrics['loss']:.5f} "
                f"val_CC={validation_metrics['cc']:.4f} "
                f"val_KLD={validation_metrics['kld']:.4f} best_CC={best_cc:.4f}",
                flush=True,
            )
    finally:
        model.close()
    save_training_curves(
        settings.output_dir / "figures" / "teacher_training_curves.png",
        history,
        prefix="teacher",
    )
    return best_path


def train_student(
    loaded: LoadedVisionBackbone,
    bundle: SALICONBundle,
    settings: Any,
) -> Path:
    model = build_student(loaded, settings)
    assert_student_trainability(model)
    train_loader, validation_loader = build_loaders(
        bundle,
        settings,
        training=True,
        train_batch_size=settings.student_batch_size,
    )
    optimizer = _student_optimizer(model, settings)
    report = trainable_parameter_report(model, prefix="student")
    report["optical_breakdown"] = model.core.parameter_breakdown()
    write_json(settings.output_dir / "student_model.json", report)
    _print_report(report)
    teacher_cache = (
        _TeacherMapStore(settings) if settings.map_kd_weight > 0 else None
    )
    best_score = -math.inf
    best_path = settings.output_dir / "checkpoints" / "student_best.pt"
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(1, settings.student_epochs + 1):
            train_metrics = _train_epoch(
                "student",
                model,
                train_loader,
                loaded,
                settings,
                optimizer,
                teacher_cache=teacher_cache,
            )
            validation_metrics, _ = evaluate_model(
                model, validation_loader, loaded, settings
            )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            history.append(row)
            _write_history(
                settings.output_dir / "metrics" / "student_history.csv", history
            )
            payload = {
                "format_version": 1,
                "epoch": epoch,
                "optical_core": model.core.state_dict(),
                "saliency_head": model.head.state_dict(),
                "validation_metrics": validation_metrics,
                "train_metrics": train_metrics,
            }
            _save_checkpoint(
                settings.output_dir / "checkpoints" / "student_last.pt", payload
            )
            score = (
                float(validation_metrics["cc"])
                if settings.checkpoint_metric == "validation_cc"
                else -float(train_metrics["loss"])
            )
            if score > best_score:
                best_score = score
                _save_checkpoint(best_path, payload)
            print(
                f"[student] epoch={epoch:03d}/{settings.student_epochs:03d} "
                f"train_loss={train_metrics['loss']:.5f} "
                f"val_CC={validation_metrics['cc']:.4f} "
                f"val_SIM={validation_metrics['sim']:.4f} "
                f"val_NSS={validation_metrics['nss']:.4f}",
                flush=True,
            )
    finally:
        model.restore_native()
    save_training_curves(
        settings.output_dir / "figures" / "student_training_curves.png",
        history,
        prefix="student",
    )
    return best_path


@torch.no_grad()
def cache_teacher_maps(
    loaded: LoadedVisionBackbone,
    bundle: SALICONBundle,
    settings: Any,
) -> None:
    checkpoint = settings.teacher_checkpoint or (
        settings.output_dir / "checkpoints" / "teacher_best.pt"
    )
    model = build_teacher(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu")
    model.head.load_state_dict(payload["head"])
    model.eval()
    datasets = (
        ("train", bundle.train_records),
        ("validation", bundle.validation_records),
    )
    try:
        for split, records in datasets:
            loader = DataLoader(
                SALICONSaliencyDataset(records, settings, training=False),
                batch_size=settings.teacher_batch_size,
                shuffle=False,
                num_workers=settings.num_workers,
                pin_memory=loaded.device.type == "cuda",
                persistent_workers=settings.num_workers > 0,
                collate_fn=collate_salicon,
            )
            directory = settings.artifact_cache_dir / "teacher_logits" / split
            directory.mkdir(parents=True, exist_ok=True)
            for batch_index, batch in enumerate(loader, start=1):
                inputs = preprocess_vision(
                    loaded.processor, batch["images"], loaded.device
                )
                logits, _ = model(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
                for sample_id, value in zip(batch["sample_ids"], logits):
                    path = directory / f"{sample_id.split('/')[-1]}.pt"
                    torch.save(value.detach().cpu().to(torch.float16), path)
                if (
                    batch_index % settings.log_interval_batches == 0
                    or batch_index == len(loader)
                ):
                    print(
                        f"[teacher_map_cache] {split} "
                        f"batch={batch_index}/{len(loader)}",
                        flush=True,
                    )
    finally:
        model.close()


@torch.no_grad()
def evaluate_checkpoint(
    kind: str,
    loaded: LoadedVisionBackbone,
    bundle: SALICONBundle,
    settings: Any,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    _, validation_loader = build_loaders(bundle, settings, training=False)
    if kind == "teacher":
        model = build_teacher(loaded, settings)
        checkpoint = checkpoint or (
            settings.output_dir / "checkpoints" / "teacher_best.pt"
        )
        model.head.load_state_dict(
            torch.load(checkpoint, map_location="cpu")["head"]
        )
    elif kind == "student":
        model = build_student(loaded, settings)
        checkpoint = checkpoint or (
            settings.output_dir / "checkpoints" / "student_best.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu")
        model.core.load_state_dict(payload["optical_core"])
        model.head.load_state_dict(payload["saliency_head"])
    else:
        raise ValueError(kind)
    try:
        metrics, examples = evaluate_model(
            model,
            validation_loader,
            loaded,
            settings,
            collect_examples=settings.visualization_sample_count,
        )
        result = {
            "system": kind,
            "split": "official_validation",
            "checkpoint": str(checkpoint),
            **metrics,
        }
        write_json(settings.output_dir / "metrics" / f"{kind}_validation.json", result)
        save_examples(
            settings.output_dir / "figures" / f"{kind}_examples",
            examples,
            kind=kind,
        )
        if kind == "student":
            save_optical_parameters(
                model.core,
                settings.output_dir / "figures" / "optical_parameters",
            )
        return result
    finally:
        if kind == "teacher":
            model.close()
        else:
            model.restore_native()


def _train_epoch(
    kind: str,
    model: nn.Module,
    loader: DataLoader,
    loaded: LoadedVisionBackbone,
    settings: Any,
    optimizer: torch.optim.Optimizer,
    *,
    teacher_cache: "_TeacherMapStore | None" = None,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        density = batch["density"].to(loaded.device, non_blocking=True)
        fixation = batch["fixation"].to(loaded.device, non_blocking=True)
        inputs = preprocess_vision(loaded.processor, batch["images"], loaded.device)
        teacher_logits = (
            teacher_cache.get(batch["sample_ids"], loaded.device)
            if teacher_cache is not None
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(settings, loaded.device):
            output = model(inputs["pixel_values"], inputs["image_grid_thw"])
            logits = output[0]
            task_loss, pieces = saliency_loss(
                logits,
                density,
                fixation,
                settings,
                teacher_logits=teacher_logits,
            )
            if kind == "student":
                balance, importance = model.router_losses()
                dc = (
                    phase_dc_loss(model)
                    if settings.phase_dc_weight > 0.0
                    else logits.new_zeros(())
                )
                total = (
                    task_loss
                    + settings.router_balance_weight * balance
                    + settings.router_importance_weight * importance
                    + settings.phase_dc_weight * dc
                )
            else:
                balance = logits.new_zeros(())
                importance = logits.new_zeros(())
                dc = logits.new_zeros(())
                total = task_loss
        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite {kind} loss at batch {batch_index}: {total}"
            )
        total.backward()
        if settings.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                settings.gradient_clip_norm,
            )
        optimizer.step()
        count = len(batch["sample_ids"])
        totals["samples"] += count
        totals["loss"] += float(total.detach()) * count
        for key, value in pieces.items():
            totals[key] += float(value.detach()) * count
        totals["router_balance"] += float(balance.detach()) * count
        totals["router_importance"] += float(importance.detach()) * count
        totals["phase_dc"] += float(dc.detach()) * count
        if (
            batch_index % settings.log_interval_batches == 0
            or batch_index == len(loader)
        ):
            denominator = totals["samples"]
            print(
                f"[{kind}] batch={batch_index}/{len(loader)} "
                f"loss={totals['loss']/denominator:.5f} "
                f"KLD={totals['kl']/denominator:.4f} "
                f"CC={totals['cc']/denominator:.4f} "
                f"NSS={totals['nss']/denominator:.4f} "
                f"phase_dc={totals['phase_dc']/denominator:.4f}",
                flush=True,
            )
    denominator = totals["samples"]
    return {
        key: value / denominator
        for key, value in totals.items()
        if key != "samples"
    } | {
        "samples": int(denominator),
        "epoch_time_sec": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loaded: LoadedVisionBackbone,
    settings: Any,
    *,
    collect_examples: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = SaliencyAccumulator()
    examples: list[dict[str, Any]] = []
    for batch in loader:
        density = batch["density"].to(loaded.device, non_blocking=True)
        fixation = batch["fixation"].to(loaded.device, non_blocking=True)
        inputs = preprocess_vision(loaded.processor, batch["images"], loaded.device)
        logits = model(inputs["pixel_values"], inputs["image_grid_thw"])[0]
        accumulator.update(logits, density, fixation)
        remaining = collect_examples - len(examples)
        for index in range(min(max(0, remaining), len(logits))):
            examples.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "image": batch["images"][index].copy(),
                    "density": density[index].detach().cpu(),
                    "fixation": fixation[index].detach().cpu(),
                    "logits": logits[index].detach().cpu(),
                }
            )
    return accumulator.compute(), examples


def _student_optimizer(
    model: VisionOpticalSaliencyStudent, settings: Any
) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {
        "phase": [],
        "router": [],
        "electronic": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_phase" in name:
            groups["phase"].append(parameter)
        elif "router" in name:
            groups["router"].append(parameter)
        else:
            groups["electronic"].append(parameter)
    values = [
        {
            "params": groups["phase"],
            "lr": settings.phase_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": groups["router"],
            "lr": settings.router_learning_rate,
            "weight_decay": settings.weight_decay,
        },
        {
            "params": groups["electronic"],
            "lr": settings.student_learning_rate,
            "weight_decay": settings.weight_decay,
        },
    ]
    return torch.optim.AdamW([value for value in values if value["params"]])


class _TeacherMapStore:
    def __init__(self, settings: Any) -> None:
        self.root = settings.artifact_cache_dir / "teacher_logits" / "train"
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Teacher map cache is missing: {self.root}. "
                "Run --phase cache_teacher_maps."
            )

    def get(
        self, sample_ids: list[str], device: torch.device
    ) -> torch.Tensor:
        values = []
        for sample_id in sample_ids:
            path = self.root / f"{sample_id.split('/')[-1]}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Teacher map cache entry missing: {path}")
            values.append(torch.load(path, map_location="cpu").float())
        return torch.stack(values).to(device, non_blocking=True)


def _autocast(settings: Any, device: torch.device):
    if not settings.amp_enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}",
        flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(
            f"  {row['name']} shape={row['shape']} "
            f"params={row['parameters']:,}",
            flush=True,
        )
