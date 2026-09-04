from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .datasets import SyntheticEditingDataset, collate_samples, load_prompt_cache
from .metrics import MetricAccumulator, compose_prediction
from .modeling import InstructionOpticalEditor, build_model
from .objectives import editing_objective
from .scenes import PALETTE
from .settings import Settings


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _environment(device: torch.device) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        value.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
        try:
            value["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi"], text=True, timeout=10
            )
        except Exception as error:  # pragma: no cover - diagnostic only
            value["nvidia_smi_error"] = repr(error)
    return value


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "source_image",
        "source_classes",
        "target_classes",
        "edit_mask",
        "preserve_mask",
        "task_index",
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    return result


def build_loaders(settings: Settings) -> tuple[DataLoader[Any], DataLoader[Any]]:
    cache = load_prompt_cache(settings.prompt_cache_path)
    train_dataset = SyntheticEditingDataset(settings.train_manifest, settings, cache)
    test_dataset = SyntheticEditingDataset(settings.test_manifest, settings, cache)
    generator = torch.Generator().manual_seed(settings.seed)
    common = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_samples,
        "persistent_workers": settings.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader


def _parameter_groups(model: InstructionOpticalEditor, settings: Settings) -> list[dict[str, Any]]:
    groups: dict[str, list[nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_phase" in name:
            group = "phase"
        elif ".router." in name:
            group = "router"
        elif "readout" in name or "output_adapter" in name:
            group = "readout"
        elif name.startswith(("decoder.", "editor.", "language_pool.", "prompt_to_vision.", "post_film.", "coordinate_projection.", "task_head.")):
            group = "decoder"
        elif "input_adapter" in name:
            group = "adapter"
        else:
            group = "base"
        groups[group].append(parameter)
    rates = {
        "phase": settings.phase_learning_rate,
        "router": settings.router_learning_rate,
        "readout": settings.readout_learning_rate,
        "decoder": settings.decoder_learning_rate,
        "adapter": settings.adapter_learning_rate,
        "base": settings.learning_rate,
    }
    return [
        {"params": parameters, "lr": rates[name], "name": name}
        for name, parameters in groups.items()
        if parameters
    ]


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.shadow = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name in self.names
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name not in self.names:
                continue
            source = parameter.detach().cpu()
            self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, values: dict[str, torch.Tensor]) -> None:
        missing = self.names.difference(values)
        if missing:
            raise RuntimeError(f"EMA checkpoint is missing {len(missing)} trainable parameters")
        self.shadow = {
            name: values[name].detach().cpu().clone()
            for name in self.names
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            backup[name] = parameter.detach().cpu().clone()
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name].to(device=parameter.device, dtype=parameter.dtype))


def _schedule(total_steps: int, warmup_steps: int):
    def value(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1.0e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return value


def _phase_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if "raw_phase" in name
    }


def _phase_rms(model: nn.Module, initial: dict[str, torch.Tensor]) -> float:
    values = []
    for name, parameter in model.named_parameters():
        if name in initial:
            delta = parameter.detach().cpu().float() - initial[name]
            values.append(delta.square().mean())
    return float(torch.stack(values).mean().sqrt()) if values else 0.0


def _checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    ema: EMA,
    epoch: int,
    global_step: int,
    settings: Settings,
    train_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "qwen_lm_language2_vision2_structured_editor_v1",
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "settings": settings.to_dict(),
            "train_metrics": train_metrics,
        },
        path,
    )


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@torch.inference_mode()
def _evaluate_model(
    model: InstructionOpticalEditor,
    test_loader: DataLoader[Any],
    settings: Settings,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    accumulator = MetricAccumulator()
    predictions: list[dict[str, Any]] = []
    visualization: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch in test_loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=settings.amp_enabled and device.type == "cuda",
        ):
            outputs = model(batch["source_image"], batch["prompt_hidden"])
        predictions.extend(accumulator.update(outputs, batch))
        if len(visualization) < settings.visualization_samples:
            composed, _, _ = compose_prediction(
                outputs["palette_logits"], outputs["edit_logits"], batch["source_classes"]
            )
            for index in range(
                min(len(composed), settings.visualization_samples - len(visualization))
            ):
                visualization.append(
                    (
                        batch["source_classes"][index].detach().cpu(),
                        batch["target_classes"][index].detach().cpu(),
                        composed[index].detach().cpu(),
                    )
                )
    return accumulator.compute(), predictions, visualization


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _test_log_row(epoch: int, result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "weights": result["weights"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    for group, metrics in result["metrics"].items():
        for name, value in metrics.items():
            row[f"{group}_{name}"] = value
    return row


def _attach_overall_test(row: dict[str, Any], result: dict[str, Any]) -> None:
    for name, value in result["metrics"]["overall"].items():
        row[f"test_{name}"] = value
    row["test_seconds"] = result["elapsed_seconds"]


def _run_epoch_test(
    settings: Settings,
    model: InstructionOpticalEditor,
    ema: EMA,
    test_loader: DataLoader[Any],
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    backup = ema.copy_to(model)
    model.eval()
    try:
        metrics, predictions, visualization = _evaluate_model(
            model, test_loader, settings, device
        )
    finally:
        EMA.restore(model, backup)
    result = {
        "checkpoint_epoch": epoch,
        "weights": "EMA at the end of this epoch",
        "test_role": "monitoring only; never used for checkpoint selection",
        "validation_split": False,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    epoch_dir = settings.output_dir / "epoch_tests"
    stem = f"epoch_{epoch:03d}"
    _json(epoch_dir / f"{stem}_metrics.json", result)
    _write_predictions(epoch_dir / f"{stem}_predictions.jsonl", predictions)
    _save_visualization(epoch_dir / f"{stem}_examples.png", visualization)
    return result


def train(settings: Settings, device: torch.device) -> dict[str, Any]:
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _json(settings.output_dir / "resolved_config.json", settings.to_dict())
    _json(settings.output_dir / "environment.json", _environment(device))
    train_loader, test_loader = build_loaders(settings)
    model = build_model(settings, device)
    _json(settings.output_dir / "model.json", model.architecture_report())
    parameter_groups = _parameter_groups(model, settings)
    optimizer = AdamW(parameter_groups, weight_decay=settings.weight_decay)
    total_steps = settings.epochs * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        _schedule(total_steps, max(1, int(0.05 * total_steps))),
    )
    ema = EMA(model, settings.ema_decay)
    checkpoint_path = settings.output_dir / "checkpoints" / "last.pt"
    history = _read_history(settings.output_dir / "train_log.csv")
    test_history = _read_history(settings.output_dir / "test_log.csv")
    completed_test_epochs = {
        int(float(row["epoch"])) for row in test_history if row.get("epoch")
    }
    start_epoch = 1
    global_step = 0
    if settings.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.load_state_dict(payload["ema_model"])
        completed_epoch = int(payload["epoch"])
        start_epoch = completed_epoch + 1
        global_step = int(payload.get("global_step", completed_epoch * len(train_loader)))
        print(
            f"resumed checkpoint={checkpoint_path} epoch={completed_epoch} "
            f"global_step={global_step}",
            flush=True,
        )
        if completed_epoch not in completed_test_epochs:
            result = _run_epoch_test(
                settings, model, ema, test_loader, device, completed_epoch
            )
            test_history.append(_test_log_row(completed_epoch, result))
            for row in history:
                if int(float(row["epoch"])) == completed_epoch:
                    _attach_overall_test(row, result)
            _write_history(settings.output_dir / "test_log.csv", test_history)
            _write_history(settings.output_dir / "train_log.csv", history)
            overall = result["metrics"]["overall"]
            print(
                f"epoch={completed_epoch}/{settings.epochs} test "
                f"pixel_accuracy={overall['pixel_accuracy']:.5f} "
                f"changed_accuracy={overall['changed_pixel_accuracy']:.5f} "
                f"edit_iou={overall['edit_mask_iou']:.5f}",
                flush=True,
            )
    initial_phase = _phase_snapshot(model)
    best_loss = min(
        (float(row["total"]) for row in history if row.get("total") not in {None, ""}),
        default=float("inf"),
    )
    started = time.perf_counter()

    for epoch in range(start_epoch, settings.epochs + 1):
        phase_trainable = epoch > settings.warmup_electronic_epochs
        model.set_phase_trainable(phase_trainable)
        model.train()
        model.vision_stem.eval()
        totals: dict[str, float] = defaultdict(float)
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=settings.amp_enabled and device.type == "cuda",
            ):
                outputs = model(batch["source_image"], batch["prompt_hidden"])
                losses = editing_objective(outputs, batch, settings)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                settings.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            ema.update(model)
            global_step += 1
            count = len(batch["task"])
            sample_count += count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
            totals["grad_norm"] += float(grad_norm) * count
            if batch_index % settings.log_interval == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch}/{settings.epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={totals['total']/sample_count:.5f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"phase={'on' if phase_trainable else 'warmup'}",
                    flush=True,
                )
        row = {
            "epoch": epoch,
            "samples": sample_count,
            **{name: value / max(1, sample_count) for name, value in totals.items()},
            "phase_trainable": phase_trainable,
            "phase_delta_rms": _phase_rms(model, initial_phase),
            "prompt_vision_gate": float(torch.sigmoid(model.prompt_vision_gate).detach()),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        _checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            ema,
            epoch,
            global_step,
            settings,
            row,
        )
        test_result = _run_epoch_test(
            settings, model, ema, test_loader, device, epoch
        )
        _attach_overall_test(row, test_result)
        test_history.append(_test_log_row(epoch, test_result))
        history.append(row)
        _write_history(settings.output_dir / "train_log.csv", history)
        _write_history(settings.output_dir / "test_log.csv", test_history)
        _checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            ema,
            epoch,
            global_step,
            settings,
            row,
        )
        overall = test_result["metrics"]["overall"]
        print(
            f"epoch={epoch}/{settings.epochs} test "
            f"pixel_accuracy={overall['pixel_accuracy']:.5f} "
            f"changed_accuracy={overall['changed_pixel_accuracy']:.5f} "
            f"edit_iou={overall['edit_mask_iou']:.5f} "
            f"success={overall['sample_success']:.5f}",
            flush=True,
        )
        if row["total"] < best_loss:
            best_loss = row["total"]
            _checkpoint(
                settings.output_dir / "checkpoints" / "best_train_loss.pt",
                model,
                optimizer,
                scheduler,
                ema,
                epoch,
                global_step,
                settings,
                row,
            )

    phase_artifacts = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "raw_phase" in name
    }
    torch.save(
        {
            "parameterization": "sigmoid_to_0_2pi",
            "raw_phase": phase_artifacts,
            "phase_radians": {
                name: torch.sigmoid(value.float()) * (2.0 * math.pi)
                for name, value in phase_artifacts.items()
            },
        },
        settings.output_dir / "optical_phases.pt",
    )
    summary = {
        "epochs": settings.epochs,
        "start_epoch_this_run": start_epoch,
        "samples_per_epoch": len(train_loader.dataset),
        "best_training_loss": best_loss,
        "official_checkpoint_policy": (
            "last_epoch_ema; test is monitored every epoch but never selects checkpoints"
        ),
        "per_epoch_test": True,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "final": history[-1],
    }
    _json(settings.output_dir / "training_summary.json", summary)
    return summary


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_official_model(settings: Settings, device: torch.device) -> tuple[InstructionOpticalEditor, dict[str, Any]]:
    path = settings.output_dir / "checkpoints" / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"Official last checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(settings, device)
    model.load_state_dict(payload["model"])
    state = model.state_dict()
    for name, value in payload["ema_model"].items():
        if name in state:
            state[name] = value.to(state[name].dtype)
    model.load_state_dict(state)
    model.eval()
    return model, payload


@torch.inference_mode()
def test(settings: Settings, device: torch.device) -> dict[str, Any]:
    _, test_loader = build_loaders(settings)
    model, checkpoint = _load_official_model(settings, device)
    started = time.perf_counter()
    metrics, predictions, visualization = _evaluate_model(
        model, test_loader, settings, device
    )
    result = {
        "checkpoint": str(settings.output_dir / "checkpoints" / "last.pt"),
        "checkpoint_epoch": checkpoint["epoch"],
        "weights": "EMA from the fixed final epoch",
        "test_selection": (
            "final official export; test was also monitored every epoch but never selected checkpoints"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    _json(settings.output_dir / "test_metrics.json", result)
    _write_predictions(settings.output_dir / "test_predictions.jsonl", predictions)
    _save_visualization(settings.output_dir / "test_examples.png", visualization)
    return result


def _save_visualization(
    path: Path,
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    if not rows:
        return
    strips = []
    for source, target, prediction in rows:
        panels = [PALETTE[value.numpy().astype("uint8")] for value in (source, target, prediction)]
        strips.append(torch.from_numpy(__import__("numpy").concatenate(panels, axis=1)))
    canvas = torch.cat(strips, dim=0).numpy().astype("uint8")
    Image.fromarray(canvas).save(path)


__all__ = ["build_loaders", "seed_everything", "test", "train"]
