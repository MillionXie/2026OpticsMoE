from __future__ import annotations

import csv
import html
import json
import math
import platform
import subprocess
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.training import (
    EMA,
    seed_everything,
)

from .assets import grid_centers, load_icons, render_grid
from .datasets import OpenMojiEditingDataset, collate_samples, load_prompt_cache
from .metrics import MetricAccumulator
from .modeling import OpenMojiOpticalEditor, build_model
from .objectives import editing_objective
from .scenes import TASKS
from .settings import Settings


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _environment(device: torch.device) -> dict[str, Any]:
    value = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        value["gpu_name"] = torch.cuda.get_device_name(device)
        value["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        try:
            value["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        except Exception as error:  # pragma: no cover
            value["nvidia_smi_error"] = repr(error)
    return value


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "source_image",
        "source_grid",
        "target_grid",
        "edit_grid",
        "preserve_grid",
        "task_index",
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    return result


def build_loaders(settings: Settings) -> tuple[DataLoader[Any], DataLoader[Any]]:
    prompts = load_prompt_cache(settings.prompt_cache_path)
    train_dataset = OpenMojiEditingDataset(settings.train_manifest, settings, prompts)
    test_dataset = OpenMojiEditingDataset(settings.test_manifest, settings, prompts)
    common = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "collate_fn": collate_samples,
    }
    generator = torch.Generator().manual_seed(settings.seed)
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def _parameter_groups(model: nn.Module, settings: Settings) -> list[dict[str, Any]]:
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


def _schedule(total_steps: int):
    warmup = max(1, int(0.05 * total_steps))

    def value(step: int) -> float:
        if step < warmup:
            return max(1.0e-6, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
            values.append((parameter.detach().cpu().float() - initial[name]).square().mean())
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
    train_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "openmoji_qwen_language2_vision2_grid_editor_v1",
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


def _error_panel(target: np.ndarray, prediction: np.ndarray, settings: Settings) -> Image.Image:
    image = Image.new("RGB", (settings.image_size, settings.image_size), "white")
    draw = ImageDraw.Draw(image)
    bounds = [
        round(index * settings.image_size / settings.grid_size)
        for index in range(settings.grid_size + 1)
    ]
    for row, col in zip(*np.nonzero(target != prediction)):
        draw.rectangle(
            (bounds[col] + 1, bounds[row] + 1, bounds[col + 1] - 2, bounds[row + 1] - 2),
            fill=(255, 220, 220),
            outline=(210, 0, 0),
            width=2,
        )
        draw.text((bounds[col] + 3, bounds[row] + 3), f"{target[row,col]}>{prediction[row,col]}", fill=(120, 0, 0))
    return image


def _sample_card(sample: dict[str, Any], settings: Settings, icons: dict[int, Image.Image]) -> Image.Image:
    panels = [
        render_grid(sample["source"], settings, icons),
        render_grid(sample["target"], settings, icons),
        render_grid(sample["prediction"], settings, icons),
        _error_panel(sample["target"], sample["prediction"], settings),
    ]
    width = settings.image_size * 4
    header = 72
    canvas = Image.new("RGB", (width, header + settings.image_size + 24), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{sample['sample_id']} | task={sample['task']} | {sample['instruction']}"
    draw.multiline_text((8, 6), "\n".join(textwrap.wrap(title, width=115)), fill="black", spacing=3)
    for index, (label, panel) in enumerate(zip(("INPUT", "TARGET", "PREDICTION", "ERROR CELLS"), panels)):
        x = index * settings.image_size
        canvas.paste(panel, (x, header))
        draw.text((x + 6, header + settings.image_size + 4), label, fill="black")
    return canvas


def _save_gallery(
    directory: Path,
    galleries: dict[str, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    icons = load_icons(settings)
    links = []
    for task in TASKS:
        samples = galleries.get(task, [])
        if not samples:
            continue
        cards = [_sample_card(sample, settings, icons) for sample in samples]
        sheet = Image.new("RGB", (cards[0].width, sum(card.height for card in cards)), "white")
        y = 0
        for card in cards:
            sheet.paste(card, (0, y))
            y += card.height
        filename = f"{task}_examples.png"
        sheet.save(directory / filename, optimize=True)
        links.append((task, filename))
    body = ["<!doctype html><meta charset='utf-8'><title>OpenMoji epoch examples</title>"]
    body.append("<style>body{font-family:sans-serif} img{max-width:100%;border:1px solid #bbb}</style>")
    for task, filename in links:
        body.append(f"<h2>{html.escape(task)}</h2><img src='{html.escape(filename)}'>")
    (directory / "index.html").write_text("\n".join(body), encoding="utf-8")


@torch.inference_mode()
def _evaluate(
    model: OpenMojiOpticalEditor,
    loader: DataLoader[Any],
    settings: Settings,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    accumulator = MetricAccumulator()
    predictions: list[dict[str, Any]] = []
    galleries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in loader:
        batch = _move(raw, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=settings.amp_enabled and device.type == "cuda",
        ):
            outputs = model(batch["source_image"], batch["prompt_hidden"])
        records, composed, _ = accumulator.update(outputs, batch)
        predictions.extend(records)
        for index, task in enumerate(batch["task"]):
            if len(galleries[task]) >= settings.visualization_samples_per_task:
                continue
            galleries[task].append(
                {
                    "sample_id": batch["sample_id"][index],
                    "task": task,
                    "instruction": batch["instruction"][index],
                    "source": batch["source_grid"][index].detach().cpu().numpy(),
                    "target": batch["target_grid"][index].detach().cpu().numpy(),
                    "prediction": composed[index].detach().cpu().numpy(),
                }
            )
    return accumulator.compute(), predictions, galleries


def _test_row(epoch: int, result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "elapsed_seconds": result["elapsed_seconds"]}
    for group, metrics in result["metrics"].items():
        for name, value in metrics.items():
            row[f"{group}_{name}"] = value
    return row


def _run_epoch_test(
    epoch: int,
    model: OpenMojiOpticalEditor,
    ema: EMA,
    loader: DataLoader[Any],
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    backup = ema.copy_to(model)
    model.eval()
    started = time.perf_counter()
    try:
        metrics, predictions, galleries = _evaluate(model, loader, settings, device)
    finally:
        EMA.restore(model, backup)
    result = {
        "epoch": epoch,
        "weights": "EMA at the end of this epoch",
        "test_role": "monitoring only; never used for checkpoint selection",
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    epoch_dir = settings.output_dir / "epoch_tests" / f"epoch_{epoch:03d}"
    _json(epoch_dir / "metrics.json", result)
    with (epoch_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _save_gallery(epoch_dir / "examples", galleries, settings)
    return result


def train(settings: Settings, device: torch.device) -> dict[str, Any]:
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _json(settings.output_dir / "resolved_config.json", settings.to_dict())
    _json(settings.output_dir / "environment.json", _environment(device))
    train_loader, test_loader = build_loaders(settings)
    model = build_model(settings, device)
    _json(settings.output_dir / "model.json", model.architecture_report())
    optimizer = AdamW(_parameter_groups(model, settings), weight_decay=settings.weight_decay)
    scheduler = LambdaLR(optimizer, _schedule(settings.epochs * len(train_loader)))
    ema = EMA(model, settings.ema_decay)
    checkpoint_path = settings.output_dir / "checkpoints" / "last.pt"
    history = _read_csv(settings.output_dir / "train_log.csv")
    test_history = _read_csv(settings.output_dir / "test_log.csv")
    start_epoch = 1
    global_step = 0
    if settings.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.load_state_dict(payload["ema_model"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload.get("global_step", (start_epoch - 1) * len(train_loader)))
        print(f"resumed epoch={start_epoch-1} global_step={global_step}", flush=True)
    initial_phase = _phase_snapshot(model)
    best_loss = min((float(row["total"]) for row in history), default=float("inf"))
    started = time.perf_counter()
    for epoch in range(start_epoch, settings.epochs + 1):
        phase_trainable = epoch > settings.warmup_electronic_epochs
        model.set_phase_trainable(phase_trainable)
        model.train()
        model.vision_stem.eval()
        totals: dict[str, float] = defaultdict(float)
        count = 0
        epoch_started = time.perf_counter()
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, device)
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
            batch_count = len(batch["task"])
            count += batch_count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_count
            totals["grad_norm"] += float(grad_norm) * batch_count
            if batch_index % settings.log_interval == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch}/{settings.epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={totals['total']/count:.5f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"phase={'on' if phase_trainable else 'warmup'}",
                    flush=True,
                )
        row: dict[str, Any] = {
            "epoch": epoch,
            "samples": count,
            **{name: value / count for name, value in totals.items()},
            "phase_trainable": phase_trainable,
            "phase_delta_rms": _phase_rms(model, initial_phase),
            "prompt_vision_gate": float(torch.sigmoid(model.prompt_vision_gate).detach()),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        _checkpoint(checkpoint_path, model, optimizer, scheduler, ema, epoch, global_step, settings, row)
        test_result = _run_epoch_test(epoch, model, ema, test_loader, settings, device)
        for name, value in test_result["metrics"]["overall"].items():
            row[f"test_{name}"] = value
        row["test_seconds"] = test_result["elapsed_seconds"]
        history.append(row)
        test_history.append(_test_row(epoch, test_result))
        _write_csv(settings.output_dir / "train_log.csv", history)
        _write_csv(settings.output_dir / "test_log.csv", test_history)
        _checkpoint(checkpoint_path, model, optimizer, scheduler, ema, epoch, global_step, settings, row)
        if float(row["total"]) < best_loss:
            best_loss = float(row["total"])
            _checkpoint(settings.output_dir / "checkpoints" / "best_train_loss.pt", model, optimizer, scheduler, ema, epoch, global_step, settings, row)
        overall = test_result["metrics"]["overall"]
        print(
            f"epoch={epoch}/{settings.epochs} test scene_exact={overall['scene_exact_match']:.5f} "
            f"changed_accuracy={overall['changed_cell_accuracy']:.5f} "
            f"object_f1={overall['object_f1']:.5f} edit_iou={overall['edit_grid_iou']:.5f}",
            flush=True,
        )
    phases = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "raw_phase" in name
    }
    torch.save(
        {
            "raw_phase": phases,
            "phase_radians": {
                name: torch.sigmoid(value.float()) * (2.0 * math.pi)
                for name, value in phases.items()
            },
        },
        settings.output_dir / "optical_phases.pt",
    )
    summary = {
        "epochs": settings.epochs,
        "best_training_loss": best_loss,
        "checkpoint_policy": "last epoch EMA; per-epoch test never selects checkpoints",
        "elapsed_seconds": time.perf_counter() - started,
        "final": history[-1],
    }
    _json(settings.output_dir / "training_summary.json", summary)
    return summary


def _load_official(settings: Settings, device: torch.device) -> tuple[OpenMojiOpticalEditor, dict[str, Any]]:
    path = settings.output_dir / "checkpoints" / "last.pt"
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
    _, loader = build_loaders(settings)
    model, payload = _load_official(settings, device)
    started = time.perf_counter()
    metrics, predictions, galleries = _evaluate(model, loader, settings, device)
    result = {
        "checkpoint": str(settings.output_dir / "checkpoints" / "last.pt"),
        "checkpoint_epoch": payload["epoch"],
        "weights": "final epoch EMA",
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    _json(settings.output_dir / "test_metrics.json", result)
    with (settings.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _save_gallery(settings.output_dir / "test_examples", galleries, settings)
    return result


__all__ = ["build_loaders", "test", "train"]
