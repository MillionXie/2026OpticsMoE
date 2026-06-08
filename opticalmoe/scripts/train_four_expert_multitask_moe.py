import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.data import create_dataloaders
from opticalmoe.optics.four_expert_geometry import FourExpertLayout
from opticalmoe.optics.four_expert_multitask_moe import (
    FourExpertMultitaskMoEClassifier,
)
from opticalmoe.training import save_checkpoint
from opticalmoe.training.four_expert_reporting import (
    build_architecture_report,
    save_architecture_report,
    save_initial_state,
)
from opticalmoe.training.multitask_engine import (
    evaluate_task,
    task_switching_evaluation,
    train_multitask_one_epoch,
)
from opticalmoe.training.multitask_progressive_schedule import (
    MultitaskProgressiveUnfreezingSchedule,
)
from opticalmoe.utils import load_config, save_json, set_seed
from opticalmoe.utils.run import create_run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the separate multitask four-expert OpticalMoE experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def configure_matplotlib(config: Dict) -> None:
    font_size = int(config.get("font_size", 12))
    dpi = int(config.get("dpi", 150))
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": max(14, font_size + 2),
            "axes.labelsize": font_size,
            "xtick.labelsize": max(11, font_size - 1),
            "ytick.labelsize": max(11, font_size - 1),
            "legend.fontsize": max(11, font_size - 1),
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )


def build_model(config: Dict, task_names: Sequence[str], task_num_classes: Dict[str, int]):
    layout_cfg = config.get("layout", {})
    optics_cfg = config.get("optics", {})
    prompt_cfg = config.get("prompt", {})
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
    task_head_configs = {
        task["name"].lower(): dict(task.get("head", {}))
        for task in config["training"]["multitask"]["tasks"]
    }
    distances = optics_cfg.get("distances_m", {})
    layout = FourExpertLayout(
        canvas_height=int(layout_cfg.get("canvas_height", 700)),
        canvas_width=int(layout_cfg.get("canvas_width", 700)),
        input_size=int(layout_cfg.get("input_size", 200)),
        expert_size=int(layout_cfg.get("expert_size", 200)),
        prompt_cell_size=int(layout_cfg.get("prompt_cell_size", 300)),
        gap_pixels=int(layout_cfg.get("gap_pixels", 100)),
        outer_margin=int(layout_cfg.get("outer_margin", 100)),
    )
    return FourExpertMultitaskMoEClassifier(
        task_names=task_names,
        task_num_classes=task_num_classes,
        task_head_configs=task_head_configs,
        layout=layout,
        wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
        pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
        input_size=int(layout_cfg.get("input_size", 200)),
        num_layers=int(optics_cfg.get("num_layers", 5)),
        distances_m={
            "input_to_prompt": float(distances.get("input_to_prompt", 0.20)),
            "prompt_to_expert": float(distances.get("prompt_to_expert", 0.20)),
            "inter_layer": float(distances.get("inter_layer", 0.05)),
            "layer5_to_fc": float(distances.get("layer5_to_fc", 0.05)),
            "fc_to_detector": float(distances.get("fc_to_detector", 0.05)),
        },
        focal_length_m=float(optics_cfg.get("focal_length_m", 0.10)),
        aperture_mode=optics_cfg.get("aperture_mode", "hard"),
        phase_param=optics_cfg.get("phase_param", "unconstrained"),
        expert_phase_init=optics_cfg.get(
            "expert_phase_init", "uniform_0_2pi"
        ),
        expert_init_std=float(optics_cfg.get("expert_init_std", 0.02)),
        global_fc_phase_init=optics_cfg.get("global_fc_phase_init", "identity"),
        global_fc_init_std=float(optics_cfg.get("global_fc_init_std", 0.02)),
        prompt_amplitude_init_logits=float(
            prompt_cfg.get("amplitude_init_logits", 2.0)
        ),
        train_prompt_phase_biases=bool(
            prompt_cfg.get("train_phase_biases", True)
        ),
        detector_size=int(detector_cfg.get("detector_size", 32)),
        detector_layout=detector_cfg.get("layout", "grid"),
        normalize_detector_energy=bool(
            readout_cfg.get("normalize_detector_energy", True)
        ),
        readout_type=readout_cfg.get("type", "optical_only"),
        logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
        readout_activation=readout_cfg.get("activation", "relu"),
        evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
    )


def optimizer_settings(config: Dict) -> Dict:
    cfg = config.get("optimizer", {})
    return {
        "type": cfg.get("type", "adamw").lower(),
        "lr": float(cfg.get("lr", 0.003)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "momentum": (
            float(cfg.get("momentum", 0.9))
            if cfg.get("type", "adamw").lower() == "sgd"
            else None
        ),
    }


def build_optimizer(model, config: Dict):
    settings = optimizer_settings(config)
    kwargs = {
        "lr": settings["lr"],
        "weight_decay": settings["weight_decay"],
    }
    if settings["type"] == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if settings["type"] == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if settings["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=settings["momentum"],
            **kwargs,
        )
    raise ValueError("optimizer.type must be adam, adamw, or sgd.")


def create_task_loaders(config: Dict, seed: int, force_smoke: bool):
    tasks = config["training"]["multitask"]["tasks"]
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    class_counts = {}
    for index, task in enumerate(tasks):
        name = task["name"].lower()
        dataset_cfg = dict(task["dataset"])
        if force_smoke:
            dataset_cfg["smoke_test"] = True
        train, val, test, classes = create_dataloaders(
            dataset_cfg,
            seed=seed + index,
        )
        train_loaders[name] = train
        val_loaders[name] = val
        test_loaders[name] = test
        class_counts[name] = classes
    return train_loaders, val_loaders, test_loaders, class_counts


def fixed_batch(loader, num_samples: int):
    for batch in loader:
        return batch[0][:num_samples], batch[1][:num_samples]
    raise RuntimeError("Validation loader is empty.")


@torch.no_grad()
def fixed_batch_loss_accuracy(model, batch, device, criterion, task_name: str):
    """Measure the epoch-0000 reference without scanning the full validation set."""

    images, targets = batch
    model.eval()
    logits = model(images.to(device), task_name=task_name)
    targets = targets.to(device)
    loss = criterion(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return float(loss.item()), float(accuracy.item())


@torch.no_grad()
def task_diagnostics(model, batch, device, task_name: str):
    images, targets = batch
    model.eval()
    logits, intermediates = model(
        images.to(device),
        task_name=task_name,
        return_intermediates=True,
    )
    return {
        "targets": targets.cpu(),
        "predictions": logits.argmax(dim=1).cpu(),
        "amplitudes": intermediates["prompt_amplitudes"].detach().cpu(),
        "powers": intermediates["prompt_powers"].detach().cpu(),
        "normalized_powers": intermediates[
            "normalized_prompt_powers"
        ].detach().cpu(),
        "expert_energy_ratios": intermediates[
            "expert_energy_ratios"
        ].mean(dim=0).detach().cpu(),
        "outside_energy_ratio": float(
            intermediates["outside_energy_ratio"].mean().item()
        ),
        "detector_energies": intermediates[
            "detector_energies"
        ].mean(dim=0).detach().cpu(),
        "intermediates": intermediates,
    }


def write_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key, "") for key in fields} for row in rows]
        )


def log_image(field: torch.Tensor) -> np.ndarray:
    value = field
    if torch.is_complex(value):
        value = torch.abs(value).square()
    if value.ndim == 3:
        value = value[0]
    array = value.detach().cpu().float().numpy()
    return np.log10(array / (array.max() + 1e-12) + 1e-8)


def phase_image(phase: torch.Tensor) -> np.ndarray:
    return torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()


def save_multitask_phase_visualizations(model, run_dir: Path, epoch: int):
    epoch_name = f"epoch_{epoch:04d}"
    phase_dir = run_dir / "phases" / epoch_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        model.num_layers,
        4,
        figsize=(12, 2.6 * model.num_layers),
        squeeze=False,
    )
    for layer_index, layer in enumerate(model.expert_layers):
        phases = layer.get_phase_wrapped().detach().cpu().numpy()
        for expert_index in range(4):
            axes[layer_index, expert_index].imshow(
                phases[expert_index],
                cmap="twilight",
                vmin=0.0,
                vmax=2.0 * math.pi,
            )
            axes[layer_index, expert_index].set_title(
                f"Layer {layer_index + 1}, E{expert_index}"
            )
            axes[layer_index, expert_index].axis("off")
    fig.suptitle(f"Shared Expert Phase Masks, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(phase_dir / "shared_expert_phase_layers.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        phase_image(model.global_fc.get_phase_wrapped()),
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(f"Shared Global FC Phase, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(phase_dir / "global_fc_phase.png")
    plt.close(fig)


def save_task_epoch_visualization(
    diagnostics: Dict,
    run_dir: Path,
    epoch: int,
    task_name: str,
):
    task_dir = run_dir / "light_fields" / f"epoch_{epoch:04d}" / task_name
    prompt_dir = run_dir / "prompt" / f"epoch_{epoch:04d}" / task_name
    sample_dir = run_dir / "sample_outputs" / f"epoch_{epoch:04d}" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    intermediates = diagnostics["intermediates"]
    fields = [
        (
            "00_input_amplitude.png",
            intermediates["input_amplitude"],
            f"{task_name}: Input Amplitude",
        ),
        (
            "01_after_input_to_prompt.png",
            intermediates["after_input_to_prompt"],
            f"{task_name}: After Input-to-Prompt",
        ),
        (
            "02_after_prompt.png",
            intermediates["after_prompt"],
            f"{task_name}: After Task Prompt",
        ),
        (
            "03_expert_entrance.png",
            intermediates["expert_entrance_intensity"],
            f"{task_name}: Expert Entrance",
        ),
    ]
    for layer_index, field in enumerate(intermediates["after_each_layer"], start=1):
        fields.append(
            (
                f"{layer_index + 3:02d}_after_expert_layer_{layer_index}.png",
                field,
                f"{task_name}: After Expert Layer {layer_index}",
            )
        )
    fields.extend(
        [
            (
                "09_after_global_fc.png",
                intermediates["after_global_fc"],
                f"{task_name}: After Global FC",
            ),
            (
                "10_detector_plane.png",
                intermediates["detector_intensity"],
                f"{task_name}: Detector Plane",
            ),
        ]
    )
    for name, field, title in fields:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(log_image(field), cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(task_dir / name)
        plt.close(fig)

    fig, axes = plt.subplots(
        int(math.ceil(len(fields) / 4)),
        4,
        figsize=(15, 3.8 * int(math.ceil(len(fields) / 4))),
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, (_name, field, title) in zip(axes, fields):
        ax.imshow(log_image(field), cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(fields) :]:
        ax.axis("off")
    fig.suptitle(f"{task_name}: Light Field Overview, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(task_dir / "overview.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        phase_image(intermediates["prompt_phase"]),
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(f"{task_name}: Prompt Phase, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(prompt_dir / "prompt_phase.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        intermediates["prompt_amplitude_map"].detach().cpu().numpy(),
        cmap="viridis",
    )
    ax.set_title(f"{task_name}: Prompt Amplitude Map, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(prompt_dir / "prompt_amplitude_map.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(np.arange(4), diagnostics["amplitudes"].numpy())
    axes[0].set_xticks(np.arange(4))
    axes[0].set_xticklabels([f"E{index}" for index in range(4)])
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Prompt Amplitudes")
    axes[1].bar(np.arange(4), diagnostics["expert_energy_ratios"].numpy())
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels([f"E{index}" for index in range(4)])
    axes[1].set_ylabel("Energy / total")
    axes[1].set_title("Expert Energy")
    axes[2].bar(np.arange(len(diagnostics["detector_energies"])), diagnostics["detector_energies"].numpy())
    axes[2].set_xlabel("Class detector")
    axes[2].set_ylabel("Energy")
    axes[2].set_title("Detector Energies")
    fig.suptitle(f"{task_name}: Prompt/Energy Diagnostics, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(sample_dir / "prompt_energy_detector_bars.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(4), diagnostics["amplitudes"].numpy())
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"E{index}" for index in range(4)])
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{task_name}: Prompt Amplitudes")
    fig.tight_layout()
    fig.savefig(prompt_dir / "prompt_amplitude_bar.png")
    plt.close(fig)

    images = intermediates["input_amplitude"].detach().cpu()
    targets = diagnostics["targets"]
    predictions = diagnostics["predictions"]
    count = min(int(images.shape[0]), 8)
    if count > 0:
        fig, axes = plt.subplots(1, count, figsize=(2.4 * count, 2.8))
        axes = np.asarray(axes).reshape(-1)
        for idx in range(count):
            axes[idx].imshow(images[idx].numpy(), cmap="gray")
            axes[idx].set_title(
                f"y={int(targets[idx])}, pred={int(predictions[idx])}"
            )
            axes[idx].axis("off")
        fig.suptitle(f"{task_name}: Fixed Validation Samples, Epoch {epoch}")
        fig.tight_layout()
        fig.savefig(sample_dir / "sample_predictions.png")
        plt.close(fig)
    with open(sample_dir / "sample_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task_name": task_name,
                "targets": targets[:count].tolist(),
                "predictions": predictions[:count].tolist(),
            },
            handle,
            indent=2,
        )

    with open(task_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task_name": task_name,
                "prompt_amplitudes": diagnostics["amplitudes"].tolist(),
                "prompt_powers": diagnostics["powers"].tolist(),
                "normalized_prompt_powers": diagnostics["normalized_powers"].tolist(),
                "expert_energy_ratios": diagnostics["expert_energy_ratios"].tolist(),
                "outside_energy_ratio": diagnostics["outside_energy_ratio"],
                "detector_energies": diagnostics["detector_energies"].tolist(),
            },
            handle,
            indent=2,
        )


def plot_history(rows, task_names, run_dir):
    epochs = sorted({int(row["epoch"]) for row in rows})
    for filename, prefix, ylabel, title in [
        (
            "task_prompt_amplitude_history.png",
            "amp",
            "Amplitude",
            "Task-Specific Prompt Amplitudes",
        ),
        (
            "task_prompt_power_history.png",
            "norm_power",
            "Normalized amplitude squared",
            "Task-Specific Prompt Power",
        ),
        (
            "task_expert_energy_ratio_history.png",
            "expert_energy_ratio",
            "Energy / total",
            "Task Expert Energy Ratios",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for task_name in task_names:
            task_rows = {
                int(row["epoch"]): row
                for row in rows
                if row["task_name"] == task_name
            }
            for expert in range(4):
                key = f"{prefix}_E{expert}"
                ax.plot(
                    epochs,
                    [task_rows[epoch][key] for epoch in epochs],
                    marker="o",
                    label=f"{task_name} E{expert}",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(run_dir / filename)
        plt.close(fig)


def plot_train_val(metrics_rows, task_names, run_dir):
    epochs = [row["epoch"] for row in metrics_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for task_name in task_names:
        axes[0].plot(
            epochs,
            [row[f"{task_name}_train_loss"] for row in metrics_rows],
            label=f"{task_name} train",
        )
        axes[0].plot(
            epochs,
            [row[f"{task_name}_val_loss"] for row in metrics_rows],
            linestyle="--",
            label=f"{task_name} val",
        )
        axes[1].plot(
            epochs,
            [row[f"{task_name}_train_acc"] for row in metrics_rows],
            label=f"{task_name} train",
        )
        axes[1].plot(
            epochs,
            [row[f"{task_name}_val_acc"] for row in metrics_rows],
            linestyle="--",
            label=f"{task_name} val",
        )
    axes[0].set_title("Multitask Loss")
    axes[1].set_title("Multitask Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "multitask_train_val_curves.png")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    if config.get("training", {}).get("mode") != "multitask":
        raise ValueError(
            "Use train_four_expert_moe_v2.py for single-task experiments. "
            "This script requires training.mode=multitask."
        )
    seed = int(config.get("seed", 7))
    set_seed(seed)
    configure_matplotlib(config.get("visualization", {}))
    run_name = args.run_name or config.get("experiment", {}).get(
        "run_name", "four_expert_multitask"
    )
    run_dir = create_run_dir(run_name, base_dir=str(PROJECT_ROOT / "runs"))
    shutil.copyfile(args.config, run_dir / "config.yaml")

    train_loaders, val_loaders, test_loaders, class_counts = create_task_loaders(
        config,
        seed,
        force_smoke=args.smoke_test,
    )
    task_names = list(train_loaders.keys())
    task_num_classes = dict(class_counts)
    device = choose_device(args.device or config.get("device", "auto"))
    model = build_model(config, task_names, task_num_classes).to(device)
    optimizer = build_optimizer(model, config)
    optimizer_cfg = optimizer_settings(config)
    criterion = nn.CrossEntropyLoss()

    progressive_cfg = config["training"].get("progressive", {})
    schedule = MultitaskProgressiveUnfreezingSchedule(
        num_layers=model.num_layers,
        enabled=bool(progressive_cfg.get("enabled", True)),
        order=progressive_cfg.get("order", "backward"),
        stage_epochs=progressive_cfg.get(
            "stage_epochs", [3, 3, 3, 3, 3, 10]
        ),
        train_task_prompts_always=bool(
            progressive_cfg.get(
                "train_task_prompts_always",
                progressive_cfg.get("train_prompt_always", True),
            )
        ),
        train_global_fc_always=bool(
            progressive_cfg.get("train_global_fc_always", True)
        ),
    )
    num_epochs = (
        int(args.epochs)
        if args.epochs is not None
        else (
            schedule.total_epochs
            if schedule.enabled
            else int(config["training"].get("epochs", 25))
        )
    )
    fixed_batches = {
        name: fixed_batch(
            val_loaders[name],
            int(config.get("visualization", {}).get("num_samples", 2)),
        )
        for name in task_names
    }

    print(f"device: {device}")
    print(f"tasks: {task_names}, task classes: {task_num_classes}")
    for task_name in task_names:
        print(
            f"  {task_name} data: "
            f"train={len(train_loaders[task_name].dataset)} samples/"
            f"{len(train_loaders[task_name])} batches, "
            f"val={len(val_loaders[task_name].dataset)} samples/"
            f"{len(val_loaders[task_name])} batches, "
            f"test={len(test_loaders[task_name].dataset)} samples/"
            f"{len(test_loaders[task_name])} batches"
        )
    print(
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"lr={optimizer_cfg['lr']}, "
        f"weight_decay={optimizer_cfg['weight_decay']}"
    )

    report = build_architecture_report(
        model,
        config,
        optimizer_cfg,
        training_mode="multitask",
        task_names=task_names,
    )
    save_architecture_report(report, run_dir)

    initial_rows = []
    prompt_history = []
    visualization_enabled = bool(config.get("visualization", {}).get("enabled", True))
    for task_name in task_names:
        initial_val_loss, initial_val_acc = fixed_batch_loss_accuracy(
            model,
            fixed_batches[task_name],
            device,
            criterion,
            task_name,
        )
        diagnostics = task_diagnostics(
            model, fixed_batches[task_name], device, task_name
        )
        initial_dir = run_dir / "initial_state" / task_name
        payload = save_initial_state(
            model,
            diagnostics,
            initial_dir,
            val_loss=initial_val_loss,
            val_acc=initial_val_acc,
            task_name=task_name,
            save_images=visualization_enabled,
        )
        if payload.get("visualization_error"):
            visualization_enabled = False
            config.setdefault("visualization", {})["enabled"] = False
            print(
                "Initial visualization failed and later PNG visualizations "
                "were disabled. Training will continue. Error: "
                f"{payload['visualization_error']}"
            )
        target_dir = run_dir / "light_fields" / "epoch_0000" / task_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(initial_dir, target_dir)
        initial_rows.append(payload)
        row = {"epoch": 0, "task_name": task_name}
        for index in range(4):
            row[f"amp_E{index}"] = payload["prompt_amplitudes"][index]
            row[f"power_E{index}"] = payload["prompt_powers"][index]
            row[f"norm_power_E{index}"] = payload[
                "normalized_prompt_powers"
            ][index]
            row[f"expert_energy_ratio_E{index}"] = payload[
                "expert_energy_ratios"
            ][index]
        row["outside_energy_ratio"] = payload["outside_energy_ratio"]
        prompt_history.append(row)
    write_rows(run_dir / "initial_diagnostics.csv", initial_rows)

    metrics_rows = []
    task_metric_rows = []
    stage_records = []
    best_val_mean = -1.0
    previous_stage = None
    multitask_cfg = config["training"]["multitask"]
    evaluation_cfg = config["training"].get("evaluation", {})
    steps_per_epoch = multitask_cfg.get("steps_per_epoch")
    max_val_batches = evaluation_cfg.get("max_val_batches")
    max_test_batches = evaluation_cfg.get("max_test_batches")
    print_freq = int(
        multitask_cfg.get(
            "print_freq",
            config.get("experiment", {}).get("print_freq", 100),
        )
    )
    natural_steps = max(len(loader) for loader in train_loaders.values())
    effective_steps = (
        min(natural_steps, int(steps_per_epoch))
        if steps_per_epoch is not None and int(steps_per_epoch) > 0
        else natural_steps
    )
    print(
        f"updates per epoch: {effective_steps} "
        f"(natural full-dataset value: {natural_steps})"
    )
    print(
        "Each update processes one batch from every task before one "
        "backward/optimizer step."
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        stage_idx = schedule.stage_for_epoch(epoch) if schedule.enabled else 0
        stage_info = schedule.apply(model, stage_idx)
        if stage_idx != previous_stage:
            stage_records.append(stage_info)
            previous_stage = stage_idx
            print(
                f"stage {stage_idx}: active layers "
                f"{stage_info['active_layers'] or 'none'}"
            )

        train_result = train_multitask_one_epoch(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            task_names=task_names,
            loss_reduction=multitask_cfg.get("loss_reduction", "mean"),
            batches_per_update=int(
                multitask_cfg.get("batches_per_update", 1)
            ),
            balanced_sampling=bool(
                multitask_cfg.get("balanced_sampling", True)
            ),
            steps_per_epoch=steps_per_epoch,
            print_freq=print_freq,
        )
        train_duration_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "stage_idx": stage_idx,
            "total_loss": train_result["total_loss"],
            "joint_train_loss": train_result["joint_sample_loss"],
            "joint_train_acc": train_result["joint_accuracy"],
            "train_samples": train_result["samples"],
            "updates": train_result["steps"],
            "natural_updates": train_result["available_steps"],
            "train_duration_seconds": train_duration_seconds,
            "lr": optimizer.param_groups[0]["lr"],
            "active_layers": " ".join(
                str(value) for value in stage_info["active_layers"]
            )
            or "none",
        }
        current_diagnostics = {}
        val_accuracies = []
        val_loss_weighted_sum = 0.0
        val_correct_weighted_sum = 0.0
        val_sample_count = 0
        validation_start = time.perf_counter()
        for task_name in task_names:
            validation = evaluate_task(
                model,
                val_loaders[task_name],
                device,
                criterion,
                prompt_task=task_name,
                max_batches=max_val_batches,
            )
            row[f"{task_name}_train_loss"] = train_result[
                f"{task_name}_loss"
            ]
            row[f"{task_name}_train_acc"] = train_result[
                f"{task_name}_acc"
            ]
            row[f"{task_name}_val_loss"] = validation["loss"]
            row[f"{task_name}_val_acc"] = validation["accuracy"]
            row[f"{task_name}_val_samples"] = validation["samples"]
            val_accuracies.append(validation["accuracy"])
            val_loss_weighted_sum += validation["loss"] * validation["samples"]
            val_correct_weighted_sum += (
                validation["accuracy"] * validation["samples"]
            )
            val_sample_count += validation["samples"]
            task_metric_rows.append(
                {
                    "epoch": epoch,
                    "task_name": task_name,
                    "train_loss": train_result[f"{task_name}_loss"],
                    "train_acc": train_result[f"{task_name}_acc"],
                    "train_samples": train_result[f"{task_name}_samples"],
                    "val_loss": validation["loss"],
                    "val_acc": validation["accuracy"],
                    "val_samples": validation["samples"],
                }
            )
            diagnostics = task_diagnostics(
                model, fixed_batches[task_name], device, task_name
            )
            current_diagnostics[task_name] = diagnostics
            prompt_row = {"epoch": epoch, "task_name": task_name}
            for index in range(4):
                prompt_row[f"amp_E{index}"] = float(
                    diagnostics["amplitudes"][index]
                )
                prompt_row[f"power_E{index}"] = float(
                    diagnostics["powers"][index]
                )
                prompt_row[f"norm_power_E{index}"] = float(
                    diagnostics["normalized_powers"][index]
                )
                prompt_row[f"expert_energy_ratio_E{index}"] = float(
                    diagnostics["expert_energy_ratios"][index]
                )
                row[f"amp_{task_name}_E{index}"] = prompt_row[
                    f"amp_E{index}"
                ]
                row[f"norm_power_{task_name}_E{index}"] = prompt_row[
                    f"norm_power_E{index}"
                ]
            prompt_row["outside_energy_ratio"] = diagnostics[
                "outside_energy_ratio"
            ]
            prompt_history.append(prompt_row)
        row["joint_val_loss"] = val_loss_weighted_sum / max(
            val_sample_count, 1
        )
        row["joint_val_acc"] = val_correct_weighted_sum / max(
            val_sample_count, 1
        )
        row["macro_val_acc"] = float(np.mean(val_accuracies))
        row["val_samples"] = val_sample_count
        row["val_duration_seconds"] = (
            time.perf_counter() - validation_start
        )
        row["epoch_duration_seconds"] = time.perf_counter() - epoch_start
        metrics_rows.append(row)

        write_rows(run_dir / "multitask_metrics.csv", metrics_rows)
        write_rows(run_dir / "task_metrics.csv", task_metric_rows)
        write_rows(
            run_dir / "task_val_metrics.csv",
            [
                {
                    "epoch": item["epoch"],
                    "task_name": item["task_name"],
                    "val_loss": item["val_loss"],
                    "val_acc": item["val_acc"],
                    "val_samples": item["val_samples"],
                }
                for item in task_metric_rows
            ],
        )
        write_rows(
            run_dir / "task_prompt_amplitude_history.csv",
            [
                {
                    key: value
                    for key, value in history.items()
                    if key in {"epoch", "task_name"}
                    or key.startswith("amp_")
                }
                for history in prompt_history
            ],
        )
        write_rows(
            run_dir / "task_prompt_power_history.csv",
            [
                {
                    key: value
                    for key, value in history.items()
                    if key in {"epoch", "task_name"}
                    or key.startswith("power_")
                    or key.startswith("norm_power_")
                }
                for history in prompt_history
            ],
        )
        write_rows(
            run_dir / "task_expert_energy_history.csv",
            [
                {
                    key: value
                    for key, value in history.items()
                    if key in {"epoch", "task_name", "outside_energy_ratio"}
                    or key.startswith("expert_energy_ratio_")
                }
                for history in prompt_history
            ],
        )
        save_json(
            {"stages": stage_records},
            str(run_dir / "trainable_parameters_by_stage.json"),
        )

        checkpoint_metrics = dict(row)
        val_mean = row["joint_val_acc"]
        checkpoint_metrics["joint_val_acc"] = val_mean
        checkpoint_metrics["mean_val_acc"] = val_mean
        save_checkpoint(
            str(run_dir / "last.pt"),
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
        )
        if val_mean > best_val_mean:
            best_val_mean = val_mean
            save_checkpoint(
                str(run_dir / "best.pt"),
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
            )
        next_stage = (
            schedule.stage_for_epoch(epoch + 1)
            if schedule.enabled and epoch < num_epochs
            else stage_idx
        )
        if next_stage != stage_idx or epoch == num_epochs:
            save_checkpoint(
                str(run_dir / f"stage_{stage_idx}_last.pt"),
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
            )

        interval = int(
            config.get("visualization", {}).get("save_interval_epochs", 5)
        )
        if (
            config.get("visualization", {}).get("enabled", True)
            and interval > 0
            and (epoch % interval == 0 or epoch == num_epochs)
        ):
            try:
                save_multitask_phase_visualizations(model, run_dir, epoch)
                for task_name in task_names:
                    save_task_epoch_visualization(
                        current_diagnostics[task_name],
                        run_dir,
                        epoch,
                        task_name,
                    )
            except Exception as exc:  # pragma: no cover - environment-specific.
                config.setdefault("visualization", {})["enabled"] = False
                print(
                    "Epoch visualization failed and later PNG visualizations "
                    f"were disabled. Training will continue. Error: {repr(exc)}"
                )
        print(
            f"epoch {epoch:03d} stage {stage_idx} | "
            f"joint train loss={row['joint_train_loss']:.4f} "
            f"acc={row['joint_train_acc']:.4f} | "
            f"joint val loss={row['joint_val_loss']:.4f} "
            f"acc={row['joint_val_acc']:.4f} | "
            f"time={row['epoch_duration_seconds'] / 60.0:.1f} min | "
            + " | ".join(
                f"{name} train={row[f'{name}_train_acc']:.4f} "
                f"val={row[f'{name}_val_acc']:.4f}"
                for name in task_names
            )
        )

    if config.get("visualization", {}).get("enabled", True):
        try:
            plot_history(prompt_history, task_names, run_dir)
            plot_train_val(metrics_rows, task_names, run_dir)
        except Exception as exc:  # pragma: no cover - environment-specific.
            config.setdefault("visualization", {})["enabled"] = False
            (run_dir / "visualization_error.txt").write_text(
                "Final history plot saving failed. Metrics CSV, checkpoints, "
                "and summary files were still written.\n"
                f"{repr(exc)}\n",
                encoding="utf-8",
            )
            print(
                "Final history plot saving failed. Continuing to evaluation "
                f"and summary. Error: {repr(exc)}"
            )
    switching_rows = task_switching_evaluation(
        model,
        test_loaders,
        device,
        criterion,
        task_names,
        max_batches=max_test_batches,
    )
    write_rows(run_dir / "task_switching_eval.csv", switching_rows)
    correct_prompt_rows = [
        item
        for item in switching_rows
        if item["eval_dataset"] == item["prompt_task"]
    ]
    test_samples = sum(item["samples"] for item in correct_prompt_rows)
    joint_test_loss = sum(
        item["loss"] * item["samples"] for item in correct_prompt_rows
    ) / max(test_samples, 1)
    joint_test_acc = sum(
        item["accuracy"] * item["samples"] for item in correct_prompt_rows
    ) / max(test_samples, 1)
    task_test_rows = [
        {
            "task_name": item["eval_dataset"],
            "test_loss": item["loss"],
            "test_acc": item["accuracy"],
            "test_samples": item["samples"],
        }
        for item in correct_prompt_rows
    ]
    write_rows(run_dir / "task_test_metrics.csv", task_test_rows)
    write_rows(
        run_dir / "joint_test_metrics.csv",
        [
            {
                "joint_test_loss": joint_test_loss,
                "joint_test_acc": joint_test_acc,
                "test_samples": test_samples,
            }
        ],
    )
    summary = {
        "run_name": run_name,
        "training_mode": "multitask",
        "task_names": task_names,
        "task_num_classes": task_num_classes,
        "task_head_configs": model.task_head_configs,
        "class_semantics_note": (
            "The optical backbone is shared, but each task has its own "
            "detector/readout head. MNIST and FashionMNIST use 10 classes; "
            "EMNIST letters uses 26 classes."
        ),
        "optimizer": optimizer_cfg,
        "architecture_report": report,
        "best_joint_validation_accuracy": best_val_mean,
        "best_mean_validation_accuracy": best_val_mean,
        "final_joint_test_loss": joint_test_loss,
        "final_joint_test_accuracy": joint_test_acc,
        "per_task_test_metrics": task_test_rows,
        "epochs": num_epochs,
        "task_switching_evaluation": switching_rows,
        "layout": model.layout.to_dict(),
        "distances_m": model.distances_m,
    }
    save_json(summary, str(run_dir / "summary.json"))
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
