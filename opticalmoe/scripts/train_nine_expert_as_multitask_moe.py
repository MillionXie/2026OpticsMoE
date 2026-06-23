import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.data import create_dataloaders
from opticalmoe.optics.nine_expert_as_multitask_moe import (
    NineExpertASGlobalRouterMultitaskMoEClassifier,
)
from opticalmoe.optics.nine_expert_geometry import NineExpertFair134Layout
from opticalmoe.training import save_checkpoint
from opticalmoe.training.multitask_engine import evaluate_task, task_switching_evaluation
from opticalmoe.utils import load_config, save_json, set_seed
from opticalmoe.utils.run import create_run_dir


plt = None


def ensure_matplotlib():
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as loaded_plt

        plt = loaded_plt
    return plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 9-expert fair134 AS global-router multitask OpticalMoE."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def configure_matplotlib(config: Dict) -> None:
    plt = ensure_matplotlib()
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


def phase_dropout_settings(config: Dict) -> Dict:
    cfg = config.get("regularization", {}).get("phase_dropout", {})
    enabled = bool(cfg.get("enabled", False))
    apply_to_experts = bool(cfg.get("apply_to_experts", True))
    apply_to_global_fc = bool(cfg.get("apply_to_global_fc", False))
    mode = cfg.get("mode", "none") if enabled else "none"
    return {
        "enabled": enabled,
        "mode": mode,
        "expert_mode": mode if enabled and apply_to_experts else "none",
        "global_fc_mode": mode if enabled and apply_to_global_fc else "none",
        "expert_p": (
            float(cfg.get("expert_p", 0.0))
            if enabled and apply_to_experts
            else 0.0
        ),
        "global_fc_p": (
            float(cfg.get("global_fc_p", 0.0))
            if enabled and apply_to_global_fc
            else 0.0
        ),
        "block_size": int(cfg.get("block_size", 8)),
        "batch_shared": bool(cfg.get("batch_shared", True)),
        "apply_to_experts": apply_to_experts,
        "apply_to_global_fc": apply_to_global_fc,
        "start_epoch": int(cfg.get("start_epoch", 0)),
    }


def phase_dropout_active_for_epoch(settings: Dict, epoch: int) -> bool:
    return bool(
        settings["enabled"]
        and settings["mode"] != "none"
        and (settings["expert_p"] > 0.0 or settings["global_fc_p"] > 0.0)
        and int(epoch) >= int(settings["start_epoch"])
    )


def build_model(config: Dict, task_names: Sequence[str], task_num_classes: Dict[str, int]):
    layout_cfg = config.get("layout", {})
    optics_cfg = config.get("optics", {})
    prompt_cfg = config.get("prompt", {})
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
    phase_dropout = phase_dropout_settings(config)
    distances = optics_cfg.get("distances_m", {})
    layout = NineExpertFair134Layout(
        canvas_height=int(layout_cfg.get("canvas_height", 1000)),
        canvas_width=int(layout_cfg.get("canvas_width", 1000)),
        input_size=int(layout_cfg.get("input_size", 134)),
        expert_size=int(layout_cfg.get("expert_size", 134)),
        expert_pitch=int(layout_cfg.get("expert_pitch", 200)),
        padding=int(layout_cfg.get("padding", 200)),
        prompt_aperture_size=int(layout_cfg.get("prompt_aperture_size", 600)),
    )
    task_head_configs = {
        task["name"].lower(): dict(task.get("head", {}))
        for task in config["training"]["multitask"]["tasks"]
    }
    return NineExpertASGlobalRouterMultitaskMoEClassifier(
        task_names=task_names,
        task_num_classes=task_num_classes,
        task_head_configs=task_head_configs,
        layout=layout,
        wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
        pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
        input_size=int(layout_cfg.get("input_size", 134)),
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
        expert_phase_init=optics_cfg.get("expert_phase_init", "identity"),
        expert_init_std=float(optics_cfg.get("expert_init_std", 0.02)),
        global_fc_phase_init=optics_cfg.get("global_fc_phase_init", "identity"),
        global_fc_init_std=float(optics_cfg.get("global_fc_init_std", 0.02)),
        prompt_mode=prompt_cfg.get("mode", "complex_order_router"),
        prompt_amplitude_init_logits=float(prompt_cfg.get("amplitude_init_logits", 2.0)),
        train_prompt_phase_biases=bool(prompt_cfg.get("train_phase_biases", True)),
        grating_scale=float(prompt_cfg.get("grating_scale", 1.0)),
        grating_sign_x=float(prompt_cfg.get("grating_sign_x", 1.0)),
        grating_sign_y=float(prompt_cfg.get("grating_sign_y", 1.0)),
        prompt_normalize=prompt_cfg.get("normalize", "sum_amplitude"),
        detector_size=int(detector_cfg.get("detector_size", 32)),
        detector_layout=detector_cfg.get("layout", "grid"),
        normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
        readout_type=readout_cfg.get("type", "optical_only"),
        logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
        readout_activation=readout_cfg.get("activation", "relu"),
        readout_input_norm=readout_cfg.get("input_norm", "none"),
        readout_norm_affine=bool(readout_cfg.get("norm_affine", True)),
        readout_hidden_layers=int(readout_cfg.get("hidden_layers", 1)),
        readout_dropout=float(readout_cfg.get("dropout", 0.0)),
        expert_phase_dropout_mode=phase_dropout["expert_mode"],
        expert_phase_dropout_p=phase_dropout["expert_p"],
        global_fc_phase_dropout_mode=phase_dropout["global_fc_mode"],
        global_fc_phase_dropout_p=phase_dropout["global_fc_p"],
        phase_dropout_block_size=phase_dropout["block_size"],
        phase_dropout_batch_shared=phase_dropout["batch_shared"],
        evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
    )


def optimizer_settings(config: Dict) -> Dict:
    cfg = config.get("optimizer", {})
    return {
        "type": cfg.get("type", "adamw").lower(),
        "lr": float(cfg.get("lr", 0.001)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "momentum": float(cfg.get("momentum", 0.9)),
    }


def build_optimizer(model, config: Dict):
    settings = optimizer_settings(config)
    kwargs = {"lr": settings["lr"], "weight_decay": settings["weight_decay"]}
    if settings["type"] == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if settings["type"] == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if settings["type"] == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=settings["momentum"], **kwargs)
    raise ValueError("optimizer.type must be adamw, adam, or sgd.")


def create_task_loaders(config: Dict, seed: int, force_smoke: bool):
    train_loaders, val_loaders, test_loaders, class_counts = {}, {}, {}, {}
    for index, task in enumerate(config["training"]["multitask"]["tasks"]):
        task_name = task["name"].lower()
        dataset_cfg = dict(task["dataset"])
        if force_smoke:
            dataset_cfg["smoke_test"] = True
        train, val, test, classes = create_dataloaders(dataset_cfg, seed=seed + index)
        train_loaders[task_name] = train
        val_loaders[task_name] = val
        test_loaders[task_name] = test
        class_counts[task_name] = classes
    return train_loaders, val_loaders, test_loaders, class_counts


def fixed_batch(loader, num_samples: int):
    for images, targets in loader:
        return images[:num_samples], targets[:num_samples]
    raise RuntimeError("Loader is empty.")


def write_rows(path: Path, rows: List[Dict]) -> None:
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
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _next_batch(iterators, loaders, task_name):
    try:
        return next(iterators[task_name])
    except StopIteration:
        iterators[task_name] = iter(loaders[task_name])
        return next(iterators[task_name])


def multitask_loss_weights(config: Dict, task_names: Sequence[str]) -> Dict[str, float]:
    raw = config["training"]["multitask"].get("loss_weights", {})
    return {task_name: float(raw.get(task_name, 1.0)) for task_name in task_names}


def train_one_epoch_sequential(
    model,
    train_loaders: Dict,
    optimizer,
    device,
    criterion,
    task_names: Sequence[str],
    loss_weights: Dict[str, float],
    steps_per_epoch=None,
    balanced_sampling: bool = True,
    print_freq: int = 50,
):
    model.train()
    task_names = list(task_names)
    available_steps = (
        max(len(train_loaders[name]) for name in task_names)
        if balanced_sampling
        else min(len(train_loaders[name]) for name in task_names)
    )
    steps = (
        min(available_steps, int(steps_per_epoch))
        if steps_per_epoch is not None and int(steps_per_epoch) > 0
        else available_steps
    )
    iterators = {name: iter(train_loaders[name]) for name in task_names}
    weight_sum = sum(float(loss_weights[name]) for name in task_names)
    task_loss_sums = {name: 0.0 for name in task_names}
    task_correct = {name: 0 for name in task_names}
    task_seen = {name: 0 for name in task_names}
    total_loss_sum = 0.0

    for step_idx in range(steps):
        optimizer.zero_grad(set_to_none=True)
        update_loss_value = 0.0
        for task_name in task_names:
            images, targets = _next_batch(iterators, train_loaders, task_name)[:2]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images, task_name=task_name)
            loss = criterion(logits, targets)
            weighted_loss = loss * float(loss_weights[task_name]) / max(weight_sum, 1e-8)
            weighted_loss.backward()
            update_loss_value += float(weighted_loss.item())

            batch_size = targets.numel()
            task_loss_sums[task_name] += float(loss.item()) * batch_size
            task_correct[task_name] += int((logits.argmax(dim=1) == targets).sum().item())
            task_seen[task_name] += batch_size
        optimizer.step()
        total_loss_sum += update_loss_value
        if print_freq > 0 and ((step_idx + 1) % print_freq == 0 or step_idx + 1 == steps):
            status = " | ".join(
                f"{name}: loss={task_loss_sums[name] / max(task_seen[name], 1):.4f}, "
                f"acc={task_correct[name] / max(task_seen[name], 1):.4f}, "
                f"w={loss_weights[name]:.2f}"
                for name in task_names
            )
            print(f"  update {step_idx + 1}/{steps} | joint_loss={total_loss_sum / (step_idx + 1):.4f} | {status}")

    joint_seen = sum(task_seen.values())
    joint_loss_sum = sum(task_loss_sums.values())
    joint_correct = sum(task_correct.values())
    result = {
        "total_loss": total_loss_sum / max(steps, 1),
        "joint_sample_loss": joint_loss_sum / max(joint_seen, 1),
        "joint_accuracy": joint_correct / max(joint_seen, 1),
        "samples": joint_seen,
        "steps": steps,
        "available_steps": available_steps,
    }
    for task_name in task_names:
        result[f"{task_name}_loss_weight"] = float(loss_weights[task_name])
        result[f"{task_name}_loss"] = task_loss_sums[task_name] / max(task_seen[task_name], 1)
        result[f"{task_name}_acc"] = task_correct[task_name] / max(task_seen[task_name], 1)
        result[f"{task_name}_samples"] = task_seen[task_name]
    return result


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
        "normalized_powers": intermediates["normalized_prompt_powers"].detach().cpu(),
        "expert_energy_ratios": intermediates["expert_energy_ratios"].mean(dim=0).detach().cpu(),
        "outside_energy_ratio": float(intermediates["outside_energy_ratio"].mean().item()),
        "detector_energies": intermediates["detector_energies"].mean(dim=0).detach().cpu(),
        "intermediates": intermediates,
    }


def log_image(field: torch.Tensor) -> np.ndarray:
    value = torch.abs(field).square() if torch.is_complex(field) else field
    if value.ndim == 3:
        value = value[0]
    array = value.detach().cpu().float().numpy()
    return np.log10(array / (array.max() + 1e-12) + 1e-8)


def phase_image(phase: torch.Tensor) -> np.ndarray:
    return torch.remainder(phase.detach().cpu().float(), 2.0 * math.pi).numpy()


def save_field(path: Path, field: torch.Tensor, title: str):
    plt = ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(log_image(field), cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_phase(path: Path, phase: torch.Tensor, title: str):
    plt = ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(phase_image(phase), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar(path: Path, values, title: str, ylabel: str, labels: Sequence[str]):
    plt = ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = torch.as_tensor(values).detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def expert_labels():
    return [f"E{row}{col}" for row in range(3) for col in range(3)]


def save_phase_visualizations(model, run_dir: Path, epoch: int):
    phase_dir = run_dir / "phases" / f"epoch_{epoch:04d}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(model.num_layers, 9, figsize=(20, 2.3 * model.num_layers), squeeze=False)
    labels = expert_labels()
    for layer_index, layer in enumerate(model.expert_layers):
        phases = layer.get_phase_wrapped().detach().cpu().numpy()
        for expert_index in range(9):
            axes[layer_index, expert_index].imshow(phases[expert_index], cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
            axes[layer_index, expert_index].set_title(f"L{layer_index + 1} {labels[expert_index]}")
            axes[layer_index, expert_index].axis("off")
    fig.suptitle(f"Shared 9-Expert Phase Layers, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(phase_dir / "shared_expert_phase_layers.png")
    plt.close(fig)
    save_phase(phase_dir / "global_fc_phase.png", model.global_fc.get_phase_wrapped(), f"Global FC Phase, Epoch {epoch}")


def save_task_visualization(diagnostics: Dict, run_dir: Path, epoch: int, task_name: str):
    intermediates = diagnostics["intermediates"]
    light_dir = run_dir / "light_fields" / f"epoch_{epoch:04d}" / task_name
    prompt_dir = run_dir / "prompt" / f"epoch_{epoch:04d}" / task_name
    sample_dir = run_dir / "sample_outputs" / f"epoch_{epoch:04d}" / task_name
    light_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        ("00_input_amplitude.png", intermediates["input_amplitude"], "Input Amplitude"),
        ("01_after_input_to_prompt.png", intermediates["after_input_to_prompt"], "After Input-to-Prompt"),
        ("02_after_prompt.png", intermediates["after_prompt"], "After Prompt"),
        ("03_expert_entrance.png", intermediates["expert_entrance_intensity"], "Expert Entrance"),
        ("04_expert_entrance_after_aperture.png", intermediates["expert_entrance_after_aperture"], "Expert Entrance After Aperture"),
    ]
    for index, field in enumerate(intermediates["after_each_layer"], start=1):
        fields.append((f"{index + 4:02d}_after_expert_layer_{index}.png", field, f"After Expert Layer {index}"))
    fields.extend(
        [
            ("after_global_fc.png", intermediates["after_global_fc"], "After Global FC"),
            ("detector_plane.png", intermediates["detector_intensity"], "Detector Plane"),
        ]
    )
    for name, field, title in fields:
        save_field(light_dir / name, field, f"{task_name}: {title}")

    plt = ensure_matplotlib()
    cols = 4
    rows = int(math.ceil(len(fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, (_name, field, title) in zip(axes, fields):
        ax.imshow(log_image(field), cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(fields) :]:
        ax.axis("off")
    fig.suptitle(f"{task_name}: Light Fields, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(light_dir / "overview.png")
    plt.close(fig)

    save_field(prompt_dir / "prompt_router_amplitude.png", intermediates["prompt_router_amplitude"], f"{task_name}: Router Amplitude")
    save_phase(prompt_dir / "prompt_router_phase.png", intermediates["prompt_router_phase"], f"{task_name}: Router Phase")
    save_field(prompt_dir / "prompt_total_amplitude.png", intermediates["prompt_total_amplitude"], f"{task_name}: Prompt Total Amplitude")
    save_phase(prompt_dir / "prompt_total_phase.png", intermediates["prompt_total_phase"], f"{task_name}: Prompt Total Phase")
    labels = expert_labels()
    save_bar(prompt_dir / "prompt_amplitude_bar.png", diagnostics["amplitudes"], f"{task_name}: Prompt Amplitudes", "amplitude", labels)
    save_bar(prompt_dir / "normalized_prompt_power_bar.png", diagnostics["normalized_powers"], f"{task_name}: Normalized Prompt Power", "power fraction", labels)
    save_bar(sample_dir / "prompt_energy_detector_bars.png", diagnostics["expert_energy_ratios"], f"{task_name}: Expert Entrance Energy", "energy ratio", labels)
    save_bar(sample_dir / "detector_energy_bar.png", diagnostics["detector_energies"], f"{task_name}: Detector Energies", "energy", [f"D{i}" for i in range(len(diagnostics["detector_energies"]))])
    payload = {
        "targets": diagnostics["targets"].tolist(),
        "predictions": diagnostics["predictions"].tolist(),
        "prompt_amplitudes": diagnostics["amplitudes"].tolist(),
        "normalized_prompt_powers": diagnostics["normalized_powers"].tolist(),
        "expert_energy_ratios": diagnostics["expert_energy_ratios"].tolist(),
        "outside_energy_ratio": diagnostics["outside_energy_ratio"],
    }
    with open(sample_dir / "sample_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    inputs = intermediates["input_amplitude"].detach().cpu().float()
    sample_count = min(int(inputs.shape[0]), 8)
    fig, axes = plt.subplots(1, sample_count, figsize=(2.2 * sample_count, 2.6), squeeze=False)
    input_aperture = inputs[:, 433:567, 433:567]
    for index in range(sample_count):
        ax = axes[0, index]
        ax.imshow(input_aperture[index], cmap="gray")
        ax.set_title(
            f"y={int(diagnostics['targets'][index])}\n"
            f"p={int(diagnostics['predictions'][index])}"
        )
        ax.axis("off")
    fig.suptitle(f"{task_name}: Sample Predictions")
    fig.tight_layout()
    fig.savefig(sample_dir / "sample_predictions.png")
    plt.close(fig)


def save_architecture_report(model, config: Dict, optimizer_cfg: Dict, run_dir: Path):
    layout = model.layout.to_dict()
    report = {
        "model": "NineExpertASGlobalRouterMultitaskMoEClassifier",
        "canvas": 1000,
        "input_size": 134,
        "expert_size": 134,
        "expert_pitch": 200,
        "padding": 200,
        "prompt_aperture": "200:800, 200:800",
        "prompt_aperture_size": 600,
        "expert_phase_params_per_layer": 9 * 134 * 134,
        "baseline_4expert_phase_params_per_layer": 4 * 200 * 200,
        "relative_param_diff": 9 * 134 * 134 / (4 * 200 * 200) - 1.0,
        "layout": layout,
        "prompt_channel_table": model.prompt_bank.channel_table(),
        "task_num_classes": model.task_num_classes,
        "task_head_configs": model.task_head_configs,
        "optimizer": optimizer_cfg,
        "optical_parameter_count": model.optical_parameter_count(),
        "prompt_parameter_count": model.prompt_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "nonlinearity_statement": "Optical propagation and phase masks are linear/phase-only; |U|^2 intensity detection is nonlinear. Electronic MLP readouts are task-specific when configured.",
    }
    save_json(report, str(run_dir / "architecture_report.json"))
    lines = [
        "# Nine-Expert AS Global Router Architecture Report",
        "",
        "- canvas: 1000 x 1000",
        "- input aperture: 134 x 134 at center (500,500)",
        "- expert apertures: 9 x 134 x 134, centers 300/500/700",
        "- expert pitch: 200 px, aperture gap: 66 px",
        "- prompt aperture: y=200:800, x=200:800",
        f"- 9-expert phase params per layer: {report['expert_phase_params_per_layer']}",
        f"- 4-expert baseline phase params per layer: {report['baseline_4expert_phase_params_per_layer']}",
        f"- relative diff: {report['relative_param_diff']:.4f}",
        "- expert entrance is produced by AngularSpectrumPropagator(prompt -> expert)",
        f"- task classes: {model.task_num_classes}",
        f"- optimizer: {optimizer_cfg}",
        "",
        report["nonlinearity_statement"],
    ]
    (run_dir / "architecture_report.md").write_text("\n".join(lines), encoding="utf-8")


def loader_summary(train_loaders, val_loaders, test_loaders, task_names, effective_steps, natural_steps, balanced_sampling):
    return {
        "balanced_sampling": bool(balanced_sampling),
        "natural_updates_per_epoch": int(natural_steps),
        "effective_updates_per_epoch": int(effective_steps),
        "tasks": {
            name: {
                "train_samples": len(train_loaders[name].dataset),
                "val_samples": len(val_loaders[name].dataset),
                "test_samples": len(test_loaders[name].dataset),
                "batch_size": train_loaders[name].batch_size,
                "train_loader_steps": len(train_loaders[name]),
                "repeat_factor": float(effective_steps) / max(float(len(train_loaders[name])), 1.0),
            }
            for name in task_names
        },
    }


def plot_history(rows: List[Dict], task_names: Sequence[str], run_dir: Path):
    if not rows:
        return
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for task_name in task_names:
        xs = [r["epoch"] for r in rows if r.get("task_name") == task_name]
        ys = [r["val_acc"] for r in rows if r.get("task_name") == task_name]
        axes[0].plot(xs, ys, label=f"{task_name} val")
        ys_loss = [r["val_loss"] for r in rows if r.get("task_name") == task_name]
        axes[1].plot(xs, ys_loss, label=f"{task_name} val")
    axes[0].set_title("Validation Accuracy")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_val_curves.png")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    if config.get("training", {}).get("mode") != "multitask":
        raise ValueError("This script requires training.mode=multitask.")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    if config.get("visualization", {}).get("enabled", True):
        configure_matplotlib(config.get("visualization", {}))
    run_name = args.run_name or config.get("experiment", {}).get("run_name", "nine_expert_as_multitask")
    run_dir = create_run_dir(run_name, base_dir=str(PROJECT_ROOT / "runs"))
    shutil.copyfile(args.config, run_dir / "config.yaml")
    save_json(config, str(run_dir / "config_resolved.json"))

    train_loaders, val_loaders, test_loaders, class_counts = create_task_loaders(config, seed, args.smoke_test)
    task_names = list(train_loaders.keys())
    device = choose_device(args.device or config.get("device", "auto"))
    model = build_model(config, task_names, class_counts).to(device)
    phase_dropout = phase_dropout_settings(config)
    model.set_phase_dropout_active(False)
    optimizer = build_optimizer(model, config)
    optimizer_cfg = optimizer_settings(config)
    criterion = nn.CrossEntropyLoss()
    save_architecture_report(model, config, optimizer_cfg, run_dir)

    multitask_cfg = config["training"]["multitask"]
    evaluation_cfg = config["training"].get("evaluation", {})
    num_epochs = int(args.epochs or config["training"].get("epochs", 300))
    if args.smoke_test:
        num_epochs = int(args.epochs or 1)
    loss_weights = multitask_loss_weights(config, task_names)
    natural_steps = max(len(loader) for loader in train_loaders.values())
    steps_per_epoch = multitask_cfg.get("steps_per_epoch")
    effective_steps = (
        min(natural_steps, int(steps_per_epoch))
        if steps_per_epoch is not None and int(steps_per_epoch) > 0
        else natural_steps
    )
    summary = loader_summary(
        train_loaders,
        val_loaders,
        test_loaders,
        task_names,
        effective_steps,
        natural_steps,
        bool(multitask_cfg.get("balanced_sampling", True)),
    )
    save_json(summary, str(run_dir / "multitask_loader_summary.json"))

    print(f"device: {device}")
    print(f"tasks: {task_names}, task classes: {class_counts}")
    print(
        f"Optimizer: {optimizer.__class__.__name__}, lr={optimizer_cfg['lr']}, weight_decay={optimizer_cfg['weight_decay']}"
    )
    print(f"sequential_backward={bool(multitask_cfg.get('sequential_backward', True))}")
    print(
        "Phase dropout: "
        f"enabled={phase_dropout['enabled']}, "
        f"mode={phase_dropout['mode']}, "
        f"expert_p={phase_dropout['expert_p']}, "
        f"global_fc_p={phase_dropout['global_fc_p']}, "
        f"block_size={phase_dropout['block_size']}, "
        f"batch_shared={phase_dropout['batch_shared']}, "
        f"start_epoch={phase_dropout['start_epoch']}"
    )
    save_json(phase_dropout, str(run_dir / "phase_dropout_summary.json"))
    print(f"updates per epoch: {effective_steps} (natural full-dataset value: {natural_steps})")
    for task_name in task_names:
        s = summary["tasks"][task_name]
        print(
            f"  {task_name}: train={s['train_samples']} samples/{s['train_loader_steps']} batches, "
            f"val={s['val_samples']}, test={s['test_samples']}, batch_size={s['batch_size']}"
        )

    fixed_batches = {
        name: fixed_batch(val_loaders[name], int(config.get("visualization", {}).get("num_samples", 2)))
        for name in task_names
    }
    visualization_enabled = bool(config.get("visualization", {}).get("enabled", True))
    current_diagnostics = {}
    for task_name in task_names:
        current_diagnostics[task_name] = task_diagnostics(model, fixed_batches[task_name], device, task_name)
        if visualization_enabled:
            save_task_visualization(current_diagnostics[task_name], run_dir, 0, task_name)
    if visualization_enabled:
        save_phase_visualizations(model, run_dir, 0)

    metrics_rows, task_rows, prompt_rows = [], [], []
    best_val_acc = -1.0
    max_val_batches = evaluation_cfg.get("max_val_batches")
    max_test_batches = evaluation_cfg.get("max_test_batches")
    print_freq = int(multitask_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()
        phase_dropout_active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(phase_dropout_active)
        train_result = train_one_epoch_sequential(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            task_names=task_names,
            loss_weights=loss_weights,
            steps_per_epoch=steps_per_epoch,
            balanced_sampling=bool(multitask_cfg.get("balanced_sampling", True)),
            print_freq=print_freq,
        )
        row = {
            "epoch": epoch,
            "total_loss": train_result["total_loss"],
            "joint_train_loss": train_result["joint_sample_loss"],
            "joint_train_acc": train_result["joint_accuracy"],
            "updates": train_result["steps"],
            "lr": optimizer.param_groups[0]["lr"],
            "sequential_backward": True,
            "loss_weights": json.dumps(loss_weights, sort_keys=True),
            "phase_dropout_active": phase_dropout_active,
            "phase_dropout_mode": phase_dropout["mode"],
            "expert_phase_dropout_p": phase_dropout["expert_p"],
            "global_fc_phase_dropout_p": phase_dropout["global_fc_p"],
            "phase_dropout_block_size": phase_dropout["block_size"],
            "phase_dropout_batch_shared": phase_dropout["batch_shared"],
        }
        val_accs = []
        val_loss_sum, val_correct_sum, val_seen = 0.0, 0.0, 0
        for task_name in task_names:
            validation = evaluate_task(
                model,
                val_loaders[task_name],
                device,
                criterion,
                prompt_task=task_name,
                max_batches=max_val_batches,
            )
            row[f"{task_name}_train_loss"] = train_result[f"{task_name}_loss"]
            row[f"{task_name}_train_acc"] = train_result[f"{task_name}_acc"]
            row[f"{task_name}_loss_weight"] = train_result[f"{task_name}_loss_weight"]
            row[f"{task_name}_val_loss"] = validation["loss"]
            row[f"{task_name}_val_acc"] = validation["accuracy"]
            row[f"{task_name}_val_samples"] = validation["samples"]
            val_accs.append(validation["accuracy"])
            val_loss_sum += validation["loss"] * validation["samples"]
            val_correct_sum += validation["accuracy"] * validation["samples"]
            val_seen += validation["samples"]
            task_rows.append(
                {
                    "epoch": epoch,
                    "task_name": task_name,
                    "train_loss": train_result[f"{task_name}_loss"],
                    "train_acc": train_result[f"{task_name}_acc"],
                    "val_loss": validation["loss"],
                    "val_acc": validation["accuracy"],
                    "val_samples": validation["samples"],
                }
            )
            diagnostics = task_diagnostics(model, fixed_batches[task_name], device, task_name)
            current_diagnostics[task_name] = diagnostics
            prompt_row = {"epoch": epoch, "task_name": task_name}
            for index in range(9):
                prompt_row[f"amp_E{index}"] = float(diagnostics["amplitudes"][index])
                prompt_row[f"power_E{index}"] = float(diagnostics["powers"][index])
                prompt_row[f"norm_power_E{index}"] = float(diagnostics["normalized_powers"][index])
                prompt_row[f"expert_energy_ratio_E{index}"] = float(diagnostics["expert_energy_ratios"][index])
                row[f"amp_{task_name}_E{index}"] = prompt_row[f"amp_E{index}"]
                row[f"norm_power_{task_name}_E{index}"] = prompt_row[f"norm_power_E{index}"]
                row[f"expert_energy_ratio_{task_name}_E{index}"] = prompt_row[f"expert_energy_ratio_E{index}"]
            prompt_row["outside_energy_ratio"] = diagnostics["outside_energy_ratio"]
            row[f"{task_name}_outside_energy_ratio"] = diagnostics["outside_energy_ratio"]
            prompt_rows.append(prompt_row)
        row["joint_val_loss"] = val_loss_sum / max(val_seen, 1)
        row["joint_val_acc"] = val_correct_sum / max(val_seen, 1)
        row["macro_val_acc"] = float(np.mean(val_accs))
        row["epoch_duration_seconds"] = time.perf_counter() - start
        metrics_rows.append(row)
        write_rows(run_dir / "multitask_metrics.csv", metrics_rows)
        write_rows(run_dir / "task_metrics.csv", task_rows)
        write_rows(run_dir / "task_prompt_amplitude_history.csv", prompt_rows)
        write_rows(run_dir / "task_expert_energy_history.csv", prompt_rows)

        ckpt_dir = run_dir / "checkpoints"
        save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, row)
        save_checkpoint(str(run_dir / "last.pt"), model, optimizer, epoch, row)
        if row["joint_val_acc"] > best_val_acc:
            best_val_acc = row["joint_val_acc"]
            save_checkpoint(str(ckpt_dir / "best.pt"), model, optimizer, epoch, row)
            save_checkpoint(str(run_dir / "best.pt"), model, optimizer, epoch, row)

        interval = int(config.get("visualization", {}).get("save_interval_epochs", 5))
        if visualization_enabled and interval > 0 and (epoch % interval == 0 or epoch == num_epochs):
            save_phase_visualizations(model, run_dir, epoch)
            for task_name in task_names:
                save_task_visualization(current_diagnostics[task_name], run_dir, epoch, task_name)
        print(
            f"epoch {epoch:03d} | train={row['joint_train_acc']:.4f} val={row['joint_val_acc']:.4f} | "
            + " | ".join(f"{name} train={row[f'{name}_train_acc']:.4f} val={row[f'{name}_val_acc']:.4f}" for name in task_names)
            + f" | phase_dropout={'on' if phase_dropout_active else 'off'}"
            + f" | time={row['epoch_duration_seconds'] / 60.0:.1f} min"
        )

    if visualization_enabled:
        plot_history(task_rows, task_names, run_dir)
    switching_rows = task_switching_evaluation(
        model,
        test_loaders,
        device,
        criterion,
        task_names,
        max_batches=max_test_batches,
    )
    write_rows(run_dir / "task_switching_eval.csv", switching_rows)
    write_rows(run_dir / "task_switching_test.csv", switching_rows)
    correct_rows = [row for row in switching_rows if row["eval_dataset"] == row["prompt_task"]]
    test_samples = sum(row["samples"] for row in correct_rows)
    joint_test_acc = sum(row["accuracy"] * row["samples"] for row in correct_rows) / max(test_samples, 1)
    joint_test_loss = sum(row["loss"] * row["samples"] for row in correct_rows) / max(test_samples, 1)
    final_metrics = {
        "joint_test_loss": joint_test_loss,
        "joint_test_acc": joint_test_acc,
        "per_task": correct_rows,
    }
    save_json(final_metrics, str(run_dir / "final_test_metrics.json"))
    save_json(
        {
            "run_name": run_name,
            "training_mode": "nine_expert_as_global_router_multitask",
            "task_names": task_names,
            "task_num_classes": class_counts,
            "layout": model.layout.to_dict(),
            "distances_m": model.distances_m,
            "loss_weights": loss_weights,
            "phase_dropout": phase_dropout,
            "best_joint_validation_accuracy": best_val_acc,
            "final_test_metrics": final_metrics,
            "task_switching_evaluation": switching_rows,
        },
        str(run_dir / "summary.json"),
    )
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
