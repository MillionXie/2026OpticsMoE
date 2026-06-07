import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from opticalmoe.optics import FourExpertMoEClassifierV2
from opticalmoe.optics.four_expert_geometry import FourExpertLayout
from opticalmoe.training import ProgressiveUnfreezingSchedule, save_checkpoint
from opticalmoe.training.four_expert_reporting import (
    build_architecture_report,
    save_architecture_report,
    save_initial_state,
)
from opticalmoe.utils import load_config, save_json, set_seed
from opticalmoe.utils.run import create_run_dir


METRIC_FIELDS = [
    "epoch",
    "stage_idx",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "lr",
    "active_layers",
    "amp_E0",
    "amp_E1",
    "amp_E2",
    "amp_E3",
    "power_E0",
    "power_E1",
    "power_E2",
    "power_E3",
    "norm_power_E0",
    "norm_power_E1",
    "norm_power_E2",
    "norm_power_E3",
    "expert_energy_ratio_E0",
    "expert_energy_ratio_E1",
    "expert_energy_ratio_E2",
    "expert_energy_ratio_E3",
    "outside_energy_ratio",
]


def configure_matplotlib(vis_cfg: Dict) -> None:
    font_size = int(vis_cfg.get("font_size", 12))
    dpi = int(vis_cfg.get("dpi", 150))
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the independent four-expert OpticalMoE V2 classifier."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Force dataset smoke subsets from the YAML smoke sizes.",
    )
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def build_model(config: Dict, num_classes: int) -> FourExpertMoEClassifierV2:
    layout_cfg = config.get("layout", {})
    optics_cfg = config.get("optics", {})
    prompt_cfg = config.get("prompt", {})
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
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
    return FourExpertMoEClassifierV2(
        num_classes=num_classes,
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


def get_fixed_batch(loader, num_samples: int):
    for batch in loader:
        images, targets = batch[:2]
        return images[:num_samples], targets[:num_samples]
    return None


def auxiliary_loss(
    model: FourExpertMoEClassifierV2,
    intermediates: Optional[Dict],
    loss_cfg: Dict,
) -> torch.Tensor:
    reference = model.prompt.amplitude_logits
    loss = torch.zeros((), device=reference.device, dtype=reference.dtype)
    lambda_balance = float(loss_cfg.get("lambda_prompt_balance", 0.0))
    lambda_outside = float(loss_cfg.get("lambda_outside_energy", 0.0))
    lambda_entropy = float(loss_cfg.get("lambda_expert_energy_entropy", 0.0))

    if lambda_balance != 0.0:
        normalized_power = model.prompt.normalized_powers()
        target = torch.full_like(normalized_power, 0.25)
        loss = loss + lambda_balance * torch.mean((normalized_power - target) ** 2)
    if lambda_outside != 0.0:
        if intermediates is None:
            raise RuntimeError("outside-energy loss requires intermediates.")
        loss = loss + lambda_outside * intermediates["outside_energy_ratio"].mean()
    if lambda_entropy != 0.0:
        if intermediates is None:
            raise RuntimeError("expert-energy entropy loss requires intermediates.")
        energies = intermediates["expert_energy"]
        probabilities = energies / (energies.sum(dim=1, keepdim=True) + 1e-8)
        # sum(p log p) is minimized by a high-entropy, balanced distribution.
        negative_entropy = torch.sum(
            probabilities * torch.log(probabilities + 1e-8), dim=1
        ).mean()
        loss = loss + lambda_entropy * negative_entropy
    return loss


def needs_training_intermediates(loss_cfg: Dict) -> bool:
    return any(
        float(loss_cfg.get(key, 0.0)) != 0.0
        for key in ["lambda_outside_energy", "lambda_expert_energy_entropy"]
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    loss_cfg,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    require_intermediates = needs_training_intermediates(loss_cfg)
    for batch in loader:
        images, targets = batch[:2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if require_intermediates:
            logits, intermediates = model(images, return_intermediates=True)
        else:
            logits = model(images)
            intermediates = None
        loss = criterion(logits, targets)
        loss = loss + auxiliary_loss(model, intermediates, loss_cfg)
        loss.backward()
        optimizer.step()

        batch_size = targets.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_targets = []
    all_predictions = []
    for batch in loader:
        images, targets = batch[:2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        batch_size = targets.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == targets).sum().item())
        total_seen += batch_size
        all_targets.append(targets.cpu())
        all_predictions.append(predictions.cpu())
    return (
        total_loss / max(total_seen, 1),
        total_correct / max(total_seen, 1),
        torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long),
        torch.cat(all_predictions)
        if all_predictions
        else torch.empty(0, dtype=torch.long),
    )


@torch.no_grad()
def epoch_diagnostics(model, fixed_batch, device) -> Dict:
    images, targets = fixed_batch
    model.eval()
    logits, intermediates = model(
        images.to(device),
        return_intermediates=True,
    )
    return {
        "targets": targets.detach().cpu(),
        "predictions": logits.argmax(dim=1).detach().cpu(),
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


def write_rows(path: Path, rows: Sequence[Dict], fieldnames: Optional[List[str]] = None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{key: row.get(key, "") for key in fieldnames} for row in rows]
        )


def metric_row(
    epoch: int,
    stage_info: Dict,
    train_result,
    val_result,
    test_result,
    lr: float,
    diagnostics: Dict,
) -> Dict:
    amplitudes = diagnostics["amplitudes"].tolist()
    powers = diagnostics["powers"].tolist()
    normalized = diagnostics["normalized_powers"].tolist()
    expert_ratios = diagnostics["expert_energy_ratios"].tolist()
    row = {
        "epoch": epoch,
        "stage_idx": stage_info["stage_idx"],
        "train_loss": train_result[0],
        "train_acc": train_result[1],
        "val_loss": val_result[0],
        "val_acc": val_result[1],
        "test_loss": test_result[0],
        "test_acc": test_result[1],
        "lr": lr,
        "active_layers": " ".join(
            str(value) for value in stage_info["active_layers"]
        )
        or "none",
        "outside_energy_ratio": diagnostics["outside_energy_ratio"],
    }
    for index in range(4):
        row[f"amp_E{index}"] = amplitudes[index]
        row[f"power_E{index}"] = powers[index]
        row[f"norm_power_E{index}"] = normalized[index]
        row[f"expert_energy_ratio_E{index}"] = expert_ratios[index]
    return row


def history_rows(epoch: int, metrics: Dict, diagnostics: Dict):
    amplitude_row = {"epoch": epoch}
    expert_row = {"epoch": epoch}
    detector_row = {"epoch": epoch}
    for index in range(4):
        amplitude_row[f"amp_E{index}"] = metrics[f"amp_E{index}"]
        amplitude_row[f"power_E{index}"] = metrics[f"power_E{index}"]
        amplitude_row[f"norm_power_E{index}"] = metrics[f"norm_power_E{index}"]
        expert_row[f"expert_energy_ratio_E{index}"] = metrics[
            f"expert_energy_ratio_E{index}"
        ]
    expert_row["outside_energy_ratio"] = metrics["outside_energy_ratio"]
    for index, value in enumerate(diagnostics["detector_energies"].tolist()):
        detector_row[f"detector_{index}"] = value
    return amplitude_row, expert_row, detector_row


def plot_line_history(rows, keys, labels, path, title, ylabel):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = [row["epoch"] for row in rows]
    for key, label in zip(keys, labels):
        ax.plot(epochs, [row[key] for row in rows], marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_history_plots(run_dir, metrics_rows, amplitude_rows, expert_rows):
    plot_line_history(
        amplitude_rows,
        [f"amp_E{index}" for index in range(4)],
        [f"E{index}" for index in range(4)],
        run_dir / "prompt_amplitude_history.png",
        "Trainable Prompt Amplitudes",
        "Amplitude",
    )
    plot_line_history(
        amplitude_rows,
        [f"norm_power_E{index}" for index in range(4)],
        [f"E{index}" for index in range(4)],
        run_dir / "normalized_prompt_power_history.png",
        "Normalized Prompt Power",
        "Normalized amplitude squared",
    )
    plot_line_history(
        expert_rows,
        [f"expert_energy_ratio_E{index}" for index in range(4)]
        + ["outside_energy_ratio"],
        [f"E{index}" for index in range(4)] + ["outside"],
        run_dir / "expert_energy_ratio_history.png",
        "Expert Entrance Energy Ratios",
        "Energy / total",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = [row["epoch"] for row in metrics_rows]
    axes[0].plot(
        epochs, [row["train_loss"] for row in metrics_rows], label="train"
    )
    axes[0].plot(
        epochs, [row["val_loss"] for row in metrics_rows], label="validation"
    )
    axes[1].plot(
        epochs, [row["train_acc"] for row in metrics_rows], label="train"
    )
    axes[1].plot(
        epochs, [row["val_acc"] for row in metrics_rows], label="validation"
    )
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_val_curves.png")
    plt.close(fig)


def phase_image(phase: torch.Tensor) -> np.ndarray:
    return torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()


def save_phase_visualizations(model, run_dir: Path, epoch: int):
    epoch_name = f"epoch_{epoch:04d}"
    phase_dir = run_dir / "phases" / epoch_name
    prompt_dir = run_dir / "prompt" / epoch_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

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
    fig.suptitle(f"Four-Expert Phase Masks, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(run_dir / f"phase_expert_layers_epoch_{epoch}.png")
    fig.savefig(phase_dir / "expert_phase_layers.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        phase_image(model.global_fc.get_phase_wrapped()),
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(f"Global FC Phase, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(run_dir / f"global_fc_phase_epoch_{epoch}.png")
    fig.savefig(phase_dir / "global_fc_phase.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        phase_image(model.prompt.phase_map()),
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(f"Prompt Phase, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(prompt_dir / "prompt_phase.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        model.prompt.amplitude_map().detach().cpu().numpy(),
        cmap="viridis",
    )
    ax.set_title(f"Prompt Amplitude Map, Epoch {epoch}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(prompt_dir / "prompt_amplitude_map.png")
    plt.close(fig)


def field_log_image(field: torch.Tensor, sample_index: int = 0) -> np.ndarray:
    value = field
    if torch.is_complex(value):
        value = torch.abs(value).square()
    if value.ndim == 3:
        value = value[sample_index]
    array = value.detach().cpu().float().numpy()
    return np.log10(array / (array.max() + 1e-12) + 1e-8)


def save_light_field_visualizations(
    diagnostics: Dict,
    run_dir: Path,
    epoch: int,
):
    intermediates = diagnostics["intermediates"]
    epoch_name = f"epoch_{epoch:04d}"
    light_dir = run_dir / "light_fields" / epoch_name
    sample_dir = run_dir / "sample_outputs" / epoch_name
    light_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("Input amplitude", intermediates["input_amplitude"]),
        ("After input to prompt", intermediates["after_input_to_prompt"]),
        ("After prompt", intermediates["after_prompt"]),
        ("Expert entrance", intermediates["expert_entrance_intensity"]),
    ]
    for index, field in enumerate(intermediates["after_each_layer"], start=1):
        fields.append((f"After expert layer {index}", field))
    fields.extend(
        [
            ("After global FC", intermediates["after_global_fc"]),
            ("Detector plane", intermediates["detector_intensity"]),
        ]
    )
    columns = 4
    rows = int(math.ceil(len(fields) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, field) in zip(axes, fields):
        ax.imshow(field_log_image(field), cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(fields) :]:
        ax.axis("off")
    fig.suptitle(f"Light Field Diagnostics, Epoch {epoch}")
    fig.tight_layout()
    fig.savefig(run_dir / f"sample_light_fields_epoch_{epoch}.png")
    fig.savefig(light_dir / "overview.png")
    plt.close(fig)

    for index, (title, field) in enumerate(fields):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(field_log_image(field), cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        filename = (
            f"{index:02d}_"
            + title.lower().replace(" ", "_").replace("-", "_")
            + ".png"
        )
        fig.savefig(light_dir / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        field_log_image(intermediates["detector_intensity"]),
        cmap="inferno",
    )
    ax.set_title(f"Detector Plane, Epoch {epoch}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / f"detector_plane_epoch_{epoch}.png")
    fig.savefig(light_dir / "detector_plane.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        np.arange(4),
        diagnostics["expert_energy_ratios"].numpy(),
    )
    axes[0].set_xticks(np.arange(4))
    axes[0].set_xticklabels([f"E{i}" for i in range(4)])
    axes[0].set_title("Expert Entrance Energy")
    axes[0].set_ylabel("Energy / total")
    axes[1].bar(np.arange(4), diagnostics["amplitudes"].numpy())
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels([f"E{i}" for i in range(4)])
    axes[1].set_title("Prompt Amplitudes")
    axes[1].set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(run_dir / f"expert_prompt_bars_epoch_{epoch}.png")
    fig.savefig(sample_dir / "expert_prompt_bars.png")
    plt.close(fig)

    images = diagnostics["intermediates"]["input_amplitude"].detach().cpu()
    targets = diagnostics.get("targets", torch.empty(0, dtype=torch.long))
    predictions = diagnostics.get("predictions", torch.empty(0, dtype=torch.long))
    count = min(int(images.shape[0]), 8)
    if count > 0:
        fig, axes = plt.subplots(1, count, figsize=(2.4 * count, 2.8))
        axes = np.asarray(axes).reshape(-1)
        for idx in range(count):
            axes[idx].imshow(images[idx].numpy(), cmap="gray")
            target = int(targets[idx]) if idx < len(targets) else -1
            prediction = int(predictions[idx]) if idx < len(predictions) else -1
            axes[idx].set_title(f"y={target}, pred={prediction}")
            axes[idx].axis("off")
        fig.suptitle(f"Fixed Validation Samples, Epoch {epoch}")
        fig.tight_layout()
        fig.savefig(sample_dir / "sample_predictions.png")
        plt.close(fig)
        with open(sample_dir / "sample_predictions.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "targets": targets[:count].tolist(),
                    "predictions": predictions[:count].tolist(),
                },
                handle,
                indent=2,
            )


def save_confusion_matrix(targets, predictions, num_classes: int, path: Path):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def optimizer_from_config(model, config):
    cfg = config.get("optimizer", {})
    name = cfg.get("type", "adamw").lower()
    kwargs = {
        "lr": float(cfg.get("lr", 0.003)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(cfg.get("momentum", 0.9)),
            **kwargs,
        )
    raise ValueError("optimizer.type must be adam, adamw, or sgd.")


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


@torch.no_grad()
def fixed_batch_loss_accuracy(model, fixed_batch, device, criterion):
    images, targets = fixed_batch
    model.eval()
    images = images.to(device)
    targets = targets.to(device)
    logits = model(images)
    return (
        float(criterion(logits, targets).item()),
        float((logits.argmax(dim=1) == targets).float().mean().item()),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("training", {}).get("mode", "single") != "single":
        raise ValueError(
            "Use train_four_expert_multitask_moe.py for multitask experiments. "
            "This script is intentionally limited to one dataset."
        )
    if args.smoke_test:
        config["dataset"]["smoke_test"] = True
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    seed = int(config.get("seed", 7))
    set_seed(seed)
    configure_matplotlib(config.get("visualization", {}))

    run_name = args.run_name or config.get("experiment", {}).get(
        "run_name", "four_expert_moe_v2"
    )
    run_dir = create_run_dir(run_name, base_dir=str(PROJECT_ROOT / "runs"))
    shutil.copyfile(args.config, run_dir / "config.yaml")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        config["dataset"], seed
    )
    device_name = args.device or config.get("device", "auto")
    device = choose_device(device_name)
    model = build_model(config, num_classes).to(device)
    optimizer = optimizer_from_config(model, config)
    optimizer_cfg = optimizer_settings(config)
    criterion = nn.CrossEntropyLoss()

    progressive_cfg = config.get("training", {}).get("progressive", {})
    schedule = ProgressiveUnfreezingSchedule(
        num_layers=model.num_layers,
        enabled=bool(progressive_cfg.get("enabled", True)),
        order=progressive_cfg.get("order", "backward"),
        stage_epochs=progressive_cfg.get(
            "stage_epochs", [3, 3, 3, 3, 3, 10]
        ),
        train_prompt_always=bool(
            progressive_cfg.get("train_prompt_always", True)
        ),
        train_global_fc_always=bool(
            progressive_cfg.get("train_global_fc_always", True)
        ),
    )
    if args.epochs is not None:
        num_epochs = int(args.epochs)
    elif schedule.enabled:
        num_epochs = schedule.total_epochs
    else:
        num_epochs = int(config.get("training", {}).get("epochs", 25))

    vis_cfg = config.get("visualization", {})
    fixed_batch = get_fixed_batch(
        val_loader, int(vis_cfg.get("num_samples", 4))
    )
    if fixed_batch is None:
        raise RuntimeError("Validation loader is empty.")

    print(f"device: {device}")
    print(f"dataset: {config['dataset']['name']}, classes: {num_classes}")
    print(
        f"layout: canvas={model.canvas_shape}, expert={model.layout.expert_size}, "
        f"prompt_cell={model.layout.prompt_cell_size}"
    )
    print(
        f"parameters: optical={model.optical_parameter_count()}, "
        f"electronic={model.electronic_parameter_count()}"
    )
    print(
        f"progressive={schedule.enabled}, order={schedule.order}, "
        f"epochs={num_epochs}"
    )
    print(
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"lr={optimizer_cfg['lr']}, "
        f"weight_decay={optimizer_cfg['weight_decay']}"
    )

    architecture_report = build_architecture_report(
        model=model,
        config=config,
        optimizer_settings=optimizer_cfg,
        training_mode="single",
    )
    save_architecture_report(architecture_report, run_dir)

    initial_val_loss, initial_val_acc = fixed_batch_loss_accuracy(
        model, fixed_batch, device, criterion
    )
    initial_diagnostics = epoch_diagnostics(model, fixed_batch, device)
    initial_payload = save_initial_state(
        model=model,
        diagnostics=initial_diagnostics,
        output_dir=run_dir / "initial_state",
        val_loss=initial_val_loss,
        val_acc=initial_val_acc,
    )
    write_rows(
        run_dir / "initial_diagnostics.csv",
        [initial_payload],
    )

    metrics_rows = []
    initial_metric_stub = {
        "epoch": 0,
        **{
            f"amp_E{index}": initial_payload["prompt_amplitudes"][index]
            for index in range(4)
        },
        **{
            f"power_E{index}": initial_payload["prompt_powers"][index]
            for index in range(4)
        },
        **{
            f"norm_power_E{index}": initial_payload[
                "normalized_prompt_powers"
            ][index]
            for index in range(4)
        },
        **{
            f"expert_energy_ratio_E{index}": initial_payload[
                "expert_energy_ratios"
            ][index]
            for index in range(4)
        },
        "outside_energy_ratio": initial_payload["outside_energy_ratio"],
    }
    amplitude_rows = [
        {
            key: value
            for key, value in initial_metric_stub.items()
            if key == "epoch"
            or key.startswith("amp_")
            or key.startswith("power_")
            or key.startswith("norm_power_")
        }
    ]
    expert_rows = [
        {
            key: value
            for key, value in initial_metric_stub.items()
            if key == "epoch"
            or key.startswith("expert_energy_ratio_")
            or key == "outside_energy_ratio"
        }
    ]
    detector_rows = [
        {
            "epoch": 0,
            **{
                f"detector_{index}": value
                for index, value in enumerate(
                    initial_payload["detector_energies"]
                )
            },
        }
    ]
    stage_records = []
    best_val_acc = -1.0
    best_epoch = 0
    previous_stage = None
    final_targets = torch.empty(0, dtype=torch.long)
    final_predictions = torch.empty(0, dtype=torch.long)

    for epoch in range(1, num_epochs + 1):
        stage_idx = schedule.stage_for_epoch(epoch) if schedule.enabled else 0
        stage_info = schedule.apply(model, stage_idx)
        if stage_idx != previous_stage:
            stage_records.append(stage_info)
            print(
                f"stage {stage_idx}: active expert layers "
                f"{stage_info['active_layers'] or 'none'}, "
                f"trainable parameters={stage_info['trainable_parameter_count']}"
            )
            previous_stage = stage_idx

        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            config.get("loss", {}),
        )
        val_result = evaluate(model, val_loader, device, criterion)
        test_result = evaluate(model, test_loader, device, criterion)
        diagnostics = epoch_diagnostics(model, fixed_batch, device)
        row = metric_row(
            epoch,
            stage_info,
            train_result,
            val_result,
            test_result,
            optimizer.param_groups[0]["lr"],
            diagnostics,
        )
        metrics_rows.append(row)
        amplitude_row, expert_row, detector_row = history_rows(
            epoch, row, diagnostics
        )
        amplitude_rows.append(amplitude_row)
        expert_rows.append(expert_row)
        detector_rows.append(detector_row)

        write_rows(run_dir / "metrics.csv", metrics_rows, METRIC_FIELDS)
        write_rows(run_dir / "prompt_amplitude_history.csv", amplitude_rows)
        write_rows(run_dir / "expert_energy_history.csv", expert_rows)
        write_rows(run_dir / "detector_energy_history.csv", detector_rows)
        save_json(
            {"stages": stage_records},
            str(run_dir / "trainable_parameters_by_stage.json"),
        )

        checkpoint_metrics = dict(row)
        checkpoint_metrics["best_val_acc"] = best_val_acc
        checkpoint_metrics["best_epoch"] = best_epoch
        save_checkpoint(
            str(run_dir / "last.pt"),
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
        )
        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            best_epoch = epoch
            checkpoint_metrics["best_val_acc"] = best_val_acc
            checkpoint_metrics["best_epoch"] = best_epoch
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

        interval = int(vis_cfg.get("save_interval_epochs", 5))
        if (
            vis_cfg.get("enabled", True)
            and interval > 0
            and (epoch % interval == 0 or epoch == num_epochs)
        ):
            save_phase_visualizations(model, run_dir, epoch)
            save_light_field_visualizations(diagnostics, run_dir, epoch)

        final_targets = test_result[2]
        final_predictions = test_result[3]
        print(
            f"epoch {epoch:03d} stage {stage_idx} | "
            f"train {train_result[0]:.4f}/{train_result[1]:.4f} | "
            f"val {val_result[0]:.4f}/{val_result[1]:.4f} | "
            f"test {test_result[0]:.4f}/{test_result[1]:.4f} | "
            f"amps {[round(value, 4) for value in diagnostics['amplitudes'].tolist()]}"
        )

    save_history_plots(run_dir, metrics_rows, amplitude_rows, expert_rows)
    save_confusion_matrix(
        final_targets,
        final_predictions,
        num_classes,
        run_dir / "confusion_matrix.png",
    )
    summary = {
        "run_name": run_name,
        "dataset": config["dataset"].get("name"),
        "dataset_split": config["dataset"].get("split"),
        "num_classes": num_classes,
        "device": str(device),
        "seed": seed,
        "epochs": num_epochs,
        "best_validation_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_accuracy": metrics_rows[-1]["test_acc"],
        "final_test_loss": metrics_rows[-1]["test_loss"],
        "layout": model.layout.to_dict(),
        "distances_m": model.distances_m,
        "num_expert_layers": model.num_layers,
        "progressive_enabled": schedule.enabled,
        "progressive_order": schedule.order,
        "stage_epochs": schedule.stage_epochs,
        "optical_parameter_count": model.optical_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "final_prompt_amplitudes": amplitude_rows[-1],
        "final_expert_energy_ratios": expert_rows[-1],
        "readout_type": config.get("readout", {}).get(
            "type", "optical_only"
        ),
        "optimizer": optimizer_cfg,
        "architecture_report": architecture_report,
        "initial_diagnostics": initial_payload,
        "loss": config.get("loss", {}),
    }
    save_json(summary, str(run_dir / "summary.json"))
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
