from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_phase_masks(model: torch.nn.Module, figures_dir: Path, epoch: int) -> None:
    phases = [layer.phase_wrapped().detach().cpu().numpy() for layer in model.optical_layers]
    epoch_name = f"epoch_{epoch:04d}"
    layer_dir = figures_dir / "phase_masks" / epoch_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    for index, phase in enumerate(phases, start=1):
        fig, ax = plt.subplots(figsize=(5.2, 4.7))
        image = ax.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
        ax.set_title(f"Optical layer {index} phase mask")
        ax.axis("off")
        bar = fig.colorbar(image, ax=ax)
        bar.set_ticks([0.0, math.pi, 2.0 * math.pi])
        bar.set_ticklabels(["0", "π", "2π"])
        fig.tight_layout()
        fig.savefig(layer_dir / f"layer_{index}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1), squeeze=False)
    image = None
    for index, (ax, phase) in enumerate(zip(axes[0], phases), start=1):
        image = ax.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
        ax.set_title(f"Layer {index}")
        ax.axis("off")
    if image is not None:
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
        bar.set_ticks([0.0, math.pi, 2.0 * math.pi])
        bar.set_ticklabels(["0", "π", "2π"])
    fig.suptitle("Five wrapped phase masks")
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.03, top=0.86, wspace=0.05)
    output = figures_dir / "phase_masks" / f"{epoch_name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    fig.savefig(layer_dir / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_light_fields(diagnostics: dict[str, Any], figures_dir: Path, epoch: int, sample_index: int = 0) -> None:
    epoch_dir = figures_dir / "light_fields" / f"epoch_{epoch:04d}" / f"sample_{sample_index:03d}"
    entries: list[tuple[str, str, torch.Tensor]] = [
        ("input_amplitude", "Input amplitude", diagnostics["input_amplitude"][sample_index]),
    ]
    for layer_index, intensity in enumerate(diagnostics["layer_intensities"], start=1):
        entries.append((f"after_layer_{layer_index}_intensity", f"After layer {layer_index} intensity", intensity[sample_index]))
    entries.append(("detector_readout_input", "Detector / readout input", diagnostics["detector_input"][sample_index]))
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, title, value) in enumerate(entries):
        _save_field(value, epoch_dir / f"{index:02d}_{name}.png", title)
    columns = 4
    rows = math.ceil(len(entries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for ax, (_, title, value) in zip(axes.ravel(), entries):
        ax.imshow(_log_image(value), cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")
    fig.suptitle("Five-layer optical propagation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(epoch_dir / "overview.png", dpi=160)
    plt.close(fig)


def save_detector_outputs(diagnostics: dict[str, Any], figures_dir: Path, epoch: int, max_samples: int = 8) -> None:
    values = diagnostics["detector_input"][:max_samples].detach().cpu()
    columns = min(4, len(values))
    rows = math.ceil(len(values) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for index, (ax, value) in enumerate(zip(axes.ravel(), values)):
        ax.imshow(_log_image(value), cmap="inferno")
        ax.set_title(f"Sample {index}")
        ax.axis("off")
    for ax in axes.ravel()[len(values):]:
        ax.axis("off")
    fig.suptitle("Final detector outputs")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = figures_dir / "detector_outputs" / f"epoch_{epoch:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(epochs, [row["train_top1_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_top1_accuracy"] for row in history], label="validation")
    axes[1].set_title("Top-1 accuracy")
    axes[2].plot(epochs, [row["validation_macro_f1"] for row in history], label="macro-F1")
    axes[2].plot(epochs, [row["validation_balanced_accuracy"] for row in history], label="balanced accuracy")
    axes[2].set_title("Imbalance-aware validation metrics")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BDD100K Weather-4 confusion matrix")
    threshold = values.max() / 2.0 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, str(values[row, column]), ha="center", va="center", color="white" if values[row, column] > threshold else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _log_image(value: torch.Tensor) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().float()
    return torch.log10(tensor / tensor.max().clamp_min(1e-12) + 1e-8).numpy()


def _save_field(value: torch.Tensor, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    image = ax.imshow(_log_image(value), cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
