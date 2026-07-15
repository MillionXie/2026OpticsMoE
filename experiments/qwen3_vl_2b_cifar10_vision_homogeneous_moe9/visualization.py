from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["loss_total"] for row in history], label="total")
    axes[0].plot(epochs, [row["loss_hidden"] for row in history], label="hidden")
    axes[0].plot(epochs, [row["loss_kd"] for row in history], label="KD")
    axes[0].plot(epochs, [row["loss_ce"] for row in history], label="CE")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Student losses")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_top1_accuracy"] for row in history], label="train top-1")
    axes[1].plot(epochs, [row["validation_top1_accuracy"] for row in history], label="validation top-1")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history], label="validation macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Metric", title="Classification")
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_phase_masks(surrogate: Any, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    layers = len(surrogate.expert_layers)
    figure, axes = plt.subplots(layers, surrogate.geometry.num_experts, figsize=(18, 10))
    for layer_index, layer in enumerate(surrogate.expert_layers):
        for expert_index, phase_layer in enumerate(layer.experts):
            axis = axes[layer_index, expert_index]
            image = axis.imshow(torch.remainder(phase_layer.phase().detach().cpu(), 2.0 * math.pi), cmap="twilight", vmin=0, vmax=2.0 * math.pi)
            axis.set_title(f"L{layer_index + 1} E{expert_index}", fontsize=7)
            axis.set_xticks([]); axis.set_yticks([])
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Phase [rad]", shrink=0.7)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    global_phase = torch.remainder(surrogate.global_phase.phase.phase().detach().cpu(), 2.0 * math.pi)
    figure, axis = plt.subplots(figsize=(6, 5.5))
    image = axis.imshow(global_phase, cmap="twilight", vmin=0, vmax=2.0 * math.pi)
    axis.set_title(f"{title}: global 450x450 phase")
    axis.set_xlabel("x [pixel]"); axis.set_ylabel("y [pixel]")
    figure.colorbar(image, ax=axis, label="Phase [rad]")
    figure.tight_layout()
    figure.savefig(path.with_name(path.stem + "_global.png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title("Student confusion matrix")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, int(values[row, column]), ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="Samples")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
