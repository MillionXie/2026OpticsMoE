from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_phase_masks(surrogate: Any, path: Path, title: str) -> None:
    plt = _pyplot(); layers = list(surrogate.expert_layers)
    figure, axes = plt.subplots(len(layers) + 1, surrogate.geometry.num_experts,
                                figsize=(20, 12), squeeze=False)
    for layer_index, layer in enumerate(layers):
        for expert_index, phase_layer in enumerate(layer.experts):
            values = torch.remainder(phase_layer.phase().detach().cpu(), 2.0 * math.pi)
            image = axes[layer_index, expert_index].imshow(values, cmap="twilight", vmin=0, vmax=2.0 * math.pi)
            axes[layer_index, expert_index].set_title(f"L{layer_index + 1} E{expert_index}")
            axes[layer_index, expert_index].set_xlabel("x (pixel)"); axes[layer_index, expert_index].set_ylabel("y (pixel)")
            figure.colorbar(image, ax=axes[layer_index, expert_index], fraction=0.046)
    global_phase = torch.remainder(surrogate.global_phase.phase.phase().detach().cpu(), 2.0 * math.pi)
    image = axes[-1, 0].imshow(global_phase, cmap="twilight", vmin=0, vmax=2.0 * math.pi)
    axes[-1, 0].set_title("Global phase"); figure.colorbar(image, ax=axes[-1, 0], fraction=0.046)
    for axis in axes[-1, 1:]: axis.axis("off")
    figure.suptitle(title); path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=140, bbox_inches="tight"); plt.close(figure)


def save_training_curves(history: Sequence[dict[str, Any]], path: Path) -> None:
    if not history: return
    plt = _pyplot(); epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, label in (("loss_total", "total"), ("loss_vision", "vision"),
                       ("loss_answer", "answer"), ("loss_classification", "BCE")):
        if key in history[0]: axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set_title("Student losses"); axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in history], label="test")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[2].plot(epochs, [row["train_auroc"] for row in history], label="train")
    axes[2].plot(epochs, [row["test_auroc"] for row in history], label="test")
    axes[2].set_title("AUROC"); axes[2].legend()
    for axis in axes: axis.set_xlabel("epoch"); axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True); figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight"); plt.close(figure)


def save_confusion_matrix(matrix: Sequence[Sequence[int]], path: Path, title: str) -> None:
    plt = _pyplot(); values = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.5, 5)); image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2): axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["not_match", "match"]); axis.set_yticks([0, 1], ["not_match", "match"])
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title(title); figure.colorbar(image, ax=axis)
    path.parent.mkdir(parents=True, exist_ok=True); figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight"); plt.close(figure)
