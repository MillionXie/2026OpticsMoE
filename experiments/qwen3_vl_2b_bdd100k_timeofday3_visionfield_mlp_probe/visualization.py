from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def save_input_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def save_field(field: torch.Tensor, path: Path, title: str) -> None:
    values = field.detach().float().cpu().numpy()
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="viridis", vmin=0)
    axis.set_title(title)
    axis.set_xlabel("optical channel")
    axis.set_ylabel("visual token row")
    figure.colorbar(image, ax=axis, label="nonnegative encoded value")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_training_curves(history: Sequence[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set_title("Probe loss"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_top1_accuracy"] for row in history], label="train top-1")
    axes[1].plot(epochs, [row["validation_top1_accuracy"] for row in history], label="validation top-1")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history], label="validation macro-F1")
    axes[1].set_title("Probe metrics"); axes[1].legend(); axes[1].grid(alpha=0.25)
    for axis in axes: axis.set_xlabel("epoch")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180); plt.close(figure)


def save_confusion_matrix(matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title("Vision-field probe")
    figure.colorbar(image, ax=axis); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)

