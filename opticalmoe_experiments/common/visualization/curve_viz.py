from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


PathLike = Union[str, Path]


def save_training_curves(rows, path: PathLike) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train", linewidth=2.0)
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="Validation", linewidth=2.0)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="Train", linewidth=2.0)
    axes[1].plot(epochs, [row["val_acc"] for row in rows], label="Validation", linewidth=2.0)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(color="#b0b0b0", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, class_names, path: PathLike) -> None:
    num_classes = len(class_names)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for target, pred in zip(targets.view(-1), preds.view(-1)):
        if 0 <= int(target) < num_classes and 0 <= int(pred) < num_classes:
            matrix[int(target), int(pred)] += 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    if num_classes <= 26:
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return matrix
