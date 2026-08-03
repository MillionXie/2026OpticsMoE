from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.visualization import (
    save_optical_phase_figures,
)

from .objectives import density_from_logits


def save_examples(
    directory: Path, examples: list[dict[str, Any]], *, kind: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(examples):
        ground_truth = _array(row["density"])
        fixation = _array(row["fixation"])
        prediction = _array(density_from_logits(row["logits"].unsqueeze(0))[0])
        ground_truth_view = ground_truth / max(1.0e-8, float(ground_truth.max()))
        prediction_view = prediction / max(1.0e-8, float(prediction.max()))
        error = np.abs(prediction_view - ground_truth_view)
        figure, axes = plt.subplots(1, 5, figsize=(20, 4.3))
        panels = (
            (np.asarray(row["image"]), "Input", None),
            (ground_truth_view, "GT density", "magma"),
            (fixation, "Fixations", "gray"),
            (prediction_view, f"{kind} prediction", "magma"),
            (error, "Absolute error", "inferno"),
        )
        for axis, (value, title, cmap) in zip(axes, panels):
            image = axis.imshow(value, cmap=cmap, vmin=0, vmax=1)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            if cmap is not None:
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(str(row["sample_id"]))
        figure.tight_layout()
        figure.savefig(directory / f"sample_{index:03d}.png", dpi=160)
        plt.close(figure)


def save_training_curves(
    path: Path, history: list[dict[str, Any]], *, prefix: str
) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].set_title("Training loss")
    axes[1].plot(
        epochs, [row["validation_cc"] for row in history], label="CC"
    )
    axes[1].plot(
        epochs, [row["validation_sim"] for row in history], label="SIM"
    )
    axes[1].set_title("Validation similarity")
    axes[1].legend()
    axes[2].plot(
        epochs, [row["validation_nss"] for row in history], label="NSS"
    )
    axes[2].plot(
        epochs, [row["validation_auc_judd"] for row in history], label="AUC-J"
    )
    axes[2].set_title("Validation fixation metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.suptitle(f"{prefix} SALICON training")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_optical_parameters(core: Any, directory: Path) -> None:
    save_optical_phase_figures(core, directory)


def _array(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    return np.squeeze(array)

