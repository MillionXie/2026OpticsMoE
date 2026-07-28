from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
from PIL import Image


def save_prediction_panel(
    path: Path,
    *,
    image: Image.Image,
    ground_truth: torch.Tensor,
    student_probability: torch.Tensor | None,
    teacher_probability: torch.Tensor | None,
    title: str,
    threshold: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: list[tuple[str, np.ndarray, str | None, float | None, float | None]] = [
        ("Input image", np.asarray(image.convert("RGB")), None, None, None),
        ("Ground-truth mask", _array(ground_truth), "gray", 0.0, 1.0),
    ]
    if teacher_probability is not None:
        panels.append(("Electronic teacher probability", _array(teacher_probability), "viridis", 0.0, 1.0))
    if student_probability is not None:
        probability = _array(student_probability)
        truth = _array(ground_truth)
        panels.extend(
            [
                ("Optical student probability", probability, "viridis", 0.0, 1.0),
                ("Binary prediction", (probability >= threshold).astype(np.float32), "gray", 0.0, 1.0),
                ("Absolute error", np.abs(probability - truth), "magma", 0.0, 1.0),
            ]
        )
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for axis, (name, value, cmap, vmin, vmax) in zip(axes, panels):
        artist = axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        if cmap is not None:
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(path: Path, history: list[dict[str, Any]], *, epochs_key: str = "epoch") -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row[epochs_key]) for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    if "test_loss" in history[0]:
        axes[0].plot(epochs, [float(row["test_loss"]) for row in history], label="test/observation")
    axes[0].set_title("Segmentation loss")
    axes[0].legend()
    for metric, axis, title in (
        ("test_mean_iou", axes[1], "Test mean IoU"),
        ("test_mean_dice", axes[2], "Test mean Dice"),
    ):
        if metric in history[0]:
            axis.plot(epochs, [float(row[metric]) for row in history])
        axis.set_title(title)
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_optical_phase_figures(core: Any, directory: Path) -> None:
    """Save effective expert/global phase without changing raw checkpoint values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experts = core.expert_layers[0].experts
    phases = [_effective_phase(module).numpy() for module in experts]
    figure, axes = plt.subplots(4, 4, figsize=(14, 13))
    for index, (axis, phase) in enumerate(zip(axes.flat, phases)):
        artist = axis.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
        axis.set_title(f"Expert {index}")
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="phase (rad)")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "expert_phase_overview.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    global_phase = _effective_phase(core.global_phase.phase).numpy()
    figure, axis = plt.subplots(figsize=(7, 6))
    artist = axis.imshow(
        global_phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi
    )
    axis.set_title("Global phase")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    figure.colorbar(artist, ax=axis, label="phase (rad)")
    figure.tight_layout()
    figure.savefig(directory / "global_phase.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _array(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().squeeze().numpy()
    if array.ndim != 2:
        raise RuntimeError(f"Visualization mask must be 2-D after squeeze, got {array.shape}")
    return array


def _effective_phase(module: Any) -> torch.Tensor:
    raw = module.raw_phase.detach().float().cpu()
    if module.parameterization == "sigmoid":
        return 2.0 * math.pi * raw.sigmoid()
    return torch.remainder(raw, 2.0 * math.pi)
