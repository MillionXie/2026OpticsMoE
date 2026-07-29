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


def save_optical_debug_example(
    directory: Path,
    *,
    input_field: torch.Tensor,
    amplitude_slm: torch.Tensor,
    stage_fields: list[torch.Tensor],
    detector_intensity: torch.Tensor,
    detector_readout: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_mask: torch.Tensor,
    grid_rows: int,
    grid_cols: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _save_scalar_map(
        directory / "01_optical_input_field.png",
        input_field,
        "Optical input field after adapter, LayerNorm and Softplus",
        "amplitude",
    )
    _save_scalar_map(
        directory / "02_amplitude_slm_canvas.png",
        amplitude_slm,
        "Routed amplitude-SLM canvas",
        "amplitude",
    )
    for stage_index, field in enumerate(stage_fields, start=1):
        _save_scalar_map(
            directory / f"03_expert_stage_{stage_index:02d}_intensity.png",
            field.abs().square(),
            f"Expert stage {stage_index} output intensity",
            "intensity",
        )
    _save_scalar_map(
        directory / "04_detector_intensity.png",
        detector_intensity,
        "Physical CCD intensity over active footprint",
        "intensity",
    )
    _save_scalar_map(
        directory / "05_detector_readout_224.png",
        detector_readout,
        "224x224 pooled/LN/ReLU detector readout",
        "readout",
    )
    _save_routing_figure(
        directory / "06_routing_weights.png",
        routing_weights,
        selected_mask,
        grid_rows,
        grid_cols,
    )


def _save_scalar_map(
    path: Path,
    value: torch.Tensor,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = value.detach().float().cpu().squeeze().numpy()
    if array.ndim != 2:
        raise RuntimeError(f"Optical visualization must be 2-D, got {array.shape}")
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    artist = axis.imshow(array, cmap="viridis", vmin=0.0)
    axis.set_title(
        f"{title}\nshape={array.shape}, min={array.min():.3g}, "
        f"max={array.max():.3g}, mean={array.mean():.3g}"
    )
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    figure.colorbar(artist, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_routing_figure(
    path: Path,
    weights: torch.Tensor,
    selected: torch.Tensor,
    grid_rows: int,
    grid_cols: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = weights.detach().float().cpu().numpy()
    chosen = selected.detach().bool().cpu().numpy()
    expected = grid_rows * grid_cols
    if values.size != expected or chosen.size != expected:
        raise RuntimeError(
            f"Router visualization expected {expected} experts, got "
            f"weights={values.size}, selected={chosen.size}"
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    grid = values.reshape(grid_rows, grid_cols)
    artist = axes[0].imshow(grid, cmap="magma", vmin=0.0, vmax=max(1e-8, values.max()))
    axes[0].set_title("Top-k routing weights on 4x4 expert layout")
    axes[0].set_xlabel("expert column")
    axes[0].set_ylabel("expert row")
    figure.colorbar(artist, ax=axes[0], label="routing amplitude weight")
    colors = ["tab:blue" if flag else "lightgray" for flag in chosen]
    axes[1].bar(np.arange(expected), values, color=colors)
    axes[1].set_title("Per-expert routing weight")
    axes[1].set_xlabel("expert index")
    axes[1].set_ylabel("routing amplitude weight")
    axes[1].set_xticks(np.arange(expected))
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
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
