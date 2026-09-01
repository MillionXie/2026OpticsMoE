from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_history_plot(csv_path: Path, output_path: Path, title: str) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, label in (("train_loss", "train"), ("validation_loss", "validation"), ("test_loss", "test")):
        if name in rows[0] and rows[0][name] not in {"", "nan"}:
            axes[0].plot(epochs, [float(row[name]) for row in rows], label=label)
    for name, label in (("train_accuracy", "train"), ("validation_accuracy", "validation"), ("test_accuracy", "test")):
        if name in rows[0] and rows[0][name] not in {"", "nan"}:
            axes[1].plot(epochs, [float(row[name]) for row in rows], label=label)
    axes[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy", ylim=(0.0, 1.0))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_phase_overview(
    phases: torch.Tensor,
    residual_weights: torch.Tensor,
    output_path: Path,
    *,
    title: str,
) -> None:
    values = phases.detach().float().cpu().numpy()
    weights = residual_weights.detach().float().cpu().numpy()
    plt = _matplotlib()
    figure, axes = plt.subplots(4, 5, figsize=(13, 10), constrained_layout=True)
    image = None
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(values[index], cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
        axis.set_title(
            f"stage {index + 1}\nmain={weights[index,0]:.3f}, skip={weights[index,1]:.3f}", fontsize=8
        )
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.65, label="phase (rad)")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_optical_example(intermediates: dict[str, object], output_path: Path, *, title: str) -> None:
    stages = intermediates["stages"]
    selected = [0, 4, 9, 14, 19]
    plt = _matplotlib()
    figure, axes = plt.subplots(3, len(selected), figsize=(14, 8), constrained_layout=True)
    for column, stage_index in enumerate(selected):
        stage = stages[stage_index]
        arrays = (
            (stage["intensity"][0].detach().float().cpu().numpy(), "intensity", "viridis"),
            (stage["normalized"][0].detach().float().cpu().numpy(), "LN intensity", "coolwarm"),
            (stage["reloaded"][0].detach().float().cpu().numpy(), "reload amplitude", "magma"),
        )
        for row, (array, label, cmap) in enumerate(arrays):
            axes[row, column].imshow(array, cmap=cmap)
            axes[row, column].set_title(f"stage {stage_index + 1} {label}", fontsize=8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    plt = _matplotlib()
    normalized = matrix.astype(np.float64) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(xlabel="prediction", ylabel="ground truth", title=title)
    figure.colorbar(image, ax=axis, label="row-normalized fraction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_comparison_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(row["method"]) for row in rows]
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(methods, [float(row["test_accuracy_mean"]) for row in rows])
    axes[0].set(ylabel="test accuracy", ylim=(0.0, 1.0))
    axes[1].bar(methods, [float(row.get("relative_parameter_drift_mean", 0.0)) for row in rows])
    axes[1].set(ylabel="relative endpoint drift")
    axes[2].bar(methods, [float(row.get("endpoint_cosine_to_bp_mean", 0.0)) for row in rows])
    axes[2].set(ylabel="endpoint cosine to BP", ylim=(-1.0, 1.0))
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "method_comparison.png", dpi=180)
    plt.close(figure)
