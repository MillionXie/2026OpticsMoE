from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .io_utils import write_json


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_phase_masks(surrogate: Any, path: Path, title: str) -> None:
    plt = _pyplot()
    layers = list(surrogate.expert_layers)
    figure, axes = plt.subplots(len(layers) + 1, surrogate.geometry.num_experts, figsize=(20, 12), squeeze=False)
    for layer_index, layer in enumerate(layers):
        for expert_index, phase_layer in enumerate(layer.experts):
            values = torch.remainder(phase_layer.phase().detach().cpu(), 2.0 * math.pi)
            image = axes[layer_index, expert_index].imshow(values, cmap="twilight", vmin=0, vmax=2.0 * math.pi)
            axes[layer_index, expert_index].set_title(f"L{layer_index + 1} E{expert_index}")
            axes[layer_index, expert_index].set_xlabel("x (pixel)")
            axes[layer_index, expert_index].set_ylabel("y (pixel)")
            figure.colorbar(image, ax=axes[layer_index, expert_index], fraction=0.046)
    global_phase = torch.remainder(surrogate.global_phase.phase.phase().detach().cpu(), 2.0 * math.pi)
    image = axes[-1, 0].imshow(global_phase, cmap="twilight", vmin=0, vmax=2.0 * math.pi)
    axes[-1, 0].set_title("Global phase")
    figure.colorbar(image, ax=axes[-1, 0], fraction=0.046)
    for axis in axes[-1, 1:]:
        axis.axis("off")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history: Sequence[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    plt = _pyplot()
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [row["loss_total"] for row in history], label="total")
    axes[0].plot(epochs, [row["loss_hidden"] for row in history], label="hidden")
    axes[0].plot(epochs, [row["loss_regression"] for row in history], label="attribute")
    axes[0].set_title("Student losses"); axes[0].legend()
    axes[1].plot(epochs, [row["train_mae"] for row in history], label="train")
    axes[1].plot(epochs, [row["test_mae"] for row in history], label="test")
    axes[1].set_title("MAE (0-100)"); axes[1].legend()
    axes[2].plot(epochs, [row["train_srcc"] for row in history], label="train")
    axes[2].plot(epochs, [row["test_srcc"] for row in history], label="test")
    axes[2].plot(epochs, [row["test_plcc"] for row in history], label="test PLCC")
    axes[2].set_title("Correlation"); axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch"); axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=160, bbox_inches="tight"); plt.close(figure)


def save_scatter(true_scores: Sequence[float], predicted_scores: Sequence[float], path: Path,
                 title: str, task_name: str = "attribute") -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_scores, predicted_scores, s=12, alpha=0.45)
    axis.plot([0, 100], [0, 100], "k--", linewidth=1)
    axis.set(xlim=(0, 100), ylim=(0, 100), xlabel=f"True {task_name}",
             ylabel=f"Predicted {task_name}", title=title)
    axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=170, bbox_inches="tight"); plt.close(figure)


def save_debug_example(directory: Path, image: Any, sample_index: int, true_score: float,
                       predicted_score: float, input_field: torch.Tensor, routing: dict[str, torch.Tensor],
                       detector_intensity: torch.Tensor, student_hidden: torch.Tensor,
                       teacher_hidden: torch.Tensor, epoch: int, task_name: str) -> None:
    plt = _pyplot()
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / "input_rgb.png")
    tensors = {
        "optical_input_field": input_field.float(),
        "detector_intensity": detector_intensity.float(),
    }
    for name, values in tensors.items():
        torch.save(values, directory / f"{name}.pt")
        figure, axis = plt.subplots(figsize=(6, 5))
        shown = values.squeeze().numpy()
        plotted = axis.imshow(shown, cmap="magma")
        axis.set_title(f"{name} shape={tuple(values.shape)}")
        axis.set_xlabel("x (pixel)"); axis.set_ylabel("y (pixel)")
        figure.colorbar(plotted, ax=axis, label="intensity / feature value")
        figure.tight_layout(); figure.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(figure)
    student = student_hidden.detach().cpu().float()
    teacher = teacher_hidden.detach().cpu().float()
    difference = student - teacher
    torch.save(student, directory / "student_hidden.pt")
    torch.save(teacher, directory / "teacher_hidden.pt")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    limit = max(float(student.abs().quantile(0.99)), float(teacher.abs().quantile(0.99)), 1e-6)
    for axis, values, name in zip(axes, (teacher, student, difference), ("teacher", "student", "student-teacher")):
        bound = limit if name != "student-teacher" else max(float(difference.abs().quantile(0.99)), 1e-6)
        plotted = axis.imshow(values.numpy(), cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(name); axis.set_xlabel("hidden dim"); axis.set_ylabel("token")
        figure.colorbar(plotted, ax=axis)
    figure.tight_layout(); figure.savefig(directory / "hidden_comparison.png", dpi=140, bbox_inches="tight"); plt.close(figure)
    weights = routing["weights"][0].detach().cpu().float()
    selected = routing["selected_mask"][0].detach().cpu().bool()
    write_json(directory / "metadata.json", {
        "epoch": epoch, "sample_index": sample_index, "task": task_name,
        "true_score": true_score, "predicted_score": predicted_score,
        "absolute_error": abs(predicted_score - true_score),
        "routing_weights": weights.tolist(), "selected_experts": selected.nonzero().flatten().tolist(),
        "hidden_mse": float(F.mse_loss(student, teacher)),
        "hidden_layernorm_mse": float(F.mse_loss(F.layer_norm(student, (student.shape[-1],)),
                                                    F.layer_norm(teacher, (teacher.shape[-1],)))),
        "hidden_cosine_mean": float(F.cosine_similarity(student, teacher, dim=-1).mean()),
    })
