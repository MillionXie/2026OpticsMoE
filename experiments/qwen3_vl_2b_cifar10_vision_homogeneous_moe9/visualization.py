from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F


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
    axes[1].plot(epochs, [row["test_top1_accuracy"] for row in history], label="test top-1")
    axes[1].plot(epochs, [row["test_macro_f1"] for row in history], label="test macro-F1")
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


def save_debug_example(directory: Path, image: Any, sample_index: int, true_label: int,
                       class_names: Sequence[str], logits: torch.Tensor, input_field: torch.Tensor,
                       routing: dict[str, torch.Tensor], fanout_field: torch.Tensor,
                       stage_fields: list[dict[str, torch.Tensor]], detector_intensity: torch.Tensor,
                       student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, epoch: int) -> None:
    """Save one compact, physically named diagnostic package."""
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(directory / "input_rgb.png")
    torch.save(input_field.detach().cpu(), directory / "optical_input_field.pt")
    torch.save(student_hidden.detach().cpu(), directory / "student_hidden.pt")
    torch.save(teacher_hidden.detach().cpu(), directory / "teacher_hidden.pt")
    _save_scalar_field(input_field, directory / "optical_input_field.png", "Optical input field", "viridis")
    prompt_amplitude = routing["prompt_amplitude"][0].detach().cpu()
    _save_scalar_field(prompt_amplitude, directory / "prompt_expert_amplitude.png",
                       "Prompt amplitude: nine expert regions", "viridis")
    weights = routing["weights"][0].detach().cpu()
    probabilities = routing["probabilities"][0].detach().cpu()
    selected = routing["selected_mask"][0].detach().cpu()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(weights))
    bars = axis.bar(positions, weights.numpy(), color=["tab:orange" if value else "0.75" for value in selected.tolist()])
    axis.plot(positions, probabilities.numpy(), "o--", color="tab:blue", label="dense probability")
    axis.set(xlabel="Expert index", ylabel="Routing value", title="Input-dependent top-k routing")
    axis.set_xticks(positions); axis.set_ylim(0.0, max(1.0, float(probabilities.max()) * 1.15)); axis.legend()
    for bar, value in zip(bars, weights.tolist()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    figure.tight_layout(); figure.savefig(directory / "routing_weights.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    _save_complex_field(fanout_field[0], directory / "fanout_to_experts", "Fan-out field at expert bank input")
    for layer_index, values in enumerate(stage_fields, start=1):
        _save_complex_field(values["before_oeo"][0], directory / f"stage_{layer_index:02d}_before_oeo",
                            f"Stage {layer_index} before OEO")
        after = values["after_oeo"][0]
        _save_scalar_field(after.abs().square(), directory / f"stage_{layer_index:02d}_after_oeo_intensity.png",
                           f"Stage {layer_index} after OEO intensity", "magma")
    _save_scalar_field(detector_intensity, directory / "final_detector_intensity.png", "Final detector intensity", "magma")
    _save_hidden_comparison(student_hidden, teacher_hidden, directory / "hidden_comparison.png", normalized=False)
    _save_hidden_comparison(student_hidden, teacher_hidden, directory / "hidden_layernorm_comparison.png", normalized=True)
    prediction = int(logits.argmax().item())
    metadata = {
        "epoch": int(epoch), "sample_index": int(sample_index), "true_label": int(true_label),
        "true_name": class_names[int(true_label)], "pred_label": prediction,
        "pred_name": class_names[prediction], "correct": prediction == int(true_label),
        "visual_token_count": int(student_hidden.shape[0]), "routing_weights": weights.tolist(),
        "routing_probabilities": probabilities.tolist(),
        "selected_experts": torch.nonzero(selected, as_tuple=False).flatten().tolist(),
        "logits": logits.detach().cpu().tolist(),
        "hidden_mse": float(F.mse_loss(student_hidden.float(), teacher_hidden.float())),
        "hidden_layernorm_mse": float(F.mse_loss(
            F.layer_norm(student_hidden.float(), (student_hidden.shape[-1],)),
            F.layer_norm(teacher_hidden.float(), (teacher_hidden.shape[-1],)))),
        "hidden_cosine_mean": float(F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1).mean()),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_complex_field(field: torch.Tensor, stem: Path, title: str) -> None:
    intensity = field.detach().cpu().to(torch.complex64).abs().square()
    phase = torch.angle(field.detach().cpu().to(torch.complex64))
    _save_scalar_field(intensity, stem.with_name(stem.name + "_intensity.png"), title + " intensity", "magma")
    _save_scalar_field(phase, stem.with_name(stem.name + "_phase.png"), title + " phase", "twilight", -math.pi, math.pi)


def _save_scalar_field(tensor: torch.Tensor, path: Path, title: str, cmap: str,
                       vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt
    values = tensor.detach().cpu().float().numpy()
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set(xlabel="x [pixel]", ylabel="y [pixel]", title=f"{title}\nshape={list(values.shape)} min={values.min():.3g} max={values.max():.3g}")
    figure.colorbar(image, ax=axis)
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight"); plt.close(figure)


def _save_hidden_comparison(student: torch.Tensor, teacher: torch.Tensor, path: Path, normalized: bool) -> None:
    import matplotlib.pyplot as plt
    student = student.detach().cpu().float()
    teacher = teacher.detach().cpu().float()
    if normalized:
        student = F.layer_norm(student, (student.shape[-1],))
        teacher = F.layer_norm(teacher, (teacher.shape[-1],))
    difference = student - teacher
    limit = max(float(student.abs().quantile(0.99)), float(teacher.abs().quantile(0.99)), 1e-6)
    diff_limit = max(float(difference.abs().quantile(0.99)), 1e-6)
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for axis, values, name, bound in ((axes[0], teacher, "Teacher", limit),
                                      (axes[1], student, "Student", limit),
                                      (axes[2], difference, "Student - teacher", diff_limit)):
        image = axis.imshow(values.numpy(), aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        axis.set_ylabel("Token"); axis.set_title(name); figure.colorbar(image, ax=axis, shrink=0.8)
    axes[-1].set_xlabel("Hidden dimension")
    figure.suptitle("LayerNorm hidden comparison" if normalized else "Raw hidden comparison")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight"); plt.close(figure)
