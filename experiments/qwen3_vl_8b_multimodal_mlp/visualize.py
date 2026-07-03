from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def generate_figures(run_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inference = _read_json(run_dir / "metrics" / "inference.json")
    training = _read_json(run_dir / "metrics" / "training.json")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if training:
        history = training["history"]
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
        axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation loss")
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="MLP loss")
        axes[0].legend()
        axes[1].plot(epochs, [100 * row["validation_top1_accuracy"] for row in history], label="Top-1")
        axes[1].plot(epochs, [100 * row["validation_top5_accuracy"] for row in history], label="Top-5")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation accuracy")
        axes[1].legend()
        outputs.extend(_save(fig, figures / "training_curves", plt))

    if inference:
        metrics = inference["metrics"]
        names = ["Top-1", "Top-5", "Macro F1"]
        values = [metrics["top1_accuracy"], metrics["top5_accuracy"], metrics["macro_f1"]]
        fig, axis = plt.subplots(figsize=(6.5, 4.2))
        bars = axis.bar(names, np.asarray(values) * 100, color=["#3b82f6", "#10b981", "#8b5cf6"])
        axis.set(ylabel="Score (%)", title="Classification performance", ylim=(0, 100))
        axis.bar_label(bars, fmt="%.2f")
        outputs.extend(_save(fig, figures / "classification_metrics", plt))

        components = inference["timing"]["components"]
        ordered = [
            ("Data loading", "data_loading_sec"),
            ("Image + text preprocess", "multimodal_preprocess_sec"),
            ("Host to GPU", "host_to_device_sec"),
            ("Full multimodal forward", "multimodal_forward_sec"),
            ("Answer hidden selection", "hidden_pooling_sec"),
            ("MLP", "mlp_forward_sec"),
            ("Postprocess", "postprocess_sec"),
        ]
        labels = [label for label, key in ordered if key in components]
        latency = [components[key]["mean_per_sample_ms"] for _, key in ordered if key in components]
        fig, axis = plt.subplots(figsize=(9, 4.8))
        bars = axis.bar(labels, latency, color="#2563eb")
        axis.set(ylabel="Mean latency (ms/image)", title="Synchronized inference latency breakdown")
        axis.tick_params(axis="x", rotation=25)
        axis.bar_label(bars, fmt="%.2f", padding=2)
        outputs.extend(_save(fig, figures / "latency_breakdown", plt))

        matrix = np.asarray(metrics["confusion_matrix"])
        fig, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(matrix, cmap="Blues", interpolation="nearest", aspect="auto")
        axis.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
        if matrix.shape[0] <= 20:
            class_names = list(metrics["per_class_accuracy"])
            axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
            axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        outputs.extend(_save(fig, figures / "confusion_matrix", plt))

        per_class = metrics["per_class_accuracy"]
        ordered_classes = sorted(per_class.items(), key=lambda item: item[1])
        fig_height = max(4.5, len(ordered_classes) * 0.16)
        fig, axis = plt.subplots(figsize=(8, fig_height))
        axis.barh([name for name, _ in ordered_classes], [100 * value for _, value in ordered_classes])
        axis.set(xlabel="Accuracy (%)", title="Per-class accuracy", xlim=(0, 100))
        axis.tick_params(axis="y", labelsize=6 if len(ordered_classes) > 30 else 8)
        outputs.extend(_save(fig, figures / "per_class_accuracy", plt))
    return outputs


def _save(fig: Any, base: Path, plt: Any) -> list[Path]:
    fig.tight_layout()
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
