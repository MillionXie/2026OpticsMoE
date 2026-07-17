from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import TASK_NAMES


def save_figures(
    output_dir: Path,
    history: Sequence[dict[str, Any]] | None = None,
    predictions: Sequence[dict[str, Any]] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if history is None:
        history = _read_history(output_dir / "training_history.csv")
    if predictions is None:
        predictions = _read_predictions(output_dir / "test_predictions.csv")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if history:
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.plot(
            [int(row["epoch"]) for row in history],
            [float(row["train_loss"]) for row in history],
            color="#2563eb",
            linewidth=2,
        )
        axis.set(xlabel="Epoch", ylabel="Smooth L1 loss", title="SPAQ multitask training loss")
        axis.grid(alpha=0.25)
        outputs.append(_save(fig, figures / "training_loss.png", plt))
    if predictions:
        for task in TASK_NAMES:
            rows = [row for row in predictions if row["task"] == task]
            truth = np.asarray([float(row["true_score"]) for row in rows])
            predicted = np.asarray([float(row["predicted_score"]) for row in rows])
            fig, axis = plt.subplots(figsize=(5.6, 5.2))
            axis.scatter(truth, predicted, s=18, alpha=0.55, color="#0f766e", edgecolors="none")
            axis.plot([0, 100], [0, 100], linestyle="--", color="#dc2626", linewidth=1.4)
            axis.set(
                xlim=(0, 100), ylim=(0, 100), xlabel="True score", ylabel="Predicted score",
                title=f"{task}: predicted vs. true",
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            outputs.append(_save(fig, figures / f"scatter_{task.lower()}.png", plt))
    return outputs


def _save(fig: Any, path: Path, plt: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
