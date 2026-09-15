from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from .settings import Settings, load_settings


def visualize_saved_results(settings: Settings) -> None:
    path = settings.output_dir / "retrieval_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval results are missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    teacher = [
        row
        for row in rows
        if row["system"] == "frozen_teacher_query_vs_frozen_teacher_gallery"
    ]
    student = [
        row
        for row in rows
        if row["system"] in {
            "optical_student_query_vs_optical_student_gallery",
            "electronic_student_query_vs_electronic_student_gallery",
        }
    ]
    electronic = any(
        row["system"] == "electronic_student_query_vs_electronic_student_gallery"
        for row in student
    )
    rng = random.Random(settings.random_seed)
    if teacher:
        _plot_retrieval_rows(
            _sample(teacher, settings.visualization_sample_count, rng),
            settings.output_dir / "teacher_retrieval_examples.png",
            "Frozen Qwen3-VL-Embedding Teacher Retrieval",
        )
    _plot_retrieval_rows(
        _sample(student, settings.visualization_sample_count, rng),
        settings.output_dir / "student_retrieval_examples.png",
        "Electronic Student Retrieval" if electronic else "Optical Student Retrieval",
    )
    failures = [row for row in student if row["top1_correct"].lower() == "false"]
    _plot_retrieval_rows(
        _sample(failures, settings.visualization_sample_count, rng),
        settings.output_dir / "student_failure_cases.png",
        "Electronic Student Top-1 Failure Cases" if electronic else "Optical Student Top-1 Failure Cases",
        empty_message="No student Top-1 failures were found.",
    )
    _plot_training_loss(settings.output_dir / "train_log.csv", settings.output_dir / "training_loss.png")


def _sample(
    rows: Sequence[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    values = list(rows)
    if len(values) <= count:
        return values
    return rng.sample(values, count)


def _plot_retrieval_rows(
    rows: Sequence[dict[str, str]],
    path: Path,
    title: str,
    *,
    empty_message: str = "No examples available.",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        figure, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.5, empty_message, ha="center", va="center", fontsize=14)
        axis.set_title(title)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(14, 3.5 * len(rows)), squeeze=False, constrained_layout=True
    )
    for row_index, row in enumerate(rows):
        panels = [
            (
                row["query_image_path"],
                f"Query\ntrue: {row['true_sku']}",
            )
        ]
        for rank in range(1, 4):
            panels.append(
                (
                    row[f"top{rank}_gallery_path"],
                    f"Top-{rank}: {row[f'top{rank}_sku']}\n"
                    f"cos={float(row[f'top{rank}_similarity']):.4f}",
                )
            )
        for axis, (image_path, panel_title) in zip(axes[row_index], panels):
            with Image.open(image_path) as source:
                image = ImageOps.contain(source.convert("RGB"), (400, 400))
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(panel_title, fontsize=9)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_training_loss(csv_path: Path, path: Path) -> None:
    if not csv_path.is_file():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, label in (
        ("total_loss", "Total"),
        ("kd_loss", "Embedding KD"),
        ("retrieval_loss", "Supervised contrastive"),
        ("gallery_loss", "Prototype retrieval CE"),
    ):
        axis.plot(epochs, [float(row[key]) for row in rows], label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Student Retrieval Training Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    visualize_saved_results(settings)
    print(f"Retrieval figures saved under {settings.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
