from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalDataset,
)

from .modeling import TwoPlaneD2NNClassifier, pil_images_to_amplitude


def _heatmap(
    values: torch.Tensor | np.ndarray,
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    rectangles: Sequence[tuple[int, int, int, int, str]] = (),
) -> None:
    array = (
        values.detach().float().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    image = axis.imshow(array, origin="upper", cmap=cmap, aspect="equal")
    for x0, y0, width, height, label in rectangles:
        axis.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        axis.text(
            x0 + width / 2,
            y0 + height / 2,
            label,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
        )
    axis.set_title(
        f"{title}\nshape={array.shape}, min={array.min():.3e}, "
        f"max={array.max():.3e}, mean={array.mean():.3e}"
    )
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label(colorbar_label)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_phase_masks(
    model: TwoPlaneD2NNClassifier, directory: Path, epoch: int | str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    first = model.first_phase.phase().detach().cpu()
    second = model.second_global_phase.phase().detach().cpu()
    label = f"{int(epoch):04d}" if isinstance(epoch, int) else str(epoch)
    _heatmap(
        first,
        directory / f"epoch_{label}_phase_plane_1_local224.png",
        title="Phase plane 1: local 224x224",
        colorbar_label="phase [rad]",
        cmap="twilight",
    )
    _heatmap(
        second,
        directory / f"epoch_{label}_phase_plane_2_global986.png",
        title="Phase plane 2: global 986x986",
        colorbar_label="phase [rad]",
        cmap="twilight",
    )
    torch.save(
        {
            "phase_plane_1_local224_rad": first,
            "phase_plane_2_global986_rad": second,
        },
        directory / f"epoch_{label}_phase_masks.pt",
    )


def save_training_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train CE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Detector-region cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        epochs,
        [100.0 * row["train_top1_accuracy"] for row in rows],
        label="sampled train",
    )
    if rows[0].get("test_top1_accuracy") is not None:
        axes[1].plot(
            epochs,
            [100.0 * row["test_top1_accuracy"] for row in rows],
            label="test",
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Top-1 accuracy [%]")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Closed-set classification accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path
) -> None:
    array = np.asarray(matrix, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(array, cmap="Blues")
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            axis.text(x, y, str(array[y, x]), ha="center", va="center", fontsize=7)
    short = [name[:16] for name in class_names]
    axis.set_xticks(range(len(short)), short, rotation=45, ha="right")
    axis.set_yticks(range(len(short)), short)
    axis.set_xlabel("predicted SKU")
    axis.set_ylabel("true SKU")
    axis.set_title("Grocery-10 D2NN confusion matrix")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.83)
    colorbar.set_label("sample count")
    figure.savefig(path, dpi=170)
    plt.close(figure)


@torch.no_grad()
def save_debug_examples(
    model: TwoPlaneD2NNClassifier,
    dataset: GroceryRetrievalDataset,
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    sample_count: int,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_index in range(min(sample_count, len(dataset))):
        item = dataset[dataset_index]
        sample = item["sample"]
        image = item["image"]
        sample_dir = output_dir / f"sample_{dataset_index:04d}_{sample.sku_name}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        image.save(sample_dir / "input_original.png")
        amplitude = pil_images_to_amplitude(
            [image], model.settings.input_encoding
        ).to(device)
        logits, fields = model(amplitude, return_intermediates=True)
        prediction = int(logits.argmax(1).item())
        probabilities = torch.softmax(logits, 1)[0].detach().cpu()
        input_crop = fields["input_amplitude_canvas"][
            0,
            model.input_start : model.input_end,
            model.input_start : model.input_end,
        ]
        before_second = fields["before_second_phase"][0].abs().square()[
            model.active_start : model.active_end,
            model.active_start : model.active_end,
        ]
        after_second = fields["after_second_phase"][0].abs().square()[
            model.active_start : model.active_end,
            model.active_start : model.active_end,
        ]
        detector = fields["detector_intensity"][0]
        rectangles = [
            (
                region.x0 - model.active_start,
                region.y0 - model.active_start,
                region.x1 - region.x0,
                region.y1 - region.y0,
                str(region.class_index),
            )
            for region in model.detector.regions
        ]
        _heatmap(
            input_crop,
            sample_dir / "01_input_scalar_amplitude.png",
            title=(
                f"Input amplitude ({model.settings.input_encoding}) "
                "co-planar with phase plane 1"
            ),
            colorbar_label="amplitude",
        )
        _heatmap(
            before_second,
            sample_dir / "02_intensity_before_global_phase.png",
            title="Intensity after 10 cm propagation to global phase",
            colorbar_label="intensity",
        )
        _heatmap(
            after_second,
            sample_dir / "03_intensity_after_global_phase.png",
            title="Intensity immediately after phase plane 2",
            colorbar_label="intensity",
        )
        _heatmap(
            detector,
            sample_dir / "04_detector_intensity_with_regions.png",
            title="Final square-law CCD intensity",
            colorbar_label="detector intensity",
            rectangles=rectangles,
        )
        figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        axis.bar(range(len(class_names)), probabilities.numpy())
        axis.set_xticks(
            range(len(class_names)),
            [name[:15] for name in class_names],
            rotation=40,
            ha="right",
        )
        axis.set_xlabel("fixed detector region / SKU")
        axis.set_ylabel("normalized detector probability")
        axis.set_ylim(0, max(1.0, float(probabilities.max()) * 1.1))
        axis.set_title(
            f"true={sample.sku_name}, predicted={class_names[prediction]}"
        )
        figure.savefig(sample_dir / "05_detector_region_probabilities.png", dpi=160)
        plt.close(figure)


def save_history_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
