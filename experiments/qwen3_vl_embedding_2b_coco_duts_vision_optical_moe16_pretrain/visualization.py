from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io_utils import write_json
from .modeling import preprocess_vision


def save_training_curves(
    path: Path,
    history: list[dict[str, Any]],
    *,
    stage: str,
) -> None:
    if not history:
        return
    plt = _pyplot()
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    if "observation_loss" in history[0]:
        axes[0].plot(
            epochs,
            [float(row["observation_loss"]) for row in history],
            label="observation split",
        )
    axes[0].set_title(f"{stage}: loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    if stage == "COCO feature distillation":
        axes[1].plot(
            epochs,
            [float(row["train_cosine_similarity"]) for row in history],
            label="train cosine",
        )
        axes[1].plot(
            epochs,
            [float(row["observation_cosine_similarity"]) for row in history],
            label="val cosine",
        )
        axes[1].set_ylabel("cosine similarity")
    else:
        axes[1].plot(
            epochs,
            [float(row["train_mean_iou"]) for row in history],
            label="train mIoU",
        )
        axes[1].plot(
            epochs,
            [float(row["test_mean_iou"]) for row in history],
            label="DUTS-TE mIoU",
        )
        axes[1].plot(
            epochs,
            [float(row["test_mean_dice"]) for row in history],
            label="DUTS-TE Dice",
        )
        axes[1].set_ylabel("metric")
    axes[1].set_title(f"{stage}: quality")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def save_segmentation_examples(
    model: Any,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    epoch: int,
    phase: str,
) -> int:
    model.eval()
    saved = 0
    directory = (
        settings.output_dir
        / "figures"
        / "duts_examples"
        / f"epoch_{epoch:04d}_{phase}"
    )
    for batch in loader:
        inputs = preprocess_vision(processor, batch["images"], model.backbone.device)
        logits, spatial, ccd = model(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        probabilities = logits.float().sigmoid().cpu()
        for row in range(len(batch["images"])):
            if saved >= settings.visualization_sample_count:
                return saved
            sample_directory = directory / f"sample_{saved:03d}"
            sample_directory.mkdir(parents=True, exist_ok=True)
            image = np.asarray(batch["images"][row].convert("RGB"))
            target = batch["masks"][row, 0].cpu().numpy()
            probability = probabilities[row, 0].numpy()
            binary = probability >= 0.5
            error = np.abs(binary.astype(np.float32) - target)
            _save_prediction_panel(
                sample_directory / "prediction.png",
                image,
                target,
                probability,
                binary,
                error,
            )
            _save_heatmap(
                sample_directory / "ccd_readout_224.png",
                ccd[row].detach().cpu().float().numpy(),
                "Physical CCD pooled/LN/ReLU readout [224,224]",
                "intensity feature",
                sequential=True,
            )
            _save_heatmap(
                sample_directory / "recombined_spatial_mean.png",
                spatial[row].detach().cpu().float().mean(0).numpy(),
                "Residual recombined optical feature (channel mean)",
                "feature value",
                sequential=False,
            )
            write_json(
                sample_directory / "metadata.json",
                {
                    "epoch": epoch,
                    "phase": phase,
                    "sample_id": batch["sample_ids"][row],
                    "image_path": batch["image_paths"][row],
                    "mask_path": batch["mask_paths"][row],
                    "image_grid_thw": inputs["image_grid_thw"][row]
                    .detach()
                    .cpu()
                    .long()
                    .tolist(),
                    "optical_spatial_shape": list(spatial[row].shape),
                    "ccd_shape": list(ccd[row].shape),
                },
            )
            saved += 1
    return saved


def save_phase_masks(core: Any, directory: Path) -> None:
    plt = _pyplot()
    directory.mkdir(parents=True, exist_ok=True)
    for stage_index, plane in enumerate(core.expert_layers, start=1):
        phases = [
            expert.phase().detach().cpu().numpy()
            for expert in plane.experts
        ]
        figure, axes = plt.subplots(4, 4, figsize=(12, 11))
        for expert_index, (axis, phase) in enumerate(zip(axes.flat, phases)):
            image = axis.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
            axis.set_title(f"Expert {expert_index}")
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="phase (rad)")
        figure.suptitle(f"Optical expert phase masks — stage {stage_index}")
        figure.tight_layout()
        figure.savefig(
            directory / f"expert_phase_stage_{stage_index}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)
    global_phase = core.global_phase.phase.phase().detach().cpu().numpy()
    _save_heatmap(
        directory / "global_phase.png",
        global_phase,
        "Global phase mask",
        "phase (rad)",
        sequential=False,
        cmap="twilight",
        limits=(0.0, 2.0 * math.pi),
    )


def _save_prediction_panel(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    binary: np.ndarray,
    error: np.ndarray,
) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 5, figsize=(18, 4))
    values = [
        (image, "Input RGB", None, None),
        (target, "Ground truth", "gray", (0.0, 1.0)),
        (probability, "Prediction probability", "viridis", (0.0, 1.0)),
        (binary, "Binary prediction", "gray", (0.0, 1.0)),
        (error, "Absolute binary error", "magma", (0.0, 1.0)),
    ]
    for axis, (value, title, cmap, limits) in zip(axes, values):
        kwargs: dict[str, Any] = {"cmap": cmap}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        shown = axis.imshow(value, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        if cmap is not None:
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_heatmap(
    path: Path,
    value: np.ndarray,
    title: str,
    colorbar_label: str,
    *,
    sequential: bool,
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.8, 5.8))
    if limits is None and not sequential:
        limit = float(max(abs(np.nanmin(value)), abs(np.nanmax(value)), 1e-8))
        limits = (-limit, limit)
    shown = axis.imshow(
        value,
        cmap=cmap or ("viridis" if sequential else "coolwarm"),
        vmin=None if limits is None else limits[0],
        vmax=None if limits is None else limits[1],
    )
    axis.set_title(
        f"{title}\nshape={list(value.shape)} min={np.nanmin(value):.4g} "
        f"max={np.nanmax(value):.4g} mean={np.nanmean(value):.4g}"
    )
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    figure.colorbar(shown, ax=axis, label=colorbar_label)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
