from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from .datasets import denormalize_clip_image
from .io_utils import write_json


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        raise RuntimeError("matplotlib is required for experiment visualizations") from error
    return plt


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | int | list[int]]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "num_negative": int(value.lt(0).sum()),
        "negative_ratio": float(value.lt(0).float().mean()),
        "energy_sum": float(value.square().sum()),
    }


def save_heatmap(
    tensor: torch.Tensor,
    path: Path,
    title: str,
    *,
    center_zero: bool,
    percentile: float,
    cmap: str | None = None,
) -> None:
    plt = _pyplot()
    value = tensor.detach().float().cpu().squeeze()
    if value.ndim != 2:
        raise ValueError(f"Heatmap requires 2-D tensor, got {tuple(value.shape)}")
    flat = value.abs().flatten()
    limit = float(torch.quantile(flat, percentile / 100.0)) if flat.numel() else 1.0
    limit = max(limit, 1e-12)
    if center_zero:
        vmin, vmax = -limit, limit
        cmap = cmap or "coolwarm"
    else:
        vmin, vmax = 0.0, limit
        cmap = cmap or "viridis"
    stats = tensor_stats(value)
    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = axis.imshow(value.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axis.set_xlabel("x / feature column")
    axis.set_ylabel("y / token row")
    axis.set_title(
        f"{title}\nshape={stats['shape']} min={stats['min']:.3g} "
        f"max={stats['max']:.3g} mean={stats['mean']:.3g} std={stats['std']:.3g}"
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("signed value" if center_zero else "intensity / amplitude")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_debug_examples(
    *,
    epoch: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_indices: torch.Tensor,
    paths: list[str],
    class_names: list[str],
    model,
    output_dir: Path,
    sample_count: int,
    percentile: float,
    save_raw: bool,
) -> int:
    plt = _pyplot()
    count = min(int(sample_count), len(images))
    debug = model.debug_state()
    root = output_dir / "figures" / "debug_examples" / f"epoch_{epoch:04d}"
    for sample in range(count):
        sample_index = int(sample_indices[sample])
        directory = root / f"sample_{sample_index:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        image = denormalize_clip_image(images[sample].detach().cpu())
        figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        axis.set_title(
            f"ImageNet input: {class_names[int(labels[sample])]}\n{Path(paths[sample]).name}"
        )
        figure.savefig(directory / "input_image.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        metadata: dict[str, Any] = {
            "epoch": int(epoch),
            "sample_index": sample_index,
            "image_path": paths[sample],
            "label": int(labels[sample]),
            "class_name": class_names[int(labels[sample])],
            "captured_blocks": sorted(debug),
            "blocks": {},
        }
        for block_index, state in debug.items():
            block_dir = directory / f"block_{block_index:02d}"
            block_dir.mkdir(parents=True, exist_ok=True)
            block_report: dict[str, Any] = {
                "selected_indices": (
                    state["selected_indices"][sample].tolist()
                    if "selected_indices" in state
                    else None
                ),
                "routing_weights": (
                    state["routing_weights"][sample].tolist()
                    if "routing_weights" in state
                    else None
                ),
            }
            for key in ("token_optical_input", "channel_optical_input"):
                if key in state:
                    tensor = state[key][sample]
                    save_heatmap(
                        tensor,
                        block_dir / f"{key}.png",
                        f"block {block_index} {key}",
                        center_zero=False,
                        percentile=percentile,
                    )
                    block_report[key] = tensor_stats(tensor)
                    if save_raw:
                        torch.save(tensor, block_dir / f"{key}.pt")
            for index, tensor_batch in enumerate(state.get("amplitude_loads", []), 1):
                tensor = tensor_batch[sample]
                save_heatmap(
                    tensor,
                    block_dir / f"amplitude_slm_load_{index}.png",
                    f"block {block_index} direct amplitude load {index}",
                    center_zero=False,
                    percentile=percentile,
                )
                block_report[f"amplitude_slm_load_{index}"] = tensor_stats(tensor)
                if save_raw:
                    torch.save(tensor, block_dir / f"amplitude_slm_load_{index}.pt")
            for index, tensor_batch in enumerate(state.get("stage_detector_intensity", []), 1):
                tensor = tensor_batch[sample]
                save_heatmap(
                    tensor,
                    block_dir / f"stage_{index}_detector_intensity.png",
                    f"block {block_index} stage {index} physical detector intensity",
                    center_zero=False,
                    percentile=percentile,
                )
                block_report[f"stage_{index}_detector_intensity"] = tensor_stats(tensor)
                if save_raw:
                    torch.save(tensor, block_dir / f"stage_{index}_detector_intensity.pt")
            for index, tensor_batch in enumerate(
                state.get("stage_reloaded_amplitude", []), 1
            ):
                tensor = tensor_batch[sample]
                save_heatmap(
                    tensor,
                    block_dir / f"stage_{index}_reloaded_amplitude.png",
                    f"block {block_index} stage {index} OEO reloaded amplitude",
                    center_zero=False,
                    percentile=percentile,
                )
                block_report[f"stage_{index}_reloaded_amplitude"] = tensor_stats(
                    tensor
                )
                if save_raw:
                    torch.save(
                        tensor,
                        block_dir / f"stage_{index}_reloaded_amplitude.pt",
                    )
            for key in (
                "token_detector_roi",
                "channel_detector_roi",
                "token_detector_readout",
                "channel_detector_readout",
                "token_delta",
                "channel_delta",
                "block_output",
            ):
                if key not in state:
                    continue
                tensor = state[key][sample]
                signed = "readout" in key or "delta" in key or key == "block_output"
                save_heatmap(
                    tensor,
                    block_dir / f"{key}.png",
                    f"block {block_index} {key}",
                    center_zero=signed,
                    percentile=percentile,
                )
                block_report[key] = tensor_stats(tensor)
                if save_raw:
                    torch.save(tensor, block_dir / f"{key}.pt")
            metadata["blocks"][str(block_index)] = block_report
        write_json(directory / "metadata.json", metadata)
    return count


def save_phase_overview(model, path: Path) -> None:
    plt = _pyplot()
    blocks = list(model.blocks)
    columns = 6
    figure, axes = plt.subplots(
        len(blocks),
        columns,
        figsize=(3.5 * columns, 3.25 * len(blocks)),
        constrained_layout=True,
    )
    if len(blocks) == 1:
        axes = axes[None, :]
    for block_index, block in enumerate(blocks):
        geometry = block.core.geometry
        for stage, plane in enumerate(block.core.phase_planes):
            mosaic = torch.full(
                (geometry.active_size, geometry.active_size), float("nan")
            )
            active_start = geometry.active_start
            for expert_index, (aperture, expert) in enumerate(
                zip(geometry.expert_apertures, plane.experts)
            ):
                y0 = aperture.y0 - active_start
                x0 = aperture.x0 - active_start
                mosaic[
                    y0 : y0 + geometry.expert_size,
                    x0 : x0 + geometry.expert_size,
                ] = expert.phase().detach().cpu() % (2 * math.pi)
            axis = axes[block_index, stage]
            image = axis.imshow(
                mosaic.numpy(), cmap="twilight", vmin=0, vmax=2 * math.pi
            )
            axis.set_title(f"block {block_index} expert stage {stage + 1}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(image, ax=axis, fraction=0.046, label="phase (rad)")
        global_phase = (
            block.core.global_phase.phase.phase().detach().cpu() % (2 * math.pi)
        )
        axis = axes[block_index, 5]
        image = axis.imshow(
            global_phase.numpy(), cmap="twilight", vmin=0, vmax=2 * math.pi
        )
        axis.set_title(f"block {block_index} shared global phase")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.046, label="phase (rad)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_router_charts(router_report: dict, path: Path, title: str) -> None:
    plt = _pyplot()
    values = torch.tensor(router_report["selection_rate"]).float()
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = axis.imshow(values.numpy(), cmap="magma", vmin=0, vmax=max(1e-6, float(values.max())))
    axis.set_xlabel("expert index")
    axis.set_ylabel("Mixer block index")
    axis.set_xticks(range(values.shape[1]))
    axis.set_yticks(range(values.shape[0]))
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="selection rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history_path: Path, output_path: Path) -> None:
    if not history_path.is_file():
        return
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    plt = _pyplot()
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key in ("train_loss_total", "train_loss_feature", "train_loss_kd", "train_loss_ce"):
        if key in rows[0]:
            axes[0, 0].plot(epochs, [float(row[key]) for row in rows], label=key.removeprefix("train_"))
    axes[0, 0].set_title("Training losses")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    for key in ("train_top1_accuracy", "validation_top1_accuracy", "validation_top5_accuracy"):
        if key in rows[0]:
            axes[0, 1].plot(epochs, [float(row[key]) for row in rows], label=key)
    axes[0, 1].set_title("ImageNet accuracy")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].legend()
    for key in ("train_clip_cosine", "validation_clip_cosine"):
        if key in rows[0]:
            axes[1, 0].plot(epochs, [float(row[key]) for row in rows], label=key)
    axes[1, 0].set_title("CLIP embedding alignment")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("cosine")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [float(row["learning_rate"]) for row in rows])
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("lr")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
