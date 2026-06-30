import math
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PathLike = Union[str, Path]
PHASE_CMAP = "twilight"
PHASE_VMIN = 0.0
PHASE_VMAX = 2.0 * math.pi
PHASE_TICKS = (0.0, math.pi, 2.0 * math.pi)
PHASE_TICK_LABELS = ("0", r"$\pi$", r"$2\pi$")
PADDING_COLOR = "#e6e6e6"


def phase_colormap():
    cmap = plt.get_cmap(PHASE_CMAP).copy()
    cmap.set_bad(PADDING_COLOR)
    return cmap


def _phase_array(phase: torch.Tensor, aperture_mask: Optional[torch.Tensor] = None) -> np.ma.MaskedArray:
    values = torch.as_tensor(phase).detach().cpu().float().squeeze().numpy()
    invalid = ~np.isfinite(values)
    if aperture_mask is not None:
        mask = torch.as_tensor(aperture_mask).detach().cpu().bool().squeeze().numpy()
        if mask.shape != values.shape:
            raise ValueError(f"phase mask shape {mask.shape} does not match phase shape {values.shape}")
        invalid = np.logical_or(invalid, ~mask)
    return np.ma.array(values, mask=invalid)


def _add_phase_colorbar(fig, image, axes) -> None:
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Phase (rad)")
    colorbar.set_ticks(PHASE_TICKS)
    colorbar.set_ticklabels(PHASE_TICK_LABELS)


def save_phase_image(
    phase: torch.Tensor,
    path: PathLike,
    title: str,
    aperture_mask: Optional[torch.Tensor] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = ax.imshow(
        _phase_array(phase, aperture_mask),
        cmap=phase_colormap(),
        vmin=PHASE_VMIN,
        vmax=PHASE_VMAX,
    )
    ax.set_title(title)
    ax.axis("off")
    _add_phase_colorbar(fig, image, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_global_fc(model, out: Path) -> None:
    if not hasattr(model, "global_fc"):
        return
    phase = model.global_fc.get_phase_wrapped().detach().cpu()
    save_phase_image(phase, out / "global_fc_phase_window.png", "Global FC phase window")
    save_phase_image(phase, out / "global_fc_phase.png", "Global FC phase window")
    if hasattr(model.global_fc, "get_phase_canvas_wrapped"):
        canvas = model.global_fc.get_phase_canvas_wrapped().detach().cpu()
        region_mask = None
        if hasattr(model.global_fc, "phase_region"):
            y0, y1, x0, x1 = model.global_fc.phase_region()
            region_mask = torch.zeros_like(canvas, dtype=torch.bool)
            region_mask[y0:y1, x0:x1] = True
        save_phase_image(
            canvas,
            out / "global_fc_phase_canvas.png",
            "Global FC phase: center window trainable, outside transparent",
            aperture_mask=region_mask,
        )
    if hasattr(model.global_fc, "phase_region"):
        region = model.global_fc.phase_region()
        mask = torch.zeros(getattr(model.global_fc, "canvas_shape", phase.shape), dtype=torch.float32)
        mask[region[0]:region[1], region[2]:region[3]] = 1.0
        path = out / "global_fc_phase_region_on_canvas.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Global FC trainable region y[{region[0]}:{region[1]}], x[{region[2]}:{region[3]}]")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def save_expert_phase_layers(model, out_dir: PathLike) -> None:
    save_phase_masks(model, out_dir)


def save_phase_masks(model, out_dir: PathLike) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "expert_layers"):
        _save_moe_phase_masks(model, out)
        _save_global_fc(model, out)
        return
    if hasattr(model, "layers"):
        _save_d2nn_phase_masks(model, out)
        _save_global_fc(model, out)


def _expert_labels(model, num_experts: int):
    layout = getattr(model, "layout", None)
    apertures = getattr(layout, "expert_apertures", None)
    if apertures and len(apertures) == num_experts:
        return [aperture.name for aperture in apertures]
    grid_dim = int(round(num_experts ** 0.5))
    if grid_dim * grid_dim == num_experts:
        return [f"E{index // grid_dim}{index % grid_dim}" for index in range(num_experts)]
    return [f"E{index:02d}" for index in range(num_experts)]


def _save_moe_phase_masks(model, out: Path) -> None:
    phases = []
    for layer in model.expert_layers:
        if hasattr(layer, "get_phase_wrapped"):
            phases.append(layer.get_phase_wrapped().detach().cpu())
    if not phases:
        return
    num_layers = len(phases)
    num_experts = phases[0].shape[0]
    labels = _expert_labels(model, num_experts)
    fig, axes = plt.subplots(
        num_layers,
        num_experts,
        figsize=(max(7.0, 1.55 * num_experts), max(2.2, 1.65 * num_layers)),
        squeeze=False,
    )
    image = None
    for layer_idx, layer_phase in enumerate(phases):
        for expert_idx in range(num_experts):
            ax = axes[layer_idx, expert_idx]
            image = ax.imshow(
                _phase_array(layer_phase[expert_idx]),
                cmap=phase_colormap(),
                vmin=PHASE_VMIN,
                vmax=PHASE_VMAX,
            )
            ax.set_title(f"L{layer_idx + 1} {labels[expert_idx]}", fontsize=8)
            ax.axis("off")
    if image is not None:
        _add_phase_colorbar(fig, image, axes.ravel().tolist())
    profile = getattr(getattr(model, "layout", None), "geometry_profile", "")
    if profile:
        fig.suptitle(f"Expert phase masks ({profile})", fontsize=10)
    fig.subplots_adjust(left=0.015, right=0.93, bottom=0.02, top=0.90 if profile else 0.94, wspace=0.06, hspace=0.22)
    fig.savefig(out / "expert_phase_layers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_d2nn_phase_masks(model, out: Path) -> None:
    phases = []
    for idx, layer in enumerate(model.layers, start=1):
        if hasattr(layer, "get_phase_wrapped"):
            phase = layer.get_phase_wrapped().detach().cpu()
            phases.append(phase)
            save_phase_image(phase, out / f"d2nn_phase_layer_{idx}.png", f"D2NN local phase layer {idx}")
    if not phases:
        return
    cols = len(phases)
    fig, axes = plt.subplots(1, cols, figsize=(max(4.0, 2.15 * cols), 2.5), squeeze=False)
    image = None
    for idx, (ax, phase) in enumerate(zip(axes[0], phases), start=1):
        image = ax.imshow(_phase_array(phase), cmap=phase_colormap(), vmin=PHASE_VMIN, vmax=PHASE_VMAX)
        ax.set_title(f"L{idx}")
        ax.axis("off")
    if image is not None:
        _add_phase_colorbar(fig, image, axes.ravel().tolist())
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.04, top=0.88, wspace=0.08)
    fig.savefig(out / "d2nn_all_phase_layers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
