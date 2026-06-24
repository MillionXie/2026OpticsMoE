from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PathLike = Union[str, Path]


def _save_phase_image(phase: torch.Tensor, path: PathLike, title: str) -> None:
    phase = phase.detach().cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(phase, cmap="hsv", vmin=0, vmax=2 * torch.pi)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_global_fc(model, out: Path) -> None:
    if not hasattr(model, "global_fc"):
        return
    phase = model.global_fc.get_phase_wrapped().detach().cpu()
    _save_phase_image(phase, out / "global_fc_phase.png", "global FC phase")


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
        return


def _save_moe_phase_masks(model, out: Path) -> None:
    phases = []
    for layer in model.expert_layers:
        if hasattr(layer, "get_phase_wrapped"):
            phases.append(layer.get_phase_wrapped().detach().cpu())
    if not phases:
        return
    num_layers = len(phases)
    num_experts = phases[0].shape[0]
    grid_dim = int(round(num_experts ** 0.5))
    fig, axes = plt.subplots(num_layers * grid_dim, grid_dim, figsize=(1.8 * grid_dim, 1.8 * num_layers * grid_dim))
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(num_layers * grid_dim, grid_dim)
    for layer_idx, layer_phase in enumerate(phases):
        for expert_idx in range(num_experts):
            row, col = divmod(expert_idx, grid_dim)
            ax = axes[layer_idx * grid_dim + row, col]
            ax.imshow(layer_phase[expert_idx], cmap="hsv", vmin=0, vmax=2 * torch.pi)
            ax.set_title(f"L{layer_idx+1} E{row}{col}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "expert_phase_layers.png", dpi=140)
    plt.close(fig)


def _save_d2nn_phase_masks(model, out: Path) -> None:
    phases = []
    for idx, layer in enumerate(model.layers, start=1):
        if hasattr(layer, "get_phase_wrapped"):
            phase = layer.get_phase_wrapped().detach().cpu()
            phases.append(phase)
            _save_phase_image(phase, out / f"d2nn_phase_layer_{idx}.png", f"D2NN local phase layer {idx}")
    if not phases:
        return
    cols = len(phases)
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.2))
    if cols == 1:
        axes = [axes]
    for idx, (ax, phase) in enumerate(zip(axes, phases), start=1):
        ax.imshow(phase, cmap="hsv", vmin=0, vmax=2 * torch.pi)
        ax.set_title(f"L{idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "d2nn_all_phase_layers.png", dpi=140)
    plt.close(fig)
