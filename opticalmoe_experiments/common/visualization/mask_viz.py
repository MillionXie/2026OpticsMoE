from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


PathLike = Union[str, Path]


def save_expert_phase_layers(model, out_dir: PathLike) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "expert_layers"):
        return
    phases = []
    for layer in model.expert_layers:
        if hasattr(layer, "get_phase_wrapped"):
            phases.append(layer.get_phase_wrapped().detach().cpu())
    if not phases:
        return
    num_layers = len(phases)
    num_experts = phases[0].shape[0]
    grid_dim = int(round(num_experts ** 0.5))
    fig, axes = plt.subplots(num_layers, num_experts, figsize=(1.8 * num_experts, 1.8 * num_layers))
    if num_layers == 1:
        axes = axes[None, :]
    for layer_idx, layer_phase in enumerate(phases):
        for expert_idx in range(num_experts):
            ax = axes[layer_idx, expert_idx]
            ax.imshow(layer_phase[expert_idx], cmap="hsv", vmin=0, vmax=2 * torch.pi)
            row, col = divmod(expert_idx, grid_dim)
            ax.set_title(f"L{layer_idx+1} E{row}{col}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "expert_phase_layers.png", dpi=140)
    plt.close(fig)
    if hasattr(model, "global_fc"):
        phase = model.global_fc.get_phase_wrapped().detach().cpu()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(phase, cmap="hsv", vmin=0, vmax=2 * torch.pi)
        ax.set_title("global FC phase")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "global_fc_phase.png", dpi=140)
        plt.close(fig)
