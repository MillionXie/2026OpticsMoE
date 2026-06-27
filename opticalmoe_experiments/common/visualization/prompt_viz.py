import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lightfield_viz import save_image
from .mask_viz import save_phase_image


PathLike = Union[str, Path]

TASK_COLORS = {
    "mnist": "#0072B2",
    "fashionmnist": "#D55E00",
    "emnist_letters": "#009E73",
    "kmnist": "#CC79A7",
    "shape": "#0072B2",
    "scale": "#D55E00",
    "x_position_4bin": "#009E73",
    "y_position_4bin": "#CC79A7",
    "usps": "#E69F00",
}


def _as_2d_mask(intermediates: Mapping) -> Optional[torch.Tensor]:
    mask = intermediates.get("prompt_aperture_mask")
    if mask is None:
        return None
    return torch.as_tensor(mask).detach().cpu().bool().squeeze()


def _masked_amplitude(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu()
    if mask is None:
        return value
    return value * mask.to(value.dtype)


def save_prompt_maps(intermediates: dict, out_dir: PathLike, expert_labels=None) -> None:
    out = Path(out_dir)
    mask = _as_2d_mask(intermediates)
    if "prompt_router_amplitude" in intermediates:
        save_image(
            _masked_amplitude(intermediates["prompt_router_amplitude"], mask),
            out / "prompt_router_amplitude.png",
            "Router amplitude",
            cmap="inferno",
            log_intensity=False,
        )
    if "prompt_router_phase" in intermediates:
        save_phase_image(
            intermediates["prompt_router_phase"],
            out / "prompt_router_phase.png",
            "Router phase",
            aperture_mask=mask,
        )
    if "prompt_total_amplitude" in intermediates:
        save_image(
            _masked_amplitude(intermediates["prompt_total_amplitude"], mask),
            out / "prompt_total_amplitude.png",
            "Total prompt amplitude",
            cmap="inferno",
            log_intensity=False,
        )
    if "prompt_total_phase" in intermediates:
        save_phase_image(
            intermediates["prompt_total_phase"],
            out / "prompt_total_phase.png",
            "Total prompt phase",
            aperture_mask=mask,
        )
    if mask is not None:
        save_image(
            mask.float(),
            out / "prompt_aperture_region_on_canvas.png",
            "Prompt aperture on canvas",
            cmap="gray",
            log_intensity=False,
        )
    if "prompt_amplitudes" in intermediates:
        labels = expert_labels or _default_expert_labels(len(intermediates["prompt_amplitudes"]))
        _bar(intermediates["prompt_amplitudes"], labels, out / "prompt_amplitude_bar.png", "Prompt amplitudes")
    if "normalized_prompt_powers" in intermediates:
        labels = expert_labels or _default_expert_labels(len(intermediates["normalized_prompt_powers"]))
        _bar(
            intermediates["normalized_prompt_powers"],
            labels,
            out / "normalized_prompt_power_bar.png",
            "Normalized prompt power",
            ylim=(0.0, 1.0),
        )


def _default_expert_labels(count: int):
    grid_dim = int(round(math.sqrt(int(count))))
    if grid_dim * grid_dim == int(count):
        return [f"E{index // grid_dim}{index % grid_dim}" for index in range(int(count))]
    return [f"E{index:02d}" for index in range(int(count))]


def _bar(values: torch.Tensor, labels, path: Path, title: str, ylim=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = torch.as_tensor(values).detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(values)), values, color="#0072B2")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#b0b0b0", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_task_expert_weights_grouped(
    task_weights: Mapping[str, torch.Tensor],
    path: PathLike,
    expert_labels: Optional[Sequence[str]] = None,
    task_order: Optional[Sequence[str]] = None,
) -> bool:
    """Save task-specific normalized prompt powers as one grouped bar chart."""
    tasks = [task for task in (task_order or task_weights.keys()) if task in task_weights]
    if not tasks:
        return False
    vectors = [torch.as_tensor(task_weights[task]).detach().cpu().float().reshape(-1) for task in tasks]
    num_experts = int(vectors[0].numel())
    if num_experts == 0 or any(int(vector.numel()) != num_experts for vector in vectors):
        raise ValueError("All task prompt-power vectors must have the same non-zero expert count.")
    labels = list(expert_labels or _default_expert_labels(num_experts))
    if len(labels) != num_experts:
        raise ValueError("expert_labels length must match the prompt-power vector length.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(num_experts, dtype=np.float32)
    width = min(0.8 / max(len(tasks), 1), 0.24)
    fig_width = max(7.0, 0.82 * num_experts + 0.62 * len(tasks))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    fallback = plt.get_cmap("tab10")
    for index, (task, vector) in enumerate(zip(tasks, vectors)):
        offset = (index - (len(tasks) - 1) / 2.0) * width
        color = TASK_COLORS.get(str(task).lower(), fallback(index % 10))
        ax.bar(x + offset, vector.numpy(), width=width, label=str(task), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Normalized prompt power")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#b0b0b0", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(len(tasks), 4), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def save_task_expert_weights_from_model(
    model,
    path: PathLike,
    task_names: Optional[Sequence[str]] = None,
) -> bool:
    prompt_bank = getattr(model, "prompt_bank", None)
    if prompt_bank is None or not hasattr(prompt_bank, "normalized_powers"):
        return False
    names = list(task_names or getattr(prompt_bank, "task_names", []) or getattr(model, "task_names", []))
    weights = {name: prompt_bank.normalized_powers(name) for name in names}
    labels = None
    layout = getattr(model, "layout", None)
    if layout is not None and hasattr(layout, "expert_apertures"):
        labels = [aperture.name for aperture in layout.expert_apertures]
    return save_task_expert_weights_grouped(weights, path, expert_labels=labels, task_order=names)
