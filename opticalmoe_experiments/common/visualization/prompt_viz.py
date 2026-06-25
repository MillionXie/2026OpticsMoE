from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .lightfield_viz import save_image


PathLike = Union[str, Path]


def save_prompt_maps(intermediates: dict, out_dir: PathLike, expert_labels=None) -> None:
    out = Path(out_dir)
    if "prompt_router_amplitude" in intermediates:
        save_image(intermediates["prompt_router_amplitude"], out / "prompt_router_amplitude.png", "router amplitude", cmap="viridis", log_intensity=False)
    if "prompt_router_phase" in intermediates:
        save_image(intermediates["prompt_router_phase"], out / "prompt_router_phase.png", "router phase", cmap="hsv", log_intensity=False)
    if "prompt_total_amplitude" in intermediates:
        save_image(intermediates["prompt_total_amplitude"], out / "prompt_total_amplitude.png", "total amplitude", cmap="viridis", log_intensity=False)
    if "prompt_total_phase" in intermediates:
        save_image(intermediates["prompt_total_phase"], out / "prompt_total_phase.png", "total phase", cmap="hsv", log_intensity=False)
    if "prompt_aperture_mask" in intermediates:
        save_image(intermediates["prompt_aperture_mask"], out / "prompt_aperture_region_on_canvas.png", "prompt aperture on canvas", cmap="gray", log_intensity=False)
    if "prompt_amplitudes" in intermediates:
        labels = expert_labels or [f"E{i}" for i in range(len(intermediates["prompt_amplitudes"]))]
        _bar(intermediates["prompt_amplitudes"], labels, out / "prompt_amplitude_bar.png", "prompt amplitudes")
    if "normalized_prompt_powers" in intermediates:
        labels = expert_labels or [f"E{i}" for i in range(len(intermediates["normalized_prompt_powers"]))]
        _bar(intermediates["normalized_prompt_powers"], labels, out / "normalized_prompt_power_bar.png", "normalized prompt power")


def _bar(values: torch.Tensor, labels, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = values.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
