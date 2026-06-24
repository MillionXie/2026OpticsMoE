from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _image_from_tensor(value: torch.Tensor, log_intensity: bool = True):
    value = value.detach().cpu()
    if torch.is_complex(value):
        value = value.abs().square()
    elif value.ndim >= 2:
        value = value.float()
    if value.ndim == 3:
        value = value[0]
    if log_intensity:
        value = torch.log10(value / (value.max() + 1e-12) + 1e-8)
    return value.numpy()


PathLike = Union[str, Path]


def save_image(value: torch.Tensor, path: PathLike, title: str = "", cmap: str = "inferno", log_intensity: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(_image_from_tensor(value, log_intensity=log_intensity), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_light_fields(intermediates: dict, out_dir: PathLike, sample_index: int = 0) -> None:
    out = Path(out_dir)
    fields = [
        ("00_input_amplitude.png", "input_amplitude"),
        ("01_after_input_to_prompt.png", "after_input_to_prompt"),
        ("02_after_prompt.png", "after_prompt"),
        ("03_expert_entrance_before_aperture.png", "expert_entrance_before_aperture"),
        ("04_expert_entrance_after_aperture.png", "expert_entrance_after_aperture"),
        ("05_after_expert_layer_1.png", "after_expert_layer_1"),
        ("06_after_expert_layer_last.png", "after_expert_layer_last"),
        ("07_after_global_fc.png", "after_global_fc"),
        ("08_detector_plane.png", "detector_field"),
    ]
    if "after_each_layer" in intermediates and intermediates["after_each_layer"]:
        intermediates["after_expert_layer_last"] = intermediates["after_each_layer"][-1]
    for filename, key in fields:
        if key not in intermediates:
            continue
        value = intermediates[key]
        if isinstance(value, list):
            value = value[-1]
        if value.ndim >= 3:
            value = value[sample_index]
        save_image(value, out / filename, key)
