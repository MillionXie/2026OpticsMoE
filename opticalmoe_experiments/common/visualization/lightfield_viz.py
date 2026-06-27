import math
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _image_from_tensor(value: torch.Tensor, log_intensity: bool = True):
    value = torch.as_tensor(value).detach().cpu()
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
    image = ax.imshow(_image_from_tensor(value, log_intensity=log_intensity), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _sample(value, sample_index: int):
    if isinstance(value, (list, tuple)):
        value = value[-1]
    value = torch.as_tensor(value)
    if value.ndim >= 3:
        value = value[sample_index]
    return value


def _light_field_entries(intermediates: dict, sample_index: int) -> List[Tuple[str, str, torch.Tensor]]:
    entries: List[Tuple[str, str, torch.Tensor]] = []

    def add(key: str, title: str) -> None:
        if key in intermediates:
            entries.append((key, title, _sample(intermediates[key], sample_index)))

    add("input_amplitude", "Input amplitude")
    add("after_input_to_prompt", "After input-to-prompt propagation")
    add("after_prompt", "After prompt")
    add("expert_entrance_before_aperture", "Expert entrance before aperture")
    add("expert_entrance_after_aperture", "Expert entrance after aperture")

    layer_fields = intermediates.get("after_each_layer") or []
    is_expert_model = "expert_entrance_before_aperture" in intermediates
    layer_label = "expert layer" if is_expert_model else "D2NN layer"
    layer_key = "after_expert_layer" if is_expert_model else "after_d2nn_layer"
    for layer_index, field in enumerate(layer_fields, start=1):
        entries.append(
            (
                f"{layer_key}_{layer_index}",
                f"After {layer_label} {layer_index}",
                _sample(field, sample_index),
            )
        )

    add("after_layer5_to_fc", "After propagation to global FC")
    add("after_global_fc", "After global FC phase")
    add("detector_field", "Detector plane")
    return entries


def _entry_filename(index: int, key: str) -> str:
    if key.startswith("after_expert_layer_"):
        layer_index = int(key.rsplit("_", 1)[1])
        return f"{index:02d}_after_expert_layer_{layer_index}.png"
    if key.startswith("after_d2nn_layer_"):
        layer_index = int(key.rsplit("_", 1)[1])
        return f"{index:02d}_after_d2nn_layer_{layer_index}.png"
    names = {
        "input_amplitude": "input_amplitude",
        "after_input_to_prompt": "after_input_to_prompt",
        "after_prompt": "after_prompt",
        "expert_entrance_before_aperture": "expert_entrance_before_aperture",
        "expert_entrance_after_aperture": "expert_entrance_after_aperture",
        "after_layer5_to_fc": "after_last_layer_to_fc",
        "after_global_fc": "after_global_fc",
        "detector_field": "detector_plane",
    }
    return f"{index:02d}_{names.get(key, key)}.png"


def _save_overview(entries: List[Tuple[str, str, torch.Tensor]], path: Path) -> None:
    if not entries:
        return
    columns = min(4, len(entries))
    rows = int(math.ceil(len(entries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.9 * rows), squeeze=False)
    for ax, (_, title, value) in zip(axes.ravel(), entries):
        ax.imshow(_image_from_tensor(value, log_intensity=True), cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")
    fig.suptitle("Light-field propagation overview", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_light_fields(intermediates: dict, out_dir: PathLike, sample_index: int = 0) -> None:
    out = Path(out_dir)
    entries = _light_field_entries(intermediates, sample_index)
    for index, (key, title, value) in enumerate(entries):
        save_image(value, out / _entry_filename(index, key), title)
    _save_overview(entries, out / "overview.png")
