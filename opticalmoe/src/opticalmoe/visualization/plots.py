from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _intensity(field: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(field):
        return torch.abs(field) ** 2
    return field.float()


def _phase(field: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(field):
        return torch.angle(field)
    return torch.zeros_like(field.float())


def save_detector_layout(detector, path: str) -> None:
    masks = detector.get_masks().detach().cpu()
    layout = masks.sum(dim=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(layout.numpy(), cmap="magma")
    plt.title("Detector layout")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_phase_layers(model, path: str, title: str = "Phase layers") -> None:
    num_layers = len(model.phase_layers)
    cols = min(num_layers, 5)
    rows = int(np.ceil(num_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, layer in enumerate(model.phase_layers):
        phase = _to_numpy(layer.get_phase_wrapped())
        axes[idx].imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
        axes[idx].set_title(f"Layer {idx + 1}")
        axes[idx].axis("off")

    for idx in range(num_layers, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_sample_outputs(model, batch, path: str, device: torch.device, num_samples: int = 4) -> Dict[str, torch.Tensor]:
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    logits, intermediates = model(images, return_intermediates=True)
    pred = torch.argmax(logits, dim=1)
    intensity = intermediates["detector_intensity"]

    count = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(count, 2, figsize=(6, 3 * count))
    axes = np.array(axes).reshape(count, 2)

    for i in range(count):
        axes[i, 0].imshow(_to_numpy(images[i, 0]), cmap="gray")
        axes[i, 0].set_title(f"Input y={int(labels[i])}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(_to_numpy(intensity[i]), cmap="inferno")
        axes[i, 1].set_title(f"Detector pred={int(pred[i])}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return intermediates


def save_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int, path: str) -> None:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_detector_energy_bar(energies: torch.Tensor, path: str) -> None:
    values = _to_numpy(energies[0])
    plt.figure(figsize=(7, 3))
    plt.bar(np.arange(len(values)), values)
    plt.xlabel("Class detector")
    plt.ylabel("Energy")
    plt.title("Detector energies")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_light_field_debug(
    intermediates: Dict[str, torch.Tensor],
    path: str,
    detector_masks: Optional[torch.Tensor] = None,
    num_samples: int = 2,
    save_npz: bool = False,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(intermediates.keys())
    count = min(num_samples, next(iter(intermediates.values())).shape[0])

    for sample_idx in range(count):
        rows = len(keys)
        fig, axes = plt.subplots(rows, 2, figsize=(8, 2.4 * rows))
        axes = np.array(axes).reshape(rows, 2)
        npz_payload = {}

        for row, key in enumerate(keys):
            value = intermediates[key][sample_idx]
            if value.ndim == 1:
                axes[row, 0].bar(np.arange(value.numel()), _to_numpy(value))
                axes[row, 0].set_title(key)
                axes[row, 1].axis("off")
                npz_payload[key] = _to_numpy(value)
                continue

            intensity = _intensity(value)
            phase = _phase(value)
            intensity_np = _to_numpy(intensity)
            phase_np = _to_numpy(phase)
            npz_payload[f"{key}_intensity"] = intensity_np
            npz_payload[f"{key}_phase"] = phase_np

            axes[row, 0].imshow(intensity_np, cmap="inferno")
            if key == "detector_intensity" and detector_masks is not None:
                overlay = detector_masks.detach().cpu().sum(dim=0).numpy()
                axes[row, 0].contour(overlay, levels=[0.5], colors="cyan", linewidths=0.5)
            axes[row, 0].set_title(f"{key} intensity")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(phase_np, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            axes[row, 1].set_title(f"{key} phase")
            axes[row, 1].axis("off")

        plt.tight_layout()
        out_path = Path(path)
        png_path = out_path.with_name(f"{out_path.stem}_sample_{sample_idx}.png")
        plt.savefig(png_path, dpi=120)
        plt.close(fig)

        if save_npz:
            np.savez_compressed(out_path.with_name(f"{out_path.stem}_sample_{sample_idx}.npz"), **npz_payload)
