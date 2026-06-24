import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import save_json, write_rows


def _as_image(field):
    if isinstance(field, torch.Tensor):
        array = field.detach().cpu()
        if array.ndim == 4:
            array = array[0, 0]
        elif array.ndim == 3:
            array = array[0]
        if torch.is_complex(array):
            array = torch.abs(array).square()
        array = array.float()
        if array.max() > 0:
            array = array / array.max()
        return array.numpy()
    return field


def save_intensity(field, path, title=None, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _as_image(field)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="inferno")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_phase(phase, path, title=None, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = phase.detach().cpu()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(image, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_phase_masks(model, out_dir, dpi=150):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = model.phase_stack_wrapped()
    for idx, phase in enumerate(phases, start=1):
        save_phase(phase, out_dir / f"phase_layer_{idx}.png", f"phase layer {idx}", dpi=dpi)
    fig, axes = plt.subplots(1, len(phases), figsize=(3.2 * len(phases), 3.2))
    if len(phases) == 1:
        axes = [axes]
    for ax, phase, idx in zip(axes, phases, range(1, len(phases) + 1)):
        ax.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
        ax.set_title(f"L{idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "all_phase_layers.png", dpi=dpi)
    plt.close(fig)
    save_phase_region_diagram(model, out_dir / "centered_phase_mask_region.png", dpi=dpi)


def save_phase_region_diagram(model, path, dpi=150):
    canvas = torch.zeros(model.canvas_size, model.canvas_size)
    y0, y1, x0, x1 = model.phase_mask_region()
    canvas[y0:y1, x0:x1] = 1.0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"trainable phase region y[{y0}:{y1}], x[{x0}:{x1}]")
    ax.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir, epoch_name, class_names, enabled=True, dpi=150):
    if not enabled:
        return
    model.eval()
    images, targets = batch
    logits, intermediates = model(images, return_intermediates=True)
    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)

    light_dir = Path(run_dir) / "figures" / "light_fields" / epoch_name / "sample_000"
    save_intensity(intermediates["input_256"], light_dir / "00_input_256.png", "input 256", dpi=dpi)
    save_intensity(intermediates["canvas_input_400"], light_dir / "01_canvas_input_400.png", "canvas input 400", dpi=dpi)
    save_intensity(intermediates["after_input_to_layer"], light_dir / "02_after_input_to_layer.png", "after input-to-layer propagation", dpi=dpi)
    file_index = 3
    for idx in range(1, model.num_layers + 1):
        save_intensity(
            intermediates[f"after_phase_modulation_{idx}"],
            light_dir / f"{file_index:02d}_after_phase_modulation_{idx}.png",
            f"after phase modulation {idx}",
            dpi=dpi,
        )
        file_index += 1
        if idx < model.num_layers:
            save_intensity(
                intermediates[f"after_propagation_{idx}"],
                light_dir / f"{file_index:02d}_after_propagation_{idx}.png",
                f"after propagation {idx}",
                dpi=dpi,
            )
            file_index += 1
    save_intensity(intermediates["detector_field"], light_dir / f"{file_index:02d}_detector_plane.png", "detector plane", dpi=dpi)
    save_overview(intermediates, light_dir / "overview.png", model.num_layers, dpi=dpi)

    save_phase_masks(model, Path(run_dir) / "figures" / "phase_masks" / epoch_name, dpi=dpi)

    detector_dir = Path(run_dir) / "figures" / "detector_outputs" / epoch_name
    detector_dir.mkdir(parents=True, exist_ok=True)
    save_intensity(intermediates["detector_field"], detector_dir / "detector_plane_sample_000.png", "detector plane", dpi=dpi)
    save_bar(intermediates["detector_energies"][0], detector_dir / "detector_energy_bar_sample_000.png", "detector energy sample 000", dpi=dpi)
    save_bar(intermediates["detector_energies"].mean(dim=0), detector_dir / "detector_energy_mean_bar.png", "mean detector energy", dpi=dpi)

    sample_dir = Path(run_dir) / "figures" / "samples" / epoch_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(min(len(images), 8)):
        rows.append(
            {
                "sample_index": idx,
                "true": int(targets[idx].item()),
                "pred": int(preds[idx].item()),
                "confidence": float(probs[idx, preds[idx]].item()),
            }
        )
    save_json(rows, sample_dir / "sample_predictions.json")
    save_sample_predictions(images, targets, preds, probs, sample_dir / "sample_predictions.png", class_names, dpi=dpi)


def save_overview(intermediates, path, num_layers, dpi=150):
    keys = ["input_256", "canvas_input_400", "after_input_to_layer"]
    for idx in range(1, num_layers + 1):
        keys.append(f"after_phase_modulation_{idx}")
        if idx < num_layers:
            keys.append(f"after_propagation_{idx}")
    keys.append("detector_field")
    cols = 4
    rows = int(math.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = axes.reshape(-1)
    for ax, key in zip(axes, keys):
        ax.imshow(_as_image(intermediates[key]), cmap="inferno")
        ax.set_title(key)
        ax.axis("off")
    for ax in axes[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_bar(values, path, title, dpi=150):
    values = values.detach().cpu().float()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(values)), values)
    ax.set_xlabel("class")
    ax.set_ylabel("energy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_sample_predictions(images, targets, preds, probs, path, class_names, dpi=150):
    count = min(len(images), 8)
    fig, axes = plt.subplots(1, count, figsize=(2.4 * count, 2.8))
    if count == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx, 0].detach().cpu(), cmap="gray")
        true = int(targets[idx].item())
        pred = int(preds[idx].item())
        conf = float(probs[idx, pred].item())
        ax.set_title(f"T:{class_names[true]}\nP:{class_names[pred]} {conf:.2f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_training_curves(rows, path, dpi=150):
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in rows], label="test")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="train")
    axes[1].plot(epochs, [row["test_acc"] for row in rows], label="test")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def confusion_matrix(preds, targets, num_classes=10):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, pred in zip(targets.view(-1).cpu(), preds.view(-1).cpu()):
        matrix[int(true), int(pred)] += 1
    return matrix


def save_confusion_matrix(matrix, path, class_names=None, dpi=150):
    matrix = matrix.detach().cpu().float()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    if class_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def save_confusion_csv(matrix, path):
    rows = []
    for i in range(matrix.shape[0]):
        row = {"row": i}
        for j in range(matrix.shape[1]):
            row[str(j)] = int(matrix[i, j].item())
        rows.append(row)
    write_rows(path, rows)
