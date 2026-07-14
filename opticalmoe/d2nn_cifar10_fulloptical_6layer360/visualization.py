import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def _save(fig, path, dpi=150):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _array(value, intensity=False):
    value = value.detach().cpu()
    while value.ndim > 2: value = value[0]
    if torch.is_complex(value): value = value.abs().square() if intensity else value.abs()
    return value.float().numpy()


def detector_bounds(detector):
    result = []
    for mask in detector.masks.detach().cpu():
        points = mask.nonzero()
        y0, x0 = points.min(0).values.tolist(); y1, x1 = (points.max(0).values + 1).tolist()
        result.append((int(y0), int(y1), int(x0), int(x1)))
    return result


def save_map(value, path, title, cmap="inferno", vmin=None, vmax=None, colorbar_label="intensity (a.u.)"):
    fig, ax = plt.subplots(figsize=(5.4, 4.7), constrained_layout=True)
    image = ax.imshow(_array(value, intensity=True), cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title); ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04); colorbar.set_label(colorbar_label)
    _save(fig, path)


def save_phase_overview(model, path):
    phases = model.phase_stack_wrapped().detach().cpu()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, (axis, phase) in enumerate(zip(axes.flat, phases), 1):
        image = axis.imshow(phase, cmap="twilight", vmin=0, vmax=2 * math.pi, origin="upper")
        axis.set_title(f"Phase layer {index}"); axis.set_xlabel("x pixel"); axis.set_ylabel("y pixel")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04); colorbar.set_label("phase (rad)")
    _save(fig, path)


def save_detector_summary(image, intensity, energies, detector, class_names, path, true_name, pred_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    source = axes[0].imshow(_array(image), cmap="gray", vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Input amplitude: 300x300, padded to 360x360")
    axes[0].set_xlabel("x pixel"); axes[0].set_ylabel("y pixel")
    fig.colorbar(source, ax=axes[0], fraction=0.046, pad=0.04).set_label("amplitude")
    detector_image = axes[1].imshow(_array(intensity), cmap="inferno", origin="upper")
    for index, (y0, y1, x0, x1) in enumerate(detector_bounds(detector)):
        detector_image.axes.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1.4))
        detector_image.axes.text(x0+2, y0+10, class_names[index], color="cyan", fontsize=7)
    axes[1].set_title("Detector intensity and integration regions")
    axes[1].set_xlabel("x pixel"); axes[1].set_ylabel("y pixel")
    fig.colorbar(detector_image, ax=axes[1], fraction=0.046, pad=0.04).set_label("intensity (a.u.)")
    values = energies.detach().cpu().float()
    axes[2].bar(range(len(values)), values)
    axes[2].set_xticks(range(len(values)), class_names, rotation=35, ha="right")
    axes[2].set_xlabel("class detector"); axes[2].set_ylabel("normalized region energy")
    axes[2].set_title(f"true={true_name}, predicted={pred_name}")
    axes[2].grid(axis="y", alpha=0.25)
    _save(fig, path)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir, tag, class_names, enabled=True):
    if not enabled: return
    model.eval(); images, targets = batch
    logits, items = model(images, return_intermediates=True)
    root = Path(run_dir) / "figures" / "epoch_artifacts" / tag
    save_phase_overview(model, root / "phase_masks_overview.png")
    save_map(items["input_canvas"][0], root / "input_amplitude.png", "Input amplitude", cmap="gray", vmin=0, vmax=1, colorbar_label="amplitude")
    if items.get("optoelectronic_interlayers_enabled",False):
        conversion_root=root/"optoelectronic_interlayers"
        for index,(detected,normalized,amplitude) in enumerate(zip(items["interlayer_detector_intensities"],items["interlayer_layer_normalized"],items["interlayer_reloaded_amplitudes"]),1):
            save_map(detected[0],conversion_root/f"layer_{index:02d}_square_detector_intensity.png",f"Plane {index}: intensity after 20 cm",colorbar_label="square-law intensity (a.u.)")
            save_map(normalized[0],conversion_root/f"layer_{index:02d}_layernorm.png",f"Plane {index}: non-affine spatial LayerNorm",cmap="coolwarm",colorbar_label="normalized real value")
            save_map(amplitude[0],conversion_root/f"layer_{index:02d}_relu_amplitude_reload.png",f"Plane {index}: ReLU amplitude reload",cmap="viridis",colorbar_label="reloaded amplitude")
    else:
        for index, field in enumerate(items["after_each_layer"], 1):
            save_map(field[0], root / f"after_layer_{index}_intensity.png", f"Intensity after phase/propagation layer {index}")
    pred = int(logits[0].argmax()); target = int(targets[0])
    save_detector_summary(
        images[0], items["detector_intensity"][0], logits[0], model.detector, class_names,
        root / "detector_summary.png", class_names[target], class_names[pred],
    )


def confusion_matrix(predictions, targets, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()): matrix[target, prediction] += 1
    return matrix


def save_confusion(matrix, figure_path, csv_path, class_names):
    figure_path = Path(figure_path); figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax).set_label("samples"); _save(fig, figure_path)
    csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["true/pred", *class_names])
        for name, row in zip(class_names, matrix.tolist()): writer.writerow([name, *row])


def save_training_curves(rows, path):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in rows], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Detector-plane loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="train")
    axes[1].plot(epochs, [row["test_acc"] for row in rows], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=.25)
    _save(fig, path)
