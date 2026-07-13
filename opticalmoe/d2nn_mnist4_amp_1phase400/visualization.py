import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from utils import save_json, write_rows


def _scalar_image(field, normalize=False):
    array = field.detach().cpu() if isinstance(field, torch.Tensor) else torch.as_tensor(field)
    while array.ndim > 2:
        array = array[0]
    if torch.is_complex(array):
        array = torch.abs(array).square()
    array = array.float()
    if normalize and float(array.max().item()) > 0.0:
        array = array / array.max()
    return array.numpy()


def _amplitude_image(field, normalize=False):
    array = field.detach().cpu() if isinstance(field, torch.Tensor) else torch.as_tensor(field)
    while array.ndim > 2:
        array = array[0]
    array = torch.abs(array).float() if torch.is_complex(array) else array.float()
    if normalize and float(array.max().item()) > 0.0:
        array = array / array.max()
    return array.numpy()


def _label_image_axes(ax):
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")


def _save_figure(fig, path, dpi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_intensity(field, path, title=None, dpi=150, normalize=True):
    image = _scalar_image(field, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(image, cmap="inferno", origin="upper")
    _label_image_axes(ax)
    ax.set_title(title or "Intensity")
    label = "normalized intensity (a.u.)" if normalize else "intensity (a.u.)"
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(label)
    _save_figure(fig, path, dpi)


def save_phase(phase, path, title=None, dpi=150):
    image = phase.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(image, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi, origin="upper")
    _label_image_axes(ax)
    ax.set_title(title or "Phase mask")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("phase (rad)")
    _save_figure(fig, path, dpi)


def save_phase_overlay(input_field, phase, path, title="Input amplitude with phase-mask overlay", dpi=150):
    amplitude = _amplitude_image(input_field, normalize=True)
    phase_image = phase.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    ax.imshow(amplitude, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    overlay = ax.imshow(phase_image, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi, alpha=0.55, origin="upper")
    _label_image_axes(ax)
    ax.set_title(title)
    colorbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("phase overlay (rad)")
    _save_figure(fig, path, dpi)


def save_phase_masks(model, out_dir, input_field=None, dpi=150):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = model.phase_stack_wrapped()
    for idx, phase in enumerate(phases, start=1):
        save_phase(phase, out_dir / f"phase_layer_{idx}.png", f"Phase layer {idx}", dpi=dpi)
    fig, axes = plt.subplots(1, len(phases), figsize=(5.2 * len(phases), 4.5), constrained_layout=True)
    if len(phases) == 1:
        axes = [axes]
    for ax, phase, idx in zip(axes, phases, range(1, len(phases) + 1)):
        im = ax.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi, origin="upper")
        ax.set_title(f"Phase layer {idx}")
        _label_image_axes(ax)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("phase (rad)")
    _save_figure(fig, out_dir / "all_phase_layers.png", dpi)
    save_phase_region_diagram(model, out_dir / "trainable_phase_region.png", dpi=dpi)
    if input_field is not None:
        save_phase_overlay(input_field, phases[0], out_dir / "phase_mask_overlay.png", dpi=dpi)


def save_phase_region_diagram(model, path, dpi=150):
    canvas = torch.zeros(model.canvas_size, model.canvas_size)
    y0, y1, x0, x1 = model.phase_mask_region()
    canvas[y0:y1, x0:x1] = 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(canvas, cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_title(f"Trainable phase region: y[{y0}:{y1}], x[{x0}:{x1}]")
    _label_image_axes(ax)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("trainable mask indicator")
    _save_figure(fig, path, dpi)


def save_k_space_mask(propagator, path, dpi=150):
    mask = torch.fft.fftshift(propagator.k_space_mask.detach().cpu()).float().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(mask, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    ax.set_xlabel("shifted kx bin")
    ax.set_ylabel("shifted ky bin")
    ax.set_title(
        f"K-space angular pass mask: enabled={propagator.k_space_constraint_enabled}, "
        f"theta_max={propagator.theta_max_deg:.3f} deg"
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("pass=1, reject=0")
    _save_figure(fig, path, dpi)


def _detector_bounds(detector):
    bounds = []
    for mask in detector.masks.detach().cpu():
        indices = mask.nonzero(as_tuple=False)
        y0, x0 = indices.min(dim=0).values.tolist()
        y1, x1 = (indices.max(dim=0).values + 1).tolist()
        bounds.append((int(y0), int(y1), int(x0), int(x1)))
    return bounds


def save_detector_mask_overlay(detector, path, class_names, dpi=150):
    label_map = torch.zeros(detector.grid_size, dtype=torch.float32)
    for index, mask in enumerate(detector.masks.detach().cpu(), start=1):
        label_map[mask.bool()] = float(index)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(label_map, cmap="viridis", vmin=0.0, vmax=float(detector.num_classes), origin="upper")
    for index, (y0, y1, x0, x1) in enumerate(_detector_bounds(detector)):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.5))
        ax.text(x0 + 3, y0 + 13, str(class_names[index]), color="white", fontsize=8, weight="bold")
    ax.set_title("Ideal square detector-region mask overlay")
    _label_image_axes(ax)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("detector class index + 1")
    _save_figure(fig, path, dpi)


def save_detector_intensity_with_regions(intensity, detector, path, class_names, title, dpi=150):
    image = _scalar_image(intensity, normalize=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(image, cmap="inferno", origin="upper")
    colors = ["cyan", "lime", "deepskyblue", "magenta"]
    for index, (y0, y1, x0, x1) in enumerate(_detector_bounds(detector)):
        color = colors[index % len(colors)]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.0))
        ax.text(x0 + 3, y0 + 13, str(class_names[index]), color=color, fontsize=8, weight="bold")
    ax.set_title(title)
    _label_image_axes(ax)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("normalized detector intensity (a.u.)")
    _save_figure(fig, path, dpi)


def save_detector_sample_summary(image, intensity, energies, detector, path, class_names, true_name, pred_name, dpi=150):
    input_image = _scalar_image(image, normalize=False)
    detector_image = _scalar_image(intensity, normalize=True)
    values = energies.detach().cpu().float()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(input_image, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    axes[0].set_title("Input amplitude")
    _label_image_axes(axes[0])
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("amplitude")
    im1 = axes[1].imshow(detector_image, cmap="inferno", origin="upper")
    for index, (y0, y1, x0, x1) in enumerate(_detector_bounds(detector)):
        axes[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1.5))
        axes[1].text(x0 + 3, y0 + 13, str(class_names[index]), color="cyan", fontsize=8, weight="bold")
    axes[1].set_title("Detector intensity and square integration regions")
    _label_image_axes(axes[1])
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("normalized intensity (a.u.)")
    axes[2].bar(range(len(values)), values, color="tab:blue")
    axes[2].set_xticks(range(len(values)), class_names)
    axes[2].set_xlabel("detector class")
    axes[2].set_ylabel("normalized region energy")
    axes[2].set_title("Detector-region readout")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(f"True: {true_name}    Predicted: {pred_name}")
    _save_figure(fig, path, dpi)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir, epoch_name, class_names, enabled=True, dpi=150):
    if not enabled:
        return
    model.eval()
    images, targets = batch
    logits, intermediates = model(images, return_intermediates=True)
    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    phases = model.phase_stack_wrapped()

    phase_dir = Path(run_dir) / "figures" / "phase_masks" / epoch_name
    save_phase_masks(model, phase_dir, input_field=intermediates["canvas_input_400"][0], dpi=dpi)

    detector_dir = Path(run_dir) / "figures" / "detector_outputs" / epoch_name
    save_detector_mask_overlay(model.detector, detector_dir / "ideal_detector_mask_overlay.png", class_names, dpi=dpi)
    save_k_space_mask(model.detector_prop, detector_dir / "k_space_constraint_mask.png", dpi=dpi)
    save_bar(intermediates["detector_energies"].mean(dim=0), detector_dir / "detector_energy_mean_bar.png", "Mean detector-region energy", class_names, dpi=dpi)

    rows = []
    for sample_index in range(len(images)):
        true = int(targets[sample_index].item())
        pred = int(preds[sample_index].item())
        rows.append(
            {
                "sample_index": sample_index,
                "true": true,
                "true_name": class_names[true],
                "pred": pred,
                "pred_name": class_names[pred],
                "confidence": float(probs[sample_index, pred].item()),
                "detector_energies": [float(value) for value in intermediates["detector_energies"][sample_index].cpu()],
            }
        )
        light_dir = Path(run_dir) / "figures" / "light_fields" / epoch_name / f"sample_{sample_index:03d}"
        save_intensity(intermediates["input_preprocessed_400"][sample_index], light_dir / "00_input_amplitude.png", "Input amplitude", dpi=dpi, normalize=False)
        save_intensity(intermediates["canvas_input_400"][sample_index], light_dir / "01_canvas_input_400.png", "336 resized input, zero-padded to 400", dpi=dpi, normalize=False)
        save_intensity(intermediates["after_input_to_layer"][sample_index], light_dir / "02_at_phase_plate.png", "Field intensity at phase plate (z=0)", dpi=dpi)
        save_phase_overlay(intermediates["canvas_input_400"][sample_index], phases[0], light_dir / "03_phase_mask_overlay.png", dpi=dpi)
        file_index = 4
        for layer_index in range(1, model.num_layers + 1):
            save_intensity(
                intermediates[f"after_phase_modulation_{layer_index}"][sample_index],
                light_dir / f"{file_index:02d}_after_phase_modulation_{layer_index}.png",
                f"Intensity immediately after phase modulation {layer_index}",
                dpi=dpi,
            )
            file_index += 1
            if layer_index < model.num_layers:
                save_intensity(
                    intermediates[f"after_propagation_{layer_index}"][sample_index],
                    light_dir / f"{file_index:02d}_after_propagation_{layer_index}.png",
                    f"Intensity after propagation {layer_index}",
                    dpi=dpi,
                )
                file_index += 1
        save_detector_intensity_with_regions(
            intermediates["detector_intensity"][sample_index],
            model.detector,
            light_dir / f"{file_index:02d}_detector_intensity_with_regions.png",
            class_names,
            "Detector intensity with square integration regions",
            dpi=dpi,
        )
        save_bar(
            intermediates["detector_energies"][sample_index],
            light_dir / f"{file_index + 1:02d}_detector_region_energy_bar.png",
            "Detector-region energy",
            class_names,
            dpi=dpi,
        )
        save_detector_sample_summary(
            images[sample_index],
            intermediates["detector_intensity"][sample_index],
            intermediates["detector_energies"][sample_index],
            model.detector,
            light_dir / "sample_summary.png",
            class_names,
            class_names[true],
            class_names[pred],
            dpi=dpi,
        )
    sample_dir = Path(run_dir) / "figures" / "samples" / epoch_name
    save_json(rows, sample_dir / "sample_predictions.json")
    save_sample_predictions(images, targets, preds, probs, sample_dir / "sample_predictions.png", class_names, dpi=dpi)


def save_bar(values, path, title, class_names=None, dpi=150):
    values = values.detach().cpu().float()
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    ax.bar(range(len(values)), values, color="tab:blue")
    ax.set_xticks(range(len(values)), class_names or [str(i) for i in range(len(values))])
    ax.set_xlabel("detector class")
    ax.set_ylabel("normalized region energy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path, dpi)


def save_sample_predictions(images, targets, preds, probs, path, class_names, dpi=150):
    count = min(len(images), 12)
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), constrained_layout=True)
    axes = list(getattr(axes, "flat", [axes]))
    for idx, ax in enumerate(axes[:count]):
        im = ax.imshow(images[idx, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        true = int(targets[idx].item())
        pred = int(preds[idx].item())
        conf = float(probs[idx, pred].item())
        ax.set_title(f"True: {class_names[true]} | Pred: {class_names[pred]} ({conf:.2f})")
        _label_image_axes(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("amplitude")
    for ax in axes[count:]:
        ax.axis("off")
    _save_figure(fig, path, dpi)


def save_training_curves(rows, path, dpi=150):
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train loss")
    axes[0].plot(epochs, [row["test_loss"] for row in rows], label="test loss")
    axes[0].set_title("Training objective loss trend")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="train accuracy")
    axes[1].plot(epochs, [row["test_acc"] for row in rows], label="test accuracy")
    axes[1].set_title("Classification accuracy trend")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    _save_figure(fig, path, dpi)


def confusion_matrix(preds, targets, num_classes=10):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, pred in zip(targets.view(-1).cpu(), preds.view(-1).cpu()):
        matrix[int(true), int(pred)] += 1
    return matrix


def save_confusion_matrix(matrix, path, class_names=None, dpi=150):
    matrix = matrix.detach().cpu().float()
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", origin="upper")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("sample count")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix")
    if class_names:
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
    _save_figure(fig, path, dpi)


def save_confusion_csv(matrix, path):
    rows = []
    for i in range(matrix.shape[0]):
        row = {"row": i}
        for j in range(matrix.shape[1]):
            row[str(j)] = int(matrix[i, j].item())
        rows.append(row)
    write_rows(path, rows)
