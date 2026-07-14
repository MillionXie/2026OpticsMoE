import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _array(value):
    value = value.detach().cpu()
    return value.numpy()


def _save_map(value, path, title, kind="intensity"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = value.detach().cpu()
    if kind == "intensity":
        array = tensor.abs().square().numpy() if tensor.is_complex() else tensor.numpy()
        cmap = "inferno"
        label = "optical intensity"
        kwargs = {"vmin": 0.0}
    elif kind == "amplitude":
        array = tensor.abs().numpy()
        cmap = "viridis"
        label = "field amplitude"
        kwargs = {"vmin": 0.0}
    elif kind == "power":
        array = tensor.numpy()
        cmap = "viridis"
        label = "normalized coupled mode power"
        kwargs = {"vmin": 0.0}
    else:
        array = torch.angle(tensor).numpy() if tensor.is_complex() else torch.remainder(tensor, 2 * math.pi).numpy()
        cmap = "twilight"
        label = "phase (rad)"
        kwargs = {"vmin": -math.pi if tensor.is_complex() else 0.0, "vmax": math.pi if tensor.is_complex() else 2 * math.pi}
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(array, cmap=cmap, origin="lower", **kwargs)
    axis.set_title(f"{title}\nshape={array.shape}, min={array.min():.3g}, max={array.max():.3g}")
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _detector_bounds(detector):
    result = []
    for mask in detector.masks.cpu():
        positions = mask.nonzero()
        result.append((int(positions[:, 0].min()), int(positions[:, 1].min()), int(positions[:, 0].max()) + 1, int(positions[:, 1].max()) + 1))
    return result


def save_detector(intensity, energies, detector, path, class_names, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = intensity.detach().cpu().numpy()
    values = energies.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    image = axes[0].imshow(array, cmap="inferno", origin="lower", vmin=0)
    for index, (y0, x0, y1, x1) in enumerate(_detector_bounds(detector)):
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1.4))
        axes[0].text(x0, y1 + 3, class_names[index], color="cyan", fontsize=8)
    axes[0].set_title(title)
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("square-law intensity")
    axes[1].bar(range(len(values)), values)
    axes[1].set_xticks(range(len(values)), class_names, rotation=20)
    axes[1].set_xlabel("class detector")
    axes[1].set_ylabel("normalized detector energy")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_expert_type_outputs(local_outputs, expert_types, path):
    """Save one representative complex output for each expert type."""
    representatives = [expert_types.index(value) for value in ("d2nn", "fourier", "fiber")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    for row, index in enumerate(representatives):
        field = local_outputs[index].detach().cpu()
        intensity = field.abs().square().numpy()
        phase = torch.angle(field).numpy()
        image_i = axes[row, 0].imshow(intensity, origin="lower", cmap="inferno", vmin=0)
        image_p = axes[row, 1].imshow(phase, origin="lower", cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[row, 0].set_title(f"E{index} {expert_types[index]} output intensity")
        axes[row, 1].set_title(f"E{index} {expert_types[index]} output phase")
        for axis in axes[row]:
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
        fig.colorbar(image_i, ax=axes[row, 0], fraction=0.046, pad=0.04, label="intensity")
        fig.colorbar(image_p, ax=axes[row, 1], fraction=0.046, pad=0.04, label="phase (rad)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir, tag, class_names, enabled=True):
    if not enabled:
        return
    images, labels = batch
    was_training = model.training
    model.eval()
    logits, items = model(images, return_intermediates=True, capture_expert_outputs=True)
    predictions = logits.argmax(1)
    root = Path(run_dir) / "figures" / tag
    sample = root / "sample_000"
    _save_map(items["input_canvas"][0], sample / "01_input_amplitude.png", "Input amplitude", "amplitude")
    _save_map(items["prompt_amplitude"][0], sample / "02_prompt_amplitude.png", "Input-dependent top-k prompt amplitude", "amplitude")
    _save_map(items["prompt_phase"][0], sample / "03_prompt_phase.png", "Fixed continuous prompt phase", "phase")
    _save_map(items["expert_entrance"][0], sample / "04_expert_entrance_intensity.png", "Expert-bank entrance", "intensity")
    _save_map(items["expert_bank_output"][0], sample / "05_expert_bank_output_intensity.png", "Reassembled heterogeneous-bank output", "intensity")
    _save_map(items["expert_bank_output"][0], sample / "06_expert_bank_output_phase.png", "Reassembled heterogeneous-bank output", "phase")
    local_outputs = items["expert_local_outputs"][0]
    save_expert_type_outputs(local_outputs, items["expert_types"], sample / "07_representative_expert_type_outputs.png")
    expert_root = sample / "expert_outputs"
    for index, (expert_type, field) in enumerate(zip(items["expert_types"], local_outputs)):
        _save_map(field, expert_root / f"expert_{index:02d}_{expert_type}_intensity.png", f"E{index} {expert_type} output", "intensity")
        _save_map(field, expert_root / f"expert_{index:02d}_{expert_type}_phase.png", f"E{index} {expert_type} output", "phase")
    intermediate_root = sample / "expert_intermediates"
    for expert_index, (expert, details) in enumerate(zip(model.expert_bank.experts, items["expert_details"])):
        if expert.expert_type == "fourier":
            for block_index, field in enumerate(details["fourier_block_fields"], 1):
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fourier_block_{block_index:02d}_intensity.png", f"E{expert_index} FourierConv{block_index} output", "intensity")
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fourier_block_{block_index:02d}_phase.png", f"E{expert_index} FourierConv{block_index} output", "phase")
            for layer_index, field in enumerate(details["tail_spatial_fields"], 1):
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fourier_tail_{layer_index:02d}_intensity.png", f"E{expert_index} Fourier tail spatial layer {layer_index}", "intensity")
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fourier_tail_{layer_index:02d}_phase.png", f"E{expert_index} Fourier tail spatial layer {layer_index}", "phase")
        elif expert.expert_type == "fiber":
            for layer_index, field in enumerate(details["encoder_fields"], 1):
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fiber_encoder_{layer_index:02d}_intensity.png", f"E{expert_index} Fiber encoder layer {layer_index}", "intensity")
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fiber_encoder_{layer_index:02d}_phase.png", f"E{expert_index} Fiber encoder layer {layer_index}", "phase")
            _save_map(details["encoded_field"][0], intermediate_root / f"expert_{expert_index:02d}_fiber_encoded_intensity.png", f"E{expert_index} field presented to fiber modes", "intensity")
            _save_map(details["reconstructed_field"][0], intermediate_root / f"expert_{expert_index:02d}_fiber_reconstructed_intensity.png", f"E{expert_index} coherent fiber reconstruction", "intensity")
            _save_map(details["reconstructed_field"][0], intermediate_root / f"expert_{expert_index:02d}_fiber_reconstructed_phase.png", f"E{expert_index} coherent fiber reconstruction", "phase")
            for layer_index, field in enumerate(details["decoder_fields"], 1):
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fiber_decoder_{layer_index:02d}_intensity.png", f"E{expert_index} Fiber decoder layer {layer_index}", "intensity")
                _save_map(field[0], intermediate_root / f"expert_{expert_index:02d}_fiber_decoder_{layer_index:02d}_phase.png", f"E{expert_index} Fiber decoder layer {layer_index}", "phase")
            mode_distribution = items["fiber_mode_power_distribution"][0, expert_index, : expert.num_modes].reshape(expert.mode_grid)
            _save_map(mode_distribution, intermediate_root / f"expert_{expert_index:02d}_fiber_mode_power_distribution.png", f"E{expert_index} normalized per-mode coupled power", "power")
            metrics = {
                "expert_index": expert_index,
                "coupling_efficiency": float(items["fiber_coupling_efficiency"][0, expert_index].cpu()),
                "effective_mode_number": float(items["fiber_effective_mode_number"][0, expert_index].cpu()),
                "input_power": float(items["expert_input_power"][0, expert_index].cpu()),
                "output_power": float(items["expert_output_power"][0, expert_index].cpu()),
                "per_mode_power_distribution": mode_distribution.cpu().flatten().tolist(),
            }
            metrics_path = intermediate_root / f"expert_{expert_index:02d}_fiber_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_map(items["at_global_fc"][0], sample / "08_at_global_fc_intensity.png", "At global FC", "intensity")
    _save_map(items["global_fc_phase"], sample / "09_global_fc_phase.png", "Global FC phase", "phase")
    save_detector(
        items["detector_intensity"][0],
        items["detector_energies"][0],
        model.detector,
        sample / "10_detector_and_bars.png",
        class_names,
        f"Detector | true={class_names[int(labels[0])]} pred={class_names[int(predictions[0])]}",
    )
    parameter_root = root / "expert_parameters"
    for index, expert in enumerate(model.expert_bank.experts):
        if expert.expert_type == "d2nn":
            for layer_index, phase in enumerate(expert.phase_stack(), 1):
                _save_map(phase, parameter_root / f"expert_{index:02d}_d2nn_phase_{layer_index:02d}.png", f"E{index} D2NN phase {layer_index}", "phase")
        elif expert.expert_type == "fourier":
            for block_index, phase in enumerate(expert.frequency_phase_stack(), 1):
                _save_map(phase, parameter_root / f"expert_{index:02d}_fourier_frequency_phase_{block_index:02d}.png", f"E{index} Fourier block {block_index} frequency phase", "phase")
            for layer_index, phase in enumerate(expert.spatial_phase_stack(), 1):
                _save_map(phase, parameter_root / f"expert_{index:02d}_fourier_tail_spatial_phase_{layer_index:02d}.png", f"E{index} Fourier tail spatial phase {layer_index}", "phase")
        else:
            for layer_index, phase in enumerate(expert.phase_stack(), 1):
                region = "encoder" if layer_index <= expert.num_pre_layers else "decoder"
                local_index = layer_index if region == "encoder" else layer_index - expert.num_pre_layers
                _save_map(phase, parameter_root / f"expert_{index:02d}_fiber_{region}_phase_{local_index:02d}.png", f"E{index} Fiber {region} phase {local_index}", "phase")
            rows, cols = expert.mode_grid
            _save_map(expert.mode_phase().reshape(rows, cols), parameter_root / f"expert_{index:02d}_fiber_mode_phase.png", f"E{index} fiber mode phase", "phase")
            _save_map(expert.mode_amplitude().reshape(rows, cols), parameter_root / f"expert_{index:02d}_fiber_mode_amplitude.png", f"E{index} fiber mode amplitude", "amplitude")
    model.train(was_training)


def confusion_matrix(predictions, targets, classes):
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def save_confusion(matrix, path, class_names):
    array = matrix.numpy()
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(array, cmap="Blues")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(column, row, str(array[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(class_names)), class_names, rotation=25)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    fig.colorbar(image, ax=axis, label="samples")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(rows, path):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in rows], label="test")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="train")
    axes[1].plot(epochs, [row["test_acc"] for row in rows], label="test")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
