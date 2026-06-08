import json
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _field_image(field: torch.Tensor, sample_index: int = 0) -> np.ndarray:
    value = field
    if torch.is_complex(value):
        value = torch.abs(value).square()
    if value.ndim == 3:
        value = value[sample_index]
    array = value.detach().cpu().float().numpy()
    return np.log10(array / (array.max() + 1e-12) + 1e-8)


def _save_field(field: torch.Tensor, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(_field_image(field), cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_phase(phase: torch.Tensor, path: Path, title: str) -> None:
    wrapped = torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        wrapped,
        cmap="twilight",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_bar(values, path: Path, title: str, ylabel: str, prefix: str) -> None:
    values = torch.as_tensor(values).detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels([f"{prefix}{index}" for index in range(len(values))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_expert_phases(model, path: Path) -> None:
    fig, axes = plt.subplots(
        model.num_layers,
        4,
        figsize=(12, 2.6 * model.num_layers),
        squeeze=False,
    )
    for layer_index, layer in enumerate(model.expert_layers):
        phases = layer.get_phase_wrapped().detach().cpu().numpy()
        for expert_index in range(4):
            axes[layer_index, expert_index].imshow(
                phases[expert_index],
                cmap="twilight",
                vmin=0.0,
                vmax=2.0 * math.pi,
            )
            axes[layer_index, expert_index].set_title(
                f"Layer {layer_index + 1}, E{expert_index}"
            )
            axes[layer_index, expert_index].axis("off")
    fig.suptitle("Initial Expert Phase Layers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_initial_state(
    model,
    diagnostics: Dict,
    output_dir: Path,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    task_name: Optional[str] = None,
    save_images: bool = True,
) -> Dict:
    """Save one pre-optimization optical state as separate Word-ready images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates = diagnostics["intermediates"]
    suffix = "epoch_0000"
    fields = [
        (
            "input_amplitude",
            intermediates["input_amplitude"],
            "Input Amplitude",
        ),
        (
            "after_input_to_prompt",
            intermediates["after_input_to_prompt"],
            "After Input-to-Prompt Propagation",
        ),
        ("after_prompt", intermediates["after_prompt"], "After Prompt"),
        (
            "expert_entrance",
            intermediates["expert_entrance_intensity"],
            "Expert Entrance Plane",
        ),
    ]
    for index, field in enumerate(intermediates["after_each_layer"], start=1):
        fields.append(
            (
                f"after_expert_layer_{index}",
                field,
                f"After Expert Layer {index}",
            )
        )
    fields.extend(
        [
            (
                "after_global_fc",
                intermediates["after_global_fc"],
                "After Global FC",
            ),
            (
                "detector_plane",
                intermediates["detector_intensity"],
                "Detector Plane",
            ),
        ]
    )
    visualization_error = None
    if save_images:
        try:
            for file_stem, field, title in fields:
                _save_field(
                    field,
                    output_dir / f"{file_stem}_{suffix}.png",
                    title,
                )

            _save_bar(
                diagnostics["amplitudes"],
                output_dir / f"prompt_amplitude_bar_{suffix}.png",
                "Initial Prompt Amplitudes",
                "Amplitude",
                "E",
            )
            _save_bar(
                diagnostics["expert_energy_ratios"],
                output_dir / f"expert_energy_bar_{suffix}.png",
                "Initial Expert Entrance Energy Ratios",
                "Energy / total",
                "E",
            )
            _save_bar(
                diagnostics["detector_energies"],
                output_dir / f"detector_energy_bar_{suffix}.png",
                "Initial Detector Energies",
                "Detector energy",
                "D",
            )
            _save_phase(
                intermediates["prompt_phase"],
                output_dir / f"prompt_phase_{suffix}.png",
                "Initial Prompt Phase",
            )
            _save_phase(
                intermediates["global_fc_phase"],
                output_dir / f"global_fc_phase_{suffix}.png",
                "Initial Global FC Phase",
            )
            _save_expert_phases(
                model,
                output_dir / f"expert_phase_layers_{suffix}.png",
            )
        except Exception as exc:  # pragma: no cover - depends on server packages.
            visualization_error = repr(exc)
            (output_dir / "visualization_error.txt").write_text(
                "Initial optical field image saving failed, so training can "
                "continue without PNG visualizations.\n"
                f"{visualization_error}\n",
                encoding="utf-8",
            )
            plt.close("all")

    payload = {
        "epoch": 0,
        "stage": "init",
        "task_name": task_name,
        "initial_val_loss": val_loss,
        "initial_val_acc": val_acc,
        "prompt_amplitudes": diagnostics["amplitudes"].tolist(),
        "prompt_powers": diagnostics["powers"].tolist(),
        "normalized_prompt_powers": diagnostics["normalized_powers"].tolist(),
        "expert_energy_ratios": diagnostics["expert_energy_ratios"].tolist(),
        "outside_energy_ratio": diagnostics["outside_energy_ratio"],
        "detector_energy_mean": float(
            diagnostics["detector_energies"].float().mean().item()
        ),
        "detector_energy_max": float(
            diagnostics["detector_energies"].float().max().item()
        ),
        "detector_energies": diagnostics["detector_energies"].tolist(),
        "visualization_saved": save_images and visualization_error is None,
        "visualization_error": visualization_error,
    }
    with open(
        output_dir / "initial_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    return payload


def build_architecture_report(
    model,
    config: Dict,
    optimizer_settings: Dict,
    training_mode: str,
    task_names=None,
) -> Dict:
    readout_cfg = config.get("readout", {})
    readout_type = readout_cfg.get("type", "optical_only")
    activation = readout_cfg.get("activation") if readout_type == "mlp" else None
    hidden_dim = readout_cfg.get("hidden_dim") if readout_type == "mlp" else None
    nonlinear_activation = readout_type == "mlp"
    if readout_type == "optical_only":
        statement = (
            "No electronic nonlinear activation is used. The only nonlinearity "
            "is optical intensity detection |U|^2 before detector energy readout."
        )
    elif readout_type == "mlp":
        statement = "Electronic nonlinear readout is enabled."
    else:
        statement = (
            "A trainable electronic linear readout is enabled, without an "
            "electronic nonlinear activation."
        )
    return {
        "training_mode": training_mode,
        "task_names": list(task_names or []),
        "shared_detector_class_count": model.num_classes,
        "optical_propagation_is_linear": True,
        "phase_masks_are_phase_only": True,
        "intensity_detection_abs_u_squared": True,
        "electronic_readout_exists": readout_type != "optical_only",
        "readout_type": readout_type,
        "electronic_activation": activation,
        "electronic_hidden_dim": hidden_dim,
        "electronic_nonlinear_activation_exists": nonlinear_activation,
        "electronic_trainable_parameters_exist": (
            model.electronic_parameter_count() > 0
        ),
        "nonlinearity_statement": statement,
        "total_optical_parameter_count": model.optical_parameter_count(),
        "optical_parameter_count_includes_prompt": True,
        "total_prompt_parameter_count": model.prompt_parameter_count(),
        "total_electronic_parameter_count": model.electronic_parameter_count(),
        "optimizer": optimizer_settings,
        "multitask_label_note": (
            "Tasks share detector indices 0-9, but class semantics differ by task."
            if training_mode == "multitask"
            else None
        ),
    }


def save_architecture_report(report: Dict, run_dir: Path) -> None:
    with open(
        run_dir / "architecture_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)
    lines = [
        "# Four-Expert MoE Architecture Report",
        "",
        f"- training mode: {report['training_mode']}",
        f"- optical propagation is linear: {report['optical_propagation_is_linear']}",
        f"- phase masks are phase-only: {report['phase_masks_are_phase_only']}",
        f"- intensity detection uses |U|^2: {report['intensity_detection_abs_u_squared']}",
        f"- readout type: {report['readout_type']}",
        f"- electronic nonlinear activation: {report['electronic_nonlinear_activation_exists']}",
        f"- electronic activation: {report['electronic_activation']}",
        f"- electronic hidden dimension: {report['electronic_hidden_dim']}",
        f"- optical parameters: {report['total_optical_parameter_count']}",
        "- optical parameter count includes the prompt parameters listed below",
        f"- prompt parameters: {report['total_prompt_parameter_count']}",
        f"- electronic parameters: {report['total_electronic_parameter_count']}",
        f"- optimizer: {report['optimizer']}",
        "",
        report["nonlinearity_statement"],
    ]
    if report.get("multitask_label_note"):
        lines.extend(["", report["multitask_label_note"]])
    (run_dir / "architecture_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
