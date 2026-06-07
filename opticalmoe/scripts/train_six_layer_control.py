import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.data import create_dataloaders
from opticalmoe.optics.six_layer_control import SixLayerNoPromptControl
from opticalmoe.training import save_checkpoint
from opticalmoe.utils import load_config, save_json, set_seed
from opticalmoe.utils.run import create_run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the separate six-layer no-prompt, no-expert control."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def build_model(config: Dict, num_classes: int) -> SixLayerNoPromptControl:
    layout = config.get("layout", {})
    optics = config.get("optics", {})
    control = config.get("control", {})
    detector = config.get("detector", {})
    readout = config.get("readout", {})
    distances = optics.get("distances_m", {})
    return SixLayerNoPromptControl(
        num_classes=num_classes,
        canvas_shape=(
            int(layout.get("canvas_height", 700)),
            int(layout.get("canvas_width", 700)),
        ),
        input_size=int(layout.get("input_size", 200)),
        num_masks=int(control.get("num_masks", 6)),
        parameter_grid_size=int(control.get("parameter_grid_size", 464)),
        wavelength_m=float(optics.get("wavelength_m", 532e-9)),
        pixel_size_m=float(optics.get("pixel_size_m", 8e-6)),
        distances_m={
            "input_to_identity_prompt": float(
                distances.get("input_to_identity_prompt", 0.20)
            ),
            "identity_prompt_to_first_mask": float(
                distances.get("identity_prompt_to_first_mask", 0.20)
            ),
            "inter_mask": float(distances.get("inter_mask", 0.05)),
            "last_mask_to_detector": float(
                distances.get("last_mask_to_detector", 0.05)
            ),
        },
        phase_param=optics.get("phase_param", "unconstrained"),
        phase_init=optics.get("phase_init", "uniform_0_2pi"),
        phase_init_std=float(optics.get("phase_init_std", 0.02)),
        detector_size=int(detector.get("detector_size", 32)),
        detector_layout=detector.get("layout", "grid"),
        normalize_detector_energy=bool(
            readout.get("normalize_detector_energy", True)
        ),
        readout_type=readout.get("type", "optical_only"),
        logit_scale=float(readout.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout.get("hidden_dim", 64)),
        readout_activation=readout.get("activation", "relu"),
        evanescent_mode=optics.get("evanescent_mode", "zero"),
    )


def build_optimizer(model, config: Dict):
    cfg = config.get("optimizer", {})
    name = cfg.get("type", "adamw").lower()
    kwargs = {
        "lr": float(cfg.get("lr", 0.003)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(cfg.get("momentum", 0.9)),
            **kwargs,
        )
    raise ValueError("optimizer.type must be adamw, adam, or sgd.")


def stage_for_epoch(epoch: int, stage_epochs: List[int]) -> int:
    cumulative = 0
    for stage_idx, length in enumerate(stage_epochs):
        cumulative += int(length)
        if epoch <= cumulative:
            return stage_idx
    return len(stage_epochs) - 1


def apply_progressive_stage(model, stage_idx: int, enabled: bool) -> Dict:
    if enabled:
        active_start = model.num_masks - int(stage_idx) - 1
        active_indices = list(range(max(0, active_start), model.num_masks))
    else:
        active_indices = list(range(model.num_masks))
    for index, phase_mask in enumerate(model.phase_masks):
        requires_grad = index in active_indices
        for parameter in phase_mask.parameters():
            parameter.requires_grad = requires_grad
    for parameter in model.readout.parameters():
        parameter.requires_grad = True
    return {
        "stage_idx": int(stage_idx),
        "active_masks": [index + 1 for index in active_indices],
        "trainable_parameter_names": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
    }


def train_one_epoch(model, loader, optimizer, device, criterion, print_freq=100):
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for batch_idx, batch in enumerate(loader):
        images, targets = batch[:2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.numel()
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch_size
        if print_freq > 0 and (
            (batch_idx + 1) % int(print_freq) == 0
            or batch_idx + 1 == len(loader)
        ):
            print(
                f"  batch {batch_idx + 1}/{len(loader)} | "
                f"loss={loss_sum / max(seen, 1):.4f} | "
                f"acc={correct / max(seen, 1):.4f}"
            )
    return loss_sum / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion, max_batches: Optional[int] = None):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    targets_all = []
    predictions_all = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images, targets = batch[:2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        batch_size = targets.numel()
        loss_sum += float(loss.item()) * batch_size
        correct += int((predictions == targets).sum().item())
        seen += batch_size
        targets_all.append(targets.cpu())
        predictions_all.append(predictions.cpu())
    return {
        "loss": loss_sum / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "samples": seen,
        "targets": torch.cat(targets_all) if targets_all else torch.empty(0),
        "predictions": (
            torch.cat(predictions_all) if predictions_all else torch.empty(0)
        ),
    }


def write_rows(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _field_image(value: torch.Tensor) -> np.ndarray:
    if torch.is_complex(value):
        value = torch.abs(value).square()
    if value.ndim == 3:
        value = value[0]
    array = value.detach().cpu().float().numpy()
    return np.log10(array / (array.max() + 1e-12) + 1e-8)


@torch.no_grad()
def save_optical_state(model, batch, device, output_dir: Path, title: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch[0].to(device)
    targets = batch[1]
    logits, intermediates = model(images, return_intermediates=True)
    predictions = logits.argmax(dim=1).detach().cpu()
    fields = [
        ("00_input.png", intermediates["input_amplitude"], "Input"),
        (
            "01_after_input_to_prompt.png",
            intermediates["after_input_to_prompt"],
            "After Input-to-Identity-Prompt Propagation",
        ),
        (
            "02_after_identity_prompt.png",
            intermediates["after_identity_prompt"],
            "After Identity Prompt",
        ),
        (
            "03_first_mask_entrance.png",
            intermediates["first_mask_entrance"],
            "First Mask Entrance",
        ),
    ]
    for index, field in enumerate(intermediates["after_each_mask"], start=1):
        fields.append(
            (
                f"{index + 3:02d}_after_mask_{index}.png",
                field,
                f"After Mask {index}",
            )
        )
    fields.append(
        (
            "10_detector_plane.png",
            intermediates["detector_intensity"],
            "Detector Plane",
        )
    )
    for filename, value, plane_title in fields:
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(_field_image(value), cmap="inferno")
        ax.set_title(f"{title}: {plane_title}")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    columns = 4
    rows = int(np.ceil(len(fields) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (_filename, value, plane_title) in zip(axes, fields):
        ax.imshow(_field_image(value), cmap="inferno")
        ax.set_title(plane_title)
        ax.axis("off")
    for ax in axes[len(fields) :]:
        ax.axis("off")
    fig.suptitle(f"{title}: Light Field Overview")
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=150)
    plt.close(fig)

    phases = model.get_phase_masks_wrapped().detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for index, ax in enumerate(axes.flat):
        image = ax.imshow(phases[index], cmap="twilight", vmin=0.0, vmax=2 * np.pi)
        ax.set_title(f"Mask {index + 1}")
        ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle(f"{title}: Six Parameter-Matched Phase Masks")
    fig.savefig(output_dir / "phase_masks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    count = min(int(images.shape[0]), 8)
    if count > 0:
        input_canvas = intermediates["input_amplitude"].detach().cpu()
        fig, axes = plt.subplots(1, count, figsize=(2.4 * count, 2.8))
        axes = np.asarray(axes).reshape(-1)
        for index in range(count):
            axes[index].imshow(input_canvas[index].numpy(), cmap="gray")
            axes[index].set_title(
                f"y={int(targets[index])}, pred={int(predictions[index])}"
            )
            axes[index].axis("off")
        fig.suptitle(f"{title}: Fixed Sample Predictions")
        fig.tight_layout()
        fig.savefig(output_dir / "sample_predictions.png", dpi=150)
        plt.close(fig)
    with open(output_dir / "sample_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "targets": targets[:count].tolist(),
                "predictions": predictions[:count].tolist(),
            },
            handle,
            indent=2,
        )


def save_confusion_matrix(targets, predictions, num_classes, path):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for target, prediction in zip(targets.long(), predictions.long()):
        matrix[target, prediction] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Six-Layer Control Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    if config.get("experiment", {}).get("type") != "six_layer_control":
        raise ValueError(
            "This script requires experiment.type=six_layer_control."
        )
    if args.smoke_test:
        config["dataset"]["smoke_test"] = True
    seed = int(config.get("seed", 7))
    set_seed(seed)
    run_name = args.run_name or config["experiment"].get(
        "run_name", "six_layer_control"
    )
    run_dir = create_run_dir(run_name, base_dir=str(PROJECT_ROOT / "runs"))
    shutil.copyfile(args.config, run_dir / "config.yaml")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        config["dataset"], seed
    )
    device = choose_device(args.device or config.get("device", "auto"))
    model = build_model(config, num_classes).to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    training = config.get("training", {})
    progressive = training.get("progressive", {})
    progressive_enabled = bool(progressive.get("enabled", True))
    stage_epochs = list(progressive.get("stage_epochs", [3, 3, 3, 3, 3, 10]))
    if len(stage_epochs) != 6:
        raise ValueError("Six-layer progressive stage_epochs must have 6 values.")
    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else (
            sum(stage_epochs)
            if progressive_enabled
            else int(training.get("epochs", 25))
        )
    )
    evaluation = training.get("evaluation", {})
    max_val_batches = evaluation.get("max_val_batches")
    max_test_batches = evaluation.get("max_test_batches")
    print_freq = int(training.get("print_freq", 100))
    fixed_batch = next(iter(val_loader))

    target_parameters = int(
        config.get("comparison", {}).get(
            "target_moe_optical_parameters", 1290008
        )
    )
    actual_parameters = model.optical_parameter_count()
    parameter_difference = actual_parameters - target_parameters
    parameter_difference_ratio = parameter_difference / max(target_parameters, 1)
    report = {
        "experiment_type": "six_layer_no_prompt_no_partition_control",
        "visual_prompt": False,
        "expert_partitions": False,
        "hard_expert_apertures": False,
        "num_full_canvas_masks": model.num_masks,
        "canvas_shape": list(model.canvas_shape),
        "parameter_grid_size_per_mask": model.parameter_grid_size,
        "target_moe_optical_parameters": target_parameters,
        "actual_control_optical_parameters": actual_parameters,
        "parameter_difference": parameter_difference,
        "parameter_difference_ratio": parameter_difference_ratio,
        "electronic_parameters": model.electronic_parameter_count(),
        "propagation_segments": model.num_propagation_segments,
        "distances_m": model.distances_m,
        "parameterization_note": (
            "Each 464x464 trainable phase grid is periodically interpolated "
            "to a full 700x700 unit-magnitude modulation."
        ),
    }
    save_json(report, str(run_dir / "control_architecture_report.json"))
    (run_dir / "control_architecture_report.md").write_text(
        "# Six-Layer Control Architecture\n\n"
        "- No visual prompt modulation; the prompt plane is identity.\n"
        "- No expert partitions or blocked expert gaps.\n"
        "- Six full-canvas phase modulations.\n"
        f"- Target MoE optical parameters: {target_parameters}\n"
        f"- Actual control optical parameters: {actual_parameters}\n"
        f"- Relative difference: {parameter_difference_ratio:.4%}\n",
        encoding="utf-8",
    )

    print(f"device: {device}")
    print(f"dataset: {config['dataset']['name']}")
    print(
        f"control optical parameters: {actual_parameters}; "
        f"target MoE parameters: {target_parameters}; "
        f"difference: {parameter_difference_ratio:.3%}"
    )
    print(
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"lr={optimizer.param_groups[0]['lr']}"
    )
    save_optical_state(
        model, fixed_batch, device, run_dir / "initial_state", "Initial"
    )

    rows = []
    stage_records = []
    previous_stage = None
    best_val_acc = -1.0
    final_test = None
    vis_interval = int(
        config.get("visualization", {}).get("save_interval_epochs", 5)
    )
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        stage_idx = (
            stage_for_epoch(epoch, stage_epochs)
            if progressive_enabled
            else 5
        )
        stage_info = apply_progressive_stage(
            model, stage_idx, progressive_enabled
        )
        if stage_idx != previous_stage:
            stage_records.append(stage_info)
            previous_stage = stage_idx
            print(
                f"stage {stage_idx}: active masks "
                f"{stage_info['active_masks']}"
            )
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            print_freq=print_freq,
        )
        val_result = evaluate(
            model,
            val_loader,
            device,
            criterion,
            max_batches=max_val_batches,
        )
        test_result = evaluate(
            model,
            test_loader,
            device,
            criterion,
            max_batches=max_test_batches,
        )
        final_test = test_result
        row = {
            "epoch": epoch,
            "stage_idx": stage_idx,
            "active_masks": " ".join(
                str(value) for value in stage_info["active_masks"]
            ),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_result["loss"],
            "val_acc": val_result["accuracy"],
            "val_samples": val_result["samples"],
            "test_loss": test_result["loss"],
            "test_acc": test_result["accuracy"],
            "test_samples": test_result["samples"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_duration_seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        write_rows(run_dir / "metrics.csv", rows)
        save_json(
            {"stages": stage_records},
            str(run_dir / "trainable_parameters_by_stage.json"),
        )
        save_checkpoint(
            str(run_dir / "last.pt"), model, optimizer, epoch, row
        )
        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            save_checkpoint(
                str(run_dir / "best.pt"), model, optimizer, epoch, row
            )
        if vis_interval > 0 and (
            epoch % vis_interval == 0 or epoch == epochs
        ):
            save_optical_state(
                model,
                fixed_batch,
                device,
                run_dir / "light_fields" / f"epoch_{epoch:04d}",
                f"Epoch {epoch}",
            )
        print(
            f"epoch {epoch:03d} | train loss={train_loss:.4f} "
            f"acc={train_acc:.4f} | val loss={val_result['loss']:.4f} "
            f"acc={val_result['accuracy']:.4f} | "
            f"test acc={test_result['accuracy']:.4f} | "
            f"time={row['epoch_duration_seconds'] / 60.0:.1f} min"
        )

    save_confusion_matrix(
        final_test["targets"],
        final_test["predictions"],
        num_classes,
        run_dir / "confusion_matrix.png",
    )
    summary = {
        **report,
        "run_name": run_name,
        "dataset": config["dataset"]["name"],
        "best_validation_accuracy": best_val_acc,
        "final_test_accuracy": final_test["accuracy"],
        "final_test_loss": final_test["loss"],
        "epochs": epochs,
        "optimizer": config.get("optimizer", {}),
    }
    save_json(summary, str(run_dir / "summary.json"))
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
