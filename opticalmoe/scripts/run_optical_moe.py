import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.data import create_dataloaders, create_mixed_mnist_fashion_dataloaders
from opticalmoe.optics import OpticalMoEClassifier, build_moe_layout
from opticalmoe.utils import cm_to_m, load_config, nm_to_m, set_seed, um_to_m


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "lr",
    "left_branch_energy_ratio_mean",
    "right_branch_energy_ratio_mean",
    "outside_energy_ratio_mean",
    "target_branch_energy_ratio_mean",
    "wrong_branch_energy_ratio_mean",
    "left_top1_acc_if_used",
    "right_top1_acc_if_used",
    "paired_sum_top1_acc",
    "energy_gated_local_top1_acc",
    "route_acc_if_task_id_available",
    "detector_margin_mean",
    "centroid_error_first_layer_mean",
    "centroid_drift_per_layer_mean",
    "edge_energy_ratio_mean",
    "mnist_acc",
    "fashion_acc",
    "overall_acc",
    "mnist_route_acc",
    "fashion_route_acc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run large-canvas OpticalMoE experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default=None, choices=["eval", "finetune", "train_scratch", "prompt_train"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target_side", default=None, choices=["left", "right"])
    parser.add_argument("--left_ckpt", default=None)
    parser.add_argument("--right_ckpt", default=None)
    parser.add_argument("--moe_ckpt", default=None, help="Load a full OpticalMoE checkpoint produced by this script.")
    parser.add_argument("--left_moe_ckpt", default=None, help="Copy the left side from an OpticalMoE checkpoint into the current left expert.")
    parser.add_argument("--right_moe_ckpt", default=None, help="Copy the right side from an OpticalMoE checkpoint into the current right expert.")
    parser.add_argument("--left_config", default=None)
    parser.add_argument("--right_config", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--readout_mode", default=None)
    parser.add_argument("--prompt_mode", default=None)
    parser.add_argument("--freeze_policy", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_train_size", type=int, default=None)
    parser.add_argument("--smoke_test_size", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--baseline_acc", type=float, default=None)
    parser.add_argument("--strict_geometry_check", action="store_true")
    parser.add_argument("--resume_optimizer", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def init_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics_csv(path: Path, row: Dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def create_moe_run_dir(run_name: str) -> Path:
    run_dir = PROJECT_ROOT / "runs" / run_name
    for subdir in ["phases", "light_fields", "detector_energies"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_config(config: Dict, args) -> Dict:
    config = dict(config)
    config.setdefault("experiment", {})
    config.setdefault("checkpoints", {})
    config.setdefault("dataset", {})
    config.setdefault("model", {})
    config.setdefault("optimizer", {})
    config.setdefault("training", {})
    config.setdefault("evaluation", {})

    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
    if args.target_side is not None:
        config["model"]["target_side"] = args.target_side
    if args.readout_mode is not None:
        config["model"]["readout_mode"] = args.readout_mode
    if args.prompt_mode is not None:
        config["model"]["prompt_mode"] = args.prompt_mode
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["optimizer"]["lr"] = float(args.lr)
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = int(args.batch_size)
    if args.smoke_test:
        config["dataset"]["smoke_test"] = True
    if args.smoke_train_size is not None:
        config["dataset"]["smoke_train_size"] = int(args.smoke_train_size)
    if args.smoke_test_size is not None:
        config["dataset"]["smoke_test_size"] = int(args.smoke_test_size)
    if args.freeze_policy is not None:
        config["training"]["freeze_policy"] = args.freeze_policy

    if args.left_ckpt is not None:
        config["checkpoints"]["left_ckpt"] = args.left_ckpt
    if args.right_ckpt is not None:
        config["checkpoints"]["right_ckpt"] = args.right_ckpt
    if args.moe_ckpt is not None:
        config["checkpoints"]["moe_ckpt"] = args.moe_ckpt
    if args.left_moe_ckpt is not None:
        config["checkpoints"]["left_moe_ckpt"] = args.left_moe_ckpt
    if args.right_moe_ckpt is not None:
        config["checkpoints"]["right_moe_ckpt"] = args.right_moe_ckpt
    if args.left_config is not None:
        config["checkpoints"]["left_config"] = args.left_config
    if args.right_config is not None:
        config["checkpoints"]["right_config"] = args.right_config

    if args.mode is not None:
        config["experiment"]["mode"] = args.mode
    if args.run_name is not None:
        config["experiment"]["run_name"] = args.run_name
    if args.baseline_acc is not None:
        config["experiment"]["baseline_acc"] = float(args.baseline_acc)
    if args.print_freq is not None:
        config["experiment"]["print_freq"] = int(args.print_freq)
    if args.strict_geometry_check:
        config["experiment"]["strict_geometry_check"] = True
    if args.resume_optimizer:
        config["experiment"]["resume_optimizer"] = True

    if "mode" in config and "mode" not in config["experiment"]:
        config["experiment"]["mode"] = config["mode"]
    if "run_name" in config and "run_name" not in config["experiment"]:
        config["experiment"]["run_name"] = config["run_name"]
    config["mode"] = config["experiment"].get("mode", "eval")
    config["run_name"] = config["experiment"].get("run_name", "optical_moe_run")
    return config


def build_model(config: Dict, num_classes: int, mode: str) -> OpticalMoEClassifier:
    optics_cfg = config.get("optics", {})
    layout_cfg = config.get("layout", {})
    model_cfg = config.get("model", {})
    detector_cfg = config.get("detector", {})
    distances_cm = optics_cfg.get("distances_cm", {})
    distances_m = {
        "input_to_prompt": cm_to_m(distances_cm.get("input_to_prompt", 1.0)),
        "prompt_to_first_layer": cm_to_m(distances_cm.get("prompt_to_first_layer", 24.0)),
        "inter_layer": cm_to_m(distances_cm.get("inter_layer", 5.0)),
        "last_layer_to_detector": cm_to_m(distances_cm.get("last_layer_to_detector", 5.0)),
    }

    prompt_mode = model_cfg.get("prompt_mode", "fixed_grating")
    freeze_policy = config.get("training", {}).get("freeze_policy")
    if freeze_policy in {None, "auto"}:
        freeze_policy = {
            "finetune": "compensation_only",
            "prompt_train": "compensation_only",
        }.get(mode)
    if mode in {"finetune", "prompt_train"} and freeze_policy == "compensation_only" and prompt_mode == "fixed_grating":
        prompt_mode = "trainable_residual_on_grating"

    layout = build_moe_layout(layout_cfg)
    return OpticalMoEClassifier(
        num_classes=num_classes,
        layout=layout,
        wavelength_m=nm_to_m(optics_cfg.get("wavelength_nm", 532.0)),
        pixel_size_m=um_to_m(optics_cfg.get("pixel_size_um", 8.0)),
        distances_m=distances_m,
        num_layers=int(optics_cfg.get("num_layers", 5)),
        phase_param=optics_cfg.get("phase_param", "unconstrained"),
        phase_init=optics_cfg.get("phase_init", "uniform"),
        detector_size=int(detector_cfg.get("detector_size", 32)),
        detector_layout=detector_cfg.get("layout", "grid"),
        readout_mode=model_cfg.get("readout_mode", "auto"),
        detector_normalization=model_cfg.get("detector_normalization", "local"),
        logit_scale=float(model_cfg.get("logit_scale", 10.0)),
        mode=model_cfg.get("mode", "single_side"),
        prompt_mode=prompt_mode,
        prompt_init=model_cfg.get("prompt_init", "fixed_grating"),
        target_side=model_cfg.get("target_side", "left"),
        prompt_slope_sign=int(model_cfg.get("prompt_slope_sign", 1)),
        use_entrance_detilt=bool(model_cfg.get("use_entrance_detilt", True)),
        use_aperture_masks=bool(model_cfg.get("use_aperture_masks", True)),
        evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
    )


def create_loaders(config: Dict, seed: int):
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("name", "mnist").lower() == "mixed_mnist_fashion":
        return create_mixed_mnist_fashion_dataloaders(dataset_cfg, seed=seed), True
    return create_dataloaders(dataset_cfg, seed=seed), False


def unpack_batch(batch, device: torch.device):
    if len(batch) == 3:
        images, targets, task_ids = batch
        return images.to(device), targets.to(device), task_ids.to(device)
    images, targets = batch
    return images.to(device), targets.to(device), None


def apply_freeze_policy(model: OpticalMoEClassifier, policy: str, target_side: Optional[str]) -> List[str]:
    warnings = []
    policy = policy or "frozen"
    if policy == "frozen":
        model.set_all_requires_grad(False)
    elif policy == "compensation_only":
        model.set_all_requires_grad(False)
        if model.prompt_raw_phase is not None:
            model.prompt_raw_phase.requires_grad = True
        else:
            warnings.append("compensation_only has no trainable prompt parameter; use prompt_mode=trainable_residual_on_grating.")
    elif policy == "first_layer_only":
        model.set_all_requires_grad(False)
        model.expert_layers[0].set_side_requires_grad(target_side, True)
    elif policy == "last_layer_only":
        model.set_all_requires_grad(False)
        model.expert_layers[-1].set_side_requires_grad(target_side, True)
    elif policy == "all_side":
        model.set_all_requires_grad(False)
        model.set_side_requires_grad(target_side, True)
    elif policy == "all":
        model.set_all_requires_grad(True)
    else:
        raise ValueError(f"Unsupported freeze_policy: {policy}")
    return warnings


def train_one_epoch(model, loader, optimizer, device, criterion, print_freq: int = 50) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        images, targets, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * targets.numel()
        total_correct += (preds == targets).sum().item()
        total_seen += targets.numel()
        if print_freq > 0 and (batch_idx == 1 or batch_idx % print_freq == 0 or batch_idx == num_batches):
            running_loss = total_loss / max(1, total_seen)
            running_acc = total_correct / max(1, total_seen)
            print(
                f"  batch {batch_idx:04d}/{num_batches:04d} | "
                f"loss {running_loss:.4f} | acc {running_acc:.4f}",
                flush=True,
            )
    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def _top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (torch.argmax(logits, dim=1) == targets).float().mean().item()


def _detector_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_value = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets.view(-1, 1), float("-inf"))
    return target_value - masked.max(dim=1).values


def task_to_side(task_name: Optional[str]) -> Optional[str]:
    if task_name is None:
        return None
    task = str(task_name).lower()
    if task in {"left", "mnist", "digit", "digits"}:
        return "left"
    if task in {"right", "fashion", "fashionmnist", "fashion_mnist"}:
        return "right"
    if task in {"none", "auto", "default"}:
        return None
    raise ValueError(f"Unsupported current_task: {task_name}")


@torch.no_grad()
def evaluate_moe(
    model,
    loader,
    device,
    criterion,
    target_side: Optional[str],
    mixed: bool,
    routing_mode: str = "model_default",
    current_task: Optional[str] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_targets = []
    all_preds = []
    all_task_ids = []

    accum = {
        "left_ratio": [],
        "right_ratio": [],
        "outside_ratio": [],
        "left_acc": [],
        "right_acc": [],
        "paired_acc": [],
        "energy_gated_acc": [],
        "route_acc": [],
        "margin": [],
        "first_centroid_error": [],
        "layer_drift": [],
        "edge": [],
    }

    original_side = model.target_side
    fixed_routing_side = task_to_side(current_task) if routing_mode == "fixed_task" else None

    def process_group(images, targets, task_ids, routing_side: Optional[str]):
        if routing_side in {"left", "right"} and hasattr(model, "set_fixed_routing_side"):
            model.set_fixed_routing_side(routing_side)
        logits, intermediates = model(images, return_intermediates=True)
        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        nonlocal total_loss, total_correct, total_seen
        total_loss += loss.item() * targets.numel()
        total_correct += (preds == targets).sum().item()
        total_seen += targets.numel()
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())
        if task_ids is not None:
            all_task_ids.append(task_ids.cpu())

        ratios = intermediates["branch_energy_ratios"]
        left_ratio = ratios[:, 0]
        right_ratio = ratios[:, 1]
        outside_ratio = ratios[:, 2]
        accum["left_ratio"].append(left_ratio.detach().cpu())
        accum["right_ratio"].append(right_ratio.detach().cpu())
        accum["outside_ratio"].append(outside_ratio.detach().cpu())

        left_logits = intermediates["detector_energies_left_local_norm"] * model.logit_scale
        right_logits = intermediates["detector_energies_right_local_norm"] * model.logit_scale
        paired_logits = intermediates["detector_energies_paired_sum"] * model.logit_scale
        gate_den = left_ratio + right_ratio + 1e-8
        gated_logits = (
            (left_ratio / gate_den).unsqueeze(1) * intermediates["detector_energies_left_local_norm"]
            + (right_ratio / gate_den).unsqueeze(1) * intermediates["detector_energies_right_local_norm"]
        ) * model.logit_scale
        accum["left_acc"].append(torch.tensor(_top1(left_logits, targets)))
        accum["right_acc"].append(torch.tensor(_top1(right_logits, targets)))
        accum["paired_acc"].append(torch.tensor(_top1(paired_logits, targets)))
        accum["energy_gated_acc"].append(torch.tensor(_top1(gated_logits, targets)))
        accum["margin"].append(_detector_margin(logits, targets).detach().cpu())

        metric_side = routing_side if routing_side in {"left", "right"} else target_side
        if metric_side in {"left", "right"}:
            target_y, target_x = model.layout.target_center(metric_side)
            first = intermediates["centroid_per_plane"]["after_prompt_to_first_layer"]
            error = torch.sqrt((first[:, 0] - target_y) ** 2 + (first[:, 1] - target_x) ** 2)
            accum["first_centroid_error"].append(error.detach().cpu())

        drift_terms = []
        previous = intermediates["centroid_per_plane"].get("after_entrance_detilt")
        for layer_idx in range(1, model.num_layers + 1):
            current = intermediates["centroid_per_plane"].get(f"after_layer_{layer_idx}_propagation")
            if previous is not None and current is not None:
                drift_terms.append(torch.abs(current[:, 1] - previous[:, 1]))
            previous = current
        if drift_terms:
            accum["layer_drift"].append(torch.stack(drift_terms, dim=1).mean(dim=1).detach().cpu())

        edge_values = list(intermediates["edge_energy_ratio_per_plane"].values())
        if edge_values:
            accum["edge"].append(torch.stack(edge_values, dim=1).mean(dim=1).detach().cpu())

        if task_ids is not None:
            if routing_mode in {"task_aware", "fixed_task"} and routing_side in {"left", "right"}:
                route_pred = torch.zeros_like(task_ids) if routing_side == "left" else torch.ones_like(task_ids)
            else:
                route_pred = torch.where(right_ratio > left_ratio, torch.ones_like(task_ids), torch.zeros_like(task_ids))
            accum["route_acc"].append((route_pred == task_ids).float().detach().cpu())

    for batch in loader:
        images, targets, task_ids = unpack_batch(batch, device)
        if routing_mode == "task_aware" and task_ids is not None:
            for task_value, side in [(0, "left"), (1, "right")]:
                mask = task_ids == task_value
                if mask.any():
                    process_group(images[mask], targets[mask], task_ids[mask], side)
        elif routing_mode == "fixed_task":
            process_group(images, targets, task_ids, fixed_routing_side)
        else:
            process_group(images, targets, task_ids, target_side)

    if original_side in {"left", "right"} and hasattr(model, "set_fixed_routing_side"):
        model.set_fixed_routing_side(original_side)

    targets_cpu = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    preds_cpu = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    task_cpu = torch.cat(all_task_ids) if all_task_ids else None

    def mean_accum(name: str) -> float:
        values = accum[name]
        if not values:
            return float("nan")
        return float(torch.cat([v.view(-1) for v in values]).float().mean().item())

    result = {
        "loss": total_loss / max(1, total_seen),
        "acc": total_correct / max(1, total_seen),
        "targets": targets_cpu,
        "preds": preds_cpu,
        "left_branch_energy_ratio_mean": mean_accum("left_ratio"),
        "right_branch_energy_ratio_mean": mean_accum("right_ratio"),
        "outside_energy_ratio_mean": mean_accum("outside_ratio"),
        "left_top1_acc_if_used": mean_accum("left_acc"),
        "right_top1_acc_if_used": mean_accum("right_acc"),
        "paired_sum_top1_acc": mean_accum("paired_acc"),
        "energy_gated_local_top1_acc": mean_accum("energy_gated_acc"),
        "route_acc_if_task_id_available": mean_accum("route_acc"),
        "detector_margin_mean": mean_accum("margin"),
        "centroid_error_first_layer_mean": mean_accum("first_centroid_error"),
        "centroid_drift_per_layer_mean": mean_accum("layer_drift"),
        "edge_energy_ratio_mean": mean_accum("edge"),
        "mnist_acc": float("nan"),
        "fashion_acc": float("nan"),
        "overall_acc": total_correct / max(1, total_seen),
        "mnist_route_acc": float("nan"),
        "fashion_route_acc": float("nan"),
    }

    if target_side == "left":
        result["target_branch_energy_ratio_mean"] = result["left_branch_energy_ratio_mean"]
        result["wrong_branch_energy_ratio_mean"] = result["right_branch_energy_ratio_mean"]
    elif target_side == "right":
        result["target_branch_energy_ratio_mean"] = result["right_branch_energy_ratio_mean"]
        result["wrong_branch_energy_ratio_mean"] = result["left_branch_energy_ratio_mean"]
    else:
        result["target_branch_energy_ratio_mean"] = float("nan")
        result["wrong_branch_energy_ratio_mean"] = float("nan")

    if mixed and task_cpu is not None and targets_cpu.numel() > 0:
        for task_id, key in [(0, "mnist"), (1, "fashion")]:
            mask = task_cpu == task_id
            if mask.any():
                result[f"{key}_acc"] = float((preds_cpu[mask] == targets_cpu[mask]).float().mean().item())
        if accum["route_acc"]:
            route_values = torch.cat([v.view(-1) for v in accum["route_acc"]])
            for task_id, key in [(0, "mnist"), (1, "fashion")]:
                mask = task_cpu == task_id
                if mask.any():
                    result[f"{key}_route_acc"] = float(route_values[mask].float().mean().item())

    return result


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: Dict, extra: Dict) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "extra": extra,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_confusion_matrix(targets: Iterable[int], preds: Iterable[int], num_classes: int, path: Path) -> None:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(targets, preds):
        matrix[int(true), int(pred)] += 1
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def save_detector_layouts(model: OpticalMoEClassifier, run_dir: Path) -> None:
    for side, name in [("left", "detector_layout_left.png"), ("right", "detector_layout_right.png"), ("paired", "detector_layout_paired.png")]:
        masks = model.detector.get_masks(side).detach().cpu().sum(dim=0).numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(masks, cmap="magma")
        plt.title(side)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(run_dir / name, dpi=100)
        plt.close()


def save_phase_layers(model: OpticalMoEClassifier, run_dir: Path, suffix: str) -> None:
    for side in ["left", "right"]:
        cols = min(model.num_layers, 5)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
        axes = np.array(axes).reshape(-1)
        for idx, layer in enumerate(model.expert_layers):
            phase = layer.get_phase_wrapped(side).detach().cpu().numpy()
            axes[idx].imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
            axes[idx].set_title(f"{side} L{idx + 1}")
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(run_dir / "phases" / f"{side}_phase_layers_{suffix}.png", dpi=100)
        plt.close(fig)

    prompt_phase = torch.remainder(model.get_prompt_phase().detach().cpu(), 2.0 * np.pi).numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(prompt_phase[::2, ::2], cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    plt.title("Prompt phase")
    plt.axis("off")
    plt.colorbar(fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(run_dir / "phases" / f"prompt_phase_{suffix}.png", dpi=100)
    plt.close()

    detilt = torch.remainder(model.entrance_detilt_phase.detach().cpu(), 2.0 * np.pi).numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(detilt[::2, ::2], cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    plt.title("Entrance de-tilt phase")
    plt.axis("off")
    plt.colorbar(fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(run_dir / "phases" / "detilt_phase.png", dpi=100)
    plt.close()


def _save_intensity_image(value: torch.Tensor, path: Path, title: str) -> None:
    if torch.is_complex(value):
        image = torch.abs(value) ** 2
    else:
        image = value.float()
    if image.ndim == 3:
        image = image[0]
    if image.ndim == 2:
        data = image.detach().cpu().numpy()
    else:
        return
    data = np.log10(data / (data.max() + 1e-12) + 1e-8)
    plt.figure(figsize=(8, 4))
    plt.imshow(data[::2, ::2], cmap="inferno")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=100)
    plt.close()


def collect_class_samples(loader, num_classes: int, samples_per_class: int = 1, max_classes: int = 10):
    wanted_classes = set(range(min(int(num_classes), int(max_classes))))
    buckets = {class_idx: [] for class_idx in wanted_classes}
    for batch in loader:
        images, targets, task_ids = batch if len(batch) == 3 else (batch[0], batch[1], None)
        for idx in range(targets.shape[0]):
            label = int(targets[idx])
            if label not in buckets or len(buckets[label]) >= samples_per_class:
                continue
            task_value = int(task_ids[idx]) if task_ids is not None else -1
            buckets[label].append((images[idx].clone(), int(label), task_value))
        if all(len(values) >= samples_per_class for values in buckets.values()):
            break

    samples = []
    for class_idx in sorted(buckets):
        samples.extend(buckets[class_idx])
    if not samples:
        return None
    images = torch.stack([item[0] for item in samples], dim=0)
    labels = torch.tensor([item[1] for item in samples], dtype=torch.long)
    task_ids = torch.tensor([item[2] for item in samples], dtype=torch.long)
    return images, labels, task_ids


def routing_side_for_sample(
    routing_mode: str,
    current_task: Optional[str],
    task_id: Optional[int],
    default_side: Optional[str],
) -> Optional[str]:
    if routing_mode == "task_aware" and task_id is not None and task_id >= 0:
        return "left" if int(task_id) == 0 else "right"
    if routing_mode == "fixed_task":
        return task_to_side(current_task)
    return default_side


@torch.no_grad()
def save_visualization_samples(
    model,
    batch,
    run_dir: Path,
    device: torch.device,
    routing_mode: str,
    current_task: Optional[str],
    max_light_field_samples: int = 4,
) -> None:
    if batch is None:
        return
    images, labels, task_ids = batch
    images = images.to(device)
    labels = labels.to(device)
    task_ids = task_ids.to(device)

    original_side = model.target_side
    logits_list = []
    left_energy_list = []
    right_energy_list = []
    overview_items = []

    for idx in range(images.shape[0]):
        side = routing_side_for_sample(
            routing_mode,
            current_task,
            int(task_ids[idx].item()) if task_ids is not None else None,
            original_side,
        )
        if side in {"left", "right"} and hasattr(model, "set_fixed_routing_side"):
            model.set_fixed_routing_side(side)
        logits, intermediates = model(images[idx : idx + 1], return_intermediates=True)
        pred = int(torch.argmax(logits, dim=1).item())
        logits_list.append(logits.detach().cpu())
        left_energy_list.append(intermediates["detector_energies_left_local_norm"][0].detach().cpu())
        right_energy_list.append(intermediates["detector_energies_right_local_norm"][0].detach().cpu())
        overview_items.append((idx, int(labels[idx].item()), int(task_ids[idx].item()), side, pred))

        if idx < int(max_light_field_samples):
            sample_dir = run_dir / "light_fields" / f"sample_{idx:03d}_label_{int(labels[idx])}_task_{int(task_ids[idx])}_{side}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_intermediate_images(model, intermediates, sample_dir)

    if original_side in {"left", "right"} and hasattr(model, "set_fixed_routing_side"):
        model.set_fixed_routing_side(original_side)

    save_sample_overview(images.detach().cpu(), overview_items, run_dir / "sample_outputs" / "class_samples_overview.png")
    save_detector_energy_samples(
        left_energy_list,
        right_energy_list,
        overview_items,
        run_dir / "detector_energies" / "detector_energy_left_right_bars_samples.png",
    )


def _save_intermediate_images(model, intermediates: Dict, sample_dir: Path) -> None:
    key_order = [
        ("00_input.png", "padded_input_on_canvas"),
        ("01_after_input_to_prompt.png", "after_input_to_prompt"),
        ("03_after_prompt.png", "after_prompt"),
        ("04_after_prompt_to_first_layer.png", "after_prompt_to_first_layer"),
        ("05_before_detilt.png", "before_entrance_detilt"),
        ("06_after_detilt.png", "after_entrance_detilt"),
    ]
    for idx in range(1, model.num_layers + 1):
        key_order.append((f"{6 + 2 * idx - 1:02d}_after_layer{idx}_modulation.png", f"after_layer_{idx}_modulation"))
        key_order.append((f"{6 + 2 * idx:02d}_after_layer{idx}_propagation.png", f"after_layer_{idx}_propagation"))
    key_order.append(("detector_plane.png", "detector_field"))
    for file_name, key in key_order:
        if key in intermediates:
            _save_intensity_image(intermediates[key], sample_dir / file_name, key)

    prompt = torch.remainder(intermediates["prompt_phase"].detach().cpu(), 2.0 * np.pi).numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(prompt[::2, ::2], cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    plt.title("prompt phase")
    plt.axis("off")
    plt.colorbar(fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(sample_dir / "02_prompt_phase.png", dpi=100)
    plt.close()

    left = intermediates["detector_energies_left_local_norm"][0].detach().cpu().numpy()
    right = intermediates["detector_energies_right_local_norm"][0].detach().cpu().numpy()
    x = np.arange(left.shape[0])
    plt.figure(figsize=(8, 3))
    plt.bar(x - 0.18, left, width=0.36, label="left")
    plt.bar(x + 0.18, right, width=0.36, label="right")
    plt.legend()
    plt.xlabel("Class")
    plt.ylabel("Local normalized energy")
    plt.tight_layout()
    plt.savefig(sample_dir / "detector_energy_left_right_bar.png", dpi=120)
    plt.close()


def save_sample_overview(images: torch.Tensor, overview_items: List[Tuple], path: Path) -> None:
    count = images.shape[0]
    cols = min(5, max(1, count))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= count:
            ax.axis("off")
            continue
        _, label, task_id, side, pred = overview_items[ax_idx]
        image = images[ax_idx, 0].numpy() if images.ndim == 4 else images[ax_idx].numpy()
        task_name = "mnist" if task_id == 0 else "fashion" if task_id == 1 else "single"
        ax.imshow(image, cmap="gray")
        ax.set_title(f"y={label} pred={pred}\n{task_name}->{side}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close(fig)


def save_detector_energy_samples(
    left_energy_list: List[torch.Tensor],
    right_energy_list: List[torch.Tensor],
    overview_items: List[Tuple],
    path: Path,
) -> None:
    count = len(left_energy_list)
    cols = min(4, max(1, count))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)
    x = np.arange(left_energy_list[0].numel()) if left_energy_list else np.arange(10)
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= count:
            ax.axis("off")
            continue
        _, label, task_id, side, pred = overview_items[ax_idx]
        left = left_energy_list[ax_idx].numpy()
        right = right_energy_list[ax_idx].numpy()
        ax.bar(x - 0.18, left, width=0.36, label="left")
        ax.bar(x + 0.18, right, width=0.36, label="right")
        ax.set_title(f"y={label} pred={pred} task={task_id}->{side}", fontsize=8)
        ax.set_xticks(x)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close()


def get_fixed_batch(loader):
    for batch in loader:
        return batch
    return None


def write_summary_md(path: Path, summary: Dict) -> None:
    warnings = summary.get("warnings", [])
    lines = [
        "# OpticalMoE Run Summary",
        "",
        "## Geometry",
        "",
        f"- canvas: {summary['geometry']['layout']['canvas_shape']}",
        f"- left aperture: {summary['geometry']['layout']['left_aperture']}",
        f"- right aperture: {summary['geometry']['layout']['right_aperture']}",
        f"- prompt slope sign: {summary['geometry']['prompt_slope_sign']}",
        f"- grating period px: {summary['geometry']['steering']['grating_period_px']:.3f}",
        f"- steering angle deg: {summary['geometry']['steering']['theta_deg']:.3f}",
        f"- predicted drift per inter-layer px: {summary['geometry']['steering']['drift_per_inter_layer_px']:.1f}",
        "",
        "## Experiment",
        "",
        f"- mode: {summary['mode']}",
        f"- dataset: {summary['dataset']}",
        f"- target_side: {summary.get('target_side')}",
        f"- readout_mode: {summary['geometry']['readout_mode']}",
        f"- prompt_mode: {summary['geometry']['prompt_mode']}",
        f"- routing_mode: {summary.get('routing_mode')}",
        f"- current_task: {summary.get('current_task')}",
        f"- task_aware_routing: {summary.get('task_aware_routing')}",
        f"- freeze_policy: {summary.get('freeze_policy')}",
        f"- loaded checkpoints: {summary.get('loaded_checkpoints')}",
        "",
        "## Results",
        "",
        f"- final test accuracy: {summary.get('final_test_acc')}",
        f"- final test loss: {summary.get('final_test_loss')}",
        f"- best validation accuracy: {summary.get('best_val_acc')}",
        f"- branch energy ratios: {summary.get('branch_energy_ratios')}",
        f"- migration judgment: {summary.get('migration_judgment')}",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- none")
    path.write_text("\n".join(lines), encoding="utf-8")


def migration_judgment(baseline_acc: Optional[float], bank_acc: float) -> Optional[str]:
    if baseline_acc is None:
        return None
    drop = float(baseline_acc) - float(bank_acc)
    if drop < 0.03:
        return f"migration looks stable; acc_drop={drop:.4f}"
    if drop < 0.10:
        return f"minor/moderate drop; compensation-only finetune recommended; acc_drop={drop:.4f}"
    return f"large drop; train in bank geometry or finetune all_side; acc_drop={drop:.4f}"


def main():
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    experiment_cfg = config.get("experiment", {})
    checkpoint_cfg = config.get("checkpoints", {})
    mode = config.get("mode", experiment_cfg.get("mode", "eval"))
    run_name = config.get("run_name", experiment_cfg.get("run_name", "optical_moe_run"))
    seed = int(config.get("seed", 7))
    set_seed(seed)

    run_dir = create_moe_run_dir(run_name)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    write_yaml(run_dir / "resolved_config.yaml", config)

    (loaders, mixed) = create_loaders(config, seed)
    train_loader, val_loader, test_loader, num_classes = loaders

    device_cfg = config.get("device", "auto")
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg if device_cfg != "auto" else "cpu")
    model = build_model(config, num_classes=num_classes, mode=mode).to(device)

    warnings = []
    loaded_checkpoints = []
    start_epoch = 1
    resume_payload = None
    moe_ckpt = checkpoint_cfg.get("moe_ckpt")
    left_ckpt = checkpoint_cfg.get("left_ckpt")
    right_ckpt = checkpoint_cfg.get("right_ckpt")
    left_moe_ckpt = checkpoint_cfg.get("left_moe_ckpt")
    right_moe_ckpt = checkpoint_cfg.get("right_moe_ckpt")
    left_config = checkpoint_cfg.get("left_config")
    right_config = checkpoint_cfg.get("right_config")
    strict_geometry_check = bool(experiment_cfg.get("strict_geometry_check", False))
    resume_optimizer = bool(experiment_cfg.get("resume_optimizer", False))
    baseline_acc = experiment_cfg.get("baseline_acc")

    if moe_ckpt:
        resume_payload = torch.load(moe_ckpt, map_location=device)
        model.load_state_dict(resume_payload["model_state_dict"])
        loaded_checkpoints.append(
            {
                "type": "full_optical_moe",
                "source_checkpoint_path": moe_ckpt,
                "epoch": resume_payload.get("epoch"),
            }
        )
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        print(f"loaded full OpticalMoE checkpoint: {moe_ckpt}")

    if left_ckpt:
        loaded_checkpoints.append(
            model.load_single_expert_checkpoint_into_side(
                left_ckpt,
                side="left",
                old_config_path=left_config,
                strict_geometry_check=strict_geometry_check,
            )
        )
    if left_moe_ckpt:
        loaded_checkpoints.append(
            model.load_moe_checkpoint_side_into_side(
                left_moe_ckpt,
                source_side="left",
                target_side="left",
            )
        )
    if right_ckpt:
        loaded_checkpoints.append(
            model.load_single_expert_checkpoint_into_side(
                right_ckpt,
                side="right",
                old_config_path=right_config,
                strict_geometry_check=strict_geometry_check,
            )
        )
    if right_moe_ckpt:
        loaded_checkpoints.append(
            model.load_moe_checkpoint_side_into_side(
                right_moe_ckpt,
                source_side="right",
                target_side="right",
            )
        )
    for item in loaded_checkpoints:
        warnings.extend(item.get("warnings", []))

    default_policy = {
        "eval": "frozen",
        "finetune": "compensation_only",
        "train_scratch": "all_side",
        "prompt_train": "compensation_only",
    }[mode]
    freeze_policy = config.get("training", {}).get("freeze_policy", default_policy)
    if freeze_policy in {None, "auto"}:
        freeze_policy = default_policy
    target_side = config.get("model", {}).get("target_side")
    warnings.extend(apply_freeze_policy(model, freeze_policy, target_side))

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = None
    if mode in {"finetune", "train_scratch", "prompt_train"}:
        if not trainable_params:
            warnings.append("No trainable parameters after applying freeze_policy; running evaluation only.")
        else:
            opt_cfg = config.get("optimizer", {})
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=float(opt_cfg.get("lr", 1e-3)),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            )
            if resume_optimizer and resume_payload is not None and "optimizer_state_dict" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    print(f"device: {device}")
    print(f"mode: {mode}")
    print(f"dataset: {config['dataset'].get('name')}")
    print(f"target_side: {target_side}")
    print(f"readout_mode: {model.readout_mode}")
    print(f"prompt_mode: {model.prompt_mode}")
    print(f"trainable parameters: {sum(p.numel() for p in trainable_params)}")

    save_detector_layouts(model, run_dir)
    vis_cfg = config.get("visualization", {})
    fixed_batch = collect_class_samples(
        test_loader,
        num_classes=num_classes,
        samples_per_class=int(vis_cfg.get("num_samples_per_class", 1)),
        max_classes=int(vis_cfg.get("max_classes", num_classes)),
    )
    criterion = nn.CrossEntropyLoss()
    metrics_path = run_dir / "metrics.csv"
    init_metrics_csv(metrics_path)
    evaluation_cfg = config.get("evaluation", {})
    routing_mode = evaluation_cfg.get("routing_mode", None)
    if routing_mode is None:
        routing_mode = "task_aware" if bool(evaluation_cfg.get("task_aware_routing", False)) else "model_default"
    current_task = evaluation_cfg.get("current_task", None)

    best_val_acc = -1.0
    best_epoch = 0
    final_test = None

    if mode == "eval" or optimizer is None:
        val_metrics = evaluate_moe(model, val_loader, device, criterion, target_side, mixed, routing_mode, current_task)
        test_metrics = evaluate_moe(model, test_loader, device, criterion, target_side, mixed, routing_mode, current_task)
        row = {"epoch": 0, "train_loss": "", "train_acc": "", "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"], "test_loss": test_metrics["loss"], "test_acc": test_metrics["acc"], "lr": ""}
        row.update({key: test_metrics.get(key, "") for key in METRIC_FIELDS})
        row["epoch"] = 0
        row["val_loss"] = val_metrics["loss"]
        row["val_acc"] = val_metrics["acc"]
        row["test_loss"] = test_metrics["loss"]
        row["test_acc"] = test_metrics["acc"]
        append_metrics_csv(metrics_path, row)
        final_test = test_metrics
        best_val_acc = val_metrics["acc"]
        save_checkpoint(run_dir / "last.pt", model, optimizer, 0, row, {"loaded_checkpoints": loaded_checkpoints})
        save_checkpoint(run_dir / "best.pt", model, optimizer, 0, row, {"loaded_checkpoints": loaded_checkpoints})
        print(f"eval | val_acc {val_metrics['acc']:.4f} | test_acc {test_metrics['acc']:.4f}")
    else:
        epochs = int(config.get("training", {}).get("epochs", 1))
        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion,
                print_freq=int(experiment_cfg.get("print_freq", 50)),
            )
            val_metrics = evaluate_moe(model, val_loader, device, criterion, target_side, mixed, routing_mode, current_task)
            test_metrics = evaluate_moe(model, test_loader, device, criterion, target_side, mixed, routing_mode, current_task)
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "lr": lr,
            }
            row.update({key: test_metrics.get(key, row.get(key, "")) for key in METRIC_FIELDS})
            row["epoch"] = epoch
            row["train_loss"] = train_loss
            row["train_acc"] = train_acc
            row["val_loss"] = val_metrics["loss"]
            row["val_acc"] = val_metrics["acc"]
            row["test_loss"] = test_metrics["loss"]
            row["test_acc"] = test_metrics["acc"]
            row["lr"] = lr
            append_metrics_csv(metrics_path, row)
            print(
                f"epoch {epoch:04d} | train_acc {train_acc:.4f} | "
                f"val_acc {val_metrics['acc']:.4f} | test_acc {test_metrics['acc']:.4f}"
            )
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, row, {"loaded_checkpoints": loaded_checkpoints})
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, row, {"loaded_checkpoints": loaded_checkpoints})
            final_test = test_metrics

    save_phase_layers(model, run_dir, "epoch_final")
    save_visualization_samples(
        model,
        fixed_batch,
        run_dir,
        device,
        routing_mode=routing_mode,
        current_task=current_task,
        max_light_field_samples=int(vis_cfg.get("max_light_field_samples", 4)),
    )
    if final_test is not None:
        save_confusion_matrix(final_test["targets"], final_test["preds"], num_classes, run_dir / "confusion_matrix.png")

    branch_ratios = None
    if final_test is not None:
        branch_ratios = {
            "left": final_test.get("left_branch_energy_ratio_mean"),
            "right": final_test.get("right_branch_energy_ratio_mean"),
            "outside": final_test.get("outside_energy_ratio_mean"),
        }
        if final_test.get("outside_energy_ratio_mean", 0.0) > 0.2:
            warnings.append("high outside energy ratio; check aperture alignment or prompt steering.")
        edge = final_test.get("edge_energy_ratio_mean")
        if isinstance(edge, float) and edge == edge and edge > 0.05:
            warnings.append("high edge energy ratio; possible FFT wrap-around or boundary contamination.")
        wrong = final_test.get("wrong_branch_energy_ratio_mean")
        target = final_test.get("target_branch_energy_ratio_mean")
        if isinstance(wrong, float) and isinstance(target, float) and target == target and wrong == wrong and wrong > 0.25 * max(target, 1e-8):
            warnings.append("wrong branch energy is high relative to target branch.")

    judgment = migration_judgment(baseline_acc, final_test["acc"] if final_test else 0.0)
    summary = {
        "run_name": run_name,
        "mode": mode,
        "dataset": config["dataset"].get("name"),
        "target_side": target_side,
        "freeze_policy": freeze_policy,
        "geometry": model.geometry_summary(),
        "loaded_checkpoints": loaded_checkpoints,
        "final_test_acc": final_test["acc"] if final_test else None,
        "final_test_loss": final_test["loss"] if final_test else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "branch_energy_ratios": branch_ratios,
        "routing_mode": routing_mode,
        "current_task": current_task,
        "task_aware_routing": routing_mode == "task_aware",
        "baseline_acc": baseline_acc,
        "migration_judgment": judgment,
        "warnings": warnings,
    }
    save_json(run_dir / "summary.json", summary)
    write_summary_md(run_dir / "summary.md", summary)
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
