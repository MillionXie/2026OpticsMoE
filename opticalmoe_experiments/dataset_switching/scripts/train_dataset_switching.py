import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.data.datasets import create_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.optics.dataset_switching_moe import (
    DatasetSwitchingASGlobalRouterMoEClassifier,
    DatasetSwitchingSharedD2NNClassifier,
)
from common.optics.expert_layout import ExpertLayout
from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import load_checkpoint, save_checkpoint
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
from common.training.task_heads import resolve_dataset_switching_task_heads
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import make_run_dir, write_text
from common.utils.git_info import collect_environment, collect_git_info
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_training_curves
from common.visualization.lightfield_viz import save_image, save_light_fields
from common.visualization.mask_viz import save_expert_phase_layers
from common.visualization.prompt_viz import save_prompt_maps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def _task_name(task_cfg: Dict) -> str:
    return str(task_cfg["name"]).lower()


def task_configs(config: Dict) -> List[Dict]:
    return list(config.get("training", {}).get("multitask", {}).get("tasks", []))


def create_task_loaders(config: Dict, seed: int, smoke_test: bool):
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    task_num_classes, class_names = {}, {}
    loader_summaries = {}
    for index, task in enumerate(task_configs(config)):
        name = _task_name(task)
        dataset_cfg = dict(task.get("dataset", {}))
        if smoke_test:
            dataset_cfg["smoke_test"] = True
            dataset_cfg.setdefault("smoke_train_size", 16)
            dataset_cfg.setdefault("smoke_test_size", 8)
            apply_smoke_loader_overrides(dataset_cfg)
        bundle = create_dataloaders(dataset_cfg, seed=seed + index)
        train_loaders[name] = bundle.train_loader
        val_loaders[name] = bundle.val_loader
        test_loaders[name] = bundle.test_loader
        task_num_classes[name] = int(bundle.num_classes)
        class_names[name] = bundle.class_names
        loader_summaries[name] = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, dataset_cfg)
    return train_loaders, val_loaders, test_loaders, task_num_classes, class_names, loader_summaries


def _layout_from_config(config: Dict) -> ExpertLayout:
    layout_cfg = config.get("layout", {})
    return ExpertLayout(
        num_experts=int(config.get("model", {}).get("num_experts", 9)),
        canvas_size=int(layout_cfg.get("canvas_height", layout_cfg.get("canvas_size", 1000))),
        input_size=int(layout_cfg.get("input_size", 134)),
        expert_size=int(layout_cfg.get("expert_size", 134)),
        expert_pitch=int(layout_cfg.get("expert_pitch", 200)),
        padding=int(layout_cfg.get("padding", 200)),
        prompt_aperture_size=int(layout_cfg.get("prompt_aperture_size", 600)),
    )


def task_head_configs(config: Dict, task_names: Sequence[str]) -> Dict[str, Dict]:
    return resolve_dataset_switching_task_heads(config, task_names)


def build_model(config: Dict, task_names: Sequence[str], task_num_classes: Dict[str, int]):
    model_cfg = config.get("model", {})
    optics_cfg = config.get("optics", {})
    prompt_cfg = config.get("prompt", {})
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
    dropout = phase_dropout_settings(config)
    model_type = str(model_cfg.get("type", "learnable_route_moe")).lower()
    distances = optics_cfg.get("distances_m", {})
    if model_type in {"learnable_route_moe", "fixed_route_moe"}:
        fixed = model_type == "fixed_route_moe"
        layout = _layout_from_config(config)
        return DatasetSwitchingASGlobalRouterMoEClassifier(
            task_names=task_names,
            task_num_classes=task_num_classes,
            task_head_configs=task_head_configs(config, task_names),
            layout=layout,
            wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
            num_layers=int(optics_cfg.get("num_layers", model_cfg.get("num_layers", 5))),
            distances_m=distances,
            focal_length_m=float(optics_cfg.get("focal_length_m", 0.10)),
            aperture_mode=optics_cfg.get("aperture_mode", "hard"),
            phase_param=optics_cfg.get("phase_param", "unconstrained"),
            expert_phase_init=optics_cfg.get("expert_phase_init", "identity"),
            expert_init_std=float(optics_cfg.get("expert_init_std", 0.02)),
            global_fc_phase_init=optics_cfg.get("global_fc_phase_init", "identity"),
            global_fc_init_std=float(optics_cfg.get("global_fc_init_std", 0.02)),
            global_fc_phase_mode=optics_cfg.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics_cfg.get("global_fc_phase_size", layout.active_window_size),
            global_fc_padding_mode=optics_cfg.get("global_fc_padding_mode", "transparent"),
            prompt_mode=prompt_cfg.get("mode", "complex_order_router"),
            prompt_amplitude_init_logits=float(prompt_cfg.get("amplitude_init_logits", 2.0)),
            train_prompt_amplitudes=bool(prompt_cfg.get("train_amplitudes", not fixed)) and not fixed,
            train_prompt_phase_biases=bool(prompt_cfg.get("train_phase_biases", not fixed)) and not fixed,
            grating_scale=float(prompt_cfg.get("grating_scale", 1.0)),
            grating_sign_x=float(prompt_cfg.get("grating_sign_x", 1.0)),
            grating_sign_y=float(prompt_cfg.get("grating_sign_y", 1.0)),
            prompt_normalize=prompt_cfg.get("normalize", "sum_amplitude"),
            detector_size=int(detector_cfg.get("detector_size", 32)),
            detector_layout=detector_cfg.get("layout", "grid"),
            normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
            readout_type=readout_cfg.get("type", "mlp"),
            logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
            readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
            readout_activation=readout_cfg.get("activation", "gelu"),
            readout_input_norm=readout_cfg.get("input_norm", "layernorm"),
            readout_norm_affine=bool(readout_cfg.get("norm_affine", True)),
            readout_hidden_layers=int(readout_cfg.get("hidden_layers", 1)),
            readout_dropout=float(readout_cfg.get("dropout", 0.1)),
            expert_phase_dropout_mode=dropout["expert_mode"],
            expert_phase_dropout_p=dropout["expert_p"],
            global_fc_phase_dropout_mode=dropout["global_fc_mode"],
            global_fc_phase_dropout_p=dropout["global_fc_p"],
            phase_dropout_block_size=dropout["block_size"],
            phase_dropout_batch_shared=dropout["batch_shared"],
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )
    if model_type == "shared_d2nn":
        return DatasetSwitchingSharedD2NNClassifier(
            task_names=task_names,
            task_num_classes=task_num_classes,
            task_head_configs=task_head_configs(config, task_names),
            canvas_size=int(model_cfg.get("canvas_size", config.get("layout", {}).get("canvas_height", 1000))),
            input_size=int(model_cfg.get("input_size", config.get("layout", {}).get("input_size", 134))),
            d2nn_phase_grid_size=int(model_cfg.get("d2nn_phase_grid_size", 402)),
            num_layers=int(model_cfg.get("d2nn_num_layers", optics_cfg.get("num_layers", 5))),
            wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
            distances_m=distances,
            phase_param=optics_cfg.get("phase_param", "unconstrained"),
            phase_init=optics_cfg.get("expert_phase_init", "identity"),
            init_std=float(optics_cfg.get("expert_init_std", 0.02)),
            global_fc_phase_mode=optics_cfg.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics_cfg.get("global_fc_phase_size"),
            global_fc_padding_mode=optics_cfg.get("global_fc_padding_mode", "transparent"),
            detector_size=int(detector_cfg.get("detector_size", 32)),
            detector_layout=detector_cfg.get("layout", "grid"),
            normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
            readout_type=readout_cfg.get("type", "mlp"),
            logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
            readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
            readout_activation=readout_cfg.get("activation", "gelu"),
            readout_input_norm=readout_cfg.get("input_norm", "layernorm"),
            readout_norm_affine=bool(readout_cfg.get("norm_affine", True)),
            readout_hidden_layers=int(readout_cfg.get("hidden_layers", 1)),
            readout_dropout=float(readout_cfg.get("dropout", 0.1)),
            phase_dropout_mode=dropout["expert_mode"],
            phase_dropout_p=dropout["expert_p"],
            global_fc_phase_dropout_mode=dropout["global_fc_mode"],
            global_fc_phase_dropout_p=dropout["global_fc_p"],
            phase_dropout_block_size=dropout["block_size"],
            phase_dropout_batch_shared=dropout["batch_shared"],
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )
    raise ValueError(f"Unsupported dataset-switching model.type: {model_type}")


def build_optimizer(model, config: Dict):
    cfg = config.get("optimizer", {})
    opt_type = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.9)))
    raise ValueError(f"Unsupported optimizer.type: {opt_type}")


def fixed_batch(loader, device, max_items=4):
    images, targets = next(iter(loader))
    return images[:max_items].to(device), targets[:max_items].to(device)


def _next_batch(iterators, loaders, task_name):
    try:
        return next(iterators[task_name])
    except StopIteration:
        iterators[task_name] = iter(loaders[task_name])
        return next(iterators[task_name])


def train_epoch(model, loaders, task_names, loss_weights, criterion, optimizer, device, steps, print_freq=50):
    model.train()
    iterators = {name: iter(loaders[name]) for name in task_names}
    weight_sum = sum(float(loss_weights.get(name, 1.0)) for name in task_names)
    loss_sum = {name: 0.0 for name in task_names}
    correct = {name: 0 for name in task_names}
    seen = {name: 0 for name in task_names}
    total_loss = 0.0
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for task_name in task_names:
            images, targets = _next_batch(iterators, loaders, task_name)[:2]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images, task_name=task_name)
            loss = criterion(logits, targets)
            weighted = loss * float(loss_weights.get(task_name, 1.0)) / max(weight_sum, 1e-8)
            weighted.backward()
            update_loss += float(weighted.item())
            batch = targets.numel()
            loss_sum[task_name] += float(loss.item()) * batch
            correct[task_name] += int((logits.argmax(dim=1) == targets).sum().item())
            seen[task_name] += batch
        optimizer.step()
        total_loss += update_loss
        if print_freq > 0 and ((step + 1) % int(print_freq) == 0 or step + 1 == steps):
            parts = [f"{name}: loss={loss_sum[name] / max(seen[name], 1):.4f}, acc={correct[name] / max(seen[name], 1):.4f}" for name in task_names]
            print(f"  update {step + 1}/{steps} | joint_loss={total_loss / max(step + 1, 1):.4f} | " + " | ".join(parts))
    result = {"joint_train_loss": total_loss / max(steps, 1)}
    total_seen = sum(seen.values())
    result["joint_train_acc"] = sum(correct.values()) / max(total_seen, 1)
    task_accs = []
    for name in task_names:
        result[f"{name}_train_loss"] = loss_sum[name] / max(seen[name], 1)
        result[f"{name}_train_acc"] = correct[name] / max(seen[name], 1)
        result[f"{name}_train_samples"] = seen[name]
        task_accs.append(result[f"{name}_train_acc"])
    result["macro_train_acc"] = sum(task_accs) / max(len(task_accs), 1)
    return result


@torch.no_grad()
def evaluate_task(model, loader, device, criterion, task_name, prompt_task_name=None, readout_task_name=None, max_batches=None):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images, task_name=task_name, prompt_task_name=prompt_task_name, readout_task_name=readout_task_name)
        if logits.shape[1] == int(targets.max().item()) + 1 or logits.shape[1] > int(targets.max().item()):
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.numel()
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += targets.numel()
    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1), "samples": seen}


def label_space_matched(task_num_classes: Dict[str, int], eval_dataset: str, readout_task: str) -> bool:
    return int(task_num_classes[eval_dataset]) == int(task_num_classes[readout_task])


@torch.no_grad()
def prompt_swap_evaluation(model, test_loaders, task_names, task_num_classes, device, criterion, max_batches=None):
    rows = []
    for eval_dataset in task_names:
        for prompt_task in task_names:
            for readout_task in task_names:
                matched = label_space_matched(task_num_classes, eval_dataset, readout_task)
                if matched:
                    metrics = evaluate_task(
                        model,
                        test_loaders[eval_dataset],
                        device,
                        criterion,
                        task_name=eval_dataset,
                        prompt_task_name=prompt_task,
                        readout_task_name=readout_task,
                        max_batches=max_batches,
                    )
                    acc, loss, samples = metrics["acc"], metrics["loss"], metrics["samples"]
                else:
                    acc, loss = "", ""
                    samples = sum(targets.numel() for index, (_images, targets) in enumerate(test_loaders[eval_dataset]) if max_batches is None or index < int(max_batches))
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "prompt_task": prompt_task,
                        "readout_task": readout_task,
                        "label_space_matched": matched,
                        "accuracy": acc,
                        "loss": loss,
                        "samples": samples,
                        "is_diagonal_prompt": eval_dataset == prompt_task,
                        "is_diagonal_readout": eval_dataset == readout_task,
                    }
                )
    return rows


def prompt_swap_summary(rows: List[Dict], task_names: Sequence[str]):
    summary = {}
    for task_name in task_names:
        diagonal = [r for r in rows if r["eval_dataset"] == task_name and r["prompt_task"] == task_name and r["readout_task"] == task_name]
        wrong = [r for r in rows if r["eval_dataset"] == task_name and r["readout_task"] == task_name and r["prompt_task"] != task_name and r["accuracy"] != ""]
        diag_acc = float(diagonal[0]["accuracy"]) if diagonal and diagonal[0]["accuracy"] != "" else ""
        wrong_acc = sum(float(r["accuracy"]) for r in wrong) / max(len(wrong), 1) if wrong else ""
        gap = diag_acc - wrong_acc if diag_acc != "" and wrong_acc != "" else ""
        summary[task_name] = {
            "diagonal_accuracy": diag_acc,
            "mean_wrong_prompt_accuracy": wrong_acc,
            "prompt_swap_gap": gap,
        }
    return summary


@torch.no_grad()
def collect_task_diagnostics(model, batch, device, task_name):
    images, targets = batch
    model.eval()
    logits, intermediates = model(images.to(device), task_name=task_name, return_intermediates=True)
    diagnostics = {
        "targets": targets.detach().cpu(),
        "predictions": logits.argmax(dim=1).detach().cpu(),
        "intermediates": intermediates,
    }
    if "prompt_amplitudes" in intermediates:
        diagnostics["prompt_amplitudes"] = intermediates["prompt_amplitudes"].detach().cpu()
        diagnostics["prompt_powers"] = intermediates["prompt_powers"].detach().cpu()
        diagnostics["normalized_prompt_powers"] = intermediates["normalized_prompt_powers"].detach().cpu()
    if "expert_energy_ratios" in intermediates:
        diagnostics["expert_energy_ratios"] = intermediates["expert_energy_ratios"].mean(dim=0).detach().cpu()
    if "outside_energy_ratio" in intermediates:
        diagnostics["outside_energy_ratio"] = float(intermediates["outside_energy_ratio"].mean().item())
    if "detector_energies" in intermediates:
        diagnostics["detector_energy_mean"] = intermediates["detector_energies"].mean(dim=0).detach().cpu()
    return diagnostics


def expert_labels(model):
    if hasattr(model, "layout"):
        return [ap.name for ap in model.layout.expert_apertures]
    return []


def save_epoch_artifacts(model, fixed_batches, run_dir: Path, epoch_name: str, task_names, device, class_names, enabled=True):
    if not enabled:
        return {}
    diagnostics = {}
    labels = expert_labels(model)
    for task_name in task_names:
        diag = collect_task_diagnostics(model, fixed_batches[task_name], device, task_name)
        diagnostics[task_name] = diag
        intermediates = diag["intermediates"]
        save_light_fields(intermediates, run_dir / "figures" / "light_fields" / epoch_name / task_name)
        save_prompt_maps(intermediates, run_dir / "figures" / "prompt" / epoch_name / task_name, expert_labels=labels or None)
        detector_dir = run_dir / "figures" / "detector_outputs" / epoch_name / task_name
        detector_dir.mkdir(parents=True, exist_ok=True)
        if "detector_field" in intermediates:
            field = intermediates["detector_field"][0] if intermediates["detector_field"].ndim >= 3 else intermediates["detector_field"]
            save_image(field, detector_dir / "detector_plane_sample_000.png", "detector plane")
        if "detector_energy_mean" in diag:
            _save_bar(diag["detector_energy_mean"], detector_dir / "detector_energy_mean_bar.png", "detector energy mean")
        samples_dir = run_dir / "figures" / "samples" / epoch_name / task_name
        samples_dir.mkdir(parents=True, exist_ok=True)
        payload = []
        names = class_names.get(task_name, [])
        for idx in range(min(len(diag["targets"]), 8)):
            target = int(diag["targets"][idx])
            pred = int(diag["predictions"][idx])
            payload.append(
                {
                    "sample_index": idx,
                    "true": target,
                    "pred": pred,
                    "true_name": names[target] if target < len(names) else str(target),
                    "pred_name": names[pred] if pred < len(names) else str(pred),
                }
            )
        save_json(payload, samples_dir / "sample_predictions.json")
    phase_dir = run_dir / "figures" / "phase_masks" / epoch_name
    save_expert_phase_layers(model, phase_dir)
    if (phase_dir / "expert_phase_layers.png").exists():
        (phase_dir / "expert_phase_layers.png").replace(phase_dir / "shared_expert_phase_layers.png")
    if (phase_dir / "global_fc_phase.png").exists():
        (phase_dir / "global_fc_phase.png").replace(phase_dir / "shared_global_fc_phase.png")
    return diagnostics


def _save_bar(values, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = torch.as_tensor(values).detach().cpu().float()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(values)), values.numpy())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_prompt_swap_plot(rows: List[Dict], path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = [
        row for row in rows
        if bool(row.get("label_space_matched")) and row.get("eval_dataset") == row.get("readout_task") and row.get("accuracy") != ""
    ]
    if not matched:
        return
    tasks = sorted({row["eval_dataset"] for row in matched})
    prompts = sorted({row["prompt_task"] for row in matched})
    values = torch.full((len(tasks), len(prompts)), float("nan"))
    for row in matched:
        values[tasks.index(row["eval_dataset"]), prompts.index(row["prompt_task"])] = float(row["accuracy"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Prompt swap accuracy")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_expert_usage_heatmap(rows: List[Dict], path: Path, value_key: str = "normalized_prompt_power"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    latest_epoch = max(int(row["epoch"]) for row in rows)
    latest = [row for row in rows if int(row["epoch"]) == latest_epoch and row.get(value_key, "") != ""]
    if not latest:
        return
    tasks = sorted({row["task_name"] for row in latest})
    experts = sorted({row["expert_id"] for row in latest})
    values = torch.zeros(len(tasks), len(experts))
    for row in latest:
        values[tasks.index(row["task_name"]), experts.index(row["expert_id"])] = float(row[value_key])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(values.numpy(), cmap="magma")
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(experts)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title(value_key)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def expert_usage_rows(run_id, epoch, model_type, diagnostics: Dict):
    rows = []
    for task_name, diag in diagnostics.items():
        if "prompt_amplitudes" not in diag:
            continue
        for idx, label in enumerate(expert_labels_from_count(len(diag["prompt_amplitudes"]))):
            rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "task_name": task_name,
                    "expert_id": label,
                    "model_type": model_type,
                    "prompt_amplitude": float(diag["prompt_amplitudes"][idx]),
                    "prompt_power": float(diag["prompt_powers"][idx]),
                    "normalized_prompt_power": float(diag["normalized_prompt_powers"][idx]),
                    "expert_entrance_energy_ratio": float(diag["expert_energy_ratios"][idx]) if "expert_energy_ratios" in diag else "",
                    "outside_energy_ratio": diag.get("outside_energy_ratio", ""),
                    "detector_energy_mean": float(diag["detector_energy_mean"].float().mean().item()) if "detector_energy_mean" in diag else "",
                }
            )
    return rows


def expert_labels_from_count(count: int):
    dim = int(round(math.sqrt(int(count))))
    return [f"E{r}{c}" for r in range(dim) for c in range(dim)]


def prompt_similarity_rows(run_id, epoch, diagnostics: Dict):
    rows = []
    tasks = list(diagnostics)
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1:]:
            da, db = diagnostics[task_a], diagnostics[task_b]
            if "normalized_prompt_powers" not in da or "normalized_prompt_powers" not in db:
                continue
            pa = da["normalized_prompt_powers"].float()
            pb = db["normalized_prompt_powers"].float()
            aa = da["prompt_amplitudes"].float()
            ab = db["prompt_amplitudes"].float()
            ia = da.get("intermediates", {})
            ib = db.get("intermediates", {})
            router_corr = ""
            total_corr = ""
            if "prompt_router_amplitude" in ia and "prompt_router_amplitude" in ib:
                ra = ia["prompt_router_amplitude"].detach().float().reshape(-1)
                rb = ib["prompt_router_amplitude"].detach().float().reshape(-1)
                router_corr = float(torch.nn.functional.cosine_similarity(ra, rb, dim=0).item())
            if "prompt_total_amplitude" in ia and "prompt_total_amplitude" in ib:
                ta = ia["prompt_total_amplitude"].detach().float().reshape(-1)
                tb = ib["prompt_total_amplitude"].detach().float().reshape(-1)
                total_corr = float(torch.nn.functional.cosine_similarity(ta, tb, dim=0).item())
            rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "task_a": task_a,
                    "task_b": task_b,
                    "amplitude_cosine": float(torch.nn.functional.cosine_similarity(aa, ab, dim=0).item()),
                    "normalized_power_cosine": float(torch.nn.functional.cosine_similarity(pa, pb, dim=0).item()),
                    "phase_bias_l2": "",
                    "complex_router_map_correlation": router_corr,
                    "prompt_total_field_correlation": total_corr,
                }
            )
    return rows


def optical_energy_rows(run_id, epoch, diagnostics: Dict):
    rows = []
    for task_name, diag in diagnostics.items():
        ints = diag.get("intermediates", {})
        for stage, key in [
            ("input", "input_amplitude"),
            ("after_input_to_prompt", "after_input_to_prompt"),
            ("after_prompt", "after_prompt"),
            ("expert_entrance_before_aperture", "expert_entrance_before_aperture"),
            ("expert_entrance_after_aperture", "expert_entrance_after_aperture"),
            ("after_expert_layer_1", "after_expert_layer_1"),
            ("after_expert_layer_last", "after_expert_layer_last"),
            ("after_global_fc", "after_global_fc"),
            ("detector_plane", "detector_field"),
        ]:
            if key not in ints:
                continue
            value = ints[key]
            if isinstance(value, list):
                value = value[-1]
            energy = torch.abs(value.to(torch.complex64)).square().sum(dim=(-2, -1)).mean()
            rows.append({"run_id": run_id, "epoch": epoch, "task_name": task_name, "stage": stage, "total_energy": float(energy.item())})
    return rows


def save_architecture_report(model, config, run_dir: Path):
    model_type = config.get("model", {}).get("type")
    global_fc = getattr(model, "global_fc", None)
    report = {
        "model_type": model_type,
        "task_names": list(model.task_names),
        "task_num_classes": model.task_num_classes,
        "shared_backbone": True,
        "task_specific_prompt": hasattr(model, "prompt_bank"),
        "task_specific_detector_readout": True,
        "task_head_configs": getattr(model, "task_head_configs", {}),
        "task_detector_configs": model.task_detector_configs() if hasattr(model, "task_detector_configs") else {},
        "task_readout_parameter_counts": model.task_readout_parameter_counts() if hasattr(model, "task_readout_parameter_counts") else {},
        "task_readout_shared": False,
        "task_readout_modules_are_independent": True,
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "phase_dropout_config": config.get("regularization", {}).get("phase_dropout", {}),
    }
    if global_fc is not None:
        report.update(
            {
                "global_fc_phase_mode": getattr(global_fc, "phase_mode", ""),
                "global_fc_phase_size": list(getattr(global_fc, "phase_size", [])),
                "global_fc_phase_region": global_fc.phase_region() if hasattr(global_fc, "phase_region") else "",
                "global_fc_padding_mode": getattr(global_fc, "padding_mode", ""),
                "global_fc_padding_is_trainable": bool(getattr(global_fc, "phase_mode", "") == "full_canvas"),
                "global_fc_parameter_count": int(global_fc.trainable_parameter_count()) if hasattr(global_fc, "trainable_parameter_count") else "",
            }
        )
    if hasattr(model, "layout"):
        report["layout"] = model.layout.to_dict()
        report["expert_union_size"] = model.layout.expert_union_size
        report["active_window_size"] = model.layout.active_window_size
        report["active_window_region"] = model.layout.active_window_aperture.to_dict()
        report["prompt_aperture_region"] = model.layout.prompt_aperture.to_dict()
        report["prompt_trainable_type"] = "channel_amplitude_and_phase_bias"
        report["prompt_trainable_pixelwise"] = False
        report["prompt_fixed_lens_grating_buffers_are_not_counted_as_parameters"] = True
        report["prompt_channel_table"] = model.prompt_bank.channel_table()
    save_json(report, run_dir / "architecture_report.json")
    lines = [
        "# Dataset Switching Architecture",
        "",
        f"- model_type: {model_type}",
        f"- tasks: {', '.join(model.task_names)}",
        f"- task_num_classes: {model.task_num_classes}",
        "- shared optical expert bank / propagation backbone / global FC phase: true",
        f"- task-specific optical prompt: {str(hasattr(model, 'prompt_bank')).lower()}",
        "- task-specific detector/readout heads: true",
        "- task readout modules are independent: true",
        f"- task head configs: {report['task_head_configs']}",
        f"- task readout parameter counts: {report['task_readout_parameter_counts']}",
        "- expert entrance is produced by AngularSpectrumPropagator(prompt_to_expert), not FFT convolution.",
        f"- optical_parameter_count: {report['optical_parameter_count']}",
        f"- prompt_parameter_count: {report['prompt_parameter_count']}",
        f"- electronic_parameter_count: {report['electronic_parameter_count']}",
        f"- global_fc_phase_mode: {report.get('global_fc_phase_mode', '')}",
        f"- global_fc_parameter_count: {report.get('global_fc_parameter_count', '')}",
        f"- global_fc_padding_is_trainable: {report.get('global_fc_padding_is_trainable', '')}",
        f"- active_window_size: {report.get('active_window_size', '')}",
    ]
    write_text(run_dir / "architecture_report.md", "\n".join(lines) + "\n")
    return report


def rebuild_dataset_switching_tables(runs_dir: Path, out_dir: Path):
    keys = {
        "runs": "master_runs.csv",
        "epoch_metrics": "master_epoch_metrics.csv",
        "task_metrics": "master_task_metrics.csv",
        "final_metrics": "master_final_metrics.csv",
        "prompt_swap": "master_prompt_swap.csv",
        "expert_usage": "master_expert_usage.csv",
        "optical_energy": "master_optical_energy.csv",
        "prompt_similarity": "master_prompt_similarity.csv",
        "model_params": "master_model_params.csv",
        "independent_baseline": "master_independent_baseline.csv",
        "scaling_results": "master_scaling_results.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, filename in keys.items():
        rows = []
        for path in sorted(runs_dir.glob(f"*/summary_for_master/{key}_rows.json")):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.extend(payload)
        write_rows(out_dir / filename, rows)
        counts[key] = len(rows)
    return counts


def run_training(config, args):
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_name = config.get("experiment", {}).get("run_name", f"dataset_switching_{int(time.time())}")
    run_dir = make_run_dir(EXPERIMENT_ROOT, "dataset_switching", run_name)
    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    save_json(collect_git_info(REPO_ROOT), run_dir / "git_info.json")
    save_json(collect_environment(), run_dir / "environment.json")
    write_text(run_dir / "command.txt", " ".join(sys.argv))

    loader_result = create_task_loaders(config, seed, args.smoke_test)
    if len(loader_result) == 6:
        train_loaders, val_loaders, test_loaders, task_num_classes, class_names, loader_summaries = loader_result
    else:
        train_loaders, val_loaders, test_loaders, task_num_classes, class_names = loader_result
        loader_summaries = {
            name: loader_summary_from_loaders(train_loaders[name], val_loaders[name], test_loaders[name], {})
            for name in train_loaders
        }
    save_json(loader_summaries, run_dir / "loader_summary.json")
    task_names = list(train_loaders)
    model = build_model(config, task_names, task_num_classes).to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    phase_dropout = phase_dropout_settings(config)
    model.set_phase_dropout_active(False)
    arch = save_architecture_report(model, config, run_dir)
    config.setdefault("resolved", {})["task_head_configs"] = getattr(model, "task_head_configs", {})
    save_json(config, run_dir / "config_resolved.json")
    print(f"device: {device}")
    print(f"tasks: {task_names}, task classes: {task_num_classes}")
    print(f"model: {config.get('model', {}).get('type')}")
    print(f"Optimizer: {optimizer.__class__.__name__}, lr={optimizer.param_groups[0]['lr']}, weight_decay={optimizer.param_groups[0].get('weight_decay', 0.0)}")
    for name in task_names:
        print_loader_summary(loader_summaries[name], prefix=f"loader/{name}")

    multitask_cfg = config.get("training", {}).get("multitask", {})
    natural_steps = max(len(loader) for loader in train_loaders.values())
    steps_cfg = multitask_cfg.get("steps_per_epoch")
    steps = min(natural_steps, int(steps_cfg)) if steps_cfg is not None and int(steps_cfg) > 0 else natural_steps
    if args.smoke_test:
        steps = min(steps, 1)
    print(f"updates per epoch: {steps} (natural full-dataset value: {natural_steps})")
    for name in task_names:
        print(f"  {name}: train={len(train_loaders[name].dataset)} samples/{len(train_loaders[name])} batches, val={len(val_loaders[name].dataset)}, test={len(test_loaders[name].dataset)}, batch_size={train_loaders[name].batch_size}")

    fixed = {name: fixed_batch(val_loaders[name], device, int(config.get("visualization", {}).get("num_samples", 4))) for name in task_names}
    viz_enabled = bool(config.get("visualization", {}).get("enabled", True))
    diagnostics = save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", task_names, device, class_names, enabled=viz_enabled)
    metrics_rows, task_rows, usage_rows, prompt_sim_rows, opt_rows = [], [], [], [], []
    usage_rows.extend(expert_usage_rows(run_name, 0, config.get("model", {}).get("type"), diagnostics))
    prompt_sim_rows.extend(prompt_similarity_rows(run_name, 0, diagnostics))
    opt_rows.extend(optical_energy_rows(run_name, 0, diagnostics))

    loss_weights = {name: float(multitask_cfg.get("loss_weights", {}).get(name, 1.0)) for name in task_names}
    epochs = int(config.get("training", {}).get("epochs", 200))
    if args.smoke_test:
        epochs = int(args.epochs or 1)
    best = {"epoch": 0, "joint_val_acc": -1.0}
    run_start = time.perf_counter()
    max_val_batches = config.get("training", {}).get("evaluation", {}).get("max_val_batches")
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train_metrics = train_epoch(
            model,
            train_loaders,
            task_names,
            loss_weights,
            criterion,
            optimizer,
            device,
            steps,
            print_freq=int(multitask_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50))),
        )
        row = {
            "run_id": run_name,
            "epoch": epoch,
            "phase_dropout_active": active,
            "phase_dropout_mode": phase_dropout["mode"],
            "expert_phase_dropout_p": phase_dropout["expert_p"],
            "global_fc_phase_dropout_p": phase_dropout["global_fc_p"],
            "phase_dropout_block_size": phase_dropout["block_size"],
            **train_metrics,
        }
        joint_correct, joint_seen, joint_loss = 0.0, 0, 0.0
        for task_name in task_names:
            val = evaluate_task(model, val_loaders[task_name], device, criterion, task_name, max_batches=max_val_batches)
            row[f"{task_name}_val_loss"] = val["loss"]
            row[f"{task_name}_val_acc"] = val["acc"]
            row[f"{task_name}_val_samples"] = val["samples"]
            joint_correct += val["acc"] * val["samples"]
            joint_loss += val["loss"] * val["samples"]
            joint_seen += val["samples"]
            head = model.task_head_configs[task_name]
            task_rows.append(
                {
                    "run_id": run_name,
                    "epoch": epoch,
                    "task_name": task_name,
                    "train_loss": row[f"{task_name}_train_loss"],
                    "train_acc": row[f"{task_name}_train_acc"],
                    "val_loss": val["loss"],
                    "val_acc": val["acc"],
                    "val_samples": val["samples"],
                    "readout_type": head["readout_type"],
                    "hidden_dim": head["hidden_dim"],
                    "hidden_layers": head["hidden_layers"],
                    "activation": head["activation"],
                    "dropout": head["dropout"],
                }
            )
        row["joint_val_acc"] = joint_correct / max(joint_seen, 1)
        row["joint_val_loss"] = joint_loss / max(joint_seen, 1)
        row["macro_val_acc"] = sum(row[f"{task}_val_acc"] for task in task_names) / max(len(task_names), 1)
        row["macro_val_loss"] = sum(row[f"{task}_val_loss"] for task in task_names) / max(len(task_names), 1)
        row["epoch_time_sec"] = time.perf_counter() - epoch_start
        metrics_rows.append(row)
        diagnostics = {name: collect_task_diagnostics(model, fixed[name], device, name) for name in task_names}
        usage_rows.extend(expert_usage_rows(run_name, epoch, config.get("model", {}).get("type"), diagnostics))
        prompt_sim_rows.extend(prompt_similarity_rows(run_name, epoch, diagnostics))
        opt_rows.extend(optical_energy_rows(run_name, epoch, diagnostics))

        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if row["joint_val_acc"] > best["joint_val_acc"]:
            best = {"epoch": epoch, "joint_val_acc": row["joint_val_acc"], "row": row}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
        if viz_enabled and interval > 0 and epoch % interval == 0:
            save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", task_names, device, class_names, enabled=True)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
        write_rows(run_dir / "metrics" / "task_metrics.csv", task_rows)
        write_rows(run_dir / "diagnostics" / "expert_usage.csv", usage_rows)
        write_rows(run_dir / "diagnostics" / "task_prompt_amplitude_history.csv", usage_rows)
        write_rows(run_dir / "diagnostics" / "task_expert_energy_history.csv", usage_rows)
        write_rows(run_dir / "diagnostics" / "prompt_similarity.csv", prompt_sim_rows)
        write_rows(run_dir / "diagnostics" / "optical_energy_by_stage.csv", opt_rows)
        limit_note = f" | val_limited=max_val_batches={max_val_batches}" if max_val_batches is not None else ""
        print(
            f"epoch {epoch:03d} | joint_train={row['joint_train_acc']:.4f} "
            f"joint_val={row['joint_val_acc']:.4f} | macro_val={row['macro_val_acc']:.4f} "
            f"| phase_dropout={'on' if active else 'off'} | time={row['epoch_time_sec']:.1f}s{limit_note}"
        )
        task_width = max(len(name) for name in task_names)
        for task_name in task_names:
            print(
                f"  {task_name:<{task_width}} "
                f"train_loss={row[f'{task_name}_train_loss']:.4f} "
                f"train_acc={row[f'{task_name}_train_acc']:.4f} "
                f"val_loss={row[f'{task_name}_val_loss']:.4f} "
                f"val_acc={row[f'{task_name}_val_acc']:.4f}"
            )

    final_diag = save_epoch_artifacts(model, fixed, run_dir, "final_epoch", task_names, device, class_names, enabled=viz_enabled)
    swap_rows = prompt_swap_evaluation(model, test_loaders, task_names, task_num_classes, device, criterion, max_batches=max_test_batches)
    for row in swap_rows:
        row["run_id"] = run_name
        row["model_type"] = config.get("model", {}).get("type")
    swap_summary = prompt_swap_summary(swap_rows, task_names)
    write_rows(run_dir / "metrics" / "prompt_swap_matrix.csv", swap_rows)
    save_json(swap_summary, run_dir / "metrics" / "prompt_swap_summary.json")
    save_prompt_swap_plot(swap_rows, run_dir / "figures" / "prompt_swap_matrix.png")
    save_expert_usage_heatmap(usage_rows, run_dir / "figures" / "task_expert_usage_heatmap.png")
    global_fc = getattr(model, "global_fc", None)
    layout = getattr(model, "layout", None)
    fc_summary = {
        "global_fc_phase_size": getattr(global_fc, "phase_size", ""),
        "global_fc_parameter_count": int(global_fc.trainable_parameter_count()) if global_fc is not None and hasattr(global_fc, "trainable_parameter_count") else "",
        "global_fc_phase_mode": getattr(global_fc, "phase_mode", ""),
        "global_fc_padding_is_trainable": bool(getattr(global_fc, "phase_mode", "") == "full_canvas") if global_fc is not None else "",
        "active_window_size": getattr(layout, "active_window_size", ""),
        "active_window_region": getattr(layout, "active_window_aperture", None).to_dict() if getattr(layout, "active_window_aperture", None) else "",
        "expert_union_size": getattr(layout, "expert_union_size", ""),
    }
    final_task_rows = []
    for task_name in task_names:
        test = evaluate_task(model, test_loaders[task_name], device, criterion, task_name, max_batches=max_test_batches)
        gap = swap_summary.get(task_name, {}).get("prompt_swap_gap", "")
        final_task_rows.append(
            {
                "run_id": run_name,
                "model_type": config.get("model", {}).get("type"),
                "task_name": task_name,
                "dataset_name": task_name,
                "num_tasks": len(task_names),
                "num_experts": config.get("model", {}).get("num_experts", ""),
                "prompt_type": config.get("model", {}).get("prompt_type", ""),
                "routing_type": config.get("model", {}).get("routing_type", ""),
                "num_classes": task_num_classes[task_name],
                "best_epoch": best["epoch"],
                "best_val_acc": best["joint_val_acc"],
                "final_test_acc": test["acc"],
                "final_test_loss": test["loss"],
                "prompt_swap_gap": gap,
                "total_wall_time_sec": time.perf_counter() - run_start,
                "total_train_time_sec": sum(float(r.get("epoch_time_sec", 0.0)) for r in metrics_rows),
                "optical_parameter_count": int(model.optical_parameter_count()),
                "electronic_parameter_count": int(model.electronic_parameter_count()),
                "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                **fc_summary,
                "run_dir": str(run_dir),
            }
        )
    save_json(final_task_rows, run_dir / "metrics" / "final_test_metrics.json")
    if metrics_rows:
        curve_rows = [{"epoch": r["epoch"], "train_loss": r["joint_train_loss"], "val_loss": r["joint_val_loss"], "train_acc": r["joint_train_acc"], "val_acc": r["joint_val_acc"]} for r in metrics_rows]
        save_training_curves(curve_rows, run_dir / "figures" / "training_curves.png")
    run_row = {
        "run_id": run_name,
        "exp_family": "dataset_switching",
        "model_type": config.get("model", {}).get("type"),
        "tasks": ",".join(task_names),
        "num_tasks": len(task_names),
        "num_experts": config.get("model", {}).get("num_experts", ""),
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "best_joint_val_acc": best["joint_val_acc"],
        "total_wall_time_sec": time.perf_counter() - run_start,
        "loader_summary": loader_summaries,
    }
    model_params = {
        "run_id": run_name,
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "task_readout_parameter_counts": model.task_readout_parameter_counts() if hasattr(model, "task_readout_parameter_counts") else {},
        "task_readout_shared": False,
        "task_readout_modules_are_independent": True,
        **fc_summary,
    }
    summary = {
        "run_id": run_name,
        "task_names": task_names,
        "task_num_classes": task_num_classes,
        "architecture": arch,
        "phase_dropout": phase_dropout,
        "best": best,
        "prompt_swap_summary": swap_summary,
        "final_test_metrics": final_task_rows,
        "loader_summary": loader_summaries,
        "task_head_configs": getattr(model, "task_head_configs", {}),
        "task_detector_configs": model.task_detector_configs() if hasattr(model, "task_detector_configs") else {},
        "task_readout_parameter_counts": model.task_readout_parameter_counts() if hasattr(model, "task_readout_parameter_counts") else {},
        "task_readout_shared": False,
        "task_readout_modules_are_independent": True,
    }
    save_json(summary, run_dir / "summary.json")
    save_json(run_row, run_dir / "summary_for_master" / "runs_rows.json")
    save_json(metrics_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json(task_rows, run_dir / "summary_for_master" / "task_metrics_rows.json")
    save_json(final_task_rows, run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json(swap_rows, run_dir / "summary_for_master" / "prompt_swap_rows.json")
    save_json(usage_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    save_json(opt_rows, run_dir / "summary_for_master" / "optical_energy_rows.json")
    save_json(prompt_sim_rows, run_dir / "summary_for_master" / "prompt_similarity_rows.json")
    save_json([model_params], run_dir / "summary_for_master" / "model_params_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_dataset_switching_tables(EXPERIMENT_ROOT / "dataset_switching" / "runs", EXPERIMENT_ROOT / "dataset_switching" / "results")
    print(f"saved run outputs to: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    config = load_yaml(args.config)
    run_training(config, args)


if __name__ == "__main__":
    main()
