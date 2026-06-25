import argparse
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

from common.data.dsprites_multitask import create_same_input_multitask_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides
from common.optics.expert_layout import ExpertLayout
from common.optics.task_prompt_moe import SameInputSharedD2NNClassifier, SameInputTaskPromptMoEClassifier
from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import load_checkpoint, save_checkpoint
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
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


def _layout(config):
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


def build_model(config: Dict, task_names: Sequence[str], task_num_classes: Dict[str, int]):
    model_cfg = config.get("model", {})
    optics = config.get("optics", {})
    prompt = config.get("prompt", {})
    detector = config.get("detector", {})
    readout = config.get("readout", {})
    dropout = phase_dropout_settings(config)
    model_type = str(model_cfg.get("type", "learnable_route_moe")).lower()
    distances = optics.get("distances_m", {})
    if model_type in {"learnable_route_moe", "fixed_route_moe"}:
        fixed = model_type == "fixed_route_moe"
        return SameInputTaskPromptMoEClassifier(
            task_names=task_names,
            task_num_classes=task_num_classes,
            layout=_layout(config),
            wavelength_m=float(optics.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics.get("pixel_size_m", 8e-6)),
            num_layers=int(optics.get("num_layers", 5)),
            distances_m=distances,
            focal_length_m=float(optics.get("focal_length_m", 0.10)),
            aperture_mode=optics.get("aperture_mode", "hard"),
            phase_param=optics.get("phase_param", "unconstrained"),
            expert_phase_init=optics.get("expert_phase_init", "identity"),
            expert_init_std=float(optics.get("expert_init_std", 0.02)),
            global_fc_phase_init=optics.get("global_fc_phase_init", "identity"),
            global_fc_init_std=float(optics.get("global_fc_init_std", 0.02)),
            global_fc_phase_mode=optics.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics.get("global_fc_phase_size", _layout(config).active_window_size),
            global_fc_padding_mode=optics.get("global_fc_padding_mode", "transparent"),
            prompt_mode=prompt.get("mode", "complex_order_router"),
            prompt_amplitude_init_logits=float(prompt.get("amplitude_init_logits", 2.0)),
            train_prompt_amplitudes=bool(prompt.get("train_amplitudes", not fixed)) and not fixed,
            train_prompt_phase_biases=bool(prompt.get("train_phase_biases", not fixed)) and not fixed,
            grating_scale=float(prompt.get("grating_scale", 1.0)),
            grating_sign_x=float(prompt.get("grating_sign_x", 1.0)),
            grating_sign_y=float(prompt.get("grating_sign_y", 1.0)),
            prompt_normalize=prompt.get("normalize", "sum_amplitude"),
            detector_size=int(detector.get("detector_size", 32)),
            detector_layout=detector.get("layout", "grid"),
            normalize_detector_energy=bool(readout.get("normalize_detector_energy", True)),
            readout_type=readout.get("type", "mlp"),
            logit_scale=float(readout.get("logit_scale", 10.0)),
            readout_hidden_dim=int(readout.get("hidden_dim", 64)),
            readout_activation=readout.get("activation", "gelu"),
            readout_input_norm=readout.get("input_norm", "layernorm"),
            readout_norm_affine=bool(readout.get("norm_affine", True)),
            readout_hidden_layers=int(readout.get("hidden_layers", 1)),
            readout_dropout=float(readout.get("dropout", 0.1)),
            expert_phase_dropout_mode=dropout["expert_mode"],
            expert_phase_dropout_p=dropout["expert_p"],
            global_fc_phase_dropout_mode=dropout["global_fc_mode"],
            global_fc_phase_dropout_p=dropout["global_fc_p"],
            phase_dropout_block_size=dropout["block_size"],
            phase_dropout_batch_shared=dropout["batch_shared"],
            evanescent_mode=optics.get("evanescent_mode", "zero"),
        )
    if model_type == "shared_d2nn":
        return SameInputSharedD2NNClassifier(
            task_names=task_names,
            task_num_classes=task_num_classes,
            canvas_size=int(model_cfg.get("canvas_size", config.get("layout", {}).get("canvas_height", 1000))),
            input_size=int(model_cfg.get("input_size", config.get("layout", {}).get("input_size", 134))),
            d2nn_phase_grid_size=int(model_cfg.get("d2nn_phase_grid_size", 402)),
            num_layers=int(model_cfg.get("d2nn_num_layers", optics.get("num_layers", 5))),
            wavelength_m=float(optics.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics.get("pixel_size_m", 8e-6)),
            distances_m=distances,
            phase_param=optics.get("phase_param", "unconstrained"),
            phase_init=optics.get("expert_phase_init", "identity"),
            init_std=float(optics.get("expert_init_std", 0.02)),
            global_fc_phase_mode=optics.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics.get("global_fc_phase_size"),
            global_fc_padding_mode=optics.get("global_fc_padding_mode", "transparent"),
            detector_size=int(detector.get("detector_size", 32)),
            detector_layout=detector.get("layout", "grid"),
            normalize_detector_energy=bool(readout.get("normalize_detector_energy", True)),
            readout_type=readout.get("type", "mlp"),
            logit_scale=float(readout.get("logit_scale", 10.0)),
            readout_hidden_dim=int(readout.get("hidden_dim", 64)),
            readout_activation=readout.get("activation", "gelu"),
            readout_input_norm=readout.get("input_norm", "layernorm"),
            readout_norm_affine=bool(readout.get("norm_affine", True)),
            readout_hidden_layers=int(readout.get("hidden_layers", 1)),
            readout_dropout=float(readout.get("dropout", 0.1)),
            phase_dropout_mode=dropout["expert_mode"],
            phase_dropout_p=dropout["expert_p"],
            global_fc_phase_dropout_mode=dropout["global_fc_mode"],
            global_fc_phase_dropout_p=dropout["global_fc_p"],
            phase_dropout_block_size=dropout["block_size"],
            phase_dropout_batch_shared=dropout["batch_shared"],
            evanescent_mode=optics.get("evanescent_mode", "zero"),
        )
    raise ValueError(f"Unsupported same-input model.type: {model_type}")


def build_optimizer(model, config):
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


def labels_to_device(labels: Dict[str, torch.Tensor], device):
    return {key: value.to(device) for key, value in labels.items()}


def task_accuracy(logits, targets):
    return int((logits.argmax(dim=1) == targets).sum().item()), int(targets.numel())


def train_one_epoch(model, loader, task_names, loss_weights, criterion, optimizer, device, print_freq=50):
    model.train()
    task_loss_sum = {task: 0.0 for task in task_names}
    task_correct = {task: 0 for task in task_names}
    task_seen = {task: 0 for task in task_names}
    joint_loss_sum = 0.0
    weight_sum = sum(float(loss_weights.get(task, 1.0)) for task in task_names)
    steps = 0
    for images, labels in loader:
        steps += 1
        images = images.to(device, non_blocking=True)
        labels = labels_to_device(labels, device)
        optimizer.zero_grad(set_to_none=True)
        weighted_sum = 0.0
        for task in task_names:
            logits = model(images, task_name=task)
            loss = criterion(logits, labels[task])
            weighted_sum = weighted_sum + float(loss_weights.get(task, 1.0)) * loss
            correct, seen = task_accuracy(logits, labels[task])
            task_loss_sum[task] += float(loss.item()) * seen
            task_correct[task] += correct
            task_seen[task] += seen
        loss = weighted_sum / max(weight_sum, 1e-8)
        loss.backward()
        optimizer.step()
        joint_loss_sum += float(loss.item())
        if print_freq > 0 and (steps % int(print_freq) == 0):
            print(f"  update {steps} | paired_loss={joint_loss_sum / max(steps, 1):.4f}")
    row = {"joint_train_loss": joint_loss_sum / max(steps, 1)}
    accs = []
    for task in task_names:
        row[f"{task}_train_loss"] = task_loss_sum[task] / max(task_seen[task], 1)
        row[f"{task}_train_acc"] = task_correct[task] / max(task_seen[task], 1)
        accs.append(row[f"{task}_train_acc"])
    row["joint_train_acc"] = sum(task_correct.values()) / max(sum(task_seen.values()), 1)
    row["macro_train_acc"] = sum(accs) / max(len(accs), 1)
    return row


@torch.no_grad()
def evaluate(model, loader, task_names, criterion, device, max_batches=None):
    model.eval()
    loss_sum = {task: 0.0 for task in task_names}
    correct = {task: 0 for task in task_names}
    seen = {task: 0 for task in task_names}
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images = images.to(device)
        labels = labels_to_device(labels, device)
        for task in task_names:
            logits = model(images, task_name=task)
            loss = criterion(logits, labels[task])
            c, s = task_accuracy(logits, labels[task])
            loss_sum[task] += float(loss.item()) * s
            correct[task] += c
            seen[task] += s
    row = {}
    accs, losses = [], []
    for task in task_names:
        row[f"{task}_loss"] = loss_sum[task] / max(seen[task], 1)
        row[f"{task}_acc"] = correct[task] / max(seen[task], 1)
        row[f"{task}_samples"] = seen[task]
        accs.append(row[f"{task}_acc"])
        losses.append(row[f"{task}_loss"])
    row["joint_loss"] = sum(loss_sum.values()) / max(sum(seen.values()), 1)
    row["joint_acc"] = sum(correct.values()) / max(sum(seen.values()), 1)
    row["macro_acc"] = sum(accs) / max(len(accs), 1)
    row["macro_loss"] = sum(losses) / max(len(losses), 1)
    return row


@torch.no_grad()
def collect_diagnostics(model, batch, task_names, device):
    images, labels = batch
    images = images.to(device)
    diagnostics = {}
    model.eval()
    for task in task_names:
        logits, intermediates = model(images, task_name=task, return_intermediates=True)
        diagnostics[task] = {
            "intermediates": intermediates,
            "predictions": logits.argmax(dim=1).detach().cpu(),
            "targets": labels[task].detach().cpu(),
        }
        for key in ["prompt_amplitudes", "prompt_powers", "normalized_prompt_powers"]:
            if key in intermediates:
                diagnostics[task][key] = intermediates[key].detach().cpu()
        if "expert_energy_ratios" in intermediates:
            diagnostics[task]["expert_energy_ratios"] = intermediates["expert_energy_ratios"].mean(dim=0).detach().cpu()
            diagnostics[task]["outside_energy_ratio"] = float(intermediates["outside_energy_ratio"].mean().item())
        if "detector_energies" in intermediates:
            diagnostics[task]["detector_energy_mean"] = intermediates["detector_energies"].mean(dim=0).detach().cpu()
    return diagnostics


def fixed_batch(loader, device, max_items=4):
    images, labels = next(iter(loader))
    return images[:max_items].to(device), {key: value[:max_items] for key, value in labels.items()}


def expert_labels(count: int):
    dim = int(round(math.sqrt(count)))
    return [f"E{r}{c}" for r in range(dim) for c in range(dim)]


def save_epoch_artifacts(model, diagnostics, run_dir: Path, epoch_name: str, task_names, enabled=True):
    if not enabled:
        return
    for task in task_names:
        ints = diagnostics[task]["intermediates"]
        labels = expert_labels(len(ints["prompt_amplitudes"])) if "prompt_amplitudes" in ints else None
        save_light_fields(ints, run_dir / "figures" / "light_fields" / epoch_name / task)
        save_prompt_maps(ints, run_dir / "figures" / "prompt" / epoch_name / task, expert_labels=labels)
        det_dir = run_dir / "figures" / "detector_outputs" / epoch_name / task
        det_dir.mkdir(parents=True, exist_ok=True)
        if "detector_field" in ints:
            save_image(ints["detector_field"][0], det_dir / "detector_plane_sample_000.png", "detector plane")
        samples_dir = run_dir / "figures" / "samples" / epoch_name / task
        samples_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"sample_index": idx, "target": int(diagnostics[task]["targets"][idx]), "prediction": int(diagnostics[task]["predictions"][idx])}
            for idx in range(min(len(diagnostics[task]["targets"]), 8))
        ]
        save_json(rows, samples_dir / "sample_predictions.json")
    phase_dir = run_dir / "figures" / "phase_masks" / epoch_name
    save_expert_phase_layers(model, phase_dir)
    if (phase_dir / "expert_phase_layers.png").exists():
        (phase_dir / "expert_phase_layers.png").replace(phase_dir / "shared_expert_phase_layers.png")
    if (phase_dir / "global_fc_phase.png").exists():
        (phase_dir / "global_fc_phase.png").replace(phase_dir / "shared_global_fc_phase.png")


def expert_usage_rows(run_id, epoch, diagnostics):
    rows = []
    for task, diag in diagnostics.items():
        if "prompt_amplitudes" not in diag:
            continue
        labels = expert_labels(len(diag["prompt_amplitudes"]))
        for idx, label in enumerate(labels):
            rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "task_name": task,
                    "expert_id": label,
                    "prompt_amplitude": float(diag["prompt_amplitudes"][idx]),
                    "prompt_power": float(diag["prompt_powers"][idx]),
                    "normalized_prompt_power": float(diag["normalized_prompt_powers"][idx]),
                    "expert_entrance_energy_ratio": float(diag["expert_energy_ratios"][idx]) if "expert_energy_ratios" in diag else "",
                    "outside_energy_ratio": diag.get("outside_energy_ratio", ""),
                }
            )
    return rows


def prompt_similarity_rows(run_id, epoch, diagnostics):
    rows = []
    tasks = list(diagnostics)
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            da, db = diagnostics[a], diagnostics[b]
            if "normalized_prompt_powers" not in da or "normalized_prompt_powers" not in db:
                continue
            pa, pb = da["normalized_prompt_powers"].float(), db["normalized_prompt_powers"].float()
            aa, ab = da["prompt_amplitudes"].float(), db["prompt_amplitudes"].float()
            ia, ib = da["intermediates"], db["intermediates"]
            router_corr = ""
            total_corr = ""
            if "prompt_router_amplitude" in ia and "prompt_router_amplitude" in ib:
                router_corr = float(torch.nn.functional.cosine_similarity(ia["prompt_router_amplitude"].float().reshape(-1), ib["prompt_router_amplitude"].float().reshape(-1), dim=0))
            if "prompt_total_amplitude" in ia and "prompt_total_amplitude" in ib:
                total_corr = float(torch.nn.functional.cosine_similarity(ia["prompt_total_amplitude"].float().reshape(-1), ib["prompt_total_amplitude"].float().reshape(-1), dim=0))
            rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "task_a": a,
                    "task_b": b,
                    "amplitude_cosine": float(torch.nn.functional.cosine_similarity(aa, ab, dim=0)),
                    "normalized_power_cosine": float(torch.nn.functional.cosine_similarity(pa, pb, dim=0)),
                    "phase_bias_l2": "",
                    "complex_router_map_correlation": router_corr,
                    "prompt_total_field_correlation": total_corr,
                }
            )
    return rows


@torch.no_grad()
def same_input_task_switching(model, batch, task_names, device, run_id):
    images, labels = batch
    images = images.to(device)
    rows = []
    per_task = {}
    predictions = {}
    targets = {}
    for task in task_names:
        logits = model(images, task_name=task)
        pred = logits.argmax(dim=1).detach().cpu()
        target = labels[task].detach().cpu()
        correct = pred.eq(target)
        per_task[task] = float(correct.float().mean().item())
        predictions[task] = pred.tolist()
        targets[task] = target.tolist()
        for idx in range(len(target)):
            rows.append({"run_id": run_id, "sample_index": idx, "task_name": task, "target": int(target[idx]), "prediction": int(pred[idx]), "correct": bool(correct[idx])})
    payload = {
        "run_id": run_id,
        "num_samples": int(len(images)),
        "task_names": list(task_names),
        "per_task_accuracy": per_task,
        "macro_accuracy": sum(per_task.values()) / max(len(per_task), 1),
        "predictions": predictions,
        "targets": targets,
    }
    return rows, payload


@torch.no_grad()
def prompt_swap_eval(model, loader, task_names, criterion, device, max_batches=None):
    rows = []
    model.eval()
    for readout_task in task_names:
        for prompt_task in task_names:
            loss_sum, correct, seen = 0.0, 0, 0
            for batch_idx, (images, labels) in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                images = images.to(device)
                targets = labels[readout_task].to(device)
                logits = model(images, task_name=readout_task, prompt_task_name=prompt_task, readout_task_name=readout_task)
                loss = criterion(logits, targets)
                loss_sum += float(loss.item()) * targets.numel()
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                seen += int(targets.numel())
            rows.append(
                {
                    "eval_task": readout_task,
                    "prompt_task": prompt_task,
                    "readout_task": readout_task,
                    "accuracy": correct / max(seen, 1),
                    "loss": loss_sum / max(seen, 1),
                    "samples": seen,
                    "is_diagonal": readout_task == prompt_task,
                }
            )
    return rows


def prompt_swap_summary(rows, task_names):
    per_task = {}
    diag_accs, wrong_accs, gaps = [], [], []
    for task in task_names:
        diag = [row for row in rows if row["readout_task"] == task and row["prompt_task"] == task]
        wrong = [row for row in rows if row["readout_task"] == task and row["prompt_task"] != task]
        diag_acc = float(diag[0]["accuracy"]) if diag else 0.0
        wrong_acc = sum(float(row["accuracy"]) for row in wrong) / max(len(wrong), 1)
        gap = diag_acc - wrong_acc
        per_task[task] = {"diagonal_accuracy": diag_acc, "wrong_prompt_accuracy_mean": wrong_acc, "prompt_swap_gap": gap}
        diag_accs.append(diag_acc)
        wrong_accs.append(wrong_acc)
        gaps.append(gap)
    return {
        "diagonal_accuracy_per_task": {task: per_task[task]["diagonal_accuracy"] for task in task_names},
        "wrong_prompt_accuracy_mean_per_task": {task: per_task[task]["wrong_prompt_accuracy_mean"] for task in task_names},
        "prompt_swap_gap_per_task": {task: per_task[task]["prompt_swap_gap"] for task in task_names},
        "macro_diagonal_accuracy": sum(diag_accs) / max(len(diag_accs), 1),
        "macro_wrong_prompt_accuracy": sum(wrong_accs) / max(len(wrong_accs), 1),
        "macro_prompt_swap_gap": sum(gaps) / max(len(gaps), 1),
    }


def save_matrix_plot(rows, task_names, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = torch.zeros(len(task_names), len(task_names))
    for row in rows:
        values[task_names.index(row["readout_task"]), task_names.index(row["prompt_task"])] = float(row["accuracy"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_xlabel("prompt task")
    ax.set_ylabel("readout/eval task")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_expert_usage_heatmap(rows, path: Path, value_key: str = "normalized_prompt_power"):
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_epoch = max(int(row["epoch"]) for row in rows)
    latest = [row for row in rows if int(row["epoch"]) == latest_epoch and row.get(value_key, "") != ""]
    if not latest:
        return
    tasks = list(dict.fromkeys(row["task_name"] for row in latest))
    experts = list(dict.fromkeys(row["expert_id"] for row in latest))
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


def save_prompt_similarity_heatmap(rows, task_names, path: Path, value_key: str = "normalized_power_cosine"):
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest_epoch = max(int(row["epoch"]) for row in rows)
    latest = [row for row in rows if int(row["epoch"]) == latest_epoch and row.get(value_key, "") != ""]
    values = torch.eye(len(task_names))
    for row in latest:
        a, b = row["task_a"], row["task_b"]
        if a not in task_names or b not in task_names:
            continue
        value = float(row[value_key])
        values[task_names.index(a), task_names.index(b)] = value
        values[task_names.index(b), task_names.index(a)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(values.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title(value_key)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_same_input_samples_plot(batch, payload, task_names, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, _labels = batch
    images = images.detach().cpu()
    num_samples = min(images.shape[0], int(payload.get("num_samples", images.shape[0])), 8)
    if num_samples <= 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, num_samples, figsize=(2.3 * num_samples, 3.2))
    if num_samples == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx, 0].float().numpy(), cmap="gray")
        lines = []
        for task in task_names:
            target = payload["targets"][task][idx]
            pred = payload["predictions"][task][idx]
            lines.append(f"{task}: {target}/{pred}")
        ax.set_title("\n".join(lines), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rebuild_same_input_tables(runs_dir: Path, out_dir: Path):
    keys = {
        "runs": "master_runs.csv",
        "epoch_metrics": "master_epoch_metrics.csv",
        "task_metrics": "master_task_metrics.csv",
        "final_metrics": "master_final_metrics.csv",
        "same_input_switching": "master_same_input_switching.csv",
        "prompt_swap": "master_prompt_swap.csv",
        "expert_usage": "master_expert_usage.csv",
        "prompt_similarity": "master_prompt_similarity.csv",
        "model_params": "master_model_params.csv",
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


def save_architecture_report(model, config, run_dir: Path):
    global_fc = getattr(model, "global_fc", None)
    report = {
        "model_type": config.get("model", {}).get("type"),
        "task_names": list(model.task_names),
        "task_num_classes": model.task_num_classes,
        "same_input_paired_training": True,
        "shared_optical_backbone": True,
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
    save_json(report, run_dir / "architecture_report.json")
    lines = [
        "# Same-Input Multitask Architecture",
        "",
        "- paired same-input training: true",
        "- shared 9-expert AS global-router backbone for MoE configs",
        "- task switching changes prompt and readout head only",
        f"- tasks: {', '.join(model.task_names)}",
        f"- optical_parameter_count: {report['optical_parameter_count']}",
        f"- prompt_parameter_count: {report['prompt_parameter_count']}",
        f"- electronic_parameter_count: {report['electronic_parameter_count']}",
        f"- global_fc_phase_mode: {report.get('global_fc_phase_mode', '')}",
        f"- global_fc_parameter_count: {report.get('global_fc_parameter_count', '')}",
        f"- active_window_size: {report.get('active_window_size', '')}",
    ]
    write_text(run_dir / "architecture_report.md", "\n".join(lines) + "\n")
    return report


def run_training(config, args):
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    if args.smoke_test:
        config.setdefault("dataset", {})["smoke_test"] = True
        apply_smoke_loader_overrides(config["dataset"])
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    train_loader, val_loader, test_loader, task_num_classes, task_names = create_same_input_multitask_dataloaders(config, seed)
    model = build_model(config, task_names, task_num_classes).to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    phase_dropout = phase_dropout_settings(config)
    model.set_phase_dropout_active(False)
    run_name = config.get("experiment", {}).get("run_name", f"same_input_{int(time.time())}")
    run_dir = make_run_dir(EXPERIMENT_ROOT, "same_input_multitask", run_name)
    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    save_json(collect_git_info(REPO_ROOT), run_dir / "git_info.json")
    save_json(collect_environment(), run_dir / "environment.json")
    write_text(run_dir / "command.txt", " ".join(sys.argv))
    arch = save_architecture_report(model, config, run_dir)
    print(f"device: {device}")
    print(f"tasks: {task_names}, classes={task_num_classes}")
    print(f"paired same-input batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    fixed = fixed_batch(val_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    diagnostics = collect_diagnostics(model, fixed, task_names, device)
    viz_enabled = bool(config.get("visualization", {}).get("enabled", True))
    save_epoch_artifacts(model, diagnostics, run_dir, "epoch_0000", task_names, enabled=viz_enabled)
    usage_rows = expert_usage_rows(run_name, 0, diagnostics)
    prompt_sim_rows = prompt_similarity_rows(run_name, 0, diagnostics)
    metrics_rows, task_rows = [], []
    best = {"epoch": 0, "macro_val_acc": -1.0}
    epochs = int(config.get("training", {}).get("epochs", 200))
    if args.smoke_test:
        epochs = int(args.epochs or 1)
    loss_weights = {task: float(config.get("training", {}).get("loss_weights", {}).get(task, 1.0)) for task in task_names}
    max_val_batches = config.get("training", {}).get("evaluation", {}).get("max_val_batches")
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train = train_one_epoch(model, train_loader, task_names, loss_weights, criterion, optimizer, device, int(config.get("training", {}).get("print_freq", 50)))
        val = evaluate(model, val_loader, task_names, criterion, device, max_batches=max_val_batches)
        row = {
            "run_id": run_name,
            "epoch": epoch,
            "phase_dropout_active": active,
            "phase_dropout_mode": phase_dropout["mode"],
            "expert_phase_dropout_p": phase_dropout["expert_p"],
            "global_fc_phase_dropout_p": phase_dropout["global_fc_p"],
            **train,
            "joint_val_loss": val["joint_loss"],
            "joint_val_acc": val["joint_acc"],
            "macro_val_acc": val["macro_acc"],
            "macro_val_loss": val["macro_loss"],
        }
        for task in task_names:
            row[f"{task}_val_loss"] = val[f"{task}_loss"]
            row[f"{task}_val_acc"] = val[f"{task}_acc"]
            task_rows.append({"run_id": run_name, "epoch": epoch, "task_name": task, "train_loss": train[f"{task}_train_loss"], "train_acc": train[f"{task}_train_acc"], "val_loss": val[f"{task}_loss"], "val_acc": val[f"{task}_acc"]})
        row["epoch_time_sec"] = time.perf_counter() - epoch_start
        metrics_rows.append(row)
        diagnostics = collect_diagnostics(model, fixed, task_names, device)
        usage_rows.extend(expert_usage_rows(run_name, epoch, diagnostics))
        prompt_sim_rows.extend(prompt_similarity_rows(run_name, epoch, diagnostics))
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if row["macro_val_acc"] > best["macro_val_acc"]:
            best = {"epoch": epoch, "macro_val_acc": row["macro_val_acc"], "row": row}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        if viz_enabled and epoch % int(config.get("visualization", {}).get("save_interval_epochs", 10)) == 0:
            save_epoch_artifacts(model, diagnostics, run_dir, f"epoch_{epoch:04d}", task_names, enabled=True)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
        write_rows(run_dir / "metrics" / "task_metrics.csv", task_rows)
        write_rows(run_dir / "diagnostics" / "task_prompt_amplitude_history.csv", usage_rows)
        write_rows(run_dir / "diagnostics" / "task_expert_energy_history.csv", usage_rows)
        write_rows(run_dir / "diagnostics" / "prompt_similarity.csv", prompt_sim_rows)
        print(f"epoch {epoch:03d} macro_train={row['macro_train_acc']:.4f} macro_val={row['macro_val_acc']:.4f} phase_dropout={'on' if active else 'off'}")

    test = evaluate(model, test_loader, task_names, criterion, device, max_batches=max_test_batches)
    fixed_test = fixed_batch(test_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    same_rows, same_payload = same_input_task_switching(model, fixed_test, task_names, device, run_name)
    swap_rows = prompt_swap_eval(model, test_loader, task_names, criterion, device, max_batches=max_test_batches)
    for row in swap_rows:
        row["run_id"] = run_name
    swap_summary = prompt_swap_summary(swap_rows, task_names)
    write_rows(run_dir / "metrics" / "same_input_task_switching.csv", same_rows)
    save_json(same_payload, run_dir / "metrics" / "same_input_task_switching.json")
    write_rows(run_dir / "metrics" / "prompt_swap_matrix.csv", swap_rows)
    save_json(swap_summary, run_dir / "metrics" / "prompt_swap_summary.json")
    save_json(test, run_dir / "metrics" / "final_test_metrics.json")
    save_matrix_plot(swap_rows, task_names, run_dir / "figures" / "prompt_swap_matrix.png")
    save_expert_usage_heatmap(usage_rows, run_dir / "figures" / "task_expert_usage_heatmap.png")
    save_prompt_similarity_heatmap(prompt_sim_rows, task_names, run_dir / "figures" / "prompt_similarity_heatmap.png")
    if viz_enabled:
        save_same_input_samples_plot(fixed_test, same_payload, task_names, run_dir / "figures" / "same_input_task_switching_samples.png")
    if metrics_rows:
        curve_rows = [{"epoch": r["epoch"], "train_loss": r["joint_train_loss"], "val_loss": r["joint_val_loss"], "train_acc": r["macro_train_acc"], "val_acc": r["macro_val_acc"]} for r in metrics_rows]
        save_training_curves(curve_rows, run_dir / "figures" / "training_curves.png")
    total_wall_time_sec = time.perf_counter() - run_start
    total_train_time_sec = sum(float(r.get("epoch_time_sec", 0.0)) for r in metrics_rows)
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
    final_rows = []
    for task in task_names:
        gap = swap_summary["prompt_swap_gap_per_task"].get(task, "")
        final_rows.append(
            {
                "run_id": run_name,
                "model_type": config.get("model", {}).get("type"),
                "stage": len(task_names),
                "num_tasks": len(task_names),
                "task_name": task,
                "num_classes": task_num_classes[task],
                "best_epoch": best["epoch"],
                "best_val_acc": best["macro_val_acc"],
                "final_test_acc": test[f"{task}_acc"],
                "same_input_accuracy": same_payload["per_task_accuracy"].get(task, ""),
                "prompt_swap_gap": gap,
                "optical_parameter_count": int(model.optical_parameter_count()),
                "prompt_parameter_count": int(model.prompt_parameter_count()),
                "electronic_parameter_count": int(model.electronic_parameter_count()),
                "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                **fc_summary,
                "total_wall_time_sec": total_wall_time_sec,
                "total_train_time_sec": total_train_time_sec,
                "run_dir": str(run_dir),
            }
        )
    scaling = {
        "run_id": run_name,
        "model_type": config.get("model", {}).get("type"),
        "stage": len(task_names),
        "num_tasks": len(task_names),
        "task_names": ",".join(task_names),
        "macro_final_test_acc": test["macro_acc"],
        "min_final_test_acc": min(test[f"{task}_acc"] for task in task_names),
        "macro_same_input_acc": same_payload["macro_accuracy"],
        "macro_prompt_swap_gap": swap_summary["macro_prompt_swap_gap"],
        "mean_prompt_similarity": sum(float(r["normalized_power_cosine"]) for r in prompt_sim_rows) / max(len(prompt_sim_rows), 1) if prompt_sim_rows else "",
        "mean_expert_entropy": "",
        "optical_parameter_count": int(model.optical_parameter_count()),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    model_params = {"run_id": run_name, "optical_parameter_count": int(model.optical_parameter_count()), "prompt_parameter_count": int(model.prompt_parameter_count()), "electronic_parameter_count": int(model.electronic_parameter_count()), "total_parameter_count": int(sum(p.numel() for p in model.parameters())), **fc_summary}
    final_metrics = {
        "run_id": run_name,
        "best_epoch": best["epoch"],
        "best_macro_val_acc": best["macro_val_acc"],
        "final_test": test,
        "same_input_task_switching": same_payload,
        "prompt_swap_summary": swap_summary,
        "total_wall_time_sec": total_wall_time_sec,
        "total_train_time_sec": total_train_time_sec,
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    summary = {"run_id": run_name, "task_names": task_names, "task_num_classes": task_num_classes, "architecture": arch, "best": best, "final_metrics": final_metrics, "same_input_task_switching": same_payload, "prompt_swap_summary": swap_summary}
    save_json(summary, run_dir / "summary.json")
    save_json({"run_id": run_name, "model_type": config.get("model", {}).get("type"), "stage": len(task_names), "num_tasks": len(task_names), "total_wall_time_sec": total_wall_time_sec, "total_train_time_sec": total_train_time_sec, "run_dir": str(run_dir)}, run_dir / "summary_for_master" / "runs_rows.json")
    save_json(metrics_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json(task_rows, run_dir / "summary_for_master" / "task_metrics_rows.json")
    save_json(final_rows, run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json(same_rows, run_dir / "summary_for_master" / "same_input_switching_rows.json")
    save_json(swap_rows, run_dir / "summary_for_master" / "prompt_swap_rows.json")
    save_json(usage_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    save_json(prompt_sim_rows, run_dir / "summary_for_master" / "prompt_similarity_rows.json")
    save_json([model_params], run_dir / "summary_for_master" / "model_params_rows.json")
    save_json([scaling], run_dir / "summary_for_master" / "scaling_results_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_same_input_tables(EXPERIMENT_ROOT / "same_input_multitask" / "runs", EXPERIMENT_ROOT / "same_input_multitask" / "results")
    print(f"saved run outputs to: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    config = load_yaml(args.config)
    run_training(config, args)


if __name__ == "__main__":
    main()
