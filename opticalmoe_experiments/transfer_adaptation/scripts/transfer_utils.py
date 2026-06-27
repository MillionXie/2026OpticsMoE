from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
TRANSFER_ROOT = EXPERIMENT_ROOT / "transfer_adaptation"

import sys

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.data.datasets import EMNIST_NUM_CLASSES, create_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders
from common.optics.detector import DetectorArray
from common.optics.global_router_prompt import GlobalRouterPrompt
from common.optics.readout import ElectronicReadout
from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import save_checkpoint
from common.training.task_heads import normalize_head_config
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import write_text
from common.visualization.curve_viz import save_training_curves
from common.visualization.prompt_viz import save_task_expert_weights_from_model
from dataset_switching.scripts import train_dataset_switching as ds_train


SOURCE_TASKS = ["mnist", "fashionmnist", "emnist_letters"]
TARGET_TASKS = {"usps": 10, "kmnist": 10}
DEFAULT_BACKBONE_DIR = (
    "opticalmoe_experiments/transfer_adaptation/pretrained_backbones/"
    "dataset_switching_moe_mnist_fashion_emnist_letters"
)

DEFAULT_TARGET_HEAD = {
    "detector_size": 32,
    "detector_layout": "grid",
    "readout_type": "optical_only",
    "normalize_detector_energy": True,
    "logit_scale": 10.0,
    "input_norm": "none",
    "norm_affine": False,
    "hidden_dim": 64,
    "hidden_layers": 0,
    "activation": "relu",
    "dropout": 0.0,
}


def resolve_path(path_like, *, prefer_experiment_root: bool = False) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    bases = [Path.cwd(), EXPERIMENT_ROOT, REPO_ROOT]
    if not prefer_experiment_root:
        bases = [Path.cwd(), REPO_ROOT, EXPERIMENT_ROOT]
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    if path.parts and path.parts[0] == "opticalmoe_experiments":
        return (REPO_ROOT / path).resolve()
    return ((EXPERIMENT_ROOT if prefer_experiment_root else Path.cwd()) / path).resolve()


def source_artifact_paths(config: Mapping) -> Dict[str, Path]:
    source = dict(config.get("source", {}) or {})
    checkpoint_dir = resolve_path(source.get("checkpoint_dir", DEFAULT_BACKBONE_DIR))
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint": checkpoint_dir / source.get("checkpoint_name", "source_best.pt"),
        "config": checkpoint_dir / source.get("config_name", "source_config.yaml"),
        "architecture_report": checkpoint_dir / source.get("architecture_report_name", "source_architecture_report.json"),
    }


def missing_checkpoint_message(config: Mapping) -> str:
    source = dict(config.get("source", {}) or {})
    path = Path(source.get("checkpoint_dir", DEFAULT_BACKBONE_DIR)) / source.get("checkpoint_name", "source_best.pt")
    return "Please place the pretrained dataset-switching OpticalMoE checkpoint at:\n" + str(path).replace("\\", "/")


def missing_source_config_message(config: Mapping) -> str:
    source = dict(config.get("source", {}) or {})
    path = Path(source.get("checkpoint_dir", DEFAULT_BACKBONE_DIR)) / source.get("config_name", "source_config.yaml")
    return "Please place the pretrained dataset-switching OpticalMoE config at:\n" + str(path).replace("\\", "/")


def validate_source_artifacts(config: Mapping) -> Dict[str, Path]:
    paths = source_artifact_paths(config)
    if not paths["checkpoint"].exists():
        raise FileNotFoundError(missing_checkpoint_message(config))
    if not paths["config"].exists():
        raise FileNotFoundError(missing_source_config_message(config))
    return paths


def source_task_configs(source_config: Mapping) -> List[Dict]:
    return [dict(task) for task in source_config.get("training", {}).get("multitask", {}).get("tasks", [])]


def source_task_names(source_config: Mapping) -> List[str]:
    names = [str(task.get("name", "")).lower() for task in source_task_configs(source_config)]
    return [name for name in names if name]


def _dataset_num_classes(dataset_cfg: Mapping, task_name: str) -> int:
    name = str(dataset_cfg.get("name", task_name)).lower().replace("-", "")
    if task_name == "emnist_letters":
        return 26
    if name == "emnist":
        split = str(dataset_cfg.get("split", "letters")).lower()
        return int(EMNIST_NUM_CLASSES.get(split, 26))
    if name in {"mnist", "fashionmnist", "kmnist", "usps", "cifar10"}:
        return 10
    if name in TARGET_TASKS:
        return int(TARGET_TASKS[name])
    raise ValueError(f"Cannot infer num_classes for dataset={name!r}, task={task_name!r}.")


def infer_task_num_classes(config: Mapping, names: Sequence[str]) -> Dict[str, int]:
    task_cfgs = {str(task.get("name", "")).lower(): dict(task) for task in source_task_configs(config)}
    result = {}
    for name in names:
        task = task_cfgs.get(str(name).lower(), {})
        result[str(name).lower()] = _dataset_num_classes(task.get("dataset", {}) or {}, str(name).lower())
    return result


def validate_source_config(source_config: Mapping, expected_source_tasks: Sequence[str]) -> List[str]:
    model_type = str(source_config.get("model", {}).get("type", "")).lower()
    if model_type != "learnable_route_moe":
        raise ValueError(f"source_config.yaml must describe model.type: learnable_route_moe, got {model_type!r}.")
    names = source_task_names(source_config)
    missing = [task for task in expected_source_tasks if task not in names]
    if missing:
        raise ValueError(f"source_config.yaml is missing expected source tasks: {missing}; found {names}.")
    return names


def validate_source_model(model, expected_source_tasks: Sequence[str]) -> None:
    missing_attrs = [
        name
        for name in ["prompt_bank", "expert_layers", "global_fc"]
        if not hasattr(model, name)
    ]
    has_heads = hasattr(model, "task_readouts") or hasattr(model, "task_detectors")
    if missing_attrs or not has_heads:
        raise ValueError(
            "Source checkpoint must be a dataset-switching OpticalMoE with prompt_bank, "
            f"expert_layers, global_fc, and task heads. Missing: {missing_attrs}, has_heads={has_heads}."
        )
    model_tasks = set(getattr(model, "task_names", []))
    missing_tasks = [task for task in expected_source_tasks if task not in model_tasks]
    if missing_tasks:
        raise ValueError(f"Source model is missing expected source tasks: {missing_tasks}.")


def _extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ["model_state_dict", "state_dict"]:
            value = payload.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return dict(payload)
    raise ValueError("Checkpoint must contain model_state_dict, state_dict, or be a raw state_dict.")


def _state_dict_variants(state_dict: Mapping[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    variants = [dict(state_dict)]
    for prefix in ["module.", "model."]:
        if any(str(key).startswith(prefix) for key in state_dict):
            stripped = {
                str(key)[len(prefix):] if str(key).startswith(prefix) else str(key): value
                for key, value in state_dict.items()
            }
            variants.append(stripped)
    unique = []
    seen = set()
    for variant in variants:
        signature = tuple(sorted(variant.keys()))
        if signature not in seen:
            unique.append(variant)
            seen.add(signature)
    return unique


def _is_core_source_key(key: str, source_tasks: Sequence[str]) -> bool:
    if key.startswith("expert_layers.") or key.startswith("global_fc."):
        return True
    for task in source_tasks:
        if key.startswith(f"prompt_bank.prompts.{task}."):
            return True
        if key.startswith(f"task_readouts.{task}."):
            return True
    return False


def robust_load_source_checkpoint(model, checkpoint_path: Path, source_tasks: Sequence[str], device) -> Dict:
    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _extract_state_dict(payload)
    strict_errors = []
    for variant in _state_dict_variants(state_dict):
        try:
            model.load_state_dict(variant, strict=True)
            return {"loaded_strict": True, "missing_keys": [], "unexpected_keys": []}
        except RuntimeError as exc:
            strict_errors.append(str(exc))
    variant = _state_dict_variants(state_dict)[-1]
    incompatible = model.load_state_dict(variant, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    core_missing = [key for key in missing if _is_core_source_key(key, source_tasks)]
    if core_missing:
        raise RuntimeError(
            "Strict checkpoint loading failed and source backbone core keys are missing.\n"
            f"Missing core keys: {core_missing[:20]}\n"
            f"Unexpected keys: {unexpected[:20]}\n"
            f"Strict errors: {strict_errors[-1] if strict_errors else ''}"
        )
    print("Strict checkpoint loading failed; retrying with strict=False.")
    print(f"  missing keys: {missing[:20]}")
    print(f"  unexpected keys: {unexpected[:20]}")
    return {"loaded_strict": False, "missing_keys": missing, "unexpected_keys": unexpected}


def load_source_backbone(config: Mapping, device):
    paths = validate_source_artifacts(config)
    source_config = load_yaml(paths["config"])
    expected = [str(task).lower() for task in config.get("source", {}).get("expected_source_tasks", SOURCE_TASKS)]
    task_names = validate_source_config(source_config, expected)
    task_num_classes = infer_task_num_classes(source_config, task_names)
    model = ds_train.build_model(source_config, task_names, task_num_classes).to(device)
    load_info = robust_load_source_checkpoint(model, paths["checkpoint"], task_names, device)
    validate_source_model(model, expected)
    return model, source_config, task_names, task_num_classes, paths, load_info


def _prompt_from_reference(model, reference_prompt) -> GlobalRouterPrompt:
    return GlobalRouterPrompt(
        layout=model.layout,
        wavelength_m=reference_prompt.wavelength_m,
        pixel_size_m=reference_prompt.pixel_size_m,
        prompt_to_expert_m=reference_prompt.prompt_to_expert_m,
        focal_length_m=reference_prompt.focal_length_m,
        mode=reference_prompt.mode,
        amplitude_init_logits=2.0,
        train_amplitudes=True,
        train_phase_biases=True,
        grating_scale=reference_prompt.grating_scale,
        grating_sign_x=reference_prompt.grating_sign_x,
        grating_sign_y=reference_prompt.grating_sign_y,
        normalize=reference_prompt.normalize,
    )


def _copy_prompt_parameters(model, target: str, source: str) -> None:
    with torch.no_grad():
        model.prompt_bank.prompts[target].amplitude_logits.copy_(model.prompt_bank.prompts[source].amplitude_logits)
        model.prompt_bank.prompts[target].phase_biases.copy_(model.prompt_bank.prompts[source].phase_biases)


def _mean_source_prompt_parameters(model, target: str, source_tasks: Sequence[str]) -> None:
    with torch.no_grad():
        amp = torch.stack([model.prompt_bank.prompts[task].amplitude_logits.detach() for task in source_tasks]).mean(dim=0)
        phase = torch.stack([model.prompt_bank.prompts[task].phase_biases.detach() for task in source_tasks]).mean(dim=0)
        model.prompt_bank.prompts[target].amplitude_logits.copy_(amp)
        model.prompt_bank.prompts[target].phase_biases.copy_(phase)


def add_transfer_target_task(
    model,
    target_task_name: str,
    target_num_classes: int,
    init_from_source_prompt: str = "mnist",
    target_head_config: Optional[Mapping] = None,
    train_target_prompt: bool = True,
    train_target_readout: bool = False,
):
    target = str(target_task_name).lower()
    if target in getattr(model, "task_names", []):
        raise ValueError(f"Target task {target!r} already exists in source model.")
    if not hasattr(model, "prompt_bank"):
        raise ValueError("Transfer prompt tuning requires a source model with prompt_bank.")
    device = next(model.parameters()).device
    source_tasks = list(model.task_names)
    reference_task = source_tasks[0]
    reference_prompt = model.prompt_bank.prompts[reference_task]
    prompt = _prompt_from_reference(model, reference_prompt).to(device)
    model.prompt_bank.prompts[target] = prompt
    model.prompt_bank.task_names.append(target)
    model.task_names.append(target)
    model.task_num_classes[target] = int(target_num_classes)

    settings = normalize_head_config(target_head_config or DEFAULT_TARGET_HEAD, DEFAULT_TARGET_HEAD)
    model.task_detectors[target] = DetectorArray(
        num_classes=int(target_num_classes),
        grid_size=model.canvas_shape,
        detector_size=settings["detector_size"],
        layout=settings["detector_layout"],
        normalize_total_energy=settings["normalize_detector_energy"],
    ).to(device)
    model.task_readouts[target] = ElectronicReadout(
        num_classes=int(target_num_classes),
        readout_type=settings["readout_type"],
        logit_scale=settings["logit_scale"],
        hidden_dim=settings["hidden_dim"],
        activation=settings["activation"],
        input_norm=settings["input_norm"],
        norm_affine=settings["norm_affine"],
        hidden_layers=settings["hidden_layers"],
        dropout=settings["dropout"],
    ).to(device)
    model.task_head_configs[target] = settings

    init = str(init_from_source_prompt).lower()
    if init in model.prompt_bank.prompts and init != target:
        _copy_prompt_parameters(model, target, init)
    elif init == "uniform":
        with torch.no_grad():
            model.prompt_bank.prompts[target].amplitude_logits.fill_(2.0)
            model.prompt_bank.prompts[target].phase_biases.zero_()
    elif init == "mean_source_prompts":
        _mean_source_prompt_parameters(model, target, source_tasks)
    else:
        raise ValueError(
            f"Unsupported init_from_source_prompt={init_from_source_prompt!r}; "
            f"use one of source tasks {source_tasks}, uniform, or mean_source_prompts."
        )

    for parameter in model.prompt_bank.prompts[target].parameters():
        parameter.requires_grad = bool(train_target_prompt)
    for parameter in model.task_readouts[target].parameters():
        parameter.requires_grad = bool(train_target_readout)
    return model


def _display_parameter_name(name: str) -> str:
    return name.replace("prompt_bank.prompts.", "prompt_bank.")


def _is_target_prompt_param(name: str, target: str) -> bool:
    return name.startswith(f"prompt_bank.prompts.{target}.")


def _is_electronic_param(name: str) -> bool:
    return name.startswith("task_readouts.")


def parameter_count_summary(model, target: str, transfer_cfg: Mapping) -> Dict:
    trainable_optical = 0
    trainable_electronic = 0
    frozen_optical = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            if _is_electronic_param(name):
                trainable_electronic += count
            else:
                trainable_optical += count
        elif not _is_electronic_param(name):
            frozen_optical += count
    return {
        "train_target_prompt": bool(transfer_cfg.get("train_target_prompt", True)),
        "train_target_readout": bool(transfer_cfg.get("train_target_readout", False)),
        "freeze_expert_layers": bool(transfer_cfg.get("freeze_expert_layers", True)),
        "freeze_global_fc": bool(transfer_cfg.get("freeze_global_fc", True)),
        "freeze_source_prompts": bool(transfer_cfg.get("freeze_source_prompts", True)),
        "freeze_source_readouts": bool(transfer_cfg.get("freeze_source_readouts", True)),
        "trainable_optical_params": trainable_optical,
        "trainable_electronic_params": trainable_electronic,
        "total_trainable_params": trainable_optical + trainable_electronic,
        "frozen_optical_params": frozen_optical,
        "total_model_params": total,
    }


def apply_transfer_freeze_policy(model, target_task_name: str, transfer_cfg: Mapping, run_dir: Optional[Path] = None) -> Dict:
    target = str(target_task_name).lower()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if bool(transfer_cfg.get("train_target_prompt", True)):
        for parameter in model.prompt_bank.prompts[target].parameters():
            parameter.requires_grad = True
    train_readout = bool(transfer_cfg.get("train_target_readout", False)) and not bool(transfer_cfg.get("freeze_target_readout", True))
    if train_readout:
        for parameter in model.task_readouts[target].parameters():
            parameter.requires_grad = True

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    allowed = {name for name, _parameter in model.named_parameters() if _is_target_prompt_param(name, target)}
    if train_readout:
        allowed.update(name for name, _parameter in model.task_readouts[target].named_parameters(prefix=f"task_readouts.{target}"))
    extra = sorted(set(trainable) - allowed)
    if extra and not bool(transfer_cfg.get("allow_extra_trainable_params", False)):
        raise RuntimeError(f"Transfer prompt-only policy expected only target prompt trainable; extra parameters: {extra}")

    summary = parameter_count_summary(model, target, transfer_cfg)
    payload = {
        "target_task": target,
        "trainable_parameter_names": trainable,
        "trainable_parameter_display_names": [_display_parameter_name(name) for name in trainable],
        "frozen_parameter_names": frozen,
        **summary,
    }
    if run_dir is not None:
        freeze_dir = Path(run_dir) / "parameter_freeze"
        write_text(freeze_dir / "trainable_parameter_names.txt", "\n".join(trainable) + ("\n" if trainable else ""))
        write_text(freeze_dir / "frozen_parameter_names.txt", "\n".join(frozen) + ("\n" if frozen else ""))
        save_json(payload, freeze_dir / "parameter_freeze_summary.json")
    print("Trainable parameters:")
    for name in trainable:
        print(f"  {_display_parameter_name(name)}")
    print("Frozen parameter groups:")
    print("  expert_layers")
    print("  global_fc")
    print("  source prompts")
    print("  source readouts")
    print("  target readout")
    return payload


def create_target_loaders(config: Mapping, seed: int, smoke_test: bool = False):
    dataset_cfg = dict(config.get("target", {}).get("dataset", {}) or {})
    if smoke_test:
        dataset_cfg["smoke_test"] = True
        dataset_cfg.setdefault("smoke_train_size", 64)
        dataset_cfg.setdefault("smoke_test_size", 32)
        apply_smoke_loader_overrides(dataset_cfg)
    bundle = create_dataloaders(dataset_cfg, seed=seed)
    summary = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, dataset_cfg)
    return bundle, summary, dataset_cfg


def create_source_task_loaders(source_config: Mapping, seed: int, smoke_test: bool = False):
    result = ds_train.create_task_loaders(dict(source_config), seed, smoke_test)
    if len(result) == 6:
        train_loaders, val_loaders, test_loaders, task_num_classes, class_names, summaries = result
    else:
        train_loaders, val_loaders, test_loaders, task_num_classes, class_names = result
        summaries = {
            name: loader_summary_from_loaders(train_loaders[name], val_loaders[name], test_loaders[name], {})
            for name in train_loaders
        }
    return train_loaders, val_loaders, test_loaders, task_num_classes, class_names, summaries


def fixed_batch(loader, device, max_items: int = 4):
    images, targets = next(iter(loader))
    return images[:max_items].to(device), targets[:max_items].to(device)


def train_target_epoch(model, loader, target_task: str, criterion, optimizer, device, print_freq: int = 50) -> Dict:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, task_name=target_task)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch = int(targets.numel())
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch
        if print_freq > 0 and (step % int(print_freq) == 0 or step == len(loader)):
            print(f"  update {step}/{len(loader)} | target_loss={total_loss / max(seen, 1):.4f} | target_acc={correct / max(seen, 1):.4f}")
    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1), "samples": seen}


@torch.no_grad()
def evaluate_task(model, loader, device, criterion, task_name: str, prompt_task_name: Optional[str] = None, readout_task_name: Optional[str] = None, max_batches=None) -> Dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(
            images,
            task_name=task_name,
            prompt_task_name=prompt_task_name,
            readout_task_name=readout_task_name,
        )
        loss = criterion(logits, targets)
        batch = int(targets.numel())
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch
    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1), "samples": seen}


def build_optimizer(model, config: Mapping):
    cfg = dict(config.get("optimizer", {}) or {})
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters after transfer freeze policy.")
    opt_type = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 0.01))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.9)))
    raise ValueError(f"Unsupported optimizer.type: {opt_type}")


def target_prompt_swap_rows(model, target_loader, source_tasks: Sequence[str], target_task: str, target_dataset: str, device, criterion, run_id: str, max_batches=None) -> Tuple[List[Dict], Dict]:
    prompts = [target_task] + [task for task in source_tasks if task != target_task]
    rows = []
    for prompt_task in prompts:
        metrics = evaluate_task(
            model,
            target_loader,
            device,
            criterion,
            task_name=target_task,
            prompt_task_name=prompt_task,
            readout_task_name=target_task,
            max_batches=max_batches,
        )
        rows.append(
            {
                "run_id": run_id,
                "target_task": target_task,
                "target_dataset": target_dataset,
                "prompt_task": prompt_task,
                "readout_task": target_task,
                "accuracy": metrics["acc"],
                "loss": metrics["loss"],
                "samples": metrics["samples"],
                "is_target_prompt": prompt_task == target_task,
            }
        )
    target_row = next(row for row in rows if row["is_target_prompt"])
    source_rows = [row for row in rows if not row["is_target_prompt"]]
    best_source = max(source_rows, key=lambda row: float(row["accuracy"])) if source_rows else None
    mean_source = sum(float(row["accuracy"]) for row in source_rows) / max(len(source_rows), 1) if source_rows else 0.0
    summary = {
        "target_prompt_accuracy": float(target_row["accuracy"]),
        "mean_source_prompt_accuracy": mean_source,
        "best_source_prompt_accuracy": float(best_source["accuracy"]) if best_source else "",
        "target_prompt_gap": float(target_row["accuracy"]) - mean_source,
        "best_source_prompt": best_source["prompt_task"] if best_source else "",
        "source_prompt_accuracies": {row["prompt_task"]: float(row["accuracy"]) for row in source_rows},
    }
    return rows, summary


def save_target_prompt_swap_plot(rows: Sequence[Mapping], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    prompts = [str(row["prompt_task"]) for row in rows]
    accs = [float(row["accuracy"]) for row in rows]
    colors = ["#0072B2" if bool(row.get("is_target_prompt")) else "#999999" for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(rows)), accs, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(prompts, rotation=30, ha="right")
    ax.set_ylabel("target accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Target prompt swap")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_source_tasks(model, source_test_loaders: Mapping, source_tasks: Sequence[str], device, criterion, max_batches=None) -> Dict[str, Dict]:
    return {
        task: evaluate_task(model, source_test_loaders[task], device, criterion, task_name=task, max_batches=max_batches)
        for task in source_tasks
        if task in source_test_loaders
    }


def source_retention_rows(run_id: str, before: Mapping[str, Mapping], after: Mapping[str, Mapping], source_tasks: Sequence[str]) -> Tuple[List[Dict], Dict]:
    rows = []
    for task in source_tasks:
        before_acc = float(before[task]["acc"])
        after_acc = float(after[task]["acc"])
        rows.append(
            {
                "run_id": run_id,
                "source_task": task,
                "before_acc": before_acc,
                "after_acc": after_acc,
                "acc_drop": before_acc - after_acc,
                "samples": after[task]["samples"],
            }
        )
    drops = [float(row["acc_drop"]) for row in rows]
    summary = {
        "mean_source_acc_before": sum(float(row["before_acc"]) for row in rows) / max(len(rows), 1),
        "mean_source_acc_after": sum(float(row["after_acc"]) for row in rows) / max(len(rows), 1),
        "mean_source_acc_drop": sum(drops) / max(len(drops), 1),
        "max_source_acc_drop": max(drops) if drops else 0.0,
    }
    return rows, summary


def _complex_field_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().to(torch.complex64).reshape(-1)
    bv = b.detach().to(torch.complex64).reshape(-1)
    denom = torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv) + 1e-8
    return float(torch.abs(torch.sum(torch.conj(av) * bv)) / denom)


def prompt_similarity_rows(model, run_id: str, target_task: str, source_tasks: Sequence[str]) -> List[Dict]:
    rows = []
    target_prompt = model.prompt_bank.prompts[target_task]
    target_amp = target_prompt.amplitudes().detach().float()
    target_power = target_prompt.normalized_powers().detach().float()
    target_field = target_prompt.transmission().detach()
    for source_task in source_tasks:
        source_prompt = model.prompt_bank.prompts[source_task]
        amp = source_prompt.amplitudes().detach().float()
        power = source_prompt.normalized_powers().detach().float()
        rows.append(
            {
                "run_id": run_id,
                "target_task": target_task,
                "source_task": source_task,
                "amplitude_cosine": float(torch.nn.functional.cosine_similarity(target_amp, amp, dim=0).item()),
                "normalized_power_cosine": float(torch.nn.functional.cosine_similarity(target_power, power, dim=0).item()),
                "phase_bias_l2": float(torch.linalg.vector_norm(target_prompt.phase_biases.detach() - source_prompt.phase_biases.detach()).item()),
                "prompt_total_field_correlation": _complex_field_correlation(target_field, source_prompt.transmission().detach()),
            }
        )
    return rows


def save_prompt_similarity_plot(rows: Sequence[Mapping], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    tasks = [str(row["source_task"]) for row in rows]
    amp = [float(row["amplitude_cosine"]) for row in rows]
    power = [float(row["normalized_power_cosine"]) for row in rows]
    field = [float(row["prompt_total_field_correlation"]) for row in rows]
    x = torch.arange(len(tasks)).float()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.25
    ax.bar((x - width).numpy(), amp, width=width, label="amplitude")
    ax.bar(x.numpy(), power, width=width, label="power")
    ax.bar((x + width).numpy(), field, width=width, label="field")
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _expert_labels(count: int) -> List[str]:
    dim = int(round(math.sqrt(int(count))))
    if dim * dim == int(count):
        return [f"E{r}{c}" for r in range(dim) for c in range(dim)]
    return [f"E{idx:02d}" for idx in range(int(count))]


@torch.no_grad()
def collect_task_diagnostics(model, batch, device, task_name: str) -> Dict:
    images, targets = batch
    model.eval()
    logits, intermediates = model(images.to(device), task_name=task_name, return_intermediates=True)
    diagnostics = {
        "targets": targets.detach().cpu(),
        "predictions": logits.argmax(dim=1).detach().cpu(),
        "intermediates": intermediates,
    }
    for key in ["prompt_amplitudes", "prompt_powers", "normalized_prompt_powers"]:
        if key in intermediates:
            diagnostics[key] = intermediates[key].detach().cpu()
    if "expert_energy_ratios" in intermediates:
        diagnostics["expert_energy_ratios"] = intermediates["expert_energy_ratios"].mean(dim=0).detach().cpu()
    if "outside_energy_ratio" in intermediates:
        diagnostics["outside_energy_ratio"] = float(intermediates["outside_energy_ratio"].mean().item())
    if "detector_energies" in intermediates:
        diagnostics["detector_energy_mean"] = float(intermediates["detector_energies"].mean().item())
    return diagnostics


def expert_usage_rows(model, run_id: str, diagnostics: Mapping[str, Dict], source_tasks: Sequence[str], target_task: str) -> List[Dict]:
    rows = []
    for task_name, diag in diagnostics.items():
        if "prompt_amplitudes" not in diag:
            continue
        labels = _expert_labels(len(diag["prompt_amplitudes"]))
        for idx, expert_id in enumerate(labels):
            rows.append(
                {
                    "run_id": run_id,
                    "task_name": task_name,
                    "task_group": "target" if task_name == target_task else "source",
                    "expert_id": expert_id,
                    "prompt_amplitude": float(diag["prompt_amplitudes"][idx]),
                    "prompt_power": float(diag["prompt_powers"][idx]),
                    "normalized_prompt_power": float(diag["normalized_prompt_powers"][idx]),
                    "expert_entrance_energy_ratio": float(diag["expert_energy_ratios"][idx]) if "expert_energy_ratios" in diag else "",
                    "outside_energy_ratio": diag.get("outside_energy_ratio", ""),
                    "detector_energy_mean": diag.get("detector_energy_mean", ""),
                }
            )
    return rows


def save_expert_usage_heatmap(rows: Sequence[Mapping], path: Path, value_key: str = "normalized_prompt_power") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row.get(value_key, "") != ""]
    if not rows:
        return
    tasks = []
    for row in rows:
        if row["task_name"] not in tasks:
            tasks.append(row["task_name"])
    experts = []
    for row in rows:
        if row["expert_id"] not in experts:
            experts.append(row["expert_id"])
    values = torch.full((len(tasks), len(experts)), float("nan"))
    for row in rows:
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


def optical_energy_rows(run_id: str, diagnostics: Mapping[str, Dict]) -> List[Dict]:
    rows = []
    stages = [
        ("input", "input_amplitude"),
        ("after_input_to_prompt", "after_input_to_prompt"),
        ("after_prompt", "after_prompt"),
        ("expert_entrance_before_aperture", "expert_entrance_before_aperture"),
        ("expert_entrance_after_aperture", "expert_entrance_after_aperture"),
        ("after_global_fc", "after_global_fc"),
        ("detector_plane", "detector_field"),
    ]
    for task_name, diag in diagnostics.items():
        intermediates = diag.get("intermediates", {})
        for stage, key in stages:
            if key not in intermediates:
                continue
            value = intermediates[key]
            if isinstance(value, list):
                value = value[-1]
            energy = torch.abs(value.to(torch.complex64)).square().sum(dim=(-2, -1)).mean()
            rows.append({"run_id": run_id, "task_name": task_name, "stage": stage, "total_energy": float(energy.item())})
    return rows


def copy_source_artifacts_to_run(paths: Mapping[str, Path], run_dir: Path) -> None:
    shutil.copy2(paths["config"], run_dir / "source_config.yaml")
    if paths["architecture_report"].exists():
        shutil.copy2(paths["architecture_report"], run_dir / "source_architecture_report.json")


def save_transfer_architecture_report(model, source_config: Mapping, transfer_config: Mapping, run_dir: Path) -> Dict:
    report = {
        "experiment_family": "transfer_adaptation",
        "source_model_type": source_config.get("model", {}).get("type"),
        "task_names": list(model.task_names),
        "task_num_classes": dict(model.task_num_classes),
        "shared_backbone_frozen": True,
        "target_prompt_only_default": True,
        "task_head_configs": getattr(model, "task_head_configs", {}),
        "task_detector_configs": model.task_detector_configs() if hasattr(model, "task_detector_configs") else {},
        "task_readout_parameter_counts": model.task_readout_parameter_counts() if hasattr(model, "task_readout_parameter_counts") else {},
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "transfer": dict(transfer_config.get("transfer", {}) or {}),
    }
    if hasattr(model, "layout"):
        report["layout"] = model.layout.to_dict()
        report["prompt_channel_table"] = model.prompt_bank.channel_table()
    save_json(report, run_dir / "architecture_report.json")
    return report


def save_epoch_visual_artifacts(model, fixed_batches: Mapping[str, Tuple[torch.Tensor, torch.Tensor]], run_dir: Path, epoch_name: str, task_names: Sequence[str], device, class_names: Mapping, enabled: bool = True) -> Dict:
    if not enabled:
        return {}
    diagnostics = ds_train.save_epoch_artifacts(model, fixed_batches, run_dir, epoch_name, task_names, device, class_names, enabled=True)
    all_tasks = list(getattr(model, "task_names", task_names))
    grouped_path = run_dir / "figures" / "prompt" / epoch_name / "task_expert_weights_grouped.png"
    if save_task_expert_weights_from_model(model, grouped_path, task_names=all_tasks) and epoch_name == "final_epoch":
        save_task_expert_weights_from_model(
            model,
            run_dir / "figures" / "task_expert_weights_grouped.png",
            task_names=all_tasks,
        )
    return diagnostics


def save_checkpoint_file(path: Path, model, optimizer, epoch: int, metrics: Mapping, config: Mapping) -> None:
    save_checkpoint(path, model, optimizer, epoch, dict(metrics), dict(config))


def write_master_rows(run_dir: Path, rows_by_key: Mapping[str, object]) -> None:
    out = run_dir / "summary_for_master"
    for key, rows in rows_by_key.items():
        save_json(rows, out / f"{key}_rows.json")


def save_final_tables(results_dir: Path, rows_by_key: Mapping[str, Iterable[Mapping]]) -> Dict[str, int]:
    filenames = {
        "runs": "master_transfer_runs.csv",
        "epoch_metrics": "master_transfer_epoch_metrics.csv",
        "final_metrics": "master_transfer_final_metrics.csv",
        "prompt_swap": "master_transfer_prompt_swap.csv",
        "source_retention": "master_transfer_source_retention.csv",
        "prompt_similarity": "master_transfer_prompt_similarity.csv",
        "expert_usage": "master_transfer_expert_usage.csv",
        "model_params": "master_transfer_model_params.csv",
        "scaling": "master_transfer_scaling.csv",
    }
    counts = {}
    results_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in filenames.items():
        rows = list(rows_by_key.get(key, []))
        write_rows(results_dir / filename, rows)
        counts[key] = len(rows)
    return counts


def rebuild_transfer_tables(runs_dir: Path, out_dir: Path) -> Dict[str, int]:
    keys = [
        "runs",
        "epoch_metrics",
        "final_metrics",
        "prompt_swap",
        "source_retention",
        "prompt_similarity",
        "expert_usage",
        "model_params",
        "scaling",
    ]
    rows_by_key = {key: [] for key in keys}
    for key in keys:
        for path in sorted(Path(runs_dir).glob(f"*/summary_for_master/{key}_rows.json")):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                rows_by_key[key].extend(payload)
            elif payload:
                rows_by_key[key].append(payload)
    return save_final_tables(Path(out_dir), rows_by_key)


def load_transfer_run_model(run_dir: Path, checkpoint: str, device):
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    source_config = load_yaml(run_dir / "source_config.yaml")
    source_tasks = validate_source_config(
        source_config,
        config.get("source", {}).get("expected_source_tasks", SOURCE_TASKS),
    )
    source_num_classes = infer_task_num_classes(source_config, source_tasks)
    model = ds_train.build_model(source_config, source_tasks, source_num_classes).to(device)
    target_task = str(config.get("target", {}).get("task_name", config.get("target", {}).get("dataset", {}).get("name", "usps"))).lower()
    target_num_classes = int(TARGET_TASKS.get(target_task, 10))
    add_transfer_target_task(
        model,
        target_task_name=target_task,
        target_num_classes=target_num_classes,
        init_from_source_prompt=str(config.get("transfer", {}).get("init_from_source_prompt", "mnist")).lower(),
        target_head_config=config.get("target_head", {}) or DEFAULT_TARGET_HEAD,
        train_target_prompt=bool(config.get("transfer", {}).get("train_target_prompt", True)),
        train_target_readout=bool(config.get("transfer", {}).get("train_target_readout", False)),
    )
    checkpoint_path = run_dir / "checkpoints" / checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / checkpoint
    payload = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(_extract_state_dict(payload), strict=True)
    if hasattr(model, "set_phase_dropout_active"):
        model.set_phase_dropout_active(False)
    return model, config, source_config, source_tasks, target_task
