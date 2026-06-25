import argparse
import json
import sys
import time
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data.datasets import create_dataloaders
from common.reporting.aggregate_results import rebuild_master_tables
from common.reporting.metrics_writer import write_rows
from common.reporting.run_manifest import architecture_report, save_run_manifest
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.training.checkpointing import save_checkpoint
from common.training.eval_loop import evaluate, predict_all
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
from common.training.train_loop import train_one_epoch
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix, save_training_curves
from common.visualization.lightfield_viz import save_light_fields
from common.visualization.mask_viz import save_expert_phase_layers
from common.visualization.prompt_viz import save_prompt_maps
from baselines.model_factory import build_model, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def fixed_batch(loader, device, max_items=4):
    images, targets = next(iter(loader))
    return images[:max_items].to(device), targets[:max_items].to(device)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir: Path, epoch_name: str, class_names, enabled: bool = True):
    if not enabled:
        return
    images, targets = batch
    model.eval()
    output = model(images, return_intermediates=True)
    if isinstance(output, tuple):
        logits, intermediates = output
    else:
        logits, intermediates = output, {}
    preds = logits.argmax(dim=1)
    labels = getattr(getattr(model, "layout", None), "expert_apertures", None)
    expert_labels = [ap.name for ap in labels] if labels else None
    save_light_fields(intermediates, run_dir / "figures" / "light_fields" / epoch_name / "sample_000")
    save_prompt_maps(intermediates, run_dir / "figures" / "prompt" / epoch_name, expert_labels=expert_labels)
    save_expert_phase_layers(model, run_dir / "figures" / "phase_masks" / epoch_name)
    detector_dir = run_dir / "figures" / "detector_outputs" / epoch_name
    detector_dir.mkdir(parents=True, exist_ok=True)
    if "detector_energies" in intermediates:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = intermediates["detector_energies"][0].detach().cpu()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(values)), values)
        ax.set_title("detector energy sample 000")
        fig.tight_layout()
        fig.savefig(detector_dir / "detector_energy_bar_sample_000.png", dpi=140)
        plt.close(fig)
    samples_dir = run_dir / "figures" / "samples" / epoch_name
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(min(len(targets), 8)):
        rows.append(
            {
                "sample_index": idx,
                "true": int(targets[idx].item()),
                "pred": int(preds[idx].item()),
                "true_name": class_names[int(targets[idx].item())] if int(targets[idx].item()) < len(class_names) else str(int(targets[idx].item())),
                "pred_name": class_names[int(preds[idx].item())] if int(preds[idx].item()) < len(class_names) else str(int(preds[idx].item())),
            }
        )
    save_json(rows, samples_dir / "sample_predictions.json")


def _field_total_energy(field: torch.Tensor) -> torch.Tensor:
    if field.ndim == 4 and field.shape[1] == 1:
        field = field[:, 0]
    if torch.is_complex(field):
        intensity = torch.abs(field).square()
    else:
        intensity = field.float().square()
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    return intensity.sum(dim=(-2, -1))


def _expert_energy_stats(field: torch.Tensor, model):
    if not hasattr(model, "expert_masks"):
        return None
    if field.ndim == 4 and field.shape[1] == 1:
        field = field[:, 0]
    if torch.is_complex(field):
        intensity = torch.abs(field).square()
    else:
        intensity = field.float().square()
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    masks = model.expert_masks.to(device=intensity.device, dtype=intensity.dtype)
    energies = torch.einsum("bhw,khw->bk", intensity, masks)
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12)
    ratios = energies / total.unsqueeze(1)
    inside = energies.sum(dim=1)
    outside_ratio = (total - inside).clamp_min(0.0) / total
    ratio_mean = ratios.mean(dim=0)
    normalized_inside = energies / energies.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(normalized_inside * normalized_inside.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "energies": energies,
        "ratios": ratios,
        "ratio_mean": ratio_mean,
        "outside_ratio": outside_ratio,
        "outside_ratio_mean": outside_ratio.mean(),
        "inside_energy_mean": inside.mean(),
        "total_energy_mean": total.mean(),
        "active_expert_count": int((ratio_mean > 0.02).sum().item()),
        "max_expert_energy_ratio": ratio_mean.max(),
        "energy_entropy": entropy.mean(),
    }


def _mean_entropy(values: torch.Tensor) -> float:
    if values is None:
        return ""
    values = values.detach().float()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    probs = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return float(entropy.mean().item())


@torch.no_grad()
def collect_optical_diagnostics(model, batch, device):
    diagnostics = {"intermediates": {}}
    if not hasattr(model, "forward"):
        return diagnostics
    model.eval()
    images, _targets = batch
    output = model(images.to(device), return_intermediates=True)
    if isinstance(output, tuple):
        _logits, intermediates = output
    else:
        intermediates = {}
    diagnostics["intermediates"] = intermediates

    if "expert_energy_ratios" in intermediates:
        diagnostics["mean_expert_entrance_energy_ratio"] = intermediates["expert_energy_ratios"].detach().mean(dim=0).cpu()
    if "outside_energy_ratio" in intermediates:
        diagnostics["outside_energy_ratio"] = float(intermediates["outside_energy_ratio"].detach().mean().item())
    if "detector_energies" in intermediates:
        diagnostics["detector_entropy"] = _mean_entropy(intermediates["detector_energies"])

    last_expert_field = None
    if intermediates.get("after_each_layer"):
        last_expert_field = intermediates["after_each_layer"][-1]
    elif "after_expert_layer_last" in intermediates:
        last_expert_field = intermediates["after_expert_layer_last"]
    if last_expert_field is not None:
        stats = _expert_energy_stats(last_expert_field, model)
        if stats is not None:
            diagnostics["mean_expert_output_energy_ratio"] = stats["ratio_mean"].detach().cpu()

    entrance_ratios = diagnostics.get("mean_expert_entrance_energy_ratio")
    if entrance_ratios is not None and len(entrance_ratios) > 0:
        probs = entrance_ratios / entrance_ratios.sum().clamp_min(1e-12)
        diagnostics["mean_expert_entropy"] = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
        diagnostics["max_expert_energy_ratio"] = float(entrance_ratios.max().item())
    return diagnostics


def _intermediate_stage_fields(intermediates):
    stages = []
    candidates = [
        ("input", "input_amplitude"),
        ("after_input_to_prompt", "after_input_to_prompt"),
        ("after_prompt", "after_prompt"),
        ("expert_entrance_before_aperture", "expert_entrance_before_aperture"),
        ("expert_entrance_after_aperture", "expert_entrance_after_aperture"),
        ("after_expert_layer_1", "after_expert_layer_1"),
    ]
    for stage, key in candidates:
        if key in intermediates:
            stages.append((stage, intermediates[key]))
    if intermediates.get("after_each_layer"):
        layers = intermediates["after_each_layer"]
        if layers:
            stages.append(("after_expert_layer_last", layers[-1]))
    elif "after_expert_layer_last" in intermediates:
        stages.append(("after_expert_layer_last", intermediates["after_expert_layer_last"]))
    for stage, key in [("after_global_fc", "after_global_fc"), ("detector_plane", "detector_field")]:
        if key in intermediates:
            stages.append((stage, intermediates[key]))
    return stages


def optical_energy_rows_from_intermediates(run_id, epoch, intermediates, model=None):
    if not intermediates:
        return []
    if model is not None and not (hasattr(model, "prompt") or hasattr(model, "global_fc")):
        return []
    rows = []
    has_experts = model is not None and hasattr(model, "expert_masks")
    for stage, field in _intermediate_stage_fields(intermediates):
        total_energy = _field_total_energy(field).mean()
        row = {
            "run_id": run_id,
            "epoch": epoch,
            "stage": stage,
            "total_energy": float(total_energy.item()),
            "inside_expert_energy": "",
            "outside_expert_ratio": "",
            "active_expert_count": "",
            "max_expert_energy_ratio": "",
            "energy_entropy": "",
        }
        if has_experts:
            stats = _expert_energy_stats(field, model)
            if stats is not None:
                row.update(
                    {
                        "inside_expert_energy": float(stats["inside_energy_mean"].item()),
                        "outside_expert_ratio": float(stats["outside_ratio_mean"].item()),
                        "active_expert_count": stats["active_expert_count"],
                        "max_expert_energy_ratio": float(stats["max_expert_energy_ratio"].item()),
                        "energy_entropy": float(stats["energy_entropy"].item()),
                    }
                )
        rows.append(row)
    return rows


def expert_usage_row(run_id, epoch, dataset_name, model_type, model, diagnostics=None):
    rows = []
    if not hasattr(model, "prompt"):
        return rows
    amps = model.prompt.amplitudes().detach().cpu()
    powers = model.prompt.normalized_powers().detach().cpu()
    labels = [ap.name for ap in model.layout.expert_apertures]
    diagnostics = diagnostics or {}
    entrance = diagnostics.get("mean_expert_entrance_energy_ratio")
    output = diagnostics.get("mean_expert_output_energy_ratio")
    for idx, label in enumerate(labels):
        rows.append(
            {
                "run_id": run_id,
                "epoch": epoch,
                "dataset_name": dataset_name,
                "model_type": model_type,
                "expert_id": label,
                "prompt_amplitude": float(amps[idx]),
                "normalized_prompt_power": float(powers[idx]),
                "expert_entrance_energy_ratio": float(entrance[idx].item()) if entrance is not None and idx < len(entrance) else "",
                "expert_output_energy_ratio": float(output[idx].item()) if output is not None and idx < len(output) else "",
            }
        )
    return rows


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.smoke_test:
        config.setdefault("dataset", {})["smoke_test"] = True
        config["dataset"].setdefault("smoke_train_size", 64)
        config["dataset"].setdefault("smoke_test_size", 32)
        apply_smoke_loader_overrides(config["dataset"])
        config.setdefault("training", {})["epochs"] = min(int(config["training"].get("epochs", 1)), 1)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False

    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_name = config.get("experiment", {}).get("run_name") or f"{config.get('dataset', {}).get('name', 'dataset')}_{config.get('model', {}).get('type', 'model')}_{int(time.time())}"
    run_dir = make_run_dir(EXPERIMENT_ROOT, "single_task", run_name)
    command = " ".join(sys.argv)
    save_run_manifest(run_dir, config, command, REPO_ROOT)

    bundle = create_dataloaders(config.get("dataset", {}), seed=seed)
    loader_summary = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, config.get("dataset", {}))
    save_json(loader_summary, run_dir / "loader_summary.json")
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()
    phase_dropout = phase_dropout_settings(config)
    architecture_report(model, config, run_dir)

    print(f"device: {device}")
    print(f"dataset: {config.get('dataset', {}).get('name')} classes={bundle.num_classes}")
    print(f"model: {config.get('model', {}).get('type')}")
    print_loader_summary(loader_summary, prefix="loader")
    print(
        "Phase dropout: "
        f"enabled={phase_dropout['enabled']}, mode={phase_dropout['mode']}, "
        f"expert_p={phase_dropout['expert_p']}, global_fc_p={phase_dropout['global_fc_p']}, "
        f"block_size={phase_dropout['block_size']}, start_epoch={phase_dropout['start_epoch']}"
    )

    viz_cfg = config.get("visualization", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))
    save_interval = int(viz_cfg.get("save_interval_epochs", config.get("training", {}).get("save_interval_epochs", 10)))
    fixed = fixed_batch(bundle.val_loader, device, int(viz_cfg.get("num_samples", 4)))
    run_start_time = time.perf_counter()
    initial_artifact_start = time.perf_counter()
    save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", bundle.class_names, enabled=viz_enabled)
    initial_artifact_time_sec = time.perf_counter() - initial_artifact_start

    metrics_rows = []
    usage_rows = []
    optical_energy_rows = []
    best = {"epoch": 0, "val_acc": -1.0}
    epochs = int(config.get("training", {}).get("epochs", 200))
    print_freq = int(config.get("training", {}).get("print_freq", 50))
    eval_max_batches = config.get("training", {}).get("max_val_batches")
    if eval_max_batches is not None:
        eval_max_batches = int(eval_max_batches)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(model, bundle.train_loader, criterion, optimizer, device, print_freq=print_freq)
        epoch_train_time_sec = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = evaluate(model, bundle.val_loader, criterion, device, max_batches=eval_max_batches)
        epoch_val_time_sec = time.perf_counter() - val_start
        diagnostics = collect_optical_diagnostics(model, fixed, device)
        optical_energy_rows.extend(optical_energy_rows_from_intermediates(run_name, epoch, diagnostics.get("intermediates", {}), model))
        epoch_artifact_time_sec = 0.0
        row = {
            "run_id": run_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "phase_dropout_active": active,
            "phase_dropout_mode": phase_dropout["mode"],
            "expert_phase_dropout_p": phase_dropout["expert_p"],
            "global_fc_phase_dropout_p": phase_dropout["global_fc_p"],
            "phase_dropout_block_size": phase_dropout["block_size"],
            "epoch_train_time_sec": epoch_train_time_sec,
            "epoch_val_time_sec": epoch_val_time_sec,
            "epoch_artifact_time_sec": 0.0,
            "epoch_total_time_sec": 0.0,
            "epoch_time_sec": 0.0,
            "outside_energy_ratio": diagnostics.get("outside_energy_ratio", ""),
            "mean_expert_entropy": diagnostics.get("mean_expert_entropy", ""),
            "max_expert_energy_ratio": diagnostics.get("max_expert_energy_ratio", ""),
            "detector_entropy": diagnostics.get("detector_entropy", ""),
        }
        metrics_rows.append(row)
        usage_rows.extend(expert_usage_row(run_name, epoch, config.get("dataset", {}).get("name"), config.get("model", {}).get("type"), model, diagnostics))
        if val_metrics["acc"] > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val_metrics["acc"], "row": row}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
            artifact_start = time.perf_counter()
            save_epoch_artifacts(model, fixed, run_dir, "best_epoch", bundle.class_names, enabled=viz_enabled)
            epoch_artifact_time_sec += time.perf_counter() - artifact_start
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if save_interval > 0 and epoch % save_interval == 0:
            artifact_start = time.perf_counter()
            save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", bundle.class_names, enabled=viz_enabled)
            epoch_artifact_time_sec += time.perf_counter() - artifact_start
        epoch_total_time_sec = time.perf_counter() - epoch_start
        row["epoch_artifact_time_sec"] = epoch_artifact_time_sec
        row["epoch_total_time_sec"] = epoch_total_time_sec
        row["epoch_time_sec"] = epoch_total_time_sec
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
        print(f"epoch {epoch:03d} train={row['train_acc']:.4f} val={row['val_acc']:.4f} phase_dropout={'on' if active else 'off'}")

    test_start = time.perf_counter()
    test_metrics = evaluate(model, bundle.test_loader, criterion, device)
    test_time_sec = time.perf_counter() - test_start
    preds, targets = predict_all(model, bundle.test_loader, device)
    conf = save_confusion_matrix(preds, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png")
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", [{"row": i, **{str(j): int(conf[i, j]) for j in range(conf.shape[1])}} for i in range(conf.shape[0])])
    save_training_curves(metrics_rows, run_dir / "figures" / "training_curves.png")
    final_artifact_start = time.perf_counter()
    save_epoch_artifacts(model, fixed, run_dir, "final_epoch", bundle.class_names, enabled=viz_enabled)
    final_artifact_time_sec = time.perf_counter() - final_artifact_start
    run_end_time = time.perf_counter()

    total_wall_time_sec = run_end_time - run_start_time
    total_epoch_time_sec = sum(float(row.get("epoch_total_time_sec", row.get("epoch_time_sec", 0.0)) or 0.0) for row in metrics_rows)
    total_train_time_sec = sum(float(row.get("epoch_train_time_sec", 0.0) or 0.0) for row in metrics_rows)
    total_val_time_sec = sum(float(row.get("epoch_val_time_sec", 0.0) or 0.0) for row in metrics_rows)
    total_artifact_time_sec = (
        initial_artifact_time_sec
        + final_artifact_time_sec
        + sum(float(row.get("epoch_artifact_time_sec", 0.0) or 0.0) for row in metrics_rows)
    )
    avg_epoch_time_sec = total_epoch_time_sec / max(1, len(metrics_rows))
    avg_train_time_per_epoch_sec = total_train_time_sec / max(1, len(metrics_rows))
    time_to_best_epoch_sec = sum(
        float(row.get("epoch_total_time_sec", row.get("epoch_time_sec", 0.0)) or 0.0)
        for row in metrics_rows
        if int(row.get("epoch", 0)) <= int(best.get("epoch", 0))
    )
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    time_metrics = {
        "total_wall_time_sec": total_wall_time_sec,
        "total_epoch_time_sec": total_epoch_time_sec,
        "total_train_time_sec": total_train_time_sec,
        "total_val_time_sec": total_val_time_sec,
        "total_artifact_time_sec": total_artifact_time_sec,
        "avg_epoch_time_sec": avg_epoch_time_sec,
        "avg_train_time_per_epoch_sec": avg_train_time_per_epoch_sec,
        "time_to_best_epoch_sec": time_to_best_epoch_sec,
        "test_time_sec": test_time_sec,
        "initial_artifact_time_sec": initial_artifact_time_sec,
        "final_artifact_time_sec": final_artifact_time_sec,
        "completed_at": completed_at,
        "status": "completed",
    }

    final_metrics = {
        "run_id": run_name,
        "final_test_loss": test_metrics["loss"],
        "final_test_acc": test_metrics["acc"],
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best.get("row", {}).get("val_loss", ""),
        "train_acc_at_best": best.get("row", {}).get("train_acc", ""),
        "generalization_gap": (best.get("row", {}).get("train_acc", 0.0) - best["val_acc"]) if best["epoch"] else "",
        **time_metrics,
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")

    global_fc = getattr(model, "global_fc", None)
    layout = getattr(model, "layout", None)
    model_params = {
        "run_id": run_name,
        "optical_param_count": int(model.optical_parameter_count()),
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_param_count": int(model.prompt_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_param_count": int(model.electronic_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_param_count": int(sum(p.numel() for p in model.parameters())),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "global_fc_phase_size": getattr(global_fc, "phase_size", ""),
        "global_fc_parameter_count": int(global_fc.trainable_parameter_count()) if global_fc is not None and hasattr(global_fc, "trainable_parameter_count") else "",
        "global_fc_phase_mode": getattr(global_fc, "phase_mode", ""),
        "global_fc_padding_is_trainable": bool(getattr(global_fc, "phase_mode", "") == "full_canvas") if global_fc is not None else "",
        "active_window_size": getattr(layout, "active_window_size", ""),
        "active_window_region": getattr(layout, "active_window_aperture", None).to_dict() if getattr(layout, "active_window_aperture", None) else "",
        "expert_union_size": getattr(layout, "expert_union_size", ""),
    }
    summary = {
        "run_id": run_name,
        "dataset_name": config.get("dataset", {}).get("name"),
        "model_type": config.get("model", {}).get("type"),
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "phase_dropout": phase_dropout,
        "loader_summary": loader_summary,
        **final_metrics,
        **model_params,
    }
    save_json(summary, run_dir / "summary.json")

    run_row = {
        "run_id": run_name,
        "exp_family": "single_task",
        "dataset_name": config.get("dataset", {}).get("name"),
        "model_type": config.get("model", {}).get("type"),
        "num_experts": config.get("model", {}).get("num_experts", ""),
        "prompt_type": config.get("model", {}).get("prompt_type", ""),
        "routing_type": config.get("model", {}).get("routing_type", ""),
        "input_size": config.get("model", {}).get("input_size", config.get("dataset", {}).get("input_size", "")),
        "canvas_size": config.get("model", {}).get("canvas_size", ""),
        "expert_size": config.get("model", {}).get("expert_size", ""),
        "expert_pitch": config.get("model", {}).get("expert_pitch", ""),
        "num_layers": config.get("model", {}).get("num_layers", ""),
        "detector_layout": config.get("detector", {}).get("layout", ""),
        "readout_type": config.get("readout", {}).get("type", ""),
        "phase_dropout_enabled": phase_dropout["enabled"],
        "phase_dropout_p": phase_dropout["expert_p"],
        "seed": seed,
        **final_metrics,
        **model_params,
        "run_dir": str(run_dir),
        "status": time_metrics["status"],
        "completed_at": time_metrics["completed_at"],
        "loader_summary": loader_summary,
    }
    save_json(run_row, run_dir / "summary_for_master" / "runs_rows.json")
    save_json(metrics_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json([run_row], run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json(usage_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    save_json(optical_energy_rows, run_dir / "summary_for_master" / "optical_energy_rows.json")
    save_json([model_params], run_dir / "summary_for_master" / "model_params_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_master_tables(
            EXPERIMENT_ROOT / "single_task" / "runs",
            EXPERIMENT_ROOT / "single_task" / "results",
        )
    else:
        write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_epoch_metrics.csv", metrics_rows)
        write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_expert_usage.csv", usage_rows)
        write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_optical_energy.csv", optical_energy_rows)
        write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_final_metrics.csv", [run_row])
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
