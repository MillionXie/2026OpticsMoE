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
from common.reporting.metrics_writer import write_rows
from common.reporting.run_manifest import architecture_report, save_run_manifest
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


def expert_usage_row(run_id, epoch, dataset_name, model_type, model):
    rows = []
    if not hasattr(model, "prompt"):
        return rows
    amps = model.prompt.amplitudes().detach().cpu()
    powers = model.prompt.normalized_powers().detach().cpu()
    labels = [ap.name for ap in model.layout.expert_apertures]
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
                "expert_entrance_energy_ratio": "",
                "expert_output_energy_ratio": "",
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
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()
    phase_dropout = phase_dropout_settings(config)
    architecture_report(model, config, run_dir)

    print(f"device: {device}")
    print(f"dataset: {config.get('dataset', {}).get('name')} classes={bundle.num_classes}")
    print(f"model: {config.get('model', {}).get('type')}")
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
    save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", bundle.class_names, enabled=viz_enabled)

    metrics_rows = []
    usage_rows = []
    best = {"epoch": 0, "val_acc": -1.0}
    epochs = int(config.get("training", {}).get("epochs", 200))
    print_freq = int(config.get("training", {}).get("print_freq", 50))
    eval_max_batches = config.get("training", {}).get("max_val_batches")
    if eval_max_batches is not None:
        eval_max_batches = int(eval_max_batches)

    for epoch in range(1, epochs + 1):
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train_metrics = train_one_epoch(model, bundle.train_loader, criterion, optimizer, device, print_freq=print_freq)
        val_metrics = evaluate(model, bundle.val_loader, criterion, device, max_batches=eval_max_batches)
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
            "epoch_time_sec": train_metrics["time_sec"],
            "outside_energy_ratio": "",
            "mean_expert_entropy": "",
            "max_expert_energy_ratio": "",
            "detector_entropy": "",
        }
        metrics_rows.append(row)
        usage_rows.extend(expert_usage_row(run_name, epoch, config.get("dataset", {}).get("name"), config.get("model", {}).get("type"), model))
        if val_metrics["acc"] > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val_metrics["acc"], "row": row}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
            save_epoch_artifacts(model, fixed, run_dir, "best_epoch", bundle.class_names, enabled=viz_enabled)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if save_interval > 0 and epoch % save_interval == 0:
            save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", bundle.class_names, enabled=viz_enabled)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
        print(f"epoch {epoch:03d} train={row['train_acc']:.4f} val={row['val_acc']:.4f} phase_dropout={'on' if active else 'off'}")

    test_metrics = evaluate(model, bundle.test_loader, criterion, device)
    preds, targets = predict_all(model, bundle.test_loader, device)
    conf = save_confusion_matrix(preds, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png")
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", [{"row": i, **{str(j): int(conf[i, j]) for j in range(conf.shape[1])}} for i in range(conf.shape[0])])
    save_training_curves(metrics_rows, run_dir / "figures" / "training_curves.png")
    save_epoch_artifacts(model, fixed, run_dir, "final_epoch", bundle.class_names, enabled=viz_enabled)

    final_metrics = {
        "run_id": run_name,
        "final_test_loss": test_metrics["loss"],
        "final_test_acc": test_metrics["acc"],
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best.get("row", {}).get("val_loss", ""),
        "train_acc_at_best": best.get("row", {}).get("train_acc", ""),
        "generalization_gap": (best.get("row", {}).get("train_acc", 0.0) - best["val_acc"]) if best["epoch"] else "",
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")

    model_params = {
        "run_id": run_name,
        "optical_param_count": int(model.optical_parameter_count()),
        "prompt_param_count": int(model.prompt_parameter_count()),
        "electronic_param_count": int(model.electronic_parameter_count()),
        "total_param_count": int(sum(p.numel() for p in model.parameters())),
    }
    summary = {
        "run_id": run_name,
        "dataset_name": config.get("dataset", {}).get("name"),
        "model_type": config.get("model", {}).get("type"),
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "phase_dropout": phase_dropout,
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
    }
    save_json(run_row, run_dir / "summary_for_master" / "runs_rows.json")
    save_json(metrics_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json([run_row], run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json(usage_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    save_json([model_params], run_dir / "summary_for_master" / "model_params_rows.json")
    write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_epoch_metrics.csv", metrics_rows)
    write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_expert_usage.csv", usage_rows)
    write_rows(EXPERIMENT_ROOT / "single_task" / "results" / "master_final_metrics.csv", [run_row])
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()

