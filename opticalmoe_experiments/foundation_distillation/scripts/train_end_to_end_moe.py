import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.datasets import create_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.reporting.metrics_writer import write_rows
from common.reporting.run_manifest import save_run_manifest
from common.training.checkpointing import save_checkpoint
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
from common.utils.config import load_yaml, save_json
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix, save_training_curves
from common.visualization.lightfield_viz import save_light_fields
from common.visualization.mask_viz import save_phase_masks
from common.visualization.prompt_viz import save_prompt_maps
from foundation_distillation.runtime import (
    build_end_to_end_student,
    build_optimizer,
    end_to_end_architecture_payload,
    predict_supervised,
    run_supervised_epoch,
)
from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables
from foundation_distillation.visualization.plot_expert_usage import save_expert_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CE-only detector-feature OpticalMoE baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def _fixed_batch(loader, device, count=4):
    images, labels = next(iter(loader))
    return images[:count].to(device), labels[:count].to(device)


@torch.no_grad()
def _diagnostics(model, images, epoch, run_id):
    model.eval()
    _logits, _feature, intermediates = model(images, return_intermediates=True)
    powers = intermediates["normalized_prompt_powers"].detach().cpu()
    amplitudes = intermediates["prompt_amplitudes"].detach().cpu()
    entrance = intermediates["expert_energy_ratios"].detach().mean(dim=0).cpu()
    labels = [aperture.name for aperture in model.layout.expert_apertures]
    usage_rows = []
    for index, label in enumerate(labels):
        usage_rows.append(
            {
                "run_id": run_id,
                "epoch": int(epoch),
                "expert_index": index,
                "expert_id": label,
                "prompt_amplitude": float(amplitudes[index]),
                "normalized_prompt_power": float(powers[index]),
                "expert_entrance_energy_ratio": float(entrance[index]),
                "outside_energy_ratio": float(intermediates["outside_energy_ratio"].mean().item()),
            }
        )
    energy_rows = []
    for stage, key in (
        ("input", "input_amplitude"),
        ("after_input_to_prompt", "after_input_to_prompt"),
        ("after_prompt", "after_prompt"),
        ("expert_entrance_before_aperture", "expert_entrance_before_aperture"),
        ("expert_entrance_after_aperture", "expert_entrance_after_aperture"),
        ("after_global_fc", "after_global_fc"),
        ("detector_plane", "detector_field"),
    ):
        value = intermediates.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        intensity = tensor.abs().square() if torch.is_complex(tensor) else tensor.float().square()
        if intensity.ndim == 2:
            intensity = intensity.unsqueeze(0)
        energy_rows.append(
            {
                "run_id": run_id,
                "epoch": int(epoch),
                "stage": stage,
                "total_energy": float(intensity.sum((-2, -1)).mean().item()),
            }
        )
    return intermediates, usage_rows, energy_rows


def _save_gray_png(image, path):
    array = (torch.as_tensor(image).detach().cpu().float().clamp(0.0, 1.0) * 255.0).byte().numpy()
    Image.fromarray(array, mode="L").save(path)


@torch.no_grad()
def _save_artifacts(model, fixed_batch, run_dir, epoch_name, enabled=True, class_names=None, dataset_name=""):
    if not enabled:
        return
    images, targets = fixed_batch
    logits, _feature, intermediates = model(images, return_intermediates=True)
    sample_dir = run_dir / "figures" / "light_fields" / epoch_name / "sample_000"
    save_light_fields(intermediates, sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    gray = images[0, 0]
    _save_gray_png(gray, sample_dir / "input_student_gray.png")
    _save_gray_png(gray, sample_dir / "input_amplitude.png")
    target = int(targets[0].item())
    prediction = int(logits[0].argmax().item())
    target_name = class_names[target] if class_names and target < len(class_names) else str(target)
    prediction_name = class_names[prediction] if class_names and prediction < len(class_names) else str(prediction)
    (sample_dir / "label.txt").write_text(
        f"true label: {target} ({target_name})\n"
        f"predicted label: {prediction} ({prediction_name})\n"
        "sample index: fixed visualization batch\n"
        f"dataset name: {dataset_name}\n",
        encoding="utf-8",
    )
    (sample_dir / "prediction.txt").write_text(
        f"predicted label: {prediction} ({prediction_name})\n", encoding="utf-8"
    )
    labels = [aperture.name for aperture in model.layout.expert_apertures]
    save_prompt_maps(intermediates, run_dir / "figures" / "prompt" / epoch_name, expert_labels=labels)
    save_phase_masks(model, run_dir / "figures" / "phase_masks" / epoch_name)


def _confusion_rows(matrix, class_names):
    rows = []
    for index, class_name in enumerate(class_names):
        row = {"true_class": class_name}
        row.update({name: int(matrix[index, column].item()) for column, name in enumerate(class_names)})
        rows.append(row)
    return rows


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.smoke_test:
        config["dataset"]["batch_size"] = int(config["dataset"].get("smoke_batch_size", 1))
        apply_smoke_loader_overrides(config["dataset"])
        config.setdefault("training", {})["max_train_batches"] = 1
        config.setdefault("training", {}).setdefault("evaluation", {})["max_val_batches"] = 1
        config["training"]["evaluation"]["max_test_batches"] = 1

    run_name = config.get("experiment", {}).get("run_name", "end_to_end_optical_moe")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    bundle = create_dataloaders(config["dataset"], seed=seed)
    loader_summary = loader_summary_from_loaders(
        bundle.train_loader, bundle.val_loader, bundle.test_loader, config["dataset"]
    )
    print(f"device: {device}")
    print_loader_summary(loader_summary, prefix=str(config["dataset"].get("name")))
    model = build_end_to_end_student(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    dropout = phase_dropout_settings(config)
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "foundation_distillation", run_name)
    save_run_manifest(run_dir, config, " ".join(sys.argv), REPO_ROOT)
    save_json(loader_summary, run_dir / "loader_summary.json")
    architecture = end_to_end_architecture_payload(model, config, config["dataset"].get("name"))
    save_json(architecture, run_dir / "architecture_report.json")
    print(
        f"model: end_to_end_optical_moe optical={model.optical_parameter_count()} "
        f"classifier={model.classifier_parameter_count()} projector=0"
    )

    visualization_enabled = bool(config.get("visualization", {}).get("enabled", True)) and not args.disable_visualization
    fixed_batch = _fixed_batch(
        bundle.val_loader, device, int(config.get("visualization", {}).get("num_samples", 4))
    )
    _save_artifacts(
        model, fixed_batch, run_dir, "epoch_0000", visualization_enabled,
        bundle.class_names, config["dataset"].get("name", "")
    )
    training_cfg = config.get("training", {})
    evaluation_cfg = training_cfg.get("evaluation", {})
    epochs = int(training_cfg.get("epochs", 200))
    print_freq = int(training_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    save_interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
    epoch_rows, curve_rows, expert_rows, optical_energy_rows = [], [], [], []
    best_epoch, best_val_acc, best_val_loss = 0, -math.inf, math.inf
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        dropout_active = phase_dropout_active_for_epoch(dropout, epoch)
        model.set_phase_dropout_active(dropout_active)
        train_metrics = run_supervised_epoch(
            model,
            bundle.train_loader,
            device,
            optimizer=optimizer,
            print_freq=print_freq,
            max_batches=training_cfg.get("max_train_batches"),
        )
        model.set_phase_dropout_active(False)
        val_metrics = run_supervised_epoch(
            model,
            bundle.val_loader,
            device,
            max_batches=evaluation_cfg.get("max_val_batches"),
        )
        epoch_time = time.perf_counter() - epoch_start
        row = {
            "run_id": run_name,
            "experiment_variant": "end_to_end_ce_baseline",
            "epoch": epoch,
            "train_total_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["loss"],
            "train_feature_loss": 0.0,
            "train_acc": train_metrics["acc"],
            "val_total_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["loss"],
            "val_feature_loss": 0.0,
            "val_acc": val_metrics["acc"],
            "phase_dropout_active": dropout_active,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc"],
            }
        )
        _intermediates, usage, energy = _diagnostics(model, fixed_batch[0], epoch, run_name)
        expert_rows.extend(usage)
        optical_energy_rows.extend(energy)
        if val_metrics["acc"] > best_val_acc:
            best_epoch, best_val_acc, best_val_loss = epoch, val_metrics["acc"], val_metrics["loss"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if visualization_enabled and epoch % save_interval == 0:
            _save_artifacts(
                model, fixed_batch, run_dir, f"epoch_{epoch:04d}", True,
                bundle.class_names, config["dataset"].get("name", "")
            )
        print(
            f"epoch {epoch:03d} | train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"ce={train_metrics['loss']:.4f} | phase_dropout={'on' if dropout_active else 'off'} | time={epoch_time:.1f}s"
        )
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
        write_rows(run_dir / "diagnostics" / "expert_usage.csv", expert_rows)
        write_rows(run_dir / "diagnostics" / "prompt_weights.csv", expert_rows)
        write_rows(run_dir / "diagnostics" / "optical_energy_by_stage.csv", optical_energy_rows)

    model.set_phase_dropout_active(False)
    test_metrics = run_supervised_epoch(
        model,
        bundle.test_loader,
        device,
        max_batches=evaluation_cfg.get("max_test_batches"),
    )
    predictions, targets = predict_supervised(
        model,
        bundle.test_loader,
        device,
        max_batches=evaluation_cfg.get("max_test_batches"),
    )
    total_wall = time.perf_counter() - run_start
    final_metrics = {
        "run_id": run_name,
        "dataset_name": config["dataset"].get("name"),
        "experiment_variant": "end_to_end_ce_baseline",
        "teacher_type": "none",
        "teacher_model_name": "",
        "teacher_input_mode": "",
        "student_model_type": config.get("student", {}).get("model_type"),
        "feature_detector_type": config.get("feature_detector", {}).get("type"),
        "feature_dim": model.optical_feature_dim,
        "teacher_feature_dim": 0,
        "ce_weight": 1.0,
        "feature_distill_weight": 0.0,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["loss"],
        "final_feature_cosine": "",
        "optical_parameter_count": model.optical_parameter_count(),
        "prompt_parameter_count": model.prompt_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "projector_parameter_count": 0,
        "inference_parameter_count": model.total_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
        "total_wall_time_sec": total_wall,
        "run_dir": str(run_dir),
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    matrix = save_confusion_matrix(
        predictions, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png"
    )
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", _confusion_rows(matrix, bundle.class_names))
    save_training_curves(curve_rows, run_dir / "figures" / "training_curves.png")
    save_expert_usage(expert_rows, run_dir / "figures" / "expert_usage_heatmap.png")
    save_expert_usage(expert_rows, run_dir / "figures" / "prompt_weights.png")
    _save_artifacts(
        model, fixed_batch, run_dir, "final_epoch", visualization_enabled,
        bundle.class_names, config["dataset"].get("name", "")
    )
    summary = {
        **final_metrics,
        **architecture,
        "loader_summary": loader_summary,
        "architecture": architecture,
        "phase_dropout": dropout,
    }
    save_json(summary, run_dir / "summary.json")
    save_json([summary], run_dir / "summary_for_master" / "runs_rows.json")
    save_json(epoch_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json([final_metrics], run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json([architecture], run_dir / "summary_for_master" / "model_params_rows.json")
    save_json([], run_dir / "summary_for_master" / "feature_similarity_rows.json")
    save_json(expert_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_distillation_tables(
            EXPERIMENTS_ROOT / "foundation_distillation" / "runs",
            EXPERIMENTS_ROOT / "foundation_distillation" / "results",
        )
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
