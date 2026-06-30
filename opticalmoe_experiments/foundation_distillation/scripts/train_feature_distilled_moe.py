import argparse
import math
import sys
import time
from pathlib import Path

import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.foundation_distillation import create_cached_distillation_loaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.reporting.metrics_writer import write_rows
from common.reporting.run_manifest import save_run_manifest
from common.training.checkpointing import save_checkpoint
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
from common.utils.config import load_yaml, save_json
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix
from common.visualization.lightfield_viz import save_light_fields
from common.visualization.mask_viz import save_phase_masks
from common.visualization.prompt_viz import save_prompt_maps
from foundation_distillation.runtime import (
    architecture_payload,
    build_optimizer,
    build_student,
    predict_distillation,
    resolve_cache_dir,
    run_distillation_epoch,
)
from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables
from foundation_distillation.visualization.plot_distillation_curves import save_distillation_curves
from foundation_distillation.visualization.plot_expert_usage import save_expert_usage
from foundation_distillation.visualization.plot_feature_similarity import save_feature_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Train detector-feature-distilled OpticalMoE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def _fixed_batch(loader, device, count=4):
    images, labels, teacher, indices = next(iter(loader))
    return images[:count].to(device), labels[:count].to(device), teacher[:count].to(device), indices[:count]


@torch.no_grad()
def _diagnostics(model, images, epoch, run_id):
    model.eval()
    _logits, _feature, _projected, intermediates = model(images, return_intermediates=True)
    powers = intermediates["normalized_prompt_powers"].detach().cpu()
    amplitudes = intermediates["prompt_amplitudes"].detach().cpu()
    entrance = intermediates["expert_energy_ratios"].detach().mean(dim=0).cpu()
    labels = [aperture.name for aperture in model.layout.expert_apertures]
    rows = []
    for index, label in enumerate(labels):
        rows.append(
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
    stage_map = [
        ("input", "input_amplitude"),
        ("after_input_to_prompt", "after_input_to_prompt"),
        ("after_prompt", "after_prompt"),
        ("expert_entrance_before_aperture", "expert_entrance_before_aperture"),
        ("expert_entrance_after_aperture", "expert_entrance_after_aperture"),
        ("after_global_fc", "after_global_fc"),
        ("detector_plane", "detector_field"),
    ]
    energy_rows = []
    for stage, key in stage_map:
        value = intermediates.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        intensity = tensor.abs().square() if torch.is_complex(tensor) else tensor.float().square()
        if intensity.ndim == 2:
            intensity = intensity.unsqueeze(0)
        energy_rows.append({"run_id": run_id, "epoch": int(epoch), "stage": stage, "total_energy": float(intensity.sum((-2, -1)).mean().item())})
    return intermediates, rows, energy_rows


@torch.no_grad()
def _save_artifacts(model, fixed_batch, run_dir, epoch_name, enabled=True):
    if not enabled:
        return
    images, _labels, _teacher, _indices = fixed_batch
    _logits, _feature, _projected, intermediates = model(images, return_intermediates=True)
    save_light_fields(intermediates, run_dir / "figures" / "light_fields" / epoch_name / "sample_000")
    labels = [aperture.name for aperture in model.layout.expert_apertures]
    save_prompt_maps(intermediates, run_dir / "figures" / "prompt" / epoch_name, expert_labels=labels)
    save_phase_masks(model, run_dir / "figures" / "phase_masks" / epoch_name)


def _confusion_rows(matrix, class_names):
    rows = []
    for index, class_name in enumerate(class_names):
        row = {"true_class": class_name}
        row.update({name: int(matrix[index, col].item()) for col, name in enumerate(class_names)})
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
    run_name = config.get("experiment", {}).get("run_name", "feature_distilled_moe")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    cache_dir = resolve_cache_dir(config["teacher_cache"]["cache_dir"], EXPERIMENTS_ROOT)
    config["teacher_cache"]["cache_dir"] = str(cache_dir)
    if not (cache_dir / "metadata.json").exists():
        if bool(config["teacher_cache"].get("build_if_missing", False)):
            from foundation_distillation.scripts.build_teacher_feature_cache import build_cache

            build_cache(config, device, overwrite=False)
        else:
            raise FileNotFoundError(
                f"Teacher feature cache not found at {cache_dir}. Run build_teacher_feature_cache.py first."
            )
    bundle = create_cached_distillation_loaders(
        config["dataset"], config["teacher"], config["teacher_cache"], seed=seed
    )
    teacher_metadata = dict(getattr(bundle, "teacher_metadata", {}) or {})
    if teacher_metadata:
        config["teacher"]["resolved_backend"] = teacher_metadata.get(
            "teacher_backend", config["teacher"].get("backend", "auto")
        )
        config["teacher"]["resolved_feature_type"] = teacher_metadata.get(
            "feature_type", config["teacher"].get("feature_type", "image_embedding")
        )
        config["teacher"]["teacher_feature_dim"] = int(bundle.teacher_feature_dim)
    loader_summary = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, config["dataset"])
    print(f"device: {device}")
    print_loader_summary(loader_summary, prefix=str(config["dataset"].get("name")))
    model = build_student(config, bundle.num_classes, bundle.teacher_feature_dim).to(device)
    optimizer = build_optimizer(model, config)
    dropout = phase_dropout_settings(config)
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "foundation_distillation", run_name)
    save_run_manifest(run_dir, config, " ".join(sys.argv), REPO_ROOT)
    save_json(loader_summary, run_dir / "loader_summary.json")
    architecture = architecture_payload(model, config, config["dataset"].get("name"), config["teacher"])
    save_json(architecture, run_dir / "architecture_report.json")
    print(
        f"model: feature_distilled_optical_moe optical={model.optical_parameter_count()} "
        f"classifier={model.classifier_parameter_count()} projector={model.projector_parameter_count()}"
    )

    visualization_enabled = bool(config.get("visualization", {}).get("enabled", True)) and not args.disable_visualization
    fixed_batch = _fixed_batch(bundle.val_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    _save_artifacts(model, fixed_batch, run_dir, "epoch_0000", visualization_enabled)
    loss_cfg = config.get("loss", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = training_cfg.get("evaluation", {})
    epochs = int(training_cfg.get("epochs", 200))
    print_freq = int(training_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    save_interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
    epoch_rows, expert_rows, optical_energy_rows = [], [], []
    best_epoch, best_val_acc, best_val_loss = 0, -math.inf, math.inf
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        dropout_active = phase_dropout_active_for_epoch(dropout, epoch)
        model.set_phase_dropout_active(dropout_active)
        train_metrics = run_distillation_epoch(
            model,
            bundle.train_loader,
            device,
            loss_cfg,
            optimizer=optimizer,
            print_freq=print_freq,
            max_batches=training_cfg.get("max_train_batches"),
        )
        model.set_phase_dropout_active(False)
        val_metrics = run_distillation_epoch(
            model,
            bundle.val_loader,
            device,
            loss_cfg,
            max_batches=evaluation_cfg.get("max_val_batches"),
        )
        epoch_time = time.perf_counter() - epoch_start
        row = {
            "run_id": run_name,
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_feature_loss": train_metrics["feature_loss"],
            "train_feature_cosine": train_metrics["feature_cosine"],
            "train_acc": train_metrics["acc"],
            "val_total_loss": val_metrics["total_loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_feature_loss": val_metrics["feature_loss"],
            "val_feature_cosine": val_metrics["feature_cosine"],
            "val_acc": val_metrics["acc"],
            "phase_dropout_active": dropout_active,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        intermediates, usage, energy = _diagnostics(model, fixed_batch[0], epoch, run_name)
        expert_rows.extend(usage)
        optical_energy_rows.extend(energy)
        is_best = val_metrics["acc"] > best_val_acc
        if is_best:
            best_epoch, best_val_acc, best_val_loss = epoch, val_metrics["acc"], val_metrics["total_loss"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if visualization_enabled and epoch % save_interval == 0:
            _save_artifacts(model, fixed_batch, run_dir, f"epoch_{epoch:04d}", True)
        print(
            f"epoch {epoch:03d} | train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"ce={train_metrics['ce_loss']:.4f} feat={train_metrics['feature_loss']:.4f} "
            f"cos={train_metrics['feature_cosine']:.4f} | phase_dropout={'on' if dropout_active else 'off'} | time={epoch_time:.1f}s"
        )
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
        write_rows(run_dir / "diagnostics" / "expert_usage.csv", expert_rows)
        write_rows(run_dir / "diagnostics" / "prompt_weights.csv", expert_rows)
        write_rows(run_dir / "diagnostics" / "optical_energy_by_stage.csv", optical_energy_rows)

    model.set_phase_dropout_active(False)
    test_metrics = run_distillation_epoch(
        model, bundle.test_loader, device, loss_cfg, max_batches=evaluation_cfg.get("max_test_batches")
    )
    predictions, targets, similarities = predict_distillation(
        model, bundle.test_loader, device, max_batches=evaluation_cfg.get("max_test_batches")
    )
    total_wall = time.perf_counter() - run_start
    final_metrics = {
        "run_id": run_name,
        "dataset_name": config["dataset"].get("name"),
        "experiment_variant": "feature_distillation",
        "teacher_type": config["teacher"].get("type"),
        "teacher_backend": config["teacher"].get("resolved_backend", config["teacher"].get("backend", "auto")),
        "teacher_model_name": config["teacher"].get("model_name"),
        "feature_type": config["teacher"].get("resolved_feature_type", config["teacher"].get("feature_type", "image_embedding")),
        "teacher_input_mode": config["teacher"].get("input_mode"),
        "student_model_type": config.get("student", {}).get("model_type"),
        "feature_detector_type": config.get("feature_detector", {}).get("type"),
        "feature_dim": model.optical_feature_dim,
        "teacher_feature_dim": model.teacher_feature_dim,
        "ce_weight": float(loss_cfg.get("ce_weight", 1.0)),
        "feature_distill_weight": float(loss_cfg.get("feature_distill_weight", 0.5)),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["total_loss"],
        "final_feature_cosine": test_metrics["feature_cosine"],
        "optical_parameter_count": model.optical_parameter_count(),
        "prompt_parameter_count": model.prompt_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "projector_parameter_count": model.projector_parameter_count(),
        "inference_parameter_count": model.total_parameter_count() - model.projector_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
        "total_wall_time_sec": total_wall,
        "run_dir": str(run_dir),
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    matrix = save_confusion_matrix(predictions, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png")
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", _confusion_rows(matrix, bundle.class_names))
    similarity_rows = [
        {"run_id": run_name, "epoch": row["epoch"], "train_feature_cosine": row["train_feature_cosine"], "val_feature_cosine": row["val_feature_cosine"]}
        for row in epoch_rows
    ]
    write_rows(run_dir / "diagnostics" / "feature_similarity.csv", similarity_rows)
    save_distillation_curves(epoch_rows, run_dir / "figures" / "training_curves.png")
    save_feature_similarity(epoch_rows, run_dir / "figures" / "feature_similarity_curve.png")
    save_expert_usage(expert_rows, run_dir / "figures" / "expert_usage_heatmap.png")
    save_expert_usage(expert_rows, run_dir / "figures" / "prompt_weights.png")
    _save_artifacts(model, fixed_batch, run_dir, "final_epoch", visualization_enabled)
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
    save_json(similarity_rows, run_dir / "summary_for_master" / "feature_similarity_rows.json")
    save_json(expert_rows, run_dir / "summary_for_master" / "expert_usage_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_distillation_tables(
            EXPERIMENTS_ROOT / "foundation_distillation" / "runs",
            EXPERIMENTS_ROOT / "foundation_distillation" / "results",
        )
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
