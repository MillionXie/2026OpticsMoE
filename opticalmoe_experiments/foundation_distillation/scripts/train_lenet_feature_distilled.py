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
from common.utils.config import load_yaml, save_json
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix
from foundation_distillation.electronic_baselines import FeatureDistilledLeNetClassifier
from foundation_distillation.runtime import (
    build_optimizer,
    predict_distillation,
    resolve_cache_dir,
    run_distillation_epoch,
)
from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables
from foundation_distillation.visualization.plot_distillation_curves import save_distillation_curves
from foundation_distillation.visualization.plot_feature_similarity import save_feature_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LeNet teacher-feature distillation diagnostic baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def build_lenet_student(config, num_classes: int, teacher_feature_dim: int):
    student_dim = int(config.get("student", {}).get("feature_dim", 900))
    lenet_dim = int(config.get("lenet", {}).get("output_feature_dim", 900))
    if student_dim != 900 or lenet_dim != student_dim:
        raise ValueError("student.feature_dim and lenet.output_feature_dim must both be 900.")
    return FeatureDistilledLeNetClassifier(
        num_classes=num_classes,
        teacher_feature_dim=teacher_feature_dim,
        lenet_config=config.get("lenet", {}),
        feature_preprocess_config=config.get("feature_preprocess", {}),
        projector_config=config.get("projector", {}),
        classifier_config=config.get("classifier", {}),
    )


def architecture_payload(model, config, dataset_name, teacher_cfg):
    return {
        "model": "FeatureDistilledLeNetClassifier",
        "experiment_variant": "lenet_feature_distillation",
        "dataset_name": dataset_name,
        "teacher_type": teacher_cfg.get("type"),
        "teacher_backend": teacher_cfg.get("resolved_backend", teacher_cfg.get("backend", "auto")),
        "teacher_model_name": teacher_cfg.get("model_name"),
        "teacher_input_mode": teacher_cfg.get("input_mode"),
        "feature_type": teacher_cfg.get("resolved_feature_type", teacher_cfg.get("feature_type", "image_embedding")),
        "teacher_feature_dim": model.teacher_feature_dim,
        "student_model_type": "feature_distilled_lenet",
        "student_backbone_type": "lenet",
        "student_feature_dim": model.student_feature_dim,
        "lenet": model.lenet_config,
        "feature_preprocess": model.feature_preprocess_config,
        "projector": model.projector_config,
        "classifier": model.classifier_config,
        "classifier_input": "semantic_feature",
        "feature_loss_input": "semantic_feature_normalized",
        "leak_loss": "disabled",
        "leak_loss_weight": 0.0,
        "optical_parameter_count": 0,
        "prompt_parameter_count": 0,
        "lenet_parameter_count": model.lenet_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "projector_parameter_count": model.projector_parameter_count(),
        "classifier_parameter_count": model.classifier_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "inference_parameter_count": model.total_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
    }


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

    run_name = config.get("experiment", {}).get("run_name", "feature_distilled_lenet")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    cache_dir = resolve_cache_dir(config["teacher_cache"]["cache_dir"], EXPERIMENTS_ROOT)
    config["teacher_cache"]["cache_dir"] = str(cache_dir)
    if not (cache_dir / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Teacher feature cache not found at {cache_dir}. Run build_teacher_feature_cache.py first."
        )
    bundle = create_cached_distillation_loaders(
        config["dataset"], config["teacher"], config["teacher_cache"], seed=seed
    )
    teacher_metadata = dict(getattr(bundle, "teacher_metadata", {}) or {})
    config.setdefault("teacher", {})["resolved_backend"] = teacher_metadata.get(
        "teacher_backend", config["teacher"].get("backend", "auto")
    )
    config["teacher"]["resolved_feature_type"] = teacher_metadata.get(
        "feature_type", config["teacher"].get("feature_type", "image_embedding")
    )
    config["teacher"]["teacher_feature_dim"] = int(bundle.teacher_feature_dim)
    model = build_lenet_student(config, bundle.num_classes, bundle.teacher_feature_dim).to(device)
    optimizer = build_optimizer(model, config)
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "foundation_distillation", run_name)
    save_run_manifest(run_dir, config, " ".join(sys.argv), REPO_ROOT)
    loader_summary = loader_summary_from_loaders(
        bundle.train_loader, bundle.val_loader, bundle.test_loader, config["dataset"]
    )
    save_json(loader_summary, run_dir / "loader_summary.json")
    architecture = architecture_payload(model, config, config["dataset"].get("name"), config["teacher"])
    save_json(architecture, run_dir / "architecture_report.json")
    print(f"device: {device}")
    print_loader_summary(loader_summary, prefix=str(config["dataset"].get("name")))
    print(
        f"model: feature_distilled_lenet lenet={model.lenet_parameter_count()} "
        f"projector={model.projector_parameter_count()} classifier={model.classifier_parameter_count()}"
    )

    training_cfg = config.get("training", {})
    evaluation_cfg = training_cfg.get("evaluation", {})
    loss_cfg = config.get("loss", {})
    epochs = int(training_cfg.get("epochs", 200))
    print_freq = int(training_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    epoch_rows = []
    best_epoch, best_val_acc, best_val_loss = 0, -math.inf, math.inf
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_distillation_epoch(
            model, bundle.train_loader, device, loss_cfg, optimizer=optimizer,
            print_freq=print_freq, max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = run_distillation_epoch(
            model, bundle.val_loader, device, loss_cfg,
            max_batches=evaluation_cfg.get("max_val_batches"),
        )
        epoch_time = time.perf_counter() - epoch_start
        row = {
            "run_id": run_name,
            "experiment_variant": "lenet_feature_distillation",
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
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        if val_metrics["acc"] > best_val_acc:
            best_epoch, best_val_acc, best_val_loss = epoch, val_metrics["acc"], val_metrics["total_loss"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
        print(
            f"epoch {epoch:03d} | train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"ce={train_metrics['ce_loss']:.4f} feat={train_metrics['feature_loss']:.4f} "
            f"cos={train_metrics['feature_cosine']:.4f} | time={epoch_time:.1f}s"
        )

    test_metrics = run_distillation_epoch(
        model, bundle.test_loader, device, loss_cfg,
        max_batches=evaluation_cfg.get("max_test_batches"),
    )
    predictions, targets, similarities = predict_distillation(
        model, bundle.test_loader, device, max_batches=evaluation_cfg.get("max_test_batches")
    )
    final_metrics = {
        "run_id": run_name,
        "dataset_name": config["dataset"].get("name"),
        "experiment_variant": "lenet_feature_distillation",
        "teacher_type": config["teacher"].get("type"),
        "teacher_backend": config["teacher"].get("resolved_backend"),
        "teacher_model_name": config["teacher"].get("model_name"),
        "teacher_input_mode": config["teacher"].get("input_mode"),
        "feature_type": config["teacher"].get("resolved_feature_type"),
        "teacher_feature_dim": model.teacher_feature_dim,
        "student_model_type": "feature_distilled_lenet",
        "student_backbone_type": "lenet",
        "student_feature_dim": model.student_feature_dim,
        "ce_weight": float(loss_cfg.get("ce_weight", 1.0)),
        "feature_distill_weight": float(loss_cfg.get("feature_distill_weight", 0.5)),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["total_loss"],
        "final_feature_cosine": float(similarities.mean().item()),
        "optical_parameter_count": 0,
        "prompt_parameter_count": 0,
        "lenet_parameter_count": model.lenet_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "projector_parameter_count": model.projector_parameter_count(),
        "classifier_parameter_count": model.classifier_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "inference_parameter_count": model.total_parameter_count(),
        "training_parameter_count": model.total_parameter_count(),
        "total_parameter_count": model.total_parameter_count(),
        "total_wall_time_sec": time.perf_counter() - run_start,
        "run_dir": str(run_dir),
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    matrix = save_confusion_matrix(
        predictions, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png"
    )
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", _confusion_rows(matrix, bundle.class_names))
    similarity_rows = [
        {
            "run_id": run_name,
            "epoch": row["epoch"],
            "train_feature_cosine": row["train_feature_cosine"],
            "val_feature_cosine": row["val_feature_cosine"],
        }
        for row in epoch_rows
    ]
    write_rows(run_dir / "diagnostics" / "feature_similarity.csv", similarity_rows)
    save_distillation_curves(epoch_rows, run_dir / "figures" / "training_curves.png")
    save_feature_similarity(epoch_rows, run_dir / "figures" / "feature_similarity_curve.png")
    summary = {**final_metrics, **architecture, "loader_summary": loader_summary, "architecture": architecture}
    save_json(summary, run_dir / "summary.json")
    save_json([summary], run_dir / "summary_for_master" / "runs_rows.json")
    save_json(epoch_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json([final_metrics], run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json([architecture], run_dir / "summary_for_master" / "model_params_rows.json")
    save_json(similarity_rows, run_dir / "summary_for_master" / "feature_similarity_rows.json")
    save_json([], run_dir / "summary_for_master" / "expert_usage_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_distillation_tables(
            EXPERIMENTS_ROOT / "foundation_distillation" / "runs",
            EXPERIMENTS_ROOT / "foundation_distillation" / "results",
        )
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
