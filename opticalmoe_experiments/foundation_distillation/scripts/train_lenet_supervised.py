import argparse
import math
import sys
import time
from pathlib import Path

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
from common.utils.config import load_yaml, save_json
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix, save_training_curves
from foundation_distillation.electronic_baselines import SupervisedLeNetClassifier
from foundation_distillation.runtime import build_optimizer, predict_supervised, run_supervised_epoch
from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CE-only supervised LeNet diagnostic baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def build_model(config, num_classes):
    student_dim = int(config.get("student", {}).get("feature_dim", 900))
    lenet_dim = int(config.get("lenet", {}).get("output_feature_dim", 900))
    if student_dim != 900 or lenet_dim != student_dim:
        raise ValueError("student.feature_dim and lenet.output_feature_dim must both be 900.")
    return SupervisedLeNetClassifier(
        num_classes=num_classes,
        lenet_config=config.get("lenet", {}),
        feature_preprocess_config=config.get("feature_preprocess", {}),
        classifier_config=config.get("classifier", {}),
    )


def architecture_payload(model, config, dataset_name):
    return {
        "model": "SupervisedLeNetClassifier",
        "experiment_variant": "lenet_supervised",
        "dataset_name": dataset_name,
        "teacher_type": "none",
        "teacher_model_name": "none",
        "teacher_feature_dim": None,
        "student_model_type": "supervised_lenet",
        "student_backbone_type": "lenet",
        "student_feature_dim": model.student_feature_dim,
        "lenet": model.lenet_config,
        "feature_preprocess": model.feature_preprocess_config,
        "classifier": model.classifier_config,
        "classifier_input": "lenet_feature",
        "feature_distill_weight": 0.0,
        "optical_parameter_count": 0,
        "prompt_parameter_count": 0,
        "lenet_parameter_count": model.lenet_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "projector_parameter_count": 0,
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

    run_name = config.get("experiment", {}).get("run_name", "cifar10_gray_lenet_supervised")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    bundle = create_dataloaders(config["dataset"], seed=seed)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    run_dir = make_run_dir(EXPERIMENTS_ROOT, "foundation_distillation", run_name)
    save_run_manifest(run_dir, config, " ".join(sys.argv), REPO_ROOT)
    loader_summary = loader_summary_from_loaders(
        bundle.train_loader, bundle.val_loader, bundle.test_loader, config["dataset"]
    )
    save_json(loader_summary, run_dir / "loader_summary.json")
    architecture = architecture_payload(model, config, config["dataset"].get("name"))
    save_json(architecture, run_dir / "architecture_report.json")
    print(f"device: {device}")
    print_loader_summary(loader_summary, prefix=str(config["dataset"].get("name")))
    print(
        f"model: supervised_lenet lenet={model.lenet_parameter_count()} "
        f"classifier={model.classifier_parameter_count()} teacher=none"
    )

    training_cfg = config.get("training", {})
    evaluation_cfg = training_cfg.get("evaluation", {})
    epochs = int(training_cfg.get("epochs", 200))
    print_freq = int(training_cfg.get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    epoch_rows, curve_rows = [], []
    best_epoch, best_val_acc, best_val_loss = 0, -math.inf, math.inf
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_supervised_epoch(
            model, bundle.train_loader, device, optimizer=optimizer,
            print_freq=print_freq, max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = run_supervised_epoch(
            model, bundle.val_loader, device, max_batches=evaluation_cfg.get("max_val_batches")
        )
        epoch_time = time.perf_counter() - epoch_start
        row = {
            "run_id": run_name,
            "experiment_variant": "lenet_supervised",
            "epoch": epoch,
            "train_total_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["loss"],
            "train_feature_loss": "",
            "train_feature_cosine": "",
            "train_acc": train_metrics["acc"],
            "val_total_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["loss"],
            "val_feature_loss": "",
            "val_feature_cosine": "",
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        epoch_rows.append(row)
        curve_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],
        })
        if val_metrics["acc"] > best_val_acc:
            best_epoch, best_val_acc, best_val_loss = epoch, val_metrics["acc"], val_metrics["loss"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
        print(
            f"epoch {epoch:03d} | train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"| ce={train_metrics['loss']:.4f} | time={epoch_time:.1f}s"
        )

    test_metrics = run_supervised_epoch(
        model, bundle.test_loader, device, max_batches=evaluation_cfg.get("max_test_batches")
    )
    predictions, targets = predict_supervised(
        model, bundle.test_loader, device, max_batches=evaluation_cfg.get("max_test_batches")
    )
    final_metrics = {
        "run_id": run_name,
        "dataset_name": config["dataset"].get("name"),
        "experiment_variant": "lenet_supervised",
        "teacher_type": "none",
        "teacher_model_name": "none",
        "teacher_feature_dim": None,
        "student_model_type": "supervised_lenet",
        "student_backbone_type": "lenet",
        "student_feature_dim": model.student_feature_dim,
        "ce_weight": 1.0,
        "feature_distill_weight": 0.0,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["loss"],
        "final_feature_cosine": None,
        "optical_parameter_count": 0,
        "prompt_parameter_count": 0,
        "lenet_parameter_count": model.lenet_parameter_count(),
        "feature_preprocess_parameter_count": model.feature_preprocess_parameter_count(),
        "projector_parameter_count": 0,
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
    save_training_curves(curve_rows, run_dir / "figures" / "training_curves.png")
    summary = {**final_metrics, **architecture, "loader_summary": loader_summary, "architecture": architecture}
    save_json(summary, run_dir / "summary.json")
    save_json([summary], run_dir / "summary_for_master" / "runs_rows.json")
    save_json(epoch_rows, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json([final_metrics], run_dir / "summary_for_master" / "final_metrics_rows.json")
    save_json([architecture], run_dir / "summary_for_master" / "model_params_rows.json")
    save_json([], run_dir / "summary_for_master" / "feature_similarity_rows.json")
    save_json([], run_dir / "summary_for_master" / "expert_usage_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_distillation_tables(
            EXPERIMENTS_ROOT / "foundation_distillation" / "runs",
            EXPERIMENTS_ROOT / "foundation_distillation" / "results",
        )
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
