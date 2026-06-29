import argparse
import copy
import sys
import time
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.data.datasets import create_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.optics.optical_models import GeneralD2NNClassifier
from common.reporting.metrics_writer import write_rows
from common.reporting.run_manifest import architecture_report, save_run_manifest
from common.training.checkpointing import load_checkpoint, save_checkpoint
from common.training.eval_loop import evaluate, predict_all
from common.training.phase_dropout import phase_dropout_active_for_epoch, phase_dropout_settings
from common.training.task_heads import global_head_defaults, normalize_head_config
from common.training.train_loop import train_one_epoch
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import make_run_dir
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix, save_training_curves
from common.visualization.lightfield_viz import save_light_fields
from common.visualization.mask_viz import save_phase_masks
from dataset_switching.scripts.train_dataset_switching import build_optimizer, rebuild_dataset_switching_tables


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--task", default=None, help="Run only this task from a combined independent config.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def _task_configs(config, selected_task=None):
    tasks = list(config.get("training", {}).get("tasks", []))
    if selected_task is not None:
        selected = str(selected_task).lower()
        tasks = [task for task in tasks if str(task.get("name", "")).lower() == selected]
        if not tasks:
            valid = [str(task.get("name", "")).lower() for task in config.get("training", {}).get("tasks", [])]
            raise ValueError(f"Unknown --task {selected_task!r}; valid tasks: {valid}")
    if not tasks:
        raise ValueError("Independent D2NN config must define at least one training.tasks entry.")
    return tasks


def _effective_task_config(config, task):
    resolved = copy.deepcopy(config)
    for section in ["model", "optics", "detector", "readout", "regularization", "optimizer", "visualization"]:
        override = task.get(section)
        if isinstance(override, dict):
            resolved.setdefault(section, {}).update(copy.deepcopy(override))
    resolved["dataset"] = copy.deepcopy(task["dataset"])
    head = normalize_head_config(task.get("head", {}), global_head_defaults(resolved))
    resolved["detector"] = {
        "detector_size": head["detector_size"],
        "layout": head["detector_layout"],
    }
    resolved["readout"] = {
        "type": head["readout_type"],
        "normalize_detector_energy": head["normalize_detector_energy"],
        "logit_scale": head["logit_scale"],
        "input_norm": head["input_norm"],
        "norm_affine": head["norm_affine"],
        "hidden_dim": head["hidden_dim"],
        "hidden_layers": head["hidden_layers"],
        "activation": head["activation"],
        "dropout": head["dropout"],
    }
    resolved.setdefault("model", {})["type"] = "general_d2nn"
    resolved.setdefault("experiment", {})["dataset"] = str(task["name"]).lower()
    return resolved, head


def build_independent_model(config, num_classes):
    model_cfg = config.get("model", {})
    optics = config.get("optics", {})
    detector = config.get("detector", {})
    readout = config.get("readout", {})
    dropout = phase_dropout_settings(config)
    return GeneralD2NNClassifier(
        num_classes=num_classes,
        canvas_size=int(model_cfg.get("canvas_size", 400)),
        input_size=int(model_cfg.get("input_size", 134)),
        d2nn_phase_grid_size=int(model_cfg.get("d2nn_phase_grid_size", 220)),
        num_layers=int(model_cfg.get("d2nn_num_layers", 5)),
        wavelength_m=float(optics.get("wavelength_m", 532e-9)),
        pixel_size_m=float(optics.get("pixel_size_m", 8e-6)),
        distances_m=optics.get("distances_m", {}),
        phase_param=optics.get("phase_param", "unconstrained"),
        phase_init=optics.get("expert_phase_init", "identity"),
        init_std=float(optics.get("expert_init_std", 0.02)),
        global_fc_phase_mode=optics.get("global_fc_phase_mode", "center_window"),
        global_fc_phase_size=optics.get("global_fc_phase_size", 220),
        global_fc_padding_mode=optics.get("global_fc_padding_mode", "transparent"),
        detector_size=int(detector.get("detector_size", 32)),
        detector_layout=detector.get("layout", "grid"),
        normalize_detector_energy=bool(readout.get("normalize_detector_energy", True)),
        readout_type=readout.get("type", "optical_only"),
        logit_scale=float(readout.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout.get("hidden_dim", 64)),
        readout_activation=readout.get("activation", "gelu"),
        readout_input_norm=readout.get("input_norm", "layernorm"),
        readout_norm_affine=bool(readout.get("norm_affine", False)),
        readout_hidden_layers=int(readout.get("hidden_layers", 1)),
        readout_dropout=float(readout.get("dropout", 0.0)),
        phase_dropout_mode=dropout["expert_mode"],
        phase_dropout_p=dropout["expert_p"],
        global_fc_phase_dropout_mode=dropout["global_fc_mode"],
        global_fc_phase_dropout_p=dropout["global_fc_p"],
        phase_dropout_block_size=dropout["block_size"],
        phase_dropout_batch_shared=dropout["batch_shared"],
        evanescent_mode=optics.get("evanescent_mode", "zero"),
    )


@torch.no_grad()
def _save_artifacts(model, batch, out_dir):
    images, _targets = batch
    model.eval()
    _logits, intermediates = model(images, return_intermediates=True)
    save_light_fields(intermediates, out_dir / "light_fields" / "sample_000")
    save_phase_masks(model, out_dir / "phase_masks")


def _parameter_report(model, config, reference_task_count):
    model_cfg = config.get("model", {})
    optical = int(model.optical_parameter_count())
    local = int(model.d2nn_local_phase_parameter_count())
    global_fc = int(model.d2nn_global_fc_parameter_count())
    planned_total = optical * int(reference_task_count)
    reference = model_cfg.get("moe_reference_optical_params")
    ratio = float(planned_total) / float(reference) if reference not in {None, "", 0} else ""
    return {
        "canvas_size": int(model.canvas_shape[0]),
        "input_size": int(model.input_size),
        "d2nn_phase_grid_size": int(model.d2nn_phase_grid_size),
        "d2nn_num_layers": int(model.num_layers),
        "d2nn_local_phase_parameter_count": local,
        "global_fc_phase_size": int(model.global_fc.phase_size[0]),
        "global_fc_parameter_count": global_fc,
        "optical_parameter_count": optical,
        "reference_num_tasks": int(reference_task_count),
        "planned_total_independent_optical_params": planned_total,
        "moe_reference_optical_params": reference if reference is not None else "",
        "planned_param_ratio_to_moe": ratio,
        "configured_target_local_phase_param_count": model_cfg.get("target_local_phase_param_count", ""),
        "configured_expected_total_optical_param_count": model_cfg.get("expected_total_optical_param_count", ""),
    }


def run_training(config, args):
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False

    all_tasks = list(config.get("training", {}).get("tasks", []))
    tasks = _task_configs(config, args.task)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_name = config.get("experiment", {}).get("run_name", f"independent_d2nn_{int(time.time())}")
    if args.task and args.run_name is None:
        run_name = f"{run_name}_{str(args.task).lower()}"
    run_dir = make_run_dir(EXPERIMENT_ROOT, "dataset_switching", run_name)
    save_run_manifest(run_dir, config, " ".join(sys.argv), REPO_ROOT)

    reference_task_count = int(config.get("model", {}).get("reference_num_tasks", len(all_tasks) or 3))
    epochs = int(config.get("training", {}).get("epochs", 100))
    if args.smoke_test:
        epochs = 1
    print_freq = int(config.get("training", {}).get("print_freq", 50))
    max_val_batches = config.get("training", {}).get("max_val_batches")
    max_test_batches = config.get("training", {}).get("max_test_batches")
    visualization = config.get("visualization", {})
    viz_enabled = bool(visualization.get("enabled", True))
    save_interval = int(visualization.get("save_interval_epochs", 10))
    num_samples = int(visualization.get("num_samples", 4))

    task_rows = []
    epoch_rows_all = []
    model_param_rows = []
    loader_summaries = {}
    run_start = time.perf_counter()

    for task_index, task in enumerate(tasks):
        task_name = str(task["name"]).lower()
        task_config, resolved_head = _effective_task_config(config, task)
        dataset_cfg = task_config["dataset"]
        if args.smoke_test:
            dataset_cfg["smoke_test"] = True
            dataset_cfg.setdefault("smoke_train_size", 16)
            dataset_cfg.setdefault("smoke_test_size", 8)
            apply_smoke_loader_overrides(dataset_cfg)
        canonical_seed_offset = {"mnist": 0, "fashionmnist": 1, "emnist_letters": 2, "kmnist": 3}
        loader_seed = seed + canonical_seed_offset.get(task_name, task_index)
        bundle = create_dataloaders(dataset_cfg, loader_seed)
        loader_summary = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, dataset_cfg)
        loader_summaries[task_name] = loader_summary
        print_loader_summary(loader_summary, prefix=f"loader/{task_name}")

        model = build_independent_model(task_config, bundle.num_classes).to(device)
        optimizer = build_optimizer(model, task_config)
        criterion = torch.nn.CrossEntropyLoss()
        dropout = phase_dropout_settings(task_config)
        task_dir = run_dir / task_name
        save_yaml(task_config, task_dir / "config_resolved.yaml")
        report = architecture_report(model, task_config, task_dir)
        parameter_report = _parameter_report(model, task_config, reference_task_count)
        report.update(parameter_report)
        report.update({"independent_task": task_name, "resolved_head": resolved_head})
        save_json(report, task_dir / "architecture_report.json")

        print(
            f"task={task_name} canvas={parameter_report['canvas_size']} local_grid={parameter_report['d2nn_phase_grid_size']} "
            f"local_params={parameter_report['d2nn_local_phase_parameter_count']} "
            f"global_fc_params={parameter_report['global_fc_parameter_count']} "
            f"optical_params={parameter_report['optical_parameter_count']}"
        )
        expected = task_config.get("model", {}).get("expected_total_optical_param_count")
        if expected is not None and int(expected) != parameter_report["optical_parameter_count"]:
            raise ValueError(
                f"Configured expected_total_optical_param_count={expected} does not match actual "
                f"{parameter_report['optical_parameter_count']} for {task_name}."
            )

        fixed_images, fixed_targets = next(iter(bundle.val_loader))
        fixed = (fixed_images[:num_samples].to(device), fixed_targets[:num_samples].to(device))
        if viz_enabled:
            _save_artifacts(model, fixed, task_dir / "figures" / "epoch_0000")

        best = {"epoch": 0, "val_acc": -1.0}
        epoch_rows = []
        task_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            active = phase_dropout_active_for_epoch(dropout, epoch)
            model.set_phase_dropout_active(active)
            train = train_one_epoch(model, bundle.train_loader, criterion, optimizer, device, print_freq=print_freq)
            val = evaluate(model, bundle.val_loader, criterion, device, max_batches=1 if args.smoke_test else max_val_batches)
            row = {
                "run_id": run_name,
                "task_name": task_name,
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_acc": train["acc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "lr": optimizer.param_groups[0]["lr"],
                "phase_dropout_active": active,
                "phase_dropout_mode": dropout["mode"],
                "phase_dropout_p": dropout["expert_p"],
                "epoch_time_sec": time.perf_counter() - epoch_start,
            }
            epoch_rows.append(row)
            epoch_rows_all.append(row)
            save_checkpoint(task_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, task_config)
            if val["acc"] > best["val_acc"]:
                best = {"epoch": epoch, "val_acc": val["acc"], "val_loss": val["loss"]}
                save_checkpoint(task_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, task_config)
            if viz_enabled and save_interval > 0 and epoch % save_interval == 0:
                _save_artifacts(model, fixed, task_dir / "figures" / f"epoch_{epoch:04d}")
            write_rows(task_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
            print(
                f"  {task_name} epoch {epoch:03d} train_loss={train['loss']:.4f} train_acc={train['acc']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} time={row['epoch_time_sec']:.1f}s"
            )

        load_checkpoint(task_dir / "checkpoints" / "best.pt", model, map_location=device)
        model.set_phase_dropout_active(False)
        test = evaluate(model, bundle.test_loader, criterion, device, max_batches=1 if args.smoke_test else max_test_batches)
        predictions, targets = predict_all(model, bundle.test_loader, device)
        matrix = save_confusion_matrix(predictions, targets, bundle.class_names, task_dir / "figures" / "confusion_matrix.png")
        write_rows(
            task_dir / "metrics" / "confusion_matrix.csv",
            [{"row": index, **{str(col): int(matrix[index, col]) for col in range(matrix.shape[1])}} for index in range(matrix.shape[0])],
        )
        save_training_curves(epoch_rows, task_dir / "figures" / "training_curves.png")
        if viz_enabled:
            _save_artifacts(model, fixed, task_dir / "figures" / "final_epoch")

        optical = int(model.optical_parameter_count())
        electronic = int(model.electronic_parameter_count())
        total = int(sum(parameter.numel() for parameter in model.parameters()))
        final = {
            "run_id": run_name,
            "task_name": task_name,
            "dataset_name": dataset_cfg.get("name", task_name),
            "independent_group_id": config.get("experiment", {}).get("independent_group_id", "independent_d2nn_canvas400_grid220_seed7"),
            "independent_run_dir": str(task_dir),
            "best_epoch": best["epoch"],
            "best_val_acc": best["val_acc"],
            "test_acc": test["acc"],
            "test_loss": test["loss"],
            "optical_parameter_count": optical,
            "electronic_parameter_count": electronic,
            "total_parameter_count": total,
            "total_wall_time_sec": time.perf_counter() - task_start,
            "is_upper_bound": False,
            **parameter_report,
        }
        save_json(final, task_dir / "metrics" / "final_metrics.json")
        task_rows.append(final)
        model_param_rows.append(
            {
                "run_id": run_name,
                "task_name": task_name,
                **parameter_report,
                "electronic_parameter_count": electronic,
                "total_parameter_count": total,
            }
        )

    executed_optical = sum(int(row["optical_parameter_count"]) for row in task_rows)
    planned_optical = task_rows[0]["planned_total_independent_optical_params"]
    reference = task_rows[0]["moe_reference_optical_params"]
    ratio = float(planned_optical) / float(reference) if reference not in {"", None, 0} else ""
    for row in task_rows:
        row["executed_total_independent_optical_params"] = executed_optical
        row["total_independent_optical_params"] = planned_optical
        row["comparison_to_moe_params_ratio"] = ratio
        row["param_ratio_to_moe"] = ratio

    write_rows(run_dir / "metrics" / "independent_baseline.csv", task_rows)
    save_json(loader_summaries, run_dir / "loader_summary.json")
    save_json(task_rows, run_dir / "summary_for_master" / "independent_baseline_rows.json")
    save_json(epoch_rows_all, run_dir / "summary_for_master" / "epoch_metrics_rows.json")
    save_json(model_param_rows, run_dir / "summary_for_master" / "model_params_rows.json")
    runs_row = {
        "run_id": run_name,
        "model_type": "independent_d2nn",
        "independent_baseline_is_upper_bound": False,
        "tasks_executed": ",".join(row["task_name"] for row in task_rows),
        "reference_num_tasks": reference_task_count,
        "executed_total_independent_optical_params": executed_optical,
        "planned_total_independent_optical_params": planned_optical,
        "moe_reference_optical_params": reference,
        "comparison_to_moe_params_ratio": ratio,
        "total_wall_time_sec": time.perf_counter() - run_start,
        "loader_summary": loader_summaries,
        "run_dir": str(run_dir),
    }
    summary = {**runs_row, "rows": task_rows}
    save_json(summary, run_dir / "summary.json")
    save_json(runs_row, run_dir / "summary_for_master" / "runs_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_dataset_switching_tables(EXPERIMENT_ROOT / "dataset_switching" / "runs", EXPERIMENT_ROOT / "dataset_switching" / "results")
    print(
        f"independent plan: {reference_task_count} x {task_rows[0]['optical_parameter_count']} = {planned_optical}; "
        f"MoE reference={reference}; ratio={ratio if ratio != '' else 'n/a'}"
    )
    print(f"saved independent baseline to {run_dir}")
    return run_dir


def main():
    args = parse_args()
    config = load_yaml(args.config)
    run_training(config, args)


if __name__ == "__main__":
    main()
