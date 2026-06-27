from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transfer_utils as tu
from common.reporting.metrics_writer import write_rows
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import write_text
from common.utils.git_info import collect_environment, collect_git_info
from common.utils.seed import choose_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def make_transfer_run_dir(run_name: str) -> Path:
    run_dir = tu.TRANSFER_ROOT / "runs" / run_name
    for child in [
        "checkpoints",
        "metrics",
        "diagnostics",
        "figures/light_fields",
        "figures/prompt",
        "figures/detector_outputs",
        "figures/phase_masks",
        "figures/samples",
        "parameter_freeze",
        "summary_for_master",
    ]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_name(config, args) -> str:
    if args.run_name:
        return str(args.run_name)
    return str(config.get("experiment", {}).get("run_name", f"transfer_prompt_{int(time.time())}"))


def _training_epochs(config, args) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    if args.smoke_test:
        return 1
    return int(config.get("training", {}).get("epochs", 100))


def _source_loader_samples(source_test_loaders, source_tasks):
    return {task: len(source_test_loaders[task].dataset) for task in source_tasks if task in source_test_loaders}


def _print_startup(source_paths, source_tasks, target_task, target_summary, freeze_payload):
    print(f"source checkpoint: {source_paths['checkpoint']}")
    print(f"source tasks: {source_tasks}")
    print(f"target task: {target_task}")
    print(
        "target train/val/test samples: "
        f"{target_summary.get('train_samples')}/"
        f"{target_summary.get('val_samples')}/"
        f"{target_summary.get('test_samples')}"
    )
    print("frozen groups: expert_layers, global_fc, source prompts, source readouts, target readout")
    print("trainable parameter names:")
    for name in freeze_payload["trainable_parameter_names"]:
        print(f"  {name}")


def run_training(config, args):
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False

    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_id = _run_name(config, args)
    run_dir = make_transfer_run_dir(run_id)
    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    save_json(collect_git_info(REPO_ROOT), run_dir / "git_info.json")
    save_json(collect_environment(), run_dir / "environment.json")
    write_text(run_dir / "command.txt", " ".join(sys.argv))

    model, source_config, source_tasks, source_task_num_classes, source_paths, load_info = tu.load_source_backbone(config, device)
    tu.copy_source_artifacts_to_run(source_paths, run_dir)
    if hasattr(model, "set_phase_dropout_active"):
        model.set_phase_dropout_active(False)

    target_task = str(config.get("target", {}).get("task_name", config.get("target", {}).get("dataset", {}).get("name", "usps"))).lower()
    if target_task not in tu.TARGET_TASKS:
        raise ValueError(f"Unsupported transfer target {target_task!r}; use usps or kmnist.")
    target_bundle, target_summary, target_dataset_cfg = tu.create_target_loaders(config, seed + 10_000, smoke_test=args.smoke_test)
    source_train, source_val, source_test, _source_nums, source_class_names, source_loader_summaries = tu.create_source_task_loaders(
        source_config,
        seed + 20_000,
        smoke_test=args.smoke_test,
    )
    del source_train, source_val
    save_json({"target": target_summary, "source": source_loader_summaries}, run_dir / "loader_summary.json")

    target_head = dict(config.get("target_head", {}) or {})
    transfer_cfg = dict(config.get("transfer", {}) or {})
    init_prompt = str(transfer_cfg.get("init_from_source_prompt", "mnist")).lower()
    tu.add_transfer_target_task(
        model,
        target_task_name=target_task,
        target_num_classes=int(target_bundle.num_classes),
        init_from_source_prompt=init_prompt,
        target_head_config=target_head,
        train_target_prompt=bool(transfer_cfg.get("train_target_prompt", True)),
        train_target_readout=bool(transfer_cfg.get("train_target_readout", False)),
    )
    freeze_payload = tu.apply_transfer_freeze_policy(model, target_task, transfer_cfg, run_dir)
    tu.save_transfer_architecture_report(model, source_config, config, run_dir)
    config.setdefault("resolved", {})["source_tasks"] = source_tasks
    config.setdefault("resolved", {})["target_head_config"] = model.task_head_configs[target_task]
    config["resolved"]["source_checkpoint"] = str(source_paths["checkpoint"])
    config["resolved"]["checkpoint_load_info"] = load_info
    save_json(config, run_dir / "config_resolved.json")

    _print_startup(source_paths, source_tasks, target_task, target_summary, freeze_payload)

    criterion = nn.CrossEntropyLoss()
    max_val_batches = config.get("training", {}).get("evaluation", {}).get("max_val_batches")
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    if max_val_batches is not None or max_test_batches is not None:
        print(f"evaluation is limited: max_val_batches={max_val_batches}, max_test_batches={max_test_batches}")

    source_before = tu.evaluate_source_tasks(model, source_test, source_tasks, device, criterion, max_batches=max_test_batches)
    optimizer = tu.build_optimizer(model, config)
    trainable_summary = {
        "optical": freeze_payload["trainable_optical_params"],
        "electronic": freeze_payload["trainable_electronic_params"],
        "total": freeze_payload["total_trainable_params"],
    }
    print(
        "trainable_params: "
        f"optical={trainable_summary['optical']} "
        f"electronic={trainable_summary['electronic']} "
        f"total={trainable_summary['total']}"
    )

    viz_enabled = bool(config.get("visualization", {}).get("enabled", True))
    class_names = dict(source_class_names)
    class_names[target_task] = target_bundle.class_names
    fixed_batches = {
        target_task: tu.fixed_batch(target_bundle.val_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    }
    for task in source_tasks:
        fixed_batches[task] = tu.fixed_batch(source_test[task], device, int(config.get("visualization", {}).get("num_samples", 4)))
    tu.save_epoch_visual_artifacts(model, fixed_batches, run_dir, "epoch_0000", [target_task], device, class_names, enabled=viz_enabled)

    epochs = _training_epochs(config, args)
    best = {"epoch": 0, "target_best_val_acc": -1.0}
    epoch_rows = []
    run_start = time.perf_counter()
    print_freq = int(config.get("training", {}).get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if hasattr(model, "set_phase_dropout_active"):
            model.set_phase_dropout_active(False)
        train = tu.train_target_epoch(model, target_bundle.train_loader, target_task, criterion, optimizer, device, print_freq=print_freq)
        val = tu.evaluate_task(model, target_bundle.val_loader, device, criterion, target_task, max_batches=max_val_batches)
        row = {
            "run_id": run_id,
            "epoch": epoch,
            "target_task": target_task,
            "target_dataset": target_dataset_cfg.get("name", target_task),
            "train_loss": train["loss"],
            "train_acc": train["acc"],
            "train_samples": train["samples"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_samples": val["samples"],
            "phase_dropout_active": False,
            "epoch_time_sec": time.perf_counter() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"trainable_{key}_params": value for key, value in trainable_summary.items()},
        }
        epoch_rows.append(row)
        tu.save_checkpoint_file(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if row["val_acc"] > best["target_best_val_acc"]:
            best = {"epoch": epoch, "target_best_val_acc": row["val_acc"], "row": row}
            tu.save_checkpoint_file(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", epoch_rows)
        print(
            f"epoch {epoch:03d} | target={target_task} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"phase_dropout=off time={row['epoch_time_sec']:.1f}s"
        )
        print(
            "  trainable_params: "
            f"optical={trainable_summary['optical']} "
            f"electronic={trainable_summary['electronic']} "
            f"total={trainable_summary['total']}"
        )
        print(f"  prompt_init: {init_prompt}")
        print(f"  lr={optimizer.param_groups[0]['lr']:.6f}")
        interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
        if viz_enabled and interval > 0 and epoch % interval == 0:
            tu.save_epoch_visual_artifacts(model, fixed_batches, run_dir, f"epoch_{epoch:04d}", [target_task], device, class_names, enabled=True)

    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        payload = torch.load(str(best_path), map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    tu.save_epoch_visual_artifacts(model, fixed_batches, run_dir, "final_epoch", [target_task], device, class_names, enabled=viz_enabled)

    final_target = tu.evaluate_task(model, target_bundle.test_loader, device, criterion, target_task, max_batches=max_test_batches)
    save_json({"run_id": run_id, "target_task": target_task, **final_target}, run_dir / "metrics" / "final_target_metrics.json")

    swap_rows, swap_summary = tu.target_prompt_swap_rows(
        model,
        target_bundle.test_loader,
        source_tasks,
        target_task,
        str(target_dataset_cfg.get("name", target_task)),
        device,
        criterion,
        run_id,
        max_batches=max_test_batches,
    )
    write_rows(run_dir / "metrics" / "target_prompt_swap.csv", swap_rows)
    save_json(swap_summary, run_dir / "metrics" / "target_prompt_swap_summary.json")
    tu.save_target_prompt_swap_plot(swap_rows, run_dir / "figures" / "target_prompt_swap.png")

    source_after = tu.evaluate_source_tasks(model, source_test, source_tasks, device, criterion, max_batches=max_test_batches)
    retention_rows, retention_summary = tu.source_retention_rows(run_id, source_before, source_after, source_tasks)
    write_rows(run_dir / "metrics" / "source_retention.csv", retention_rows)
    save_json(retention_summary, run_dir / "metrics" / "source_retention_summary.json")
    if float(retention_summary["max_source_acc_drop"]) > 1e-4:
        print(f"WARNING: source retention drop exceeded threshold: {retention_summary['max_source_acc_drop']:.6g}")

    similarity_rows = tu.prompt_similarity_rows(model, run_id, target_task, source_tasks)
    write_rows(run_dir / "diagnostics" / "prompt_similarity.csv", similarity_rows)
    tu.save_prompt_similarity_plot(similarity_rows, run_dir / "figures" / "prompt_similarity.png")

    final_diagnostics = {}
    for task in source_tasks + [target_task]:
        final_diagnostics[task] = tu.collect_task_diagnostics(model, fixed_batches[task], device, task)
    usage_rows = tu.expert_usage_rows(model, run_id, final_diagnostics, source_tasks, target_task)
    optical_rows = tu.optical_energy_rows(run_id, final_diagnostics)
    write_rows(run_dir / "diagnostics" / "expert_usage.csv", usage_rows)
    write_rows(run_dir / "diagnostics" / "optical_energy_by_stage.csv", optical_rows)
    tu.save_expert_usage_heatmap(usage_rows, run_dir / "figures" / "source_target_expert_usage_heatmap.png")
    tu.save_expert_usage_heatmap(usage_rows, run_dir / "figures" / "source_target_expert_entrance_energy_ratio_heatmap.png", value_key="expert_entrance_energy_ratio")

    if epoch_rows:
        tu.save_training_curves(
            [
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "train_acc": row["train_acc"],
                    "val_acc": row["val_acc"],
                }
                for row in epoch_rows
            ],
            run_dir / "figures" / "training_curves.png",
        )

    total_wall = time.perf_counter() - run_start
    final_row = {
        "run_id": run_id,
        "source_checkpoint": str(source_paths["checkpoint"]),
        "source_tasks": ",".join(source_tasks),
        "target_task": target_task,
        "target_dataset": target_dataset_cfg.get("name", target_task),
        "target_total_size": (target_dataset_cfg.get("sampling_protocol", {}) or {}).get("total_size", ""),
        "target_train_samples": len(target_bundle.train_loader.dataset),
        "target_val_samples": len(target_bundle.val_loader.dataset),
        "target_test_samples": len(target_bundle.test_loader.dataset),
        "method": transfer_cfg.get("method", "prompt_only"),
        "init_from_source_prompt": init_prompt,
        "train_target_prompt": bool(transfer_cfg.get("train_target_prompt", True)),
        "train_target_readout": bool(transfer_cfg.get("train_target_readout", False)),
        "target_best_epoch": best["epoch"],
        "target_best_val_acc": best["target_best_val_acc"],
        "target_final_test_acc": final_target["acc"],
        "target_final_test_loss": final_target["loss"],
        "target_prompt_accuracy": swap_summary["target_prompt_accuracy"],
        "mean_source_prompt_accuracy": swap_summary["mean_source_prompt_accuracy"],
        "target_prompt_gap": swap_summary["target_prompt_gap"],
        "source_retention_mean_drop": retention_summary["mean_source_acc_drop"],
        "source_retention_max_drop": retention_summary["max_source_acc_drop"],
        "trainable_optical_params": freeze_payload["trainable_optical_params"],
        "trainable_electronic_params": freeze_payload["trainable_electronic_params"],
        "total_trainable_params": freeze_payload["total_trainable_params"],
        "frozen_optical_params": freeze_payload["frozen_optical_params"],
        "total_model_params": freeze_payload["total_model_params"],
        "run_dir": str(run_dir),
    }
    run_row = {
        "run_id": run_id,
        "exp_family": "transfer_adaptation",
        "source_checkpoint": str(source_paths["checkpoint"]),
        "source_tasks": ",".join(source_tasks),
        "target_task": target_task,
        "target_dataset": target_dataset_cfg.get("name", target_task),
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "best_val_acc": best["target_best_val_acc"],
        "total_wall_time_sec": total_wall,
        "source_test_samples": _source_loader_samples(source_test, source_tasks),
    }
    model_params_row = {
        "run_id": run_id,
        "target_task": target_task,
        **{key: freeze_payload[key] for key in [
            "trainable_optical_params",
            "trainable_electronic_params",
            "total_trainable_params",
            "frozen_optical_params",
            "total_model_params",
        ]},
        "task_readout_parameter_counts": model.task_readout_parameter_counts(),
    }
    scaling_row = {
        "run_id": run_id,
        "target_task": target_task,
        "target_total_size": final_row["target_total_size"],
        "target_final_test_acc": final_target["acc"],
        "target_prompt_gap": swap_summary["target_prompt_gap"],
        "total_trainable_params": freeze_payload["total_trainable_params"],
        "trainable_optical_params": freeze_payload["trainable_optical_params"],
        "trainable_electronic_params": freeze_payload["trainable_electronic_params"],
    }
    summary = {
        "run_id": run_id,
        "source_tasks": source_tasks,
        "target_task": target_task,
        "target_dataset": target_dataset_cfg.get("name", target_task),
        "source_checkpoint": str(source_paths["checkpoint"]),
        "source_checkpoint_load_info": load_info,
        "freeze": freeze_payload,
        "best": best,
        "final_target_metrics": final_target,
        "target_prompt_swap_summary": swap_summary,
        "source_retention_summary": retention_summary,
        "trainable_summary": trainable_summary,
        "total_wall_time_sec": total_wall,
    }
    save_json(summary, run_dir / "summary.json")
    tu.write_master_rows(
        run_dir,
        {
            "runs": run_row,
            "epoch_metrics": epoch_rows,
            "final_metrics": [final_row],
            "prompt_swap": swap_rows,
            "source_retention": retention_rows,
            "prompt_similarity": similarity_rows,
            "expert_usage": usage_rows,
            "model_params": [model_params_row],
            "scaling": [scaling_row],
        },
    )
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        counts = tu.rebuild_transfer_tables(tu.TRANSFER_ROOT / "runs", tu.TRANSFER_ROOT / "results")
        print(f"rebuilt transfer master tables: {counts}")
    print(f"saved transfer run outputs to: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    config_path = tu.resolve_path(args.config, prefer_experiment_root=True)
    config = load_yaml(config_path)
    run_training(config, args)


if __name__ == "__main__":
    main()

