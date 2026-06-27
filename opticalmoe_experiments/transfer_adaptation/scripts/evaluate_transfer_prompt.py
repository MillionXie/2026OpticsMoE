from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transfer_utils as tu
from common.reporting.metrics_writer import write_rows
from common.utils.config import save_json
from common.utils.seed import choose_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = tu.resolve_path(args.run_dir, prefer_experiment_root=True)
    device = choose_device(args.device)
    model, config, source_config, source_tasks, target_task = tu.load_transfer_run_model(run_dir, args.checkpoint, device)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    target_bundle, _target_summary, _dataset_cfg = tu.create_target_loaders(config, seed + 10_000, smoke_test=False)
    _source_train, _source_val, source_test, _source_nums, _source_class_names, _source_summaries = tu.create_source_task_loaders(
        source_config,
        seed + 20_000,
        smoke_test=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    target_metrics = tu.evaluate_task(model, target_bundle.test_loader, device, criterion, target_task, max_batches=max_test_batches)
    source_after = tu.evaluate_source_tasks(model, source_test, source_tasks, device, criterion, max_batches=max_test_batches)
    rows, summary = tu.source_retention_rows(run_dir.name, source_after, source_after, source_tasks)
    save_json({"run_id": run_dir.name, "target_task": target_task, **target_metrics}, run_dir / "metrics" / "final_target_metrics.json")
    write_rows(run_dir / "metrics" / "source_retention.csv", rows)
    save_json(summary, run_dir / "metrics" / "source_retention_summary.json")
    print(f"target test acc={target_metrics['acc']:.4f}, loss={target_metrics['loss']:.4f}")
    print(f"saved evaluation metrics to {run_dir / 'metrics'}")


if __name__ == "__main__":
    main()

