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
    model, config, _source_config, source_tasks, target_task = tu.load_transfer_run_model(run_dir, args.checkpoint, device)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    bundle, _summary, dataset_cfg = tu.create_target_loaders(config, seed + 10_000, smoke_test=False)
    criterion = torch.nn.CrossEntropyLoss()
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    rows, summary = tu.target_prompt_swap_rows(
        model,
        bundle.test_loader,
        source_tasks,
        target_task,
        str(dataset_cfg.get("name", target_task)),
        device,
        criterion,
        run_dir.name,
        max_batches=max_test_batches,
    )
    write_rows(run_dir / "metrics" / "target_prompt_swap.csv", rows)
    save_json(summary, run_dir / "metrics" / "target_prompt_swap_summary.json")
    tu.save_target_prompt_swap_plot(rows, run_dir / "figures" / "target_prompt_swap.png")
    save_json(rows, run_dir / "summary_for_master" / "prompt_swap_rows.json")
    print(f"saved target prompt swap to {run_dir / 'metrics' / 'target_prompt_swap.csv'}")


if __name__ == "__main__":
    main()

