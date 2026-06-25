import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.data.dsprites_multitask import create_same_input_multitask_dataloaders
from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import load_checkpoint
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device
from same_input_multitask.scripts.train_same_input_multitask import (
    build_model,
    prompt_swap_eval,
    prompt_swap_summary,
    save_matrix_plot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = EXPERIMENT_ROOT / run_dir
    config = load_yaml(run_dir / "config.yaml")
    device = choose_device(args.device)
    _train, _val, test_loader, task_num_classes, task_names = create_same_input_multitask_dataloaders(config, int(config.get("seed", 7)))
    model = build_model(config, task_names, task_num_classes).to(device)
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "best.pt"
    load_checkpoint(ckpt_path, model, map_location=device)
    criterion = nn.CrossEntropyLoss()
    max_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    rows = prompt_swap_eval(model, test_loader, task_names, criterion, device, max_batches=max_batches)
    for row in rows:
        row["run_id"] = run_dir.name
    summary = prompt_swap_summary(rows, task_names)
    write_rows(run_dir / "metrics" / "prompt_swap_matrix.csv", rows)
    save_json(summary, run_dir / "metrics" / "prompt_swap_summary.json")
    save_matrix_plot(rows, task_names, run_dir / "figures" / "prompt_swap_matrix.png")
    print(f"saved prompt swap evaluation to {run_dir / 'metrics' / 'prompt_swap_matrix.csv'}")


if __name__ == "__main__":
    main()
