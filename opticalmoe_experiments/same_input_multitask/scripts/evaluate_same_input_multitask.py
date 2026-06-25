import argparse
import sys
from pathlib import Path

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
    evaluate,
    fixed_batch,
    prompt_swap_eval,
    prompt_swap_summary,
    same_input_task_switching,
    save_matrix_plot,
    save_same_input_samples_plot,
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
    test = evaluate(model, test_loader, task_names, criterion, device, max_batches=max_batches)
    fixed = fixed_batch(test_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    same_rows, same_payload = same_input_task_switching(model, fixed, task_names, device, run_dir.name)
    swap_rows = prompt_swap_eval(model, test_loader, task_names, criterion, device, max_batches=max_batches)
    for row in swap_rows:
        row["run_id"] = run_dir.name
    swap_summary = prompt_swap_summary(swap_rows, task_names)
    save_json(test, run_dir / "metrics" / "eval_test_metrics.json")
    write_rows(run_dir / "metrics" / "same_input_task_switching.csv", same_rows)
    save_json(same_payload, run_dir / "metrics" / "same_input_task_switching.json")
    write_rows(run_dir / "metrics" / "prompt_swap_matrix.csv", swap_rows)
    save_json(swap_summary, run_dir / "metrics" / "prompt_swap_summary.json")
    save_same_input_samples_plot(fixed, same_payload, task_names, run_dir / "figures" / "same_input_task_switching_samples.png")
    save_matrix_plot(swap_rows, task_names, run_dir / "figures" / "prompt_swap_matrix.png")
    print(f"saved evaluation metrics under {run_dir / 'metrics'}")


if __name__ == "__main__":
    main()
