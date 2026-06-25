import argparse
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.training.checkpointing import load_checkpoint
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device, set_seed
from common.reporting.metrics_writer import write_rows
from dataset_switching.scripts.train_dataset_switching import (
    build_model,
    create_task_loaders,
    prompt_swap_evaluation,
    prompt_swap_summary,
    save_prompt_swap_plot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device)
    train, val, test, task_num_classes, _class_names = create_task_loaders(config, seed, smoke_test=False)
    task_names = list(test)
    model = build_model(config, task_names, task_num_classes).to(device)
    ckpt = run_dir / "checkpoints" / args.checkpoint
    if not ckpt.exists():
        ckpt = run_dir / args.checkpoint
    load_checkpoint(ckpt, model, map_location=device)
    criterion = torch.nn.CrossEntropyLoss()
    rows = prompt_swap_evaluation(
        model,
        test,
        task_names,
        task_num_classes,
        device,
        criterion,
        max_batches=config.get("training", {}).get("evaluation", {}).get("max_test_batches"),
    )
    for row in rows:
        row["run_id"] = run_dir.name
        row["model_type"] = config.get("model", {}).get("type")
    write_rows(run_dir / "metrics" / "prompt_swap_matrix.csv", rows)
    save_json(prompt_swap_summary(rows, task_names), run_dir / "metrics" / "prompt_swap_summary.json")
    save_prompt_swap_plot(rows, run_dir / "figures" / "prompt_swap_matrix.png")
    save_json(rows, run_dir / "summary_for_master" / "prompt_swap_rows.json")
    print(f"saved prompt swap evaluation to {run_dir / 'metrics' / 'prompt_swap_matrix.csv'}")


if __name__ == "__main__":
    main()
