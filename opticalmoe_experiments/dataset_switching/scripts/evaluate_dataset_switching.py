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
from dataset_switching.scripts.train_dataset_switching import build_model, create_task_loaders, evaluate_task


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
    _train, _val, test, task_num_classes, _class_names = create_task_loaders(config, seed, smoke_test=False)
    task_names = list(test)
    model = build_model(config, task_names, task_num_classes).to(device)
    ckpt = run_dir / "checkpoints" / args.checkpoint
    if not ckpt.exists():
        ckpt = run_dir / args.checkpoint
    load_checkpoint(ckpt, model, map_location=device)
    criterion = torch.nn.CrossEntropyLoss()
    rows = []
    for task_name in task_names:
        metrics = evaluate_task(model, test[task_name], device, criterion, task_name)
        rows.append({"task_name": task_name, **metrics})
    save_json(rows, run_dir / "metrics" / "eval_metrics.json")
    print(rows)


if __name__ == "__main__":
    main()
