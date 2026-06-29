import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.datasets import create_dataloaders
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device
from common.visualization.curve_viz import save_confusion_matrix
from foundation_distillation.runtime import (
    build_end_to_end_student,
    load_checkpoint_state,
    predict_supervised,
    run_supervised_epoch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    device = choose_device(args.device)
    bundle = create_dataloaders(config["dataset"], seed=int(config.get("seed", 7)))
    model = build_end_to_end_student(config, bundle.num_classes).to(device)
    checkpoint = run_dir / "checkpoints" / args.checkpoint
    load_checkpoint_state(model, checkpoint, device)
    max_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    metrics = run_supervised_epoch(model, bundle.test_loader, device, max_batches=max_batches)
    predictions, targets = predict_supervised(model, bundle.test_loader, device, max_batches=max_batches)
    payload = {
        "checkpoint": str(checkpoint),
        "test_acc": metrics["acc"],
        "test_loss": metrics["loss"],
        "samples": metrics["samples"],
    }
    save_json(payload, run_dir / "metrics" / "eval_metrics.json")
    save_confusion_matrix(predictions, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png")
    print(payload)


if __name__ == "__main__":
    main()

