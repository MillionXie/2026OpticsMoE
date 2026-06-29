import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.foundation_distillation import create_cached_distillation_loaders
from common.reporting.metrics_writer import write_rows
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device
from common.visualization.curve_viz import save_confusion_matrix
from foundation_distillation.runtime import (
    build_student,
    load_checkpoint_state,
    predict_distillation,
    resolve_cache_dir,
    run_distillation_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    device = choose_device(args.device)
    config["teacher_cache"]["cache_dir"] = str(resolve_cache_dir(config["teacher_cache"]["cache_dir"], EXPERIMENTS_ROOT))
    bundle = create_cached_distillation_loaders(config["dataset"], config["teacher"], config["teacher_cache"], seed=int(config.get("seed", 7)))
    model = build_student(config, bundle.num_classes, bundle.teacher_feature_dim).to(device)
    checkpoint = run_dir / "checkpoints" / args.checkpoint
    load_checkpoint_state(model, checkpoint, device)
    metrics = run_distillation_epoch(
        model,
        bundle.test_loader,
        device,
        config.get("loss", {}),
        max_batches=config.get("training", {}).get("evaluation", {}).get("max_test_batches"),
    )
    max_test_batches = config.get("training", {}).get("evaluation", {}).get("max_test_batches")
    predictions, targets, similarities = predict_distillation(model, bundle.test_loader, device, max_batches=max_test_batches)
    payload = {
        "checkpoint": str(checkpoint),
        "test_acc": metrics["acc"],
        "test_loss": metrics["total_loss"],
        "mean_feature_cosine": float(similarities.mean().item()),
        "samples": metrics["samples"],
    }
    save_json(payload, run_dir / "metrics" / "eval_metrics.json")
    matrix = save_confusion_matrix(predictions, targets, bundle.class_names, run_dir / "figures" / "confusion_matrix.png")
    rows = []
    for index, name in enumerate(bundle.class_names):
        row = {"true_class": name}
        row.update({pred_name: int(matrix[index, col]) for col, pred_name in enumerate(bundle.class_names)})
        rows.append(row)
    write_rows(run_dir / "metrics" / "confusion_matrix.csv", rows)
    images, labels, _teacher, _indices = next(iter(bundle.test_loader))
    sample_count = min(8, len(images))
    with torch.no_grad():
        logits, _optical, _projected = model(images[:sample_count].to(device))
    sample_predictions = logits.argmax(dim=1).cpu()
    fig, axes = plt.subplots(2, 4, figsize=(9, 5), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if index >= sample_count:
            continue
        ax.imshow(images[index, 0], cmap="gray")
        ax.set_title(f"true={int(labels[index])} pred={int(sample_predictions[index])}")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "sample_predictions.png", dpi=160)
    plt.close(fig)
    print(payload)


if __name__ == "__main__":
    main()
