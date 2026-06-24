import argparse
from pathlib import Path

import torch

from data import create_mnist_loaders
from model import D2NNClassifier
from train_d2nn_mnist256 import evaluate_model, fixed_batch
from utils import choose_device, load_yaml, save_json
from visualization import confusion_matrix, save_confusion_csv, save_confusion_matrix, save_epoch_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained standalone MNIST D2NN baseline.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    device = choose_device(args.device)
    _, test_loader, class_names = create_mnist_loaders(config, seed=int(config.get("seed", 7)), smoke_test=False)
    model = D2NNClassifier(config, num_classes=10).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / "checkpoints" / ckpt_path
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device)
    matrix = confusion_matrix(metrics["preds"], metrics["targets"], num_classes=10)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json({"checkpoint": str(ckpt_path), "test_loss": metrics["loss"], "test_acc": metrics["acc"]}, eval_dir / "eval_metrics.json")
    save_confusion_matrix(matrix, eval_dir / "confusion_matrix.png", class_names)
    save_confusion_csv(matrix, eval_dir / "confusion_matrix.csv")
    fixed = fixed_batch(test_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    save_epoch_artifacts(model, fixed, eval_dir, "eval_samples", class_names, enabled=True, dpi=int(config.get("visualization", {}).get("dpi", 150)))
    print(f"test_acc={metrics['acc']:.4f} test_loss={metrics['loss']:.4f}")
    print(f"saved eval outputs to: {eval_dir}")


if __name__ == "__main__":
    main()

