import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for path in (EXPERIMENTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.data.foundation_distillation import load_cache_payloads
from common.data.loader_utils import dataloader_kwargs
from common.reporting.metrics_writer import write_rows
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.seed import choose_device, set_seed
from common.visualization.curve_viz import save_confusion_matrix, save_training_curves
from foundation_distillation.runtime import resolve_cache_dir
from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier probe on cached frozen-teacher features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe_type", choices=("matched_mlp", "linear"), default="matched_mlp")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def _activation(name: str) -> nn.Module:
    choices = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    key = str(name).lower()
    if key not in choices:
        raise ValueError(f"Unsupported probe activation: {name!r}.")
    return choices[key]()


def build_probe(input_dim: int, num_classes: int, probe_type: str, classifier_cfg: dict) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(int(input_dim), int(num_classes))
    hidden_layers = int(classifier_cfg.get("hidden_layers", 1))
    hidden_dim = int(classifier_cfg.get("hidden_dim", 128))
    dropout = float(classifier_cfg.get("dropout", 0.1))
    layers = []
    current = int(input_dim)
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(current, hidden_dim), _activation(classifier_cfg.get("activation", "gelu"))])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, int(num_classes)))
    return nn.Sequential(*layers)


def _load_metadata(cache_dir: Path) -> dict:
    path = cache_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Teacher cache metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_loaders(payloads: dict, dataset_cfg: dict, seed: int):
    loaders = {}
    for split in ("train", "val", "test"):
        payload = payloads[split]
        dataset = TensorDataset(
            torch.as_tensor(payload["features"], dtype=torch.float32),
            torch.as_tensor(payload["labels"], dtype=torch.long),
        )
        cfg = dict(dataset_cfg)
        if split != "train":
            cfg["batch_size"] = int(dataset_cfg.get("eval_batch_size", dataset_cfg.get("batch_size", 64)))
        loaders[split] = DataLoader(
            dataset,
            **dataloader_kwargs(cfg, shuffle=split == "train", seed=seed + 700_000 if split == "train" else None),
        )
    return loaders


def _run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    all_predictions, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (features, targets) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            count = int(targets.numel())
            total_loss += float(loss.detach().item()) * count
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            samples += count
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
    denominator = max(samples, 1)
    return {
        "loss": total_loss / denominator,
        "acc": correct / denominator,
        "samples": samples,
        "predictions": torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long),
        "targets": torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long),
    }


def main():
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    cache_dir = resolve_cache_dir(config["teacher_cache"]["cache_dir"], EXPERIMENTS_ROOT)
    metadata = _load_metadata(cache_dir)
    payloads = load_cache_payloads(cache_dir)
    teacher_dim = int(metadata.get("teacher_feature_dim", payloads["train"]["features"].shape[1]))
    class_names = list(metadata.get("class_names", []))
    max_label = max(int(torch.as_tensor(payloads[split]["labels"]).max().item()) for split in payloads)
    num_classes = len(class_names) or max_label + 1
    if not class_names:
        class_names = [str(index) for index in range(num_classes)]

    classifier_cfg = dict(config.get("classifier", {}))
    probe = build_probe(teacher_dim, num_classes, args.probe_type, classifier_cfg).to(device)
    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 5e-4)),
    )
    loaders = _make_loaders(payloads, config.get("dataset", {}), seed)
    epochs = int(args.epochs or config.get("training", {}).get("epochs", 100))
    max_batches = 1 if args.smoke_test else None
    run_name = args.run_name or f"{config.get('dataset', {}).get('name', 'dataset')}_{args.probe_type}_teacher_probe"
    run_dir = EXPERIMENTS_ROOT / "foundation_distillation" / "teacher_probe_runs" / run_name
    for child in ("metrics", "figures", "summary_for_master"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    rows = []
    best_epoch, best_val_acc, best_state = 0, -math.inf, None
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(probe, loaders["train"], device, optimizer, max_batches)
        val_metrics = _run_epoch(probe, loaders["val"], device, max_batches=max_batches)
        row = {
            "run_id": run_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        rows.append(row)
        if val_metrics["acc"] > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_metrics["acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
        print(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

    if best_state is not None:
        probe.load_state_dict(best_state)
    test_metrics = _run_epoch(probe, loaders["test"], device, max_batches=max_batches)
    probe_parameter_count = sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad)
    final_metrics = {
        "run_id": run_name,
        "dataset_name": config.get("dataset", {}).get("name"),
        "teacher_type": metadata.get("teacher_type", config.get("teacher", {}).get("type")),
        "teacher_backend": metadata.get("teacher_backend", config.get("teacher", {}).get("backend", "")),
        "teacher_model_name": metadata.get("teacher_model_name", config.get("teacher", {}).get("model_name")),
        "feature_type": metadata.get("feature_type", config.get("teacher", {}).get("feature_type", "image_embedding")),
        "teacher_feature_dim": teacher_dim,
        "probe_type": args.probe_type,
        "probe_parameter_count": probe_parameter_count,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["loss"],
        "run_dir": str(run_dir),
    }
    write_rows(run_dir / "metrics" / "epoch_metrics.csv", rows)
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    save_training_curves(rows, run_dir / "figures" / "training_curves.png")
    save_confusion_matrix(
        test_metrics["predictions"], test_metrics["targets"], class_names, run_dir / "figures" / "confusion_matrix.png"
    )
    summary = {
        **final_metrics,
        "teacher_cache_dir": str(cache_dir),
        "probe_classifier_config": classifier_cfg if args.probe_type == "matched_mlp" else {"type": "linear"},
    }
    save_json(summary, run_dir / "summary.json")
    save_json([summary], run_dir / "summary_for_master" / "runs_rows.json")
    save_json([final_metrics], run_dir / "summary_for_master" / "final_metrics_rows.json")
    rebuild_distillation_tables(
        EXPERIMENTS_ROOT / "foundation_distillation" / "runs",
        EXPERIMENTS_ROOT / "foundation_distillation" / "results",
    )
    print(f"teacher probe complete: {run_dir}")


if __name__ == "__main__":
    main()
