import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_loaders
from model import FullOpticalD2NNClassifier
from utils import BASE_DIR, choose_device, environment_info, git_info, load_yaml, save_json, save_yaml, set_seed, write_rows
from visualization import confusion_matrix, save_confusion, save_epoch_artifacts, save_training_curves


def detector_plane_mse_loss(intensity, target_plane, scale, normalize, eps):
    """Full-plane MSE with optional per-sample total-energy matching."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError("loss.detector_plane_mse_normalization_eps must be positive")
    prediction = intensity
    if bool(normalize):
        prediction_energy = prediction.sum(dim=(-2, -1), keepdim=True)
        target_energy = target_plane.sum(dim=(-2, -1), keepdim=True)
        prediction = prediction * target_energy / (prediction_energy + eps)
    return float(scale) * F.mse_loss(prediction, target_plane)


def build_optimizer(model, config):
    cfg = config.get("optimizer", {}); opt_type = str(cfg.get("type", "adam")).strip().lower()
    kwargs = {"lr": float(cfg.get("lr", cfg.get("learning_rate", 0.01))), "weight_decay": float(cfg.get("weight_decay", 0.0))}
    if opt_type == "adam": return torch.optim.Adam(model.parameters(), **kwargs)
    if opt_type == "adamw": return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer.type: {opt_type}. Expected 'adam' or 'adamw'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pure-optical six-layer 360x360 CIFAR-10 D2NN (4 or 10 classes).")
    parser.add_argument("--config", default="configs/cifar10_4class.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-samples-per-class-per-epoch", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-visualization", action="store_true")
    return parser.parse_args()


def forward_loss(model, images, targets, loss_cfg):
    loss_type = str(loss_cfg.get("type", "detector_plane_mse"))
    scale = float(loss_cfg.get("scale", 100.0))
    # Training only needs detector intensity. Do not retain six complex layer
    # fields and their autograd graphs unless visualization explicitly asks.
    logits, items = model(images, return_intermediates=True, capture_layer_fields=False)
    if loss_type == "detector_plane_mse":
        target = model.detector.masks[targets].to(images.device)
        loss = detector_plane_mse_loss(
            items["detector_intensity"], target, scale,
            loss_cfg.get("normalize_detector_plane_mse", False),
            loss_cfg.get("detector_plane_mse_normalization_eps", 1.0e-8),
        )
    elif loss_type == "cross_entropy":
        loss = scale * F.cross_entropy(logits, targets)
    else:
        raise ValueError(f"Unsupported loss.type: {loss_type}")
    return logits, loss


def run_epoch(model, loader, loss_cfg, device, optimizer=None, print_freq=50):
    training = optimizer is not None; model.train(training)
    total_loss = 0.0; correct = 0; count = 0; predictions = []; targets_all = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            logits, loss = forward_loss(model, images, targets, loss_cfg)
            if training: loss.backward(); optimizer.step()
            batch = targets.numel(); total_loss += float(loss) * batch
            prediction = logits.argmax(1); correct += int((prediction == targets).sum()); count += batch
            predictions.append(prediction.detach().cpu()); targets_all.append(targets.detach().cpu())
            if training and print_freq > 0 and step % print_freq == 0:
                print(f"  batch {step}/{len(loader)} loss={total_loss/count:.5f} acc={correct/count:.4f}")
    return {
        "loss": total_loss / max(1, count), "acc": correct / max(1, count),
        "preds": torch.cat(predictions), "targets": torch.cat(targets_all),
    }


def fixed_batch(loader, device, count):
    images, targets = next(iter(loader)); count = min(int(count), len(images))
    return images[:count].to(device), targets[:count].to(device)


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics, "config": config}, path)


def detector_bounds(detector):
    result = []
    for mask in detector.masks.cpu():
        points = mask.nonzero(); y0, x0 = points.min(0).values; y1, x1 = points.max(0).values + 1
        result.append([int(y0), int(y1), int(x0), int(x1)])
    return result


def main():
    args = parse_args(); config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = config_path.resolve() if config_path.is_file() else BASE_DIR / config_path
    config = load_yaml(config_path)
    if args.epochs is not None: config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None: config.setdefault("dataset", {})["batch_size"] = args.batch_size
    if args.train_samples_per_class_per_epoch is not None: config.setdefault("dataset", {})["train_samples_per_class_per_epoch"] = args.train_samples_per_class_per_epoch
    if args.run_name: config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.smoke_test:
        config["dataset"]["batch_size"] = 2; config["dataset"]["num_workers"] = 0; config["training"]["epochs"] = 1
    if args.disable_visualization: config.setdefault("visualization", {})["enabled"] = False
    seed = int(config.get("seed", 7)); set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_dir = BASE_DIR / "runs" / str(config.get("experiment", {}).get("run_name", "fulloptical_6layer360"))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml"); save_json(config, run_dir / "config_resolved.json")
    save_json(environment_info(), run_dir / "environment.json"); save_json(git_info(), run_dir / "git_info.json")
    shutil.copy2(config_path, run_dir / "source_config.yaml"); (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    train_loader, test_loader, class_names = create_loaders(config, seed, args.smoke_test)
    save_json({
        "dataset": config["dataset"].get("name"), "class_names": class_names,
        "source_class_indices": config["dataset"].get("class_indices"),
        "train_samples": len(train_loader.dataset), "test_samples": len(test_loader.dataset),
        "per_class_train_counts": {class_names[index]: count for index, count in train_loader.dataset.class_counts.items()},
        "per_class_test_counts": {class_names[index]: count for index, count in test_loader.dataset.class_counts.items()},
        "train_samples_per_class": config["dataset"].get("train_samples_per_class"),
        "test_samples_per_class": config["dataset"].get("test_samples_per_class"),
        "batch_size": train_loader.batch_size, "shuffle_train": True,
        "train_samples_per_class_per_epoch": config["dataset"].get("train_samples_per_class_per_epoch"), "train_samples_per_epoch": len(train_loader.sampler),
        "sampling_note": "Per-class settings cap the whole dataset; per-class-per-epoch rotates through a smaller balanced subset; batch_size controls optimizer mini-batches.",
        "input": "grayscale amplitude 300x300, centered zero padding to 360x360",
    }, run_dir / "dataset.json")
    model = FullOpticalD2NNClassifier(config, len(class_names)).to(device)
    optimizer = build_optimizer(model, config)
    optics = config["optics"]
    conversion_parameters=model.interlayer_conversion_parameter_count()
    save_json({
        "model": "FullOpticalD2NNClassifier", "classes": class_names,
        "path": "grayscale amplitude -> zero pad -> 6 phase-only planes -> square-law detector regions",
        "input_size": model.input_size, "canvas_size": model.canvas_size, "num_phase_layers": model.num_layers,
        "wavelength_m": optics["wavelength_m"], "pixel_size_m": optics["pixel_size_m"],
        "input_to_layer_distance_m": optics["input_to_layer_distance_m"], "inter_layer_distance_m": optics["inter_layer_distance_m"], "detector_distance_m": optics["detector_distance_m"],
        "optoelectronic_interlayers": {**config.get("optoelectronic_interlayers", {}), "trainable_parameters": conversion_parameters, "conversion_count": 5 if model.optoelectronic_enabled else 0, "sequence": "phase -> propagation -> square detection -> stage-independent affine spatial LayerNorm -> ReLU -> zero-phase amplitude reload" if model.optoelectronic_enabled else "disabled; continuous coherent propagation", "routing_amplitude_reapplied": False},
        "detector_bounds": detector_bounds(model.detector),
        "parameters": {"optical_phase": model.optical_parameter_count(), "interlayer_affine": conversion_parameters, "electronic": model.electronic_parameter_count(), "total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "optimizer": {"type": str(config.get("optimizer", {}).get("type", "adam")).lower(), "lr": optimizer.param_groups[0]["lr"], "weight_decay": float(config.get("optimizer", {}).get("weight_decay", 0.0))},
    }, run_dir / "model.json")
    print(f"device={device} classes={class_names} train={len(train_loader.dataset)} test={len(test_loader.dataset)} batch_size={train_loader.batch_size}")
    print(f"optical phase params={model.optical_parameter_count()}; interlayer affine electronic params={conversion_parameters}")
    fixed = fixed_batch(test_loader, device, config.get("visualization", {}).get("num_samples", 4))
    enabled = bool(config.get("visualization", {}).get("enabled", True)); interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
    save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", class_names, enabled)
    loss_cfg = config.get("loss", {}); rows = []; best_acc = -1.0; best_epoch = 0; start = time.perf_counter()
    epochs = int(config.get("training", {}).get("epochs", 200)); print_freq = int(config.get("training", {}).get("print_freq", 50))
    dropout = config.get("regularization", {}).get("phase_dropout", {})
    for epoch in range(1, epochs + 1):
        active = bool(dropout.get("enabled", False)) and epoch >= int(dropout.get("start_epoch", 0)); model.set_phase_dropout_active(active)
        epoch_start = time.perf_counter(); train_metrics = run_epoch(model, train_loader, loss_cfg, device, optimizer, print_freq); test_metrics = run_epoch(model, test_loader, loss_cfg, device)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"], "test_loss": test_metrics["loss"], "test_acc": test_metrics["acc"], "lr": optimizer.param_groups[0]["lr"], "phase_dropout_active": active, "epoch_time_sec": time.perf_counter() - epoch_start}
        rows.append(row); write_rows(run_dir / "metrics" / "training_history.csv", rows); save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if row["test_acc"] > best_acc:
            best_acc = row["test_acc"]; best_epoch = epoch; save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config); save_epoch_artifacts(model, fixed, run_dir, "best_epoch", class_names, enabled)
        if interval > 0 and epoch % interval == 0: save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", class_names, enabled)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.5f} train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f}")
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    final = run_epoch(model, test_loader, loss_cfg, device)
    matrix = confusion_matrix(final["preds"], final["targets"], len(class_names))
    save_confusion(matrix, run_dir / "figures" / "confusion_matrix.png", run_dir / "metrics" / "confusion_matrix.csv", class_names)
    save_training_curves(rows, run_dir / "figures" / "training_curves.png"); save_epoch_artifacts(model, fixed, run_dir, "final_best", class_names, enabled)
    write_rows(run_dir / "metrics" / "test_predictions.csv", [
        {"sample_index": index, "true_label": int(target), "true_name": class_names[int(target)], "pred_label": int(pred), "pred_name": class_names[int(pred)], "correct": bool(target == pred)}
        for index, (target, pred) in enumerate(zip(final["targets"].tolist(), final["preds"].tolist()))
    ])
    save_json({"best_epoch": best_epoch, "best_test_acc": best_acc, "final_test_acc": final["acc"], "final_test_loss": final["loss"], "wall_time_sec": time.perf_counter() - start}, run_dir / "metrics" / "final_metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
