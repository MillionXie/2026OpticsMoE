import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_mnist_loaders
from model import D2NNClassifier
from utils import (
    BASE_DIR,
    choose_device,
    environment_info,
    git_info,
    load_yaml,
    make_run_dir,
    phase_dropout_active_for_epoch,
    phase_dropout_settings,
    save_json,
    save_yaml,
    set_seed,
    write_rows,
)
from visualization import (
    confusion_matrix,
    save_confusion_csv,
    save_confusion_matrix,
    save_epoch_artifacts,
    save_training_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a standalone MNIST-4 amplitude-input one-phase D2NN baseline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--disable_visualization", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def build_optimizer(model, config):
    cfg = config.get("optimizer", {})
    opt_type = cfg.get("type", "adamw").lower()
    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.0005))
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer.type: {opt_type}")


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def forward_and_loss(model, images, targets, loss_cfg):
    loss_type = str(loss_cfg.get("type", "detector_plane_mse"))
    scale = float(loss_cfg.get("scale", 100.0 if loss_type == "detector_plane_mse" else 1.0))
    if loss_type == "detector_plane_mse":
        logits, intermediates = model(images, return_intermediates=True)
        target_intensity = model.detector.masks[targets].to(device=images.device, dtype=torch.float32)
        loss = scale * F.mse_loss(intermediates["detector_intensity"], target_intensity, reduction="mean")
        return logits, loss
    if loss_type == "cross_entropy":
        logits = model(images)
        return logits, scale * F.cross_entropy(logits, targets)
    raise ValueError(f"Unsupported loss.type: {loss_type}")


def train_one_epoch(model, loader, loss_cfg, optimizer, device, print_freq=50):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = forward_and_loss(model, images, targets, loss_cfg)
        loss.backward()
        optimizer.step()
        batch = targets.numel()
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += batch
        if print_freq > 0 and step % int(print_freq) == 0:
            print(f"  step {step}/{len(loader)} loss={total_loss / max(1, total_count):.4f} acc={total_correct / max(1, total_count):.4f}")
    return {"loss": total_loss / max(1, total_count), "acc": total_correct / max(1, total_count)}


@torch.no_grad()
def evaluate_model(model, loader, loss_cfg, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits, loss = forward_and_loss(model, images, targets, loss_cfg)
        preds = logits.argmax(dim=1)
        batch = targets.numel()
        total_loss += float(loss.item()) * batch
        total_correct += int((preds == targets).sum().item())
        total_count += batch
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "preds": torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long),
        "targets": torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long),
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def fixed_batch(loader, device, max_items):
    image_parts = []
    target_parts = []
    collected = 0
    for images, targets in loader:
        take = min(len(images), int(max_items) - collected)
        image_parts.append(images[:take])
        target_parts.append(targets[:take])
        collected += take
        if collected >= int(max_items):
            break
    if not image_parts:
        raise RuntimeError("Cannot create visualization batch from an empty loader.")
    return torch.cat(image_parts, dim=0).to(device), torch.cat(target_parts, dim=0).to(device)


def architecture_report(model, config, class_names=None):
    dataset = config.get("dataset", {})
    optics = config.get("optics", {})
    detector = config.get("detector", {})
    readout = config.get("readout", {})
    phase_dropout = phase_dropout_settings(config)
    loss_cfg = config.get("loss", {"type": "detector_plane_mse", "scale": 100.0})
    class_names = list(class_names or [str(index) for index in range(model.num_classes)])
    return {
        "model": "D2NNClassifier",
        "dataset": dataset.get("name", "mnist4"),
        "class_names": class_names,
        "num_classes": int(model.num_classes),
        "input_preprocessing": {
            "mode": dataset.get("preprocess_mode", "resize_then_pad"),
            "resize_size": int(dataset.get("resize_size", 336)),
            "output_size": int(dataset.get("input_size", 400)),
            "interpolation": dataset.get("interpolation", "bicubic"),
            "train_samples_per_class": dataset.get("train_samples_per_class"),
            "test_samples_per_class": dataset.get("test_samples_per_class"),
            "use_full_dataset": bool(dataset.get("use_full_dataset", False)),
            "input_encoding": "grayscale amplitude",
        },
        "input_size": int(optics.get("input_size", 256)),
        "canvas_size": int(optics.get("canvas_size", 400)),
        "phase_mask_size": int(optics.get("phase_mask_size", optics.get("input_size", 256))),
        "phase_mask_mode": optics.get("phase_mask_mode", "centered_local"),
        "phase_mask_region": model.phase_mask_region(),
        "trainable_phase_params_per_layer": int(model.phase_layers[0].raw_phase.numel()),
        "trainable_phase_params_total": int(model.optical_parameter_count()),
        "padding_is_trainable": model.phase_mask_mode == "full_canvas",
        "num_layers": int(optics.get("num_layers", 5)),
        "pixel_size_m": float(optics.get("pixel_size_m", 8.0e-6)),
        "wavelength_m": float(optics.get("wavelength_m", 5.32e-7)),
        "input_to_layer_distance_m": float(optics.get("input_to_layer_distance_m", 0.05)),
        "inter_layer_distance_m": float(optics.get("inter_layer_distance_m", 0.05)),
        "detector_distance_m": float(optics.get("detector_distance_m", 0.05)),
        "k_space_constraint": {
            "enabled": bool(optics.get("k_space_constraint_enabled", False)),
            "theta_max_deg": float(optics.get("theta_max_deg", 1.0)),
            "max_sampled_angle_deg": float(model.detector_prop.max_sampled_angle_deg),
            "pass_fraction": float(model.detector_prop.k_space_pass_fraction),
        },
        "detector_size": int(detector.get("detector_size", 32)),
        "detector_layout": detector.get("layout", "grid"),
        "detector_start_pos_x": int(detector.get("start_pos_x", 75)),
        "detector_start_pos_y": int(detector.get("start_pos_y", 75)),
        "detector_N_det_sets": detector.get("N_det_sets", [2, 2]),
        "detector_steps_x": detector.get("det_steps_x", [150, 150]),
        "detector_steps_y": int(detector.get("det_steps_y", 150)),
        "readout_type": readout.get("type", "mlp"),
        "detector_output_is_final_logits": readout.get("type") == "detector_only",
        "phase_dropout": phase_dropout,
        "loss": {
            "type": loss_cfg.get("type", "detector_plane_mse"),
            "scale": float(loss_cfg.get("scale", 100.0)),
            "target": "full detector-plane class mask" if loss_cfg.get("type", "detector_plane_mse") == "detector_plane_mse" else "class label",
        },
        "parameter_count": {
            "optical": model.optical_parameter_count(),
            "electronic": model.electronic_parameter_count(),
            "total": sum(p.numel() for p in model.parameters()),
        },
    }


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    config = load_yaml(config_path)
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    if args.smoke_test:
        config.setdefault("dataset", {})["smoke_train_size"] = 32
        config.setdefault("dataset", {})["smoke_test_size"] = 16
        config["dataset"]["batch_size"] = min(4, int(config["dataset"].get("batch_size", 128)))
        config.setdefault("training", {})["epochs"] = min(int(config.get("training", {}).get("epochs", 1)), 1)

    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_name = config.get("experiment", {}).get("run_name", f"d2nn_mnist256_{int(time.time())}")
    run_dir = make_run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    save_json(environment_info(), run_dir / "environment.json")
    save_json(git_info(), run_dir / "git_info.json")
    shutil.copy2(config_path, run_dir / "source_config.yaml")

    train_loader, test_loader, class_names = create_mnist_loaders(config, seed=seed, smoke_test=args.smoke_test)
    model = D2NNClassifier(config, num_classes=len(class_names)).to(device)
    optimizer = build_optimizer(model, config)
    loss_cfg = config.get("loss", {"type": "detector_plane_mse", "scale": 100.0})
    phase_dropout = phase_dropout_settings(config)
    save_json(architecture_report(model, config, class_names), run_dir / "architecture_report.json")

    print(f"device: {device}")
    print(f"MNIST train samples={len(train_loader.dataset)} test samples={len(test_loader.dataset)} batch_size={train_loader.batch_size}")
    optics_cfg = config.get("optics", {})
    y0, y1, x0, x1 = model.phase_mask_region()
    print(
        "D2NN geometry: "
        f"input_size={model.input_size}, canvas_size={model.canvas_size}, phase_mask_size={model.phase_mask_size}, "
        f"phase_mask_region=y[{y0}:{y1}], x[{x0}:{x1}]"
    )
    print(f"Loss: type={loss_cfg.get('type', 'detector_plane_mse')} scale={float(loss_cfg.get('scale', 100.0))}")
    print(
        "Trainable phase params: "
        f"per_layer={model.phase_layers[0].raw_phase.numel()}, total={model.optical_parameter_count()}, "
        f"padding_is_trainable={model.phase_mask_mode == 'full_canvas'}"
    )
    print(
        "Distances: "
        f"input_to_layer={float(optics_cfg.get('input_to_layer_distance_m', 0.05))} m, "
        f"inter_layer={float(optics_cfg.get('inter_layer_distance_m', 0.05))} m, "
        f"detector={float(optics_cfg.get('detector_distance_m', 0.05))} m"
    )
    print(
        "K-space constraint: "
        f"enabled={model.detector_prop.k_space_constraint_enabled} "
        f"theta_max_deg={model.detector_prop.theta_max_deg:.4f} "
        f"max_sampled_angle_deg={model.detector_prop.max_sampled_angle_deg:.4f} "
        f"pass_fraction={model.detector_prop.k_space_pass_fraction:.6f}"
    )
    print(
        "Phase dropout: "
        f"enabled={phase_dropout['enabled']} mode={phase_dropout['mode']} p={phase_dropout['p']} "
        f"block_size={phase_dropout['block_size']} start_epoch={phase_dropout['start_epoch']}"
    )

    viz_cfg = config.get("visualization", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))
    viz_interval = int(viz_cfg.get("save_interval_epochs", config.get("training", {}).get("save_interval_epochs", 10)))
    fixed = fixed_batch(test_loader, device, int(viz_cfg.get("num_samples", 4)))
    fixed_final = fixed_batch(test_loader, device, int(viz_cfg.get("final_num_samples", 12)))
    dpi = int(viz_cfg.get("dpi", 150))
    init_artifact_start = time.perf_counter()
    save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", class_names, enabled=viz_enabled, dpi=dpi)
    init_artifact_time = time.perf_counter() - init_artifact_start

    epochs = int(config.get("training", {}).get("epochs", 100))
    print_freq = int(config.get("training", {}).get("print_freq", config.get("experiment", {}).get("print_freq", 50)))
    metrics_rows = []
    best = {"epoch": 0, "test_acc": -1.0, "test_loss": ""}
    run_start = time.perf_counter()
    total_artifact_time = init_artifact_time

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, loss_cfg, optimizer, device, print_freq=print_freq)
        train_time = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        test_metrics = evaluate_model(model, test_loader, loss_cfg, device)
        eval_time = time.perf_counter() - eval_start
        artifact_time = 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "phase_dropout_active": active,
            "phase_dropout_mode": phase_dropout["mode"],
            "phase_dropout_p": phase_dropout["p"],
            "phase_dropout_block_size": phase_dropout["block_size"],
            "epoch_time_sec": 0.0,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "artifact_time_sec": 0.0,
        }
        if test_metrics["acc"] > best["test_acc"]:
            best = {"epoch": epoch, "test_acc": test_metrics["acc"], "test_loss": test_metrics["loss"]}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
            artifact_start = time.perf_counter()
            save_epoch_artifacts(model, fixed, run_dir, "best_epoch", class_names, enabled=viz_enabled, dpi=dpi)
            artifact_time += time.perf_counter() - artifact_start
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if viz_interval > 0 and epoch % viz_interval == 0:
            artifact_start = time.perf_counter()
            save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", class_names, enabled=viz_enabled, dpi=dpi)
            artifact_time += time.perf_counter() - artifact_start
        row["artifact_time_sec"] = artifact_time
        row["epoch_time_sec"] = time.perf_counter() - epoch_start
        total_artifact_time += artifact_time
        metrics_rows.append(row)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
        print(f"epoch {epoch:03d} train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f} phase_dropout={'on' if active else 'off'}")

    final_eval = evaluate_model(model, test_loader, loss_cfg, device)
    matrix = confusion_matrix(final_eval["preds"], final_eval["targets"], num_classes=len(class_names))
    save_confusion_matrix(matrix, run_dir / "figures" / "confusion_matrix.png", class_names)
    save_confusion_csv(matrix, run_dir / "metrics" / "confusion_matrix.csv")
    save_training_curves(metrics_rows, run_dir / "figures" / "training_curves.png")
    final_artifact_start = time.perf_counter()
    save_epoch_artifacts(model, fixed_final, run_dir, "final_epoch", class_names, enabled=viz_enabled, dpi=dpi)
    final_artifact_time = time.perf_counter() - final_artifact_start
    total_artifact_time += final_artifact_time
    total_wall_time = time.perf_counter() - run_start
    total_train_time = sum(float(row["train_time_sec"]) for row in metrics_rows)
    total_eval_time = sum(float(row["eval_time_sec"]) for row in metrics_rows)
    avg_epoch_time = sum(float(row["epoch_time_sec"]) for row in metrics_rows) / max(1, len(metrics_rows))

    final_metrics = {
        "run_name": run_name,
        "best_epoch": best["epoch"],
        "best_test_acc": best["test_acc"],
        "best_test_loss": best["test_loss"],
        "final_test_acc": final_eval["acc"],
        "final_test_loss": final_eval["loss"],
        "total_wall_time_sec": total_wall_time,
        "total_train_time_sec": total_train_time,
        "total_eval_time_sec": total_eval_time,
        "total_artifact_time_sec": total_artifact_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "optical_param_count": model.optical_parameter_count(),
        "electronic_param_count": model.electronic_parameter_count(),
        "total_param_count": sum(p.numel() for p in model.parameters()),
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    save_json({**final_metrics, "architecture": architecture_report(model, config, class_names)}, run_dir / "summary.json")
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
