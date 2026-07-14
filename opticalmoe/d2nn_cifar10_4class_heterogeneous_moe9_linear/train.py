import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_loaders
from model import HeterogeneousOpticalMoEClassifier
from slm_export import export_best_slm_package
from utils import BASE_DIR, choose_device, environment_info, git_info, load_yaml, save_json, save_yaml, set_seed, write_rows
from visualization import confusion_matrix, save_confusion, save_epoch_artifacts, save_training_curves


def build_optimizer(model, config):
    cfg = config.get("optimizer", {})
    optimizer_type = str(cfg.get("type", "adamw")).lower()
    kwargs = {"lr": float(cfg.get("lr", 0.005)), "weight_decay": float(cfg.get("weight_decay", 0.0))}
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError("optimizer.type must be adam or adamw")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 heterogeneous nine-expert linear optical MoE")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-samples-per-class-per-epoch", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-visualization", action="store_true")
    return parser.parse_args()


def forward_loss(model, images, targets, loss_cfg):
    logits, items = model(images, return_intermediates=True, capture_expert_outputs=False)
    loss_type = str(loss_cfg.get("type", "detector_plane_mse"))
    scale = float(loss_cfg.get("scale", 100.0))
    if loss_type == "detector_plane_mse":
        target_plane = model.detector.masks[targets].to(images.device)
        classification = scale * F.mse_loss(items["detector_intensity"], target_plane)
    elif loss_type == "cross_entropy":
        classification = F.cross_entropy(logits, targets)
    else:
        raise ValueError(f"Unsupported loss.type: {loss_type}")
    balance = items["router_balance_loss"]
    importance = items["router_importance_loss"]
    total = (
        classification
        + float(loss_cfg.get("router_balance_weight", 0.0)) * balance
        + float(loss_cfg.get("router_importance_weight", 0.0)) * importance
    )
    return logits, total, {
        "classification": classification,
        "router_balance": balance,
        "router_importance": importance,
        "router_entropy": items["router_normalized_entropy"],
        "routing_weights": items["routing_weights"],
        "selected_mask": items["routing_selected_mask"],
        "expert_input_power": items["expert_input_power"],
        "expert_output_power": items["expert_output_power"],
    }


def run_epoch(model, loader, loss_cfg, device, optimizer=None, print_freq=0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "classification": 0.0, "balance": 0.0, "importance": 0.0, "entropy": 0.0}
    count = correct = 0
    predictions = []
    targets_all = []
    routing_weight_sum = torch.zeros(9)
    selection_count = torch.zeros(9)
    input_power_sum = torch.zeros(9)
    output_power_sum = torch.zeros(9)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, loss, parts = forward_loss(model, images, targets, loss_cfg)
            if training:
                loss.backward()
                optimizer.step()
            batch = len(targets)
            totals["loss"] += float(loss.item()) * batch
            for key in ("classification", "balance", "importance", "entropy"):
                source = "router_" + key if key != "classification" else key
                totals[key] += float(parts[source].item()) * batch
            predicted = logits.argmax(1)
            correct += int((predicted == targets).sum())
            count += batch
            predictions.append(predicted.detach().cpu())
            targets_all.append(targets.detach().cpu())
            routing_weight_sum += parts["routing_weights"].detach().cpu().sum(0)
            selection_count += parts["selected_mask"].detach().cpu().float().sum(0)
            input_power_sum += parts["expert_input_power"].detach().cpu().sum(0)
            output_power_sum += parts["expert_output_power"].detach().cpu().sum(0)
            if training and print_freq > 0 and step % print_freq == 0:
                print(f"  batch {step}/{len(loader)} loss={totals['loss']/count:.5f} acc={correct/count:.4f}")
    mean_input = input_power_sum / max(1, count)
    mean_output = output_power_sum / max(1, count)
    mean_routing_weights = routing_weight_sum / max(1, count)
    expert_selection_rates = selection_count / max(1, count)
    type_metrics = {}
    for expert_type in sorted(set(model.expert_bank.expert_types)):
        indices = [index for index, value in enumerate(model.expert_bank.expert_types) if value == expert_type]
        type_metrics[expert_type] = {
            "mean_routing_weight": float(mean_routing_weights[indices].mean()),
            "mean_selection_rate": float(expert_selection_rates[indices].mean()),
            "mean_input_power": float(mean_input[indices].mean()),
            "mean_output_power": float(mean_output[indices].mean()),
        }
    return {
        "loss": totals["loss"] / max(1, count),
        "classification_loss": totals["classification"] / max(1, count),
        "router_balance_loss": totals["balance"] / max(1, count),
        "router_importance_loss": totals["importance"] / max(1, count),
        "router_normalized_entropy": totals["entropy"] / max(1, count),
        "acc": correct / max(1, count),
        "preds": torch.cat(predictions),
        "targets": torch.cat(targets_all),
        "mean_routing_weights": mean_routing_weights.tolist(),
        "expert_selection_rates": expert_selection_rates.tolist(),
        "mean_expert_input_power": mean_input.tolist(),
        "mean_expert_output_power": mean_output.tolist(),
        "mean_metrics_by_type": type_metrics,
    }


def fixed_batch(loader, device, count=4):
    images = []
    targets = []
    collected = 0
    for values, labels in loader:
        take = min(len(values), count - collected)
        images.append(values[:take])
        targets.append(labels[:take])
        collected += take
        if collected >= count:
            break
    return torch.cat(images).to(device), torch.cat(targets).to(device)


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
            "expert_parameter_report": model.expert_parameter_report(),
        },
        path,
    )


def architecture_report(model, config, class_names, train_loader, test_loader):
    parameters = model.expert_parameter_report()
    return {
        "model": "HeterogeneousOpticalMoEClassifier",
        "task": "CIFAR-10 four-class heterogeneous linear optical MoE",
        "class_names": class_names,
        "layout": model.layout.to_dict(),
        "expert_type_map_row_major": model.expert_bank.expert_types,
        "expert_parameter_report": parameters,
        "linearity": {
            "complex_field_in_out": True,
            "intensity_detection_inside_experts": False,
            "sample_power_normalization": False,
            "activation_or_reencoding": False,
        },
        "dataset": {
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "train_samples_per_epoch": len(train_loader.sampler),
            "batch_size": train_loader.batch_size,
        },
        "distances_m": config.get("optics", {}).get("distances_m", {}),
        "d2nn_local_propagation": config.get("experts", {}).get("d2nn", {}),
        "routing": {
            "type": "input_topk",
            "top_k": int(config.get("prompt", {}).get("top_k", 3)),
            "balance_weight": float(config.get("loss", {}).get("router_balance_weight", 0.0)),
            "importance_weight": float(config.get("loss", {}).get("router_importance_weight", 0.0)),
        },
        "parameters": {
            "heterogeneous_expert_bank": model.expert_parameter_count(),
            "global_fc_phase": model.global_fc_parameter_count(),
            "optical_total": model.optical_parameter_count(),
            "electronic_router": model.router_parameter_count(),
            "electronic_classifier": 0,
            "trainable_total": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
    }


def main():
    args = parse_args()
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else BASE_DIR / config_path
    config = load_yaml(config_path)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("dataset", {})["batch_size"] = args.batch_size
    if args.train_samples_per_class_per_epoch is not None:
        config.setdefault("dataset", {})["train_samples_per_class_per_epoch"] = args.train_samples_per_class_per_epoch
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.smoke_test:
        config["dataset"]["batch_size"] = 2
        config["dataset"]["num_workers"] = 0
        config["training"]["epochs"] = 1
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    run_dir = BASE_DIR / "runs" / config.get("experiment", {}).get("run_name", "heterogeneous_moe9_linear")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    save_json(environment_info(), run_dir / "environment.json")
    save_json(git_info(), run_dir / "git_info.json")
    shutil.copy2(config_path, run_dir / "source_config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    train_loader, test_loader, class_names = create_loaders(config, seed, args.smoke_test)
    save_json(
        {
            "dataset": "cifar10_4class",
            "class_names": class_names,
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "per_class_train_counts": {class_names[index]: count for index, count in train_loader.dataset.class_counts.items()},
            "per_class_test_counts": {class_names[index]: count for index, count in test_loader.dataset.class_counts.items()},
            "train_samples_per_class_per_epoch": config.get("dataset", {}).get("train_samples_per_class_per_epoch"),
            "train_samples_per_epoch": len(train_loader.sampler),
        },
        run_dir / "dataset.json",
    )
    model = HeterogeneousOpticalMoEClassifier(config, len(class_names)).to(device)
    optimizer = build_optimizer(model, config)
    report = architecture_report(model, config, class_names, train_loader, test_loader)
    save_json(report, run_dir / "architecture_report.json")
    print(f"device={device} train={len(train_loader.dataset)} test={len(test_loader.dataset)} classes={class_names}")
    print(f"expert_types={model.expert_bank.expert_types}")
    print(f"parameters={report['parameters']}")
    loss_cfg = config.get("loss", {})
    fixed = fixed_batch(test_loader, device, int(config.get("visualization", {}).get("num_samples", 4)))
    visualization_enabled = bool(config.get("visualization", {}).get("enabled", True))
    interval = int(config.get("visualization", {}).get("save_interval_epochs", 10))
    save_epoch_artifacts(model, fixed, run_dir, "epoch_0000", class_names, visualization_enabled)
    epochs = int(config.get("training", {}).get("epochs", 200))
    print_freq = int(config.get("training", {}).get("print_freq", 50))
    rows = []
    best_acc = -1.0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, loss_cfg, device, optimizer, print_freq)
        test_metrics = run_epoch(model, test_loader, loss_cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_router_balance_loss": train_metrics["router_balance_loss"],
            "train_router_importance_loss": train_metrics["router_importance_loss"],
            "train_router_normalized_entropy": train_metrics["router_normalized_entropy"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.perf_counter() - epoch_start,
        }
        for index in range(9):
            row[f"train_expert_{index}_selection_rate"] = train_metrics["expert_selection_rates"][index]
            row[f"train_expert_{index}_mean_weight"] = train_metrics["mean_routing_weights"][index]
            row[f"train_expert_{index}_mean_input_power"] = train_metrics["mean_expert_input_power"][index]
            row[f"train_expert_{index}_mean_output_power"] = train_metrics["mean_expert_output_power"][index]
        for expert_type, values in train_metrics["mean_metrics_by_type"].items():
            row[f"train_{expert_type}_mean_selection_rate"] = values["mean_selection_rate"]
            row[f"train_{expert_type}_mean_routing_weight"] = values["mean_routing_weight"]
            row[f"train_{expert_type}_mean_input_power"] = values["mean_input_power"]
            row[f"train_{expert_type}_mean_output_power"] = values["mean_output_power"]
        rows.append(row)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", rows)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, config)
        if row["test_acc"] > best_acc:
            best_acc = row["test_acc"]
            best_epoch = epoch
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, config)
            save_epoch_artifacts(model, fixed, run_dir, "best_epoch", class_names, visualization_enabled)
        if interval > 0 and epoch % interval == 0:
            save_epoch_artifacts(model, fixed, run_dir, f"epoch_{epoch:04d}", class_names, visualization_enabled)
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.5f} balance={row['train_router_balance_loss']:.5f} "
            f"importance={row['train_router_importance_loss']:.5f} entropy={row['train_router_normalized_entropy']:.4f} "
            f"train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f} "
            f"selection_rates={[round(value, 3) for value in train_metrics['expert_selection_rates']]}"
        )
    final = run_epoch(model, test_loader, loss_cfg, device)
    matrix = confusion_matrix(final["preds"], final["targets"], len(class_names))
    save_confusion(matrix, run_dir / "figures" / "confusion_matrix.png", class_names)
    save_training_curves(rows, run_dir / "figures" / "training_curves.png")
    save_epoch_artifacts(model, fixed, run_dir, "final_epoch", class_names, visualization_enabled)
    final_metrics = {
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "final_test_acc": final["acc"],
        "final_test_loss": final["loss"],
        "final_router_balance_loss": final["router_balance_loss"],
        "final_router_importance_loss": final["router_importance_loss"],
        "final_router_normalized_entropy": final["router_normalized_entropy"],
        "final_expert_selection_rates": final["expert_selection_rates"],
        "final_mean_routing_weights": final["mean_routing_weights"],
        "final_mean_expert_input_power": final["mean_expert_input_power"],
        "final_mean_expert_output_power": final["mean_expert_output_power"],
        "final_mean_metrics_by_type": final["mean_metrics_by_type"],
        "wall_time_sec": time.perf_counter() - start,
    }
    save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
    write_rows(
        run_dir / "metrics" / "test_predictions.csv",
        [
            {
                "sample_index": index,
                "true_label": int(target),
                "true_name": class_names[int(target)],
                "pred_label": int(prediction),
                "pred_name": class_names[int(prediction)],
                "correct": bool(prediction == target),
            }
            for index, (target, prediction) in enumerate(zip(final["targets"].tolist(), final["preds"].tolist()))
        ],
    )
    export_best_slm_package(model, test_loader, run_dir / "checkpoints" / "best.pt", run_dir / "slm_export_best", config, device, class_names)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
