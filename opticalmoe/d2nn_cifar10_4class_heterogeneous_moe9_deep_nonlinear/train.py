import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_loaders
from model import DeepHeterogeneousOpticalMoENonlinearClassifier
from slm_export import export_best_slm_package
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


def detector_region_cross_entropy(detector_energies, targets, eps):
    """Optional NLL over relative class-detector energies."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError("loss.detector_ce_eps must be positive")
    probabilities = (detector_energies + eps) / (
        detector_energies.sum(dim=1, keepdim=True) + detector_energies.shape[1] * eps
    )
    return F.nll_loss(torch.log(probabilities), targets)


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
    parser = argparse.ArgumentParser(description="Train CIFAR-10 4/10-class deep heterogeneous staged-OEO optical MoE")
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
        plane_mse = detector_plane_mse_loss(
            items["detector_intensity"],
            target_plane,
            scale,
            loss_cfg.get("normalize_detector_plane_mse", False),
            loss_cfg.get("detector_plane_mse_normalization_eps", 1.0e-8),
        )
        ce_weight = float(loss_cfg.get("detector_ce_weight", 0.0))
        detector_ce = logits.new_zeros(())
        if ce_weight != 0.0:
            detector_ce = detector_region_cross_entropy(logits, targets, loss_cfg.get("detector_ce_eps", 1.0e-8))
        classification = plane_mse + ce_weight * detector_ce
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
        "detector_plane_mse": plane_mse if loss_type == "detector_plane_mse" else logits.new_zeros(()),
        "detector_ce": detector_ce if loss_type == "detector_plane_mse" else logits.new_zeros(()),
        "router_balance": balance,
        "router_importance": importance,
        "router_entropy": items["router_normalized_entropy"],
        "routing_weights": items["routing_weights"],
        "selected_mask": items["routing_selected_mask"],
        "expert_input_power": items["expert_input_power"],
        "expert_output_power": items["expert_output_power"],
        "fiber_coupling_efficiency": items["fiber_coupling_efficiency"],
        "fiber_effective_mode_number": items["fiber_effective_mode_number"],
        "fiber_reconstruction_power": items["fiber_reconstruction_power"],
        "fiber_mode_power_distribution": items["fiber_mode_power_distribution"],
        "stage_details": items["expert_stage_details"],
    }


def run_epoch(model, loader, loss_cfg, device, optimizer=None, print_freq=0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "classification": 0.0, "plane_mse": 0.0, "detector_ce": 0.0, "balance": 0.0, "importance": 0.0, "entropy": 0.0}
    count = correct = 0
    predictions = []
    targets_all = []
    routing_weight_sum = torch.zeros(9)
    selection_count = torch.zeros(9)
    input_power_sum = torch.zeros(9)
    output_power_sum = torch.zeros(9)
    fiber_metric_count = torch.zeros(9)
    fiber_coupling_sum = torch.zeros(9)
    fiber_effective_modes_sum = torch.zeros(9)
    fiber_reconstruction_power_sum = torch.zeros(9)
    fiber_mode_distribution_sum = torch.zeros(9, model.expert_bank.max_fiber_modes)
    stage_sums = {
        key: torch.zeros(model.expert_bank.num_stages, 9)
        for key in ("pre_power", "normalized_power", "output_power", "active_ratio")
    }
    stage_counts = torch.zeros(model.expert_bank.num_stages, 9)
    stage_norm_mean_sum = torch.zeros(model.expert_bank.num_stages)
    stage_norm_std_sum = torch.zeros(model.expert_bank.num_stages)
    stage_norm_count = torch.zeros(model.expert_bank.num_stages)
    stage_linear_input_sum = torch.zeros(model.expert_bank.num_stages, 9)
    stage_linear_output_sum = torch.zeros(model.expert_bank.num_stages, 9)
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
            totals["plane_mse"] += float(parts["detector_plane_mse"].item()) * batch
            totals["detector_ce"] += float(parts["detector_ce"].item()) * batch
            predicted = logits.argmax(1)
            correct += int((predicted == targets).sum())
            count += batch
            predictions.append(predicted.detach().cpu())
            targets_all.append(targets.detach().cpu())
            routing_weight_sum += parts["routing_weights"].detach().cpu().sum(0)
            selection_count += parts["selected_mask"].detach().cpu().float().sum(0)
            input_power_sum += parts["expert_input_power"].detach().cpu().sum(0)
            output_power_sum += parts["expert_output_power"].detach().cpu().sum(0)
            coupling = parts["fiber_coupling_efficiency"].detach().cpu()
            effective = parts["fiber_effective_mode_number"].detach().cpu()
            valid = torch.isfinite(coupling)
            fiber_metric_count += valid.sum(0)
            fiber_coupling_sum += torch.where(valid, coupling, torch.zeros_like(coupling)).sum(0)
            fiber_effective_modes_sum += torch.where(valid, effective, torch.zeros_like(effective)).sum(0)
            reconstruction = parts["fiber_reconstruction_power"].detach().cpu()
            fiber_reconstruction_power_sum += torch.where(valid, reconstruction, torch.zeros_like(reconstruction)).sum(0)
            fiber_mode_distribution_sum += parts["fiber_mode_power_distribution"].detach().cpu().sum(0)
            for stage_index, stage in enumerate(parts["stage_details"]):
                stage_linear_input_sum[stage_index] += stage["linear_input_power"].detach().cpu().sum(0)
                stage_linear_output_sum[stage_index] += stage["linear_output_power"].detach().cpu().sum(0)
                oeo = stage["oeo"]
                if oeo["pre_power"] is None:
                    continue
                values = oeo["pre_power"].detach().cpu()
                valid = torch.isfinite(values)
                stage_counts[stage_index] += valid.sum(0)
                for key in stage_sums:
                    tensor = oeo[key].detach().cpu()
                    stage_sums[key][stage_index] += torch.where(valid, tensor, torch.zeros_like(tensor)).sum(0)
                norm_mean = oeo["normalization_mean"].detach().cpu()
                norm_std = oeo["normalization_std"].detach().cpu()
                stage_norm_mean_sum[stage_index] += norm_mean.sum()
                stage_norm_std_sum[stage_index] += norm_std.sum()
                stage_norm_count[stage_index] += norm_mean.numel()
            if training and print_freq > 0 and step % print_freq == 0:
                print(f"  batch {step}/{len(loader)} loss={totals['loss']/count:.5f} acc={correct/count:.4f}")
    mean_input = input_power_sum / max(1, count)
    mean_output = output_power_sum / max(1, count)
    mean_fiber_coupling = fiber_coupling_sum / fiber_metric_count.clamp_min(1)
    mean_fiber_effective_modes = fiber_effective_modes_sum / fiber_metric_count.clamp_min(1)
    mean_fiber_reconstruction_power = fiber_reconstruction_power_sum / fiber_metric_count.clamp_min(1)
    mean_fiber_mode_distribution = fiber_mode_distribution_sum / fiber_metric_count[:, None].clamp_min(1)
    mean_routing_weights = routing_weight_sum / max(1, count)
    expert_selection_rates = selection_count / max(1, count)
    stage_metrics = []
    for stage_index in range(model.expert_bank.num_stages):
        per_expert = []
        for expert_index, expert_type in enumerate(model.expert_bank.expert_types):
            divisor = stage_counts[stage_index, expert_index].clamp_min(1)
            enabled = model.expert_bank.experts[expert_index].nonlinear_enabled(stage_index)
            per_expert.append(
                {
                    "expert_index": expert_index,
                    "expert_type": expert_type,
                    "nonlinear_enabled": bool(model.expert_bank.nonlinearity_enabled and enabled),
                    "mean_linear_input_power": float(stage_linear_input_sum[stage_index, expert_index] / max(1, count)),
                    "mean_linear_output_power": float(stage_linear_output_sum[stage_index, expert_index] / max(1, count)),
                    **{
                        f"mean_{key}": float(stage_sums[key][stage_index, expert_index] / divisor)
                        if stage_counts[stage_index, expert_index] > 0 else None
                        for key in stage_sums
                    },
                }
            )
        by_type = {}
        for expert_type in sorted(set(model.expert_bank.expert_types)):
            all_type_values = [value for value in per_expert if value["expert_type"] == expert_type]
            enabled_values = [value for value in per_expert if value["expert_type"] == expert_type and value["nonlinear_enabled"]]
            by_type[expert_type] = {
                "enabled_expert_count": len(enabled_values),
                "mean_linear_input_power": sum(value["mean_linear_input_power"] for value in all_type_values) / len(all_type_values),
                "mean_linear_output_power": sum(value["mean_linear_output_power"] for value in all_type_values) / len(all_type_values),
                **{
                    f"mean_{key}": (
                        sum(value[f"mean_{key}"] for value in enabled_values if value[f"mean_{key}"] is not None)
                        / max(1, sum(value[f"mean_{key}"] is not None for value in enabled_values))
                    ) if enabled_values else None
                    for key in stage_sums
                },
            }
        stage_metrics.append(
            {
                "stage": stage_index + 1,
                "normalization_input_mean": float(stage_norm_mean_sum[stage_index] / stage_norm_count[stage_index].clamp_min(1)),
                "normalization_input_std": float(stage_norm_std_sum[stage_index] / stage_norm_count[stage_index].clamp_min(1)),
                "learned_gain": False,
                "learned_threshold": False,
                "per_expert": per_expert,
                "by_type": by_type,
            }
        )
    type_metrics = {}
    for expert_type in sorted(set(model.expert_bank.expert_types)):
        indices = [index for index, value in enumerate(model.expert_bank.expert_types) if value == expert_type]
        type_metrics[expert_type] = {
            "mean_routing_weight": float(mean_routing_weights[indices].mean()),
            "mean_selection_rate": float(expert_selection_rates[indices].mean()),
            "mean_input_power": float(mean_input[indices].mean()),
            "mean_output_power": float(mean_output[indices].mean()),
        }
        if expert_type == "fiber":
            type_metrics[expert_type]["mean_coupling_efficiency"] = float(mean_fiber_coupling[indices].mean())
            type_metrics[expert_type]["mean_effective_mode_number"] = float(mean_fiber_effective_modes[indices].mean())
            type_metrics[expert_type]["mean_reconstruction_power"] = float(mean_fiber_reconstruction_power[indices].mean())
    return {
        "loss": totals["loss"] / max(1, count),
        "classification_loss": totals["classification"] / max(1, count),
        "detector_plane_mse_loss": totals["plane_mse"] / max(1, count),
        "detector_ce_loss": totals["detector_ce"] / max(1, count),
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
        "mean_fiber_coupling_efficiency": mean_fiber_coupling.tolist(),
        "mean_fiber_effective_mode_number": mean_fiber_effective_modes.tolist(),
        "mean_fiber_reconstruction_power": mean_fiber_reconstruction_power.tolist(),
        "mean_fiber_mode_power_distribution": mean_fiber_mode_distribution.tolist(),
        "stage_nonlinearity_metrics": stage_metrics,
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
            "nonlinearity_parameter_report": model.nonlinearity_parameter_report(),
        },
        path,
    )


def architecture_report(model, config, class_names, train_loader, test_loader):
    parameters = model.expert_parameter_report()
    normalization_cfg = config.get("nonlinearity", {}).get("normalization", {})
    per_expert_normalization = bool(normalization_cfg.get("per_expert_enabled", True))
    affine_enabled = bool(normalization_cfg.get("elementwise_affine", True))
    normalization_scope = "per_expert" if per_expert_normalization else "stage_global"
    return {
        "model": "DeepHeterogeneousOpticalMoENonlinearClassifier",
        "task": f"CIFAR-10 {len(class_names)}-class deep heterogeneous staged-OEO optical MoE",
        "class_names": class_names,
        "layout": model.layout.to_dict(),
        "expert_type_map_row_major": model.expert_bank.expert_types,
        "expert_parameter_report": parameters,
        "stage_nonlinearity": {
            "complex_field_in_out": True,
            "intensity_detection_between_stages": True,
            "normalization": f"per_sample_{normalization_scope}_layernorm",
            "per_expert_instantaneous_normalization": per_expert_normalization,
            "elementwise_affine": affine_enabled,
            "affine_sharing": str(normalization_cfg.get("affine_sharing", "per_expert")),
            "activation": f"relu_after_{normalization_scope}_layernorm",
            "routing_amplitude_reapplied_after_normalization": False,
            "zero_phase_reencoding": True,
            "post_global_fc_oeo": False,
            "report": model.nonlinearity_parameter_report(),
        },
        "dataset": {
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "train_samples_per_epoch": len(train_loader.sampler),
            "batch_size": train_loader.batch_size,
        },
        "distances_m": config.get("optics", {}).get("distances_m", {}),
        "expert_structures": config.get("expert_bank", {}),
        "fourier_non_foldability": "Every Fourier mask is separated by finite spatial/frequency apertures, center crop, and padded free-space propagation; spatial truncation is non-diagonal in Fourier space.",
        "routing": {
            "type": "input_topk",
            "top_k": int(config.get("prompt", {}).get("top_k", 3)),
            "balance_weight": float(config.get("loss", {}).get("router_balance_weight", 0.0)),
            "importance_weight": float(config.get("loss", {}).get("router_importance_weight", 0.0)),
        },
        "parameters": {
            "heterogeneous_expert_bank": model.expert_parameter_count(),
            "stage_oeo_trainable_parameters": model.nonlinearity_parameter_report()["trainable_parameters"],
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
    if not config_path.is_absolute():
        config_path = config_path.resolve() if config_path.is_file() else BASE_DIR / config_path
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
    run_dir = BASE_DIR / "runs" / config.get("experiment", {}).get("run_name", "heterogeneous_moe9_deep_nonlinear")
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
    model = DeepHeterogeneousOpticalMoENonlinearClassifier(config, len(class_names)).to(device)
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
            "train_detector_plane_mse_loss": train_metrics["detector_plane_mse_loss"],
            "train_detector_ce_loss": train_metrics["detector_ce_loss"],
            "train_router_balance_loss": train_metrics["router_balance_loss"],
            "train_router_importance_loss": train_metrics["router_importance_loss"],
            "train_router_normalized_entropy": train_metrics["router_normalized_entropy"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_detector_plane_mse_loss": test_metrics["detector_plane_mse_loss"],
            "test_detector_ce_loss": test_metrics["detector_ce_loss"],
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
            if expert_type == "fiber":
                row["train_fiber_mean_coupling_efficiency"] = values["mean_coupling_efficiency"]
                row["train_fiber_mean_effective_mode_number"] = values["mean_effective_mode_number"]
                row["train_fiber_mean_reconstruction_power"] = values["mean_reconstruction_power"]
        for index, expert_type in enumerate(model.expert_bank.expert_types):
            if expert_type == "fiber":
                row[f"train_expert_{index}_fiber_coupling_efficiency"] = train_metrics["mean_fiber_coupling_efficiency"][index]
                row[f"train_expert_{index}_fiber_effective_mode_number"] = train_metrics["mean_fiber_effective_mode_number"][index]
                row[f"train_expert_{index}_fiber_reconstruction_power"] = train_metrics["mean_fiber_reconstruction_power"][index]
        for stage in train_metrics["stage_nonlinearity_metrics"]:
            stage_number = stage["stage"]
            row[f"stage_{stage_number}_normalization_input_mean"] = stage["normalization_input_mean"]
            row[f"stage_{stage_number}_normalization_input_std"] = stage["normalization_input_std"]
            for expert in stage["per_expert"]:
                prefix = f"stage_{stage_number}_expert_{expert['expert_index']}"
                row[f"{prefix}_nonlinear_enabled"] = expert["nonlinear_enabled"]
                row[f"{prefix}_linear_input_power"] = expert["mean_linear_input_power"]
                row[f"{prefix}_linear_output_power"] = expert["mean_linear_output_power"]
                row[f"{prefix}_pre_power"] = expert["mean_pre_power"]
                row[f"{prefix}_normalized_power"] = expert["mean_normalized_power"]
                row[f"{prefix}_output_power"] = expert["mean_output_power"]
                row[f"{prefix}_active_ratio"] = expert["mean_active_ratio"]
        rows.append(row)
        write_rows(run_dir / "metrics" / "epoch_metrics.csv", rows)
        save_json(
            {
                "epoch": epoch,
                "fiber_expert_indices": [index for index, value in enumerate(model.expert_bank.expert_types) if value == "fiber"],
                "mean_coupling_efficiency": train_metrics["mean_fiber_coupling_efficiency"],
                "mean_effective_mode_number": train_metrics["mean_fiber_effective_mode_number"],
                "mean_reconstruction_power": train_metrics["mean_fiber_reconstruction_power"],
                "mean_mode_power_distribution": train_metrics["mean_fiber_mode_power_distribution"],
            },
            run_dir / "metrics" / f"fiber_metrics_epoch_{epoch:04d}.json",
        )
        save_json(
            {
                "epoch": epoch,
                "train": train_metrics["stage_nonlinearity_metrics"],
                "test": test_metrics["stage_nonlinearity_metrics"],
                "fiber_stage2_bypass": True,
                "post_global_fc_oeo": False,
            },
            run_dir / "metrics" / f"stage_nonlinearity_epoch_{epoch:04d}.json",
        )
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
        "final_detector_plane_mse_loss": final["detector_plane_mse_loss"],
        "final_detector_ce_loss": final["detector_ce_loss"],
        "final_router_balance_loss": final["router_balance_loss"],
        "final_router_importance_loss": final["router_importance_loss"],
        "final_router_normalized_entropy": final["router_normalized_entropy"],
        "final_expert_selection_rates": final["expert_selection_rates"],
        "final_mean_routing_weights": final["mean_routing_weights"],
        "final_mean_expert_input_power": final["mean_expert_input_power"],
        "final_mean_expert_output_power": final["mean_expert_output_power"],
        "final_mean_metrics_by_type": final["mean_metrics_by_type"],
        "final_mean_fiber_coupling_efficiency": final["mean_fiber_coupling_efficiency"],
        "final_mean_fiber_effective_mode_number": final["mean_fiber_effective_mode_number"],
        "final_mean_fiber_reconstruction_power": final["mean_fiber_reconstruction_power"],
        "final_mean_fiber_mode_power_distribution": final["mean_fiber_mode_power_distribution"],
        "final_stage_nonlinearity_metrics": final["stage_nonlinearity_metrics"],
        "nonlinearity_parameter_report": model.nonlinearity_parameter_report(),
        "fiber_stage2_bypasses_nonlinearity": True,
        "post_global_fc_oeo": False,
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
