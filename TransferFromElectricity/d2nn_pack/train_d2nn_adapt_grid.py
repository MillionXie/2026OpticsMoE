"""Inverse-design D2NN training (grid detector): CLIP features → Mask Generator → Optical propagation → grid energy → MSE loss.

Uses 10 hard rectangular detector regions (grid) — same as baseline, but phase
masks are generated from CLIP features via MaskGenerator (not direct parameters).

Supports multi-GPU training via DDP:
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_d2nn_adapt_grid.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # single GPU

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import create_mnist_clip_loaders
from model_adapt import D2NNOpticalBackbone
from mask_generator import MaskGenerator
from optics import DetectorArray
from utils import (
    BASE_DIR,
    choose_device,
    environment_info,
    git_info,
    load_yaml,
    make_run_dir,
    save_json,
    save_yaml,
    set_seed,
    write_rows,
)
from visualization import (
    save_confusion_matrix,
    save_confusion_csv,
    save_training_curves,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def ddp_setup():
    """Initialise DDP if running under torchrun, else return (False, device)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return False, None, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, torch.device(f"cuda:{local_rank}")


def is_main_process(ddp_enabled: bool) -> bool:
    if not ddp_enabled:
        return True
    return dist.get_rank() == 0


def ddp_barrier(ddp_enabled: bool):
    if ddp_enabled:
        dist.barrier()


# ---------------------------------------------------------------------------
# Arg parsing, optimiser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inverse-design D2NN training with CLIP features."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--disable_visualization", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--transformer_style", default="pytorch", choices=["pytorch", "tf"])
    return parser.parse_args()


def build_optimizer(mask_generator, config):
    cfg = config.get("optimizer", {})
    opt_type = cfg.get("type", "adamw").lower()
    lr = 0.001  # hard-coded: 1e-3
    weight_decay = 0.0005
    print(f"[DEBUG build_optimizer] final lr={lr}")
    params = mask_generator.parameters()
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer.type: {opt_type}")


def build_scheduler(optimizer, config, steps_per_epoch):
    """Linear decay: LR halves every 5 epochs."""
    decay_epochs = 5  # halve every 5 epochs
    decay_steps = decay_epochs * steps_per_epoch
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step):
        return 0.5 ** (step / max(1, decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    backbone, mask_gen, detector, loader, criterion, optimizer,
    scheduler, device, ddp_enabled, grad_clip=0.0, print_freq=50,
):
    """mask_gen can be a DDP wrapper or raw nn.Module."""
    backbone.train()
    mask_gen.train()
    if ddp_enabled and hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(0)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for step, (images, clip_feats, labels) in enumerate(loader, start=1):
        images = images.to(device)
        clip_feats = clip_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        phase_masks = mask_gen(clip_feats)
        sensor_field = backbone(images, phase_masks, return_complex=True)
        logits = detector(sensor_field)  # [B, 10] energy per class
        targets = F.one_hot(labels, num_classes=10).float()
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mask_gen.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
        batch = labels.numel()
        total_loss += float(loss.item()) * batch
        total_correct += int((preds == labels).sum().item())
        total_count += batch

        if is_main_process(ddp_enabled) and print_freq > 0 and step % int(print_freq) == 0:
            print(
                f"  step {step}/{len(loader)} "
                f"loss={total_loss / max(1, total_count):.4f} "
                f"acc={total_correct / max(1, total_count):.4f}"
            )

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


@torch.no_grad()
def evaluate_model(backbone, mask_gen, detector, loader, criterion, device):
    backbone.eval()
    mask_gen.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_targets = []
    for images, clip_feats, labels in loader:
        images = images.to(device)
        clip_feats = clip_feats.to(device)
        labels = labels.to(device)

        phase_masks = mask_gen(clip_feats)
        sensor_field = backbone(images, phase_masks, return_complex=True)
        logits = detector(sensor_field)
        targets = F.one_hot(labels, num_classes=10).float()
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        batch = labels.numel()
        total_loss += float(loss.item()) * batch
        total_correct += int((preds == labels).sum().item())
        total_count += batch
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "preds": torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long),
        "targets": torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Checkpoint & visualisation
# ---------------------------------------------------------------------------

def save_checkpoint(path, mask_gen, optimizer, epoch, metrics, config, ddp_enabled):
    if not is_main_process(ddp_enabled):
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP before saving
    state = mask_gen.module.state_dict() if ddp_enabled else mask_gen.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "mask_generator_state_dict": state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def fixed_clip_batch(loader, device, max_items):
    images, clip_feats, labels = next(iter(loader))
    return (
        images[:max_items].to(device),
        clip_feats[:max_items].to(device),
        labels[:max_items].to(device),
    )


def save_phase_visualization(phase_masks, run_dir, tag, dpi=150):
    """Save first sample's phase mask images."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # phase_masks: [B, num_layers, H, W] or [num_layers, H, W]
    if phase_masks.ndim == 4:
        phase_masks = phase_masks[0]  # show first sample
    for i, phase in enumerate(phase_masks):
        path = run_dir / "figures" / f"{tag}_phase_layer{i+1:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        phase_np = phase.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(phase_np, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     ticks=[0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


def save_sensor_visualization(sensor_out, labels, preds, run_dir, tag, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / "figures" / f"{tag}_sensor_sample.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(4, sensor_out.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for j in range(n):
        axes[j].imshow(sensor_out[j].detach().cpu().numpy(), cmap="inferno")
        axes[j].set_title(f"L={labels[j].item()} P={preds[j].item()}")
        axes[j].axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_target_patterns_fig(target_patterns, run_dir, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / "figures" / "target_patterns.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    C = target_patterns.shape[0]
    cols = 5
    rows = (C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for c in range(C):
        ax = axes[c // cols][c % cols] if rows > 1 else axes[c % cols]
        ax.imshow(target_patterns[c].cpu().numpy(), cmap="inferno")
        ax.set_title(str(c))
        ax.axis("off")
    for c in range(C, rows * cols):
        axes[c // cols][c % cols].axis("off") if rows > 1 else axes[c].axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def architecture_report(config):
    optics = config.get("optics", {})
    mg_cfg = config.get("mask_generator", {})
    det_cfg = config.get("detector", {})
    return {
        "model": "D2NNInverseDesign_CLIP_Grid",
        "dataset": "MNIST",
        "input_size": int(optics.get("input_size", 256)),
        "canvas_size": int(optics.get("canvas_size", 400)),
        "phase_mask_size": int(optics.get("phase_mask_size", 256)),
        "num_layers": int(optics.get("num_layers", 5)),
        "output_size": int(optics.get("output_size", 256)),
        "detector": {
            "type": "grid",
            "detector_size": int(det_cfg.get("detector_size", 32)),
            "layout": det_cfg.get("layout", "grid"),
            "normalize_detector_energy": bool(det_cfg.get("normalize_detector_energy", True)),
        },
        "mask_generator": {
            "clip_dim": int(mg_cfg.get("clip_dim", 1024)),
            "transformer_dim": int(mg_cfg.get("transformer_dim", 768)),
            "num_heads": int(mg_cfg.get("num_heads", 8)),
            "num_transformer_layers": int(mg_cfg.get("num_transformer_layers", 4)),
            "mlp_hidden_dim": int(mg_cfg.get("mlp_hidden_dim", 3072)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ddp_enabled, local_rank, rank_device = ddp_setup()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    config = load_yaml(config_path)
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("dataset", {})["batch_size"] = int(args.batch_size)
    if args.disable_visualization:
        config.setdefault("visualization", {})["enabled"] = False
    if args.smoke_test:
        config.setdefault("dataset", {})["smoke_train_size"] = 256
        config.setdefault("dataset", {})["smoke_test_size"] = 128
        config["dataset"]["batch_size"] = min(64, int(config["dataset"].get("batch_size", 128)))
        config.setdefault("training", {})["epochs"] = min(
            int(config.get("training", {}).get("epochs", 1)), 2
        )

    # Override device with DDP device
    if ddp_enabled:
        device = rank_device
    else:
        device = choose_device(args.device or config.get("device", "auto"))

    seed = int(config.get("seed", 7))
    set_seed(seed)

    run_name = config.get("experiment", {}).get(
        "run_name", f"d2nn_adapt_grid_{int(time.time())}"
    )
    run_dir = make_run_dir(run_name)

    # --- Only main process writes config ---
    if is_main_process(ddp_enabled):
        save_yaml(config, run_dir / "config.yaml")
        save_json(config, run_dir / "config_resolved.json")
        (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
        save_json(environment_info(), run_dir / "environment.json")
        save_json(git_info(), run_dir / "git_info.json")
        shutil.copy2(config_path, run_dir / "source_config.yaml")

    ddp_barrier(ddp_enabled)

    # Data loaders
    train_loader, test_loader, class_names = create_mnist_clip_loaders(
        config, seed=seed, smoke_test=args.smoke_test,
    )

    # Wrap with DistributedSampler if DDP
    if ddp_enabled:
        train_sampler = DistributedSampler(train_loader.dataset, shuffle=True, seed=seed)
        test_sampler = DistributedSampler(test_loader.dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_loader.dataset,
            batch_size=test_loader.batch_size,
            sampler=test_sampler,
            num_workers=test_loader.num_workers,
            pin_memory=True,
        )

    # Build models
    backbone = D2NNOpticalBackbone(config).to(device)

    mg_cfg = config.get("mask_generator", {})
    optics_cfg = config.get("optics", {})
    mask_gen = MaskGenerator(
        clip_dim=int(mg_cfg.get("clip_dim", 1024)),
        transformer_dim=int(mg_cfg.get("transformer_dim", 768)),
        num_heads=int(mg_cfg.get("num_heads", 8)),
        num_transformer_layers=int(mg_cfg.get("num_transformer_layers", 4)),
        mlp_hidden_dim=int(mg_cfg.get("mlp_hidden_dim", 3072)),
        dropout=float(mg_cfg.get("dropout", 0.1)),
        final_dropout=float(mg_cfg.get("final_dropout", 0.2)),
        num_layers=int(optics_cfg.get("num_layers", 5)),
        mask_size=int(optics_cfg.get("phase_mask_size", 256)),
        transformer_style=args.transformer_style,
    ).to(device)

    if ddp_enabled:
        mask_gen = DDP(mask_gen, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=False)

    optimizer = build_optimizer(mask_gen, config)

    # LR scheduler: linear decay, halve every 5 epochs
    scheduler = None
    grad_clip = 1.0
    if is_main_process(ddp_enabled):
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(optimizer, config, steps_per_epoch)
        print(f"Optimizer: lr={optimizer.param_groups[0]['lr']:.4f}, "
              f"schedule=halve_every_5ep, grad_clip={grad_clip}")

    # Grid detector (10 hard rectangular regions, same as baseline)
    det_cfg = config.get("detector", {})
    detector = DetectorArray(
        num_classes=10,
        grid_size=backbone.canvas_size,
        detector_size=int(det_cfg.get("detector_size", 32)),
        layout=det_cfg.get("layout", "grid"),
        normalize_total_energy=bool(det_cfg.get("normalize_detector_energy", True)),
    ).to(device)
    criterion = torch.nn.MSELoss()

    if is_main_process(ddp_enabled):
        save_json(architecture_report(config), run_dir / "architecture_report.json")

    # --- Logging ---
    if is_main_process(ddp_enabled):
        ngpu = 3 if ddp_enabled else 1
        print(f"device: {device}  (DDP={'on' if ddp_enabled else 'off'}, GPUs={ngpu})")
        print(
            f"MNIST train samples={len(train_loader.dataset)} "
            f"test samples={len(test_loader.dataset)} "
            f"batch_size={train_loader.batch_size}"
        )
        optics_cfg_ = config.get("optics", {})
        y0, y1, x0, x1 = backbone.injectors[0].phase_mask_region()
        print(
            "D2NN geometry: "
            f"input_size={backbone.input_size}, canvas_size={backbone.canvas_size}, "
            f"phase_mask_size={backbone.phase_mask_size}, "
            f"phase_mask_region=y[{y0}:{y1}], x[{x0}:{x1}]"
        )
        raw_mg = mask_gen.module if ddp_enabled else mask_gen
        print(f"Mask Generator params: {sum(p.numel() for p in raw_mg.parameters()):,}")
        print(
            "Distances: "
            f"input_to_layer={float(optics_cfg_.get('input_to_layer_distance_m', 0.05))} m, "
            f"inter_layer={float(optics_cfg_.get('inter_layer_distance_m', 0.05))} m, "
            f"detector={float(optics_cfg_.get('detector_distance_m', 0.05))} m"
        )

    # Visualisation
    viz_cfg = config.get("visualization", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))
    viz_interval = int(viz_cfg.get("save_interval_epochs",
                                   config.get("training", {}).get("save_interval_epochs", 50)))
    fixed_imgs, fixed_clip, fixed_labels = fixed_clip_batch(
        test_loader, device, int(viz_cfg.get("num_samples", 4)),
    )
    dpi = int(viz_cfg.get("dpi", 150))

    if viz_enabled and is_main_process(ddp_enabled):
        with torch.no_grad():
            init_phase = mask_gen(fixed_clip)
        save_phase_visualization(init_phase, run_dir, "epoch_0000", dpi=dpi)

    epochs = int(config.get("training", {}).get("epochs", 300))
    print_freq = int(config.get("training", {}).get(
        "print_freq", config.get("experiment", {}).get("print_freq", 50)))
    metrics_rows = []
    best = {"epoch": 0, "test_acc": -1.0, "test_loss": ""}
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            backbone, mask_gen, detector, train_loader, criterion,
            optimizer, scheduler, device, ddp_enabled,
            grad_clip=grad_clip, print_freq=print_freq,
        )
        train_time = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        test_metrics = evaluate_model(
            backbone, mask_gen, detector, test_loader, criterion, device,
        )
        eval_time = time.perf_counter() - eval_start

        artifact_time = 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": 0.0,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "artifact_time_sec": 0.0,
        }

        if test_metrics["acc"] > best["test_acc"]:
            best = {"epoch": epoch, "test_acc": test_metrics["acc"], "test_loss": test_metrics["loss"]}
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                mask_gen, optimizer, epoch, row, config, ddp_enabled,
            )
            ddp_barrier(ddp_enabled)
            if viz_enabled and is_main_process(ddp_enabled):
                art_start = time.perf_counter()
                with torch.no_grad():
                    best_phase = mask_gen(fixed_clip)
                save_phase_visualization(best_phase, run_dir, "best_epoch", dpi=dpi)
                artifact_time += time.perf_counter() - art_start

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            mask_gen, optimizer, epoch, row, config, ddp_enabled,
        )

        if viz_enabled and is_main_process(ddp_enabled) and viz_interval > 0 and epoch % viz_interval == 0:
            art_start = time.perf_counter()
            with torch.no_grad():
                epoch_phase = mask_gen(fixed_clip)
            save_phase_visualization(epoch_phase, run_dir, f"epoch_{epoch:04d}", dpi=dpi)
            artifact_time += time.perf_counter() - art_start

        row["artifact_time_sec"] = artifact_time
        row["epoch_time_sec"] = time.perf_counter() - epoch_start
        metrics_rows.append(row)
        if is_main_process(ddp_enabled):
            write_rows(run_dir / "metrics" / "epoch_metrics.csv", metrics_rows)
            print(
                f"epoch {epoch:03d} "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
                f"test_loss={row['test_loss']:.4f} test_acc={row['test_acc']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    # Final evaluation
    final_eval = evaluate_model(
        backbone, mask_gen, detector, test_loader, criterion, device,
    )

    if is_main_process(ddp_enabled):
        if "preds" in final_eval:
            from visualization import confusion_matrix as cm_fn
            matrix = cm_fn(final_eval["preds"], final_eval["targets"], num_classes=10)
            save_confusion_matrix(matrix, run_dir / "figures" / "confusion_matrix.png", class_names)
            save_confusion_csv(matrix, run_dir / "metrics" / "confusion_matrix.csv")

        save_training_curves(metrics_rows, run_dir / "figures" / "training_curves.png")

        if viz_enabled:
            with torch.no_grad():
                final_phase = mask_gen(fixed_clip)
            save_phase_visualization(final_phase, run_dir, "final_epoch", dpi=dpi)

    ddp_barrier(ddp_enabled)

    if is_main_process(ddp_enabled):
        total_wall = time.perf_counter() - run_start
        total_train = sum(float(r["train_time_sec"]) for r in metrics_rows)
        total_eval = sum(float(r["eval_time_sec"]) for r in metrics_rows)
        avg_epoch = sum(float(r["epoch_time_sec"]) for r in metrics_rows) / max(1, len(metrics_rows))
        raw_mg = mask_gen.module if ddp_enabled else mask_gen
        final_metrics = {
            "run_name": run_name,
            "best_epoch": best["epoch"],
            "best_test_acc": best["test_acc"],
            "best_test_loss": best["test_loss"],
            "final_test_acc": final_eval["acc"],
            "final_test_loss": final_eval["loss"],
            "total_wall_time_sec": total_wall,
            "total_train_time_sec": total_train,
            "total_eval_time_sec": total_eval,
            "avg_epoch_time_sec": avg_epoch,
            "mask_generator_param_count": sum(p.numel() for p in raw_mg.parameters()),
        }
        save_json(final_metrics, run_dir / "metrics" / "final_metrics.json")
        save_json(
            {**final_metrics, "architecture": architecture_report(config)},
            run_dir / "summary.json",
        )
        with torch.no_grad():
            final_phase_np = mask_gen(fixed_clip)
            if final_phase_np.ndim == 4:
                final_phase_np = final_phase_np[0]  # save first sample
            final_phase_np = final_phase_np.cpu().numpy()
        np.save(run_dir / "final_phase_masks.npy", final_phase_np)
        print(f"saved run outputs to: {run_dir}")

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
