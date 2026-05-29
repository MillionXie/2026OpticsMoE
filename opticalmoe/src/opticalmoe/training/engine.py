from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from opticalmoe.training.checkpoint import save_checkpoint
from opticalmoe.training.logging import append_metrics_csv, init_metrics_csv
from opticalmoe.training.metrics import accuracy
from opticalmoe.visualization import (
    save_confusion_matrix,
    save_detector_energy_bar,
    save_light_field_debug,
    save_phase_layers,
    save_sample_outputs,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        total_seen += batch_size

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_seen += batch_size
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

    return (
        total_loss / max(1, total_seen),
        total_correct / max(1, total_seen),
        torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long),
        torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long),
    )


def _maybe_visualize(
    model: nn.Module,
    fixed_batch,
    run_dir: Path,
    device: torch.device,
    epoch: int,
    vis_cfg: Dict,
    final: bool = False,
) -> None:
    if not vis_cfg.get("enabled", True) or fixed_batch is None:
        return

    suffix = "final" if final else f"epoch_{epoch:04d}"
    num_samples = int(vis_cfg.get("num_samples", 4))

    save_phase_layers(
        model,
        str(run_dir / "phases" / f"phase_layers_{suffix}.png"),
        title=f"Phase layers {suffix}",
    )
    intermediates = save_sample_outputs(
        model,
        fixed_batch,
        str(run_dir / "sample_outputs" / f"sample_outputs_{suffix}.png"),
        device=device,
        num_samples=num_samples,
    )
    save_detector_energy_bar(
        intermediates["detector_energies"],
        str(run_dir / "detector_energy_bar.png" if final else run_dir / "sample_outputs" / f"detector_energy_bar_{suffix}.png"),
    )

    if vis_cfg.get("save_train_intermediates", True) or vis_cfg.get("save_eval_intermediates", True):
        save_light_field_debug(
            intermediates,
            str(run_dir / "light_fields" / f"light_fields_{suffix}.png"),
            detector_masks=model.detector.get_masks(),
            num_samples=min(2, num_samples),
            save_npz=bool(vis_cfg.get("save_npz", False)),
        )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    run_dir: Path,
    num_epochs: int,
    num_classes: int,
    visualization_cfg: Dict,
    fixed_vis_batch=None,
    start_epoch: int = 1,
    best_val_acc: float = -1.0,
    best_epoch: int = 0,
) -> Dict:
    criterion = nn.CrossEntropyLoss()
    metrics_path = run_dir / "metrics.csv"
    init_metrics_csv(str(metrics_path), append=start_epoch > 1)

    last_test_acc = 0.0
    last_test_loss = 0.0
    last_targets = torch.empty(0, dtype=torch.long)
    last_preds = torch.empty(0, dtype=torch.long)

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        test_loss, test_acc, test_targets, test_preds = evaluate(model, test_loader, device, criterion)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
        }
        append_metrics_csv(str(metrics_path), row)

        print(
            f"epoch {epoch:04d} | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
            f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | lr {lr:.2e}"
        )

        checkpoint_metrics = dict(row)
        checkpoint_metrics["best_val_acc"] = best_val_acc
        checkpoint_metrics["best_epoch"] = best_epoch
        save_checkpoint(str(run_dir / "last.pt"), model, optimizer, epoch, checkpoint_metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_metrics["best_val_acc"] = best_val_acc
            checkpoint_metrics["best_epoch"] = best_epoch
            save_checkpoint(str(run_dir / "best.pt"), model, optimizer, epoch, checkpoint_metrics)

        interval = int(visualization_cfg.get("save_interval_epochs", 5))
        if interval > 0 and epoch % interval == 0:
            _maybe_visualize(model, fixed_vis_batch, run_dir, device, epoch, visualization_cfg, final=False)

        last_test_loss = test_loss
        last_test_acc = test_acc
        last_targets = test_targets
        last_preds = test_preds

    _maybe_visualize(model, fixed_vis_batch, run_dir, device, num_epochs, visualization_cfg, final=True)
    save_confusion_matrix(last_targets, last_preds, num_classes, str(run_dir / "confusion_matrix.png"))

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_loss": last_test_loss,
        "final_test_acc": last_test_acc,
    }
