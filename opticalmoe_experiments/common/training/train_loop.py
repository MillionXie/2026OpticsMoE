import time
from typing import Dict, Optional

import torch

from .eval_loop import evaluate
from .phase_dropout import phase_dropout_active_for_epoch


def train_one_epoch(model, loader, criterion, optimizer, device, print_freq: int = 50) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    start = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch = targets.numel()
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch
        if print_freq > 0 and batch_index % int(print_freq) == 0:
            print(f"  batch {batch_index}/{len(loader)} loss={total_loss/max(1, seen):.4f} acc={correct/max(1, seen):.4f}")
    return {
        "loss": total_loss / max(1, seen),
        "acc": correct / max(1, seen),
        "samples": seen,
        "time_sec": time.perf_counter() - start,
    }


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    phase_dropout: Dict,
    on_epoch_end=None,
    print_freq: int = 50,
    eval_max_batches: Optional[int] = None,
):
    rows = []
    best = {"epoch": 0, "val_acc": -1.0, "state_dict": None}
    for epoch in range(1, int(epochs) + 1):
        active = phase_dropout_active_for_epoch(phase_dropout, epoch)
        model.set_phase_dropout_active(active)
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, print_freq)
        val_metrics = evaluate(model, val_loader, criterion, device, max_batches=eval_max_batches)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "phase_dropout_active": active,
            "phase_dropout_mode": phase_dropout["mode"],
            "expert_phase_dropout_p": phase_dropout["expert_p"],
            "global_fc_phase_dropout_p": phase_dropout["global_fc_p"],
            "phase_dropout_block_size": phase_dropout["block_size"],
            "epoch_time_sec": train_metrics["time_sec"],
        }
        rows.append(row)
        if val_metrics["acc"] > best["val_acc"]:
            best = {
                "epoch": epoch,
                "val_acc": val_metrics["acc"],
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
        print(
            f"epoch {epoch:03d} train={row['train_acc']:.4f} val={row['val_acc']:.4f} "
            f"phase_dropout={'on' if active else 'off'} time={row['epoch_time_sec']/60.0:.1f} min"
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, row, best)
    return rows, best

