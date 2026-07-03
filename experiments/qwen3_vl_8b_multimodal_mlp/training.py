from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .io_utils import synchronize, write_csv, write_json
from .metrics import ClassificationResult, metrics_from_logits


def train_head(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    validation_fraction: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    progress: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    train_indices, validation_indices = _split_indices(len(train_features), validation_fraction, seed)
    train_dataset = TensorDataset(train_features[train_indices], train_labels[train_indices])
    validation_features = train_features[validation_indices]
    validation_labels = train_labels[validation_indices]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        samples = 0
        synchronize(device)
        started = time.perf_counter()
        iterator: Any = loader
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(loader, desc=f"MLP epoch {epoch}/{epochs}", leave=False)
            except ImportError:
                pass
        for features, labels in iterator:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            samples += len(labels)
        synchronize(device)
        train_time = time.perf_counter() - started
        validation_logits, validation_loss, validation_time = evaluate_features(
            head, validation_features, validation_labels, batch_size, device
        )
        validation = metrics_from_logits(validation_logits, validation_labels, class_names)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(samples, 1),
            "validation_loss": validation_loss,
            "validation_top1_accuracy": validation.top1_accuracy,
            "validation_top5_accuracy": validation.top5_accuracy,
            "validation_macro_f1": validation.macro_f1,
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
        }
        history.append(row)
        if validation.top1_accuracy > best_accuracy:
            best_accuracy = validation.top1_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})

    if best_state is None:
        raise RuntimeError("MLP training produced no checkpoint")
    head.load_state_dict(best_state)
    head.to(device).eval()
    test_logits, test_loss, test_time = evaluate_features(
        head, test_features, test_labels, batch_size, device
    )
    test_metrics = metrics_from_logits(test_logits, test_labels, class_names)
    checkpoint = {
        "state_dict": best_state,
        "feature_dim": int(train_features.shape[1]),
        "hidden_dim": int(head.network[0].out_features),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "best_epoch": best_epoch,
    }
    checkpoint_path = output_dir / "checkpoints" / "best_mlp.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    write_csv(output_dir / "metrics" / "training_history.csv", history, list(history[0]))
    report = {
        "best_epoch": best_epoch,
        "best_validation_top1_accuracy": best_accuracy,
        "total_train_time_sec": sum(row["train_time_sec"] for row in history),
        "total_validation_time_sec": sum(row["validation_time_sec"] for row in history),
        "test_loss": test_loss,
        "test_evaluation_time_sec": test_time,
        "test_metrics": vars(test_metrics),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    write_json(output_dir / "metrics" / "training.json", report)
    return head, report


def evaluate_features(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    outputs: list[torch.Tensor] = []
    loss_sum = 0.0
    head.eval()
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = head(batch_features)
            outputs.append(logits.cpu())
            loss_sum += float(criterion(logits, batch_labels))
    synchronize(device)
    elapsed = time.perf_counter() - started
    return torch.cat(outputs), loss_sum / max(len(labels), 1), elapsed


def load_head(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    from .modeling import MLPHead

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = MLPHead(
        checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_classes"], 0.0
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval(), checkpoint


def _split_indices(size: int, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator)
    validation_size = max(1, int(round(size * fraction)))
    return order[validation_size:], order[:validation_size]
