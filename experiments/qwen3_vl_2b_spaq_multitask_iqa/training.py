from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .io_utils import write_csv, write_json
from .metrics import multitask_metrics
from .modeling import MultitaskRegressionHead
from .settings import Settings


HISTORY_FIELDS = ["epoch", "learning_rate", "train_loss", "epoch_time_sec"]


def train_regression_head(
    train_cache: Mapping[str, Any],
    settings: Settings,
    device: torch.device,
) -> tuple[MultitaskRegressionHead, list[dict[str, Any]]]:
    features = train_cache["features"].float()
    labels = train_cache["normalized_scores"].float()
    if features.ndim != 2 or features.shape[1] != settings.expected_feature_dim:
        raise RuntimeError(f"Unexpected train feature shape: {tuple(features.shape)}")
    if labels.shape != (len(features),):
        raise RuntimeError(f"Unexpected normalized label shape: {tuple(labels.shape)}")
    if torch.any((labels < 0.0) | (labels > 1.0)):
        raise RuntimeError("Normalized SPAQ labels must remain in [0,1]")
    head = MultitaskRegressionHead(
        feature_dim=settings.expected_feature_dim,
        hidden_dim=settings.head_hidden_dim,
        dropout=settings.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=settings.smooth_l1_beta)
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.head_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    history_path = settings.output_dir / "training_history.csv"
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        head.train()
        loss_sum = 0.0
        samples = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = head(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            count = len(batch_labels)
            loss_sum += float(loss.detach()) * count
            samples += count
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(samples, 1),
            "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        write_csv(history_path, history, HISTORY_FIELDS)
        print(
            f"epoch {epoch}/{settings.epochs} train_loss={row['train_loss']:.6f} "
            f"lr={row['learning_rate']:.3e}"
        )
    checkpoint = {
        "state_dict": head.state_dict(),
        "metadata": {
            **head.specification(),
            "epochs_completed": settings.epochs,
            "selection_strategy": "fixed_final_epoch",
            "validation_set_used": False,
            "test_set_used_for_selection": False,
            "loss": "SmoothL1Loss",
            "smooth_l1_beta": settings.smooth_l1_beta,
            "label_scale_during_training": [0.0, 1.0],
        },
    }
    checkpoint_path = settings.output_dir / "checkpoints" / "final_regression_head.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    write_json(
        settings.output_dir / "metrics" / "training_summary.json",
        {
            "epochs_completed": settings.epochs,
            "final_train_loss": history[-1]["train_loss"],
            "checkpoint": str(checkpoint_path),
            "selection_strategy": "fixed_final_epoch",
            "validation_set_used": False,
            "test_set_used_during_training": False,
            "history": history,
        },
    )
    return head, history


def load_final_head(settings: Settings, device: torch.device) -> MultitaskRegressionHead:
    path = settings.output_dir / "checkpoints" / "final_regression_head.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Final regression checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "feature_dim": settings.expected_feature_dim,
        "hidden_dim": settings.head_hidden_dim,
        "dropout": settings.dropout,
    }
    mismatches = {
        key: {"saved": metadata.get(key), "current": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Regression checkpoint configuration mismatch: {mismatches}")
    head = MultitaskRegressionHead(
        settings.expected_feature_dim, settings.head_hidden_dim, settings.dropout
    )
    head.load_state_dict(payload["state_dict"])
    return head.to(device)


def evaluate_test(
    head: nn.Module,
    test_cache: Mapping[str, Any],
    settings: Settings,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = test_cache["features"].float()
    scores = test_cache["scores"].float()
    dataset = TensorDataset(features, torch.arange(len(features), dtype=torch.long))
    loader = DataLoader(dataset, batch_size=settings.head_batch_size, shuffle=False)
    predictions = torch.empty(len(features), dtype=torch.float32)
    head.eval()
    with torch.inference_mode():
        for batch_features, indices in loader:
            output = head(batch_features.to(device, non_blocking=True)).cpu()
            predictions[indices] = output
    predicted_scores = (predictions.clamp(0.0, 1.0) * 100.0).tolist()
    rows = [
        {
            "sample_index": int(test_cache["sample_indices"][index]),
            "image_name": test_cache["image_names"][index],
            "image_path": test_cache["image_paths"][index],
            "task": test_cache["tasks"][index],
            "true_score": float(scores[index]),
            "predicted_score": float(predicted_scores[index]),
            "absolute_error": abs(float(predicted_scores[index]) - float(scores[index])),
        }
        for index in range(len(features))
    ]
    fields = [
        "sample_index", "image_name", "image_path", "task", "true_score",
        "predicted_score", "absolute_error",
    ]
    write_csv(settings.output_dir / "test_predictions.csv", rows, fields)
    metrics = multitask_metrics(rows)
    metrics.update(
        {
            "checkpoint": str(settings.output_dir / "checkpoints" / "final_regression_head.pt"),
            "checkpoint_selection": "fixed_final_epoch",
            "test_used_for_epoch_selection": False,
        }
    )
    write_json(settings.output_dir / "test_metrics.json", metrics)
    return rows, metrics

