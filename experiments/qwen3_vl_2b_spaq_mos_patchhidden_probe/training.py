from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.io_utils import write_csv, write_json
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.metrics import regression_metrics
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.modeling import NormalizedLinearRegressionHead
from experiments.qwen3_vl_2b_spaq_mos_vision_homogeneous_moe9.visualization import save_scatter

from .features import load_feature_cache


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return regression_metrics((targets.float() * 100).tolist(), (predictions.float() * 100).tolist())


def build_head(feature_dim: int, settings: Any) -> NormalizedLinearRegressionHead:
    return NormalizedLinearRegressionHead(feature_dim, settings.output_activation)


def train_probe(settings: Any, device: torch.device) -> nn.Module:
    train = load_feature_cache(settings.output_dir / "features" / "train_patch_hidden.pt")
    test = load_feature_cache(settings.output_dir / "features" / "test_patch_hidden.pt")
    feature_dim = int(train["metadata"]["feature_dim"])
    head = build_head(feature_dim, settings).to(device)
    loader = DataLoader(TensorDataset(train["features"].float(), train["targets"].float()),
                        batch_size=settings.head_batch_size, shuffle=True)
    test_features = test["features"].float().to(device)
    test_targets = test["targets"].float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
                 if settings.scheduler == "cosine" else None)
    criterion = nn.SmoothL1Loss(beta=settings.smooth_l1_beta)
    history: list[dict[str, Any]] = []
    best_srcc = float("-inf")
    checkpoint_dir = settings.output_dir / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); head.train(); loss_sum = 0.0; seen = 0
        train_predictions: list[torch.Tensor] = []; train_targets: list[torch.Tensor] = []
        for features, targets in loader:
            features = features.to(device); targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = head(features); loss = criterion(predictions, targets)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(targets); seen += len(targets)
            train_predictions.append(predictions.detach().cpu()); train_targets.append(targets.detach().cpu())
        head.eval()
        with torch.inference_mode():
            test_predictions = _batched_predict(head, test_features, settings.inference_batch_size).cpu()
        train_report = _metrics(torch.cat(train_predictions), torch.cat(train_targets))
        test_report = _metrics(test_predictions, test_targets)
        if scheduler is not None: scheduler.step()
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / seen,
               **{f"train_{key}": value for key, value in train_report.items()},
               **{f"test_{key}": value for key, value in test_report.items()},
               "epoch_time_sec": time.perf_counter() - started}
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        _save_head(head, checkpoint_dir / "head_last.pt", settings, feature_dim, epoch, row)
        if test_report["srcc"] > best_srcc:
            best_srcc = test_report["srcc"]
            _save_head(head, checkpoint_dir / "head_best_test_srcc.pt", settings, feature_dim, epoch, row)
            write_json(settings.output_dir / "metrics" / "best_test_srcc.json", row)
        if epoch % settings.log_interval_epochs == 0:
            print(f"epoch {epoch:03d}/{settings.epochs} loss={loss_sum/seen:.5f} "
                  f"train_MAE={train_report['mae']:.3f} train_SRCC={train_report['srcc']:.4f} "
                  f"test_MAE={test_report['mae']:.3f} test_SRCC={test_report['srcc']:.4f}", flush=True)
    return head


@torch.inference_mode()
def inference(settings: Any, device: torch.device, dataset: Any, checkpoint: str = "last") -> dict[str, Any]:
    cache = load_feature_cache(settings.output_dir / "features" / "test_patch_hidden.pt")
    feature_dim = int(cache["metadata"]["feature_dim"])
    filename = "head_last.pt" if checkpoint == "last" else "head_best_test_srcc.pt"
    payload = torch.load(settings.output_dir / "checkpoints" / filename, map_location="cpu", weights_only=True)
    head = build_head(feature_dim, settings).to(device); head.load_state_dict(payload["state_dict"]); head.eval()
    predictions = _batched_predict(head, cache["features"].float().to(device), settings.inference_batch_size).cpu()
    targets = cache["targets"].float(); report: dict[str, Any] = _metrics(predictions, targets)
    report.update({"dataset": "SPAQ", "task": "MOS", "model": "frozen_qwen_patch_hidden_direct_head",
                   "checkpoint": checkpoint, "feature_dim": feature_dim, "vision_transformer_used": False,
                   "optical_moe_used": False, "language_model_used": False,
                   "head_parameters": sum(parameter.numel() for parameter in head.parameters())})
    write_json(settings.output_dir / "metrics" / f"test_metrics_{checkpoint}.json", report)
    rows = []
    for index, target, prediction in zip(cache["sample_indices"].tolist(), targets.tolist(), predictions.tolist()):
        metadata = dataset.sample_metadata(index)
        rows.append({**metadata, "sample_index": index, "true_score": target * 100,
                     "predicted_score": prediction * 100, "absolute_error": abs(prediction-target) * 100})
    write_csv(settings.output_dir / "metrics" / f"test_predictions_{checkpoint}.csv", rows, list(rows[0]))
    save_scatter([row["true_score"] for row in rows], [row["predicted_score"] for row in rows],
                 settings.output_dir / "figures" / f"scatter_{checkpoint}.png",
                 f"SPAQ MOS patch-hidden probe ({checkpoint})")
    _write_comparison(settings, report)
    return report


def _batched_predict(head: nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.cat([head(features[start:start+batch_size]) for start in range(0, len(features), batch_size)])


def _save_head(head: nn.Module, path: Path, settings: Any, feature_dim: int,
               epoch: int, metrics: dict[str, Any]) -> None:
    torch.save({"state_dict": head.state_dict(), "feature_dim": feature_dim, "epoch": epoch,
                "metrics": metrics, "output_activation": settings.output_activation}, path)


def _write_comparison(settings: Any, probe: dict[str, Any]) -> None:
    result: dict[str, Any] = {"patch_hidden_probe": probe}
    for name, filename in (("electronic_teacher", "teacher_inference.json"),
                           ("optical_student", "student_inference.json")):
        path = settings.source_output_dir / "metrics" / filename
        if path.is_file():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    write_json(settings.output_dir / "metrics" / "comparison.json", result)

