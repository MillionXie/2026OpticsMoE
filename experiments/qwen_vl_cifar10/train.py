from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import EvaluationResult, classification_metrics, evaluate_head
from .features import extract_feature_batch
from .utils import cuda_synchronize


@dataclass
class TrainingResult:
    train_loss: float
    evaluation: EvaluationResult
    elapsed_sec: float
    images_per_second: float
    history: list[dict[str, float | int]]
    trainable_model_state: dict[str, torch.Tensor] | None = None


def train_mlp_head(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> TrainingResult:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_evaluation: EvaluationResult | None = None
    best_train_loss = 0.0
    history: list[dict[str, float | int]] = []
    training_elapsed = 0.0

    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        example_count = 0
        cuda_synchronize(device)
        epoch_start = time.perf_counter()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            example_count += len(labels)
        cuda_synchronize(device)
        training_elapsed += time.perf_counter() - epoch_start
        train_loss = loss_sum / max(example_count, 1)
        evaluation = evaluate_head(
            head, test_features, test_labels, batch_size, device, class_names
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": float(evaluation.loss or 0.0),
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
            }
        )
        if evaluation.accuracy > best_accuracy:
            best_accuracy = evaluation.accuracy
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in head.state_dict().items()}
            )
            best_evaluation = evaluation
            best_train_loss = train_loss

    if best_state is None or best_evaluation is None:
        raise RuntimeError("MLP training completed without producing a checkpoint.")
    head.load_state_dict(best_state)
    processed = len(train_features) * epochs
    return TrainingResult(
        train_loss=best_train_loss,
        evaluation=best_evaluation,
        elapsed_sec=training_elapsed,
        images_per_second=processed / training_elapsed if training_elapsed > 0 else 0.0,
        history=history,
    )


def train_lora_classifier(
    model: nn.Module,
    processor: Any,
    head: nn.Module,
    train_loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    test_loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    feature_source: str,
    class_names: Sequence[str],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    train_sample_count: int,
) -> TrainingResult:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA mode found no trainable adapter parameters.")
    head.to(device)
    optimizer = torch.optim.AdamW(
        [*trainable, *head.parameters()], lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_head_state: dict[str, torch.Tensor] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_evaluation: EvaluationResult | None = None
    best_train_loss = 0.0
    history: list[dict[str, float | int]] = []
    training_elapsed = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        loss_sum = 0.0
        example_count = 0
        cuda_synchronize(device)
        epoch_start = time.perf_counter()
        for images, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = extract_feature_batch(
                model, processor, images, feature_source, device, requires_grad=True
            )
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            example_count += len(labels)
        cuda_synchronize(device)
        training_elapsed += time.perf_counter() - epoch_start
        train_loss = loss_sum / max(example_count, 1)
        evaluation = evaluate_image_classifier(
            model, processor, head, test_loader, feature_source, class_names, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": float(evaluation.loss or 0.0),
                "accuracy": evaluation.accuracy,
                "macro_f1": evaluation.macro_f1,
            }
        )
        if evaluation.accuracy > best_accuracy:
            best_accuracy = evaluation.accuracy
            best_head_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in head.state_dict().items()}
            )
            best_model_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            best_evaluation = evaluation
            best_train_loss = train_loss

    if best_head_state is None or best_model_state is None or best_evaluation is None:
        raise RuntimeError("LoRA training completed without producing a checkpoint.")
    head.load_state_dict(best_head_state)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in best_model_state.items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    processed = train_sample_count * epochs
    return TrainingResult(
        train_loss=best_train_loss,
        evaluation=best_evaluation,
        elapsed_sec=training_elapsed,
        images_per_second=processed / training_elapsed if training_elapsed > 0 else 0.0,
        history=history,
        trainable_model_state=best_model_state,
    )


def evaluate_image_classifier(
    model: nn.Module,
    processor: Any,
    head: nn.Module,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    feature_source: str,
    class_names: Sequence[str],
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    head.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels_all: list[int] = []
    predictions: list[int] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(device)
            features = extract_feature_batch(model, processor, images, feature_source, device)
            logits = head(features)
            loss_sum += float(criterion(logits, labels).item())
            labels_all.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    result = classification_metrics(labels_all, predictions, class_names)
    return EvaluationResult(
        loss=loss_sum / max(len(labels_all), 1),
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        per_class_accuracy=result.per_class_accuracy,
        labels=result.labels,
        predictions=result.predictions,
        confusion_matrix=result.confusion_matrix,
    )
