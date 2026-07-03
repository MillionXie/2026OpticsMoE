from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .features import (
    move_inputs,
    multimodal_forward_features,
    pool_answer_hidden_state,
    preprocess_image_text,
)
from .io_utils import synchronize, write_csv, write_json
from .metrics import classification_metrics
from .optics import VisionBlockReplacement


def train_optical_student(
    model: nn.Module,
    processor: Any,
    replacement: VisionBlockReplacement,
    teacher_head: nn.Module,
    student_head: nn.Module,
    train_loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    validation_loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    class_names: Sequence[str],
    prompt: str,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    hidden_weight: float,
    kd_weight: float,
    ce_weight: float,
    progress: bool,
) -> dict[str, Any]:
    model.requires_grad_(False)
    model.eval()
    teacher_head.requires_grad_(False).eval()
    replacement.surrogate.requires_grad_(True).train()
    student_head.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(
        [*replacement.trainable_parameters(), *student_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_head_state: dict[str, torch.Tensor] | None = None
    best_optical_state: dict[str, torch.Tensor] | None = None
    first_shapes: dict[str, list[int]] | None = None

    for epoch in range(1, epochs + 1):
        replacement.surrogate.train()
        student_head.train()
        totals = {"total": 0.0, "hidden": 0.0, "kd": 0.0, "ce": 0.0}
        sample_count = 0
        synchronize(device)
        started = time.perf_counter()
        iterator: Any = train_loader
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(train_loader, desc=f"Optical student {epoch}/{epochs}", leave=False)
            except ImportError:
                pass
        for images, labels in iterator:
            labels = labels.to(device, non_blocking=True)
            inputs = move_inputs(preprocess_image_text(processor, images, prompt), device)

            replacement.use_teacher()
            replacement.capture.clear()
            with torch.no_grad():
                teacher_hidden = multimodal_forward_features(model, inputs)
                teacher_features, _ = pool_answer_hidden_state(
                    teacher_hidden, inputs["attention_mask"]
                )
                teacher_logits = teacher_head(teacher_features)
                teacher_block_output = replacement.capture.output_hidden
            if teacher_block_output is None:
                raise RuntimeError("Teacher hook did not capture the replaced block output")

            replacement.use_student()
            optimizer.zero_grad(set_to_none=True)
            student_hidden = multimodal_forward_features(model, inputs)
            student_features, _ = pool_answer_hidden_state(
                student_hidden, inputs["attention_mask"]
            )
            student_logits = student_head(student_features)
            student_block_output = replacement.surrogate.last_output
            if student_block_output is None:
                raise RuntimeError("Optical surrogate did not expose its block output")

            loss_hidden = normalized_hidden_mse(
                student_block_output, teacher_block_output
            )
            loss_kd = knowledge_distillation_loss(
                student_logits, teacher_logits, temperature
            )
            loss_ce = F.cross_entropy(student_logits, labels)
            loss = hidden_weight * loss_hidden + kd_weight * loss_kd + ce_weight * loss_ce
            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            sample_count += batch_size
            totals["total"] += float(loss.detach()) * batch_size
            totals["hidden"] += float(loss_hidden.detach()) * batch_size
            totals["kd"] += float(loss_kd.detach()) * batch_size
            totals["ce"] += float(loss_ce.detach()) * batch_size
            if first_shapes is None:
                first_shapes = {
                    "teacher_block_input": list(replacement.capture.input_hidden.shape)
                    if replacement.capture.input_hidden is not None
                    else [],
                    "teacher_block_output": list(teacher_block_output.shape),
                    "student_block_input": list(replacement.surrogate.last_input.shape)
                    if replacement.surrogate.last_input is not None
                    else [],
                    "student_block_output": list(student_block_output.shape),
                    "teacher_logits": list(teacher_logits.shape),
                    "student_logits": list(student_logits.shape),
                }
        synchronize(device)
        train_time = time.perf_counter() - started
        validation = evaluate_online(
            model,
            processor,
            replacement,
            student_head,
            validation_loader,
            class_names,
            prompt,
            device,
            student=True,
        )
        row = {
            "epoch": epoch,
            "loss_total": totals["total"] / max(sample_count, 1),
            "loss_hidden": totals["hidden"] / max(sample_count, 1),
            "loss_kd": totals["kd"] / max(sample_count, 1),
            "loss_ce": totals["ce"] / max(sample_count, 1),
            "validation_top1_accuracy": validation["top1_accuracy"],
            "validation_top5_accuracy": validation["top5_accuracy"],
            "train_time_sec": train_time,
        }
        history.append(row)
        if validation["top1_accuracy"] > best_accuracy:
            best_accuracy = validation["top1_accuracy"]
            best_epoch = epoch
            best_head_state = _cpu_state(student_head)
            best_optical_state = _cpu_state(replacement.surrogate)

    if best_head_state is None or best_optical_state is None:
        raise RuntimeError("Student training did not produce a checkpoint")
    student_head.load_state_dict(best_head_state)
    replacement.surrogate.load_state_dict(best_optical_state)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_head_state,
            "feature_dim": student_head.feature_dim,
            "hidden_dim": student_head.network[0].out_features,
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "best_epoch": best_epoch,
        },
        checkpoint_dir / "student_mlp.pt",
    )
    torch.save(
        {
            "state_dict": best_optical_state,
            "vision_block_index": replacement.block_index,
            "best_epoch": best_epoch,
        },
        checkpoint_dir / "optical_surrogate.pt",
    )
    write_csv(
        output_dir / "metrics" / "student_training_history.csv",
        history,
        list(history[0]),
    )
    report = {
        "best_epoch": best_epoch,
        "best_validation_top1_accuracy": best_accuracy,
        "loss_weights": {
            "hidden": hidden_weight,
            "kd": kd_weight,
            "ce": ce_weight,
            "temperature": temperature,
        },
        "captured_shapes": first_shapes or {},
        "history": history,
    }
    write_json(output_dir / "metrics" / "student_training.json", report)
    return report


def normalized_hidden_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            f"Teacher/student block output shapes differ: {teacher.shape} vs {student.shape}"
        )
    normalized_student = F.layer_norm(student.float(), (student.shape[-1],))
    normalized_teacher = F.layer_norm(teacher.float(), (teacher.shape[-1],))
    return F.mse_loss(normalized_student, normalized_teacher)


def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return (
        F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        * temperature**2
    )


def evaluate_online(
    model: nn.Module,
    processor: Any,
    replacement: VisionBlockReplacement,
    head: nn.Module,
    loader: Iterable[tuple[list[Image.Image], torch.Tensor]],
    class_names: Sequence[str],
    prompt: str,
    device: torch.device,
    student: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    replacement.use_student() if student else replacement.use_teacher()
    model.eval()
    replacement.surrogate.eval()
    head.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    top5_all: list[list[int]] = []
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = move_inputs(preprocess_image_text(processor, images, prompt), device)
            hidden = multimodal_forward_features(model, inputs)
            features, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
            logits = head(features).float().cpu()
            predictions = logits.argmax(dim=-1)
            top5 = logits.topk(min(5, len(class_names)), dim=-1).indices
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())
            top5_all.extend(top5.tolist())
    result = classification_metrics(labels_all, predictions_all, top5_all, class_names)
    return vars(result)


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy(
        {name: value.detach().cpu() for name, value in module.state_dict().items()}
    )

