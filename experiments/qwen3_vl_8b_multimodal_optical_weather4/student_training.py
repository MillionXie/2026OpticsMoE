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
    replacement.set_surrogates_trainable(True)
    replacement.train()
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
    best_optical_states: list[dict[str, torch.Tensor]] | None = None
    first_shapes: dict[str, Any] | None = None
    group_metric_names = [
        f"hidden_group_{start}_{end}" for start, end in replacement.block_groups
    ]

    for epoch in range(1, epochs + 1):
        replacement.train()
        student_head.train()
        totals = {
            "total": 0.0,
            "hidden": 0.0,
            "kd": 0.0,
            "ce": 0.0,
            **{name: 0.0 for name in group_metric_names},
        }
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
            replacement.clear_captures()
            with torch.no_grad():
                teacher_hidden = multimodal_forward_features(model, inputs)
                teacher_features, _ = pool_answer_hidden_state(
                    teacher_hidden, inputs["attention_mask"]
                )
                teacher_logits = teacher_head(teacher_features)
                teacher_group_outputs = replacement.teacher_outputs()

            replacement.use_student()
            optimizer.zero_grad(set_to_none=True)
            student_hidden = multimodal_forward_features(model, inputs)
            student_features, _ = pool_answer_hidden_state(
                student_hidden, inputs["attention_mask"]
            )
            student_logits = student_head(student_features)
            student_group_outputs = replacement.student_outputs()

            group_hidden_losses = [
                normalized_hidden_mse(student_output, teacher_output)
                for student_output, teacher_output in zip(
                    student_group_outputs, teacher_group_outputs
                )
            ]
            loss_hidden = torch.stack(group_hidden_losses).mean()
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
            for name, group_loss in zip(group_metric_names, group_hidden_losses):
                totals[name] += float(group_loss.detach()) * batch_size
            if first_shapes is None:
                first_shapes = {
                    "distillation_groups": [
                        {
                            "teacher_blocks": [capture.block_start, capture.block_end],
                            "teacher_group_input": list(capture.input_hidden.shape)
                            if capture.input_hidden is not None
                            else [],
                            "teacher_group_output": list(teacher_output.shape),
                            "student_optical_input": list(surrogate.last_input.shape)
                            if surrogate.last_input is not None
                            else [],
                            "student_optical_output": list(student_output.shape),
                        }
                        for capture, surrogate, teacher_output, student_output in zip(
                            replacement.captures,
                            replacement.surrogates,
                            teacher_group_outputs,
                            student_group_outputs,
                        )
                    ],
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
            **{
                f"loss_{name}": totals[name] / max(sample_count, 1)
                for name in group_metric_names
            },
            "validation_top1_accuracy": validation["top1_accuracy"],
            "validation_top5_accuracy": validation["top5_accuracy"],
            "train_time_sec": train_time,
        }
        history.append(row)
        if validation["top1_accuracy"] > best_accuracy:
            best_accuracy = validation["top1_accuracy"]
            best_epoch = epoch
            best_head_state = _cpu_state(student_head)
            best_optical_states = replacement.cpu_state_dicts()

    if best_head_state is None or best_optical_states is None:
        raise RuntimeError("Student training did not produce a checkpoint")
    student_head.load_state_dict(best_head_state)
    replacement.load_state_dicts(best_optical_states)
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
            "state_dicts": best_optical_states,
            "block_groups": [list(group) for group in replacement.block_groups],
            "optical_conversions": len(replacement.surrogates),
            "phase_masks_per_conversion": 1,
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
        "distillation_block_groups": [list(group) for group in replacement.block_groups],
        "hidden_group_reduction": "mean",
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
    replacement.eval()
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
