from __future__ import annotations

import contextlib
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import DatasetBundle, FSSSaliencyDataset, collate_saliency
from .io_utils import write_csv, write_json
from .modeling import (
    FrozenQwenVisionTeacher,
    LoadedVisionBackbone,
    VisionOpticalSaliencyStudent,
    build_teacher,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import SegmentationAccumulator, segmentation_loss
from .teacher_cache import TeacherMaskCache, expected_cache_identity
from .visualization import (
    save_optical_phase_figures,
    save_prediction_panel,
    save_training_curves,
)


HISTORY_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_bce", "train_dice_loss",
    "train_mask_kd", "train_router_balance", "train_router_importance",
    "train_mean_iou", "train_mean_dice", "train_mae", "train_pixel_accuracy",
    "test_loss", "test_mean_iou", "test_mean_dice", "test_mae",
    "test_pixel_accuracy", "epoch_time_sec",
]


def build_loaders(
    bundle: DatasetBundle,
    settings: Any,
    *,
    train_batch_size: int,
    train_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train = FSSSaliencyDataset(
        bundle.train_records, settings, training=train_augmentation
    )
    test = FSSSaliencyDataset(bundle.test_records, settings, training=False)
    common = {
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_saliency,
        "persistent_workers": settings.num_workers > 0,
    }
    train_loader = DataLoader(
        train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader


def train_teacher(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
) -> dict[str, Any]:
    teacher = build_teacher(loaded, settings)
    report = trainable_parameter_report(teacher.head, prefix="teacher_segmentation_head")
    report["segmentation_head"] = teacher.head.specification()
    write_json(settings.output_dir / "model_teacher.json", report)
    _print_parameter_report(report)
    train_loader, test_loader = build_loaders(
        bundle, settings, train_batch_size=settings.teacher_batch_size
    )
    optimizer = torch.optim.AdamW(
        teacher.head.parameters(),
        lr=settings.teacher_learning_rate,
        weight_decay=settings.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_path = settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt"
    last_path = settings.output_dir / "checkpoints" / "teacher_last.pt"
    for epoch in range(1, settings.teacher_epochs + 1):
        started = time.perf_counter()
        train_metrics, train_parts = _train_epoch(
            teacher, loaded.processor, train_loader, optimizer, settings,
            epoch=epoch, model_kind="teacher",
        )
        improved = train_metrics["loss"] < best_train_loss
        if improved:
            best_train_loss = float(train_metrics["loss"])
        payload = {
            "epoch": epoch,
            "head_state_dict": teacher.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics_observation_only": None,
            "evaluation_status": "train_complete_test_pending",
            "selection_criterion": "minimum_train_loss",
            "head_specification": teacher.head.specification(),
        }
        # Selection uses training loss, so save the completed epoch before test
        # evaluation. A bad evaluation record must not discard a full epoch.
        _save_checkpoint(last_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        test_metrics = (
            evaluate_model(
                teacher, loaded.processor, test_loader, settings,
                model_kind="teacher", save_predictions=False,
            )[0]
            if settings.evaluate_test_each_epoch else _empty_test_metrics()
        )
        row = _history_row(
            epoch, optimizer.param_groups[0]["lr"], train_metrics, train_parts,
            test_metrics, time.perf_counter() - started,
        )
        history.append(row)
        _save_history(settings.output_dir / "metrics" / "teacher_training_history.csv", history)
        write_json(settings.output_dir / "metrics" / "teacher_training_latest.json", row)
        payload["test_metrics_observation_only"] = test_metrics
        payload["evaluation_status"] = "train_and_test_complete"
        _save_checkpoint(last_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        print(
            f"teacher epoch {epoch:03d}/{settings.teacher_epochs} "
            f"train_loss={train_metrics['loss']:.5f} train_mIoU={train_metrics['mean_iou']:.4f} "
            f"test_mIoU={test_metrics['mean_iou']:.4f} test_Dice={test_metrics['mean_dice']:.4f} "
            f"best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    save_training_curves(
        settings.output_dir / "figures" / "teacher_training_curves.png", history
    )
    teacher.close()
    return {"best_checkpoint": str(best_path), "best_train_loss": best_train_loss}


def train_student(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
) -> dict[str, Any]:
    student = VisionOpticalSaliencyStudent(
        loaded,
        settings,
        head=_student_head(loaded, settings),
    )
    student.activate()
    initialization = _initialize_student(
        student, settings.student_initial_checkpoint
    )
    report = trainable_parameter_report(student, prefix="optical_student")
    report["segmentation_head"] = student.head.specification()
    report["optical_core"] = student.core.parameter_breakdown()
    report["output_adapter_used"] = False
    report["output_adapter_trainable"] = any(
        parameter.requires_grad for parameter in student.core.output_adapter.parameters()
    )
    report["initialization"] = initialization
    write_json(settings.output_dir / "model_student.json", report)
    _print_parameter_report(report)
    train_loader, test_loader = build_loaders(
        bundle,
        settings,
        train_batch_size=settings.student_batch_size,
        train_augmentation=settings.augmentation_enabled,
    )
    optimizer = _student_optimizer(student, settings)
    mask_cache = _load_mask_cache(settings, "train") if settings.mask_kd_weight > 0 else None
    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_path = settings.output_dir / "checkpoints" / "student_best_train_loss.pt"
    last_path = settings.output_dir / "checkpoints" / "student_last.pt"
    for epoch in range(1, settings.student_epochs + 1):
        started = time.perf_counter()
        train_metrics, train_parts = _train_epoch(
            student, loaded.processor, train_loader, optimizer, settings,
            epoch=epoch, model_kind="student", mask_cache=mask_cache,
        )
        improved = train_metrics["loss"] < best_train_loss
        if improved:
            best_train_loss = float(train_metrics["loss"])
        payload = {
            "epoch": epoch,
            "core_state_dict": student.core.state_dict(),
            "head_state_dict": student.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics_observation_only": None,
            "evaluation_status": "train_complete_test_pending",
            "selection_criterion": "minimum_train_loss",
            "mask_kd_weight": settings.mask_kd_weight,
            "initialization": initialization,
            "head_specification": student.head.specification(),
        }
        _save_checkpoint(last_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        test_metrics = (
            evaluate_model(
                student, loaded.processor, test_loader, settings,
                model_kind="student", save_predictions=False,
            )[0]
            if settings.evaluate_test_each_epoch else _empty_test_metrics()
        )
        row = _history_row(
            epoch, optimizer.param_groups[0]["lr"], train_metrics, train_parts,
            test_metrics, time.perf_counter() - started,
        )
        history.append(row)
        _save_history(settings.output_dir / "metrics" / "student_training_history.csv", history)
        write_json(settings.output_dir / "metrics" / "student_training_latest.json", row)
        payload["test_metrics_observation_only"] = test_metrics
        payload["evaluation_status"] = "train_and_test_complete"
        _save_checkpoint(last_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        print(
            f"student epoch {epoch:03d}/{settings.student_epochs} "
            f"train_loss={train_metrics['loss']:.5f} train_mIoU={train_metrics['mean_iou']:.4f} "
            f"test_mIoU={test_metrics['mean_iou']:.4f} test_Dice={test_metrics['mean_dice']:.4f} "
            f"balance={train_parts['router_balance']:.5f} "
            f"best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    save_training_curves(
        settings.output_dir / "figures" / "student_training_curves.png", history
    )
    student.restore_native()
    return {"best_checkpoint": str(best_path), "best_train_loss": best_train_loss}


def test_teacher(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or (
        settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt"
    )
    teacher = build_teacher(loaded, settings)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    _, test_loader = build_loaders(
        bundle, settings, train_batch_size=settings.teacher_batch_size
    )
    metrics, _ = evaluate_model(
        teacher, loaded.processor, test_loader, settings,
        model_kind="teacher", save_predictions=True,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_selection"] = "minimum_train_loss; test was not used for selection"
    write_json(settings.output_dir / "metrics" / "teacher_test_metrics.json", metrics)
    teacher.close()
    return metrics


def test_student(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or (
        settings.output_dir / "checkpoints" / "student_best_train_loss.pt"
    )
    student = VisionOpticalSaliencyStudent(
        loaded, settings, head=_student_head(loaded, settings)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student.core.load_state_dict(checkpoint["core_state_dict"], strict=True)
    student.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    student.activate()
    _, test_loader = build_loaders(
        bundle, settings, train_batch_size=settings.student_batch_size
    )
    metrics, _ = evaluate_model(
        student, loaded.processor, test_loader, settings,
        model_kind="student", save_predictions=True,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_selection"] = "minimum_train_loss; test was not used for selection"
    metrics["mask_kd_enabled"] = settings.mask_kd_weight > 0
    write_json(settings.output_dir / "metrics" / "student_test_metrics.json", metrics)
    save_optical_phase_figures(
        student.core, settings.output_dir / "figures" / "optical_parameters"
    )
    student.restore_native()
    return metrics


@torch.no_grad()
def save_teacher_student_comparison_examples(
    loaded: LoadedVisionBackbone,
    bundle: DatasetBundle,
    settings: Any,
    *,
    teacher_checkpoint: Path,
    student_checkpoint: Path,
) -> None:
    """Render matched examples without holding two Qwen backbones in GPU memory."""
    _, test_loader = build_loaders(
        bundle, settings, train_batch_size=settings.student_batch_size
    )
    teacher = build_teacher(loaded, settings)
    teacher_payload = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher.head.load_state_dict(teacher_payload["head_state_dict"], strict=True)
    teacher.eval()
    selected: list[dict[str, Any]] = []
    for batch in test_loader:
        inputs = preprocess_vision(loaded.processor, batch["images"], loaded.device)
        logits, _ = teacher(inputs["pixel_values"], inputs["image_grid_thw"])
        for index in range(len(batch["sample_ids"])):
            selected.append(
                {
                    "image": batch["images"][index],
                    "mask": batch["masks"][index],
                    "sample_id": batch["sample_ids"][index],
                    "teacher_probability": logits[index].float().sigmoid().cpu(),
                }
            )
            if len(selected) >= settings.visualization_sample_count:
                break
        if len(selected) >= settings.visualization_sample_count:
            break
    teacher.close()

    student = VisionOpticalSaliencyStudent(
        loaded, settings, head=_student_head(loaded, settings)
    )
    student_payload = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    student.core.load_state_dict(student_payload["core_state_dict"], strict=True)
    student.head.load_state_dict(student_payload["head_state_dict"], strict=True)
    student.activate()
    student.eval()
    for index, item in enumerate(selected):
        inputs = preprocess_vision(loaded.processor, [item["image"]], loaded.device)
        logits, _, _ = student(inputs["pixel_values"], inputs["image_grid_thw"])
        probability = logits[0].float().sigmoid().cpu()
        truth = item["mask"].float()
        prediction = probability.ge(0.5)
        binary_truth = truth.ge(0.5)
        intersection = float((prediction & binary_truth).sum())
        union = float((prediction | binary_truth).sum())
        iou = intersection / union if union else 1.0
        filename = f"{index:03d}_{item['sample_id'].replace('/', '_')}.png"
        save_prediction_panel(
            settings.output_dir / "figures" / "comparison_examples" / filename,
            image=item["image"],
            ground_truth=truth,
            teacher_probability=item["teacher_probability"],
            student_probability=probability,
            title=f"{item['sample_id']} | optical IoU={iou:.3f}",
        )
        if iou < 0.25:
            save_prediction_panel(
                settings.output_dir / "figures" / "failure_cases" / filename,
                image=item["image"],
                ground_truth=truth,
                teacher_probability=item["teacher_probability"],
                student_probability=probability,
                title=f"Failure: {item['sample_id']} | optical IoU={iou:.3f}",
            )
    student.restore_native()


def _student_head(loaded: LoadedVisionBackbone, settings: Any) -> nn.Module:
    from .modeling import LightweightSegmentationHead

    return LightweightSegmentationHead(
        settings.detector_output_size,
        settings.segmentation_projection_dim,
        settings.segmentation_channels,
        settings.segmentation_groupnorm_groups,
        settings.image_size,
    ).to(loaded.device)


def _student_optimizer(
    student: VisionOpticalSaliencyStudent,
    settings: Any,
) -> torch.optim.Optimizer:
    phase_ids = {
        id(parameter)
        for module in (student.core.expert_layers, student.core.global_phase)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    router_ids = {
        id(parameter) for parameter in student.core.router.parameters()
        if parameter.requires_grad
    }
    groups: list[dict[str, Any]] = []
    used: set[int] = set()
    for ids, lr, name in (
        (phase_ids, settings.phase_learning_rate, "phase"),
        (router_ids, settings.router_learning_rate, "router"),
    ):
        parameters = [
            parameter for parameter in student.parameters()
            if parameter.requires_grad and id(parameter) in ids and id(parameter) not in used
        ]
        if parameters:
            groups.append(
                {"params": parameters, "lr": lr or settings.student_learning_rate, "name": name}
            )
            used.update(id(parameter) for parameter in parameters)
    remaining = [
        parameter for parameter in student.parameters()
        if parameter.requires_grad and id(parameter) not in used
    ]
    groups.append(
        {"params": remaining, "lr": settings.student_learning_rate, "name": "adapter_and_head"}
    )
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _initialize_student(
    student: VisionOpticalSaliencyStudent,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "mode": "random_initialization",
            "checkpoint": None,
            "optimizer_state_restored": False,
        }
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Student initialization checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = [
        name for name in ("core_state_dict", "head_state_dict")
        if name not in payload
    ]
    if missing:
        raise RuntimeError(
            f"Student initialization checkpoint is missing {missing}: {checkpoint_path}"
        )
    student.core.load_state_dict(payload["core_state_dict"], strict=True)
    student.head.load_state_dict(payload["head_state_dict"], strict=True)
    report = {
        "mode": "checkpoint_weights_fresh_optimizer",
        "checkpoint": str(checkpoint_path),
        "source_epoch": int(payload.get("epoch", -1)),
        "source_train_loss": payload.get("train_metrics", {}).get("loss"),
        "optimizer_state_restored": False,
    }
    print(
        "student initialized from "
        f"{checkpoint_path} (source_epoch={report['source_epoch']}); "
        "optimizer is freshly initialized",
        flush=True,
    )
    return report


def _train_epoch(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    settings: Any,
    *,
    epoch: int,
    model_kind: str,
    mask_cache: TeacherMaskCache | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.train()
    accumulator = SegmentationAccumulator()
    parts = {
        "bce": 0.0, "dice_loss": 0.0, "mask_kd": 0.0,
        "router_balance": 0.0, "router_importance": 0.0,
    }
    sample_count = 0
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=settings.amp_enabled and settings.dtype.lower() == "float16",
    )
    for batch_index, batch in enumerate(loader, start=1):
        inputs = preprocess_vision(processor, batch["images"], next(model.parameters()).device)
        masks = batch["masks"].to(next(model.parameters()).device, non_blocking=True)
        teacher_logits = (
            mask_cache.fetch(batch["sample_ids"], masks.device)
            if mask_cache is not None else None
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(settings):
            logits, _ = _model_forward(model, inputs)
            loss, loss_parts = segmentation_loss(
                logits, masks,
                bce_weight=settings.bce_weight,
                dice_weight=settings.dice_weight,
                teacher_logits=teacher_logits,
                mask_kd_weight=settings.mask_kd_weight,
                mask_kd_temperature=settings.mask_kd_temperature,
            )
            balance = logits.new_zeros(())
            importance = logits.new_zeros(())
            if model_kind == "student":
                balance, importance = model.router_losses()
                loss = (
                    loss
                    + settings.router_balance_weight * balance
                    + settings.router_importance_weight * importance
                )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {model_kind} loss at epoch={epoch}, batch={batch_index}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(masks.shape[0])
        sample_count += batch_size
        accumulator.update(logits.detach(), masks, loss=float(loss.detach()))
        for name, value in loss_parts.items():
            parts[name] += float(value.detach()) * batch_size
        parts["router_balance"] += float(balance.detach()) * batch_size
        parts["router_importance"] += float(importance.detach()) * batch_size
        if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
            print(
                f"[{model_kind}_train] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={float(loss.detach()):.5f} bce={float(loss_parts['bce']):.5f} "
                f"dice={float(loss_parts['dice_loss']):.5f} "
                f"balance={float(balance):.5f}",
                flush=True,
            )
    return accumulator.compute(), {name: value / sample_count for name, value in parts.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    *,
    model_kind: str,
    save_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = SegmentationAccumulator()
    rows: list[dict[str, Any]] = []
    saved = 0
    worst_examples: list[dict[str, Any]] = []
    routing_samples = 0
    routing_selected = torch.zeros(settings.num_experts, dtype=torch.float64)
    routing_weight = torch.zeros(settings.num_experts, dtype=torch.float64)
    routing_probability = torch.zeros(settings.num_experts, dtype=torch.float64)
    routing_entropy_sum = 0.0
    routing_balance_sum = 0.0
    routing_importance_sum = 0.0
    for batch in loader:
        inputs = preprocess_vision(processor, batch["images"], next(model.parameters()).device)
        masks = batch["masks"].to(next(model.parameters()).device, non_blocking=True)
        with _autocast(settings):
            logits, _ = _model_forward(model, inputs)
            loss, _ = segmentation_loss(
                logits, masks,
                bce_weight=settings.bce_weight,
                dice_weight=settings.dice_weight,
            )
        accumulator.update(logits, masks, loss=loss)
        if model_kind == "student":
            routing = model.core.last_routing
            required = {
                "selected_mask", "weights", "probabilities",
                "normalized_entropy", "balance_loss", "importance_loss",
            }
            missing = sorted(required - set(routing))
            if missing:
                raise RuntimeError(
                    f"Student router diagnostics are missing fields: {missing}"
                )
            selected = routing["selected_mask"].detach().double().cpu()
            weights = routing["weights"].detach().double().cpu()
            probabilities_cpu = routing["probabilities"].detach().double().cpu()
            batch_size = int(selected.shape[0])
            routing_samples += batch_size
            routing_selected += selected.sum(0)
            routing_weight += weights.sum(0)
            routing_probability += probabilities_cpu.sum(0)
            routing_entropy_sum += (
                float(routing["normalized_entropy"].detach()) * batch_size
            )
            routing_balance_sum += (
                float(routing["balance_loss"].detach()) * batch_size
            )
            routing_importance_sum += (
                float(routing["importance_loss"].detach()) * batch_size
            )
        probabilities = logits.float().sigmoid()
        binary = probabilities.ge(0.5)
        truth = masks.ge(0.5)
        intersection = (binary & truth).flatten(1).sum(1).float()
        union = (binary | truth).flatten(1).sum(1).float()
        per_iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        for index, sample_id in enumerate(batch["sample_ids"]):
            row = {
                "sample_index": int(batch["sample_indices"][index]),
                "sample_id": sample_id,
                "class_name": batch["class_names"][index],
                "image_path": batch["image_paths"][index],
                "mask_path": batch["mask_paths"][index],
                "mean_probability": float(probabilities[index].mean()),
                "foreground_fraction_true": float(masks[index].mean()),
                "foreground_fraction_predicted": float(binary[index].float().mean()),
                "iou": float(per_iou[index]),
            }
            rows.append(row)
            if save_predictions and saved < settings.visualization_sample_count:
                folder = settings.output_dir / "figures" / f"{model_kind}_examples"
                save_prediction_panel(
                    folder / f"{saved:03d}_{sample_id.replace('/', '_')}.png",
                    image=batch["images"][index],
                    ground_truth=masks[index],
                    student_probability=probabilities[index] if model_kind == "student" else None,
                    teacher_probability=probabilities[index] if model_kind == "teacher" else None,
                    title=f"{sample_id} | IoU={float(per_iou[index]):.3f}",
                )
                saved += 1
            if save_predictions:
                worst_examples.append(
                    {
                        "iou": float(per_iou[index]),
                        "sample_id": sample_id,
                        "image": batch["images"][index].copy(),
                        "truth": masks[index].detach().cpu(),
                        "probability": probabilities[index].detach().cpu(),
                    }
                )
                worst_examples.sort(key=lambda item: item["iou"])
                del worst_examples[settings.visualization_sample_count:]
    metrics = accumulator.compute()
    if model_kind == "student":
        if routing_samples != metrics["samples"]:
            raise RuntimeError(
                f"Router diagnostics saw {routing_samples} samples, segmentation "
                f"metrics saw {metrics['samples']}"
            )
        selected_denominator = routing_selected.clamp_min(1.0)
        metrics["router"] = {
            "samples": routing_samples,
            "top_k": settings.top_k,
            "selection_rate_per_expert": (
                routing_selected / routing_samples
            ).tolist(),
            "mean_sparse_weight_per_expert": (
                routing_weight / routing_samples
            ).tolist(),
            "mean_weight_when_selected_per_expert": (
                routing_weight / selected_denominator
            ).tolist(),
            "mean_dense_probability_per_expert": (
                routing_probability / routing_samples
            ).tolist(),
            "normalized_entropy": routing_entropy_sum / routing_samples,
            "balance_loss": routing_balance_sum / routing_samples,
            "importance_loss": routing_importance_sum / routing_samples,
        }
    if save_predictions:
        write_csv(
            settings.output_dir / "metrics" / f"{model_kind}_test_predictions.csv",
            rows,
            [
                "sample_index", "sample_id", "class_name", "image_path", "mask_path",
                "mean_probability", "foreground_fraction_true",
                "foreground_fraction_predicted", "iou",
            ],
        )
        for rank, item in enumerate(worst_examples):
            save_prediction_panel(
                settings.output_dir / "figures" / "failure_cases"
                / f"{model_kind}_{rank:03d}_{item['sample_id'].replace('/', '_')}.png",
                image=item["image"],
                ground_truth=item["truth"],
                student_probability=item["probability"] if model_kind == "student" else None,
                teacher_probability=item["probability"] if model_kind == "teacher" else None,
                title=(
                    f"Worst-case rank {rank + 1}: {item['sample_id']} | "
                    f"IoU={item['iou']:.3f}"
                ),
            )
    return metrics, rows


def _model_forward(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, FrozenQwenVisionTeacher):
        return model(inputs["pixel_values"], inputs["image_grid_thw"])
    if isinstance(model, VisionOpticalSaliencyStudent):
        logits, spatial, _ = model(inputs["pixel_values"], inputs["image_grid_thw"])
        return logits, spatial
    raise TypeError(f"Unsupported model type {type(model).__name__}")


def load_teacher_head_for_cache(
    loaded: LoadedVisionBackbone,
    settings: Any,
    checkpoint_path: Path,
) -> FrozenQwenVisionTeacher:
    teacher = build_teacher(loaded, settings)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher.head.load_state_dict(payload["head_state_dict"], strict=True)
    teacher.eval()
    return teacher


def _load_mask_cache(settings: Any, split: str) -> TeacherMaskCache:
    checkpoint = settings.teacher_checkpoint
    directory = settings.teacher_mask_cache
    if checkpoint is None or directory is None:
        raise RuntimeError(
            "Mask KD requires mask_kd.teacher_checkpoint and mask_kd.teacher_mask_cache"
        )
    return TeacherMaskCache(
        directory, split, expected_cache_identity(settings, checkpoint)
    )


def _history_row(
    epoch: int,
    learning_rate: float,
    train: dict[str, Any],
    parts: dict[str, float],
    test: dict[str, Any],
    epoch_time: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train["loss"],
        "train_bce": parts["bce"],
        "train_dice_loss": parts["dice_loss"],
        "train_mask_kd": parts["mask_kd"],
        "train_router_balance": parts["router_balance"],
        "train_router_importance": parts["router_importance"],
        "train_mean_iou": train["mean_iou"],
        "train_mean_dice": train["mean_dice"],
        "train_mae": train["mae"],
        "train_pixel_accuracy": train["pixel_accuracy"],
        "test_loss": test["loss"],
        "test_mean_iou": test["mean_iou"],
        "test_mean_dice": test["mean_dice"],
        "test_mae": test["mae"],
        "test_pixel_accuracy": test["pixel_accuracy"],
        "epoch_time_sec": epoch_time,
    }


def _empty_test_metrics() -> dict[str, float]:
    return {
        "loss": float("nan"), "mean_iou": float("nan"),
        "mean_dice": float("nan"), "mae": float("nan"),
        "pixel_accuracy": float("nan"),
    }


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    write_csv(path, history, HISTORY_FIELDS)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _autocast(settings: Any):
    if not settings.amp_enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if settings.dtype.lower() == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _print_parameter_report(report: dict[str, Any]) -> None:
    print(
        f"trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}",
        flush=True,
    )
    for item in report["trainable_parameter_list"]:
        print(
            f"  {item['name']} shape={item['shape']} params={item['parameters']:,}",
            flush=True,
        )
