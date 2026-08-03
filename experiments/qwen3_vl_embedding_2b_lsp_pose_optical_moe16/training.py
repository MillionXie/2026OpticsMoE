from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)

from .datasets import (
    JOINT_NAMES,
    SKELETON_EDGES,
    DatasetBundle,
    LSPPoseDataset,
    pose_collate,
)
from .losses import hardargmax_coordinates, masked_coordinate_loss, masked_heatmap_mse
from .metrics import PoseMetricAccumulator
from .modeling import (
    FrozenQwenVisionPoseTeacher,
    VisionOpticalPoseStudent,
    build_student,
    build_teacher,
    preprocess_vision,
    trainable_parameter_report,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    bundle: DatasetBundle,
    settings: Any,
    *,
    training: bool,
    train_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train = LSPPoseDataset(bundle.train, settings, training=training)
    test = LSPPoseDataset(bundle.test, settings, training=False)
    generator = torch.Generator().manual_seed(settings.random_seed)
    common = {
        "num_workers": settings.num_workers,
        "collate_fn": pose_collate,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": settings.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    return (
        DataLoader(
            train, batch_size=(train_batch_size or settings.teacher_batch_size),
            shuffle=training, generator=generator, **common,
        ),
        DataLoader(test, batch_size=settings.inference_batch_size, shuffle=False, **common),
    )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def train_teacher(loaded: Any, bundle: DatasetBundle, settings: Any) -> dict[str, Any]:
    train_loader, test_loader = build_loaders(
        bundle, settings, training=True, train_batch_size=settings.teacher_batch_size,
    )
    model = build_teacher(loaded, settings)
    report = trainable_parameter_report(model, "teacher")
    _write_model_report(settings, "teacher", report, model.head.specification())
    _print_report(report)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=settings.teacher_learning_rate,
        weight_decay=settings.weight_decay,
    )
    history, best_loss = [], float("inf")
    (settings.output_dir / "metrics" / "teacher_training_history.csv").unlink(missing_ok=True)
    try:
        for epoch in range(1, settings.teacher_epochs + 1):
            started = time.perf_counter()
            train_metrics = _train_epoch(
                model, "teacher", train_loader, loaded.processor, loaded.device,
                optimizer, settings, epoch,
            )
            test_metrics, _ = evaluate_model(
                model, "teacher", test_loader, loaded.processor, loaded.device,
                settings, phase="teacher_test", epoch=epoch, save_outputs=False,
            )
            row = _history_row(epoch, train_metrics, test_metrics, time.perf_counter() - started)
            history.append(row)
            _append_history(settings.output_dir / "metrics" / "teacher_training_history.csv", row)
            print(
                f"teacher epoch {epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"train_PCK={train_metrics['pck_at_0.2_torso']:.4f} "
                f"test_PCK={test_metrics['pck_at_0.2_torso']:.4f} "
                f"test_PCKh={test_metrics['pckh_at_0.5_head']:.4f}", flush=True,
            )
            _save_teacher(model, settings.output_dir / "checkpoints" / "teacher_last.pt", epoch, settings)
            if train_metrics["loss"] < best_loss:
                best_loss = train_metrics["loss"]
                _save_teacher(
                    model, settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt",
                    epoch, settings,
                )
        _save_curves(history, settings.output_dir / "figures" / "teacher_training_curves.png")
    finally:
        model.close()
    return {"best_train_loss": best_loss, "epochs": settings.teacher_epochs}


def train_student(loaded: Any, bundle: DatasetBundle, settings: Any) -> dict[str, Any]:
    train_loader, test_loader = build_loaders(
        bundle, settings, training=True, train_batch_size=settings.student_batch_size,
    )
    model = build_student(loaded, settings)
    report = trainable_parameter_report(model, "student")
    optical = model.core.parameter_breakdown()
    _write_model_report(settings, "student", report, model.head.specification(), optical)
    _print_report(report)
    phase_parameters = list(model.core.expert_layers.parameters()) + list(model.core.global_phase.parameters())
    router_parameters = list(model.core.router.parameters())
    special = {id(p) for p in [*phase_parameters, *router_parameters]}
    base_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in special]
    optimizer = torch.optim.AdamW([
        {"params": base_parameters, "lr": settings.student_learning_rate},
        {"params": router_parameters, "lr": settings.router_learning_rate},
        {"params": phase_parameters, "lr": settings.phase_learning_rate},
    ], weight_decay=settings.weight_decay)
    history, best_loss = [], float("inf")
    (settings.output_dir / "metrics" / "student_training_history.csv").unlink(missing_ok=True)
    try:
        for epoch in range(1, settings.student_epochs + 1):
            started = time.perf_counter()
            train_metrics = _train_epoch(
                model, "student", train_loader, loaded.processor, loaded.device,
                optimizer, settings, epoch,
            )
            test_metrics, _ = evaluate_model(
                model, "student", test_loader, loaded.processor, loaded.device,
                settings, phase="student_test", epoch=epoch, save_outputs=False,
            )
            row = _history_row(epoch, train_metrics, test_metrics, time.perf_counter() - started)
            history.append(row)
            _append_history(settings.output_dir / "metrics" / "student_training_history.csv", row)
            print(
                f"student epoch {epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"train_PCK={train_metrics['pck_at_0.2_torso']:.4f} "
                f"test_PCK={test_metrics['pck_at_0.2_torso']:.4f} "
                f"test_PCKh={test_metrics['pckh_at_0.5_head']:.4f} "
                f"balance={train_metrics['router_balance_loss']:.4f}", flush=True,
            )
            _save_student(model, settings.output_dir / "checkpoints" / "student_last.pt", epoch, settings)
            if train_metrics["loss"] < best_loss:
                best_loss = train_metrics["loss"]
                _save_student(
                    model, settings.output_dir / "checkpoints" / "student_best_train_loss.pt",
                    epoch, settings,
                )
        _save_curves(history, settings.output_dir / "figures" / "student_training_curves.png")
    finally:
        model.restore_native()
    return {"best_train_loss": best_loss, "epochs": settings.student_epochs}


def infer_teacher(loaded: Any, bundle: DatasetBundle, settings: Any) -> dict[str, Any]:
    _, loader = build_loaders(bundle, settings, training=False)
    model = build_teacher(loaded, settings)
    checkpoint = settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt"
    payload = _load(checkpoint, loaded.device)
    model.head.load_state_dict(payload["head"])
    try:
        metrics, _ = evaluate_model(
            model, "teacher", loader, loaded.processor, loaded.device, settings,
            phase="teacher_inference", epoch=int(payload["epoch"]), save_outputs=True,
        )
    finally:
        model.close()
    return metrics


def infer_student(loaded: Any, bundle: DatasetBundle, settings: Any) -> dict[str, Any]:
    _, loader = build_loaders(bundle, settings, training=False)
    model = build_student(loaded, settings)
    checkpoint = settings.output_dir / "checkpoints" / "student_best_train_loss.pt"
    payload = _load(checkpoint, loaded.device)
    model.core.load_state_dict(payload["core"])
    model.head.load_state_dict(payload["head"])
    model.core.set_intermediate_field_capture(
        True, min(4, settings.visualization_sample_count)
    )
    try:
        metrics, _ = evaluate_model(
            model, "student", loader, loaded.processor, loaded.device, settings,
            phase="student_inference", epoch=int(payload["epoch"]), save_outputs=True,
        )
        _save_optical_debug(model, settings.output_dir / "figures" / "student_optical_debug")
    finally:
        model.restore_native()
    return metrics


def _train_epoch(
    model: nn.Module,
    kind: str,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    settings: Any,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    accumulator = PoseMetricAccumulator()
    sums = {"loss": 0.0, "heatmap_loss": 0.0, "coordinate_loss": 0.0,
            "router_balance_loss": 0.0, "router_importance_loss": 0.0,
            "phase_dc_loss": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        inputs = preprocess_vision(processor, batch["images"], device)
        target_heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=_amp_dtype(settings.dtype),
            enabled=settings.amp_enabled and device.type == "cuda",
        ):
            outputs = model(**inputs)
            predictions = outputs[0]
            heatmap_loss = masked_heatmap_mse(predictions, target_heatmaps, visible)
            coordinate_loss = masked_coordinate_loss(
                predictions, keypoints, visible, settings.image_size,
            )
            if kind == "student":
                balance, importance = model.router_losses()
                dc = (
                    phase_dc_loss(model)
                    if settings.phase_dc_weight > 0.0
                    else predictions.new_zeros(())
                )
            else:
                balance = importance = predictions.new_zeros(())
                dc = predictions.new_zeros(())
            loss = (
                settings.heatmap_loss_weight * heatmap_loss
                + settings.coordinate_loss_weight * coordinate_loss
                + settings.router_balance_weight * balance
                + settings.router_importance_weight * importance
                + settings.phase_dc_weight * dc
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {kind} loss at epoch={epoch} batch={batch_index}")
        loss.backward()
        optimizer.step()
        count = len(batch["images"])
        samples += count
        values = {
            "loss": loss, "heatmap_loss": heatmap_loss,
            "coordinate_loss": coordinate_loss, "router_balance_loss": balance,
            "router_importance_loss": importance,
            "phase_dc_loss": dc,
        }
        for name, value in values.items():
            sums[name] += float(value.detach()) * count
        accumulator.update(
            predictions, keypoints, visible, batch["torso_scale"],
            batch["head_scale"], settings.image_size,
        )
        if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
            print(
                f"[{kind}_train] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={sums['loss']/samples:.5f} heatmap={sums['heatmap_loss']/samples:.5f} "
                f"coord={sums['coordinate_loss']/samples:.5f} "
                f"phase_dc={sums['phase_dc_loss']/samples:.5f}", flush=True,
            )
    result = {name: value / max(samples, 1) for name, value in sums.items()}
    result.update(accumulator.compute())
    result["samples"] = samples
    return result


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    kind: str,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    settings: Any,
    *,
    phase: str,
    epoch: int,
    save_outputs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulator = PoseMetricAccumulator()
    sums = {"loss": 0.0, "heatmap_loss": 0.0, "coordinate_loss": 0.0}
    samples, rows, examples, printed_shapes = 0, [], [], False
    for batch in loader:
        inputs = preprocess_vision(processor, batch["images"], device)
        targets = batch["heatmaps"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        outputs = model(**inputs)
        heatmaps, spatial = outputs[0], outputs[1]
        heatmap_loss = masked_heatmap_mse(heatmaps, targets, visible)
        coordinate_loss = masked_coordinate_loss(heatmaps, keypoints, visible, settings.image_size)
        loss = settings.heatmap_loss_weight * heatmap_loss + settings.coordinate_loss_weight * coordinate_loss
        count = len(batch["images"])
        samples += count
        sums["loss"] += float(loss) * count
        sums["heatmap_loss"] += float(heatmap_loss) * count
        sums["coordinate_loss"] += float(coordinate_loss) * count
        predicted = accumulator.update(
            heatmaps, keypoints, visible, batch["torso_scale"], batch["head_scale"],
            settings.image_size,
        )
        if not printed_shapes:
            token_counts = [int(t * h * w) for t, h, w in inputs["image_grid_thw"].detach().cpu().tolist()]
            detector_shape = list(outputs[2].shape) if kind == "student" else None
            print(json.dumps({
                "input_image_shape": [count, 3, settings.image_size, settings.image_size],
                "qwen_pixel_values_shape": list(inputs["pixel_values"].shape),
                "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
                "visual_token_counts": token_counts,
                "spatial_feature_shape": list(spatial.shape),
                "vision_ccd_shape": detector_shape,
                "pose_heatmap_shape": list(heatmaps.shape),
            }, ensure_ascii=False), flush=True)
            printed_shapes = True
        target_cpu = batch["keypoints"].cpu()
        visible_cpu = batch["visible"].cpu()
        for sample_index in range(count):
            for joint, name in enumerate(JOINT_NAMES):
                valid = bool(visible_cpu[sample_index, joint])
                target_xy = target_cpu[sample_index, joint]
                pred_xy = predicted[sample_index, joint]
                error = (
                    float(torch.linalg.vector_norm(pred_xy - target_xy)) if valid else None
                )
                rows.append({
                    "sample_id": batch["sample_id"][sample_index],
                    "source": batch["source"][sample_index],
                    "image_path": batch["image_path"][sample_index],
                    "joint_index": joint,
                    "joint_name": name,
                    "visible": valid,
                    "true_x": float(target_xy[0]) if valid else "",
                    "true_y": float(target_xy[1]) if valid else "",
                    "pred_x": float(pred_xy[0]),
                    "pred_y": float(pred_xy[1]),
                    "pixel_error": error if error is not None else "",
                })
            if len(examples) < settings.visualization_sample_count:
                examples.append((
                    batch["images"][sample_index].copy(),
                    target_cpu[sample_index], predicted[sample_index],
                    visible_cpu[sample_index],
                    heatmaps[sample_index].detach().cpu(),
                    batch["sample_id"][sample_index],
                ))
    metrics = {name: value / max(samples, 1) for name, value in sums.items()}
    metrics.update(accumulator.compute())
    metrics.update({"samples": samples, "phase": phase, "epoch": epoch, "model": kind})
    if save_outputs:
        metrics_dir = settings.output_dir / "metrics"
        figures_dir = settings.output_dir / "figures" / phase
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{phase}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_predictions(metrics_dir / f"{phase}_predictions.csv", rows)
        _save_pose_examples(examples, figures_dir)
    return metrics, rows


def save_comparison(settings: Any, teacher: dict[str, Any], student: dict[str, Any]) -> None:
    keys = ("pck_at_0.2_torso", "pckh_at_0.5_head", "mean_pixel_error", "normalized_mean_error_torso")
    result = {
        "electronic_teacher": teacher,
        "optical_student": student,
        "student_minus_teacher": {
            key: (student[key] - teacher[key])
            for key in keys if student.get(key) is not None and teacher.get(key) is not None
        },
        "protocol": "best training-loss checkpoints; test is observational and not used for selection",
    }
    path = settings.output_dir / "metrics" / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_teacher(model: FrozenQwenVisionPoseTeacher, path: Path, epoch: int, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "head": model.head.state_dict(), "settings": settings.to_dict()}, path)


def _save_student(model: VisionOpticalPoseStudent, path: Path, epoch: int, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "core": model.core.state_dict(), "head": model.head.state_dict(),
        "settings": settings.to_dict(),
    }, path)


def _load(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _history_row(epoch: int, train: dict[str, Any], test: dict[str, Any], seconds: float) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "epoch_time_sec": seconds}
    row.update({f"train_{key}": value for key, value in train.items() if not isinstance(value, dict)})
    row.update({f"test_{key}": value for key, value in test.items() if not isinstance(value, dict)})
    return row


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_model_report(
    settings: Any,
    name: str,
    trainable: dict[str, Any],
    head: dict[str, Any],
    optical: dict[str, Any] | None = None,
) -> None:
    path = settings.output_dir / "metrics" / f"{name}_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": name, "trainable": trainable, "pose_head": head,
        "optical_parameter_breakdown": optical,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"trainable parameters={report['trainable_parameters']:,} "
        f"tensors={report['trainable_tensors']}", flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(f"  {row['name']} shape={row['shape']} params={row['parameters']:,}", flush=True)


def _save_pose_examples(examples: list[tuple[Any, ...]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for image, target, predicted, visible, heatmaps, sample_id in examples:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for a, b in SKELETON_EDGES:
            if bool(visible[a]) and bool(visible[b]):
                draw.line([tuple(target[a].tolist()), tuple(target[b].tolist())], fill=(0, 255, 0), width=2)
            draw.line([tuple(predicted[a].tolist()), tuple(predicted[b].tolist())], fill=(255, 64, 64), width=2)
        for joint in range(len(JOINT_NAMES)):
            px, py = predicted[joint].tolist()
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 0, 0))
            if bool(visible[joint]):
                tx, ty = target[joint].tolist()
                draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=(0, 255, 0))
        canvas.save(directory / f"{sample_id}_green_gt_red_prediction.png")
        _save_heatmap_grid(
            heatmaps, directory / f"{sample_id}_predicted_joint_heatmaps.png",
        )


def _save_heatmap_grid(heatmaps: torch.Tensor, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 7, figsize=(16, 5.5), constrained_layout=True)
    image = None
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(heatmaps[index].float().numpy(), cmap="magma", origin="upper")
        axis.set_title(JOINT_NAMES[index], fontsize=8)
        axis.set_xlabel("heatmap x")
        axis.set_ylabel("heatmap y")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="predicted heatmap logit")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_optical_debug(model: VisionOpticalPoseStudent, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "optical_input_field": model.core.last_input_fields,
        "amplitude_slm_canvas": model.core.last_amplitude_slm_canvas,
        "detector_intensity": model.core.last_detector_intensity,
        "detector_readout": model.core.last_detector_readout,
    }
    for stage, value in enumerate(model.core.last_stage_fields, 1):
        tensors[f"expert_stage_{stage}_complex_field"] = value
    for name, value in tensors.items():
        if value is None:
            continue
        torch.save(value, directory / f"{name}.pt")
        display = value[0]
        if torch.is_complex(display):
            display = display.abs().square()
        while display.ndim > 2:
            display = display[0]
        _save_scalar_field(display.float(), directory / f"{name}.png", name)
    routing = {
        key: value.detach().cpu()
        for key, value in model.core.last_routing.items()
        if torch.is_tensor(value)
    }
    torch.save(routing, directory / "routing.pt")


def _save_scalar_field(value: torch.Tensor, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = axis.imshow(value.numpy(), cmap="magma", origin="upper")
    axis.set_title(
        f"{title}\nshape={tuple(value.shape)} min={float(value.min()):.3g} "
        f"max={float(value.max()):.3g} mean={float(value.mean()):.3g}"
    )
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    figure.colorbar(image, ax=axis, label="intensity / amplitude")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_curves(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="test")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Pose loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_pck_at_0.2_torso"] for row in history], label="train PCK")
    axes[1].plot(epochs, [row["test_pck_at_0.2_torso"] for row in history], label="test PCK")
    axes[1].plot(epochs, [row["test_pckh_at_0.5_head"] for row in history], label="test PCKh")
    axes[1].set(xlabel="epoch", ylabel="score", ylim=(0, 1), title="Pose accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _amp_dtype(name: str) -> torch.dtype:
    return torch.float16 if name.lower() == "float16" else torch.bfloat16
