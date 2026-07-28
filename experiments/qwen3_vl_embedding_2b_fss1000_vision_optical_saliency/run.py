from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FSSSaliencyDataset, collate_saliency, prepare_fss1000
from .io_utils import environment_report, seed_everything, write_json
from .modeling import (
    build_student,
    build_teacher,
    load_vision_backbone,
    preprocess_vision,
)
from .settings import load_settings, save_resolved_config
from .teacher_cache import build_teacher_mask_cache
from .training import (
    build_loaders,
    load_teacher_head_for_cache,
    save_teacher_student_comparison_examples,
    test_student,
    test_teacher,
    train_student,
    train_teacher,
)


PHASES = {
    "prepare_data",
    "shape_smoke",
    "teacher_train",
    "teacher_test",
    "cache_teacher_masks",
    "student_train",
    "student_test",
    "all",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FSS-1000 frozen-Qwen Vision and Optical MoE16 saliency"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_fss1000(settings, persist=True)
    if args.phase == "prepare_data":
        _print_dataset(bundle.metadata)
        return 0

    device = _device(settings.device)
    loaded = load_vision_backbone(settings, device)
    save_resolved_config(settings)
    write_json(
        settings.output_dir / "qwen_vision.json",
        {
            "model_id": settings.model_id,
            "vision_depth": settings.vision_depth,
            "vision_hidden_size": settings.vision_hidden_size,
            "processor_min_pixels": settings.processor_min_pixels,
            "processor_max_pixels": settings.processor_max_pixels,
            "language_model_forward_used": False,
            "load_time_sec": loaded.load_time_sec,
        },
    )
    if args.phase == "shape_smoke":
        _shape_smoke(loaded, bundle, settings)
        return 0
    if args.phase == "teacher_train":
        train_teacher(loaded, bundle, settings)
        return 0
    if args.phase == "teacher_test":
        metrics = test_teacher(
            loaded, bundle, settings, _teacher_checkpoint(settings)
        )
        _print_metrics("Electronic teacher", metrics)
        return 0
    if args.phase == "cache_teacher_masks":
        _cache_teacher_masks(loaded, bundle, settings)
        return 0
    if args.phase == "student_train":
        train_student(loaded, bundle, settings)
        return 0
    if args.phase == "student_test":
        metrics = test_student(loaded, bundle, settings)
        _print_metrics("Optical student", metrics)
        return 0

    # all: no validation is created. Checkpoints are selected strictly by
    # minimum train loss; per-epoch test metrics are observation only.
    _shape_smoke(loaded, bundle, settings)
    teacher_path = _teacher_checkpoint(settings)
    if settings.mask_kd_weight <= 0:
        train_teacher(loaded, bundle, settings)
        teacher_path = settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt"
    elif not teacher_path.is_file():
        raise FileNotFoundError(
            f"Mask-KD config references missing teacher checkpoint: {teacher_path}"
        )
    teacher_metrics = test_teacher(loaded, bundle, settings, teacher_path)
    if settings.mask_kd_weight > 0:
        _cache_teacher_masks(loaded, bundle, settings)
    train_student(loaded, bundle, settings)
    student_path = settings.output_dir / "checkpoints" / "student_best_train_loss.pt"
    student_metrics = test_student(loaded, bundle, settings, student_path)
    save_teacher_student_comparison_examples(
        loaded,
        bundle,
        settings,
        teacher_checkpoint=teacher_path,
        student_checkpoint=student_path,
    )
    comparison = {
        "electronic_teacher": teacher_metrics,
        "optical_student": student_metrics,
        "optical_minus_teacher": {
            metric: float(student_metrics[metric]) - float(teacher_metrics[metric])
            for metric in ("mean_iou", "mean_dice", "mae", "pixel_accuracy")
        },
        "test_used_for_checkpoint_selection": False,
        "selection_criterion": "minimum_train_loss",
        "class_disjoint_split": bundle.metadata["class_disjoint"],
        "mask_kd_enabled": settings.mask_kd_weight > 0,
    }
    write_json(settings.output_dir / "metrics" / "comparison.json", comparison)
    _print_metrics("Electronic teacher", teacher_metrics)
    _print_metrics("Optical student", student_metrics)
    return 0


@torch.no_grad()
def _shape_smoke(loaded: Any, bundle: Any, settings: Any) -> dict[str, Any]:
    dataset = FSSSaliencyDataset(bundle.train_records[:1], settings, training=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_saliency)
    batch = next(iter(loader))
    image_array = np.asarray(batch["images"][0].convert("RGB"))
    image_shape = [1, 3, int(image_array.shape[0]), int(image_array.shape[1])]
    inputs = preprocess_vision(loaded.processor, batch["images"], loaded.device)
    grids = inputs["image_grid_thw"].long()
    counts = grids.prod(dim=-1).tolist()
    if max(counts) > settings.max_visual_tokens:
        raise RuntimeError(
            f"visual token count {max(counts)} exceeds max_visual_tokens="
            f"{settings.max_visual_tokens}. Lower processor_max_pixels; silent crop, "
            "truncate, pooling, or reshape is forbidden."
        )
    teacher = build_teacher(loaded, settings)
    teacher_logits, teacher_spatial = teacher(
        inputs["pixel_values"], inputs["image_grid_thw"]
    )
    teacher.close()
    student = build_student(loaded, settings)
    student.activate()
    student_logits, student_spatial, ccd = student(
        inputs["pixel_values"], inputs["image_grid_thw"]
    )
    report = {
        "input_rgb_shape": image_shape,
        "mask_shape": list(batch["masks"].shape),
        "qwen_pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_grid_thw": grids.cpu().tolist(),
        "visual_token_counts": [int(value) for value in counts],
        "teacher_vision_spatial_hidden_shape": list(teacher_spatial.shape),
        "teacher_mask_logits_shape": list(teacher_logits.shape),
        "vision_ccd_shape": list(ccd.shape),
        "restored_optical_spatial_feature_shape": list(student_spatial.shape),
        "student_mask_logits_shape": list(student_logits.shape),
        "strict_token_mapping": True,
    }
    expected_logits = [1, 1, settings.image_size, settings.image_size]
    if report["teacher_mask_logits_shape"] != expected_logits:
        raise RuntimeError(f"Teacher smoke output mismatch: {report}")
    if report["student_mask_logits_shape"] != expected_logits:
        raise RuntimeError(f"Student smoke output mismatch: {report}")
    student.restore_native()
    write_json(settings.output_dir / "shape_smoke.json", report)
    for name, value in report.items():
        print(f"{name}: {value}", flush=True)
    return report


def _cache_teacher_masks(loaded: Any, bundle: Any, settings: Any) -> None:
    checkpoint = _teacher_checkpoint(settings)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint is required before mask caching: {checkpoint}"
        )
    directory = settings.teacher_mask_cache or (
        settings.output_dir / "teacher_mask_cache"
    )
    teacher = load_teacher_head_for_cache(loaded, settings, checkpoint)
    train_loader, test_loader = build_loaders(
        bundle,
        settings,
        train_batch_size=settings.teacher_batch_size,
        train_augmentation=False,
    )
    build_teacher_mask_cache(
        teacher, loaded.processor, train_loader, directory,
        split="train", settings=settings, checkpoint_path=checkpoint,
        device=loaded.device,
    )
    build_teacher_mask_cache(
        teacher, loaded.processor, test_loader, directory,
        split="test", settings=settings, checkpoint_path=checkpoint,
        device=loaded.device,
    )
    teacher.close()


def _teacher_checkpoint(settings: Any) -> Path:
    return settings.teacher_checkpoint or (
        settings.output_dir / "checkpoints" / "teacher_best_train_loss.pt"
    )


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration requests CUDA but torch.cuda.is_available() is false")
    return torch.device(value)


def _print_dataset(metadata: dict[str, Any]) -> None:
    print(
        f"FSS-1000 train={metadata['train_classes']} classes/"
        f"{metadata['train_images']} images, test={metadata['test_classes']} classes/"
        f"{metadata['test_images']} images, class_disjoint={metadata['class_disjoint']}",
        flush=True,
    )


def _print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name}: mIoU={metrics['mean_iou']:.4f} Dice={metrics['mean_dice']:.4f} "
        f"MAE={metrics['mae']:.4f} PixelAcc={metrics['pixel_accuracy']:.4f}",
        flush=True,
    )

