from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from torch import nn

from .io_utils import write_json
from .optics import VisionBlockReplacement
from .student_training import evaluate_online


def run_and_save_inference(
    *,
    model: nn.Module,
    processor: Any,
    replacement: VisionBlockReplacement,
    head: nn.Module,
    loader: Any,
    class_names: Sequence[str],
    prompt: str,
    device: Any,
    student: bool,
    max_batches: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    metrics = evaluate_online(
        model,
        processor,
        replacement,
        head,
        loader,
        class_names,
        prompt,
        device,
        student=student,
        max_batches=max_batches,
    )
    report = {
        "model": (
            f"qwen3_vl_with_{len(replacement.block_groups)}_single_mask_optical_"
            f"conversions_for_vision_blocks_{replacement.block_groups[0][0]}_"
            f"{replacement.block_groups[-1][1]}"
            if student
            else "full_electronic_qwen3_vl_multimodal_mlp"
        ),
        "distillation_block_groups": [list(group) for group in replacement.block_groups],
        "classes": list(class_names),
        "prompt": prompt,
        "metrics": metrics,
    }
    name = "student_inference.json" if student else "teacher_inference.json"
    write_json(output_dir / "metrics" / name, report)
    return report


def write_comparison(
    output_dir: Path,
    dataset: str,
    class_names: Sequence[str],
    prompt: str,
    replacement: dict[str, Any],
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    teacher = _read_json(output_dir / "metrics" / "teacher_inference.json")
    student = _read_json(output_dir / "metrics" / "student_inference.json")
    teacher_top1 = float(teacher["metrics"]["top1_accuracy"])
    teacher_top5 = float(teacher["metrics"]["top5_accuracy"])
    student_top1 = float(student["metrics"]["top1_accuracy"])
    student_top5 = float(student["metrics"]["top5_accuracy"])
    comparison = {
        "dataset": dataset,
        "classes": list(class_names),
        "prompt": prompt,
        "replacement": replacement,
        "teacher": {
            "model": teacher["model"],
            "top1_accuracy": teacher_top1,
            "top5_accuracy": teacher_top5,
        },
        "student": {
            "model": student["model"],
            "top1_accuracy": student_top1,
            "top5_accuracy": student_top5,
        },
        "accuracy_drop": {
            "top1": student_top1 - teacher_top1,
            "top5": student_top5 - teacher_top5,
        },
        "loss_weights": loss_weights,
    }
    write_json(output_dir / "metrics" / "comparison.json", comparison)
    _print_comparison(comparison)
    return comparison


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required metrics file is missing: {path}. Run teacher_inference and "
            "student_inference before compare."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_comparison(comparison: dict[str, Any]) -> None:
    teacher = comparison["teacher"]
    student = comparison["student"]
    drop = comparison["accuracy_drop"]
    print("\nBDD100K Weather-4 Results\n")
    print(f"{'Model':34s} {'Top-1':>9s} {'Top-5':>9s}")
    print(
        f"{'Teacher electronic Qwen3-VL+MLP':34s} "
        f"{teacher['top1_accuracy'] * 100:8.2f}% {teacher['top5_accuracy'] * 100:8.2f}%"
    )
    print(
        f"{'Optical student':34s} "
        f"{student['top1_accuracy'] * 100:8.2f}% {student['top5_accuracy'] * 100:8.2f}%"
    )
    print(
        f"{'Drop':34s} {drop['top1'] * 100:+8.2f}% {drop['top5'] * 100:+8.2f}%"
    )
