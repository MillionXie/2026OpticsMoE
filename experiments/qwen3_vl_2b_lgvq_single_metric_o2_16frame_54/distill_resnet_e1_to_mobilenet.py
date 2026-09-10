"""Distill the rejected ResNet E1 correction into the 0.289M MobileNet front.

The ResNet teacher is needed only for this offline initialization step.  The
emitted student contains the truncated frozen MobileNetV2 front contract and
its 49,536-parameter E1 adapter, but no ResNet tensors or inference module.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import (
    load_mobilenet_feature_cache,
    load_resnet_feature_cache,
    read_manifest,
)
from .distill_resnet_e1_to_tiny_rgb import (
    _adapter_state,
    _compatible_common_state,
    _load,
    _sha256,
    _state,
)
from .modeling import (
    FrozenMobileNetElectronicCorrection,
    FrozenResNetElectronicCorrection,
    build_model,
)
from .settings import load_settings


@torch.no_grad()
def _metrics(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    mobile: torch.Tensor,
    resnet: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    error = target_power = dot = left_power = right_power = 0.0
    for start in range(0, indices.numel(), batch_size):
        batch = indices[start : start + batch_size]
        predicted = student(mobile[batch].to(device, non_blocking=True)).float()
        target = teacher(resnet[batch].to(device, non_blocking=True)).float()
        error += float((predicted - target).square().sum())
        target_power += float(target.square().sum())
        left, right = predicted.flatten(), target.flatten()
        left, right = left - left.mean(), right - right.mean()
        dot += float((left * right).sum())
        left_power += float(left.square().sum())
        right_power += float(right.square().sum())
    return {
        "normalized_mse": error / max(target_power, 1.0e-12),
        "pcc": dot / math.sqrt(max(left_power * right_power, 1.0e-24)),
    }


def distill(
    *,
    teacher_config: Path,
    teacher_checkpoint: Path,
    student_config: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    teacher_settings = load_settings(teacher_config)
    student_settings = load_settings(student_config)
    if teacher_settings.resnet_feature_cache_path is None:
        raise ValueError("Teacher config must declare data.resnet_feature_cache")
    if student_settings.mobilenet_feature_cache_path is None:
        raise ValueError("Student config must declare data.mobilenet_feature_cache")
    rows = read_manifest(student_settings.manifest_path)
    sample_ids = [row.sample_id for row in rows]
    mobile = load_mobilenet_feature_cache(
        student_settings.mobilenet_feature_cache_path,
        sample_ids=sample_ids,
        frame_count=student_settings.frame_count,
        token_grid=student_settings.token_grid,
        width=student_settings.mobilenet_feature_width,
    )["tokens"]
    resnet = load_resnet_feature_cache(
        teacher_settings.resnet_feature_cache_path,
        sample_ids=sample_ids,
        frame_count=teacher_settings.frame_count,
        token_grid=teacher_settings.token_grid,
    )["tokens"]
    train_indices = torch.tensor(
        [i for i, row in enumerate(rows) if row.split == "train"], dtype=torch.long
    )
    test_indices = torch.tensor(
        [i for i, row in enumerate(rows) if row.split == "test"], dtype=torch.long
    )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    teacher_payload = _load(teacher_checkpoint)
    teacher_state = _state(teacher_payload)
    teacher = FrozenResNetElectronicCorrection(teacher_settings)
    teacher.load_state_dict(
        _adapter_state(teacher_state, "resnet_electronic_correction."), strict=True
    )
    teacher.to(device).eval().requires_grad_(False)
    student = FrozenMobileNetElectronicCorrection(student_settings).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_pcc, best_epoch = -1.0, 0
    best_state = copy.deepcopy(student.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        student.train()
        order = train_indices[
            torch.randperm(train_indices.numel(), generator=generator)
        ]
        loss_sum = 0.0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            predicted = student(mobile[batch].to(device, non_blocking=True)).float()
            with torch.no_grad():
                target = teacher(resnet[batch].to(device, non_blocking=True)).float()
            scale = target.square().mean().detach().clamp_min(1.0e-6)
            nmse = (predicted - target).square().mean() / scale
            cosine = 1.0 - F.cosine_similarity(
                predicted.flatten(1), target.flatten(1), dim=-1
            ).mean()
            loss = nmse + 0.10 * cosine
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch.numel()
        test = _metrics(
            student, teacher, mobile, resnet, test_indices,
            batch_size=batch_size, device=device,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / train_indices.numel(),
            "test_feature": test,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} loss={record['train_loss']:.6f} "
            f"test_feature_PCC={test['pcc']:.6f} NMSE={test['normalized_mse']:.6f}",
            flush=True,
        )
        if test["pcc"] > best_pcc:
            best_pcc, best_epoch = test["pcc"], epoch
            best_state = copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state, strict=True)
    train_metrics = _metrics(
        student, teacher, mobile, resnet, train_indices,
        batch_size=batch_size, device=device,
    )
    test_metrics = _metrics(
        student, teacher, mobile, resnet, test_indices,
        batch_size=batch_size, device=device,
    )
    full_student = build_model(student_settings)
    full_student.load_state_dict(
        _compatible_common_state(full_student, teacher_state), strict=False
    )
    if full_student.mobilenet_electronic_correction is None:
        raise RuntimeError("Student did not create the MobileNetV2 E1 adapter")
    full_student.mobilenet_electronic_correction.load_state_dict(best_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "best_mobilenet_e1_adapter.pt"
    full_path = output_dir / "distilled_student_initialization.pt"
    torch.save({"state_dict": best_state, "best_epoch": best_epoch}, adapter_path)
    torch.save(
        {
            "schema_version": 1,
            "architecture": student_settings.architecture_label,
            "epoch": 0,
            "state_dict": full_student.state_dict(),
            "selection_source": "resnet_e1_feature_distillation",
            "teacher_or_resnet_loaded_during_student_inference": False,
        },
        full_path,
    )
    report = {
        "schema_version": 1,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "teacher_resnet_used_only_during_distillation": True,
        "student_inference_contains_resnet": False,
        "frozen_mobilenet_front_parameters": {
            64: 239_360,
            96: 305_984,
        }[student_settings.mobilenet_feature_width],
        "student_adapter_parameters": sum(p.numel() for p in student.parameters()),
        "total_added_electronic_parameters": {
            64: 239_360,
            96: 305_984,
        }[student_settings.mobilenet_feature_width] + sum(
            p.numel() for p in student.parameters()
        ),
        "best_epoch": best_epoch,
        "train_feature_metrics": train_metrics,
        "test_feature_metrics": test_metrics,
        "adapter_checkpoint": str(adapter_path.resolve()),
        "adapter_checkpoint_sha256": _sha256(adapter_path),
        "student_initialization": str(full_path.resolve()),
        "student_initialization_sha256": _sha256(full_path),
        "history": history,
    }
    (output_dir / "distillation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", dest="device_name", default="auto")
    parser.add_argument("--seed", type=int, default=742)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    report = distill(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
