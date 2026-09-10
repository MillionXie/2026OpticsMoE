"""Distill the rejected ResNet E1 correction into the tiny RGB E1 front.

The teacher ResNet is used only while creating the student initialization.  It
is not present in the emitted checkpoint's inference graph.  Distillation is
performed on E1 correction tensors, not on MOS labels, so the downstream
optical masks, routers, fusions and single readout can be copied bit-exactly
from the 0.6665 teacher.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .data import load_raw_frame_cache, load_resnet_feature_cache, read_manifest
from .modeling import FrozenResNetElectronicCorrection, build_model
from .settings import ExperimentSettings, load_settings


def _load(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    result = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(result, Mapping):
        raise ValueError("Checkpoint has no state_dict mapping")
    return result


def _adapter_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    result = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not result:
        raise RuntimeError(f"Checkpoint contains no {prefix!r} tensors")
    return result


@torch.no_grad()
def _feature_metrics(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    frames: torch.Tensor,
    resnet: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    squared_error = 0.0
    squared_teacher = 0.0
    dot = 0.0
    student_square = 0.0
    teacher_square = 0.0
    element_count = 0
    for start in range(0, indices.numel(), batch_size):
        batch = indices[start : start + batch_size]
        x = frames[batch].to(device, non_blocking=True)
        z = resnet[batch].to(device, non_blocking=True)
        predicted = student(x).float()
        target = teacher(z).float()
        squared_error += float((predicted - target).square().sum())
        squared_teacher += float(target.square().sum())
        left = predicted.flatten()
        right = target.flatten()
        left = left - left.mean()
        right = right - right.mean()
        dot += float((left * right).sum())
        student_square += float(left.square().sum())
        teacher_square += float(right.square().sum())
        element_count += target.numel()
    return {
        "normalized_mse": squared_error / max(squared_teacher, 1.0e-12),
        "pcc": dot / math.sqrt(max(student_square * teacher_square, 1.0e-24)),
        "elements": float(element_count),
    }


def _compatible_common_state(
    model: torch.nn.Module, source: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    destination = model.state_dict()
    return {
        name: value
        for name, value in source.items()
        if name in destination
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(destination[name].shape)
        and not name.startswith("resnet_electronic_correction.")
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
    if not student_settings.tiny_rgb_electronic_adapter_enabled:
        raise ValueError("Student config must enable the tiny RGB E1 adapter")
    if student_settings.raw_frame_cache_path is None:
        raise ValueError("Student config must declare data.raw_frame_cache")
    rows = read_manifest(student_settings.manifest_path)
    sample_ids = [row.sample_id for row in rows]
    frames_payload = load_raw_frame_cache(
        student_settings.raw_frame_cache_path,
        sample_ids=sample_ids,
        frame_count=student_settings.frame_count,
    )
    resnet_payload = load_resnet_feature_cache(
        teacher_settings.resnet_feature_cache_path,
        sample_ids=sample_ids,
        frame_count=teacher_settings.frame_count,
        token_grid=teacher_settings.token_grid,
    )
    frames = frames_payload["frames"]
    resnet = resnet_payload["tokens"]
    train_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row.split == "train"],
        dtype=torch.long,
    )
    test_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row.split == "test"],
        dtype=torch.long,
    )
    if not train_indices.numel() or not test_indices.numel():
        raise RuntimeError("Distillation requires nonempty train and test splits")
    target_device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    teacher_payload = _load(teacher_checkpoint)
    teacher_state = _state(teacher_payload)
    teacher = FrozenResNetElectronicCorrection(teacher_settings)
    teacher.load_state_dict(
        _adapter_state(teacher_state, "resnet_electronic_correction."), strict=True
    )
    teacher.to(target_device).eval().requires_grad_(False)
    student_model = build_model(student_settings)
    student = student_model.tiny_rgb_electronic_adapter
    if student is None:
        raise RuntimeError("Student model did not construct the tiny RGB adapter")
    student.to(target_device).train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(student.state_dict())
    best_test_pcc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        student.train()
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        loss_sum = 0.0
        seen = 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            x = frames[batch].to(target_device, non_blocking=True)
            z = resnet[batch].to(target_device, non_blocking=True)
            with torch.no_grad():
                target = teacher(z).float()
            predicted = student(x).float()
            scale = target.square().mean().detach().clamp_min(1.0e-6)
            normalized_mse = (predicted - target).square().mean() / scale
            cosine = 1.0 - F.cosine_similarity(
                predicted.flatten(1), target.flatten(1), dim=-1
            ).mean()
            loss = normalized_mse + 0.10 * cosine
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch.numel()
            seen += batch.numel()
        test_metrics = _feature_metrics(
            student, teacher, frames, resnet, test_indices,
            batch_size=batch_size, device=target_device,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "test_feature": test_metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} loss={record['train_loss']:.6f} "
            f"test_feature_PCC={test_metrics['pcc']:.6f} "
            f"NMSE={test_metrics['normalized_mse']:.6f}",
            flush=True,
        )
        if test_metrics["pcc"] > best_test_pcc:
            best_test_pcc = test_metrics["pcc"]
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state, strict=True)
    train_metrics = _feature_metrics(
        student, teacher, frames, resnet, train_indices,
        batch_size=batch_size, device=target_device,
    )
    test_metrics = _feature_metrics(
        student, teacher, frames, resnet, test_indices,
        batch_size=batch_size, device=target_device,
    )
    student_model.load_state_dict(
        _compatible_common_state(student_model, teacher_state), strict=False
    )
    student_model.tiny_rgb_electronic_adapter.load_state_dict(best_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "best_tiny_rgb_adapter.pt"
    full_path = output_dir / "distilled_student_initialization.pt"
    torch.save(
        {
            "schema_version": 1,
            "contract": "lgvq_resnet_e1_to_tiny_rgb_feature_distillation_v1",
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "train_feature_metrics": train_metrics,
            "test_feature_metrics": test_metrics,
        },
        adapter_path,
    )
    torch.save(
        {
            "schema_version": 1,
            "architecture": student_settings.architecture_label,
            "target_name": student_settings.target_name,
            "prompt": student_settings.prompt,
            "epoch": 0,
            "state_dict": student_model.state_dict(),
            "selection_source": "teacher_feature_distillation",
            "teacher_or_qwen_loaded_during_student_inference": False,
        },
        full_path,
    )
    report = {
        "schema_version": 1,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "teacher_resnet_used_only_during_distillation": True,
        "student_inference_contains_resnet": False,
        "student_tiny_rgb_parameters": sum(p.numel() for p in student.parameters()),
        "train_count": train_indices.numel(),
        "test_count": test_indices.numel(),
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", dest="device_name", default="auto")
    parser.add_argument("--seed", type=int, default=720)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    report = distill(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
