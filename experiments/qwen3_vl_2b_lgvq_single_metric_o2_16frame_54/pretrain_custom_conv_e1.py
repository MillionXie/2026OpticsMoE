"""Pretrain the project-owned Conv E1 correction, then emit a full warm start.

The previous lightweight E1 correction is a training-only feature teacher.
The emitted checkpoint contains neither that teacher front nor its cached
features.  Formal inference uses only elementary project-owned Conv/Norm/GELU/
Linear layers followed by the unchanged optical/electronic predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import read_manifest
from .modeling import CustomConvE1Correction, build_model
from .settings import load_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state_dict mapping: {path}")
    return state


class _PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frames: torch.Tensor,
        teacher_tokens: torch.Tensor,
        indices: list[int],
    ) -> None:
        self.frames = frames
        self.teacher_tokens = teacher_tokens
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.indices[index]
        return self.frames[source], self.teacher_tokens[source]


@torch.inference_mode()
def _evaluate(
    student: CustomConvE1Correction,
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    squared_error = squared_target = count = 0.0
    for frames, tokens in loader:
        frames = frames.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        target = teacher(tokens).float()
        prediction = student(frames).float()
        squared_error += float((prediction - target).square().sum())
        squared_target += float(target.square().sum())
        count += float(target.numel())
        sum_x += float(prediction.sum())
        sum_y += float(target.sum())
        sum_x2 += float(prediction.square().sum())
        sum_y2 += float(target.square().sum())
        sum_xy += float((prediction * target).sum())
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = max(sum_x2 - sum_x * sum_x / count, 1.0e-12)
    variance_y = max(sum_y2 - sum_y * sum_y / count, 1.0e-12)
    return {
        "normalized_mse": squared_error / max(squared_target, 1.0e-12),
        "pcc": covariance / (variance_x * variance_y) ** 0.5,
        "rmse": (squared_error / count) ** 0.5,
    }


def run(
    *,
    teacher_config: Path,
    student_config: Path,
    teacher_checkpoint: Path,
    teacher_feature_cache: Path,
    raw_frame_cache: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")

    teacher_settings = load_settings(teacher_config)
    student_settings = load_settings(student_config)
    if not student_settings.custom_conv_electronic_enabled:
        raise ValueError("student config must enable the custom Conv E1 correction")

    raw = torch.load(
        raw_frame_cache.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    cached = torch.load(
        teacher_feature_cache.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    frames = raw.get("frames")
    tokens = cached.get("tokens")
    raw_ids = list(map(str, raw.get("sample_ids", [])))
    teacher_ids = list(map(str, cached.get("sample_ids", [])))
    if not torch.is_tensor(frames) or tuple(frames.shape[1:]) != (4, 3, 224, 224):
        raise ValueError("raw frame cache must contain [N,4,3,224,224]")
    if not torch.is_tensor(tokens) or tuple(tokens.shape[1:]) != (4, 196, 96):
        raise ValueError("teacher cache must contain [N,4,196,96]")
    if set(raw_ids) != set(teacher_ids) or len(raw_ids) != len(set(raw_ids)):
        raise RuntimeError("raw-frame and teacher-cache sample IDs differ")
    if raw_ids != teacher_ids:
        lookup = {sample_id: index for index, sample_id in enumerate(teacher_ids)}
        order = torch.tensor([lookup[sample_id] for sample_id in raw_ids])
        tokens = tokens.index_select(0, order)

    rows = read_manifest(student_settings.manifest_path)
    split_by_id = {row.sample_id: row.split for row in rows}
    train_indices = [
        index for index, sample_id in enumerate(raw_ids)
        if split_by_id[sample_id] == "train"
    ]
    test_indices = [
        index for index, sample_id in enumerate(raw_ids)
        if split_by_id[sample_id] == "test"
    ]

    teacher_model = build_model(teacher_settings)
    teacher_model.load_state_dict(_checkpoint_state(teacher_checkpoint), strict=False)
    teacher = teacher_model.mobilenet_electronic_correction
    if teacher is None:
        raise RuntimeError("Teacher configuration has no lightweight E1 correction")
    teacher.to(target_device).eval().requires_grad_(False)
    del teacher_model

    student = CustomConvE1Correction(student_settings).to(target_device)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != 316_568:
        raise RuntimeError(f"Custom Conv parameter contract changed: {parameter_count}")
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    train_loader = DataLoader(
        _PairDataset(frames, tokens, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        _PairDataset(frames, tokens, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    best_pcc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        student.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_frames, batch_tokens in train_loader:
            batch_frames = batch_frames.to(target_device, non_blocking=True)
            batch_tokens = batch_tokens.to(target_device, non_blocking=True)
            with torch.no_grad():
                target = teacher(batch_tokens).float()
            prediction = student(batch_frames).float()
            target_power = target.square().mean().clamp_min(1.0e-6)
            loss = F.mse_loss(prediction, target) / target_power
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_frames.shape[0]
            sample_count += batch_frames.shape[0]
        metrics = _evaluate(student, teacher, test_loader, target_device)
        record = {
            "epoch": epoch,
            "train_normalized_mse": loss_sum / sample_count,
            "test": metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} train_nmse={record['train_normalized_mse']:.6f} "
            f"test_pcc={metrics['pcc']:.5f} test_nmse={metrics['normalized_mse']:.6f}",
            flush=True,
        )
        if metrics["pcc"] > best_pcc:
            best_pcc = metrics["pcc"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Pretraining produced no checkpoint")

    full_student = build_model(student_settings)
    source = _checkpoint_state(teacher_checkpoint)
    destination = full_student.state_dict()
    compatible = {
        name: value for name, value in source.items()
        if name in destination and tuple(value.shape) == tuple(destination[name].shape)
    }
    full_student.load_state_dict(compatible, strict=False)
    if full_student.custom_conv_electronic_correction is None:
        raise RuntimeError("Full student did not construct the custom Conv correction")
    full_student.custom_conv_electronic_correction.load_state_dict(best_state)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": full_student.state_dict(),
        "epoch": 0,
        "pretraining": {
            "best_epoch": best_epoch,
            "teacher_used_during_training_only": True,
            "teacher_absent_from_inference": True,
            "custom_conv_parameters": parameter_count,
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "architecture": "project_owned_conv_norm_gelu_linear_e1_v1",
        "custom_conv_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_test_pcc": best_pcc,
        "best_test_metrics": history[best_epoch - 1]["test"],
        "history": history,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "teacher_used_during_training_only": True,
        "teacher_absent_from_inference": True,
        "full_warm_start": str(output),
        "full_warm_start_sha256": _sha256(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--student-config", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-feature-cache", required=True, type=Path)
    parser.add_argument("--raw-frame-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=746)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
