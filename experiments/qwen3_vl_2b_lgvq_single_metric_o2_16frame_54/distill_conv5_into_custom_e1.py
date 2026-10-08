"""Fold the frozen Conv5 E1 contribution into the existing raw-RGB Conv E1.

The teacher uses two additive E1 corrections::

    custom_conv(raw_rgb) + sigmoid(scale) * LayerNorm(conv5_cache)

The emitted student has no Conv5 input, cache, module, or parameter.  Its sole
project-owned raw-RGB Conv E1 is trained to reproduce the teacher's combined
correction.  The optical path and every other compatible student parameter are
copied exactly and can subsequently be jointly fine-tuned on MOS.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import load_single_metric_cache
from .modeling import CustomConvE1Correction, build_model
from .settings import load_settings, resolved_dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Pairs(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frames: torch.Tensor,
        quality: torch.Tensor,
        indices: list[int],
    ) -> None:
        self.frames = frames
        self.quality = quality
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.indices[index]
        return self.frames[source], self.quality[source]


@torch.no_grad()
def _metrics(
    student: CustomConvE1Correction,
    teacher_custom: CustomConvE1Correction,
    teacher_norm: torch.nn.Module,
    teacher_quality_scale: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    error2 = target2 = sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = count = 0.0
    for frames, quality in loader:
        frames = frames.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True).float()
        target = teacher_custom(frames).float() + teacher_quality_scale * teacher_norm(
            quality
        ).float()
        prediction = student(frames).float()
        difference = prediction - target
        error2 += float(difference.square().sum())
        target2 += float(target.square().sum())
        sum_x += float(prediction.sum())
        sum_y += float(target.sum())
        sum_x2 += float(prediction.square().sum())
        sum_y2 += float(target.square().sum())
        sum_xy += float((prediction * target).sum())
        count += float(target.numel())
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = max(sum_x2 - sum_x * sum_x / count, 1.0e-12)
    variance_y = max(sum_y2 - sum_y * sum_y / count, 1.0e-12)
    return {
        "normalized_mse": error2 / max(target2, 1.0e-12),
        "pcc": covariance / (variance_x * variance_y) ** 0.5,
        "rmse": (error2 / count) ** 0.5,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher_settings = load_settings(args.teacher_config)
    student_settings = load_settings(args.student_config)
    if not teacher_settings.electronic_quality_residual_enabled:
        raise ValueError("Teacher must enable the Conv5 E1 residual")
    if student_settings.electronic_quality_residual_enabled:
        raise ValueError("Student must disable the Conv5 E1 residual")
    if not student_settings.custom_conv_electronic_enabled:
        raise ValueError("Student must retain the sole raw-RGB Conv E1")

    cached = load_single_metric_cache(teacher_settings)
    frames = cached.get("raw_frames")
    quality = cached.get("quality_tokens")
    splits = list(map(str, cached["splits"]))
    if not torch.is_tensor(frames) or tuple(frames.shape[1:]) != (4, 3, 224, 224):
        raise ValueError("Teacher data must contain raw frames [N,4,3,224,224]")
    if not torch.is_tensor(quality) or tuple(quality.shape[1:]) != (4, 196, 192):
        raise ValueError("Teacher data must contain Conv5 tokens [N,4,196,192]")

    teacher_model = build_model(teacher_settings)
    saved = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_model.load_state_dict(saved["state_dict"], strict=True)
    if (
        teacher_model.custom_conv_electronic_correction is None
        or teacher_model.electronic_quality_norm is None
    ):
        raise RuntimeError("Teacher E1 components are absent")
    teacher_custom = copy.deepcopy(
        teacher_model.custom_conv_electronic_correction
    ).to(device).eval().requires_grad_(False)
    teacher_norm = copy.deepcopy(
        teacher_model.electronic_quality_norm
    ).to(device).eval().requires_grad_(False)
    teacher_quality_scale = torch.sigmoid(
        teacher_model.raw_electronic_quality_scale.detach()
    ).to(device)

    student_model = build_model(student_settings)
    destination = student_model.state_dict()
    compatible = {
        name: value
        for name, value in saved["state_dict"].items()
        if name in destination and tuple(value.shape) == tuple(destination[name].shape)
    }
    load_result = student_model.load_state_dict(compatible, strict=False)
    if student_model.custom_conv_electronic_correction is None:
        raise RuntimeError("Student raw-RGB Conv E1 is absent")
    student = student_model.custom_conv_electronic_correction.to(device)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != 316_568:
        raise RuntimeError(f"Custom Conv parameter contract changed: {parameter_count}")
    del teacher_model

    train_indices = [i for i, split in enumerate(splits) if split == "train"]
    test_indices = [i for i, split in enumerate(splits) if split == "test"]
    train_loader = DataLoader(
        _Pairs(frames, quality, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        _Pairs(frames, quality, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        samples = 0
        for batch_frames, batch_quality in train_loader:
            batch_frames = batch_frames.to(device, non_blocking=True)
            batch_quality = batch_quality.to(device, non_blocking=True).float()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                target = teacher_custom(batch_frames).float() + (
                    teacher_quality_scale * teacher_norm(batch_quality).float()
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                prediction = student(batch_frames).float()
                target_power = target.square().mean().clamp_min(1.0e-6)
                mse = F.mse_loss(prediction, target) / target_power
                cosine = 1.0 - F.cosine_similarity(
                    prediction.flatten(1), target.flatten(1), dim=1
                ).mean()
                loss = mse + args.cosine_weight * cosine
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * batch_frames.shape[0]
            samples += batch_frames.shape[0]
        metrics = _metrics(
            student,
            teacher_custom,
            teacher_norm,
            teacher_quality_scale,
            test_loader,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, samples),
            "test": metrics,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train={row['train_loss']:.6f} "
            f"test_nmse={metrics['normalized_mse']:.6f} pcc={metrics['pcc']:.6f}",
            flush=True,
        )
        if metrics["normalized_mse"] < best_score:
            best_score = metrics["normalized_mse"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Distillation produced no state")
    student_model.custom_conv_electronic_correction.load_state_dict(
        best_state, strict=True
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "architecture": student_settings.architecture_label,
        "target_name": student_settings.target_name,
        "prompt": student_settings.prompt,
        "epoch": 0,
        "state_dict": student_model.state_dict(),
        "settings": resolved_dict(student_settings),
        "selection_policy": "lowest held-out combined-E1 normalized MSE",
        "teacher_used_during_training_only": True,
        "conv5_present_during_student_inference": False,
        "distillation": {
            "teacher_quality_scale": float(teacher_quality_scale.cpu()),
            "custom_conv_parameters": parameter_count,
            "best_epoch": best_epoch,
            "best_test": history[best_epoch - 1]["test"],
            "compatible_teacher_tensors": len(compatible),
            "student_missing_after_compatible_load": list(load_result.missing_keys),
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "output": str(output),
        "output_sha256": _sha256(output),
        "student_architecture": student_settings.architecture_label,
        "custom_conv_parameters": parameter_count,
        "removed_conv5_parameters": 339_312,
        "conv5_present_during_student_inference": False,
        "best_epoch": best_epoch,
        "best_test": history[best_epoch - 1]["test"],
        "history": history,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--student-config", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1401)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
