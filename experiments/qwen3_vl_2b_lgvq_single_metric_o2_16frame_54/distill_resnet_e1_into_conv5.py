"""Absorb the rejected ResNet E1 correction into the small target Conv5.

The emitted student contains the existing 339k five-convolution input head,
the optical/electronic network and one MOS readout.  The pretrained ResNet is
used only to define an offline feature target and is absent at inference.
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
from torch import nn

from .data import (
    load_quality_feature_cache,
    load_raw_frame_cache,
    load_resnet_feature_cache,
    read_manifest,
)
from .modeling import (
    FrozenResNetElectronicCorrection,
    TrainableQualityFrameStem,
    build_model,
)
from .settings import load_settings


def _load(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint has no state_dict mapping")
    return state


def _prefixed(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    result = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not result:
        raise RuntimeError(f"No tensors with prefix {prefix!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_state(
    model: nn.Module, source: Mapping[str, torch.Tensor]
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


@torch.no_grad()
def _metrics(
    stem: nn.Module,
    norm: nn.Module,
    teacher: nn.Module,
    frames: torch.Tensor,
    old_quality: torch.Tensor,
    resnet: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    stem.eval()
    dot = student_square = teacher_square = 0.0
    delta_error = delta_target_square = 0.0
    full_error = full_target_square = 0.0
    for start in range(0, indices.numel(), batch_size):
        batch = indices[start : start + batch_size]
        x = frames[batch].to(device, non_blocking=True)
        q = old_quality[batch].to(device, non_blocking=True).float()
        z = resnet[batch].to(device, non_blocking=True)
        base = scale * norm(q)
        target_delta = teacher(z).float()
        predicted = scale * norm(stem(x).to(torch.float16).float())
        predicted_delta = predicted - base
        target = base + target_delta
        delta_error += float((predicted_delta - target_delta).square().sum())
        delta_target_square += float(target_delta.square().sum())
        full_error += float((predicted - target).square().sum())
        full_target_square += float(target.square().sum())
        left = predicted_delta.flatten()
        right = target_delta.flatten()
        left = left - left.mean()
        right = right - right.mean()
        dot += float((left * right).sum())
        student_square += float(left.square().sum())
        teacher_square += float(right.square().sum())
    return {
        "correction_pcc": dot
        / math.sqrt(max(student_square * teacher_square, 1.0e-24)),
        "correction_normalized_mse": delta_error
        / max(delta_target_square, 1.0e-12),
        "full_injection_normalized_mse": full_error
        / max(full_target_square, 1.0e-12),
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
        raise ValueError("Teacher config must contain the ResNet feature cache")
    if teacher_settings.quality_feature_cache_path is None:
        raise ValueError("Teacher config must contain the old Conv5 quality cache")
    if not student_settings.trainable_frame_stem_enabled:
        raise ValueError("Student config must enable the trainable Conv5 stem")
    if student_settings.raw_frame_cache_path is None:
        raise ValueError("Student config must contain the raw frame cache")
    if student_settings.frame_stem_checkpoint is None:
        raise ValueError("Student config must identify the source Conv5 checkpoint")
    rows = read_manifest(student_settings.manifest_path)
    sample_ids = [row.sample_id for row in rows]
    frame_count, grid = student_settings.frame_count, student_settings.token_grid
    frames = load_raw_frame_cache(
        student_settings.raw_frame_cache_path,
        sample_ids=sample_ids,
        frame_count=frame_count,
    )["frames"]
    old_quality = load_quality_feature_cache(
        teacher_settings.quality_feature_cache_path,
        sample_ids=sample_ids,
        frame_count=frame_count,
        token_grid=grid,
        width=student_settings.model_width,
    )["quality_tokens"]
    resnet = load_resnet_feature_cache(
        teacher_settings.resnet_feature_cache_path,
        sample_ids=sample_ids,
        frame_count=frame_count,
        token_grid=grid,
    )["tokens"]
    train_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row.split == "train"],
        dtype=torch.long,
    )
    test_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row.split == "test"],
        dtype=torch.long,
    )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    teacher_payload = _load(teacher_checkpoint)
    teacher_state = _state(teacher_payload)
    teacher = FrozenResNetElectronicCorrection(teacher_settings)
    teacher.load_state_dict(
        _prefixed(teacher_state, "resnet_electronic_correction."), strict=True
    )
    teacher.to(device).eval().requires_grad_(False)
    norm = nn.LayerNorm(student_settings.model_width)
    norm.load_state_dict(
        {
            "weight": teacher_state["electronic_quality_norm.weight"],
            "bias": teacher_state["electronic_quality_norm.bias"],
        },
        strict=True,
    )
    norm.to(device).eval().requires_grad_(False)
    scale = torch.sigmoid(
        teacher_state["raw_electronic_quality_scale"].float()
    ).to(device)
    source_payload = _load(student_settings.frame_stem_checkpoint)
    stem = TrainableQualityFrameStem(student_settings.frame_stem_refiner_depth)
    source_result = stem.load_state_dict(
        _prefixed(_state(source_payload), "frame_stem."), strict=False
    )
    unexpected = list(source_result.unexpected_keys)
    non_refiner_missing = [
        name for name in source_result.missing_keys if not name.startswith("refiners.")
    ]
    if unexpected or non_refiner_missing:
        raise RuntimeError(
            "Base Conv5 checkpoint is incompatible with the compact student: "
            f"missing={non_refiner_missing}, unexpected={unexpected}"
        )
    stem.to(device)
    optimizer = torch.optim.AdamW(
        stem.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(stem.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        stem.train()
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        loss_sum = 0.0
        seen = 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            x = frames[batch].to(device, non_blocking=True)
            q = old_quality[batch].to(device, non_blocking=True).float()
            z = resnet[batch].to(device, non_blocking=True)
            with torch.no_grad():
                base = scale * norm(q)
                target_delta = teacher(z).float()
            predicted = scale * norm(stem(x).to(torch.float16).float())
            predicted_delta = predicted - base
            target_scale = target_delta.square().mean().detach().clamp_min(1.0e-6)
            nmse = (predicted_delta - target_delta).square().mean() / target_scale
            cosine = 1.0 - F.cosine_similarity(
                predicted_delta.flatten(1), target_delta.flatten(1), dim=-1
            ).mean()
            loss = nmse + 0.10 * cosine
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stem.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch.numel()
            seen += batch.numel()
        test_metrics = _metrics(
            stem, norm, teacher, frames, old_quality, resnet, test_indices,
            scale=scale, batch_size=batch_size, device=device,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "test_feature": test_metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} loss={record['train_loss']:.6f} "
            f"correction_PCC={test_metrics['correction_pcc']:.6f} "
            f"correction_NMSE={test_metrics['correction_normalized_mse']:.6f} "
            f"full_NMSE={test_metrics['full_injection_normalized_mse']:.6f}",
            flush=True,
        )
        score = test_metrics["full_injection_normalized_mse"]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(stem.state_dict())
    stem.load_state_dict(best_state, strict=True)
    train_metrics = _metrics(
        stem, norm, teacher, frames, old_quality, resnet, train_indices,
        scale=scale, batch_size=batch_size, device=device,
    )
    test_metrics = _metrics(
        stem, norm, teacher, frames, old_quality, resnet, test_indices,
        scale=scale, batch_size=batch_size, device=device,
    )
    student_model = build_model(student_settings)
    student_model.load_state_dict(_common_state(student_model, teacher_state), strict=False)
    if student_model.frame_stem is None:
        raise RuntimeError("Student model did not create its Conv5 stem")
    student_model.frame_stem.load_state_dict(best_state, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_path = output_dir / "best_distilled_conv5.pt"
    init_path = output_dir / "distilled_student_initialization.pt"
    torch.save(
        {
            "schema_version": 1,
            "contract": "lgvq_resnet_e1_absorbed_into_conv5_v1",
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "train_feature_metrics": train_metrics,
            "test_feature_metrics": test_metrics,
        },
        stem_path,
    )
    torch.save(
        {
            "schema_version": 1,
            "architecture": student_settings.architecture_label,
            "target_name": student_settings.target_name,
            "prompt": student_settings.prompt,
            "epoch": 0,
            "state_dict": student_model.state_dict(),
            "selection_source": "resnet_e1_absorbed_into_conv5",
            "teacher_or_qwen_loaded_during_student_inference": False,
        },
        init_path,
    )
    report = {
        "schema_version": 1,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        "teacher_resnet_used_only_during_distillation": True,
        "student_inference_contains_resnet": False,
        "student_conv5_parameters": sum(p.numel() for p in stem.parameters()),
        "student_refiner_depth": student_settings.frame_stem_refiner_depth,
        "best_epoch": best_epoch,
        "train_feature_metrics": train_metrics,
        "test_feature_metrics": test_metrics,
        "conv5_checkpoint": str(stem_path.resolve()),
        "conv5_checkpoint_sha256": _sha256(stem_path),
        "student_initialization": str(init_path.resolve()),
        "student_initialization_sha256": _sha256(init_path),
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", dest="device_name", default="auto")
    parser.add_argument("--seed", type=int, default=721)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    report = distill(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
