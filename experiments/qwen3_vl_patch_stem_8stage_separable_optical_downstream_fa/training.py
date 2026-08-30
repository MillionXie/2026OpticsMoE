from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.objectives import (
    SegmentationAccumulator,
    segmentation_loss,
)
from experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation.metrics import (
    ISICSegmentationAccumulator,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import (
    masked_coordinate_loss,
    masked_heatmap_mse,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import (
    PoseMetricAccumulator,
)

from .datasets import DownstreamDataConfig, build_loaders, prepare_bundle
from .model import P11DownstreamModel
from .settings import Settings, implementation_sha256


CHECKPOINT_FORMAT = "p12-p11-downstream-fixed-feedback-v1"
RESULT_FORMAT = "p12-p11-downstream-result-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def save_torch(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def build_model(settings: Settings) -> P11DownstreamModel:
    model = P11DownstreamModel(
        stem_checkpoint=settings.paths.stem_checkpoint,
        source_checkpoint=settings.paths.source_backbone,
        p11_config=settings.p11_config,
        task=settings.task,
        num_outputs=settings.task_settings.num_outputs,
        global_hidden_dim=settings.task_settings.head_hidden_dim or 256,
        dense_output_size=settings.task_settings.output_size,
        decoder_width=settings.task_settings.decoder_width or 64,
    )
    report = model.parameter_report()
    if model.source_manifest["sha256"] != settings.paths.source_backbone_sha256:
        raise RuntimeError(
            "P11 source checkpoint SHA-256 differs from the preregistered source"
        )
    if report["optical_phase_parameters"] != settings.model.expected_optical_parameters:
        raise RuntimeError("P11 optical parameter count differs from the locked P12 recipe")
    if (
        report["optical_fraction_of_reusable_backbone"]
        < settings.model.minimum_optical_parameter_fraction
    ):
        raise RuntimeError("P12 reusable backbone fell below the optical parameter fraction")
    if report["minimum_optical_gate"] < settings.model.optical_gate_min - 1.0e-7:
        raise RuntimeError("P12 source contains an optical gate below the locked minimum")
    return model


def build_data(settings: Settings):
    config = DownstreamDataConfig(
        task=settings.task,
        data_root=settings.data_root,
        output_dir=settings.output_dir,
        image_size=settings.model.canvas_size,
        random_seed=settings.seed,
        train_limit=settings.limits.max_train_samples,
        val_limit=settings.limits.max_validation_samples,
        test_limit=settings.limits.max_test_samples,
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.evaluation_batch_size,
        num_workers=settings.num_workers,
        persistent_workers=settings.dataloader.persistent_workers,
        pin_memory=settings.dataloader.pin_memory,
        prefetch_factor=settings.dataloader.prefetch_factor,
        isic_val_fraction=settings.task_settings.validation_fraction,
        lsp_val_fraction=settings.task_settings.validation_fraction,
        augmentation_enabled=True,
        auto_download=False,
    )
    bundle = prepare_bundle(config)
    full_counts = bundle.metadata.get("full_counts", {})
    expected_counts = {
        "train": settings.task_settings.expected_train_samples,
        "test": settings.task_settings.expected_test_samples,
    }
    mismatches = {
        split: (full_counts.get(split), expected)
        for split, expected in expected_counts.items()
        if expected is not None and full_counts.get(split) != expected
    }
    if mismatches:
        raise RuntimeError(f"Formal dataset split count mismatch: {mismatches}")
    loaders = build_loaders(bundle, config)
    return config, bundle, loaders


def _trainable_groups(model: P11DownstreamModel, settings: Settings):
    optimizer_settings = settings.optimizer
    groups: list[dict[str, Any]] = []
    if settings.updates_backbone:
        phase = [parameter for parameter in model.phase_parameters() if parameter.requires_grad]
        adapter = [
            parameter
            for parameter in model.backbone.adapter_parameters()
            if parameter.requires_grad
        ]
        residual = [
            parameter
            for parameter in model.backbone.residual_parameters()
            if parameter.requires_grad
        ]
        groups.extend(
            [
                {
                    "params": phase,
                    "lr": optimizer_settings.phase_learning_rate,
                    "weight_decay": optimizer_settings.phase_weight_decay,
                    "group_name": "phase",
                },
                {
                    "params": adapter,
                    "lr": optimizer_settings.adapter_learning_rate,
                    "weight_decay": optimizer_settings.electronic_weight_decay,
                    "group_name": "adapter",
                },
                {
                    "params": residual,
                    "lr": optimizer_settings.residual_learning_rate,
                    "weight_decay": optimizer_settings.electronic_weight_decay,
                    "group_name": "residual",
                },
            ]
        )
    head = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    groups.append(
        {
            "params": head,
            "lr": optimizer_settings.head_learning_rate,
            "weight_decay": optimizer_settings.electronic_weight_decay,
            "group_name": "head",
        }
    )
    for group in groups:
        if not group["params"]:
            raise RuntimeError(f"Optimizer group {group['group_name']} is empty")
    return groups


def build_optimizer_scheduler(
    model: P11DownstreamModel,
    settings: Settings,
    steps_per_epoch: int,
):
    optimizer = torch.optim.AdamW(
        _trainable_groups(model, settings),
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
    )
    total_steps = max(int(settings.run_epochs) * int(steps_per_epoch), 1)
    warmup_steps = min(
        int(settings.optimizer.warmup_epochs) * int(steps_per_epoch), total_steps - 1
    )
    minimum = float(settings.optimizer.minimum_learning_rate_ratio)

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def task_loss(
    settings: Settings,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"]
    if settings.task == "caltech101":
        loss = F.cross_entropy(logits.float(), batch["label"], label_smoothing=0.10)
        return loss, {"cross_entropy": float(loss.detach())}
    if settings.task == "isic2016":
        loss, parts = segmentation_loss(
            logits,
            batch["mask"],
            bce_weight=1.0,
            dice_weight=1.0,
            soft_iou_weight=0.75,
            boundary_weight=0.25,
        )
        return loss, {key: float(value.detach()) for key, value in parts.items()}
    heatmap = masked_heatmap_mse(logits, batch["heatmaps"], batch["visible"])
    coordinate = masked_coordinate_loss(
        logits, batch["keypoints"], batch["visible"], image_size=224
    )
    loss = heatmap + 0.10 * coordinate
    return loss, {
        "heatmap_mse": float(heatmap.detach()),
        "coordinate_loss": float(coordinate.detach()),
    }


def _retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor | None = None,
    gallery_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute retrieval against a disjoint gallery, or leave-one-out if omitted."""

    leave_one_out = gallery_embeddings is None or gallery_labels is None
    queries = F.normalize(query_embeddings.float(), dim=-1)
    query_labels = query_labels.long()
    gallery = queries if leave_one_out else F.normalize(gallery_embeddings.float(), dim=-1)
    gallery_labels = query_labels if leave_one_out else gallery_labels.long()
    count = int(query_labels.numel())
    gallery_count = int(gallery_labels.numel())
    if count < 1 or gallery_count < 1 or (leave_one_out and count < 2):
        return {"retrieval_recall_at_1": 0.0, "retrieval_map": 0.0}
    recall_hits = 0
    average_precisions: list[float] = []
    # Chunking limits peak memory while preserving exact leave-one-out ranking.
    for begin in range(0, count, 256):
        end = min(begin + 256, count)
        similarity = queries[begin:end] @ gallery.T
        if leave_one_out:
            rows = torch.arange(end - begin)
            similarity[rows, torch.arange(begin, end)] = -torch.inf
        order = similarity.argsort(dim=1, descending=True)
        ranked_labels = gallery_labels[order]
        current_labels = query_labels[begin:end, None]
        relevant = ranked_labels.eq(current_labels)
        # The query itself was moved to the end with -inf similarity, but it
        # must also be excluded from the relevant set for true leave-one-out AP.
        if leave_one_out:
            relevant &= order.ne(torch.arange(begin, end)[:, None])
        recall_hits += int(relevant[:, 0].sum())
        positions = torch.arange(1, gallery_count + 1, dtype=torch.float32)[None, :]
        precision = relevant.cumsum(dim=1).float() / positions
        relevant_count = relevant.sum(dim=1).clamp_min(1)
        ap = (precision * relevant).sum(dim=1) / relevant_count
        average_precisions.extend(ap.tolist())
    return {
        "retrieval_recall_at_1": recall_hits / count,
        "retrieval_map": float(np.mean(average_precisions)),
    }


@torch.no_grad()
def _collect_caltech_embeddings(
    model: P11DownstreamModel,
    loader,
    settings: Settings,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if settings.training.use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp:
            output = model(batch["image"])
        embeddings.append(output["embedding"].detach().cpu())
        labels.append(batch["label"].detach().cpu())
    if not embeddings:
        raise RuntimeError("Caltech retrieval loader produced no samples")
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def evaluate(
    model: P11DownstreamModel,
    loader,
    settings: Settings,
    device: torch.device,
    *,
    ablation: str = "normal",
    max_batches: int | None = None,
    include_retrieval: bool = False,
) -> dict[str, Any]:
    model.eval()
    samples = 0
    loss_sum = 0.0
    if settings.task == "caltech101":
        correct = 0
        class_correct = torch.zeros(101, dtype=torch.long)
        class_total = torch.zeros(101, dtype=torch.long)
        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
    elif settings.task == "isic2016":
        accumulator = ISICSegmentationAccumulator()
    else:
        pose = PoseMetricAccumulator()

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if settings.training.use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp:
            output = model(batch["image"], ablation=ablation)
            loss, _ = task_loss(settings, output, batch)
        batch_size = int(batch["image"].shape[0])
        samples += batch_size
        loss_sum += float(loss) * batch_size
        if settings.task == "caltech101":
            prediction = output["logits"].argmax(dim=-1)
            labels = batch["label"]
            hits = prediction.eq(labels)
            correct += int(hits.sum())
            class_total.scatter_add_(0, labels.cpu(), torch.ones_like(labels.cpu()))
            class_correct.scatter_add_(0, labels.cpu(), hits.long().cpu())
            if include_retrieval:
                all_embeddings.append(output["embedding"].detach().cpu())
                all_labels.append(labels.detach().cpu())
        elif settings.task == "isic2016":
            accumulator.update(output["logits"], batch["mask"], loss=loss)
        else:
            pose.update(
                output["logits"],
                batch["keypoints"],
                batch["visible"],
                batch["torso_scale"],
                batch["head_scale"],
                image_size=224,
            )

    if not samples:
        raise RuntimeError("Evaluation loader produced no samples")
    if settings.task == "caltech101":
        present = class_total > 0
        metrics: dict[str, Any] = {
            "loss": loss_sum / samples,
            "top1": correct / samples,
            "balanced_accuracy": float(
                (class_correct[present].float() / class_total[present]).mean()
            ),
            "samples": samples,
        }
        if include_retrieval:
            metrics.update(
                _retrieval_metrics(torch.cat(all_embeddings), torch.cat(all_labels))
            )
        return metrics
    if settings.task == "isic2016":
        return accumulator.compute()
    metrics = pose.compute()
    metrics["loss"] = loss_sum / samples
    metrics["samples"] = samples
    # Match the preregistered settings key while retaining the verbose metric.
    metrics["pck_torso_0p2"] = metrics["pck_at_0.2_torso"]
    return metrics


def _feedback_method(settings: Settings) -> str:
    return {
        "noft": "noft",
        "bp": "bp_current",
        "fa_pretrained": "fa_pretrained",
        "fa_random": "fa_random",
    }[settings.method]


def configure_runtime_feedback(model: P11DownstreamModel, settings: Settings) -> dict[str, Any]:
    random_seed = int(settings.seed) + 8_000_003
    model.configure_feedback(_feedback_method(settings), random_seed=random_seed)
    return model.feedback_manifest()


def _gradient_list(model: P11DownstreamModel) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for stage, parameter in enumerate(model.phase_parameters(), start=1):
        if parameter.grad is None:
            raise RuntimeError(f"Missing phase gradient at stage {stage}")
        gradient = parameter.grad.detach().float().cpu().clone()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"Non-finite phase gradient at stage {stage}")
        values.append(gradient)
    return values


def _gradient_comparison(reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]):
    if len(reference) != len(candidate):
        raise ValueError("Gradient stage counts differ")
    rows = []
    for stage, (exact, approximate) in enumerate(zip(reference, candidate, strict=True), start=1):
        exact_flat = exact.flatten().double()
        approximate_flat = approximate.flatten().double()
        exact_norm = float(exact_flat.norm())
        approximate_norm = float(approximate_flat.norm())
        cosine = float(
            F.cosine_similarity(exact_flat, approximate_flat, dim=0, eps=1.0e-20)
        )
        rows.append(
            {
                "stage": stage,
                "cosine_to_bp_current": cosine,
                "norm_ratio_to_bp_current": approximate_norm / max(exact_norm, 1.0e-20),
                "bp_current_norm": exact_norm,
                "candidate_norm": approximate_norm,
            }
        )
    return rows


def _parameter_gradient_report(parameters: Iterable[nn.Parameter], *, name: str) -> dict[str, Any]:
    values = list(parameters)
    missing = sum(parameter.grad is None for parameter in values)
    nonfinite = sum(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in values
    )
    squared_norm = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in values
        if parameter.grad is not None
    )
    report = {
        "parameter_tensors": len(values),
        "missing_gradient_tensors": missing,
        "nonfinite_gradient_tensors": nonfinite,
        "zero_gradient_tensors": sum(
            parameter.grad is not None and not bool(parameter.grad.detach().ne(0).any())
            for parameter in values
        ),
        "total_l2_norm": math.sqrt(squared_norm),
    }
    if not values or missing or nonfinite or report["total_l2_norm"] <= 0.0:
        raise RuntimeError(f"Invalid {name} gradient group: {report}")
    return report


def gradient_diagnostic(
    model: P11DownstreamModel,
    raw_batch: Mapping[str, Any],
    settings: Settings,
    device: torch.device,
) -> dict[str, Any] | None:
    """Compare the configured connector with exact current BP on one eval batch."""

    if settings.method == "noft":
        return None
    was_training = model.training
    model.eval()
    batch = _move_batch(raw_batch, device)
    # Two samples are enough for connector geometry and avoid wasting a large
    # dense-task batch on this diagnostic.
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            batch[key] = value[:2]
        elif isinstance(value, list):
            batch[key] = value[:2]
    rng = capture_rng_state()

    def compute(method: str) -> tuple[list[torch.Tensor], dict[str, Any]]:
        restore_rng_state(rng)
        model.zero_grad(set_to_none=True)
        model.configure_feedback(method, random_seed=int(settings.seed) + 8_000_003)
        output = model(batch["image"])
        loss, _ = task_loss(settings, output, batch)
        loss.backward()
        return _gradient_list(model), {
            "adapter": _parameter_gradient_report(model.adapter_parameters(), name="adapter"),
            "residual": _parameter_gradient_report(model.residual_parameters(), name="residual"),
            "task_head": _parameter_gradient_report(model.head_parameters(), name="task_head"),
        }

    exact, _ = compute("bp_current")
    candidate, group_report = compute(_feedback_method(settings))
    rows = _gradient_comparison(exact, candidate)
    model.zero_grad(set_to_none=True)
    configure_runtime_feedback(model, settings)
    restore_rng_state(rng)
    model.train(was_training)
    if settings.method == "fa_pretrained" and model.phase_report()["mean_absolute_rad"] < 1.0e-7:
        minimum = min(float(row["cosine_to_bp_current"]) for row in rows)
        if minimum < 0.999:
            raise RuntimeError(
                "FA-pretrained must match BP-current at the unmodified common start; "
                f"minimum phase-gradient cosine was {minimum:.6f}"
            )
    return {
        "method": settings.method,
        "phase_motion_from_source": model.phase_report(),
        "per_stage": rows,
        "mean_cosine_stages_1_to_7": float(
            np.mean([row["cosine_to_bp_current"] for row in rows[:-1]])
        ),
        "last_stage_expected_exact_local_gradient": True,
        "trainable_gradient_groups": group_report,
    }


def _primary_metric(settings: Settings, metrics: Mapping[str, Any]) -> float:
    key = settings.task_settings.primary_metric
    if key not in metrics:
        raise RuntimeError(f"Primary metric {key!r} is missing from {sorted(metrics)}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Primary metric {key} is non-finite")
    return value


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _checkpoint_payload(
    *,
    model: P11DownstreamModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    settings: Settings,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    common_start_sha256: str | None,
    config_digest: str,
    implementation_digest: str,
    frozen_stem_digest: str,
    loader_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "epoch": int(epoch),
        "run_epochs": settings.run_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "history": list(history),
        "dataset_manifest_sha256": manifest_sha256,
        "source_checkpoint_sha256": model.source_manifest["sha256"],
        "source_phase_sha256": model.parameter_report()["source_phase_sha256"],
        "common_start_sha256": common_start_sha256,
        "feedback": model.feedback_manifest(),
        "config_digest": config_digest,
        "implementation_sha256": implementation_digest,
        "frozen_stem_state_sha256": frozen_stem_digest,
        "git_commit": git_commit(settings.repo_root),
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator_state,
        "phase_report": model.phase_report(),
        "saved_at_unix": time.time(),
    }


def _validate_checkpoint_identity(
    payload: Mapping[str, Any],
    settings: Settings,
    *,
    manifest_sha256: str,
    config_digest: str | None = None,
    implementation_digest: str | None = None,
    source_checkpoint_sha256: str | None = None,
    common_start_sha256: str | None = None,
    source_phase_sha256: str | None = None,
    feedback_manifest: Mapping[str, Any] | None = None,
    frozen_stem_digest: str | None = None,
) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported P12 checkpoint format: {payload.get('format')!r}")
    expected = {
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "dataset_manifest_sha256": manifest_sha256,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if config_digest is not None and payload.get("config_digest") != config_digest:
        mismatches["config_digest"] = (payload.get("config_digest"), config_digest)
    if (
        implementation_digest is not None
        and payload.get("implementation_sha256") != implementation_digest
    ):
        mismatches["implementation_sha256"] = (
            payload.get("implementation_sha256"),
            implementation_digest,
        )
    if (
        source_checkpoint_sha256 is not None
        and payload.get("source_checkpoint_sha256") != source_checkpoint_sha256
    ):
        mismatches["source_checkpoint_sha256"] = (
            payload.get("source_checkpoint_sha256"),
            source_checkpoint_sha256,
        )
    if payload.get("common_start_sha256") != common_start_sha256:
        mismatches["common_start_sha256"] = (
            payload.get("common_start_sha256"),
            common_start_sha256,
        )
    if (
        source_phase_sha256 is not None
        and payload.get("source_phase_sha256") != source_phase_sha256
    ):
        mismatches["source_phase_sha256"] = (
            payload.get("source_phase_sha256"),
            source_phase_sha256,
        )
    if feedback_manifest is not None and payload.get("feedback") != dict(
        feedback_manifest
    ):
        mismatches["feedback"] = (payload.get("feedback"), dict(feedback_manifest))
    if (
        frozen_stem_digest is not None
        and payload.get("frozen_stem_state_sha256") != frozen_stem_digest
    ):
        mismatches["frozen_stem_state_sha256"] = (
            payload.get("frozen_stem_state_sha256"),
            frozen_stem_digest,
        )
    if mismatches:
        raise RuntimeError(f"Checkpoint identity mismatch: {mismatches}")


def _load_common_start(
    model: P11DownstreamModel,
    settings: Settings,
    manifest_sha256: str,
    implementation_digest: str,
    *,
    require_completed_noft: bool = True,
) -> str | None:
    if settings.method == "noft":
        return None
    path = settings.paths.common_start_checkpoint
    if not path.is_file():
        raise FileNotFoundError(
            f"Updating method requires the completed 50-epoch NoFT common start: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("method") != "noft"
        or payload.get("selected_as_common_start") is not True
        or payload.get("completed_head_only_epochs") != settings.training.head_only_epochs
    ):
        raise RuntimeError("common_start.pt is not a P12 NoFT endpoint")
    expected = {
        "task": settings.task,
        "seed": settings.seed,
        "dataset_manifest_sha256": manifest_sha256,
        "source_checkpoint_sha256": model.source_manifest["sha256"],
        "implementation_sha256": implementation_digest,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if int(payload.get("epoch", -1)) not in range(1, settings.training.head_only_epochs + 1):
        mismatches["epoch"] = (payload.get("epoch"), "validation-best within 1..50")
    if mismatches:
        raise RuntimeError(f"Common-start identity mismatch: {mismatches}")
    common_sha256 = sha256_file(path)
    if require_completed_noft:
        if payload.get("synthetic_smoke_only") is True:
            raise RuntimeError("A synthetic smoke common cannot start formal training")
        noft_result_path = (
            settings.paths.output_root
            / settings.task
            / "noft"
            / f"seed_{settings.seed}"
            / "result.json"
        )
        if not noft_result_path.is_file():
            raise RuntimeError(
                f"Formal adaptation requires the completed NoFT result: {noft_result_path}"
            )
        noft_result = json.loads(noft_result_path.read_text(encoding="utf-8"))
        expected_result = {
            "format": RESULT_FORMAT,
            "status": "complete",
            "task": settings.task,
            "method": "noft",
            "seed": settings.seed,
            "epochs_completed_this_run": settings.training.head_only_epochs,
            "dataset_manifest_sha256": manifest_sha256,
            "source_checkpoint_sha256": model.source_manifest["sha256"],
            "implementation_sha256": implementation_digest,
            "common_start_sha256": common_sha256,
        }
        result_mismatches = {
            key: (noft_result.get(key), value)
            for key, value in expected_result.items()
            if noft_result.get(key) != value
        }
        if result_mismatches:
            raise RuntimeError(
                f"Completed NoFT result/common identity mismatch: {result_mismatches}"
            )
    model.load_state_dict(payload["model"], strict=True)
    if model.phase_report()["mean_absolute_rad"] > 1.0e-7:
        raise RuntimeError("NoFT common start changed the frozen P11 source phases")
    return common_sha256


def _train_epoch(
    model: P11DownstreamModel,
    loader,
    settings: Settings,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    if settings.method == "noft":
        # Frozen feature extractor semantics: no stochastic Slim-mixer dropout.
        model.backbone.eval()
        model.head.train()
    started = time.time()
    samples = 0
    loss_sum = 0.0
    parts_sum: dict[str, float] = {}
    max_batches = settings.limits.max_train_batches
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if settings.training.use_amp and device.type == "cuda"
            else nullcontext()
        )
        # Head gradients are needed for NoFT, so only the feature extractor is
        # evaluated under no_grad rather than wrapping the complete model.
        if settings.method == "noft":
            with amp:
                with torch.no_grad():
                    final, _ = model.forward_features(batch["image"])
                output = model.head(final.detach())
                loss, parts = task_loss(settings, output, batch)
        else:
            with amp:
                output = model(batch["image"])
                loss, parts = task_loss(settings, output, batch)
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(
                f"Non-finite training loss at epoch={epoch}, batch={batch_index + 1}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        phase_parameters = [
            parameter for parameter in model.phase_parameters() if parameter.requires_grad
        ]
        electronic_parameters = [
            parameter
            for parameter in (*model.backbone_parameters(), *model.head_parameters())
            if parameter.requires_grad
        ]
        if phase_parameters:
            torch.nn.utils.clip_grad_norm_(
                phase_parameters,
                settings.optimizer.phase_gradient_clip_norm,
                error_if_nonfinite=True,
            )
        torch.nn.utils.clip_grad_norm_(
            electronic_parameters,
            settings.optimizer.electronic_gradient_clip_norm,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_size = int(batch["image"].shape[0])
        samples += batch_size
        loss_sum += float(loss.detach()) * batch_size
        for key, value in parts.items():
            parts_sum[key] = parts_sum.get(key, 0.0) + float(value) * batch_size
        if (batch_index + 1) % settings.save.log_interval_batches == 0:
            print(
                f"[{settings.task}/{settings.method}/seed{settings.seed}] "
                f"epoch={epoch}/{settings.run_epochs} batch={batch_index + 1}/{len(loader)} "
                f"loss={loss_sum / max(samples, 1):.5f}",
                flush=True,
            )
    if not samples:
        raise RuntimeError("Training loader produced no samples")
    elapsed = time.time() - started
    return {
        "loss": loss_sum / samples,
        "samples": samples,
        "seconds": elapsed,
        "images_per_second": samples / max(elapsed, 1.0e-9),
        "parts": {key: value / samples for key, value in parts_sum.items()},
        "learning_rates": _learning_rates(optimizer),
    }


def _append_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                "epoch": row["epoch"],
                "train_loss": row["train"]["loss"],
                "validation_primary": row["validation_primary"],
                "validation_loss": row["validation"].get("loss"),
                "phase_mean_absolute_rad": row["phase"]["mean_absolute_rad"],
                "phase_fraction_over_0p1_rad": row["phase"]["fraction_over_0p1_rad"],
                "images_per_second": row["train"]["images_per_second"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def run_experiment(settings: Settings, *, resume: bool = True) -> dict[str, Any]:
    settings.validate_runtime_paths()
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resolved = settings.to_dict()
    implementation_digest = implementation_sha256()
    config_digest = sha256_json(
        {"settings": resolved, "implementation_sha256": implementation_digest}
    )
    seed_everything(settings.seed)
    _, bundle, loaders = build_data(settings)
    source_checkpoint_sha256 = sha256_file(settings.paths.source_backbone)
    if source_checkpoint_sha256 != settings.paths.source_backbone_sha256:
        raise RuntimeError(
            "P11 source checkpoint SHA-256 differs from the preregistered source: "
            f"expected {settings.paths.source_backbone_sha256}, "
            f"got {source_checkpoint_sha256}"
        )
    result_path = output / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            current_common_sha256 = (
                sha256_file(settings.paths.common_start_checkpoint)
                if settings.paths.common_start_checkpoint.is_file()
                else None
            )
            expected = {
                "format": RESULT_FORMAT,
                "task": settings.task,
                "method": settings.method,
                "seed": settings.seed,
                "epochs_completed_this_run": settings.run_epochs,
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "common_start_sha256": current_common_sha256,
                "config_digest": config_digest,
                "implementation_sha256": implementation_digest,
            }
            mismatches = {
                key: (existing.get(key), value)
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Completed result identity mismatch: {mismatches}")
            print(f"[P12] already complete: {output}", flush=True)
            return existing

    write_json(output / "resolved_config.json", resolved)
    launch_path = output / "launch.json"
    if not launch_path.is_file():
        write_json(
            launch_path,
            {
            "started_at_unix": time.time(),
            "git_commit": git_commit(settings.repo_root),
            "hostname": os.environ.get("HOSTNAME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "task": settings.task,
            "method": settings.method,
            "seed": settings.seed,
            "config_digest": config_digest,
            "implementation_sha256": implementation_digest,
            },
        )
    write_json(output / "dataset_summary.json", bundle.metadata)

    if not torch.cuda.is_available():
        raise RuntimeError("Formal P12 training requires a CUDA GPU")
    device = torch.device("cuda", 0)
    model = build_model(settings)
    frozen_stem_digest = module_state_sha256(model.backbone.stem)
    common_start_sha256 = _load_common_start(
        model, settings, bundle.manifest_sha256, implementation_digest
    )
    if module_state_sha256(model.backbone.stem) != frozen_stem_digest:
        raise RuntimeError("Common start changed the frozen Qwen stem state")
    model.set_backbone_trainable(settings.updates_backbone)
    feedback = configure_runtime_feedback(model, settings)
    model.to(device)
    # feedback_phase is runtime-only; configure again after the device move.
    feedback = configure_runtime_feedback(model, settings)
    source_report = model.parameter_report()
    write_json(output / "model_report.json", source_report)
    write_json(output / "feedback_manifest.json", feedback)
    torch.cuda.reset_peak_memory_stats(device)

    effective_steps = len(loaders["train"])
    if settings.limits.max_train_batches is not None:
        effective_steps = min(effective_steps, settings.limits.max_train_batches)
    optimizer, scheduler = build_optimizer_scheduler(
        model, settings, effective_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=settings.training.use_amp,
        init_scale=settings.training.amp_initial_scale,
        growth_interval=settings.training.amp_growth_interval,
    )
    start_epoch = 1
    best_metric = -math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    last_path = output / "checkpoints" / "last.pt"
    best_path = output / "checkpoints" / "best.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_identity(
            checkpoint,
            settings,
            manifest_sha256=bundle.manifest_sha256,
            config_digest=config_digest,
            implementation_digest=implementation_digest,
            source_checkpoint_sha256=source_checkpoint_sha256,
            common_start_sha256=common_start_sha256,
            source_phase_sha256=source_report["source_phase_sha256"],
            feedback_manifest=model.feedback_manifest(),
            frozen_stem_digest=frozen_stem_digest,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        configure_runtime_feedback(model, settings)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if settings.training.restore_rng_on_resume:
            restore_rng_state(checkpoint["rng_state"])
        generator_state = checkpoint.get("loader_generator_state")
        if generator_state is not None and loaders["train"].generator is not None:
            loaders["train"].generator.set_state(generator_state)
        resume_path = output / "resume_lineage.json"
        resume_lineage = (
            json.loads(resume_path.read_text(encoding="utf-8"))
            if resume_path.is_file()
            else []
        )
        resume_lineage.append(
            {
                "resumed_at_unix": time.time(),
                "from_epoch": int(checkpoint["epoch"]),
                "git_commit": git_commit(settings.repo_root),
                "config_digest": config_digest,
                "implementation_sha256": implementation_digest,
            }
        )
        write_json(resume_path, resume_lineage)
        print(f"[P12] resumed {output} at epoch {start_epoch}", flush=True)

    common_start_validation: dict[str, Any] | None = None
    common_validation_path = output / "metrics" / "common_start_validation.json"
    if settings.method != "noft":
        if start_epoch == 1:
            common_start_validation = evaluate(
                model,
                loaders["val"],
                settings,
                device,
                max_batches=settings.limits.max_validation_batches,
            )
            best_metric = _primary_metric(settings, common_start_validation)
            best_epoch = 0
            initial_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                settings=settings,
                epoch=0,
                best_metric=best_metric,
                best_epoch=0,
                history=history,
                manifest_sha256=bundle.manifest_sha256,
                common_start_sha256=common_start_sha256,
                config_digest=config_digest,
                implementation_digest=implementation_digest,
                frozen_stem_digest=frozen_stem_digest,
                loader_generator_state=(
                    loaders["train"].generator.get_state()
                    if loaders["train"].generator is not None
                    else None
                ),
            )
            save_torch(best_path, initial_payload)
            write_json(common_validation_path, common_start_validation)
            print(
                f"[P12] task={settings.task} method={settings.method} seed={settings.seed} "
                f"common_val_{settings.task_settings.primary_metric}={best_metric:.5f}",
                flush=True,
            )
        elif common_validation_path.is_file():
            common_start_validation = json.loads(
                common_validation_path.read_text(encoding="utf-8")
            )

    validation_batch = next(iter(loaders["val"]))
    initial_gradient: dict[str, Any] | None = None
    initial_gradient_path = output / "diagnostics" / "gradient_epoch_000.json"
    if start_epoch == 1:
        initial_gradient = gradient_diagnostic(
            model, validation_batch, settings, device
        )
        if initial_gradient is not None:
            write_json(initial_gradient_path, initial_gradient)
    elif initial_gradient_path.is_file():
        initial_gradient = json.loads(initial_gradient_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, settings.run_epochs + 1):
        train_metrics = _train_epoch(
            model, loaders["train"], settings, device, optimizer, scheduler, scaler, epoch
        )
        validation = evaluate(
            model,
            loaders["val"],
            settings,
            device,
            max_batches=settings.limits.max_validation_batches,
        )
        primary = _primary_metric(settings, validation)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "validation_primary": primary,
            "phase": model.phase_report(),
            "optical_gates": model.backbone.optical_gates(),
            "electronic_skip_gates": model.backbone.electronic_skip_gates(),
        }
        history.append(row)
        if min(row["optical_gates"]) < settings.model.optical_gate_min - 1.0e-7:
            raise RuntimeError("An optical gate fell below the preregistered 0.5 bound")
        if epoch in settings.save.diagnostic_epochs and epoch != 0:
            epoch_gradient = gradient_diagnostic(
                model, validation_batch, settings, device
            )
            if epoch_gradient is not None:
                write_json(
                    output / "diagnostics" / f"gradient_epoch_{epoch:03d}.json",
                    epoch_gradient,
                )
        loader_state = (
            loaders["train"].generator.get_state()
            if loaders["train"].generator is not None
            else None
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            settings=settings,
            epoch=epoch,
            best_metric=max(best_metric, primary),
            best_epoch=epoch if primary > best_metric else best_epoch,
            history=history,
            manifest_sha256=bundle.manifest_sha256,
            common_start_sha256=common_start_sha256,
            config_digest=config_digest,
            implementation_digest=implementation_digest,
            frozen_stem_digest=frozen_stem_digest,
            loader_generator_state=loader_state,
        )
        if primary > best_metric:
            best_metric = primary
            best_epoch = epoch
            save_torch(best_path, payload)
        # Save last after best so an interruption cannot advertise a best epoch
        # whose checkpoint has not reached durable storage yet.
        save_torch(last_path, payload)
        if epoch % settings.save.checkpoint_interval_epochs == 0:
            save_torch(output / "checkpoints" / f"epoch_{epoch:03d}.pt", payload)
        write_json(output / "metrics" / "history.json", history)
        _append_history_csv(output / "metrics" / "history.csv", history)
        print(
            f"[P12] task={settings.task} method={settings.method} seed={settings.seed} "
            f"epoch={epoch}/{settings.run_epochs} train_loss={train_metrics['loss']:.5f} "
            f"val_{settings.task_settings.primary_metric}={primary:.5f} "
            f"best={best_metric:.5f}@{best_epoch}",
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError("Training completed without a validation-best checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_identity(
        selected,
        settings,
        manifest_sha256=bundle.manifest_sha256,
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        source_checkpoint_sha256=source_checkpoint_sha256,
        common_start_sha256=common_start_sha256,
        source_phase_sha256=source_report["source_phase_sha256"],
        feedback_manifest=model.feedback_manifest(),
        frozen_stem_digest=frozen_stem_digest,
    )
    model.load_state_dict(selected["model"], strict=True)
    model.to(device)
    configure_runtime_feedback(model, settings)
    selected_stem_digest = module_state_sha256(model.backbone.stem)
    if selected_stem_digest != frozen_stem_digest:
        raise RuntimeError("Frozen Qwen stem state changed during downstream training")
    endpoint_gradient = gradient_diagnostic(
        model, validation_batch, settings, device
    )
    if endpoint_gradient is not None:
        write_json(
            output / "diagnostics" / "gradient_selected.json", endpoint_gradient
        )

    ablations = ["normal"]
    if settings.save.run_final_ablations:
        ablations.extend(["optical_off", "phase_random", "electronic_skip_off"])
    test: dict[str, Any] = {}
    for ablation in ablations:
        test[ablation] = evaluate(
            model,
            loaders["test"],
            settings,
            device,
            ablation=ablation,
            max_batches=settings.limits.max_test_batches,
            include_retrieval=False,
        )
    if settings.task == "caltech101":
        # Five held-out validation images per class form the labelled gallery;
        # the never-selected test images are disjoint queries. This avoids the
        # optimistic same-set retrieval protocol while keeping all 101 classes.
        gallery_embeddings, gallery_labels = _collect_caltech_embeddings(
            model,
            loaders["val"],
            settings,
            device,
            max_batches=settings.limits.max_validation_batches,
        )
        query_embeddings, query_labels = _collect_caltech_embeddings(
            model,
            loaders["test"],
            settings,
            device,
            max_batches=settings.limits.max_test_batches,
        )
        test["normal"].update(
            _retrieval_metrics(
                query_embeddings,
                query_labels,
                gallery_embeddings,
                gallery_labels,
            )
        )
        test["normal"]["retrieval_gallery_split"] = "validation"
        test["normal"]["retrieval_query_split"] = "test"

    # NoFT's selected 50-epoch endpoint is the byte-identical common start for
    # all three updating methods of this task/seed.
    if settings.method == "noft":
        common_payload = dict(selected)
        common_payload["selected_as_common_start"] = True
        common_payload["completed_head_only_epochs"] = settings.training.head_only_epochs
        save_torch(settings.paths.common_start_checkpoint, common_payload)
        common_start_sha256 = sha256_file(settings.paths.common_start_checkpoint)

    result = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "head_only_epochs": settings.training.head_only_epochs,
        "adaptation_epochs": 0 if settings.method == "noft" else settings.training.adaptation_epochs,
        "epochs_completed_this_run": settings.run_epochs,
        "inherited_pipeline_epochs": settings.inherited_pipeline_epochs,
        "best_epoch": int(selected["epoch"]),
        "best_validation_metric": float(selected["best_metric"]),
        "primary_metric": settings.task_settings.primary_metric,
        "test": test,
        "phase": model.phase_report(),
        "optical_gates": model.backbone.optical_gates(),
        "electronic_skip_gates": model.backbone.electronic_skip_gates(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "feedback": model.feedback_manifest(),
        "initial_gradient_diagnostic": initial_gradient,
        "selected_gradient_diagnostic": endpoint_gradient,
        "common_start_validation": common_start_validation,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "source_checkpoint_sha256": model.source_manifest["sha256"],
        "source_phase_sha256": source_report["source_phase_sha256"],
        "common_start_sha256": common_start_sha256,
        "config_digest": config_digest,
        "implementation_sha256": implementation_digest,
        "frozen_stem_state_sha256": selected_stem_digest,
        "git_commit": git_commit(settings.repo_root),
        "completed_at_unix": time.time(),
    }
    write_json(result_path, result)
    print(
        f"[P12] complete task={settings.task} method={settings.method} "
        f"seed={settings.seed} best={best_metric:.5f}@{best_epoch}",
        flush=True,
    )
    return result


# Stable public name used by command-line and queue launchers.
run_training = run_experiment
