from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from .io_utils import write_json
from .modeling import (
    LoadedBackbone,
    OpticalRetrievalReadout,
    build_optical_student,
    load_backbone,
    trainable_parameter_report,
    unique_trainable_parameters,
)
from .optical_artifacts import save_phase_preview, save_phase_snapshot
from .optics.physical import phase_dc_loss, phase_dc_statistics
from .optics.replacement import DeepStackMultimodalReplacement
from .prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GroceryRetrievalDataset,
    GrocerySample,
    collate_grocery,
    prepare_grocery_subset,
)
from .retrieval_metrics import evaluate_embeddings
from .settings import Settings, load_settings


class PKBatchSampler(Sampler[list[int]]):
    """Deterministic epoch-aware P-SKU x K-image batch sampler."""

    def __init__(
        self,
        samples: Sequence[GrocerySample],
        p: int,
        k: int,
        seed: int,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[sample.sku_index].append(index)
        self.grouped = {key: tuple(values) for key, values in sorted(grouped.items())}
        if len(self.grouped) < self.p:
            raise ValueError(f"PK sampler needs P={self.p} SKUs, found {len(self.grouped)}")
        if any(not values for values in self.grouped.values()):
            raise ValueError("PK sampler encountered an empty SKU")
        natural_batch_count = max(1, math.ceil(len(samples) / (self.p * self.k)))
        self.batch_count = (
            natural_batch_count
            if steps_per_epoch is None
            else int(steps_per_epoch)
        )
        if self.batch_count <= 0:
            raise ValueError("steps_per_epoch must be positive when configured")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        sku_ids = list(self.grouped)
        pools: dict[int, list[int]] = {}
        positions: dict[int, int] = {}
        for sku in sku_ids:
            order = torch.randperm(len(self.grouped[sku]), generator=generator).tolist()
            pools[sku] = [self.grouped[sku][index] for index in order]
            positions[sku] = 0
        sku_order = torch.randperm(len(sku_ids), generator=generator).tolist()
        sku_cursor = 0
        for _ in range(self.batch_count):
            if self.p == len(sku_ids):
                selected_skus = sku_ids
            else:
                if sku_cursor + self.p > len(sku_order):
                    sku_order = torch.randperm(len(sku_ids), generator=generator).tolist()
                    sku_cursor = 0
                selected_skus = [sku_ids[index] for index in sku_order[sku_cursor:sku_cursor + self.p]]
                sku_cursor += self.p
            batch: list[int] = []
            for sku in selected_skus:
                for _ in range(self.k):
                    if positions[sku] >= len(pools[sku]):
                        order = torch.randperm(
                            len(self.grouped[sku]), generator=generator
                        ).tolist()
                        pools[sku] = [self.grouped[sku][index] for index in order]
                        positions[sku] = 0
                    batch.append(pools[sku][positions[sku]])
                    positions[sku] += 1
            yield batch


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Supervised contrastive inputs must be [B,D] and [B]")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / float(temperature)
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not torch.all(valid):
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"Every contrastive anchor needs a same-SKU positive; missing anchors={missing}"
        )
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~identity
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator_mask, -torch.inf), dim=1
    )
    log_probability = logits - log_denominator[:, None]
    mean_positive_log_probability = (
        (log_probability * positive).sum(dim=1) / positive.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()


def embedding_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student/teacher embedding shapes differ: {student.shape} vs {teacher.shape}"
        )
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def relational_embedding_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """Match the Teacher's off-diagonal pairwise cosine geometry.

    Pointwise cosine KD leaves many embedding configurations with nearly the
    same loss.  On a small retrieval dataset, matching all pairwise relations
    in the current query/gallery batch supplies O(B^2) geometric constraints
    without adding a classifier or changing inference.
    """
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student/teacher embedding shapes differ: {student.shape} vs {teacher.shape}"
        )
    if student.ndim != 2:
        raise ValueError("Relational embedding KD expects [B,D] tensors")
    if len(student) < 2:
        return student.float().sum() * 0.0
    student_normalized = F.normalize(student.float(), dim=-1)
    teacher_normalized = F.normalize(teacher.float(), dim=-1)
    student_relations = student_normalized @ student_normalized.T
    teacher_relations = teacher_normalized @ teacher_normalized.T
    off_diagonal = ~torch.eye(
        len(student), dtype=torch.bool, device=student.device
    )
    return F.mse_loss(
        student_relations[off_diagonal], teacher_relations[off_diagonal]
    )


def gallery_retrieval_loss(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    *,
    stop_gradient_on_gallery: bool = False,
) -> torch.Tensor:
    """Cross-entropy retrieval against one differentiable prototype per SKU.

    Unlike the in-batch supervised contrastive objective, this loss exactly
    matches deployment: every natural query must rank its standard gallery SKU
    above all other gallery SKUs. Multiple gallery images per SKU are supported
    by normalized mean prototypes.
    """
    logits, targets = gallery_retrieval_logits(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels,
        temperature,
        stop_gradient_on_gallery=stop_gradient_on_gallery,
    )
    return F.cross_entropy(logits, targets)


def gallery_retrieval_logits(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    *,
    stop_gradient_on_gallery: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query-to-gallery-prototype logits and contiguous targets."""
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Query and gallery embeddings must both be [N,D]")
    if query_labels.ndim != 1 or gallery_labels.ndim != 1:
        raise ValueError("Query and gallery labels must both be [N]")
    if len(query_embeddings) != len(query_labels):
        raise ValueError("Query embeddings and labels have different lengths")
    if len(gallery_embeddings) != len(gallery_labels):
        raise ValueError("Gallery embeddings and labels have different lengths")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query = F.normalize(query_embeddings.float(), dim=-1)
    gallery = F.normalize(gallery_embeddings.float(), dim=-1)
    if stop_gradient_on_gallery:
        gallery = gallery.detach()
    sku_ids = torch.unique(gallery_labels, sorted=True)
    if sku_ids.numel() < 2:
        raise RuntimeError("Gallery retrieval loss needs at least two distinct SKUs")
    prototypes = []
    for sku_id in sku_ids:
        prototype = gallery[gallery_labels == sku_id].mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))
    prototype_tensor = torch.stack(prototypes)
    targets = torch.searchsorted(sku_ids, query_labels)
    valid = (targets < len(sku_ids)) & sku_ids[
        targets.clamp_max(len(sku_ids) - 1)
    ].eq(query_labels)
    if not torch.all(valid):
        missing = torch.unique(query_labels[~valid]).detach().cpu().tolist()
        raise RuntimeError(f"Gallery is missing query SKU labels: {missing}")
    logits = query @ prototype_tensor.T / float(temperature)
    return logits, targets


def retrieval_ranking_sums(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return additive Top-1/Top-3/MRR statistics for epoch aggregation."""
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("Retrieval logits/targets must be [N,C] and [N]")
    if logits.shape[1] < 2:
        raise ValueError("Retrieval metrics need at least two candidate SKUs")
    ranking = logits.detach().float().argsort(dim=1, descending=True)
    top1 = ranking[:, 0].eq(targets)
    top3 = ranking[:, : min(3, logits.shape[1])].eq(targets[:, None]).any(dim=1)
    matching_positions = ranking.eq(targets[:, None]).nonzero(as_tuple=False)
    if len(matching_positions) != len(targets):
        raise RuntimeError("Each retrieval target must occur exactly once in its ranking")
    reciprocal_rank = 1.0 / (matching_positions[:, 1].float() + 1.0)
    return {
        "top1_correct": float(top1.sum()),
        "top3_correct": float(top3.sum()),
        "reciprocal_rank_sum": float(reciprocal_rank.sum()),
        "query_count": float(len(targets)),
    }


def select_gallery_items_for_queries(
    gallery_items: Sequence[dict[str, Any]],
    query_samples: Sequence[GrocerySample],
) -> list[dict[str, Any]]:
    """Select all standard gallery images for only the SKUs in this query batch."""
    requested = sorted({int(sample.sku_index) for sample in query_samples})
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in gallery_items:
        grouped[int(item["sample"].sku_index)].append(item)
    missing = [sku_index for sku_index in requested if sku_index not in grouped]
    if missing:
        raise RuntimeError(f"Gallery is missing query SKU labels: {missing}")
    return [
        item
        for sku_index in requested
        for item in grouped[sku_index]
    ]


def _router_diagnostics(
    replacement: DeepStackMultimodalReplacement,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        routing = surrogate.core.last_routing
        selected = routing["selected_mask"].detach()
        importance = routing["importance"].detach()
        output[f"{name}_router_entropy"] = float(
            routing["normalized_entropy"].detach()
        )
        output[f"{name}_router_active_experts"] = float(
            selected.any(dim=0).sum()
        )
        output[f"{name}_router_max_importance"] = float(importance.max())
    return output


def _build_optimizer(
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    settings: Settings,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
    parameters = unique_trainable_parameters(replacement, readout)
    router_parameters = list(replacement.vision_surrogate.core.router.parameters())
    router_parameters.extend(
        replacement.language_surrogate.core.router.parameters()
    )
    router_ids = {id(parameter) for parameter in router_parameters}
    if len(router_ids) != len(router_parameters):
        raise RuntimeError("Vision/language router parameter sets unexpectedly overlap")
    phase_parameters = [
        parameter
        for surrogate in (
            replacement.vision_surrogate,
            replacement.language_surrogate,
        )
        for name, parameter in surrogate.named_parameters()
        if "raw_phase" in name and parameter.requires_grad
    ]
    phase_ids = {id(parameter) for parameter in phase_parameters}
    if len(phase_ids) != len(phase_parameters) or phase_ids & router_ids:
        raise RuntimeError("Optical phase/router parameter groups unexpectedly overlap")
    readout_parameters = [
        parameter for parameter in readout.parameters() if parameter.requires_grad
    ]
    readout_ids = {id(parameter) for parameter in readout_parameters}
    adapter_parameters = []
    for surrogate in (
        replacement.vision_surrogate,
        replacement.language_surrogate,
    ):
        for module in (
            getattr(surrogate.core, name, None)
            for name in ("input_adapter", "input_norm", "output_adapter")
        ):
            if module is None:
                continue
            adapter_parameters.extend(
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad
            )
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    reserved_ids = router_ids | phase_ids | readout_ids | adapter_ids
    if sum(map(len, (router_ids, phase_ids, readout_ids, adapter_ids))) != len(
        reserved_ids
    ):
        raise RuntimeError("Optimizer parameter groups unexpectedly overlap")
    base_parameters = [
        parameter for parameter in parameters if id(parameter) not in reserved_ids
    ]
    configured_router_lr = settings.router_learning_rate
    configured_phase_lr = settings.phase_learning_rate
    adapter_group_name = (
        "optical_adapters"
        if getattr(replacement, "has_optical_phases", True)
        else "electronic_adapters"
    )
    group_specs = (
        ("student_base", base_parameters, settings.learning_rate),
        (
            adapter_group_name,
            adapter_parameters,
            getattr(settings, "adapter_learning_rate", None)
            if getattr(settings, "adapter_learning_rate", None) is not None
            else settings.learning_rate,
        ),
        (
            "retrieval_readout",
            readout_parameters,
            getattr(settings, "readout_learning_rate", None)
            if getattr(settings, "readout_learning_rate", None) is not None
            else settings.learning_rate,
        ),
        (
            "optical_phases",
            phase_parameters,
            configured_phase_lr
            if configured_phase_lr is not None
            else settings.learning_rate,
        ),
        (
            "routers",
            router_parameters,
            configured_router_lr
            if configured_router_lr is not None
            else settings.learning_rate,
        ),
    )
    optimizer_groups = [
        {
            "params": group_parameters,
            "lr": float(group_lr),
            "configured_lr": float(group_lr),
            "group_name": group_name,
        }
        for group_name, group_parameters, group_lr in group_specs
        if group_parameters
    ]
    grouped_ids = {
        id(parameter)
        for group in optimizer_groups
        for parameter in group["params"]
    }
    if grouped_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError(
            "Optimizer parameter partition does not cover every trainable tensor exactly once"
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=settings.weight_decay,
    )
    return optimizer, parameters


def _restore_configured_group_lrs(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("configured_lr", group["lr"]))


def _set_phase_focus_trainability(
    optimizer: torch.optim.Optimizer,
    enabled: bool,
) -> None:
    """Alternate joint and phase-only block-coordinate updates.

    A phase-only epoch keeps the already learned adapters/router/readout in the
    forward graph but prevents their Adam moments and weights from absorbing
    the task gradient.  The only optimizer-owned tensors that receive a
    gradient/update are the physical expert/global raw phases.
    """
    for group in optimizer.param_groups:
        train_group = not enabled or group.get("group_name") == "optical_phases"
        group["lr"] = (
            float(group.get("configured_lr", group["lr"])) if train_group else 0.0
        )
        for parameter in group["params"]:
            parameter.requires_grad_(train_group)


def _phase_focus_epoch(settings: Settings, relative_epoch: int) -> bool:
    if not settings.phase_focus_enabled:
        return False
    if relative_epoch <= settings.phase_focus_warmup_epochs:
        return False
    offset = relative_epoch - settings.phase_focus_warmup_epochs - 1
    return offset % settings.phase_focus_interval_epochs == 0


def _phase_named_parameters(
    replacement: DeepStackMultimodalReplacement,
) -> dict[str, list[nn.Parameter]]:
    output: dict[str, list[nn.Parameter]] = {}
    for stack_name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        output[f"{stack_name}_expert"] = [
            expert.raw_phase
            for layer in surrogate.core.expert_layers
            for expert in layer.experts
        ]
        output[f"{stack_name}_global"] = [
            surrogate.core.global_phase.phase.raw_phase
        ]
    return output


@torch.no_grad()
def _phase_reference(
    phase_groups: dict[str, list[nn.Parameter]],
) -> dict[str, list[torch.Tensor]]:
    return {
        name: [parameter.detach().float().clone() for parameter in parameters]
        for name, parameters in phase_groups.items()
    }


def _physical_phase(parameter: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.sigmoid(parameter.float())


@torch.no_grad()
def _phase_motion_statistics(
    phase_groups: dict[str, list[nn.Parameter]],
    run_reference: dict[str, list[torch.Tensor]],
    epoch_reference: dict[str, list[torch.Tensor]],
) -> dict[str, float | int]:
    report: dict[str, float | int] = {}
    all_current: list[torch.Tensor] = []
    all_run_delta: list[torch.Tensor] = []
    all_epoch_delta: list[torch.Tensor] = []
    for name, parameters in phase_groups.items():
        current = torch.cat(
            [_physical_phase(parameter.detach()).reshape(-1) for parameter in parameters]
        )
        run_start = torch.cat(
            [_physical_phase(value).reshape(-1) for value in run_reference[name]]
        )
        epoch_start = torch.cat(
            [_physical_phase(value).reshape(-1) for value in epoch_reference[name]]
        )
        run_delta = current - run_start
        epoch_delta = current - epoch_start
        report[f"{name}_phase_std_rad"] = float(current.std(unbiased=False))
        report[f"{name}_phase_delta_run_rms_rad"] = float(
            run_delta.square().mean().sqrt()
        )
        report[f"{name}_phase_delta_epoch_rms_rad"] = float(
            epoch_delta.square().mean().sqrt()
        )
        report[f"{name}_phase_moved_fraction_gt_0p01"] = float(
            run_delta.abs().gt(0.01).float().mean()
        )
        all_current.append(current)
        all_run_delta.append(run_delta)
        all_epoch_delta.append(epoch_delta)
    current = torch.cat(all_current)
    run_delta = torch.cat(all_run_delta)
    epoch_delta = torch.cat(all_epoch_delta)
    raw = torch.cat(
        [
            parameter.detach().float().reshape(-1)
            for parameters in phase_groups.values()
            for parameter in parameters
        ]
    )
    report.update(
        {
            "phase_physical_std_rad": float(current.std(unbiased=False)),
            "phase_delta_from_pi_rms_rad": float(
                (current - math.pi).square().mean().sqrt()
            ),
            "phase_delta_run_rms_rad": float(run_delta.square().mean().sqrt()),
            "phase_delta_epoch_rms_rad": float(
                epoch_delta.square().mean().sqrt()
            ),
            "phase_raw_abs_mean": float(raw.abs().mean()),
            "phase_sigmoid_saturation_fraction_abs_raw_gt_4": float(
                raw.abs().gt(4.0).float().mean()
            ),
        }
    )
    return report


def _phase_gradient_statistics(
    phase_groups: dict[str, list[nn.Parameter]],
) -> dict[str, float | int]:
    report: dict[str, float | int] = {}
    all_gradients: list[torch.Tensor] = []
    missing_total = 0
    for name, parameters in phase_groups.items():
        gradients = [
            parameter.grad.detach().float().reshape(-1)
            for parameter in parameters
            if parameter.grad is not None
        ]
        missing = len(parameters) - len(gradients)
        missing_total += missing
        if gradients:
            flattened = torch.cat(gradients)
            report[f"{name}_phase_grad_rms"] = float(
                flattened.square().mean().sqrt()
            )
            report[f"{name}_phase_grad_max"] = float(flattened.abs().max())
            all_gradients.append(flattened)
        else:
            report[f"{name}_phase_grad_rms"] = 0.0
            report[f"{name}_phase_grad_max"] = 0.0
        report[f"{name}_phase_grad_missing_planes"] = missing
    if all_gradients:
        flattened = torch.cat(all_gradients)
        report["phase_grad_rms"] = float(flattened.square().mean().sqrt())
        report["phase_grad_max"] = float(flattened.abs().max())
    else:
        report["phase_grad_rms"] = 0.0
        report["phase_grad_max"] = 0.0
    report["phase_grad_missing_planes"] = missing_total
    return report


@torch.no_grad()
def initialize_parameter_ema(
    parameters: Sequence[nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().float().clone() for parameter in parameters]


@torch.no_grad()
def update_parameter_ema(
    ema_parameters: Sequence[torch.Tensor],
    parameters: Sequence[nn.Parameter],
    decay: float,
) -> None:
    if len(ema_parameters) != len(parameters):
        raise ValueError("EMA and live parameter lists have different lengths")
    if not 0.0 < decay < 1.0:
        raise ValueError("EMA decay must be strictly between 0 and 1")
    for ema, parameter in zip(ema_parameters, parameters):
        ema.mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)


@contextmanager
def use_parameter_ema(
    parameters: Sequence[nn.Parameter],
    ema_parameters: Sequence[torch.Tensor],
) -> Iterator[None]:
    """Temporarily install EMA weights and restore live weights afterwards."""
    if len(ema_parameters) != len(parameters):
        raise ValueError("EMA and live parameter lists have different lengths")
    backups = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter, ema in zip(parameters, ema_parameters):
                parameter.copy_(ema.to(dtype=parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)


def save_checkpoint(
    path: Path,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    settings: Settings,
    *,
    weight_variant: str = "live",
) -> None:
    payload = {
        "checkpoint_version": 2,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "vision_optical": replacement.vision_surrogate.state_dict(),
        "language_optical": replacement.language_surrogate.state_dict(),
        "retrieval_readout": readout.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "embedding_dim": settings.embedding_dim,
            "detector_dim": settings.detector_output_size,
            "instruction": settings.instruction,
            "model_id": settings.model_id,
            "expert_stages_per_stack": settings.expert_layers,
            "vision_tap_stages": list(settings.vision_tap_stages),
            "student_deepstack_auxiliary_count": len(settings.vision_tap_stages),
            "language_optical_layer_indexes": list(
                replacement.language_optical_layer_indexes
            ),
            "optical_architecture": "one_expert_stage_plus_one_global_phase",
            "selection_criterion": "minimum_training_total_loss",
            "test_metrics_used_for_selection": False,
            "weight_variant": weight_variant,
            "training_objective": {
                "lambda_kd": settings.lambda_kd,
                "lambda_relational_kd": settings.lambda_relational_kd,
                "lambda_ret": settings.lambda_ret,
                "lambda_gallery": settings.lambda_gallery,
                "lambda_teacher_gallery": settings.lambda_teacher_gallery,
                "lambda_router_balance": settings.lambda_router_balance,
                "lambda_router_importance": settings.lambda_router_importance,
                "phase_dc_enabled": settings.phase_dc_enabled,
                "lambda_phase_dc": settings.lambda_phase_dc,
                "phase_dc_start_epoch": settings.phase_dc_start_epoch,
                "temperature": settings.temperature,
                "gallery_temperature": settings.gallery_temperature,
                "gallery_prototype_stop_gradient": (
                    settings.gallery_prototype_stop_gradient
                ),
            },
            "learning_rate": settings.learning_rate,
            "adapter_learning_rate": settings.adapter_learning_rate,
            "readout_learning_rate": settings.readout_learning_rate,
            "router_learning_rate": settings.router_learning_rate,
            "phase_learning_rate": settings.phase_learning_rate,
            "phase_focus": {
                "enabled": settings.phase_focus_enabled,
                "warmup_epochs": settings.phase_focus_warmup_epochs,
                "interval_epochs": settings.phase_focus_interval_epochs,
            },
            "ema_decay": settings.ema_decay,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Optical retrieval checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected_layers = len(replacement.vision_surrogate.core.expert_layers)
    saved_layers = metadata.get("expert_stages_per_stack")
    if saved_layers is not None and int(saved_layers) != expected_layers:
        raise RuntimeError(
            "Optical retrieval checkpoint architecture mismatch: "
            f"saved expert stages={saved_layers}, current expert stages={expected_layers}. "
            "The corrected baseline uses one expert phase stage plus one global "
            "phase plane; do not reuse a four-stage Student checkpoint. The "
            "frozen Teacher embedding cache remains reusable."
        )
    replacement.vision_surrogate.load_state_dict(payload["vision_optical"])
    replacement.language_surrogate.load_state_dict(payload["language_optical"])
    readout.load_state_dict(payload["retrieval_readout"])
    payload["_optimizer_state_loaded"] = False
    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            payload["_optimizer_state_loaded"] = True
        except ValueError as error:
            # Checkpoints made before the phase-engagement fix used one broad
            # ``student_base`` group.  Their model weights remain compatible,
            # but restoring those Adam moments would merge phase/readout LR
            # ownership again.  Reset the optimizer explicitly and report it.
            print(
                "WARNING: checkpoint optimizer groups are incompatible with the "
                "phase-engaged optimizer; model weights were loaded and Adam state "
                f"was reset. Original error: {error}",
                flush=True,
            )
    return payload


def train_optical_retrieval(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    bundle: GroceryRetrievalBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Settings,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    train_dataset = GroceryRetrievalDataset(
        bundle.train_samples,
        settings.image_size,
        augment=settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,
        brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,
        rotation_degrees=settings.rotation_degrees,
    )
    sampler = PKBatchSampler(
        bundle.train_samples,
        settings.pk_skus_per_batch,
        settings.pk_images_per_sku,
        settings.random_seed,
        settings.optimizer_steps_per_epoch,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    gallery_dataset = GroceryRetrievalDataset(
        bundle.gallery_samples,
        settings.image_size,
        augment=False,
    )
    gallery_items = [
        gallery_dataset[index] for index in range(len(gallery_dataset))
    ]
    gallery_sku_ids = {
        int(item["sample"].sku_index) for item in gallery_items
    }
    if settings.lambda_gallery > 0 and len(gallery_sku_ids) != len(bundle.class_names):
        raise RuntimeError(
            "Gallery-aligned loss requires at least one standard image for every SKU"
        )
    optimizer, parameters = _build_optimizer(replacement, readout, settings)
    start_epoch = 1
    resumed_from_epoch: int | None = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        payload = load_checkpoint(
            resume_checkpoint,
            replacement,
            readout,
            optimizer=optimizer if settings.resume_optimizer_state else None,
        )
        resumed_from_epoch = int(payload["epoch"])
        start_epoch = resumed_from_epoch + 1
        # A continuation config may deliberately lower either learning rate.
        # Checkpoint optimizer state must never silently override that choice.
        _restore_configured_group_lrs(optimizer)
        _archive_pre_resume_outputs(
            settings.output_dir, resume_checkpoint, resumed_from_epoch
        )
        print(
            f"Resuming Student weights from epoch {resumed_from_epoch}; "
            f"optimizer_state={'loaded' if payload.get('_optimizer_state_loaded') else 'reset'}, "
            f"additional_epochs={settings.epochs}, "
            f"absolute_epoch_range={start_epoch}..{resumed_from_epoch + settings.epochs}"
        )
    report = trainable_parameter_report(
        loaded.model, replacement, readout
    )
    report["continuation"] = {
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resumed_from_epoch": resumed_from_epoch,
        "optimizer_state_resumed": bool(
            resume_checkpoint is not None
            and payload.get("_optimizer_state_loaded", False)
        ),
        "base_learning_rate": settings.learning_rate,
        "adapter_learning_rate": settings.adapter_learning_rate,
        "readout_learning_rate": settings.readout_learning_rate,
        "router_learning_rate": settings.router_learning_rate,
        "phase_learning_rate": settings.phase_learning_rate,
        "phase_dc_enabled": settings.phase_dc_enabled,
        "lambda_phase_dc": settings.lambda_phase_dc,
        "lambda_router_response_consistency": (
            settings.lambda_router_response_consistency
        ),
        "phase_dc_start_epoch": settings.phase_dc_start_epoch,
        "optimizer_groups": [
            {
                "name": group.get("group_name", "unnamed"),
                "learning_rate": float(group["lr"]),
                "parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer.param_groups
        ],
        "phase_focus": {
            "enabled": settings.phase_focus_enabled,
            "warmup_epochs": settings.phase_focus_warmup_epochs,
            "interval_epochs": settings.phase_focus_interval_epochs,
        },
        "ema_decay": settings.ema_decay,
    }
    write_json(settings.output_dir / "model.json", report)
    loaded.model.eval()
    replacement.use_student()
    best_train_loss = math.inf
    history_path = settings.output_dir / "train_log.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "learning_rate",
        "adapter_learning_rate",
        "readout_learning_rate",
        "router_learning_rate",
        "phase_learning_rate",
        "phase_focus_epoch",
        "total_loss",
        "kd_loss",
        "relational_kd_loss",
        "retrieval_loss",
        "gallery_loss",
        "teacher_gallery_loss",
        "router_balance_loss",
        "router_importance_loss",
        "router_response_consistency_loss",
        "phase_dc_loss",
        "phase_dc_weighted_loss",
        "phase_dc_current_loss",
        "phase_dc_rho_mean",
        "phase_dc_rho_max",
        "phase_dc_plane_count",
        "vision_router_entropy",
        "language_router_entropy",
        "vision_router_active_experts",
        "language_router_active_experts",
        "vision_router_max_importance",
        "language_router_max_importance",
        "vision_router_min_selection_count",
        "vision_router_max_selection_count",
        "vision_router_unselected_experts",
        "language_router_min_selection_count",
        "language_router_max_selection_count",
        "language_router_unselected_experts",
        "phase_grad_rms",
        "phase_grad_max",
        "phase_grad_missing_planes",
        "vision_expert_phase_grad_rms",
        "vision_expert_phase_grad_max",
        "vision_expert_phase_grad_missing_planes",
        "vision_global_phase_grad_rms",
        "vision_global_phase_grad_max",
        "vision_global_phase_grad_missing_planes",
        "language_expert_phase_grad_rms",
        "language_expert_phase_grad_max",
        "language_expert_phase_grad_missing_planes",
        "language_global_phase_grad_rms",
        "language_global_phase_grad_max",
        "language_global_phase_grad_missing_planes",
        "phase_physical_std_rad",
        "phase_delta_from_pi_rms_rad",
        "phase_delta_run_rms_rad",
        "phase_delta_epoch_rms_rad",
        "phase_raw_abs_mean",
        "phase_sigmoid_saturation_fraction_abs_raw_gt_4",
        "vision_expert_phase_std_rad",
        "vision_expert_phase_delta_run_rms_rad",
        "vision_expert_phase_delta_epoch_rms_rad",
        "vision_expert_phase_moved_fraction_gt_0p01",
        "vision_global_phase_std_rad",
        "vision_global_phase_delta_run_rms_rad",
        "vision_global_phase_delta_epoch_rms_rad",
        "vision_global_phase_moved_fraction_gt_0p01",
        "language_expert_phase_std_rad",
        "language_expert_phase_delta_run_rms_rad",
        "language_expert_phase_delta_epoch_rms_rad",
        "language_expert_phase_moved_fraction_gt_0p01",
        "language_global_phase_std_rad",
        "language_global_phase_delta_run_rms_rad",
        "language_global_phase_delta_epoch_rms_rad",
        "language_global_phase_moved_fraction_gt_0p01",
        "trainable_tensors_without_gradient",
        "samples",
        "model_forward_samples",
        "train_top1",
        "train_top3",
        "train_mrr",
        "train_metric_definition",
        "epoch_time_sec",
        "test_top1",
        "test_top3",
        "test_mrr",
        "ema_test_top1",
        "ema_test_top3",
        "ema_test_mrr",
        "checkpoint_selected_by",
    ]
    rows: list[dict[str, Any]] = (
        _read_history(history_path) if resume_checkpoint is not None else []
    )
    if resume_checkpoint is None and history_path.is_file():
        history_path.unlink()
    if rows:
        latest_logged_epoch = max(int(row["epoch"]) for row in rows)
        if latest_logged_epoch != resumed_from_epoch:
            raise RuntimeError(
                "Resume checkpoint/history mismatch: "
                f"checkpoint epoch={resumed_from_epoch}, latest train_log epoch="
                f"{latest_logged_epoch}. Restore matching artifacts before continuing."
            )
    best_observed_test_top1 = max(
        (
            float(row["test_top1"])
            for row in rows
            if row.get("test_top1") not in (None, "")
        ),
        default=-math.inf,
    )
    best_observed_ema_test_top1 = max(
        (
            float(row["ema_test_top1"])
            for row in rows
            if row.get("ema_test_top1") not in (None, "")
        ),
        default=-math.inf,
    )
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    ema_parameters = (
        initialize_parameter_ema(parameters)
        if settings.ema_decay is not None
        else None
    )
    phase_groups = _phase_named_parameters(replacement)
    phase_run_reference = _phase_reference(phase_groups)
    end_epoch = start_epoch + settings.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        relative_epoch = epoch - start_epoch + 1
        phase_focus = _phase_focus_epoch(settings, relative_epoch)
        _set_phase_focus_trainability(optimizer, phase_focus)
        phase_epoch_reference = _phase_reference(phase_groups)
        phase_gradient_totals: dict[str, float] = defaultdict(float)
        phase_gradient_measurements = 0
        trainable_with_gradient: set[int] = set()
        router_selection_counts = {
            "vision": torch.zeros(settings.num_experts, dtype=torch.long),
            "language": torch.zeros(settings.num_experts, dtype=torch.long),
        }
        sampler.set_epoch(epoch)
        replacement.set_student_train_mode()
        readout.train()
        totals = {
            "total": 0.0,
            "kd": 0.0,
            "relational_kd": 0.0,
            "ret": 0.0,
            "gallery": 0.0,
            "teacher_gallery": 0.0,
            "balance": 0.0,
            "importance": 0.0,
            "router_response": 0.0,
            "phase_dc": 0.0,
            "samples": 0,
            "forward_samples": 0,
            "top1_correct": 0.0,
            "top3_correct": 0.0,
            "reciprocal_rank_sum": 0.0,
            "retrieval_queries": 0,
        }
        router_totals = {
            "vision_router_entropy": 0.0,
            "language_router_entropy": 0.0,
            "vision_router_active_experts": 0.0,
            "language_router_active_experts": 0.0,
            "vision_router_max_importance": 0.0,
            "language_router_max_importance": 0.0,
        }
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            query_count = len(batch["samples"])
            gallery_training_enabled = (
                settings.lambda_gallery > 0
                or settings.lambda_teacher_gallery > 0
            )
            selected_gallery_batch = (
                collate_grocery(
                    select_gallery_items_for_queries(
                        gallery_items, batch["samples"]
                    )
                )
                if gallery_training_enabled
                else None
            )
            combined_images = (
                batch["images"] + selected_gallery_batch["images"]
                if gallery_training_enabled
                else batch["images"]
            )
            combined_samples = (
                batch["samples"] + selected_gallery_batch["samples"]
                if gallery_training_enabled
                else batch["samples"]
            )
            inputs = preprocess_images(
                loaded.processor, combined_images, settings.instruction
            )
            validate_token_budgets(inputs, settings)
            inputs = move_inputs(inputs, loaded.device)
            query_labels = torch.tensor(
                [sample.sku_index for sample in batch["samples"]],
                device=loaded.device,
                dtype=torch.long,
            )
            selected_gallery_labels = (
                torch.tensor(
                    [
                        sample.sku_index
                        for sample in selected_gallery_batch["samples"]
                    ],
                    device=loaded.device,
                    dtype=torch.long,
                )
                if gallery_training_enabled
                else None
            )
            combined_labels = (
                torch.cat((query_labels, selected_gallery_labels))
                if gallery_training_enabled
                else query_labels
            )
            teacher = teacher_store.lookup(combined_samples).to(
                loaded.device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=loaded.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                student, detector = student_embeddings(
                    loaded.model, replacement, readout, inputs
                )
                if detector.shape != (
                    len(combined_samples),
                    settings.detector_output_size,
                ):
                    raise RuntimeError(
                        f"Student detector output shape {tuple(detector.shape)} is invalid"
                    )
                kd = embedding_distillation_loss(student, teacher)
                relational_kd = relational_embedding_distillation_loss(
                    student, teacher
                )
                retrieval = supervised_contrastive_loss(
                    student, combined_labels, settings.temperature
                )
                if gallery_training_enabled:
                    gallery_logits, gallery_targets = gallery_retrieval_logits(
                        student[:query_count],
                        query_labels,
                        student[query_count:],
                        selected_gallery_labels,
                        settings.gallery_temperature,
                        stop_gradient_on_gallery=(
                            settings.gallery_prototype_stop_gradient
                        ),
                    )
                    gallery = F.cross_entropy(gallery_logits, gallery_targets)
                    teacher_gallery_logits, teacher_gallery_targets = (
                        gallery_retrieval_logits(
                            student[:query_count],
                            query_labels,
                            teacher[query_count:],
                            selected_gallery_labels,
                            settings.gallery_temperature,
                            stop_gradient_on_gallery=True,
                        )
                    )
                    teacher_gallery = F.cross_entropy(
                        teacher_gallery_logits, teacher_gallery_targets
                    )
                else:
                    gallery_logits = None
                    gallery_targets = None
                    gallery = student.new_zeros(())
                    teacher_gallery = student.new_zeros(())
                router_losses = replacement.router_losses()
                balance = 0.5 * (
                    router_losses["vision_balance"]
                    + router_losses["language_balance"]
                )
                importance = 0.5 * (
                    router_losses["vision_importance"]
                    + router_losses["language_importance"]
                )
                router_response = (
                    replacement.router_response_consistency_loss()
                    if settings.lambda_router_response_consistency > 0.0
                    else student.new_zeros(())
                )
                phase_dc_active = (
                    settings.phase_dc_enabled
                    and settings.lambda_phase_dc > 0.0
                    and relative_epoch >= settings.phase_dc_start_epoch
                )
                dc = (
                    phase_dc_loss(replacement)
                    if phase_dc_active
                    else student.new_zeros(())
                )
                total = (
                    settings.lambda_kd * kd
                    + settings.lambda_relational_kd * relational_kd
                    + settings.lambda_ret * retrieval
                    + settings.lambda_gallery * gallery
                    + settings.lambda_teacher_gallery * teacher_gallery
                    + settings.lambda_router_balance * balance
                    + settings.lambda_router_importance * importance
                    + settings.lambda_router_response_consistency
                    * router_response
                    + settings.lambda_phase_dc * dc
                )
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_index}: "
                    f"total={total}, kd={kd}, retrieval={retrieval}, "
                    f"relational_kd={relational_kd}, gallery={gallery}, "
                    f"teacher_gallery={teacher_gallery}, balance={balance}, "
                    f"importance={importance}, phase_dc={dc}"
                )
            total.backward()
            for parameter in parameters:
                if parameter.grad is not None:
                    trainable_with_gradient.add(id(parameter))
            bad_gradients = [
                index
                for index, parameter in enumerate(parameters)
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise RuntimeError(f"Non-finite gradients in trainable tensors {bad_gradients}")
            if (
                batch_index % settings.phase_gradient_measure_interval_batches == 0
                or batch_index == len(loader)
            ):
                gradient_report = _phase_gradient_statistics(phase_groups)
                for key, value in gradient_report.items():
                    phase_gradient_totals[key] += float(value)
                phase_gradient_measurements += 1
            if settings.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in parameters if parameter.requires_grad],
                    max_norm=settings.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            optimizer.step()
            if ema_parameters is not None:
                update_parameter_ema(
                    ema_parameters, parameters, float(settings.ema_decay)
                )
            count = query_count
            totals["total"] += float(total.detach()) * count
            totals["kd"] += float(kd.detach()) * count
            totals["relational_kd"] += float(relational_kd.detach()) * count
            totals["ret"] += float(retrieval.detach()) * count
            totals["gallery"] += float(gallery.detach()) * count
            totals["teacher_gallery"] += float(teacher_gallery.detach()) * count
            totals["balance"] += float(balance.detach()) * count
            totals["importance"] += float(importance.detach()) * count
            totals["router_response"] += float(router_response.detach()) * count
            totals["phase_dc"] += float(dc.detach()) * count
            totals["samples"] += count
            totals["forward_samples"] += len(combined_samples)
            if gallery_logits is not None and gallery_targets is not None:
                ranking = retrieval_ranking_sums(
                    gallery_logits, gallery_targets
                )
                totals["top1_correct"] += ranking["top1_correct"]
                totals["top3_correct"] += ranking["top3_correct"]
                totals["reciprocal_rank_sum"] += ranking[
                    "reciprocal_rank_sum"
                ]
                totals["retrieval_queries"] += int(ranking["query_count"])
            diagnostics = _router_diagnostics(replacement)
            for stack_name, surrogate in (
                ("vision", replacement.vision_surrogate),
                ("language", replacement.language_surrogate),
            ):
                router_selection_counts[stack_name] += (
                    surrogate.core.last_routing["selected_mask"]
                    .detach()
                    .sum(dim=0)
                    .cpu()
                    .long()
                )
            for key, value in diagnostics.items():
                router_totals[key] += value * count
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                print(
                    f"epoch {epoch:03d}/{end_epoch:03d} "
                    f"batch {batch_index:04d}/{len(loader):04d} "
                    f"loss={totals['total']/totals['samples']:.5f} "
                    f"kd={totals['kd']/totals['samples']:.5f} "
                    f"rel_kd={totals['relational_kd']/totals['samples']:.5f} "
                    f"ret={totals['ret']/totals['samples']:.5f} "
                    f"gallery={totals['gallery']/totals['samples']:.5f} "
                    f"teacher_gallery="
                    f"{totals['teacher_gallery']/totals['samples']:.5f} "
                    f"train_top1="
                    f"{totals['top1_correct']/max(1, totals['retrieval_queries']):.4f} "
                    f"balance={totals['balance']/totals['samples']:.5f} "
                    f"importance={totals['importance']/totals['samples']:.5f} "
                    f"router_response="
                    f"{totals['router_response']/totals['samples']:.5f} "
                    f"phase_dc={totals['phase_dc']/totals['samples']:.5f} "
                    f"phase_focus={'yes' if phase_focus else 'no'} "
                    f"phase_grad="
                    f"{phase_gradient_totals.get('phase_grad_rms', 0.0)/max(1, phase_gradient_measurements):.3e} "
                    f"active_v={diagnostics['vision_router_active_experts']:.0f}/"
                    f"{settings.num_experts} "
                    f"active_l={diagnostics['language_router_active_experts']:.0f}/"
                    f"{settings.num_experts}"
                )
        # Restore all optimizer-owned tensors before evaluation/checkpointing;
        # focus mode changes update ownership, not the saved model definition.
        _set_phase_focus_trainability(optimizer, False)
        sample_count = int(totals["samples"])
        average_total = totals["total"] / sample_count
        phase_motion = _phase_motion_statistics(
            phase_groups, phase_run_reference, phase_epoch_reference
        )
        dc_statistics = (
            phase_dc_statistics(replacement)
            if settings.phase_dc_enabled
            else {
                "phase_dc_current_loss": 0.0,
                "phase_dc_rho_mean": 0.0,
                "phase_dc_rho_max": 0.0,
                "phase_dc_plane_count": 0,
            }
        )
        phase_gradients = {
            key: value / max(1, phase_gradient_measurements)
            for key, value in phase_gradient_totals.items()
        }
        coverage: dict[str, int] = {}
        for stack_name, counts in router_selection_counts.items():
            coverage[f"{stack_name}_router_min_selection_count"] = int(counts.min())
            coverage[f"{stack_name}_router_max_selection_count"] = int(counts.max())
            coverage[f"{stack_name}_router_unselected_experts"] = int(
                counts.eq(0).sum()
            )
        expected_gradient_ids = (
            {
                id(parameter)
                for grouped in phase_groups.values()
                for parameter in grouped
            }
            if phase_focus
            else {id(parameter) for parameter in parameters}
        )
        missing_gradient_tensors = len(
            expected_gradient_ids - trainable_with_gradient
        )
        retrieval_query_count = int(totals["retrieval_queries"])
        train_top1 = (
            totals["top1_correct"] / retrieval_query_count
            if retrieval_query_count
            else None
        )
        train_top3 = (
            totals["top3_correct"] / retrieval_query_count
            if retrieval_query_count
            else None
        )
        train_mrr = (
            totals["reciprocal_rank_sum"] / retrieval_query_count
            if retrieval_query_count
            else None
        )
        test_metrics: dict[str, Any] = {}
        if settings.evaluate_test_each_epoch:
            test_metrics = evaluate_student_split(
                loaded,
                replacement,
                readout,
                bundle.test_samples,
                bundle.gallery_samples,
                bundle.class_names,
                settings,
            )
        ema_test_metrics: dict[str, Any] = {}
        if ema_parameters is not None and settings.evaluate_test_each_epoch:
            with use_parameter_ema(parameters, ema_parameters):
                ema_test_metrics = evaluate_student_split(
                    loaded,
                    replacement,
                    readout,
                    bundle.test_samples,
                    bundle.gallery_samples,
                    bundle.class_names,
                    settings,
                )
        row = {
            "epoch": epoch,
            "learning_rate": settings.learning_rate,
            "adapter_learning_rate": (
                settings.adapter_learning_rate
                if settings.adapter_learning_rate is not None
                else settings.learning_rate
            ),
            "readout_learning_rate": (
                settings.readout_learning_rate
                if settings.readout_learning_rate is not None
                else settings.learning_rate
            ),
            "router_learning_rate": (
                settings.router_learning_rate
                if settings.router_learning_rate is not None
                else settings.learning_rate
            ),
            "phase_learning_rate": (
                settings.phase_learning_rate
                if settings.phase_learning_rate is not None
                else settings.learning_rate
            ),
            "phase_focus_epoch": phase_focus,
            "total_loss": average_total,
            "kd_loss": totals["kd"] / sample_count,
            "relational_kd_loss": totals["relational_kd"] / sample_count,
            "retrieval_loss": totals["ret"] / sample_count,
            "gallery_loss": totals["gallery"] / sample_count,
            "teacher_gallery_loss": totals["teacher_gallery"] / sample_count,
            "router_balance_loss": totals["balance"] / sample_count,
            "router_importance_loss": totals["importance"] / sample_count,
            "router_response_consistency_loss": (
                totals["router_response"] / sample_count
            ),
            "phase_dc_loss": totals["phase_dc"] / sample_count,
            "phase_dc_weighted_loss": (
                settings.lambda_phase_dc * totals["phase_dc"] / sample_count
            ),
            **dc_statistics,
            **{
                key: value / sample_count
                for key, value in router_totals.items()
            },
            **coverage,
            **phase_gradients,
            **phase_motion,
            "trainable_tensors_without_gradient": missing_gradient_tensors,
            "samples": sample_count,
            "model_forward_samples": int(totals["forward_samples"]),
            "train_top1": train_top1,
            "train_top3": train_top3,
            "train_mrr": train_mrr,
            "train_metric_definition": (
                "augmented training queries vs current Student gallery prototypes "
                "for the PK-selected SKUs"
                if retrieval_query_count
                else "disabled because lambda_gallery=0"
            ),
            "epoch_time_sec": time.perf_counter() - started,
            "test_top1": test_metrics.get("top1_retrieval_accuracy"),
            "test_top3": test_metrics.get("top3_retrieval_accuracy"),
            "test_mrr": test_metrics.get("mrr"),
            "ema_test_top1": ema_test_metrics.get("top1_retrieval_accuracy"),
            "ema_test_top3": ema_test_metrics.get("top3_retrieval_accuracy"),
            "ema_test_mrr": ema_test_metrics.get("mrr"),
            "checkpoint_selected_by": "training_total_loss",
        }
        rows.append(row)
        _write_history(history_path, rows, fieldnames)
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        write_json(settings.output_dir / "metrics" / "phase_training_latest.json", {
            "epoch": epoch,
            "phase_focus_epoch": phase_focus,
            "phase_dc": dc_statistics,
            "phase_gradients": phase_gradients,
            "phase_motion": phase_motion,
            "router_coverage": coverage,
            "trainable_tensors_without_gradient": missing_gradient_tensors,
        })
        has_optical_phases = bool(
            getattr(replacement, "has_optical_phases", True)
        )
        if (
            has_optical_phases
            and
            relative_epoch >= settings.phase_motion_warning_epoch
            and phase_motion["phase_delta_run_rms_rad"]
            < settings.phase_motion_warning_threshold_rad
        ):
            print(
                "WARNING: optical phase is still effectively stationary: "
                f"delta_run_rms={phase_motion['phase_delta_run_rms_rad']:.6f} rad "
                f"after {relative_epoch} epoch(s), below configured threshold "
                f"{settings.phase_motion_warning_threshold_rad:.6f} rad. "
                "Inspect phase_grad_rms, router coverage, and phase optimizer LR.",
                flush=True,
            )
        if has_optical_phases and (
            relative_epoch % settings.phase_preview_interval_epochs == 0
            or epoch == end_epoch
        ):
            phase_epoch_dir = (
                settings.output_dir / "phase_training" / f"epoch_{epoch:04d}"
            )
            save_phase_snapshot(
                replacement,
                phase_epoch_dir,
                epoch=epoch,
                train_loss=average_total,
                weight_variant="live",
            )
            save_phase_preview(
                replacement,
                phase_epoch_dir / "phase_preview.png",
                title=(
                    f"Optical retrieval phase epoch {epoch} "
                    f"(focus={'yes' if phase_focus else 'no'})"
                ),
            )
        save_checkpoint(
            settings.output_dir / "last_checkpoint.pt",
            replacement,
            readout,
            optimizer,
            epoch,
            average_total,
            settings,
        )
        if ema_parameters is not None:
            with use_parameter_ema(parameters, ema_parameters):
                save_checkpoint(
                    settings.output_dir / "ema_last_checkpoint.pt",
                    replacement,
                    readout,
                    optimizer,
                    epoch,
                    average_total,
                    settings,
                    weight_variant="ema",
                )
        observed_test_top1 = test_metrics.get("top1_retrieval_accuracy")
        if (
            observed_test_top1 is not None
            and float(observed_test_top1) > best_observed_test_top1
        ):
            best_observed_test_top1 = float(observed_test_top1)
            observed_path = settings.output_dir / "best_observed_test_checkpoint.pt"
            save_checkpoint(
                observed_path,
                replacement,
                readout,
                optimizer,
                epoch,
                average_total,
                settings,
                weight_variant="live",
            )
            write_json(
                settings.output_dir / "metrics" / "best_observed_test.json",
                {
                    "epoch": epoch,
                    "test_top1": best_observed_test_top1,
                    "selection_criterion": "maximum repeatedly observed test Top-1",
                    "selection_biased": True,
                    "checkpoint": str(observed_path),
                },
            )
        observed_ema_test_top1 = ema_test_metrics.get("top1_retrieval_accuracy")
        if (
            ema_parameters is not None
            and observed_ema_test_top1 is not None
            and float(observed_ema_test_top1) > best_observed_ema_test_top1
        ):
            best_observed_ema_test_top1 = float(observed_ema_test_top1)
            observed_ema_path = (
                settings.output_dir / "ema_best_observed_test_checkpoint.pt"
            )
            with use_parameter_ema(parameters, ema_parameters):
                save_checkpoint(
                    observed_ema_path,
                    replacement,
                    readout,
                    optimizer,
                    epoch,
                    average_total,
                    settings,
                    weight_variant="ema",
                )
            write_json(
                settings.output_dir / "metrics" / "ema_best_observed_test.json",
                {
                    "epoch": epoch,
                    "test_top1": best_observed_ema_test_top1,
                    "selection_criterion": "maximum repeatedly observed EMA test Top-1",
                    "selection_biased": True,
                    "checkpoint": str(observed_ema_path),
                },
            )
        if average_total < best_train_loss:
            best_train_loss = average_total
            continuation_checkpoint_name = (
                f"best_continuation_from_epoch_{resumed_from_epoch:04d}.pt"
                if resumed_from_epoch is not None
                else None
            )
            continuation_metrics_name = (
                f"best_continuation_from_epoch_{resumed_from_epoch:04d}.json"
                if resumed_from_epoch is not None
                else None
            )
            best_checkpoint_name = "best_train_loss_checkpoint.pt"
            best_metrics_name = "best_train_loss.json"
            save_checkpoint(
                settings.output_dir / best_checkpoint_name,
                replacement,
                readout,
                optimizer,
                epoch,
                average_total,
                settings,
            )
            if has_optical_phases:
                save_phase_snapshot(
                    replacement,
                    settings.output_dir / "best_optical_artifacts" / "live_weights",
                    epoch=epoch,
                    train_loss=average_total,
                    weight_variant="live",
                )
                save_phase_preview(
                    replacement,
                    settings.output_dir
                    / "best_optical_artifacts"
                    / "live_weights"
                    / "phase_preview.png",
                    title=f"Best live optical phase at epoch {epoch}",
                )
            if ema_parameters is not None:
                with use_parameter_ema(parameters, ema_parameters):
                    save_checkpoint(
                        settings.output_dir / "ema_best_train_loss_checkpoint.pt",
                        replacement,
                        readout,
                        optimizer,
                        epoch,
                        average_total,
                        settings,
                        weight_variant="ema",
                    )
                    if has_optical_phases:
                        save_phase_snapshot(
                            replacement,
                            settings.output_dir
                            / "best_optical_artifacts"
                            / "ema_weights",
                            epoch=epoch,
                            train_loss=average_total,
                            weight_variant="ema",
                        )
                        save_phase_preview(
                            replacement,
                            settings.output_dir
                            / "best_optical_artifacts"
                            / "ema_weights"
                            / "phase_preview.png",
                            title=f"Best EMA optical phase at epoch {epoch}",
                        )
            write_json(
                settings.output_dir / "metrics" / best_metrics_name,
                {
                    "epoch": epoch,
                    "train_total_loss": average_total,
                    "selection_criterion": "minimum_training_total_loss",
                    "test_was_not_used_for_selection": True,
                    "resumed_from_epoch": resumed_from_epoch,
                    "checkpoint": str(settings.output_dir / best_checkpoint_name),
                },
            )
            if continuation_checkpoint_name is not None:
                shutil.copy2(
                    settings.output_dir / best_checkpoint_name,
                    settings.output_dir / continuation_checkpoint_name,
                )
                shutil.copy2(
                    settings.output_dir / "metrics" / best_metrics_name,
                    settings.output_dir
                    / "metrics"
                    / continuation_metrics_name,
                )
        architecture_diagnostics = (
            (
                f"phase_focus={'yes' if phase_focus else 'no'} "
                f"phase_delta={phase_motion['phase_delta_run_rms_rad']:.4f}rad "
                f"phase_std={phase_motion['phase_physical_std_rad']:.4f}rad "
                f"phase_grad={phase_gradients.get('phase_grad_rms', 0.0):.3e} "
                f"unselected_v/l={coverage['vision_router_unselected_experts']}/"
                f"{coverage['language_router_unselected_experts']} "
            )
            if has_optical_phases
            else "architecture=dense_electronic_no_moe "
        )
        print(
            f"epoch {epoch:03d} complete train_loss={average_total:.5f} "
            f"train_top1="
            f"{train_top1 if train_top1 is not None else float('nan'):.4f} "
            f"test_top1={test_metrics.get('top1_retrieval_accuracy', float('nan')):.4f} "
            f"ema_test_top1="
            f"{ema_test_metrics.get('top1_retrieval_accuracy', float('nan')):.4f} "
            f"{architecture_diagnostics}"
            f"best_train_loss={best_train_loss:.5f} "
            f"best_observed_test={best_observed_test_top1:.4f} "
            f"best_observed_ema_test={best_observed_ema_test_top1:.4f}"
        )
    return {
        "additional_epochs": settings.epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "resumed_from_epoch": resumed_from_epoch,
        "best_train_loss": best_train_loss,
        "last_train_loss": rows[-1]["total_loss"],
        "checkpoint_selection": "minimum training total loss (test not used)",
    }


@torch.no_grad()
def encode_student_samples(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    samples: Sequence[GrocerySample],
    settings: Settings,
) -> torch.Tensor:
    dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        # This loader is created for one finite evaluation pass. Persistent
        # workers would survive until delayed garbage collection; constructing
        # query and gallery loaders every epoch would then leak processes/file
        # descriptors and eventually fail with "Too many open files".
        persistent_workers=False,
        collate_fn=collate_grocery,
    )
    loaded.model.eval()
    replacement.use_student()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()
    chunks: list[torch.Tensor] = []
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for batch in loader:
        inputs = preprocess_images(
            loaded.processor, batch["images"], settings.instruction
        )
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        with torch.autocast(
            device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp
        ):
            embeddings, _ = student_embeddings(
                loaded.model, replacement, readout, inputs
            )
        chunks.append(embeddings.detach().cpu())
    output = torch.cat(chunks, dim=0)
    if output.shape != (len(samples), settings.embedding_dim):
        raise RuntimeError(f"Encoded student embedding shape is {tuple(output.shape)}")
    return output


@torch.no_grad()
def evaluate_student_split(
    loaded: LoadedBackbone,
    replacement: DeepStackMultimodalReplacement,
    readout: OpticalRetrievalReadout,
    query_samples: Sequence[GrocerySample],
    gallery_samples: Sequence[GrocerySample],
    class_names: Sequence[str],
    settings: Settings,
) -> dict[str, Any]:
    gallery = encode_student_samples(
        loaded, replacement, readout, gallery_samples, settings
    )
    query = encode_student_samples(
        loaded, replacement, readout, query_samples, settings
    )
    return evaluate_embeddings(
        query,
        query_samples,
        gallery,
        gallery_samples,
        class_names,
        settings.gallery_aggregation,
        system_name=(
            "optical_student_query_vs_optical_student_gallery"
            if getattr(replacement, "has_optical_phases", True)
            else "electronic_student_query_vs_electronic_student_gallery"
        ),
    ).metrics


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _archive_pre_resume_outputs(
    output_dir: Path, resume_checkpoint: Path, epoch: int
) -> None:
    """Preserve the original run before continuation starts overwriting `last`."""
    archive = output_dir / "checkpoints" / f"pre_resume_epoch_{epoch:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    candidates = {
        "resume_checkpoint.pt": resume_checkpoint,
        "best_train_loss_checkpoint.pt": output_dir
        / "best_train_loss_checkpoint.pt",
        "train_log.csv": output_dir / "train_log.csv",
        "training_latest.json": output_dir / "metrics" / "training_latest.json",
        "best_train_loss.json": output_dir / "metrics" / "best_train_loss.json",
    }
    for target_name, source in candidates.items():
        target = archive / target_name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def _write_history(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_grocery_subset(settings, persist=True)
    teacher_store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    replacement, readout = build_optical_student(loaded, settings)
    train_optical_retrieval(
        loaded,
        replacement,
        readout,
        bundle,
        teacher_store,
        settings,
        resume_checkpoint=(
            Path(args.resume_checkpoint).expanduser().resolve()
            if args.resume_checkpoint
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
