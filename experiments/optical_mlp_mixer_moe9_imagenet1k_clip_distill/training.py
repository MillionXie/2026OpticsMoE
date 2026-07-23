from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .clip_teacher import load_text_prototypes
from .datasets import EpochViewSampler, ImageNetBundle
from .io_utils import write_csv, write_json
from .metrics import ClassificationAccumulator, RouterAccumulator, ScalarAccumulator
from .optics import OpticalMixerMoE9
from .settings import ExperimentSettings
from .teacher_cache import ClipFeatureStore, DistillationViewDataset, cache_directory
from .visualization import (
    save_debug_examples,
    save_phase_overview,
    save_router_charts,
    save_training_curves,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed OpticalMixer training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device(device_name)
    return DistributedContext(rank, local_rank, world_size, device)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def finalize_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def unwrap(model: nn.Module) -> OpticalMixerMoE9:
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_model(settings: ExperimentSettings, context: DistributedContext) -> nn.Module:
    model = OpticalMixerMoE9(settings).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    return model


def _data_loader(
    dataset,
    sampler,
    *,
    batch_size: int,
    settings: ExperimentSettings,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": settings.training.num_workers,
        "pin_memory": settings.training.pin_memory,
        "persistent_workers": (
            settings.training.persistent_workers and settings.training.num_workers > 0
        ),
        "drop_last": False,
    }
    if settings.training.num_workers > 0:
        kwargs["prefetch_factor"] = settings.training.prefetch_factor
    return DataLoader(**kwargs)


def build_loaders(
    bundle: ImageNetBundle,
    settings: ExperimentSettings,
    context: DistributedContext,
) -> tuple[DataLoader, DataLoader, EpochViewSampler, EpochViewSampler]:
    train_store = ClipFeatureStore("train", bundle.train, bundle, settings)
    validation_store = ClipFeatureStore(
        "validation", bundle.validation, bundle, settings
    )
    train_dataset = DistillationViewDataset(bundle.train, train_store)
    validation_dataset = DistillationViewDataset(bundle.validation, validation_store)
    train_sampler = EpochViewSampler(
        bundle.train,
        shuffle=True,
        seed=settings.training.seed,
        rank=context.rank,
        world_size=context.world_size,
    )
    validation_sampler = EpochViewSampler(
        bundle.validation,
        shuffle=False,
        seed=settings.training.seed,
        rank=context.rank,
        world_size=context.world_size,
    )
    train_loader = _data_loader(
        train_dataset,
        train_sampler,
        batch_size=settings.training.batch_size,
        settings=settings,
    )
    validation_loader = _data_loader(
        validation_dataset,
        validation_sampler,
        batch_size=settings.training.validation_batch_size,
        settings=settings,
    )
    return train_loader, validation_loader, train_sampler, validation_sampler


def build_optimizer(model: nn.Module, settings: ExperimentSettings):
    router_parameters = []
    main_parameters = []
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if ".core.router." in name:
            router_parameters.append(parameter)
        else:
            main_parameters.append(parameter)
    config = settings.optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": config.learning_rate, "name": "main"},
            {
                "params": router_parameters,
                "lr": config.router_learning_rate,
                "name": "router",
            },
        ],
        weight_decay=0.0,
        betas=config.betas,
        eps=config.eps,
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: ExperimentSettings,
    steps_per_epoch: int,
):
    total_steps = max(1, settings.training.epochs * steps_per_epoch)
    warmup_steps = max(0, settings.optimizer.warmup_epochs * steps_per_epoch)
    minimum_ratio = settings.optimizer.minimum_learning_rate / max(
        settings.optimizer.learning_rate, 1e-12
    )

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if settings.optimizer.scheduler == "cosine":
            value = 0.5 * (1 + math.cos(math.pi * progress))
        elif settings.optimizer.scheduler == "linear":
            value = 1 - progress
        else:
            raise ValueError(f"Unsupported scheduler {settings.optimizer.scheduler!r}")
        return minimum_ratio + (1 - minimum_ratio) * value

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def compute_losses(
    output,
    teacher_embedding: torch.Tensor,
    labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    clip_logit_scale: float,
    settings: ExperimentSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    teacher_embedding = F.normalize(teacher_embedding.float(), dim=-1)
    student_embedding = output.embedding.float()
    feature = (1 - (student_embedding * teacher_embedding).sum(-1)).mean()
    scale = float(clip_logit_scale) / max(settings.clip.logit_temperature, 1e-8)
    student_clip_logits = scale * student_embedding @ text_prototypes.T
    teacher_clip_logits = scale * teacher_embedding @ text_prototypes.T
    temperature = settings.loss.distill_temperature
    kd = F.kl_div(
        F.log_softmax(student_clip_logits / temperature, dim=-1),
        F.softmax(teacher_clip_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    ce = F.cross_entropy(output.logits.float(), labels)
    balance = output.router_balance_loss
    importance = output.router_importance_loss
    total = (
        settings.loss.feature_cosine_weight * feature
        + settings.loss.clip_logit_kd_weight * kd
        + settings.loss.supervised_ce_weight * ce
        + settings.router.balance_weight * balance
        + settings.router.importance_weight * importance
    )
    values = {
        "loss_total": total,
        "loss_feature": feature,
        "loss_kd": kd,
        "loss_ce": ce,
        "loss_router_balance": balance,
        "loss_router_importance": importance,
        "clip_cosine": 1 - feature,
        "zero_shot_top1": student_clip_logits.argmax(-1).eq(labels).float().mean(),
    }
    return total, values, student_clip_logits


def train(
    bundle: ImageNetBundle,
    settings: ExperimentSettings,
    context: DistributedContext,
) -> dict:
    train_loader, validation_loader, train_sampler, validation_sampler = build_loaders(
        bundle, settings, context
    )
    model = build_model(settings, context)
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings, len(train_loader))
    prototypes_path = cache_directory(settings) / "imagenet_text_prototypes.pt"
    text_prototypes, clip_logit_scale = load_text_prototypes(
        prototypes_path, bundle.class_names, settings, context.device
    )
    start_epoch = 1
    best_top1 = -math.inf
    history: list[dict] = []
    if settings.training.resume_checkpoint is not None:
        start_epoch, best_top1, history = load_training_checkpoint(
            settings.training.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            settings,
            context.device,
        )
    if context.is_main:
        write_json(
            settings.training.output_dir / "model.json",
            model_report(unwrap(model), settings),
        )
    history_path = settings.training.output_dir / "metrics" / "training_history.csv"
    best_path = settings.training.output_dir / "checkpoints" / "best.pt"
    last_path = settings.training.output_dir / "checkpoints" / "last.pt"
    for epoch in range(start_epoch, settings.training.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        epoch_started = time.perf_counter()
        train_metrics, train_router = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            text_prototypes,
            clip_logit_scale,
            settings,
            context,
            epoch,
        )
        validation_started = time.perf_counter()
        should_validate = epoch % settings.training.validation_interval_epochs == 0
        debug_due = (
            settings.visualization.enabled
            and epoch % settings.visualization.interval_epochs == 0
        )
        if should_validate:
            validation_metrics, validation_router, debug_saved = evaluate(
                model,
                validation_loader,
                text_prototypes,
                clip_logit_scale,
                bundle.class_names,
                settings,
                context,
                epoch=epoch,
                save_debug=debug_due,
            )
        else:
            validation_metrics, validation_router, debug_saved = {}, {}, 0
        validation_time = time.perf_counter() - validation_started
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
                if key not in {"per_class_accuracy", "per_class_samples"}
            },
            "debug_examples_saved": debug_saved,
            "validation_time_sec": validation_time,
            "epoch_time_sec": time.perf_counter() - epoch_started,
        }
        if context.is_main:
            history.append(row)
            write_csv(history_path, history)
            write_json(
                settings.training.output_dir / "metrics" / "training_latest.json",
                {
                    **row,
                    "train_router": train_router,
                    "validation_router": validation_router,
                },
            )
            if validation_metrics:
                write_json(
                    settings.training.output_dir
                    / "metrics"
                    / f"validation_epoch_{epoch:04d}.json",
                    {
                        **validation_metrics,
                        "router": validation_router,
                    },
                )
                save_router_charts(
                    validation_router,
                    settings.training.output_dir
                    / "figures"
                    / "router"
                    / f"validation_selection_epoch_{epoch:04d}.png",
                    f"Validation expert selection, epoch {epoch}",
                )
            current_top1 = validation_metrics.get("top1_accuracy", -math.inf)
            if current_top1 > best_top1:
                best_top1 = current_top1
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_top1,
                    history,
                    settings,
                )
                write_json(
                    settings.training.output_dir / "metrics" / "best_validation.json",
                    {
                        "epoch": epoch,
                        "criterion": "validation_top1_accuracy",
                        **validation_metrics,
                    },
                )
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_top1,
                history,
                settings,
            )
            if epoch % settings.training.checkpoint_interval_epochs == 0:
                save_checkpoint(
                    settings.training.output_dir
                    / "checkpoints"
                    / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_top1,
                    history,
                    settings,
                )
            if debug_due and settings.visualization.save_phase_overview:
                save_phase_overview(
                    unwrap(model),
                    settings.training.output_dir
                    / "figures"
                    / "phase_masks"
                    / f"epoch_{epoch:04d}.png",
                )
            if settings.visualization.save_training_curves:
                save_training_curves(
                    history_path,
                    settings.training.output_dir / "figures" / "training_curves.png",
                )
            print(
                f"epoch {epoch:03d}/{settings.training.epochs} "
                f"train_top1={train_metrics['top1_accuracy']:.4f} "
                f"val_top1={validation_metrics.get('top1_accuracy', float('nan')):.4f} "
                f"val_top5={validation_metrics.get('top5_accuracy', float('nan')):.4f} "
                f"clip_cos={validation_metrics.get('clip_cosine', float('nan')):.4f} "
                f"best={best_top1:.4f} time={row['epoch_time_sec']:.1f}s",
                flush=True,
            )
        barrier()
        # Broadcast the rank-zero best score so checkpoint metadata remains
        # consistent if this process later becomes the source of a resumed job.
        best_tensor = torch.tensor(best_top1, device=context.device)
        if context.world_size > 1:
            torch.distributed.broadcast(best_tensor, src=0)
        best_top1 = float(best_tensor.item())
    return {"best_validation_top1": best_top1, "best_checkpoint": str(best_path)}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    text_prototypes,
    clip_logit_scale,
    settings,
    context,
    epoch,
) -> tuple[dict, dict]:
    model.train()
    unwrap(model).set_phase_dropout_active(False)
    classification = ClassificationAccumulator(
        settings.model.num_classes, context.device
    )
    scalars = ScalarAccumulator(context.device)
    routers = RouterAccumulator(
        settings.model.num_blocks, settings.geometry.num_experts, context.device
    )
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        teacher_embedding = batch["teacher_embedding"].to(
            context.device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss, values, _ = compute_losses(
            output,
            teacher_embedding,
            labels,
            text_prototypes,
            clip_logit_scale,
            settings,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings.optimizer.gradient_clip_norm
        )
        optimizer.step()
        scheduler.step()
        classification.update(output.logits.detach(), labels, loss.detach())
        scalars.update(values, labels.numel())
        routers.update(output.router_statistics, labels.numel())
        if context.is_main and (
            batch_index % settings.training.log_interval_batches == 0
            or batch_index == len(loader)
        ):
            elapsed = time.perf_counter() - started
            current = classification.compute()
            print(
                f"epoch {epoch:03d}/{settings.training.epochs} "
                f"batch {batch_index:,}/{len(loader):,} "
                f"loss={float(loss):.5f} running_top1={current['top1_accuracy']:.4f} "
                f"clip_cos={float(values['clip_cosine']):.4f} "
                f"balance={float(values['loss_router_balance']):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
    classification.reduce()
    scalars.reduce()
    routers.reduce()
    values = classification.compute()
    values.update(scalars.compute())
    values["time_sec"] = time.perf_counter() - started
    return values, routers.compute()


@torch.no_grad()
def evaluate(
    model,
    loader,
    text_prototypes,
    clip_logit_scale,
    class_names,
    settings,
    context,
    *,
    epoch: int,
    save_debug: bool,
) -> tuple[dict, dict, int]:
    model.eval()
    unwrap(model).set_phase_dropout_active(False)
    classification = ClassificationAccumulator(
        settings.model.num_classes, context.device
    )
    scalars = ScalarAccumulator(context.device)
    routers = RouterAccumulator(
        settings.model.num_blocks, settings.geometry.num_experts, context.device
    )
    debug_saved = 0
    for batch_index, batch in enumerate(loader, 1):
        capture = save_debug and context.is_main and batch_index == 1
        unwrap(model).set_debug_capture(
            settings.visualization.capture_block_indices, capture
        )
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        teacher_embedding = batch["teacher_embedding"].to(
            context.device, non_blocking=True
        )
        output = model(images)
        loss, values, _ = compute_losses(
            output,
            teacher_embedding,
            labels,
            text_prototypes,
            clip_logit_scale,
            settings,
        )
        classification.update(output.logits, labels, loss)
        scalars.update(values, labels.numel())
        routers.update(output.router_statistics, labels.numel())
        if capture:
            debug_saved = save_debug_examples(
                epoch=epoch,
                images=images,
                labels=labels,
                sample_indices=batch["sample_index"],
                paths=list(batch["path"]),
                class_names=class_names,
                model=unwrap(model),
                output_dir=settings.training.output_dir,
                sample_count=settings.visualization.sample_count,
                percentile=settings.visualization.percentile_clip,
                save_raw=settings.visualization.save_raw_tensors,
            )
            unwrap(model).set_debug_capture([], False)
    classification.reduce()
    scalars.reduce()
    routers.reduce()
    debug_tensor = torch.tensor(debug_saved, device=context.device)
    if context.world_size > 1:
        torch.distributed.broadcast(debug_tensor, src=0)
    values = classification.compute()
    values.update(scalars.compute())
    return values, routers.compute(), int(debug_tensor.item())


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_top1: float,
    history: list[dict],
    settings: ExperimentSettings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_validation_top1": float(best_top1),
            "history": history,
            "model_name": settings.model.name,
            "optical_parameter_formula": settings.optical_parameter_formula,
            "settings": settings.to_dict(),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    settings,
    device,
) -> tuple[int, float, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != settings.model.name:
        raise RuntimeError("Checkpoint model name does not match OpticalMixerMoE9")
    if payload.get("optical_parameter_formula") != settings.optical_parameter_formula:
        raise RuntimeError("Checkpoint optical geometry/parameter formula differs")
    unwrap(model).load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return (
        int(payload["epoch"]) + 1,
        float(payload.get("best_validation_top1", -math.inf)),
        list(payload.get("history", [])),
    )


def load_model_checkpoint(path: Path, model: nn.Module, settings: ExperimentSettings) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("optical_parameter_formula") != settings.optical_parameter_formula:
        raise RuntimeError("Checkpoint optical geometry does not match current config")
    unwrap(model).load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def model_report(model: OpticalMixerMoE9, settings: ExperimentSettings) -> dict:
    actual = model.parameter_breakdown()
    expected = settings.optical_parameter_formula
    if actual["optical"]["total_phase_parameters"] != expected["optical_phase_parameters_total"]:
        raise RuntimeError(
            f"Actual optical parameters {actual['optical']['total_phase_parameters']:,} "
            f"do not match formula {expected['optical_phase_parameters_total']:,}"
        )
    return {
        "architecture": {
            "name": "OpticalMixerMoE9",
            "input_shape": ["B", 3, settings.model.image_size, settings.model.image_size],
            "patch_embedding_output": [
                "B",
                settings.model.token_count,
                settings.model.hidden_size,
            ],
            "optical_field_shape": [
                "B",
                settings.geometry.expert_size,
                settings.geometry.expert_size,
            ],
            "blocks": settings.model.num_blocks,
            "token_stages_per_block": settings.model.token_stages_per_block,
            "channel_stages_per_block": settings.model.channel_stages_per_block,
            "router_calls_per_block": 1,
            "routing_reused_across_all_five_stages": True,
            "token_field_mapping": "[B,196,224] -> transpose -> zero-pad columns to [B,224,224]",
            "channel_field_mapping": "[B,196,224] -> zero-pad rows to [B,224,224]",
            "interpolation_used": False,
        },
        "geometry": model.blocks[0].core.geometry.report(),
        "physical_parameters": {
            "wavelength_nm": settings.optics.wavelength_nm,
            "pixel_pitch_um": settings.optics.pixel_pitch_um,
            "inter_layer_distance_m": settings.optics.inter_layer_distance_m,
            "readout_to_global_distance_m": settings.optics.readout_to_global_distance_m,
            "global_to_detector_distance_m": settings.optics.global_to_detector_distance_m,
            "phase_only": True,
        },
        "parameter_formula": expected,
        "parameter_breakdown": actual,
    }


def final_evaluation(
    bundle: ImageNetBundle,
    settings: ExperimentSettings,
    context: DistributedContext,
    checkpoint: Path | None = None,
) -> dict:
    _, loader, _, sampler = build_loaders(bundle, settings, context)
    sampler.set_epoch(0)
    model = build_model(settings, context)
    checkpoint = checkpoint or settings.training.output_dir / "checkpoints" / "best.pt"
    payload = load_model_checkpoint(checkpoint, model, settings)
    prototypes, logit_scale = load_text_prototypes(
        cache_directory(settings) / "imagenet_text_prototypes.pt",
        bundle.class_names,
        settings,
        context.device,
    )
    metrics, router, debug_saved = evaluate(
        model,
        loader,
        prototypes,
        logit_scale,
        bundle.class_names,
        settings,
        context,
        epoch=int(payload.get("epoch", 0)),
        save_debug=settings.visualization.enabled,
    )
    report = {
        "split": "imagenet1k_validation",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        **metrics,
        "router": router,
        "debug_examples_saved": debug_saved,
    }
    if context.is_main:
        write_json(
            settings.training.output_dir / "metrics" / "final_validation.json",
            report,
        )
        rows = [
            {
                "class_index": index,
                "class_name": bundle.class_names[index],
                "accuracy": metrics["per_class_accuracy"][index],
                "samples": metrics["per_class_samples"][index],
            }
            for index in range(len(bundle.class_names))
        ]
        write_csv(
            settings.training.output_dir / "metrics" / "validation_per_class.csv",
            rows,
        )
        save_router_charts(
            router,
            settings.training.output_dir / "figures" / "router" / "final_validation.png",
            "Final ImageNet-1K validation expert selection",
        )
    barrier()
    return report
