from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .datasets_bench2drive import Bench2DriveBCDataset, collate_bench2drive
from .io_utils import append_csv, atomic_torch_save, torch_load, write_json
from .modeling import (
    DrivingActor,
    OpticalDrivingPolicy,
    decode_normalized_action,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import behavior_cloning_loss, control_metrics


def bench_loader(
    records: list[Any], settings: Any, *, training: bool
) -> DataLoader:
    dataset = Bench2DriveBCDataset(records, settings.image_size)
    generator = torch.Generator()
    generator.manual_seed(settings.random_seed)
    return DataLoader(
        dataset,
        batch_size=settings.bc_batch_size,
        shuffle=training,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_bench2drive,
        generator=generator if training else None,
    )


def build_policy(backbone: Any, settings: Any) -> OpticalDrivingPolicy:
    actor = DrivingActor(
        visual_dim=224,
        num_commands=settings.num_commands,
        hidden_dims=settings.actor_hidden_dims,
    ).to(backbone.device)
    return OpticalDrivingPolicy(backbone, actor, settings).to(backbone.device)


def train_behavior_cloning(
    policy: OpticalDrivingPolicy,
    processor: Any,
    train_records: list[Any],
    validation_records: list[Any],
    settings: Any,
    device: torch.device,
    *,
    stage: int,
) -> dict[str, Any]:
    if stage not in {1, 2}:
        raise ValueError("Behavior-cloning stage must be 1 or 2")
    train_loader = bench_loader(train_records, settings, training=True)
    validation_loader = bench_loader(validation_records, settings, training=False)
    if stage == 1:
        policy.backbone.requires_grad_(False).eval()
        policy.actor.requires_grad_(True)
        policy.actor.log_std.requires_grad_(False)
        epochs = settings.bc_stage1_epochs
        optimizer = torch.optim.AdamW(
            [p for p in policy.actor.parameters() if p.requires_grad],
            lr=settings.bc_actor_learning_rate,
            weight_decay=settings.bc_weight_decay,
        )
    else:
        stage1_path = settings.output_dir / "checkpoints" / "bc_stage1_best.pt"
        if not stage1_path.is_file():
            raise FileNotFoundError(
                f"BC stage 2 must start from the frozen-backbone stage-1 policy: "
                f"{stage1_path}. Run --phase bc_stage1 first."
            )
        policy.actor.load_state_dict(torch_load(stage1_path)["actor_state_dict"])
        policy.backbone.requires_grad_(True)
        policy.actor.requires_grad_(True)
        policy.actor.log_std.requires_grad_(False)
        epochs = settings.bc_stage2_epochs
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": policy.actor.parameters(),
                    "lr": settings.bc_actor_learning_rate,
                },
                {
                    "params": policy.backbone.recombiner.parameters(),
                    "lr": settings.bc_linear_learning_rate,
                },
                {
                    "params": policy.backbone.core.parameters(),
                    "lr": settings.bc_optical_learning_rate,
                },
            ],
            weight_decay=settings.bc_weight_decay,
        )
    report = trainable_parameter_report(policy)
    report.update(
        {
            "stage": stage,
            "frozen_optical_backbone": stage == 1,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
    )
    write_json(
        settings.output_dir / "metrics" / f"bc_stage{stage}_model.json", report
    )
    history = settings.output_dir / "metrics" / f"bc_stage{stage}_history.csv"
    best = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = _bc_epoch(
            policy,
            processor,
            train_loader,
            settings,
            device,
            optimizer=optimizer,
            epoch=epoch,
        )
        validation_metrics = _bc_epoch(
            policy,
            processor,
            validation_loader,
            settings,
            device,
            optimizer=None,
            epoch=epoch,
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "epoch_time_sec": time.perf_counter() - started,
        }
        append_csv(history, row)
        payload = {
            "stage": stage,
            "epoch": epoch,
            "backbone": policy.backbone.checkpoint_state(),
            "actor_state_dict": policy.actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
        }
        last_path = settings.output_dir / "checkpoints" / f"bc_stage{stage}_last.pt"
        atomic_torch_save(last_path, payload)
        if validation_metrics["control_mae"] < best:
            best = validation_metrics["control_mae"]
            best_epoch = epoch
            atomic_torch_save(
                settings.output_dir / "checkpoints" / f"bc_stage{stage}_best.pt",
                payload,
            )
            if stage == 2:
                atomic_torch_save(
                    settings.output_dir / "checkpoints" / "bc_policy_best.pt",
                    payload,
                )
        print(
            f"[bc_stage{stage}] epoch={epoch:03d} "
            f"train_MAE={train_metrics['control_mae']:.5f} "
            f"validation_MAE={validation_metrics['control_mae']:.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    summary = {
        "stage": stage,
        "best_epoch": best_epoch,
        "best_validation_control_mae": best,
        "checkpoint": str(
            settings.output_dir / "checkpoints" / f"bc_stage{stage}_best.pt"
        ),
    }
    write_json(
        settings.output_dir / "metrics" / f"bc_stage{stage}_summary.json", summary
    )
    return summary


@torch.no_grad()
def evaluate_bc(
    policy: OpticalDrivingPolicy,
    processor: Any,
    records: list[Any],
    settings: Any,
    device: torch.device,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    if checkpoint is not None:
        payload = torch_load(checkpoint)
        policy.backbone.core.load_state_dict(
            payload["backbone"]["core_state_dict"]
        )
        policy.backbone.recombiner.load_state_dict(
            payload["backbone"]["recombiner_state_dict"]
        )
        policy.actor.load_state_dict(payload["actor_state_dict"])
    metrics = _bc_epoch(
        policy,
        processor,
        bench_loader(records, settings, training=False),
        settings,
        device,
        optimizer=None,
        epoch=0,
    )
    write_json(settings.output_dir / "metrics" / "bc_evaluation.json", metrics)
    return metrics


def _bc_epoch(
    policy: OpticalDrivingPolicy,
    processor: Any,
    loader: DataLoader,
    settings: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    policy.train(training)
    policy.backbone.visual.eval()
    total_loss = 0.0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    samples = 0
    for batch_index, batch in enumerate(loader, 1):
        inputs = preprocess_vision(processor, batch["images"], device)
        controls = batch["controls"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            visual = policy.encode(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            state = policy.state(
                visual,
                batch["speed"].to(device),
                batch["command"].to(device),
                batch["target_point"].to(device),
            )
            normalized = policy.actor.forward_normalized(state)
            loss, _ = behavior_cloning_loss(
                normalized,
                controls,
                steer_weight=settings.bc_steer_weight,
                throttle_weight=settings.bc_throttle_weight,
                brake_weight=settings.bc_brake_weight,
                exclusion_weight=settings.bc_exclusion_weight,
            )
            router_balance, router_importance = policy.backbone.router_losses()
            loss = (
                loss
                + settings.bc_router_balance_weight * router_balance.float()
                + settings.bc_router_importance_weight
                * router_importance.float()
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
        decoded = decode_normalized_action(normalized.detach()).cpu()
        predictions.append(decoded)
        targets.append(controls.detach().cpu())
        batch_size = controls.shape[0]
        samples += batch_size
        total_loss += float(loss.detach()) * batch_size
        # Routing is input-dependent even while the backbone is frozen. Keep
        # these values in the log so expert collapse is visible during both BC
        # stages.
        balance_value = float(router_balance.detach())
        importance_value = float(router_importance.detach())
        if training and batch_index % settings.log_interval_batches == 0:
            print(
                f"[bc] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={float(loss):.5f}",
                flush=True,
            )
    if not samples:
        raise RuntimeError("Bench2Drive behavior-cloning loader is empty")
    metrics = control_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["loss"] = total_loss / samples
    metrics["samples"] = samples
    metrics["router_balance_last_batch"] = balance_value
    metrics["router_importance_last_batch"] = importance_value
    return metrics
