from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from .datasets_bench2drive import Bench2DriveBCDataset, collate_bench2drive
from .io_utils import append_csv, atomic_torch_save, torch_load, write_json
from .modeling import (
    DrivingActor,
    OpticalDrivingPolicy,
    decode_normalized_action,
    preprocess_vision,
    trainable_parameter_report,
)
from .objectives import behavior_cloning_loss
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)


class CommandBalancedEpochSampler(Sampler[int]):
    """Deterministic rotating windows over the six navigation commands.

    A command with fewer than ``max_per_command`` records is used in full once
    per epoch. Larger commands advance by one non-overlapping window each
    epoch, wrapping only at the end. The merged window is shuffled
    deterministically, so interrupted epochs can be replayed exactly.
    """

    def __init__(
        self,
        records: list[Any],
        max_per_command: int,
        seed: int,
    ) -> None:
        self.max_per_command = int(max_per_command)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[int(record.command)].append(index)
        self.groups: dict[int, list[int]] = {}
        for command, indices in sorted(grouped.items()):
            generator = torch.Generator().manual_seed(self.seed + 10_007 * command)
            order = torch.randperm(len(indices), generator=generator).tolist()
            self.groups[command] = [indices[position] for position in order]
        self.command_counts = {
            command: len(indices) for command, indices in self.groups.items()
        }
        self.samples_per_epoch = sum(
            min(self.max_per_command, count)
            for count in self.command_counts.values()
        )
        self.full_coverage_epochs = max(
            math.ceil(count / self.max_per_command)
            for count in self.command_counts.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        selected: list[int] = []
        for indices in self.groups.values():
            count = len(indices)
            take = min(self.max_per_command, count)
            start = (self.epoch * self.max_per_command) % count
            end = start + take
            if end <= count:
                selected.extend(indices[start:end])
            else:
                selected.extend(indices[start:])
                selected.extend(indices[: end - count])
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter([selected[position] for position in order])

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "deterministic_rotating_command_balanced",
            "max_samples_per_command_per_epoch": self.max_per_command,
            "command_counts": {
                str(command): count
                for command, count in self.command_counts.items()
            },
            "samples_per_epoch": self.samples_per_epoch,
            "full_coverage_epochs": self.full_coverage_epochs,
        }


def bench_loader(
    records: list[Any], settings: Any, *, training: bool
) -> DataLoader:
    dataset = Bench2DriveBCDataset(records, settings.image_size)
    generator = torch.Generator()
    generator.manual_seed(settings.random_seed)
    sampler = None
    if training and settings.bc_samples_per_command_per_epoch is not None:
        sampler = CommandBalancedEpochSampler(
            records,
            settings.bc_samples_per_command_per_epoch,
            settings.random_seed,
        )
    return DataLoader(
        dataset,
        batch_size=settings.bc_batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
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
    train_optical = stage == 2 or settings.bc_train_optical_from_stage1
    _set_policy_trainability(policy, train_optical=train_optical)
    if stage == 1:
        epochs = settings.bc_stage1_epochs
        optimizer = _build_bc_optimizer(
            policy,
            settings,
            train_optical=train_optical,
            optical_learning_rate=settings.bc_stage1_optical_learning_rate,
            phase_learning_rate=settings.bc_stage1_phase_learning_rate,
            router_learning_rate=settings.bc_stage1_router_learning_rate,
        )
    else:
        stage1_path = settings.output_dir / "checkpoints" / "bc_stage1_best.pt"
        if not stage1_path.is_file():
            raise FileNotFoundError(
                f"BC stage 2 must start from the complete stage-1 policy: "
                f"{stage1_path}. Run --phase bc_stage1 first."
            )
        stage1_payload = torch_load(stage1_path)
        policy.backbone.core.load_state_dict(
            stage1_payload["backbone"]["core_state_dict"]
        )
        policy.backbone.recombiner.load_state_dict(
            stage1_payload["backbone"]["recombiner_state_dict"]
        )
        policy.actor.load_state_dict(stage1_payload["actor_state_dict"])
        epochs = settings.bc_stage2_epochs
        optimizer = _build_bc_optimizer(
            policy,
            settings,
            train_optical=True,
            optical_learning_rate=settings.bc_stage2_optical_learning_rate,
            phase_learning_rate=settings.bc_stage2_phase_learning_rate,
            router_learning_rate=settings.bc_stage2_router_learning_rate,
        )
    completed = _completed_stage_summary(settings, stage, epochs)
    if settings.bc_resume and completed is not None:
        print(
            f"[bc_stage{stage}] already completed through epoch {epochs}; "
            f"reusing {completed['checkpoint']}",
            flush=True,
        )
        return completed
    report = trainable_parameter_report(policy)
    sampling_report = (
        train_loader.sampler.report()
        if isinstance(train_loader.sampler, CommandBalancedEpochSampler)
        else {
            "strategy": "full_dataset_shuffle",
            "samples_per_epoch": len(train_records),
            "full_coverage_epochs": 1,
        }
    )
    report.update(
        {
            "stage": stage,
            "frozen_optical_backbone": not train_optical,
            # ``visual.blocks[0]`` owns the trainable optical replacement, so
            # scanning visual.parameters() would incorrectly report Qwen as
            # trainable. Check the retained native modules explicitly.
            "qwen_vision_frozen": all(
                not parameter.requires_grad
                for module in [
                    policy.backbone.visual.patch_embed,
                    *policy.backbone.original_blocks,
                ]
                for parameter in module.parameters()
            ),
            "backbone_initialization": settings.bc_backbone_initialization,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "sampling": sampling_report,
            "gradient_clip_norm": settings.bc_gradient_clip_norm,
            "checkpoint_interval_batches": settings.bc_checkpoint_interval_batches,
        }
    )
    write_json(
        settings.output_dir / "metrics" / f"bc_stage{stage}_model.json", report
    )
    history = settings.output_dir / "metrics" / f"bc_stage{stage}_history.csv"
    best = float("inf")
    best_epoch = 0
    resume_epoch, resume_batch, resume_partial = _load_step_checkpoint(
        policy, optimizer, settings, stage
    )
    print(
        f"[bc_stage{stage}] sampling={sampling_report} batch_size="
        f"{settings.bc_batch_size} resume_epoch={resume_epoch} "
        f"resume_batch={resume_batch}",
        flush=True,
    )
    for epoch in range(resume_epoch, epochs + 1):
        started = time.perf_counter()
        if isinstance(train_loader.sampler, CommandBalancedEpochSampler):
            global_epoch = (
                (stage - 1) * settings.bc_stage1_epochs + epoch - 1
            )
            train_loader.sampler.set_epoch(global_epoch)
        step_path = (
            settings.output_dir
            / "checkpoints"
            / f"bc_stage{stage}_step_last.pt"
        )

        def save_step(batch_index: int, partial: dict[str, Any]) -> None:
            atomic_torch_save(
                step_path,
                {
                    "stage": stage,
                    "epoch": epoch,
                    "batch": batch_index,
                    "backbone": policy.backbone.checkpoint_state(),
                    "actor_state_dict": policy.actor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "partial_metrics": partial,
                    "sampling": sampling_report,
                },
            )

        if not (epoch == resume_epoch and resume_batch > 0):
            save_step(0, {})
        train_metrics = _bc_epoch(
            policy,
            processor,
            train_loader,
            settings,
            device,
            optimizer=optimizer,
            epoch=epoch,
            skip_batches=(resume_batch if epoch == resume_epoch else 0),
            initial_partial=(resume_partial if epoch == resume_epoch else None),
            checkpoint_callback=save_step,
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
        phase_statistics = _phase_statistics(policy)
        row = {
            "stage": stage,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "epoch_time_sec": time.perf_counter() - started,
            **phase_statistics,
        }
        append_csv(history, row)
        _save_phase_preview(
            policy,
            settings.output_dir / "figures" / f"bc_stage{stage}_phase_latest.png",
            title=f"BC stage {stage}, epoch {epoch}",
        )
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
        step_path.unlink(missing_ok=True)
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
            _save_phase_preview(
                policy,
                settings.output_dir / "figures" / f"bc_stage{stage}_phase_best.png",
                title=f"BC stage {stage} best, epoch {epoch}",
            )
        print(
            f"[bc_stage{stage}] epoch={epoch:03d} "
            f"train_MAE={train_metrics['control_mae']:.5f} "
            f"validation_MAE={validation_metrics['control_mae']:.5f} "
            f"phase_std={phase_statistics['phase_physical_std_rad']:.5f}rad "
            f"phase_delta_rms={phase_statistics['phase_delta_from_zero_init_rms_rad']:.5f}rad "
            f"phase_grad_rms={train_metrics['phase_gradient_rms_mean']:.3e} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        resume_batch = 0
        resume_partial = None
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


def _set_policy_trainability(
    policy: OpticalDrivingPolicy, *, train_optical: bool
) -> None:
    """Train only deployable optics/electronics; native Qwen remains frozen."""
    policy.backbone.visual.requires_grad_(False).eval()
    policy.backbone.core.requires_grad_(train_optical)
    policy.backbone.recombiner.requires_grad_(train_optical)
    policy.actor.requires_grad_(True)
    # SAC uses this parameter, deterministic behavior cloning does not.
    policy.actor.log_std.requires_grad_(False)


def _build_bc_optimizer(
    policy: OpticalDrivingPolicy,
    settings: Any,
    *,
    train_optical: bool,
    optical_learning_rate: float,
    phase_learning_rate: float | None = None,
    router_learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in policy.actor.parameters()
                if parameter.requires_grad
            ],
            "lr": settings.bc_actor_learning_rate,
            "group_name": "actor",
        }
    ]
    if train_optical:
        core_named = [
            (name, parameter)
            for name, parameter in policy.backbone.core.named_parameters()
            if parameter.requires_grad
        ]
        phase_parameters = [
            parameter for name, parameter in core_named if "raw_phase" in name
        ]
        router_parameters = [
            parameter for name, parameter in core_named if name.startswith("router.")
        ]
        excluded = {id(parameter) for parameter in phase_parameters + router_parameters}
        optical_parameters = [
            parameter
            for _name, parameter in core_named
            if id(parameter) not in excluded
        ]
        if not phase_parameters or not router_parameters or not optical_parameters:
            raise RuntimeError(
                "Could not split Optical core into adapter/OEO, phase, and router groups"
            )
        groups.extend(
            [
                {
                    "params": [
                        parameter
                        for parameter in policy.backbone.recombiner.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": settings.bc_linear_learning_rate,
                    "group_name": "ccd_recombiner",
                },
                {
                    "params": optical_parameters,
                    "lr": optical_learning_rate,
                    "group_name": "optical_adapters_oeo",
                },
                {
                    "params": phase_parameters,
                    "lr": (
                        optical_learning_rate
                        if phase_learning_rate is None
                        else phase_learning_rate
                    ),
                    "group_name": "optical_phases",
                },
                {
                    "params": router_parameters,
                    "lr": (
                        optical_learning_rate
                        if router_learning_rate is None
                        else router_learning_rate
                    ),
                    "group_name": "optical_router",
                },
            ]
        )
    if any(not group["params"] for group in groups):
        raise RuntimeError("Behavior-cloning optimizer contains an empty parameter group")
    return torch.optim.AdamW(groups, weight_decay=settings.bc_weight_decay)


def _load_step_checkpoint(
    policy: OpticalDrivingPolicy,
    optimizer: torch.optim.Optimizer,
    settings: Any,
    stage: int,
) -> tuple[int, int, dict[str, Any] | None]:
    if not settings.bc_resume:
        return 1, 0, None
    path = (
        settings.output_dir
        / "checkpoints"
        / f"bc_stage{stage}_step_last.pt"
    )
    if not path.is_file():
        return 1, 0, None
    payload = torch_load(path)
    if int(payload.get("stage", -1)) != stage:
        raise RuntimeError(f"Invalid BC step checkpoint stage in {path}")
    policy.backbone.core.load_state_dict(
        payload["backbone"]["core_state_dict"]
    )
    policy.backbone.recombiner.load_state_dict(
        payload["backbone"]["recombiner_state_dict"]
    )
    policy.actor.load_state_dict(payload["actor_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    epoch = int(payload["epoch"])
    batch = int(payload["batch"])
    print(
        f"Resuming BC stage {stage} from {path}: epoch={epoch}, batch={batch}",
        flush=True,
    )
    return epoch, batch, payload.get("partial_metrics") or None


def _completed_stage_summary(
    settings: Any,
    stage: int,
    expected_epochs: int,
) -> dict[str, Any] | None:
    last_path = (
        settings.output_dir / "checkpoints" / f"bc_stage{stage}_last.pt"
    )
    best_path = (
        settings.output_dir / "checkpoints" / f"bc_stage{stage}_best.pt"
    )
    if not last_path.is_file() or not best_path.is_file():
        return None
    payload = torch_load(last_path)
    if int(payload.get("stage", -1)) != stage:
        return None
    if int(payload.get("epoch", 0)) < expected_epochs:
        return None
    best_payload = torch_load(best_path)
    metrics = best_payload.get("metrics", {})
    return {
        "stage": stage,
        "best_epoch": int(best_payload.get("epoch", expected_epochs)),
        "best_validation_control_mae": float(
            metrics.get("validation_control_mae", float("nan"))
        ),
        "checkpoint": str(best_path),
        "resumed_completed_stage": True,
    }


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
    skip_batches: int = 0,
    initial_partial: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    policy.train(training)
    policy.backbone.visual.eval()
    partial = initial_partial or {}
    total_loss = float(partial.get("total_loss", 0.0))
    samples = int(partial.get("samples", 0))
    error_sum = torch.tensor(
        partial.get("control_abs_error_sum", [0.0, 0.0, 0.0]),
        dtype=torch.float64,
    )
    balance_value = float(partial.get("router_balance_last_batch", 0.0))
    importance_value = float(partial.get("router_importance_last_batch", 0.0))
    phase_gradient_rms_sum = float(partial.get("phase_gradient_rms_sum", 0.0))
    phase_gradient_max = float(partial.get("phase_gradient_max", 0.0))
    gradient_batches = int(partial.get("gradient_batches", 0))
    for batch_index, batch in enumerate(loader, 1):
        if batch_index <= skip_batches:
            continue
        inputs = preprocess_vision(processor, batch["images"], device)
        controls = batch["controls"].to(device, non_blocking=True)
        _require_finite("pixel_values", inputs["pixel_values"], batch)
        _require_finite("controls", controls, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            try:
                visual = policy.encode(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Optical BC forward failed for sample_ids="
                    f"{batch['sample_ids']}: {exc}"
                ) from exc
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
            dc = (
                phase_dc_loss(policy.backbone)
                if training and settings.bc_phase_dc_weight > 0.0
                else loss.new_zeros(())
            )
            loss = (
                loss
                + settings.bc_router_balance_weight * router_balance.float()
                + settings.bc_router_importance_weight
                * router_importance.float()
                + settings.bc_phase_dc_weight * dc
            )
            _require_finite("behavior-cloning loss", loss, batch)
        if training:
            loss.backward()
            named_parameters = [
                (name, parameter)
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            ]
            nonfinite_gradients = [
                name
                for name, parameter in named_parameters
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_gradients:
                raise RuntimeError(
                    "Non-finite BC gradients before optimizer.step for "
                    f"parameters={nonfinite_gradients[:8]}, "
                    f"sample_ids={batch['sample_ids']}. The optimizer was not stepped."
                )
            phase_gradients = [
                parameter.grad.detach().float()
                for name, parameter in named_parameters
                if "raw_phase" in name and parameter.grad is not None
            ]
            if phase_gradients:
                phase_elements = sum(value.numel() for value in phase_gradients)
                phase_square_sum = sum(value.square().sum() for value in phase_gradients)
                batch_phase_rms = float(torch.sqrt(phase_square_sum / phase_elements))
                batch_phase_max = max(float(value.abs().max()) for value in phase_gradients)
                phase_gradient_rms_sum += batch_phase_rms
                phase_gradient_max = max(phase_gradient_max, batch_phase_max)
                gradient_batches += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named_parameters],
                settings.bc_gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            nonfinite_parameters = [
                name
                for name, parameter in named_parameters
                if not torch.isfinite(parameter).all()
            ]
            if nonfinite_parameters:
                raise RuntimeError(
                    "BC optimizer.step produced non-finite parameters="
                    f"{nonfinite_parameters[:8]}, sample_ids={batch['sample_ids']}. "
                    "Resume from the latest step checkpoint with a lower learning rate."
                )
        decoded = decode_normalized_action(normalized.detach()).cpu()
        target_cpu = controls.detach().cpu()
        batch_size = controls.shape[0]
        samples += batch_size
        total_loss += float(loss.detach()) * batch_size
        error_sum += (decoded.float() - target_cpu.float()).abs().double().sum(0)
        # Routing is input-dependent even while the backbone is frozen. Keep
        # these values in the log so expert collapse is visible during both BC
        # stages.
        balance_value = float(router_balance.detach())
        importance_value = float(router_importance.detach())
        running_loss = total_loss / samples
        if training and batch_index % settings.log_interval_batches == 0:
            print(
                f"[bc] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"batch_loss={float(loss):.5f} mean_loss={running_loss:.5f} "
                f"grad_norm={float(gradient_norm):.5f}",
                flush=True,
            )
        if (
            training
            and checkpoint_callback is not None
            and batch_index % settings.bc_checkpoint_interval_batches == 0
        ):
            checkpoint_callback(
                batch_index,
                _partial_metrics(
                    total_loss,
                    samples,
                    error_sum,
                    balance_value,
                    importance_value,
                    phase_gradient_rms_sum,
                    phase_gradient_max,
                    gradient_batches,
                ),
            )
    if not samples:
        raise RuntimeError("Bench2Drive behavior-cloning loader is empty")
    mean_error = error_sum / samples
    metrics = {
        "steer_mae": float(mean_error[0]),
        "throttle_mae": float(mean_error[1]),
        "brake_mae": float(mean_error[2]),
        "control_mae": float(mean_error.mean()),
    }
    metrics["loss"] = total_loss / samples
    metrics["samples"] = samples
    metrics["router_balance_last_batch"] = balance_value
    metrics["router_importance_last_batch"] = importance_value
    metrics["phase_gradient_rms_mean"] = (
        phase_gradient_rms_sum / gradient_batches if gradient_batches else 0.0
    )
    metrics["phase_gradient_max"] = phase_gradient_max
    return metrics


def _partial_metrics(
    total_loss: float,
    samples: int,
    error_sum: torch.Tensor,
    balance_value: float,
    importance_value: float,
    phase_gradient_rms_sum: float,
    phase_gradient_max: float,
    gradient_batches: int,
) -> dict[str, Any]:
    return {
        "total_loss": total_loss,
        "samples": samples,
        "control_abs_error_sum": error_sum.tolist(),
        "router_balance_last_batch": balance_value,
        "router_importance_last_batch": importance_value,
        "phase_gradient_rms_sum": phase_gradient_rms_sum,
        "phase_gradient_max": phase_gradient_max,
        "gradient_batches": gradient_batches,
    }


def _require_finite(name: str, tensor: torch.Tensor, batch: dict[str, Any]) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite {name} for sample_ids={batch['sample_ids']}; "
            "the optimizer was not stepped for this batch."
        )


@torch.no_grad()
def _phase_statistics(policy: OpticalDrivingPolicy) -> dict[str, float]:
    layers = [
        expert.raw_phase
        for plane in policy.backbone.core.expert_layers
        for expert in plane.experts
    ]
    layers.append(policy.backbone.core.global_phase.phase.raw_phase)
    count = sum(parameter.numel() for parameter in layers)
    raw_abs_sum = sum(parameter.float().abs().sum() for parameter in layers)
    raw_square_sum = sum(parameter.float().square().sum() for parameter in layers)
    physical_delta_square_sum = sum(
        (
            2.0 * math.pi * torch.sigmoid(parameter.float()) - math.pi
        ).square().sum()
        for parameter in layers
    )
    physical = torch.cat(
        [2.0 * math.pi * torch.sigmoid(parameter.float()).reshape(-1) for parameter in layers]
    )
    per_layer_delta_rms = [
        float(
            torch.sqrt(
                (2.0 * math.pi * torch.sigmoid(parameter.float()) - math.pi)
                .square()
                .mean()
            )
        )
        for parameter in layers
    ]
    return {
        "phase_raw_abs_mean": float(raw_abs_sum / count),
        "phase_raw_rms": float(torch.sqrt(raw_square_sum / count)),
        "phase_delta_from_zero_init_rms_rad": float(
            torch.sqrt(physical_delta_square_sum / count)
        ),
        "phase_physical_min_rad": float(physical.min()),
        "phase_physical_max_rad": float(physical.max()),
        "phase_physical_std_rad": float(physical.std()),
        "phase_planes_delta_rms_gt_0p01": float(
            sum(value > 0.01 for value in per_layer_delta_rms)
        ),
    }


@torch.no_grad()
def _save_phase_preview(
    policy: OpticalDrivingPolicy, path: Path, *, title: str
) -> None:
    """Save physical expert/global phase with a fixed 0..2pi color scale."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    expert_phases = [
        2.0 * math.pi * torch.sigmoid(expert.raw_phase.detach().float()).cpu()
        for plane in policy.backbone.core.expert_layers
        for expert in plane.experts
    ]
    global_phase = (
        2.0
        * math.pi
        * torch.sigmoid(
            policy.backbone.core.global_phase.phase.raw_phase.detach().float()
        )
    ).cpu()
    figure, axes = plt.subplots(4, 5, figsize=(16, 13), constrained_layout=True)
    axes_flat = list(axes.reshape(-1))
    image = None
    for index, phase in enumerate(expert_phases):
        image = axes_flat[index].imshow(
            phase.numpy(), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi
        )
        axes_flat[index].set_title(f"expert {index}")
        axes_flat[index].set_xlabel("x pixel")
        axes_flat[index].set_ylabel("y pixel")
    global_axis = axes_flat[len(expert_phases)]
    image = global_axis.imshow(
        global_phase.numpy(), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi
    )
    global_axis.set_title("global phase")
    global_axis.set_xlabel("x pixel")
    global_axis.set_ylabel("y pixel")
    for axis in axes_flat[len(expert_phases) + 1 :]:
        axis.axis("off")
    figure.suptitle(title)
    if image is not None:
        figure.colorbar(image, ax=axes_flat, label="physical phase [rad]", shrink=0.72)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
