from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_pretraining import (
    SubsetEpochViewSampler,
    stratified_base_indices,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
    load_imagenet,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
    load_settings as load_imagenet_settings,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    Context,
    atomic_save,
    load_config,
    make_loader,
    mix_batch,
    reduce_metrics,
    resolve_path,
    seed_all,
    topk_counts,
    write_json,
)
from experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.p11_matched_continue import (
    initialize_p11_epoch88_control,
)
from experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.train import (
    canonical_sha256,
    gather_rng_states,
    restore_rng_state,
    sha256_file,
    sha256_tensor,
    training_implementation_manifest,
)

from .model import QwenStemSeparableOpticalImageNetBackbone


CHECKPOINT_FORMAT = "p11-large-scale-supervised-continuation-v1"
BACKBONE_FORMAT = "p11-large-scale-supervised-backbone-v1"
IMPLEMENTATION_FILES = (
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/large_scale_continue.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/stem.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/train.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/p11_matched_continue.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/train.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/datasets.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/settings.py",
)


def unwrap(model: nn.Module) -> LargeScaleP11Model:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(core, LargeScaleP11Model):
        raise TypeError(f"Expected LargeScaleP11Model, got {type(core).__name__}")
    return core


class LargeScaleP11Model(QwenStemSeparableOpticalImageNetBackbone):
    """P11 with training-only, parameter-free stochastic depth.

    The state-dict is intentionally identical to P11.  This lets the recipe
    strict-load the locked epoch-88 source without changing the deployable
    backbone.  Stochastic depth is disabled automatically in evaluation.
    """

    def __init__(self, stem_checkpoint: str | Path, config: dict[str, Any]) -> None:
        super().__init__(stem_checkpoint, config)
        self.stage_drop_path_rate = float(config.get("stage_drop_path_rate", 0.0))
        if not 0.0 <= self.stage_drop_path_rate < 1.0:
            raise ValueError("stage_drop_path_rate must lie in [0,1)")

    def stage_drop_probabilities(self) -> list[float]:
        if self.num_stages <= 1:
            return [self.stage_drop_path_rate]
        return [
            self.stage_drop_path_rate * index / (self.num_stages - 1)
            for index in range(self.num_stages)
        ]

    @staticmethod
    def _drop_stage(
        previous: torch.Tensor,
        updated: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return updated
        keep = 1.0 - probability
        mask = torch.empty(
            (updated.shape[0], 1, 1, 1),
            dtype=updated.dtype,
            device=updated.device,
        ).bernoulli_(keep)
        return previous + (updated - previous) * mask / keep

    def forward_features(
        self,
        images: torch.Tensor,
        *,
        ablation: str = "normal",
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if ablation not in {
            "normal",
            "optical_off",
            "phase_random",
            "electronic_skip_off",
        }:
            raise ValueError(f"Unsupported ablation: {ablation}")
        amplitude, _ = self.optical_input(images)
        outputs: list[torch.Tensor] = []
        probabilities = self.stage_drop_probabilities()
        for index, stage in enumerate(self.stages):
            previous = amplitude
            amplitude = stage(
                amplitude,
                phase_override=(
                    stage.random_phase if ablation == "phase_random" else None
                ),
                optical_off=ablation == "optical_off",
                disable_electronic_skip=ablation == "electronic_skip_off",
            )
            if self.training and ablation == "normal":
                amplitude = self._drop_stage(
                    previous,
                    amplitude,
                    probabilities[index],
                )
            outputs.append(amplitude)
        return amplitude, tuple(outputs)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "training_only_stochastic_depth": True,
                "stage_drop_path_rate": self.stage_drop_path_rate,
                "stage_drop_probabilities": self.stage_drop_probabilities(),
                "stochastic_depth_adds_deployable_parameters": False,
            }
        )
        return report


def _no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    return (
        parameter.ndim <= 1
        or lowered.endswith(".bias")
        or "norm" in lowered
        or "logit" in lowered
    )


def build_layerwise_optimizer(
    model: LargeScaleP11Model,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    """AdamW with ViT/ConvNeXt-style layer-wise learning-rate decay."""

    values = config["optimizer"]
    decay = float(values.get("layer_decay", 0.90))
    if not 0.0 < decay <= 1.0:
        raise ValueError("optimizer.layer_decay must lie in (0,1]")
    weight_decay = float(values.get("weight_decay", 0.05))
    phase_lr = float(values.get("phase_learning_rate", 3.5e-3))
    electronic_lr = float(values.get("electronic_learning_rate", 2.5e-4))
    adapter_lr = float(values.get("adapter_learning_rate", electronic_lr))
    head_lr = float(values.get("head_learning_rate", 5.0e-4))
    for name, value in (
        ("phase_learning_rate", phase_lr),
        ("electronic_learning_rate", electronic_lr),
        ("adapter_learning_rate", adapter_lr),
        ("head_learning_rate", head_lr),
    ):
        if value <= 0.0:
            raise ValueError(f"optimizer.{name} must be positive")

    groups: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    assigned: dict[int, str] = {}

    def add_group(
        name: str,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
        *,
        learning_rate: float,
        decay_weights: bool,
        depth: int,
    ) -> None:
        parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
        if not parameters:
            return
        for parameter in parameters:
            identity = id(parameter)
            if identity in assigned:
                raise RuntimeError(
                    f"Parameter occurs in both {assigned[identity]!r} and {name!r}"
                )
            assigned[identity] = name
        group_decay = weight_decay if decay_weights else 0.0
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": group_decay,
                "name": name,
            }
        )
        schema.append(
            {
                "name": name,
                "depth": int(depth),
                "parameter_tensors": len(parameters),
                "parameter_elements": sum(p.numel() for p in parameters),
                "learning_rate": float(learning_rate),
                "weight_decay": float(group_decay),
            }
        )

    def split_and_add(
        prefix: str,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
        *,
        learning_rate: float,
        depth: int,
    ) -> None:
        decayed = [(name, p) for name, p in named_parameters if not _no_weight_decay(name, p)]
        exempt = [(name, p) for name, p in named_parameters if _no_weight_decay(name, p)]
        add_group(
            f"{prefix}.decay",
            decayed,
            learning_rate=learning_rate,
            decay_weights=True,
            depth=depth,
        )
        add_group(
            f"{prefix}.no_decay",
            exempt,
            learning_rate=learning_rate,
            decay_weights=False,
            depth=depth,
        )

    stage_count = len(model.stages)
    adapter_scale = decay ** (stage_count + 1)
    split_and_add(
        "adapter",
        list(model.adapter.named_parameters(prefix="adapter")),
        learning_rate=adapter_lr * adapter_scale,
        depth=0,
    )
    for index, stage in enumerate(model.stages):
        scale = decay ** (stage_count - 1 - index)
        add_group(
            f"stage{index:02d}.phase",
            [(f"stages.{index}.raw_phase", stage.raw_phase)],
            learning_rate=phase_lr * scale,
            decay_weights=False,
            depth=index + 1,
        )
        electronics = [
            (f"stages.{index}.{name}", parameter)
            for name, parameter in stage.named_parameters()
            if parameter is not stage.raw_phase
        ]
        split_and_add(
            f"stage{index:02d}.electronic",
            electronics,
            learning_rate=electronic_lr * scale,
            depth=index + 1,
        )
    split_and_add(
        "readout",
        list(model.readout.named_parameters(prefix="readout")),
        learning_rate=head_lr,
        depth=stage_count + 1,
    )

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(trainable[identity] for identity in trainable.keys() - assigned.keys())
    extra = sorted(assigned[identity] for identity in assigned.keys() - trainable.keys())
    if missing or extra:
        raise RuntimeError(
            "Layer-wise groups do not partition trainable parameters exactly: "
            f"missing={missing}, extra={extra}"
        )
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in values.get("betas", [0.9, 0.999])),
        eps=float(values.get("eps", 1.0e-8)),
    )
    return optimizer, schema


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    updates_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    values = config["optimizer"]
    total = max(int(config["training"]["epochs"]) * updates_per_epoch, 1)
    warmup = int(values.get("warmup_epochs", 2)) * updates_per_epoch
    minimum = float(values.get("minimum_learning_rate_ratio", 0.05))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must lie in [0,1]")

    def scale(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1.0 / warmup)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class TrainableEMA:
    """EMA only for trainable tensors; frozen Qwen/optical buffers are shared."""

    def __init__(self, model: nn.Module, decay: float, warmup_updates: int) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema.decay must lie in [0,1)")
        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        if self.warmup_updates > 0:
            ramp = 1.0 - math.exp(-self.updates / self.warmup_updates)
            decay = self.decay * ramp
        else:
            decay = self.decay
        live = dict(model.named_parameters())
        if not self.shadow.keys() <= live.keys():
            raise RuntimeError("EMA state contains a parameter absent from the model")
        for name, average in self.shadow.items():
            average.lerp_(live[name].detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_updates": self.warmup_updates,
            "updates": self.updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise RuntimeError("EMA decay differs from the checkpoint")
        if int(state["warmup_updates"]) != self.warmup_updates:
            raise RuntimeError("EMA warmup differs from the checkpoint")
        shadow = dict(state["shadow"])
        if shadow.keys() != self.shadow.keys():
            raise RuntimeError("EMA tensor names differ from the checkpoint")
        for name in self.shadow:
            if tuple(shadow[name].shape) != tuple(self.shadow[name].shape):
                raise RuntimeError(f"EMA tensor shape differs at {name}")
            self.shadow[name].copy_(shadow[name])
        self.updates = int(state["updates"])

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        live = dict(model.named_parameters())
        backups = {name: live[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    live[name].copy_(average)
            yield
        finally:
            with torch.no_grad():
                for name, value in backups.items():
                    live[name].copy_(value)


def random_erasing(images: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    values = config.get("augmentation", {})
    probability = float(values.get("random_erasing_probability", 0.0))
    if probability <= 0.0:
        return images
    area_min, area_max = (
        float(value) for value in values.get("random_erasing_area", [0.02, 0.20])
    )
    ratio_min, ratio_max = (
        float(value) for value in values.get("random_erasing_ratio", [0.3, 3.333333])
    )
    if not (0.0 < area_min <= area_max < 1.0 and 0.0 < ratio_min <= ratio_max):
        raise ValueError("Invalid random-erasing area or aspect-ratio range")
    output = images.clone()
    height, width = output.shape[-2:]
    total_area = height * width
    for index in range(output.shape[0]):
        if random.random() >= probability:
            continue
        for _ in range(10):
            target = total_area * random.uniform(area_min, area_max)
            ratio = math.exp(random.uniform(math.log(ratio_min), math.log(ratio_max)))
            erase_h = int(round(math.sqrt(target * ratio)))
            erase_w = int(round(math.sqrt(target / ratio)))
            if 0 < erase_h < height and 0 < erase_w < width:
                top = random.randrange(height - erase_h + 1)
                left = random.randrange(width - erase_w + 1)
                # Zero is the per-channel mean in normalized image space.
                output[index, :, top : top + erase_h, left : left + erase_w] = 0.0
                break
    return output


def classification_loss(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """CE compatibility plus the soft-target BCE used by modern vision recipes."""

    values = config.get("loss", {})
    mode = str(values.get("mode", "cross_entropy"))
    smoothing = float(values.get("label_smoothing", 0.1))
    if mode == "cross_entropy":
        loss_a = F.cross_entropy(logits, labels_a, label_smoothing=smoothing)
        loss_b = F.cross_entropy(logits, labels_b, label_smoothing=smoothing)
        return lam * loss_a + (1.0 - lam) * loss_b
    if mode != "bce_soft_targets":
        raise ValueError(f"Unsupported classification loss mode: {mode}")
    classes = logits.shape[-1]
    off = smoothing / classes
    on = 1.0 - smoothing + off
    targets_a = torch.full_like(logits, off).scatter_(1, labels_a[:, None], on)
    targets_b = torch.full_like(logits, off).scatter_(1, labels_b[:, None], on)
    targets = lam * targets_a + (1.0 - lam) * targets_b
    # Keep the standard mean reduction used by large-scale vision recipes.
    # Multiplying this value by ``classes`` would turn the zero-logit loss from
    # about 0.693 into about 693 on ImageNet-1K and invalidate the selected
    # learning rates and gradient-clipping thresholds.
    return F.binary_cross_entropy_with_logits(logits, targets)


def phase_gradient_report(model: LargeScaleP11Model) -> dict[str, Any]:
    norms: list[float] = []
    finite: list[bool] = []
    for parameter in model.phase_parameters():
        if parameter.grad is None:
            norms.append(0.0)
            finite.append(False)
        else:
            norms.append(float(parameter.grad.float().norm().detach().cpu()))
            finite.append(bool(torch.isfinite(parameter.grad).all()))
    return {
        "per_stage_l2_norm": norms,
        "all_finite": all(finite),
        "all_nonzero": all(value > 0.0 for value in norms),
    }


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    ema: TrainableEMA,
    config: Mapping[str, Any],
    context: Context,
    *,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], dict[str, Any], int]:
    model.train()
    core = unwrap(model)
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    training = config["training"]
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    limit = training.get("max_train_batches")
    effective_batches = min(len(loader), int(limit or len(loader)))
    vector = torch.zeros(5, dtype=torch.float64, device=context.device)
    started = time.perf_counter()
    use_amp = bool(training.get("use_amp", True)) and context.device.type == "cuda"
    phase_clip = float(config["optimizer"].get("phase_gradient_clip_norm", 2.0))
    electronic_clip = float(config["optimizer"].get("electronic_gradient_clip_norm", 5.0))
    phase_parameters = list(core.phase_parameters())
    phase_parameter_ids = {id(parameter) for parameter in phase_parameters}
    electronic_parameters = [
        parameter
        for _, parameter in core.named_parameters()
        if parameter.requires_grad and id(parameter) not in phase_parameter_ids
    ]
    gradient_report: dict[str, Any] | None = None
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > effective_batches:
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        images = random_erasing(images, config)
        images, labels_a, labels_b, lam, _ = mix_batch(images, labels, dict(config))
        window_start = ((batch_index - 1) // accumulation) * accumulation + 1
        window_end = min(window_start + accumulation - 1, effective_batches)
        window_size = window_end - window_start + 1
        update_now = batch_index == window_end
        no_sync = (
            not update_now
            and context.world_size > 1
            and callable(getattr(model, "no_sync", None))
        )
        sync_scope = model.no_sync() if no_sync else _null_scope()
        with sync_scope:
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = classification_loss(logits, labels_a, labels_b, lam, config)
            scaler.scale(loss / window_size).backward()
        if update_now:
            scaler.unscale_(optimizer)
            if gradient_report is None:
                gradient_report = phase_gradient_report(core)
                if bool(training.get("require_all_phase_gradients", True)) and not (
                    gradient_report["all_finite"]
                    and gradient_report["all_nonzero"]
                ):
                    raise RuntimeError(
                        "Optical phase gradient health check failed on the first update"
                    )
            torch.nn.utils.clip_grad_norm_(phase_parameters, phase_clip)
            torch.nn.utils.clip_grad_norm_(electronic_parameters, electronic_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_succeeded = scaler.get_scale() >= scale_before
            if update_succeeded:
                scheduler.step()
                global_step += 1
                ema.update(core)
            optimizer.zero_grad(set_to_none=True)
        count = labels.numel()
        correct1, correct5 = topk_counts(logits.detach(), labels)
        vector += torch.tensor(
            [float(loss.detach()) * count, correct1, correct5, count, 1],
            dtype=torch.float64,
            device=context.device,
        )
        interval = int(training.get("log_interval_batches", 100))
        if context.is_main and (batch_index == 1 or batch_index % interval == 0):
            rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            print(
                f"[train] epoch={epoch} batch={batch_index}/{effective_batches} "
                f"loss={float(loss.detach()):.4f} step={global_step} "
                f"phase_lr_last={rates.get('stage07.phase')}",
                flush=True,
            )
    metrics = reduce_metrics(vector, time.perf_counter() - started)
    metrics["samples_per_second"] = metrics["samples"] / max(metrics["seconds"], 1e-9)
    if context.device.type == "cuda":
        metrics["peak_allocated_mib"] = torch.cuda.max_memory_allocated(
            context.device
        ) / (1024.0**2)
    if gradient_report is None:
        raise RuntimeError("No optimizer update was made in the epoch")
    return metrics, gradient_report, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    config: Mapping[str, Any],
    context: Context,
    *,
    ablation: str = "normal",
) -> dict[str, float]:
    model.eval()
    vector = torch.zeros(5, dtype=torch.float64, device=context.device)
    started = time.perf_counter()
    limit = config["training"].get("max_validation_batches")
    use_amp = bool(config["training"].get("use_amp", True)) and context.device.type == "cuda"
    for batch_index, batch in enumerate(loader, 1):
        if limit is not None and batch_index > int(limit):
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        with torch.autocast(
            device_type=context.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images, ablation=ablation)
            loss = F.cross_entropy(logits, labels)
        count = labels.numel()
        correct1, correct5 = topk_counts(logits, labels)
        vector += torch.tensor(
            [float(loss) * count, correct1, correct5, count, 1],
            dtype=torch.float64,
            device=context.device,
        )
    return reduce_metrics(vector, time.perf_counter() - started)


def _dataset_loaders(config: Mapping[str, Any], context: Context):
    training = config["training"]
    settings = load_imagenet_settings(resolve_path(config["imagenet_config"]))
    bundle = load_imagenet(settings)
    train_per_class = training.get("train_samples_per_class")
    if train_per_class is None:
        train_indices = list(range(bundle.train.base_sample_count))
    else:
        train_indices = stratified_base_indices(
            bundle.train.targets,
            int(train_per_class),
            int(training.get("seed", 2026)),
        )
    validation_per_class = training.get("validation_samples_per_class")
    if validation_per_class is None:
        validation_indices = list(range(bundle.validation.base_sample_count))
    else:
        validation_indices = stratified_base_indices(
            bundle.validation.targets,
            int(validation_per_class),
            int(training.get("seed", 2026)) + 1,
        )
    train_sampler = SubsetEpochViewSampler(
        bundle.train,
        train_indices,
        shuffle=True,
        seed=int(training.get("seed", 2026)),
        rank=context.rank,
        world_size=context.world_size,
        shuffle_block_size=training.get("shuffle_block_size", 4096),
    )
    validation_sampler = SubsetEpochViewSampler(
        bundle.validation,
        validation_indices,
        shuffle=False,
        seed=int(training.get("seed", 2026)) + 1,
        rank=context.rank,
        world_size=context.world_size,
    )
    return (
        bundle,
        make_loader(bundle.train, train_sampler, dict(config), train=True),
        make_loader(bundle.validation, validation_sampler, dict(config), train=False),
        train_sampler,
        validation_sampler,
        train_indices,
        validation_indices,
    )


def _checkpoint_payload(
    *,
    role: str,
    model: LargeScaleP11Model,
    ema: TrainableEMA,
    optimizer: torch.optim.Optimizer,
    optimizer_schema: Sequence[Mapping[str, Any]],
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_raw: float,
    best_ema: float,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    initialization: Mapping[str, Any],
    implementation: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    initial_phases_sha256: str,
    rng_states: Sequence[Mapping[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_role": role,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_schema": list(optimizer_schema),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_optimizer_step": int(global_step),
        "best_raw_top1": float(best_raw),
        "best_ema_top1": float(best_ema),
        "history": list(history),
        "config_digest": config["_config_digest"],
        "model_config": dict(config["model"]),
        "model_report": model.parameter_report(),
        "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
        "initialization": dict(initialization),
        "implementation_manifest": dict(implementation),
        "dataset_identity": dict(dataset_identity),
        "initial_phases_sha256": str(initial_phases_sha256),
        "rng_states": list(rng_states),
        "world_size": int(world_size),
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_save(path, _checkpoint_payload(**kwargs))


def _run_has_artifacts(output: Path) -> bool:
    candidates = (output / "manifest.json", output / "result.json", output / "checkpoints")
    return any(path.exists() for path in candidates)


def validate_config(config: Mapping[str, Any], context: Context) -> None:
    expected_world = config["training"].get("expected_world_size")
    if expected_world is not None and context.world_size != int(expected_world):
        raise RuntimeError(
            f"Recipe expects world_size={int(expected_world)}, got {context.world_size}"
        )
    objective = str(config.get("objective", {}).get("mode", "supervised_imagenet1k"))
    if objective != "supervised_imagenet1k":
        raise ValueError(
            "This entry is the supervised large-scale recipe; use a separate run "
            "for future masked/self-distillation objectives"
        )


def run(config: dict[str, Any], context: Context, *, resume: bool) -> None:
    validate_config(config, context)
    training = config["training"]
    seed_all(int(training.get("seed", 2026)), context.rank)
    output = resolve_path(config["output_dir"])
    last_path = output / "checkpoints" / "last.pt"
    if resume and not last_path.is_file():
        raise FileNotFoundError("--resume requires checkpoints/last.pt")
    if not resume and _run_has_artifacts(output):
        raise RuntimeError("--fresh refuses an output directory containing run artifacts")
    implementation = training_implementation_manifest(relative_paths=IMPLEMENTATION_FILES)
    (
        bundle,
        train_loader,
        validation_loader,
        train_sampler,
        validation_sampler,
        train_indices,
        validation_indices,
    ) = _dataset_loaders(config, context)
    dataset_identity = {
        "dataset_digest": bundle.digest,
        "train_indices_sha256": canonical_sha256(train_indices),
        "validation_indices_sha256": canonical_sha256(validation_indices),
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
    }
    model_config = dict(config["model"])
    model_config.setdefault("seed", int(training.get("seed", 2026)))
    model = LargeScaleP11Model(resolve_path(config["stem_checkpoint"]), model_config)
    resume_payload: Mapping[str, Any] | None = None
    initial_phases_path = output / "initial_phases.pt"
    if resume:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_payload.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint format mismatch")
        if resume_payload.get("checkpoint_role") != "last":
            raise RuntimeError("Resume checkpoint is not the last-state checkpoint")
        if resume_payload.get("config_digest") != config["_config_digest"]:
            raise RuntimeError("Resume config digest mismatch")
        if resume_payload.get("implementation_manifest") != implementation:
            raise RuntimeError("Resume implementation/runtime manifest mismatch")
        if resume_payload.get("dataset_identity") != dataset_identity:
            raise RuntimeError("Resume dataset/index identity mismatch")
        saved_stem_sha256 = resume_payload.get("stem_checkpoint_sha256")
        if saved_stem_sha256 != model.stem.checkpoint_sha256:
            raise RuntimeError("Resume stem checkpoint identity mismatch")
        model.load_state_dict(resume_payload["model"], strict=True)
        initialization = dict(resume_payload["initialization"])
        if not initial_phases_path.is_file():
            raise FileNotFoundError(
                "Exact resume requires the original continuation phase snapshot"
            )
        initial_phases = torch.load(
            initial_phases_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(initial_phases, torch.Tensor):
            raise RuntimeError("Initial phase snapshot is not a tensor")
    else:
        initialization = dict(initialize_p11_epoch88_control(model, config))
        initialization.pop(
            "continuation_schedule_restarted_for_matched_20_epoch_budget",
            None,
        )
        initialization.update(
            {
                "continuation_schedule_restarted": True,
                "continuation_target_epochs": int(training["epochs"]),
                "continuation_recipe": "large_scale_supervised",
            }
        )
        initial_phases = model.phase_snapshot()
    if tuple(initial_phases.shape) != tuple(model.phase_snapshot().shape):
        raise RuntimeError("Initial phase snapshot shape differs from the model")
    initial_phases_sha256 = sha256_tensor(initial_phases)
    if resume_payload is not None and resume_payload.get(
        "initial_phases_sha256"
    ) != initial_phases_sha256:
        raise RuntimeError("Resume initial phase snapshot identity mismatch")
    report = model.parameter_report()
    required_fraction = float(model_config.get("minimum_optical_parameter_fraction", 0.50))
    if float(report["optical_fraction_of_backbone_trainable"]) < required_fraction:
        raise RuntimeError("Optical parameter fraction is below the required threshold")
    if float(report["minimum_optical_gate"]) < 0.50:
        raise RuntimeError("Optical gate is below 0.5 before training")
    model.to(context.device)
    core = model
    optimizer, optimizer_schema = build_layerwise_optimizer(core, config)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    micro_batches = min(
        len(train_loader),
        int(training.get("max_train_batches") or len(train_loader)),
    )
    updates_per_epoch = math.ceil(micro_batches / accumulation)
    scheduler = build_scheduler(optimizer, config, updates_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training.get("use_amp", True)) and context.device.type == "cuda",
        init_scale=float(training.get("amp_initial_scale", 256.0)),
        growth_interval=int(training.get("amp_growth_interval", 100000)),
    )
    ema_values = config.get("ema", {})
    ema = TrainableEMA(
        core,
        decay=float(ema_values.get("decay", 0.9998)),
        warmup_updates=int(ema_values.get("warmup_updates", 100)),
    )
    wrapped: nn.Module = core
    if context.world_size > 1:
        wrapped = DistributedDataParallel(
            core,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    start_epoch = 1
    global_step = 0
    best_raw = -math.inf
    best_ema = -math.inf
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        if list(resume_payload["optimizer_schema"]) != optimizer_schema:
            raise RuntimeError("Resume optimizer schema mismatch")
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload["scaler"])
        ema.load_state_dict(resume_payload["ema"])
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload["global_optimizer_step"])
        best_raw = float(resume_payload["best_raw_top1"])
        best_ema = float(resume_payload["best_ema_top1"])
        history = list(resume_payload.get("history", []))
        saved_world_size = int(resume_payload.get("world_size", 1))
        if saved_world_size != context.world_size:
            raise RuntimeError(
                f"Exact resume requires world_size={saved_world_size}, got {context.world_size}"
            )
        rng_states = list(resume_payload.get("rng_states", []))
        if len(rng_states) != context.world_size:
            raise RuntimeError("Resume checkpoint has incomplete per-rank RNG states")
        restore_rng_state(rng_states[context.rank], context)
    effective_global_batch = (
        int(training["batch_size"]) * accumulation * context.world_size
    )
    expected_batch = int(training.get("expected_effective_global_batch", effective_global_batch))
    if effective_global_batch != expected_batch:
        raise RuntimeError(
            f"Expected effective batch {expected_batch}, got {effective_global_batch}"
        )
    manifest = {
        "experiment": "P11 epoch-88 large-scale supervised continuation",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config": config["_config_path"],
        "config_digest": config["_config_digest"],
        "dataset_digest": bundle.digest,
        "dataset_identity": dataset_identity,
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
        "objective": "supervised ImageNet-1K continuation",
        "frozen_qwen_patch_position_stem": True,
        "full_qwen_transformer_loaded": False,
        "online_randaugment": True,
        "augmentation": dict(config.get("augmentation", {})),
        "loss": dict(config.get("loss", {})),
        "ema": dict(ema_values),
        "optimizer_groups": optimizer_schema,
        "effective_global_batch": effective_global_batch,
        "world_size": context.world_size,
        "updates_per_epoch": updates_per_epoch,
        "initialization": initialization,
        "initial_phases_sha256": initial_phases_sha256,
        "model": report,
        "implementation_manifest": implementation,
    }
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
    context.barrier()
    if not resume:
        if context.is_main:
            write_json(output / "manifest.json", manifest)
            atomic_save(initial_phases_path, initial_phases)
        validation_sampler.set_epoch(0)
        baseline = evaluate(wrapped, validation_loader, config, context)
        best_raw = float(baseline["top1_accuracy"])
        best_ema = best_raw
        rng_states = gather_rng_states(context)
        common = dict(
            model=core,
            ema=ema,
            optimizer=optimizer,
            optimizer_schema=optimizer_schema,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            global_step=0,
            best_raw=best_raw,
            best_ema=best_ema,
            history=history,
            config=config,
            initialization=initialization,
            implementation=implementation,
            dataset_identity=dataset_identity,
            initial_phases_sha256=initial_phases_sha256,
            rng_states=rng_states,
            world_size=context.world_size,
        )
        if context.is_main:
            write_json(output / "metrics" / "initial_baseline.json", baseline)
            _save_checkpoint(output / "checkpoints" / "last.pt", role="last", **common)
            _save_checkpoint(output / "checkpoints" / "best_raw.pt", role="best_raw", **common)
            _save_checkpoint(output / "checkpoints" / "best_ema.pt", role="best_ema", **common)
            print(
                f"[baseline] top1={best_raw:.6f} top5={baseline['top5_accuracy']:.6f}",
                flush=True,
            )
        context.barrier()

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        train_metrics, gradient_report, global_step = train_epoch(
            wrapped,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            ema,
            config,
            context,
            epoch=epoch,
            global_step=global_step,
        )
        raw_metrics = evaluate(wrapped, validation_loader, config, context)
        with ema.apply(core):
            ema_metrics = evaluate(wrapped, validation_loader, config, context)
        raw_top1 = float(raw_metrics["top1_accuracy"])
        ema_top1 = float(ema_metrics["top1_accuracy"])
        improved_raw = raw_top1 > best_raw
        improved_ema = ema_top1 > best_ema
        best_raw = max(best_raw, raw_top1)
        best_ema = max(best_ema, ema_top1)
        row = {
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "train": train_metrics,
            "validation_raw": raw_metrics,
            "validation_ema": ema_metrics,
            "phase_gradients": gradient_report,
            "phase_motion_from_continuation_start": core.phase_motion(initial_phases),
            "optical_gates": core.optical_gates(),
            "electronic_skip_gates": core.electronic_skip_gates(),
        }
        history.append(row)
        rng_states = gather_rng_states(context)
        common = dict(
            model=core,
            ema=ema,
            optimizer=optimizer,
            optimizer_schema=optimizer_schema,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_raw=best_raw,
            best_ema=best_ema,
            history=history,
            config=config,
            initialization=initialization,
            implementation=implementation,
            dataset_identity=dataset_identity,
            initial_phases_sha256=initial_phases_sha256,
            rng_states=rng_states,
            world_size=context.world_size,
        )
        if context.is_main:
            write_json(output / "metrics" / "history.json", history)
            write_json(output / "metrics" / "latest.json", row)
            _save_checkpoint(output / "checkpoints" / "last.pt", role="last", **common)
            if improved_raw:
                _save_checkpoint(output / "checkpoints" / "best_raw.pt", role="best_raw", **common)
            if improved_ema:
                _save_checkpoint(output / "checkpoints" / "best_ema.pt", role="best_ema", **common)
            interval = int(training.get("checkpoint_interval_epochs", 5))
            if epoch % interval == 0:
                _save_checkpoint(
                    output / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    role="last",
                    **common,
                )
            print(
                f"[epoch] {epoch}/{training['epochs']} raw_top1={raw_top1:.6f} "
                f"ema_top1={ema_top1:.6f} best_raw={best_raw:.6f} best_ema={best_ema:.6f}",
                flush=True,
            )
        context.barrier()

    best_role = "best_ema" if best_ema >= best_raw else "best_raw"
    best_path = output / "checkpoints" / f"{best_role}.pt"
    best_payload = torch.load(best_path, map_location=context.device, weights_only=False)
    core.load_state_dict(best_payload["model"], strict=True)
    ema.load_state_dict(best_payload["ema"])
    if best_role == "best_ema":
        scope = ema.apply(core)
    else:
        scope = _null_scope()
    with scope:
        validation_sampler.set_epoch(0)
        normal = evaluate(wrapped, validation_loader, config, context)
        ablations: dict[str, Any] = {}
        if bool(training.get("run_final_ablations", True)):
            for name in ("optical_off", "phase_random", "electronic_skip_off"):
                ablations[name] = evaluate(
                    wrapped,
                    validation_loader,
                    config,
                    context,
                    ablation=name,
                )
        backbone_state = core.backbone_state_dict()
        backbone_path = output / "checkpoints" / "backbone_best.pt"
        if context.is_main:
            atomic_save(
                backbone_path,
                {
                    "format": BACKBONE_FORMAT,
                    "backbone": backbone_state,
                    "state_variant": best_role,
                    "best_epoch": int(best_payload["epoch"]),
                    "source_training_checkpoint": str(best_path),
                    "source_training_checkpoint_sha256": sha256_file(best_path),
                    "config_digest": config["_config_digest"],
                    "stem_checkpoint_sha256": core.stem.checkpoint_sha256,
                    "model_config": model_config,
                    "model_report": core.parameter_report(),
                    "initialization": initialization,
                    "implementation_manifest": implementation,
                    "temporary_imagenet_readout_exported": False,
                },
            )
    context.barrier()
    result = {
        "status": "complete",
        "best_state_variant": best_role,
        "best_epoch": int(best_payload["epoch"]),
        "best_raw_top1": best_raw,
        "best_ema_top1": best_ema,
        "best_validation": normal,
        "ablations": ablations,
        "backbone_checkpoint": str(backbone_path),
        "backbone_sha256": sha256_file(backbone_path),
        "source_p11_epoch88": initialization,
        "model": core.parameter_report(),
    }
    if context.is_main:
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    context.barrier()


@contextmanager
def _null_scope() -> Iterator[None]:
    yield


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue P11 with a mature large-scale supervised recipe"
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    context = Context()
    try:
        run(load_config(args.config), context, resume=bool(args.resume))
    finally:
        context.close()


if __name__ == "__main__":
    main()
