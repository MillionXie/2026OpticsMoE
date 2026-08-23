from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

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

from .model import QwenStemOpticalImageNetBackbone


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Training config must contain a mapping")
    raw["_config_path"] = str(config_path)
    raw["_config_digest"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return raw


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Context:
    def __init__(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            if self.world_size > 1:
                raise RuntimeError("DDP requires CUDA")
            self.device = torch.device("cpu")
        if self.world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl", init_method="env://")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def close(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def seed_all(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def make_loader(dataset, sampler, config: dict[str, Any], *, train: bool) -> DataLoader:
    training = config["training"]
    workers = int(training.get("num_workers", 8))
    return DataLoader(
        dataset,
        batch_size=int(
            training.get("batch_size", 24)
            if train
            else training.get("validation_batch_size", 48)
        ),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(training.get("pin_memory", True)),
        persistent_workers=bool(training.get("persistent_workers", True)) and workers > 0,
        prefetch_factor=int(training.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=train,
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]):
    values = config["optimizer"]
    groups = [
        {
            "params": list(model.phase_parameters()),
            "lr": float(values.get("phase_learning_rate", 4.0e-3)),
            "weight_decay": 0.0,
            "name": "phase",
        },
        {
            "params": list(model.adapter_parameters()),
            "lr": float(values.get("adapter_learning_rate", 3.0e-4)),
            "weight_decay": float(values.get("weight_decay", 5.0e-4)),
            "name": "adapter",
        },
        {
            "params": list(model.residual_parameters()),
            "lr": float(values.get("residual_learning_rate", 2.0e-4)),
            "weight_decay": float(values.get("weight_decay", 5.0e-4)),
            "name": "residual",
        },
        {
            "params": list(model.head_parameters()),
            "lr": float(values.get("head_learning_rate", 5.0e-4)),
            "weight_decay": float(values.get("weight_decay", 5.0e-4)),
            "name": "head",
        },
    ]
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(x) for x in values.get("betas", [0.9, 0.999])),
        eps=float(values.get("eps", 1.0e-8)),
    )


def build_scheduler(optimizer, config: dict[str, Any], steps_per_epoch: int):
    values = config["optimizer"]
    epochs = int(config["training"]["epochs"])
    warmup = int(values.get("warmup_epochs", 2)) * steps_per_epoch
    total = max(epochs * steps_per_epoch, 1)
    minimum = float(values.get("minimum_learning_rate_ratio", 0.05))

    def scale(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1.0 / warmup)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def mix_batch(images: torch.Tensor, labels: torch.Tensor, config: dict[str, Any]):
    loss = config.get("loss", {})
    probability = float(loss.get("batch_mix_probability", 0.0))
    if probability <= 0.0 or random.random() >= probability:
        return images, labels, labels, 1.0, "none"
    mixup = float(loss.get("mixup_alpha", 0.0))
    cutmix = float(loss.get("cutmix_alpha", 0.0))
    enabled = [name for name, alpha in (("mixup", mixup), ("cutmix", cutmix)) if alpha > 0.0]
    if not enabled:
        return images, labels, labels, 1.0, "none"
    mode = random.choice(enabled)
    alpha = mixup if mode == "mixup" else cutmix
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    labels_b = labels[permutation]
    if mode == "mixup":
        return lam * images + (1.0 - lam) * images[permutation], labels, labels_b, lam, mode
    height, width = images.shape[-2:]
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    center_y = random.randrange(height)
    center_x = random.randrange(width)
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    actual = 1.0 - ((y2 - y1) * (x2 - x1)) / (height * width)
    return mixed, labels, labels_b, float(actual), mode


def topk_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    predictions = logits.topk(5, dim=-1).indices
    correct = predictions.eq(labels[:, None])
    return float(correct[:, :1].any(dim=1).sum()), float(correct.any(dim=1).sum())


def reduce_metrics(vector: torch.Tensor, elapsed: float) -> dict[str, float]:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(vector)
    loss_sum, top1, top5, samples, batches = (float(value) for value in vector.cpu())
    return {
        "loss": loss_sum / max(samples, 1.0),
        "top1_accuracy": top1 / max(samples, 1.0),
        "top5_accuracy": top5 / max(samples, 1.0),
        "samples": int(samples),
        "batches": int(batches),
        "seconds": float(elapsed),
    }


def phase_gradient_report(model: nn.Module) -> dict[str, Any]:
    norms = []
    finite = []
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
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    config: dict[str, Any],
    context: Context,
    *,
    epoch: int,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    model.train()
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    vector = torch.zeros(5, dtype=torch.float64, device=context.device)
    started = time.perf_counter()
    limit = config["training"].get("max_train_batches")
    label_smoothing = float(config.get("loss", {}).get("label_smoothing", 0.1))
    use_amp = bool(config["training"].get("use_amp", True)) and context.device.type == "cuda"
    phase_clip = float(config["optimizer"].get("phase_gradient_clip_norm", 2.0))
    electronic_clip = float(config["optimizer"].get("electronic_gradient_clip_norm", 5.0))
    log_interval = int(config["training"].get("log_interval_batches", 100))
    gradient_report = None
    core = unwrap(model)
    phase_parameters = list(core.phase_parameters())
    electronic_parameters = (
        list(core.adapter_parameters())
        + list(core.residual_parameters())
        + list(core.head_parameters())
    )
    for batch_index, batch in enumerate(loader, 1):
        if limit is not None and batch_index > int(limit):
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        images, labels_a, labels_b, lam, _ = mix_batch(images, labels, config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=context.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss_a = F.cross_entropy(logits, labels_a, label_smoothing=label_smoothing)
            loss_b = F.cross_entropy(logits, labels_b, label_smoothing=label_smoothing)
            loss = lam * loss_a + (1.0 - lam) * loss_b
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_report is None:
            gradient_report = phase_gradient_report(core)
        torch.nn.utils.clip_grad_norm_(phase_parameters, phase_clip)
        torch.nn.utils.clip_grad_norm_(electronic_parameters, electronic_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        count = labels.numel()
        correct1, correct5 = topk_counts(logits.detach(), labels)
        vector += torch.tensor(
            [float(loss.detach()) * count, correct1, correct5, count, 1],
            dtype=torch.float64,
            device=context.device,
        )
        if context.is_main and (batch_index == 1 or batch_index % log_interval == 0):
            rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            print(
                f"[train] epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={float(loss.detach()):.4f} lr={rates}",
                flush=True,
            )
    metrics = reduce_metrics(vector, time.perf_counter() - started)
    metrics["samples_per_second"] = metrics["samples"] / max(metrics["seconds"], 1.0e-9)
    if context.device.type == "cuda":
        peak = torch.tensor(
            [
                torch.cuda.max_memory_allocated(context.device),
                torch.cuda.max_memory_reserved(context.device),
            ],
            dtype=torch.float64,
            device=context.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        metrics["peak_allocated_mib"] = float(peak[0].cpu()) / (1024.0 ** 2)
        metrics["peak_reserved_mib"] = float(peak[1].cpu()) / (1024.0 ** 2)
    return metrics, gradient_report


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
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
        with torch.autocast(device_type=context.device.type, dtype=torch.float16, enabled=use_amp):
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    best_top1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    atomic_save(
        path,
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_validation_top1": float(best_top1),
            "history": history,
            "config_digest": config["_config_digest"],
        },
    )


def run(
    config: dict[str, Any],
    context: Context,
    *,
    resume: bool,
    model_class: type[nn.Module] = QwenStemOpticalImageNetBackbone,
    experiment_name: str = "P08 Qwen static patch stem + eight-stage optical ImageNet backbone",
) -> None:
    training = config["training"]
    seed_all(int(training.get("seed", 2026)), context.rank)
    output = resolve_path(config["output_dir"])
    imagenet_settings = load_imagenet_settings(resolve_path(config["imagenet_config"]))
    bundle = load_imagenet(imagenet_settings)
    train_per_class = training.get("train_samples_per_class")
    if train_per_class is None:
        train_indices = list(range(bundle.train.base_sample_count))
    else:
        train_indices = stratified_base_indices(
            bundle.train.targets, int(train_per_class), int(training.get("seed", 2026))
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
    train_loader = make_loader(bundle.train, train_sampler, config, train=True)
    validation_loader = make_loader(bundle.validation, validation_sampler, config, train=False)
    model_config = dict(config["model"])
    model_config.setdefault("seed", int(training.get("seed", 2026)))
    model = model_class(resolve_path(config["stem_checkpoint"]), model_config)
    initial_phases = model.phase_snapshot()
    report = model.parameter_report()
    fraction_scope = str(model_config.get("optical_parameter_fraction_scope", "all_trainable"))
    if fraction_scope == "all_trainable":
        measured_fraction = float(
            report.get("optical_fraction_of_all_trainable", report["optical_fraction_of_trainable"])
        )
    elif fraction_scope == "backbone_excluding_task_head":
        measured_fraction = float(report["optical_fraction_of_backbone_trainable"])
    else:
        raise ValueError(f"Unsupported optical parameter fraction scope: {fraction_scope}")
    required_fraction = float(model_config.get("minimum_optical_parameter_fraction", 0.50))
    if measured_fraction < required_fraction:
        raise RuntimeError(
            f"Optical parameter fraction {measured_fraction:.4f} in scope {fraction_scope!r} "
            f"is below the required {required_fraction:.4f}"
        )
    if report["minimum_optical_gate"] < 0.50:
        raise RuntimeError("Optical fusion gate fell below 0.5 before training")
    model.to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    optimizer = build_optimizer(unwrap(model), config)
    steps_per_epoch = min(
        len(train_loader), int(training.get("max_train_batches") or len(train_loader))
    )
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training.get("use_amp", True)) and context.device.type == "cuda",
        init_scale=float(training.get("amp_initial_scale", 256.0)),
        growth_interval=int(training.get("amp_growth_interval", 100000)),
    )
    manifest = {
        "experiment": experiment_name,
        "config": config["_config_path"],
        "config_digest": config["_config_digest"],
        "world_size": context.world_size,
        "dataset_digest": bundle.digest,
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
        "online_qwen_stem": True,
        "hidden_state_cache_used": False,
        "full_qwen_loaded": False,
        "optical_parameter_budget": {
            "scope": fraction_scope,
            "measured_fraction": measured_fraction,
            "minimum_required_fraction": required_fraction,
        },
        "model": report,
    }
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)

    last_path = output / "checkpoints" / "last.pt"
    best_path = output / "checkpoints" / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_top1 = 1, -math.inf
    if resume and last_path.is_file():
        payload = torch.load(last_path, map_location=context.device, weights_only=False)
        if payload.get("config_digest") != config["_config_digest"]:
            raise RuntimeError("Resume config digest mismatch")
        unwrap(model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1
        best_top1 = float(payload["best_validation_top1"])
        history = list(payload.get("history", []))
        initial_phases = torch.load(
            output / "initial_phases.pt", map_location="cpu", weights_only=True
        )
        if context.is_main:
            print(f"[resume] epoch={start_epoch} best={best_top1:.4f}", flush=True)
    elif context.is_main:
        atomic_save(output / "initial_phases.pt", initial_phases)
    context.barrier()

    if start_epoch == 1:
        validation_sampler.set_epoch(0)
        baseline = evaluate(model, validation_loader, config, context)
        best_top1 = baseline["top1_accuracy"]
        if context.is_main:
            write_json(output / "metrics" / "initial_baseline.json", baseline)
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch=0, best_top1=best_top1, history=history, config=config,
            )
            print(f"[baseline] top1={best_top1:.4f} top5={baseline['top5_accuracy']:.4f}", flush=True)
        best_tensor = torch.tensor(best_top1, device=context.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(best_tensor, 0)
        best_top1 = float(best_tensor)
        context.barrier()

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        train_metrics, gradient_report = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, context, epoch=epoch
        )
        validation_metrics = evaluate(model, validation_loader, config, context)
        motion = unwrap(model).phase_motion(initial_phases)
        row = {
            "epoch": epoch,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "train": train_metrics,
            "validation": validation_metrics,
            "phase_gradients": gradient_report,
            "phase_motion": motion,
            "optical_gates": unwrap(model).optical_gates(),
        }
        electronic_skip_gates = getattr(unwrap(model), "electronic_skip_gates", None)
        if callable(electronic_skip_gates):
            row["electronic_skip_gates"] = electronic_skip_gates()
        if context.is_main:
            history.append(row)
            write_json(output / "metrics" / "history.json", history)
            write_json(output / "metrics" / "latest.json", row)
            if validation_metrics["top1_accuracy"] > best_top1:
                best_top1 = validation_metrics["top1_accuracy"]
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, best_top1=best_top1, history=history, config=config,
                )
            save_checkpoint(
                last_path, model, optimizer, scheduler, scaler,
                epoch=epoch, best_top1=best_top1, history=history, config=config,
            )
            interval = int(training.get("checkpoint_interval_epochs", 5))
            if epoch % interval == 0:
                save_checkpoint(
                    output / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    model, optimizer, scheduler, scaler,
                    epoch=epoch, best_top1=best_top1, history=history, config=config,
                )
            print(
                f"[epoch] {epoch}/{training['epochs']} train_top1={train_metrics['top1_accuracy']:.4f} "
                f"val_top1={validation_metrics['top1_accuracy']:.4f} best={best_top1:.4f} "
                f"phase_motion={motion['mean_absolute_rad']:.4f}rad",
                flush=True,
            )
        best_tensor = torch.tensor(best_top1, device=context.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(best_tensor, 0)
        best_top1 = float(best_tensor)
        context.barrier()

    best_payload = torch.load(best_path, map_location=context.device, weights_only=False)
    unwrap(model).load_state_dict(best_payload["model"], strict=True)
    validation_sampler.set_epoch(0)
    normal = evaluate(model, validation_loader, config, context)
    ablations = {}
    if bool(training.get("run_final_ablations", True)):
        for name in ("optical_off", "phase_random", "electronic_skip_off"):
            ablations[name] = evaluate(model, validation_loader, config, context, ablation=name)
    if context.is_main:
        backbone_path = output / "checkpoints" / "backbone.pt"
        atomic_save(
            backbone_path,
            {
                "backbone": unwrap(model).backbone_state_dict(),
                "best_epoch": int(best_payload["epoch"]),
                "config_digest": config["_config_digest"],
                "stem_checkpoint_sha256": unwrap(model).stem.checkpoint_sha256,
                "model_report": unwrap(model).parameter_report(),
                "feature_contract": {
                    "input": "CLIP-normalized RGB [B,3,224,224]",
                    "final": "three latent optical banks [B,3,224,224]",
                    "stages": "tuple of eight [B,3,224,224] OEO feature maps",
                    "qwen_transformer_required": False,
                },
            },
        )
        result = {
            "status": "complete",
            "best_epoch": int(best_payload["epoch"]),
            "backbone_checkpoint": str(backbone_path),
            "best_validation": normal,
            "ablations": ablations,
            "model": unwrap(model).parameter_report(),
            "phase_motion": unwrap(model).phase_motion(initial_phases),
        }
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Qwen-stem eight-stage optical ImageNet backbone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = Context()
    try:
        run(load_config(args.config), context, resume=args.resume)
    finally:
        context.close()


if __name__ == "__main__":
    main()
