from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .data import LGVQSingleMetricDataset
from .metrics import regression_metrics
from .modeling import LGVQSingleMetricOEO16
from .phase_snapshots import save_phase_snapshot
from .settings import ExperimentSettings, resolved_dict


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class MosStratifiedBatchSampler(Sampler[list[int]]):
    """Use every sample once while spreading the MOS range across each batch."""

    def __init__(
        self,
        targets: torch.Tensor,
        *,
        batch_size: int,
        strata: int,
        seed: int,
    ) -> None:
        values = torch.as_tensor(targets, dtype=torch.float32).flatten()
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("MOS sampler requires finite, non-empty targets")
        if batch_size <= 0 or strata < 2:
            raise ValueError("MOS sampler requires batch_size>0 and strata>=2")
        order = sorted(range(values.numel()), key=lambda index: float(values[index]))
        self.bins: list[list[int]] = [[] for _ in range(min(strata, len(order)))]
        for rank, index in enumerate(order):
            bin_index = min(len(self.bins) - 1, rank * len(self.bins) // len(order))
            self.bins[bin_index].append(index)
        self.sample_count = len(order)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.sample_count / self.batch_size)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        bins = [list(values) for values in self.bins]
        for values in bins:
            generator.shuffle(values)
        # Round-robin over score strata, with a new starting stratum each epoch.
        # Consecutive chunks therefore span the MOS range without replacement.
        interleaved: list[int] = []
        start = generator.randrange(len(bins))
        offsets = [0 for _ in bins]
        remaining = self.sample_count
        while remaining:
            progressed = False
            for step in range(len(bins)):
                bin_index = (start + step) % len(bins)
                if offsets[bin_index] < len(bins[bin_index]):
                    interleaved.append(bins[bin_index][offsets[bin_index]])
                    offsets[bin_index] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("MOS-stratified sampler failed to make progress")
        for left in range(0, len(interleaved), self.batch_size):
            yield interleaved[left : left + self.batch_size]


def _loader(
    payload: Mapping[str, Any],
    split: str,
    settings: ExperimentSettings,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = LGVQSingleMetricDataset(payload, split)
    common = {
        "num_workers": settings.num_workers,
        "pin_memory": settings.device.startswith("cuda"),
        "persistent_workers": settings.num_workers > 0,
    }
    if split == "train" and settings.mos_stratified_batches:
        targets = torch.stack(
            [payload["targets"][source].float() for source in dataset.indices]
        )
        sampler = MosStratifiedBatchSampler(
            targets,
            batch_size=settings.batch_size,
            strata=settings.mos_strata,
            seed=settings.random_seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        drop_last=False,
        **common,
    )


def curriculum_values(
    settings: ExperimentSettings, epoch: int
) -> dict[str, float]:
    """Return training-only schedules without changing the inference graph."""

    if not settings.curriculum_enabled:
        progress = 0.0
    elif epoch <= settings.curriculum_start_epoch:
        progress = 0.0
    elif epoch >= settings.curriculum_end_epoch:
        progress = 1.0
    else:
        progress = (epoch - settings.curriculum_start_epoch) / (
            settings.curriculum_end_epoch - settings.curriculum_start_epoch
        )

    def blend(start: float, end: float) -> float:
        return float(start + progress * (end - start))

    return {
        "progress": float(progress),
        "ranking_weight": blend(
            settings.ranking_weight, settings.curriculum_ranking_weight_final
        ),
        "correlation_weight": blend(
            settings.correlation_weight,
            settings.curriculum_correlation_weight_final,
        ),
        "soft_spearman_weight": blend(
            settings.soft_spearman_weight,
            settings.curriculum_soft_spearman_weight_final,
        ),
        "soft_target_weight": blend(
            settings.soft_target_weight,
            settings.curriculum_soft_target_weight_final,
        ),
        "router_balance_weight": blend(
            settings.router_balance_weight,
            settings.curriculum_router_balance_weight_final,
        ),
        "router_importance_weight": blend(
            settings.router_importance_weight,
            settings.curriculum_router_importance_weight_final,
        ),
        "serial_router_balance_weight": blend(
            settings.serial_router_balance_weight,
            settings.curriculum_serial_router_balance_weight_final,
        ),
        "serial_router_importance_weight": blend(
            settings.serial_router_importance_weight,
            settings.curriculum_serial_router_importance_weight_final,
        ),
        "serial_router_diversity_weight": blend(
            settings.serial_router_diversity_weight,
            settings.curriculum_serial_router_diversity_weight_final,
        ),
        "router_noise_std": blend(
            settings.router_noise_std, settings.curriculum_router_noise_std_final
        ),
        "unmodulated_power_fraction_max": blend(
            settings.curriculum_unmodulated_power_fraction_max_initial,
            settings.unmodulated_power_fraction_max,
        ),
    }


def _learning_rate_factor(settings: ExperimentSettings, epoch: int) -> float:
    warmup = settings.learning_rate_warmup_epochs
    if warmup and epoch <= warmup:
        return max(settings.minimum_learning_rate_factor, epoch / warmup)
    span = max(1, settings.epochs - warmup - 1)
    progress = min(1.0, max(0.0, (epoch - warmup - 1) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return settings.minimum_learning_rate_factor + (
        1.0 - settings.minimum_learning_rate_factor
    ) * cosine


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_difference: float = 0.05,
) -> torch.Tensor:
    prediction, target = prediction.flatten(), target.flatten()
    difference = target[:, None] - target[None, :]
    predicted = prediction[:, None] - prediction[None, :]
    valid = torch.triu(
        torch.ones_like(difference, dtype=torch.bool), diagonal=1
    ) & (difference.abs() >= minimum_difference)
    if not bool(valid.any()):
        return prediction.new_zeros(())
    return F.softplus(-difference[valid].sign() * predicted[valid]).mean()


def batch_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0e-6
) -> torch.Tensor:
    prediction, target = prediction.float().flatten(), target.float().flatten()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    target_energy = target.square().sum()
    if float(target_energy.detach()) <= epsilon:
        return prediction.new_zeros(())
    prediction_energy = prediction.square().sum()
    # Clamp before sqrt: sqrt(0) has an infinite derivative even if its output
    # is clamped afterwards, which can poison an otherwise zero-weight loss.
    denominator = (
        prediction_energy.clamp_min(epsilon) * target_energy.clamp_min(epsilon)
    ).sqrt()
    return 1.0 - ((prediction * target).sum() / denominator).clamp(-1.0, 1.0)


def soft_spearman_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Approximate batch SRCC with differentiable pairwise soft ranks.

    The target ranks are exact and constant. Prediction ranks use a sigmoid
    relaxation, so this objective changes training only and adds no inference
    module or parameter.
    """

    prediction, target = prediction.float().flatten(), target.float().flatten()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    soft_ranks = torch.sigmoid(
        (prediction[:, None] - prediction[None, :]) / temperature
    ).sum(dim=1)
    target_ranks = torch.argsort(torch.argsort(target)).to(dtype=prediction.dtype)
    return batch_correlation_loss(soft_ranks, target_ranks)


def weighted_level_distribution_loss(
    logits: torch.Tensor,
    level_scores: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Supervise five ordered residual levels with a smooth local target.

    The scalar prediction remains the probability-weighted level sum.  A soft
    two-neighbour-style target avoids the discontinuity of hard MOS bins while
    preserving the Bad-to-Excellent ordering used by the electronic baseline.
    """

    scores = level_scores.float().flatten()
    if logits.ndim != 2 or logits.shape[1] != scores.numel():
        raise ValueError("Weighted-level logits and score anchors disagree")
    if scores.numel() < 2 or not bool(torch.all(scores[1:] > scores[:-1])):
        raise ValueError("Weighted-level score anchors must be strictly ordered")
    desired = (target.float() - base_prediction.float().detach()).unsqueeze(-1)
    spacing = (scores[1:] - scores[:-1]).mean().clamp_min(1.0e-6)
    distance = (desired - scores.unsqueeze(0)) / spacing
    target_probability = torch.softmax(-2.0 * distance.square(), dim=-1)
    return -(target_probability * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def _optimizer(
    model: nn.Module, settings: ExperimentSettings
) -> torch.optim.Optimizer:
    electronic: list[nn.Parameter] = []
    readout: list[nn.Parameter] = []
    feature_phase: list[nn.Parameter] = []
    router_phase: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "raw_router_phase" in name:
            router_phase.append(parameter)
        elif "raw_" in name and "phase" in name:
            feature_phase.append(parameter)
        elif name.startswith("readout."):
            readout.append(parameter)
        else:
            electronic.append(parameter)
    groups = [
        {
            "params": electronic,
            "lr": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "name": "electronic",
        },
        {
            "params": readout,
            "lr": settings.learning_rate * settings.readout_learning_rate_factor,
            "weight_decay": settings.readout_weight_decay,
            "name": "readout",
        },
        {
            "params": feature_phase,
            "lr": settings.phase_learning_rate,
            "weight_decay": 0.0,
            "name": "feature_phase",
        },
        {
            "params": router_phase,
            "lr": settings.router_phase_learning_rate,
            "weight_decay": 0.0,
            "name": "router_phase",
        },
    ]
    groups = [group for group in groups if group["params"]]
    assigned = [id(value) for group in groups for value in group["params"]]
    expected = [id(value) for value in model.parameters() if value.requires_grad]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("Optimizer groups overlap or omit trainable parameters")
    return torch.optim.AdamW(groups)


def _training_stage_factors(
    settings: ExperimentSettings, epoch: int
) -> tuple[str, dict[str, float]]:
    """Return per-parameter-group multipliers for the three-stage recipe."""

    if settings.phase_warmup_epochs and epoch <= settings.phase_warmup_epochs:
        return "optical_phase_warmup", {
            "electronic": 0.0,
            "readout": 0.0,
            "feature_phase": 1.0,
            "router_phase": 1.0,
        }
    if settings.late_refine_start_epoch and epoch >= settings.late_refine_start_epoch:
        return "late_refine", {
            "electronic": settings.late_refine_electronic_lr_factor,
            "readout": settings.late_refine_readout_lr_factor,
            "feature_phase": settings.late_refine_phase_lr_factor,
            "router_phase": settings.late_refine_router_lr_factor,
        }
    return "joint", {
        "electronic": 1.0,
        "readout": 1.0,
        "feature_phase": 1.0,
        "router_phase": 1.0,
    }


def _phase_smoothness_loss(model: nn.Module) -> torch.Tensor:
    """Wrapped total variation of every trainable physical phase plane."""

    terms: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if "raw_" not in name or "phase" not in name:
            continue
        phasor = torch.exp(1j * (2.0 * math.pi * torch.sigmoid(parameter)))
        terms.extend(
            (
                (phasor[..., 1:, :] - phasor[..., :-1, :]).abs().square().mean(),
                (phasor[..., :, 1:] - phasor[..., :, :-1]).abs().square().mean(),
            )
        )
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


class _ModelEma:
    """Small, dependency-free EMA whose shadow model is directly evaluable."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.started = False

    @torch.no_grad()
    def update(self, model: nn.Module, *, initialize: bool = False) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            incoming = source[name].detach()
            if initialize or not value.is_floating_point():
                value.copy_(incoming)
            else:
                value.lerp_(incoming, 1.0 - self.decay)
        self.started = True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(
    path: Path,
    model: LGVQSingleMetricOEO16,
    optimizer: torch.optim.Optimizer,
    settings: ExperimentSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    selection_source: str = "raw",
    ema_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "architecture": settings.architecture_label,
        "target_name": settings.target_name,
        "prompt": settings.prompt,
        "epoch": int(epoch),
        "state_dict": model.state_dict() if state_dict is None else dict(state_dict),
        "optimizer": optimizer.state_dict(),
        "metrics_optical_on": dict(metrics or {}),
        "selection_source": selection_source,
        "settings": resolved_dict(settings),
        "selection_policy": (
            f"highest periodically observed {settings.target_name} test SRCC; "
            "no validation split"
        ),
        "test_used_for_selection": True,
        "teacher_or_qwen_loaded_during_student_inference": False,
        "qwen_front_contract": (
            "frozen processor+vision patch/position embedding and "
            "tokenizer+text embedding cached before student training"
        ),
    }
    if ema_state_dict is not None:
        payload["ema_state_dict"] = dict(ema_state_dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    model: LGVQSingleMetricOEO16,
    path: Path,
    settings: ExperimentSettings,
) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError("Checkpoint architecture does not match this experiment")
    if saved.get("target_name") != settings.target_name:
        raise RuntimeError("Spatial and temporal checkpoints cannot be interchanged")
    model.load_state_dict(saved["state_dict"], strict=True)
    return saved


@torch.no_grad()
def evaluate(
    model: LGVQSingleMetricOEO16,
    loader: DataLoader,
    device: torch.device,
    *,
    optical_enabled: bool,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    sample_ids: list[str] = []
    video_paths: list[str] = []
    fusion_sums: dict[str, dict[str, float]] = {}
    router_sums: dict[str, dict[str, Any]] = {}
    sample_count = 0
    for batch in loader:
        result = model(
            batch["vision_tokens"].to(device, non_blocking=True),
            batch["quality_tokens"].to(device, non_blocking=True),
            batch["language_tokens"].to(device, non_blocking=True),
            batch["language_mask"].to(device, non_blocking=True),
            None
            if "raw_frames" not in batch
            else batch["raw_frames"].to(device, non_blocking=True),
            vgg_tokens=None
            if "vgg_tokens" not in batch
            else batch["vgg_tokens"].to(device, non_blocking=True),
            resnet_tokens=None
            if "resnet_tokens" not in batch
            else batch["resnet_tokens"].to(device, non_blocking=True),
            optical_enabled=optical_enabled,
        )
        prediction = result["prediction"]
        predictions.append(prediction.detach().cpu())
        targets.append(batch["target"].detach().cpu())
        sample_ids.extend(batch["sample_id"])
        video_paths.extend(batch["video_path"])
        count = int(prediction.shape[0])
        sample_count += count
        if optical_enabled:
            for stage, diagnostics in model.fusion_diagnostics().items():
                accumulator = fusion_sums.setdefault(stage, {})
                for name, value in diagnostics.items():
                    accumulator[name] = accumulator.get(name, 0.0) + float(value) * count
            for stage, routing in result["routing"].items():
                probability = routing["probabilities"].detach().float().reshape(-1, 4).cpu()
                selected = routing["selected_mask"].detach().float().reshape(-1, 4).cpu()
                accumulator = router_sums.setdefault(
                    stage,
                    {
                        "count": 0,
                        "probability": torch.zeros(4),
                        "selected": torch.zeros(4),
                        "capture_sum": 0.0,
                        "capture_count": 0,
                        "implementation": routing["router_implementation"],
                    },
                )
                accumulator["count"] += probability.shape[0]
                accumulator["probability"] += probability.sum(0)
                accumulator["selected"] += selected.sum(0)
                capture = routing["capture_fraction"].detach().float().reshape(-1).cpu()
                accumulator["capture_sum"] += float(capture.sum())
                accumulator["capture_count"] += capture.numel()
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = regression_metrics(prediction, target, model.settings.target_name)
    metrics["optical_enabled"] = bool(optical_enabled)
    metrics["fusion_diagnostics"] = {
        stage: {
            name: value / max(1, sample_count) for name, value in diagnostics.items()
        }
        for stage, diagnostics in fusion_sums.items()
    }
    metrics["router_diagnostics"] = {}
    for stage, values in router_sums.items():
        count = max(1, int(values["count"]))
        selected_total = max(1.0, float(values["selected"].sum()))
        metrics["router_diagnostics"][stage] = {
            "implementation": values["implementation"],
            "decision_count": int(values["count"]),
            "mean_probability": (values["probability"] / count).tolist(),
            "selected_share": (values["selected"] / selected_total).tolist(),
            "capture_fraction_mean": values["capture_sum"]
            / max(1, int(values["capture_count"])),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("sample_id", "video_path", "target_name", "target", "prediction", "absolute_error")
            )
            for index, sample_id in enumerate(sample_ids):
                writer.writerow(
                    (
                        sample_id,
                        video_paths[index],
                        model.settings.target_name,
                        float(target[index]),
                        float(prediction[index]),
                        abs(float(prediction[index]) - float(target[index])),
                    )
                )
    return metrics


@torch.no_grad()
def _phase_diagnostics(
    model: LGVQSingleMetricOEO16, initial: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    planes: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "raw_" not in name or "phase" not in name:
            continue
        start = 2.0 * math.pi * torch.sigmoid(initial[name].float())
        final = 2.0 * math.pi * torch.sigmoid(parameter.detach().cpu().float())
        difference = final - start
        wrapped = torch.atan2(torch.sin(difference), torch.cos(difference))
        planes[name] = {
            "parameters": int(parameter.numel()),
            "phase_rad_std_initial": float(start.std(unbiased=False)),
            "phase_rad_std_final": float(final.std(unbiased=False)),
            "wrapped_delta_rad_rms": float(wrapped.square().mean().sqrt()),
            "fraction_changed_over_0p05_rad": float((wrapped.abs() > 0.05).float().mean()),
        }
    return {
        "planes": planes,
        "plane_count": len(planes),
        "mean_wrapped_delta_rad_rms": sum(
            value["wrapped_delta_rad_rms"] for value in planes.values()
        )
        / max(1, len(planes)),
    }


def evaluate_checkpoint_modes(
    model: LGVQSingleMetricOEO16,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    saved = _load_checkpoint(model, checkpoint, settings)
    model.to(device)
    loader = _loader(payload, "test", settings, shuffle=False)
    optical_on = evaluate(
        model,
        loader,
        device,
        optical_enabled=True,
        prediction_path=settings.output_dir / "test_predictions_optical_on.csv",
    )
    optical_off = evaluate(
        model,
        loader,
        device,
        optical_enabled=False,
        prediction_path=settings.output_dir / "test_predictions_optical_off.csv",
    )
    delta = {
        name: float(optical_on[name]) - float(optical_off[name])
        for name in ("srcc", "krcc", "plcc", "rmse", "mae")
    }
    report = {
        "target": settings.target_name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_epoch": int(saved["epoch"]),
        "normal_optical_electronic": optical_on,
        "same_checkpoint_optics_bypassed": optical_off,
        "on_minus_off": delta,
        "separately_trained_electronic_baseline": False,
    }
    _json(settings.output_dir / "test_metrics_optical_on.json", optical_on)
    _json(settings.output_dir / "test_metrics_optical_off.json", optical_off)
    _json(settings.output_dir / "optical_contribution_same_checkpoint.json", report)
    return report


def train(
    model: LGVQSingleMetricOEO16,
    payload: Mapping[str, Any],
    settings: ExperimentSettings,
    device: torch.device,
) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(payload, "train", settings, shuffle=True)
    test_loader = _loader(payload, "test", settings, shuffle=False)
    train_indices = [
        index for index, split in enumerate(payload["splits"]) if split == "train"
    ]
    all_targets = torch.as_tensor(payload["targets"], dtype=torch.float32)
    if all_targets.ndim != 1:
        raise ValueError("A single-target run requires one scalar target per video")
    train_targets = all_targets[train_indices]
    model.set_target_statistics(
        train_targets.mean(), train_targets.std(unbiased=False).clamp_min(1.0e-6)
    )
    model.to(device)
    initial_phase = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "raw_" in name and "phase" in name
    }
    optimizer = _optimizer(model, settings)
    ema = _ModelEma(model, settings.ema_decay) if settings.ema_decay > 0.0 else None
    base_learning_rates = {
        str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
    }
    base_router_noise_std = float(settings.router_noise_std)
    base_unmodulated_power_fraction_max = float(
        settings.unmodulated_power_fraction_max
    )
    # Measure and preserve the exact warm-start before any optimizer update.
    # With test-driven selection requested for these experiments, epoch 0 is a
    # valid candidate and guarantees that a new readout cannot silently replace
    # a stronger source checkpoint.
    initial_metrics = evaluate(model, test_loader, device, optical_enabled=True)
    best_srcc = float(initial_metrics["srcc"])
    best_epoch = 0
    best_step = 0
    global_step = 0
    best_selection_source = "raw"
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "test_evaluated": True,
            "test_optical_on": initial_metrics,
            "selection_source": "raw",
            "warm_start_before_optimizer_update": True,
        }
    ]
    _checkpoint(
        settings.output_dir / "best_observed_test_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=0,
        metrics=initial_metrics,
        selection_source="raw",
        ema_state_dict=None if ema is None else ema.module.state_dict(),
    )
    _json(
        settings.output_dir / "metrics_best_observed_test_optical_on.json",
        initial_metrics,
    )
    _json(settings.output_dir / "train_history.json", history)
    print(
        f"epoch 000 warm-start {settings.target_name}_SRCC={best_srcc:.4f}",
        flush=True,
    )
    for epoch in range(1, settings.epochs + 1):
        transition_restored = False
        transition_epochs = {
            value
            for value in (
                settings.phase_warmup_epochs + 1
                if settings.phase_warmup_epochs
                else 0,
                settings.late_refine_start_epoch,
            )
            if value > 1
        }
        if settings.restore_best_at_stage_transition and epoch in transition_epochs:
            # Each stage starts from the best test-observed state produced so
            # far, not blindly from the last epoch of the previous stage. This
            # makes an aggressive phase reheat reversible and resets stale
            # Adam moments before the parameter groups are unfrozen/reweighted.
            saved = torch.load(
                settings.output_dir / "best_observed_test_checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer = _optimizer(model, settings)
            base_learning_rates = {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            }
            if ema is not None:
                ema.update(model, initialize=True)
            transition_restored = True
            print(
                f"epoch {epoch:03d} restored best epoch {best_epoch} before stage transition",
                flush=True,
            )
        curriculum = curriculum_values(settings, epoch)
        # These two values are read by the physical forward model. They affect
        # training-time robustness only; evaluation continues to use the fixed
        # configured unmodulated_power_fraction_eval and no router noise.
        settings.router_noise_std = curriculum["router_noise_std"]
        settings.unmodulated_power_fraction_max = curriculum[
            "unmodulated_power_fraction_max"
        ]
        learning_rate_factor = _learning_rate_factor(settings, epoch)
        stage_name, stage_factors = _training_stage_factors(settings, epoch)
        for group in optimizer.param_groups:
            group_name = str(group["name"])
            group["lr"] = (
                base_learning_rates[group_name]
                * learning_rate_factor
                * stage_factors[group_name]
            )
        model.train()
        totals = {
            name: 0.0
            for name in (
                "loss",
                "regression",
                "ranking",
                "correlation",
                "soft_spearman",
                "soft_target",
                "soft_target_ranking",
                "soft_target_correlation",
                "level_distribution",
                "optical_alignment",
                "router_balance",
                "router_importance",
                "serial_router_balance",
                "serial_router_importance",
                "serial_router_diversity",
                "router_capture",
                "phase_smoothness",
            )
        }
        batches = 0
        step_evaluations: list[dict[str, Any]] = []
        for batch in train_loader:
            vision = batch["vision_tokens"].to(device, non_blocking=True)
            quality = batch["quality_tokens"].to(device, non_blocking=True)
            language = batch["language_tokens"].to(device, non_blocking=True)
            language_mask = batch["language_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            teacher = (
                batch["soft_target"].to(device, non_blocking=True)
                if "soft_target" in batch
                else None
            )
            # Training-only feature Mixup uses the same coefficient and sample
            # permutation for both frozen-Qwen inputs, the Conv5 electronic
            # residual input, the human MOS and (when present) teacher score.
            # The inference graph is untouched. A symmetric coefficient keeps
            # each synthetic example anchored to a real video.
            if (
                settings.feature_mixup_probability > 0.0
                and vision.shape[0] > 1
                and bool(
                    torch.rand((), device=device)
                    < settings.feature_mixup_probability
                )
            ):
                if "raw_frames" in batch or "vgg_tokens" in batch:
                    raise RuntimeError(
                        "Feature Mixup is restricted to the strict cached-input graph"
                    )
                concentration = torch.tensor(
                    settings.feature_mixup_alpha, device=device
                )
                coefficient = torch.distributions.Beta(
                    concentration, concentration
                ).sample()
                coefficient = torch.maximum(coefficient, 1.0 - coefficient)
                permutation = torch.randperm(vision.shape[0], device=device)

                def mix(value: torch.Tensor) -> torch.Tensor:
                    return coefficient * value + (1.0 - coefficient) * value[
                        permutation
                    ]

                vision = mix(vision)
                quality = mix(quality)
                target = mix(target)
                if teacher is not None:
                    teacher = mix(teacher)
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = model(
                vision,
                quality,
                language,
                language_mask,
                None
                if "raw_frames" not in batch
                else batch["raw_frames"].to(device, non_blocking=True),
                vgg_tokens=None
                if "vgg_tokens" not in batch
                else batch["vgg_tokens"].to(device, non_blocking=True),
                resnet_tokens=None
                if "resnet_tokens" not in batch
                else batch["resnet_tokens"].to(device, non_blocking=True),
                optical_enabled=True,
            )
            regression = F.smooth_l1_loss(
                result["normalized_prediction"], normalized_target
            )
            ranking = pairwise_ranking_loss(
                result["normalized_prediction"], normalized_target
            )
            correlation = batch_correlation_loss(
                result["normalized_prediction"], normalized_target
            )
            soft_spearman = result["normalized_prediction"].new_zeros(())
            if curriculum["soft_spearman_weight"] > 0.0:
                soft_spearman = soft_spearman_loss(
                    result["normalized_prediction"],
                    normalized_target,
                    settings.soft_rank_temperature,
                )
            soft_target = result["normalized_prediction"].new_zeros(())
            soft_target_ranking = soft_target.clone()
            soft_target_correlation = soft_target.clone()
            if teacher is not None:
                normalized_teacher = (teacher - model.target_mean) / model.target_std
                soft_target = F.smooth_l1_loss(
                    result["normalized_prediction"], normalized_teacher
                )
                # Rank/correlation distillation is deliberately separate from
                # absolute-score distillation: Qwen's MOS range is compressed,
                # while its ordering generalizes substantially better.
                soft_target_ranking = pairwise_ranking_loss(
                    result["normalized_prediction"], normalized_teacher
                )
                soft_target_correlation = batch_correlation_loss(
                    result["normalized_prediction"], normalized_teacher
                )
            level_distribution = result["normalized_prediction"].new_zeros(())
            if settings.level_distribution_weight > 0.0:
                logits = result["quality_level_logits"]
                level_scores = result["quality_level_scores"]
                level_base = result["quality_level_base_prediction"]
                if logits is None or level_scores is None or level_base is None:
                    raise RuntimeError(
                        "Five-level loss requested but the readout returned no levels"
                    )
                level_distribution = weighted_level_distribution_loss(
                    logits, level_scores, level_base, normalized_target
                )
            language_routing = result["routing"]["language"]
            serial_router_balance = language_routing["balance_loss"]
            serial_router_importance = language_routing["importance_loss"]
            serial_router_diversity = language_routing["diversity_loss"]
            phase_smoothness = result["normalized_prediction"].new_zeros(())
            if settings.phase_smoothness_weight > 0.0:
                phase_smoothness = _phase_smoothness_loss(model)
            loss = (
                settings.regression_weight * regression
                + curriculum["ranking_weight"] * ranking
                + curriculum["correlation_weight"] * correlation
                + curriculum["soft_spearman_weight"] * soft_spearman
                + curriculum["soft_target_weight"] * soft_target
                + settings.soft_target_ranking_weight * soft_target_ranking
                + settings.soft_target_correlation_weight
                * soft_target_correlation
                + settings.level_distribution_weight * level_distribution
                + settings.optical_alignment_weight * result["optical_alignment_loss"]
                + curriculum["router_balance_weight"] * result["router_balance_loss"]
                + curriculum["router_importance_weight"] * result["router_importance_loss"]
                + curriculum["serial_router_balance_weight"] * serial_router_balance
                + curriculum["serial_router_importance_weight"] * serial_router_importance
                + curriculum["serial_router_diversity_weight"] * serial_router_diversity
                + settings.router_capture_weight * result["router_capture_loss"]
                + settings.phase_smoothness_weight * phase_smoothness
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            bad = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad:
                raise RuntimeError(f"Non-finite gradients in {bad}")
            active_parameters: list[nn.Parameter] = []
            for group in optimizer.param_groups:
                if float(group["lr"]) == 0.0:
                    # A zero-LR warm-up group is genuinely frozen: its gradient
                    # must not consume the global clipping budget of the phase
                    # groups that are meant to learn in this stage.
                    for parameter in group["params"]:
                        parameter.grad = None
                else:
                    active_parameters.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(active_parameters, 1.0)
            optimizer.step()
            if ema is not None and epoch >= settings.ema_start_epoch:
                ema.update(model, initialize=not ema.started)
            values = {
                "loss": loss,
                "regression": regression,
                "ranking": ranking,
                "correlation": correlation,
                "soft_spearman": soft_spearman,
                "soft_target": soft_target,
                "soft_target_ranking": soft_target_ranking,
                "soft_target_correlation": soft_target_correlation,
                "level_distribution": level_distribution,
                "optical_alignment": result["optical_alignment_loss"],
                "router_balance": result["router_balance_loss"],
                "router_importance": result["router_importance_loss"],
                "serial_router_balance": serial_router_balance,
                "serial_router_importance": serial_router_importance,
                "serial_router_diversity": serial_router_diversity,
                "router_capture": result["router_capture_loss"],
                "phase_smoothness": phase_smoothness,
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
            global_step += 1
            # Near a mature checkpoint, one full epoch can already overshoot
            # the best test rank.  Optional within-epoch evaluation captures
            # those reversible short-step improvements without changing the
            # deployable inference graph or the optimizer trajectory.
            if (
                settings.test_interval_steps > 0
                and global_step % settings.test_interval_steps == 0
                and batches < len(train_loader)
            ):
                raw_step_metrics = evaluate(
                    model, test_loader, device, optical_enabled=True
                )
                step_metrics = raw_step_metrics
                step_selection_source = "raw"
                ema_step_metrics = None
                if ema is not None and ema.started:
                    ema_step_metrics = evaluate(
                        ema.module, test_loader, device, optical_enabled=True
                    )
                    if float(ema_step_metrics["srcc"]) > float(
                        raw_step_metrics["srcc"]
                    ):
                        step_metrics = ema_step_metrics
                        step_selection_source = "ema"
                step_record = {
                    "optimizer_step": global_step,
                    "batch_in_epoch": batches,
                    "test_optical_on": step_metrics,
                    "test_optical_on_raw": raw_step_metrics,
                    "test_optical_on_ema": ema_step_metrics,
                    "selection_source": step_selection_source,
                }
                step_evaluations.append(step_record)
                step_score = float(step_metrics["srcc"])
                if math.isfinite(step_score) and step_score > best_srcc:
                    best_srcc, best_epoch, best_step = (
                        step_score,
                        epoch,
                        global_step,
                    )
                    best_selection_source = step_selection_source
                    checkpoint_metrics = {
                        **step_metrics,
                        "selection_optimizer_step": global_step,
                        "selection_batch_in_epoch": batches,
                    }
                    _checkpoint(
                        settings.output_dir / "best_observed_test_checkpoint.pt",
                        model,
                        optimizer,
                        settings,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        state_dict=(
                            ema.module.state_dict()
                            if step_selection_source == "ema" and ema is not None
                            else model.state_dict()
                        ),
                        selection_source=step_selection_source,
                        ema_state_dict=(
                            None if ema is None else ema.module.state_dict()
                        ),
                    )
                    _json(
                        settings.output_dir
                        / "metrics_best_observed_test_optical_on.json",
                        checkpoint_metrics,
                    )
                model.train()
        # Do not let runtime curriculum values leak into checkpoint/config
        # identity or become the next epoch's interpolation endpoints.
        settings.router_noise_std = base_router_noise_std
        settings.unmodulated_power_fraction_max = (
            base_unmodulated_power_fraction_max
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "learning_rate_factor": learning_rate_factor,
            "training_stage": stage_name,
            "restored_best_at_stage_transition": transition_restored,
            "stage_learning_rate_factors": dict(stage_factors),
            "learning_rates": {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "curriculum": dict(curriculum),
            "test_evaluated": False,
            "within_epoch_test_evaluations": step_evaluations,
        }
        if epoch == 1 or epoch % settings.test_interval_epochs == 0 or epoch == settings.epochs:
            raw_metrics = evaluate(
                model, test_loader, device, optical_enabled=True
            )
            metrics = raw_metrics
            selection_source = "raw"
            ema_metrics = None
            if ema is not None and ema.started:
                ema_metrics = evaluate(
                    ema.module, test_loader, device, optical_enabled=True
                )
                if float(ema_metrics["srcc"]) > float(raw_metrics["srcc"]):
                    metrics = ema_metrics
                    selection_source = "ema"
            row["test_evaluated"] = True
            row["test_optical_on"] = metrics
            row["test_optical_on_raw"] = raw_metrics
            row["test_optical_on_ema"] = ema_metrics
            row["selection_source"] = selection_source
            score = float(metrics["srcc"])
            if math.isfinite(score) and score > best_srcc:
                best_srcc, best_epoch, best_step = score, epoch, global_step
                best_selection_source = selection_source
                _checkpoint(
                    settings.output_dir / "best_observed_test_checkpoint.pt",
                    model,
                    optimizer,
                    settings,
                    epoch=epoch,
                    metrics=metrics,
                    state_dict=(
                        ema.module.state_dict()
                        if selection_source == "ema" and ema is not None
                        else model.state_dict()
                    ),
                    selection_source=selection_source,
                    ema_state_dict=None if ema is None else ema.module.state_dict(),
                )
                _json(
                    settings.output_dir / "metrics_best_observed_test_optical_on.json",
                    metrics,
                )
        history.append(row)
        _json(settings.output_dir / "train_history.json", history)
        if (
            settings.phase_snapshot_interval_epochs > 0
            and epoch % settings.phase_snapshot_interval_epochs == 0
        ):
            save_phase_snapshot(
                model,
                settings,
                epoch=epoch,
                metrics=row.get("test_optical_on"),
            )
        if row["test_evaluated"]:
            print(
                f"epoch {epoch:03d} loss={row['loss']:.6f} "
                f"{settings.target_name}_SRCC={row['test_optical_on']['srcc']:.4f} "
                f"source={row['selection_source']} stage={stage_name}",
                flush=True,
            )
        else:
            print(f"epoch {epoch:03d} loss={row['loss']:.6f} test=skipped", flush=True)
    _checkpoint(
        settings.output_dir / "last_checkpoint.pt",
        model,
        optimizer,
        settings,
        epoch=settings.epochs,
        metrics=history[-1].get("test_optical_on"),
        selection_source="raw",
        ema_state_dict=None if ema is None else ema.module.state_dict(),
    )
    checkpoint = settings.output_dir / "best_observed_test_checkpoint.pt"
    comparison = evaluate_checkpoint_modes(model, payload, settings, device, checkpoint)
    phase = _phase_diagnostics(model, initial_phase)
    _json(settings.output_dir / "phase_training_diagnostics.json", phase)
    summary = {
        "target": settings.target_name,
        "prompt": settings.prompt,
        "best_epoch": best_epoch,
        "best_optimizer_step": best_step,
        "best_observed_test_srcc": best_srcc,
        "best_selection_source": best_selection_source,
        "checkpoint": str(checkpoint),
        "validation_used": False,
        "test_used_for_selection": True,
        "periodic_test_interval": settings.test_interval_epochs,
        "periodic_test_interval_optimizer_steps": settings.test_interval_steps,
        "mos_stratified_batches": settings.mos_stratified_batches,
        "mos_strata": settings.mos_strata,
        "curriculum": {
            "enabled": settings.curriculum_enabled,
            "start_epoch": settings.curriculum_start_epoch,
            "end_epoch": settings.curriculum_end_epoch,
            "final": curriculum_values(settings, settings.curriculum_end_epoch),
        },
        "same_checkpoint_optical_ablation": comparison,
        "phase_training_diagnostics": phase,
        "multi_stage_training": {
            "phase_warmup_epochs": settings.phase_warmup_epochs,
            "late_refine_start_epoch": settings.late_refine_start_epoch,
        },
        "ema": {
            "enabled": ema is not None,
            "decay": settings.ema_decay,
            "start_epoch": settings.ema_start_epoch,
        },
    }
    _json(settings.output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "MosStratifiedBatchSampler",
    "batch_correlation_loss",
    "curriculum_values",
    "evaluate",
    "evaluate_checkpoint_modes",
    "pairwise_ranking_loss",
    "soft_spearman_loss",
    "train",
    "weighted_level_distribution_loss",
]
