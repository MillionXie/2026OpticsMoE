from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.clip_teacher import (
    load_text_prototypes,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
    CLIP_MEAN,
    CLIP_STD,
    load_imagenet,
    view_seed,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
    load_settings as load_imagenet_settings,
)
from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.teacher_cache import (
    ClipFeatureStore,
    DistillationViewDataset,
    cache_directory,
)

from .formal_settings import load_formal_settings
from .model import Ablation, OpticalClassifier
from .settings import OpticalConfig, REPO_ROOT


@dataclass(frozen=True)
class P06ModelConfig:
    selected_stage_indices: tuple[int, ...]
    pool_size: int
    projection_dim: int
    num_classes: int
    classifier_mode: str
    classifier_hidden_dim: int
    classifier_dropout: float


@dataclass(frozen=True)
class P06LossConfig:
    supervised_ce_weight: float
    feature_cosine_weight: float
    clip_logit_kd_weight: float
    contrastive_weight: float
    contrastive_temperature: float
    distill_temperature: float
    label_smoothing: float


@dataclass(frozen=True)
class P06OptimizerConfig:
    phase_learning_rate: float
    residual_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    gradient_clip_norm: float
    warmup_steps: int
    minimum_learning_rate_ratio: float


@dataclass(frozen=True)
class P06TrainingConfig:
    seed: int
    head_warmup_epochs: int
    joint_epochs: int
    train_samples_per_class: int | None
    validation_samples_per_class: int
    batch_size: int
    validation_batch_size: int
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int
    shuffle_block_size: int | None
    use_amp: bool
    amp_initial_scale: float
    amp_growth_interval: int
    log_interval_batches: int
    checkpoint_interval_epochs: int
    max_train_batches: int | None
    max_validation_batches: int | None
    run_final_ablations: bool

    @property
    def epochs(self) -> int:
        return self.head_warmup_epochs + self.joint_epochs


@dataclass(frozen=True)
class P06GateConfig:
    validation_top1_min: float
    clip_cosine_min: float
    optical_gate_min: float
    optical_disruption_relative_drop_min: float


@dataclass(frozen=True)
class P06Settings:
    config_path: Path
    output_dir: Path
    imagenet_config: Path
    architecture_config: Path
    source_checkpoint: Path
    source_checkpoint_sha256: str
    source_checkpoint_load_mode: str
    initial_checkpoint: Path | None
    initial_checkpoint_sha256: str | None
    initial_checkpoint_load_mode: str
    model: P06ModelConfig
    loss: P06LossConfig
    optimizer: P06OptimizerConfig
    training: P06TrainingConfig
    gates: P06GateConfig

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_p06_settings(path: str | Path) -> P06Settings:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P06 configuration root must be a mapping")
    model_raw = raw.get("model", {})
    loss_raw = raw.get("loss", {})
    optimizer_raw = raw.get("optimizer", {})
    training_raw = raw.get("training", {})
    gate_raw = raw.get("gates", {})
    checksum = str(raw["source_checkpoint_sha256"]).lower()
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ValueError("source_checkpoint_sha256 must be a lowercase SHA-256")
    initial_value = raw.get("initial_checkpoint")
    initial_checksum_value = raw.get("initial_checkpoint_sha256")
    if bool(initial_value) != bool(initial_checksum_value):
        raise ValueError("initial_checkpoint and initial_checkpoint_sha256 must be provided together")
    initial_checksum = str(initial_checksum_value).lower() if initial_checksum_value else None
    if initial_checksum is not None and (
        len(initial_checksum) != 64
        or any(character not in "0123456789abcdef" for character in initial_checksum)
    ):
        raise ValueError("initial_checkpoint_sha256 must be a lowercase SHA-256")
    settings = P06Settings(
        config_path=config_path,
        output_dir=_resolve(raw["output_dir"]),
        imagenet_config=_resolve(raw["imagenet_config"]),
        architecture_config=_resolve(raw["architecture_config"]),
        source_checkpoint=_resolve(raw["source_checkpoint"]),
        source_checkpoint_sha256=checksum,
        source_checkpoint_load_mode=str(
            raw.get("source_checkpoint_load_mode", "strict")
        ).lower(),
        initial_checkpoint=_resolve(initial_value) if initial_value else None,
        initial_checkpoint_sha256=initial_checksum,
        initial_checkpoint_load_mode=str(
            raw.get("initial_checkpoint_load_mode", "strict")
        ).lower(),
        model=P06ModelConfig(
            selected_stage_indices=tuple(int(value) for value in model_raw.get("selected_stage_indices", [1, 3, 5, 7])),
            pool_size=int(model_raw.get("pool_size", 4)),
            projection_dim=int(model_raw.get("projection_dim", 512)),
            num_classes=int(model_raw.get("num_classes", 1000)),
            classifier_mode=str(
                model_raw.get("classifier_mode", "projected_linear")
            ).lower(),
            classifier_hidden_dim=int(model_raw.get("classifier_hidden_dim", 512)),
            classifier_dropout=float(model_raw.get("classifier_dropout", 0.0)),
        ),
        loss=P06LossConfig(
            supervised_ce_weight=float(loss_raw.get("supervised_ce_weight", 0.5)),
            feature_cosine_weight=float(loss_raw.get("feature_cosine_weight", 1.0)),
            clip_logit_kd_weight=float(loss_raw.get("clip_logit_kd_weight", 0.5)),
            contrastive_weight=float(loss_raw.get("contrastive_weight", 0.0)),
            contrastive_temperature=float(loss_raw.get("contrastive_temperature", 0.07)),
            distill_temperature=float(loss_raw.get("distill_temperature", 2.0)),
            label_smoothing=float(loss_raw.get("label_smoothing", 0.1)),
        ),
        optimizer=P06OptimizerConfig(
            phase_learning_rate=float(optimizer_raw.get("phase_learning_rate", 1e-4)),
            residual_learning_rate=float(optimizer_raw.get("residual_learning_rate", 1e-4)),
            head_learning_rate=float(optimizer_raw.get("head_learning_rate", 5e-4)),
            weight_decay=float(optimizer_raw.get("weight_decay", 1e-4)),
            betas=tuple(float(value) for value in optimizer_raw.get("betas", [0.9, 0.999])),
            eps=float(optimizer_raw.get("eps", 1e-8)),
            gradient_clip_norm=float(optimizer_raw.get("gradient_clip_norm", 5.0)),
            warmup_steps=int(optimizer_raw.get("warmup_steps", 2000)),
            minimum_learning_rate_ratio=float(optimizer_raw.get("minimum_learning_rate_ratio", 0.05)),
        ),
        training=P06TrainingConfig(
            seed=int(training_raw.get("seed", 2026)),
            head_warmup_epochs=int(training_raw.get("head_warmup_epochs", 1)),
            joint_epochs=int(training_raw.get("joint_epochs", 5)),
            train_samples_per_class=_optional_int(
                training_raw.get("train_samples_per_class", 100)
            ),
            validation_samples_per_class=int(training_raw.get("validation_samples_per_class", 10)),
            batch_size=int(training_raw.get("batch_size", 32)),
            validation_batch_size=int(training_raw.get("validation_batch_size", 64)),
            num_workers=int(training_raw.get("num_workers", 8)),
            persistent_workers=bool(training_raw.get("persistent_workers", True)),
            pin_memory=bool(training_raw.get("pin_memory", True)),
            prefetch_factor=int(training_raw.get("prefetch_factor", 2)),
            shuffle_block_size=_optional_int(training_raw.get("shuffle_block_size")),
            use_amp=bool(training_raw.get("use_amp", True)),
            amp_initial_scale=float(training_raw.get("amp_initial_scale", 65536.0)),
            amp_growth_interval=int(training_raw.get("amp_growth_interval", 2000)),
            log_interval_batches=int(training_raw.get("log_interval_batches", 50)),
            checkpoint_interval_epochs=int(training_raw.get("checkpoint_interval_epochs", 1)),
            max_train_batches=_optional_int(training_raw.get("max_train_batches")),
            max_validation_batches=_optional_int(training_raw.get("max_validation_batches")),
            run_final_ablations=bool(training_raw.get("run_final_ablations", True)),
        ),
        gates=P06GateConfig(
            validation_top1_min=float(gate_raw.get("validation_top1_min", 0.10)),
            clip_cosine_min=float(gate_raw.get("clip_cosine_min", 0.70)),
            optical_gate_min=float(gate_raw.get("optical_gate_min", 0.50)),
            optical_disruption_relative_drop_min=float(gate_raw.get("optical_disruption_relative_drop_min", 0.30)),
        ),
    )
    if not settings.model.selected_stage_indices:
        raise ValueError("At least one selected stage is required")
    if settings.model.pool_size < 1 or settings.model.projection_dim < 1:
        raise ValueError("P06 pool/projection sizes must be positive")
    if settings.model.classifier_mode not in {
        "projected_linear",
        "descriptor_linear",
        "descriptor_mlp",
    }:
        raise ValueError("Unsupported model.classifier_mode")
    if settings.model.classifier_hidden_dim < 1:
        raise ValueError("model.classifier_hidden_dim must be positive")
    if not 0.0 <= settings.model.classifier_dropout < 1.0:
        raise ValueError("model.classifier_dropout must be in [0, 1)")
    if settings.source_checkpoint_load_mode not in {"strict", "integrity_only"}:
        raise ValueError("source_checkpoint_load_mode must be strict or integrity_only")
    if settings.initial_checkpoint_load_mode not in {
        "strict",
        "compatible",
        "expanded",
    }:
        raise ValueError(
            "initial_checkpoint_load_mode must be strict, compatible or expanded"
        )
    if (
        settings.source_checkpoint_load_mode == "integrity_only"
        and settings.initial_checkpoint_load_mode != "expanded"
    ):
        raise ValueError(
            "source integrity_only is allowed only with an expanded initial checkpoint"
        )
    if settings.training.head_warmup_epochs < 0 or settings.training.joint_epochs < 1:
        raise ValueError("P06 requires at least one joint-training epoch")
    if (
        settings.training.train_samples_per_class is not None
        and settings.training.train_samples_per_class < 1
    ) or settings.training.validation_samples_per_class < 1:
        raise ValueError("Per-class sample counts must be positive")
    if (
        settings.training.shuffle_block_size is not None
        and settings.training.shuffle_block_size < 1
    ):
        raise ValueError("training.shuffle_block_size must be positive or null")
    if settings.training.amp_initial_scale <= 0.0:
        raise ValueError("training.amp_initial_scale must be positive")
    if settings.training.amp_growth_interval < 1:
        raise ValueError("training.amp_growth_interval must be positive")
    if len(settings.optimizer.betas) != 2:
        raise ValueError("optimizer.betas must have two values")
    if settings.loss.contrastive_weight < 0 or settings.loss.contrastive_temperature <= 0:
        raise ValueError("Contrastive weight must be non-negative and temperature positive")
    return settings


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("P06 DDP requires CUDA/NCCL")
        device = torch.device("cpu")
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def finalize_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class CompactOpticalImageNetStudent(nn.Module):
    """P05 optical/OEO trunk with a small multi-stage semantic readout."""

    def __init__(
        self,
        optical: OpticalConfig,
        *,
        selected_stage_indices: Iterable[int],
        pool_size: int = 4,
        projection_dim: int = 512,
        num_classes: int = 1000,
        source_num_classes: int = 10,
        classifier_mode: str = "projected_linear",
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = OpticalClassifier(optical, source_num_classes)
        self.selected_stage_indices = tuple(int(value) for value in selected_stage_indices)
        if min(self.selected_stage_indices) < 0 or max(self.selected_stage_indices) >= optical.num_stages:
            raise ValueError("selected_stage_indices are outside the optical trunk")
        self.pool_size = int(pool_size)
        features_per_stage = 2 * optical.input_channels * self.pool_size * self.pool_size
        self.descriptor_dim = len(self.selected_stage_indices) * features_per_stage
        self.classifier_mode = str(classifier_mode)
        self.descriptor_norm = nn.LayerNorm(self.descriptor_dim)
        self.projector = nn.Linear(self.descriptor_dim, int(projection_dim))
        if self.classifier_mode == "projected_linear":
            self.classifier = nn.Linear(int(projection_dim), int(num_classes))
        elif self.classifier_mode == "descriptor_linear":
            self.classifier = nn.Linear(self.descriptor_dim, int(num_classes))
        elif self.classifier_mode == "descriptor_mlp":
            hidden = int(classifier_hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(self.descriptor_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(classifier_dropout)),
                nn.Linear(hidden, int(num_classes)),
            )
        else:
            raise ValueError(f"Unsupported classifier_mode: {self.classifier_mode}")
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def load_source(
        self,
        checkpoint: Path,
        expected_sha256: str,
        *,
        load_mode: str = "strict",
    ) -> dict[str, Any]:
        actual = sha256_file(checkpoint)
        if actual != expected_sha256:
            raise RuntimeError(
                f"P05 source checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if load_mode == "strict":
            state = payload.get("model", payload)
            if any(str(key).startswith("module.") for key in state):
                state = {
                    str(key).removeprefix("module."): value
                    for key, value in state.items()
                }
            self.encoder.load_state_dict(state, strict=True)
        elif load_mode != "integrity_only":
            raise ValueError(f"Unsupported source checkpoint load mode: {load_mode}")
        self.encoder.head = nn.Identity()
        self.encoder.configure_feedback("bp")
        return {
            "path": str(checkpoint),
            "sha256": actual,
            "selected_epoch": payload.get("selected_epoch"),
            "best_validation_accuracy": payload.get("best_validation_accuracy"),
            "load_mode": load_mode,
        }

    def denormalize_clip_input(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() * self.clip_std + self.clip_mean).clamp(0.0, 1.0)

    def load_pretraining_checkpoint(
        self,
        checkpoint: Path,
        expected_sha256: str,
        *,
        load_mode: str = "strict",
    ) -> dict[str, Any]:
        actual = sha256_file(checkpoint)
        if actual != expected_sha256:
            raise RuntimeError(
                f"P06 initial checkpoint checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        if any(str(key).startswith("module.") for key in state):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        if load_mode == "strict":
            self.load_state_dict(state, strict=True)
            load_report = {
                "mode": "strict",
                "loaded_tensors": len(state),
                "missing_keys": [],
                "shape_mismatches": {},
                "unexpected_keys": [],
            }
        elif load_mode == "compatible":
            current = self.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in current and tuple(value.shape) == tuple(current[key].shape)
            }
            shape_mismatches = {
                key: {
                    "checkpoint": list(value.shape),
                    "model": list(current[key].shape),
                }
                for key, value in state.items()
                if key in current and tuple(value.shape) != tuple(current[key].shape)
            }
            missing = sorted(set(current) - set(compatible))
            encoder_missing = [key for key in missing if key.startswith("encoder.")]
            if encoder_missing:
                raise RuntimeError(
                    "Compatible P06 load must restore the complete optical encoder; "
                    f"missing {encoder_missing[:5]}"
                )
            self.load_state_dict(compatible, strict=False)
            load_report = {
                "mode": "compatible",
                "loaded_tensors": len(compatible),
                "missing_keys": missing,
                "shape_mismatches": shape_mismatches,
                "unexpected_keys": sorted(set(state) - set(current)),
            }
        elif load_mode == "expanded":
            current = self.state_dict()
            source_stage_indices = sorted(
                {
                    int(key.split(".")[2])
                    for key in state
                    if key.startswith("encoder.stages.")
                    and key.split(".")[2].isdigit()
                }
            )
            if not source_stage_indices or source_stage_indices != list(
                range(len(source_stage_indices))
            ):
                raise RuntimeError("Expanded P06 load requires contiguous source stages")
            source_stage_count = len(source_stage_indices)
            target_stage_count = len(self.encoder.stages)
            expanded: dict[str, torch.Tensor] = {}
            stage_mapping: list[dict[str, float | int]] = []

            def adapt_stage_tensor(
                value: torch.Tensor,
                target: torch.Tensor,
                suffix: str,
            ) -> torch.Tensor | None:
                if tuple(value.shape) == tuple(target.shape):
                    return value
                if (
                    suffix == "raw_phase"
                    and value.ndim == 3
                    and target.ndim == 3
                    and value.shape[0] == target.shape[0]
                ):
                    return F.interpolate(
                        value.float().unsqueeze(0),
                        size=target.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0).to(dtype=target.dtype)
                return None

            for target_index in range(target_stage_count):
                position = (
                    0.0
                    if target_stage_count == 1
                    else target_index
                    * (source_stage_count - 1)
                    / (target_stage_count - 1)
                )
                lower = int(math.floor(position))
                upper = int(math.ceil(position))
                weight = float(position - lower)
                stage_mapping.append(
                    {
                        "target": target_index,
                        "source_lower": lower,
                        "source_upper": upper,
                        "upper_weight": weight,
                    }
                )
                prefix = f"encoder.stages.{target_index}."
                for key, target in current.items():
                    if not key.startswith(prefix):
                        continue
                    suffix = key[len(prefix) :]
                    if suffix in {"propagator.transfer_function", "random_phase"}:
                        continue
                    lower_key = f"encoder.stages.{lower}.{suffix}"
                    upper_key = f"encoder.stages.{upper}.{suffix}"
                    if lower_key not in state or upper_key not in state:
                        continue
                    lower_value = adapt_stage_tensor(state[lower_key], target, suffix)
                    upper_value = adapt_stage_tensor(state[upper_key], target, suffix)
                    if lower_value is None or upper_value is None:
                        continue
                    if weight == 0.0 or not torch.is_floating_point(lower_value):
                        expanded[key] = lower_value
                    else:
                        expanded[key] = torch.lerp(
                            lower_value.float(), upper_value.float(), weight
                        ).to(dtype=target.dtype)

            for key, value in state.items():
                if key.startswith("encoder.stages."):
                    continue
                if key in current and tuple(value.shape) == tuple(current[key].shape):
                    expanded[key] = value

            parameter_names = set(dict(self.named_parameters()))
            missing_trainable_encoder = sorted(
                key
                for key in parameter_names
                if key.startswith("encoder.") and key not in expanded
            )
            if missing_trainable_encoder:
                raise RuntimeError(
                    "Expanded P06 load did not initialise all trainable encoder tensors; "
                    f"missing {missing_trainable_encoder[:5]}"
                )
            self.load_state_dict(expanded, strict=False)
            load_report = {
                "mode": "expanded",
                "loaded_tensors": len(expanded),
                "source_stage_count": source_stage_count,
                "target_stage_count": target_stage_count,
                "stage_mapping": stage_mapping,
                "missing_keys": sorted(set(current) - set(expanded)),
                "shape_mismatches": {},
                "unexpected_keys": sorted(set(state) - set(current)),
            }
        else:
            raise ValueError(f"Unsupported checkpoint load mode: {load_mode}")
        self.encoder.configure_feedback("bp")
        return {
            "path": str(checkpoint),
            "sha256": actual,
            "epoch": payload.get("epoch"),
            "best_validation_top1": payload.get("best_validation_top1"),
            "settings_digest": payload.get("settings_digest"),
            "load": load_report,
        }

    def forward(
        self,
        images: torch.Tensor,
        *,
        detach_backbone: bool = False,
        ablation: Ablation = "normal",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = self.denormalize_clip_input(images)
        if detach_backbone:
            with torch.no_grad():
                _, stages = self.encoder.forward_features(intensity, ablation=ablation)
        else:
            _, stages = self.encoder.forward_features(intensity, ablation=ablation)
        pooled = []
        size = (self.pool_size, self.pool_size)
        for index in self.selected_stage_indices:
            feature = stages[index]
            pooled.extend((F.adaptive_avg_pool2d(feature, size), F.adaptive_max_pool2d(feature, size)))
        descriptor = self.descriptor_norm(torch.cat(pooled, dim=1).flatten(1))
        projected = self.projector(descriptor)
        embedding = F.normalize(projected.float(), dim=-1)
        # CE receives the unnormalised feature so its logit scale is not
        # artificially capped; CLIP cosine/KD use the unit embedding.
        classifier_input = (
            projected if self.classifier_mode == "projected_linear" else descriptor
        )
        logits = self.classifier(classifier_input)
        return logits, embedding, descriptor

    def parameter_report(self) -> dict[str, Any]:
        phase = sum(parameter.numel() for parameter in self.encoder.phase_parameters())
        residual = sum(parameter.numel() for parameter in self.encoder.residual_parameters())
        head = sum(
            parameter.numel()
            for module in (self.descriptor_norm, self.projector, self.classifier)
            for parameter in module.parameters()
        )
        return {
            "phase_parameters": phase,
            "residual_electronic_parameters": residual,
            "pretraining_head_parameters": head,
            "total_trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "descriptor_dim": self.descriptor_dim,
            "classifier_mode": self.classifier_mode,
            "selected_stage_indices_zero_based": list(self.selected_stage_indices),
            "selected_stage_numbers": [index + 1 for index in self.selected_stage_indices],
            "optical_gates": self.encoder.optical_weights(),
            "minimum_optical_gate": min(self.encoder.optical_weights()),
        }


def unwrap(model: nn.Module) -> CompactOpticalImageNetStudent:
    return model.module if isinstance(model, DistributedDataParallel) else model


def stratified_base_indices(targets: list[int], per_class: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)
    selected = []
    for class_id in sorted(by_class):
        values = by_class[class_id]
        if len(values) < per_class:
            raise RuntimeError(
                f"Class {class_id} only has {len(values)} samples; requested {per_class}"
            )
        generator = random.Random(int(seed) + 104729 * class_id)
        selected.extend(generator.sample(values, per_class))
    return sorted(selected)


class SubsetEpochViewSampler(Sampler[int]):
    """DDP-safe one-view-per-image sampler over fixed full-cache indices.

    ``shuffle_block_size`` preserves sequential reads inside each block while
    randomising block order every epoch.  This is important for the Hugging
    Face ImageNet Arrow store: a global per-image permutation turns compressed
    image decoding into pathological random I/O.  The upstream train split is
    already class-shuffled, so local sequential runs remain class-diverse.
    """

    def __init__(
        self,
        dataset,
        base_indices: list[int],
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_block_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.base_indices = list(base_indices)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle_block_size = (
            None if shuffle_block_size is None else int(shuffle_block_size)
        )
        self.epoch = 0
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid rank/world_size")
        if self.shuffle_block_size is not None and self.shuffle_block_size < 1:
            raise ValueError("shuffle_block_size must be positive or None")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.base_indices) / self.world_size)

    def __iter__(self):
        indices = list(self.base_indices)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            if self.shuffle_block_size is None:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in order]
            else:
                block_size = self.shuffle_block_size
                blocks = [
                    indices[start : start + block_size]
                    for start in range(0, len(indices), block_size)
                ]
                block_order = torch.randperm(len(blocks), generator=generator).tolist()
                indices = [sample for block in block_order for sample in blocks[block]]
        total_size = math.ceil(len(indices) / self.world_size) * self.world_size
        if total_size > len(indices):
            indices.extend(indices[: total_size - len(indices)])
        views = int(self.dataset.views)
        for sample_index in indices[self.rank::self.world_size]:
            offset = view_seed(self.seed, sample_index, 0) % views
            view_index = (offset + self.epoch) % views
            yield sample_index * views + view_index


def make_loader(dataset, sampler, *, batch_size: int, settings: P06Settings) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": settings.training.num_workers,
        "pin_memory": settings.training.pin_memory,
        "persistent_workers": settings.training.persistent_workers and settings.training.num_workers > 0,
        "drop_last": False,
    }
    if settings.training.num_workers > 0:
        kwargs["prefetch_factor"] = settings.training.prefetch_factor
    return DataLoader(**kwargs)


def build_optimizer(model: CompactOpticalImageNetStudent, settings: P06Settings):
    phase = list(model.encoder.phase_parameters())
    residual = list(model.encoder.residual_parameters())
    head = [
        parameter
        for module in (model.descriptor_norm, model.projector, model.classifier)
        for parameter in module.parameters()
    ]
    return torch.optim.AdamW(
        [
            {"params": phase, "lr": settings.optimizer.phase_learning_rate, "weight_decay": 0.0, "name": "phase"},
            {"params": residual, "lr": settings.optimizer.residual_learning_rate, "weight_decay": settings.optimizer.weight_decay, "name": "residual"},
            {"params": head, "lr": settings.optimizer.head_learning_rate, "weight_decay": settings.optimizer.weight_decay, "name": "head"},
        ],
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
    )


def build_scheduler(optimizer, settings: P06Settings, steps_per_epoch: int):
    total_steps = max(1, settings.training.epochs * steps_per_epoch)
    warmup_steps = min(settings.optimizer.warmup_steps, max(0, total_steps - 1))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        minimum = settings.optimizer.minimum_learning_rate_ratio
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def batch_contrastive_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Match each student to its paired teacher instead of the CLIP mean direction."""

    scores = student_embedding.float() @ teacher_embedding.float().T / float(temperature)
    targets = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (
        F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)
    )


def compute_losses(
    logits: torch.Tensor,
    embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    clip_logit_scale: float,
    settings: P06Settings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher = F.normalize(teacher_embedding.float(), dim=-1)
    cosine = (embedding.float() * teacher).sum(-1)
    feature = (1.0 - cosine).mean()
    student_clip_logits = float(clip_logit_scale) * embedding.float() @ text_prototypes.T
    teacher_clip_logits = float(clip_logit_scale) * teacher @ text_prototypes.T
    temperature = settings.loss.distill_temperature
    kd = F.kl_div(
        F.log_softmax(student_clip_logits / temperature, dim=-1),
        F.softmax(teacher_clip_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    contrastive = batch_contrastive_loss(
        embedding, teacher, settings.loss.contrastive_temperature
    )
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=settings.loss.label_smoothing)
    total = (
        settings.loss.supervised_ce_weight * ce
        + settings.loss.feature_cosine_weight * feature
        + settings.loss.clip_logit_kd_weight * kd
        + settings.loss.contrastive_weight * contrastive
    )
    return total, {
        "loss_total": total.detach(),
        "loss_ce": ce.detach(),
        "loss_feature": feature.detach(),
        "loss_kd": kd.detach(),
        "loss_contrastive": contrastive.detach(),
        "clip_cosine": cosine.mean().detach(),
        "zero_shot_correct": student_clip_logits.argmax(-1).eq(labels).sum().detach(),
        "teacher_zero_shot_correct": teacher_clip_logits.argmax(-1).eq(labels).sum().detach(),
    }


def _reduce(values: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return values


def _metric_vector(device: torch.device) -> torch.Tensor:
    # samples, losses x5, cosine, classifier top1/top5, student/teacher zero-shot, batches
    return torch.zeros(12, dtype=torch.float64, device=device)


def _update_metrics(vector, logits, labels, losses, batch_size: int) -> None:
    vector[0] += batch_size
    vector[1] += float(losses["loss_total"]) * batch_size
    vector[2] += float(losses["loss_ce"]) * batch_size
    vector[3] += float(losses["loss_feature"]) * batch_size
    vector[4] += float(losses["loss_kd"]) * batch_size
    vector[5] += float(losses["loss_contrastive"]) * batch_size
    vector[6] += float(losses["clip_cosine"]) * batch_size
    topk = logits.detach().topk(min(5, logits.shape[-1]), dim=-1).indices
    vector[7] += topk[:, :1].eq(labels[:, None]).any(-1).sum()
    vector[8] += topk.eq(labels[:, None]).any(-1).sum()
    vector[9] += losses["zero_shot_correct"]
    vector[10] += losses["teacher_zero_shot_correct"]
    vector[11] += 1


def _metrics(vector: torch.Tensor, seconds: float) -> dict[str, float]:
    vector = _reduce(vector)
    count = max(float(vector[0]), 1.0)
    return {
        "samples": int(vector[0]),
        "loss_total": float(vector[1] / count),
        "loss_ce": float(vector[2] / count),
        "loss_feature": float(vector[3] / count),
        "loss_kd": float(vector[4] / count),
        "loss_contrastive": float(vector[5] / count),
        "clip_cosine": float(vector[6] / count),
        "top1_accuracy": float(vector[7] / count),
        "top5_accuracy": float(vector[8] / count),
        "clip_zero_shot_top1": float(vector[9] / count),
        "teacher_zero_shot_top1": float(vector[10] / count),
        "batches": int(vector[11]),
        "seconds": float(seconds),
    }


def _phase_gradient_report(model: CompactOpticalImageNetStudent) -> dict[str, Any]:
    norms = []
    finite = []
    for stage in model.encoder.stages:
        gradient = stage.raw_phase.grad
        norms.append(float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0)
        finite.append(bool(gradient is not None and torch.isfinite(gradient).all()))
    return {
        "norms": norms,
        "all_finite": all(finite),
        "all_nonzero": all(value > 0.0 for value in norms),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    text_prototypes,
    clip_logit_scale: float,
    settings: P06Settings,
    context: DistributedContext,
    *,
    epoch: int,
    head_only: bool,
) -> tuple[dict[str, float], dict[str, Any] | None, dict[str, float] | None]:
    model.train()
    vector = _metric_vector(context.device)
    started = time.perf_counter()
    gradient_report = None
    input_report = None
    limit = settings.training.max_train_batches
    total_batches = min(len(loader), limit) if limit is not None else len(loader)
    for batch_index, batch in enumerate(loader, 1):
        if limit is not None and batch_index > limit:
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        teacher = batch["teacher_embedding"].to(context.device, non_blocking=True)
        if batch_index == 1:
            intensity = unwrap(model).denormalize_clip_input(images)
            input_report = {
                "normalized_min": float(images.min().detach().cpu()),
                "normalized_max": float(images.max().detach().cpu()),
                "intensity_min": float(intensity.min().detach().cpu()),
                "intensity_max": float(intensity.max().detach().cpu()),
            }
        optimizer.zero_grad(set_to_none=True)
        amp = settings.training.use_amp and context.device.type == "cuda"
        with torch.autocast(device_type=context.device.type, dtype=torch.float16, enabled=amp):
            logits, embedding, _ = model(images, detach_backbone=head_only)
            loss, losses = compute_losses(
                logits, embedding, teacher, labels, text_prototypes, clip_logit_scale, settings
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not head_only and (
            gradient_report is None or not gradient_report["all_finite"]
        ):
            # The first AMP-scaled batch may legitimately overflow. Keep
            # auditing until a real finite phase update is observed instead
            # of permanently reporting the skipped batch as the epoch result.
            gradient_report = _phase_gradient_report(unwrap(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.optimizer.gradient_clip_norm)
        scale_before_step = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        # AMP deliberately skips optimizer.step when it detects overflow.  Do
        # not advance the batch scheduler on that skipped update, otherwise the
        # first high-scale batch silently shortens warm-up (and PyTorch warns
        # that scheduler.step preceded optimizer.step).
        optimizer_updated = not scaler.is_enabled() or float(scaler.get_scale()) >= scale_before_step
        if optimizer_updated:
            scheduler.step()
        elif context.is_main:
            print(
                f"[amp] skipped non-finite optimizer update at epoch={epoch} batch={batch_index}; "
                f"scale={scale_before_step:.1f}->{float(scaler.get_scale()):.1f}",
                flush=True,
            )
        _update_metrics(vector, logits, labels, losses, labels.numel())
        if context.is_main and (
            batch_index % settings.training.log_interval_batches == 0 or batch_index == total_batches
        ):
            elapsed = time.perf_counter() - started
            print(
                f"[train] epoch={epoch}/{settings.training.epochs} stage={'head' if head_only else 'joint'} "
                f"batch={batch_index}/{total_batches} loss={float(loss):.4f} "
                f"top1={float(vector[7] / vector[0]):.4f} cos={float(vector[6] / vector[0]):.4f} "
                f"lr_phase={optimizer.param_groups[0]['lr']:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return _metrics(vector, time.perf_counter() - started), gradient_report, input_report


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    text_prototypes,
    clip_logit_scale: float,
    settings: P06Settings,
    context: DistributedContext,
    *,
    ablation: Ablation = "normal",
) -> dict[str, float]:
    model.eval()
    vector = _metric_vector(context.device)
    started = time.perf_counter()
    limit = settings.training.max_validation_batches
    for batch_index, batch in enumerate(loader, 1):
        if limit is not None and batch_index > limit:
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        teacher = batch["teacher_embedding"].to(context.device, non_blocking=True)
        amp = settings.training.use_amp and context.device.type == "cuda"
        with torch.autocast(device_type=context.device.type, dtype=torch.float16, enabled=amp):
            logits, embedding, _ = model(images, ablation=ablation)
            _, losses = compute_losses(
                logits, embedding, teacher, labels, text_prototypes, clip_logit_scale, settings
            )
        _update_metrics(vector, logits, labels, losses, labels.numel())
    return _metrics(vector, time.perf_counter() - started)


def save_checkpoint(path, model, optimizer, scheduler, scaler, *, epoch, best_top1, history, settings):
    atomic_torch_save(
        path,
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_validation_top1": float(best_top1),
            "history": history,
            "settings_digest": settings.digest(),
            "source_checkpoint_sha256": settings.source_checkpoint_sha256,
        },
    )


def load_checkpoint(path, model, optimizer, scheduler, scaler, settings, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("settings_digest") != settings.digest():
        raise RuntimeError("Resume checkpoint settings digest does not match the current P06 config")
    unwrap(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    return int(payload["epoch"]) + 1, float(payload["best_validation_top1"]), list(payload["history"])


def _gate_report(
    normal: dict[str, float],
    ablations: dict[str, dict[str, float]],
    phase_gradients: dict[str, Any] | None,
    model_report: dict[str, Any],
    settings: P06Settings,
) -> dict[str, Any]:
    disrupted = max((value["top1_accuracy"] for value in ablations.values()), default=normal["top1_accuracy"])
    relative_drop = 0.0 if normal["top1_accuracy"] <= 0 else 1.0 - disrupted / normal["top1_accuracy"]
    checks = {
        "validation_top1": normal["top1_accuracy"] >= settings.gates.validation_top1_min,
        "clip_cosine": normal["clip_cosine"] >= settings.gates.clip_cosine_min,
        "phase_gradients": bool(
            phase_gradients
            and phase_gradients.get("all_finite")
            and phase_gradients.get("all_nonzero")
        ),
        "optical_gate_floor": model_report["minimum_optical_gate"] >= settings.gates.optical_gate_min,
        "optical_disruption": relative_drop >= settings.gates.optical_disruption_relative_drop_min,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "optical_disruption_relative_drop": relative_drop,
        "thresholds": asdict(settings.gates),
    }


def run(settings: P06Settings, context: DistributedContext, *, resume: bool) -> dict[str, Any]:
    seed_everything(settings.training.seed, context.rank)
    imagenet_settings = load_imagenet_settings(settings.imagenet_config)
    if imagenet_settings.model.num_classes != settings.model.num_classes:
        raise RuntimeError("ImageNet cache and P06 class counts differ")
    bundle = load_imagenet(imagenet_settings)
    train_store = ClipFeatureStore("train", bundle.train, bundle, imagenet_settings)
    validation_store = ClipFeatureStore("validation", bundle.validation, bundle, imagenet_settings)
    train_dataset = DistillationViewDataset(bundle.train, train_store)
    validation_dataset = DistillationViewDataset(bundle.validation, validation_store)
    if settings.training.train_samples_per_class is None:
        train_indices = list(range(bundle.train.base_sample_count))
    else:
        train_indices = stratified_base_indices(
            bundle.train.targets,
            settings.training.train_samples_per_class,
            settings.training.seed,
        )
    validation_indices = stratified_base_indices(
        bundle.validation.targets,
        settings.training.validation_samples_per_class,
        settings.training.seed + 1,
    )
    train_sampler = SubsetEpochViewSampler(
        bundle.train,
        train_indices,
        shuffle=True,
        seed=settings.training.seed,
        rank=context.rank,
        world_size=context.world_size,
        shuffle_block_size=settings.training.shuffle_block_size,
    )
    validation_sampler = SubsetEpochViewSampler(
        bundle.validation,
        validation_indices,
        shuffle=False,
        seed=settings.training.seed,
        rank=context.rank,
        world_size=context.world_size,
    )
    train_loader = make_loader(train_dataset, train_sampler, batch_size=settings.training.batch_size, settings=settings)
    validation_loader = make_loader(
        validation_dataset,
        validation_sampler,
        batch_size=settings.training.validation_batch_size,
        settings=settings,
    )

    architecture = load_formal_settings(settings.architecture_config).base
    model = CompactOpticalImageNetStudent(
        architecture.optical,
        selected_stage_indices=settings.model.selected_stage_indices,
        pool_size=settings.model.pool_size,
        projection_dim=settings.model.projection_dim,
        num_classes=settings.model.num_classes,
        source_num_classes=architecture.num_classes,
        classifier_mode=settings.model.classifier_mode,
        classifier_hidden_dim=settings.model.classifier_hidden_dim,
        classifier_dropout=settings.model.classifier_dropout,
    )
    source_report = model.load_source(
        settings.source_checkpoint,
        settings.source_checkpoint_sha256,
        load_mode=settings.source_checkpoint_load_mode,
    )
    initial_report = None
    if settings.initial_checkpoint is not None:
        if settings.initial_checkpoint_sha256 is None:
            raise RuntimeError("Missing checksum for P06 initial checkpoint")
        initial_report = model.load_pretraining_checkpoint(
            settings.initial_checkpoint,
            settings.initial_checkpoint_sha256,
            load_mode=settings.initial_checkpoint_load_mode,
        )
    model.to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=settings.training.head_warmup_epochs > 0,
        )
    optimizer = build_optimizer(unwrap(model), settings)
    steps_per_epoch = min(
        len(train_loader),
        settings.training.max_train_batches or len(train_loader),
    )
    scheduler = build_scheduler(optimizer, settings, steps_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=settings.training.use_amp and context.device.type == "cuda",
        init_scale=settings.training.amp_initial_scale,
        growth_interval=settings.training.amp_growth_interval,
    )
    prototypes_path = cache_directory(imagenet_settings) / "imagenet_text_prototypes.pt"
    text_prototypes, clip_logit_scale = load_text_prototypes(
        prototypes_path, bundle.class_names, imagenet_settings, context.device
    )
    model_report = unwrap(model).parameter_report()
    manifest = {
        "settings_digest": settings.digest(),
        "world_size": context.world_size,
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
        "train_cached_views_per_image": bundle.train.views,
        "train_shuffle_block_size": settings.training.shuffle_block_size,
        "imagenet_dataset_digest": bundle.digest,
        "clip_cache_directory": str(cache_directory(imagenet_settings)),
        "source": source_report,
        "initial_checkpoint": initial_report,
        "model": model_report,
    }
    if context.is_main:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(settings.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)

    best_path = settings.output_dir / "checkpoints" / "best.pt"
    last_path = settings.output_dir / "checkpoints" / "last.pt"
    start_epoch = 1
    best_top1 = -math.inf
    history: list[dict[str, Any]] = []
    resumed = False
    if resume and last_path.is_file():
        start_epoch, best_top1, history = load_checkpoint(
            last_path, model, optimizer, scheduler, scaler, settings, context.device
        )
        resumed = True
        if context.is_main:
            print(f"[resume] checkpoint={last_path} start_epoch={start_epoch}", flush=True)

    # A refinement is never allowed to silently replace its source with a
    # worse checkpoint. Evaluate and register the immutable input as epoch 0;
    # later epochs must exceed this validation Top-1 to become best.
    if initial_report is not None and not resumed:
        validation_sampler.set_epoch(0)
        baseline = evaluate(
            model,
            validation_loader,
            text_prototypes,
            clip_logit_scale,
            settings,
            context,
        )
        best_top1 = baseline["top1_accuracy"]
        if context.is_main:
            write_json(settings.output_dir / "metrics" / "initial_baseline.json", baseline)
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                best_top1=best_top1,
                history=history,
                settings=settings,
            )
            print(
                f"[baseline] epoch=0 val_top1={baseline['top1_accuracy']:.4f} "
                f"val_top5={baseline['top5_accuracy']:.4f} "
                f"student_zero={baseline['clip_zero_shot_top1']:.4f} "
                f"teacher_zero={baseline['teacher_zero_shot_top1']:.4f}",
                flush=True,
            )
        best_tensor = torch.tensor(best_top1, device=context.device)
        if context.world_size > 1:
            torch.distributed.broadcast(best_tensor, src=0)
        best_top1 = float(best_tensor)
        barrier()

    latest_phase_gradients = next(
        (
            row["phase_gradients"]
            for row in reversed(history)
            if row.get("phase_gradients") is not None
        ),
        None,
    )
    for epoch in range(start_epoch, settings.training.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        head_only = epoch <= settings.training.head_warmup_epochs
        train_metrics, phase_gradients, input_report = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            text_prototypes,
            clip_logit_scale,
            settings,
            context,
            epoch=epoch,
            head_only=head_only,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            text_prototypes,
            clip_logit_scale,
            settings,
            context,
        )
        if phase_gradients is not None:
            latest_phase_gradients = phase_gradients
        row = {
            "epoch": epoch,
            "stage": "head_warmup" if head_only else "joint_bp",
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "train": train_metrics,
            "validation": validation_metrics,
            "phase_gradients": phase_gradients,
            "input_range": input_report,
            "optical_gates": unwrap(model).encoder.optical_weights(),
        }
        if context.is_main:
            history.append(row)
            write_json(settings.output_dir / "metrics" / "history.json", history)
            write_json(settings.output_dir / "metrics" / "latest.json", row)
            # The warm-up is an optimisation aid, not a backbone checkpoint.
            # Select the reusable source only after exact BP reaches all stages.
            if not head_only and validation_metrics["top1_accuracy"] > best_top1:
                best_top1 = validation_metrics["top1_accuracy"]
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_top1=best_top1,
                    history=history,
                    settings=settings,
                )
            if head_only:
                save_checkpoint(
                    settings.output_dir / "checkpoints" / "head_warmup.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_top1=best_top1,
                    history=history,
                    settings=settings,
                )
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_top1=best_top1,
                history=history,
                settings=settings,
            )
            if epoch % settings.training.checkpoint_interval_epochs == 0:
                save_checkpoint(
                    settings.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    best_top1=best_top1,
                    history=history,
                    settings=settings,
                )
            print(
                f"[epoch] {epoch}/{settings.training.epochs} stage={row['stage']} "
                f"train_top1={train_metrics['top1_accuracy']:.4f} "
                f"val_top1={validation_metrics['top1_accuracy']:.4f} "
                f"val_top5={validation_metrics['top5_accuracy']:.4f} "
                f"val_cos={validation_metrics['clip_cosine']:.4f} best={best_top1:.4f}",
                flush=True,
            )
        best_tensor = torch.tensor(best_top1, device=context.device)
        if context.world_size > 1:
            torch.distributed.broadcast(best_tensor, src=0)
        best_top1 = float(best_tensor)
        barrier()

    barrier()
    best_payload = torch.load(best_path, map_location=context.device, weights_only=False)
    unwrap(model).load_state_dict(best_payload["model"], strict=True)
    normal = evaluate(
        model, validation_loader, text_prototypes, clip_logit_scale, settings, context
    )
    ablations: dict[str, dict[str, float]] = {}
    if settings.training.run_final_ablations:
        for name in ("optical_off", "phase_random"):
            ablations[name] = evaluate(
                model,
                validation_loader,
                text_prototypes,
                clip_logit_scale,
                settings,
                context,
                ablation=name,
            )
    final_model_report = unwrap(model).parameter_report()
    gates = _gate_report(normal, ablations, latest_phase_gradients, final_model_report, settings)
    result = {
        "status": "complete",
        "best_epoch": int(best_payload["epoch"]),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path) if context.is_main else None,
        "validation": normal,
        "ablations": ablations,
        "phase_gradients": latest_phase_gradients,
        "model": final_model_report,
        "gates": gates,
    }
    if context.is_main:
        write_json(settings.output_dir / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    barrier()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P06 compact optical ImageNet/CLIP pretraining")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_p06_settings(args.config)
    context = initialize_distributed()
    try:
        run(settings, context, resume=args.resume)
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
