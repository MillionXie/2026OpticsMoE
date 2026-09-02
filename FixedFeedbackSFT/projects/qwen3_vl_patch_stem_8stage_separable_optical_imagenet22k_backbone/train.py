"""Independent ImageNet-21K/22K supervised trainer for the eight-stage P11.

This file intentionally does not reuse the mutable ImageNet-1K continuation
entry point.  Checkpoint formats, implementation manifests, dataset identity
locks and output names are separate, so a large-vocabulary run cannot alter or
resume the currently running P11 formal experiment by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .dataset import (
    ClassFolderMMapDataset,
    DatasetContractError,
    GlobalAffineDistributedSampler,
    image_transform,
    load_plumbing_imagenet1k,
    sha256_file,
    verify_index,
)
from .initialization import (
    construct_large_vocabulary_model,
    initialize_from_frozen_p11_backbone,
    state_dict_sha256,
)


CHECKPOINT_FORMAT = "p11-optical-imagenet-large-supervised-v1"
BACKBONE_FORMAT = "p11-optical-imagenet-large-backbone-v1"
PROJECT_RELATIVE = (
    "FixedFeedbackSFT/projects/"
    "qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone"
)
IMPLEMENTATION_FILES = (
    "experiments/__init__.py",
    f"{PROJECT_RELATIVE}/__init__.py",
    f"{PROJECT_RELATIVE}/dataset.py",
    f"{PROJECT_RELATIVE}/initialization.py",
    f"{PROJECT_RELATIVE}/train.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/stem.py",
    "FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/optics.py",
    # Used only by the explicitly labelled ImageNet-1K plumbing path.
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/datasets.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/settings.py",
)


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "FixedFeedbackSFT").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPOSITORY_ROOT = repository_root()


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as error:
        raise RuntimeError("PyYAML is required") from error
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    value["_config_path"] = str(config_path)
    digest_value = {key: item for key, item in value.items() if not key.startswith("_")}
    value["_config_digest"] = canonical_sha256(digest_value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def implementation_manifest() -> dict[str, Any]:
    records = []
    for relative in IMPLEMENTATION_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Implementation dependency is missing: {path}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "format": "training-implementation-manifest-v1",
        "files": records,
        "aggregate_sha256": canonical_sha256(records),
    }


class Context:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        if self.distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            if self.distributed:
                raise RuntimeError("Distributed large-vocabulary training requires CUDA/NCCL")
            self.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def training_rank_seed(seed: int, rank: int) -> int:
    """Decorrelate augmentation/mixing streams after deterministic model init."""

    return int(seed) + int(rank) * 1_000_003


def rng_state(context: Context) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None
        ),
    }


def gather_rng_states(context: Context) -> list[dict[str, Any]]:
    local = rng_state(context)
    if not context.distributed:
        return [local]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, local)
    return list(gathered)


def restore_rng_state(value: Mapping[str, Any], context: Context) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    if context.device.type == "cuda" and value.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(value["torch_cuda"], context.device)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _validate_index_against_config(
    manifest: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    *,
    role: str,
) -> None:
    source_value = manifest.get("source_root")
    if not isinstance(source_value, str) or not source_value.strip():
        raise DatasetContractError(f"{role} index manifest has no source_root")
    source_root = Path(source_value).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"{role} index source_root is unavailable or not a directory: {source_root}"
        )
    expected = {
        "variant_id": dataset_config["variant_id"],
        "release_id": dataset_config["release_id"],
        "num_classes": int(dataset_config["num_classes"]),
        "split_id": dataset_config[f"{role}_split_id"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise DatasetContractError(
                f"{role} index {key}={manifest.get(key)!r}; config requires {value!r}"
            )
    declared_count = dataset_config.get(f"expected_{role}_samples")
    if declared_count is not None and int(manifest["num_samples"]) != int(declared_count):
        raise DatasetContractError(
            f"{role} index has {int(manifest['num_samples']):,} samples; "
            f"config requires {int(declared_count):,}"
        )


def _taxonomy_identity(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove every split uses the same WNIDs in exactly the same label order."""

    if not manifests:
        raise DatasetContractError("At least one index manifest is required")
    keys = ("num_classes", "class_list_sha256", "class_to_idx_sha256")
    reference = {key: manifests[0].get(key) for key in keys}
    for index, manifest in enumerate(manifests[1:], 1):
        actual = {key: manifest.get(key) for key in keys}
        if actual != reference:
            raise DatasetContractError(
                "Train/validation taxonomy mismatch at manifest "
                f"{index}: expected {reference}, got {actual}"
            )
    if not all(reference.values()):
        raise DatasetContractError("Index manifest has incomplete taxonomy hashes")
    return {
        **reference,
        "taxonomy_digest": canonical_sha256(reference),
    }


def preflight(config: Mapping[str, Any], *, verify_large_files: bool = False) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    model = config.get("model", {})
    mode = str(dataset.get("mode", ""))
    target_classes = int(model.get("num_classes", 0))
    if int(dataset.get("num_classes", -1)) != target_classes:
        raise DatasetContractError("dataset.num_classes and model.num_classes must match")
    if config.get("loss", {}).get("mode") != "soft_target_cross_entropy":
        raise RuntimeError("Large-vocabulary recipes are locked to soft-target cross-entropy")
    if mode == "plumbing_smoke_only":
        if dataset.get("publishable_result") is not False:
            raise DatasetContractError("Plumbing smoke must set publishable_result: false")
        if target_classes != 21_841:
            raise DatasetContractError("The plumbing config exercises the real 21841-way head")
        maximum = training.get("max_train_batches")
        if maximum is None or not 1 <= int(maximum) <= 100:
            raise DatasetContractError("Plumbing smoke is capped at 100 train batches")
        imagenet_config = resolve_path(dataset["imagenet1k_config"])
        if not imagenet_config.is_file():
            raise FileNotFoundError(f"ImageNet-1K plumbing config is missing: {imagenet_config}")
        data_identity = {
            "mode": mode,
            "publishable_result": False,
            "warning": "ImageNet-1K images with a 21841-way head; not an IN-22K metric",
        }
    elif mode == "indexed_class_folder":
        train_index = resolve_path(dataset["train_index"])
        if not train_index.is_dir():
            raise FileNotFoundError(
                f"Formal large-data index is absent: {train_index}. "
                "No ImageNet-21K/22K run may start without a verified manifest."
            )
        train_manifest = verify_index(train_index, verify_large_files=verify_large_files)
        _validate_index_against_config(train_manifest, dataset, role="train")
        evaluation_enabled = bool(config.get("evaluation", {}).get("enabled", False))
        validation_manifest = None
        if evaluation_enabled:
            validation_value = dataset.get("validation_index")
            if not validation_value:
                raise DatasetContractError("Evaluation is enabled but validation_index is unset")
            validation_index = resolve_path(validation_value)
            if not validation_index.is_dir():
                raise FileNotFoundError(f"Validation index is absent: {validation_index}")
            validation_manifest = verify_index(
                validation_index,
                verify_large_files=verify_large_files,
            )
            _validate_index_against_config(validation_manifest, dataset, role="validation")
        taxonomy = _taxonomy_identity(
            [
                train_manifest,
                *([validation_manifest] if validation_manifest is not None else []),
            ]
        )
        data_identity = {
            "mode": mode,
            "publishable_result": True,
            "train_manifest_sha256": train_manifest["index_manifest_sha256"],
            "train_variant_id": train_manifest["variant_id"],
            "train_num_classes": train_manifest["num_classes"],
            "train_num_samples": train_manifest["num_samples"],
            "validation_manifest_sha256": (
                validation_manifest["index_manifest_sha256"] if validation_manifest else None
            ),
            "validation_num_samples": (
                validation_manifest["num_samples"] if validation_manifest else None
            ),
            **taxonomy,
        }
    else:
        raise DatasetContractError(f"Unsupported dataset.mode: {mode!r}")

    initialization = config.get("initialization", {})
    backbone = resolve_path(initialization["backbone_checkpoint"])
    stem = resolve_path(config["stem_checkpoint"])
    if not backbone.is_file():
        raise FileNotFoundError(f"Frozen P11 backbone is missing: {backbone}")
    if sha256_file(backbone) != initialization.get("expected_backbone_sha256"):
        raise RuntimeError("Frozen P11 backbone SHA does not match the config")
    if not stem.is_file():
        raise FileNotFoundError(f"Frozen Qwen stem is missing: {stem}")
    if sha256_file(stem) != initialization.get("expected_stem_sha256"):
        raise RuntimeError("Frozen Qwen stem SHA does not match the config")

    output = resolve_path(config["output_dir"])
    disk = shutil.disk_usage(_nearest_existing_parent(output))
    free_gib = disk.free / (1024**3)
    required_gib = float(training.get("minimum_free_disk_gib", 100.0))
    if free_gib < required_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free; recipe requires {required_gib:.1f} GiB"
        )
    expected_world = training.get("expected_world_size")
    return {
        "ok": True,
        "dataset": data_identity,
        "output_dir": str(output),
        "free_disk_gib": free_gib,
        "required_free_disk_gib": required_gib,
        "expected_world_size": expected_world,
        "preflight_created_output": False,
    }


def build_datasets(config: Mapping[str, Any]):
    values = config["dataset"]
    mode = values["mode"]
    if mode == "plumbing_smoke_only":
        train, identity = load_plumbing_imagenet1k(
            resolve_path(values["imagenet1k_config"]),
            limit=int(values.get("plumbing_sample_limit", 3_200)),
            seed=int(config["training"].get("seed", 2026)),
        )
        return train, None, identity
    train_index = resolve_path(values["train_index"])
    train = ClassFolderMMapDataset(
        train_index,
        transform=image_transform(train=True),
        verify_large_files=False,
    )
    validation = None
    if bool(config.get("evaluation", {}).get("enabled", False)):
        validation = ClassFolderMMapDataset(
            resolve_path(values["validation_index"]),
            transform=image_transform(train=False),
            verify_large_files=False,
        )
    identity = {
        "mode": mode,
        "publishable_result": True,
        "train_manifest_sha256": train.manifest["index_manifest_sha256"],
        "train_variant_id": train.manifest["variant_id"],
        "train_num_classes": train.manifest["num_classes"],
        "train_num_samples": len(train),
        "validation_manifest_sha256": (
            validation.manifest["index_manifest_sha256"] if validation else None
        ),
        "validation_num_samples": len(validation) if validation else None,
        **_taxonomy_identity(
            [
                train.manifest,
                *([validation.manifest] if validation is not None else []),
            ]
        ),
    }
    return train, validation, identity


def make_loader(
    dataset,
    sampler,
    values: Mapping[str, Any],
    *,
    train: bool,
) -> DataLoader:
    workers = int(values.get("num_workers", 8))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(values["batch_size"] if train else values["validation_batch_size"]),
        "sampler": sampler,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(values.get("pin_memory", True)),
        "drop_last": bool(train),
        "persistent_workers": bool(values.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(values.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def _no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim <= 1 or lowered.endswith(".bias") or "norm" in lowered or "logit" in lowered


def build_layerwise_optimizer(model: nn.Module, config: Mapping[str, Any]):
    values = config["optimizer"]
    stage_count = len(model.stages)
    layer_decay = float(values.get("layer_decay", 0.9))
    weight_decay = float(values.get("weight_decay", 0.05))
    bases = {
        "phase": float(values["phase_learning_rate"]),
        "electronic": float(values["electronic_learning_rate"]),
        "adapter": float(values.get("adapter_learning_rate", values["electronic_learning_rate"])),
        "head": float(values["head_learning_rate"]),
    }
    grouped: dict[tuple[str, int, bool], list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        depth = 0
        family = "electronic"
        if name.startswith("readout."):
            family, depth = "head", stage_count + 1
        elif name.startswith("adapter."):
            family, depth = "adapter", 0
        elif name.startswith("stages."):
            parts = name.split(".")
            depth = int(parts[1]) + 1
            family = "phase" if parts[2] == "raw_phase" else "electronic"
        decay_weights = family != "phase" and not _no_weight_decay(name, parameter)
        grouped.setdefault((family, depth, decay_weights), []).append((name, parameter))
    groups = []
    schema = []
    for (family, depth, decay_weights), named in sorted(grouped.items()):
        if family in {"head", "phase", "electronic"} and depth > 0:
            scale = layer_decay ** max(stage_count - depth, 0)
        elif family == "adapter":
            scale = layer_decay ** (stage_count + 1)
        else:
            scale = 1.0
        learning_rate = bases[family] * scale
        name = f"{family}.depth{depth}.{'decay' if decay_weights else 'no_decay'}"
        parameters = [parameter for _, parameter in named]
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay if decay_weights else 0.0,
                "name": name,
            }
        )
        schema.append(
            {
                "name": name,
                "family": family,
                "depth": depth,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay if decay_weights else 0.0,
                "parameter_tensors": len(parameters),
                "parameter_elements": sum(value.numel() for value in parameters),
            }
        )
    assigned = sum(len(group["params"]) for group in groups)
    expected = sum(1 for value in model.parameters() if value.requires_grad)
    if assigned != expected:
        raise RuntimeError("Layer-wise optimizer does not partition parameters exactly")
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in values.get("betas", [0.9, 0.999])),
        eps=float(values.get("eps", 1.0e-8)),
    )
    return optimizer, schema


def build_scheduler(optimizer, config: Mapping[str, Any], updates_per_epoch: int):
    values = config["optimizer"]
    total = max(int(config["training"]["epochs"]) * updates_per_epoch, 1)
    warmup = int(values.get("warmup_epochs", 5)) * updates_per_epoch
    minimum = float(values.get("minimum_learning_rate_ratio", 0.01))

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1.0 / warmup)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


class TrainableEMA:
    def __init__(self, model: nn.Module, *, decay: float, warmup_updates: int) -> None:
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
        decay = self.decay
        if self.warmup_updates > 0:
            decay *= 1.0 - math.exp(-self.updates / self.warmup_updates)
        live = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(live[name].detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_updates": self.warmup_updates,
            "updates": self.updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if float(value["decay"]) != self.decay or int(value["warmup_updates"]) != self.warmup_updates:
            raise RuntimeError("EMA recipe differs from resume checkpoint")
        shadow = value["shadow"]
        if shadow.keys() != self.shadow.keys():
            raise RuntimeError("EMA tensor names differ from resume checkpoint")
        for name, tensor in shadow.items():
            if tuple(tensor.shape) != tuple(self.shadow[name].shape):
                raise RuntimeError(f"EMA tensor shape mismatch: {name}")
            self.shadow[name].copy_(tensor)
        self.updates = int(value["updates"])

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        live = dict(model.named_parameters())
        backup = {name: live[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    live[name].copy_(average)
            yield
        finally:
            with torch.no_grad():
                for name, tensor in backup.items():
                    live[name].copy_(tensor)


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Soft targets {targets.shape} do not match logits {logits.shape}")
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def _one_hot(labels: torch.Tensor, classes: int, smoothing: float) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0,1)")
    target = F.one_hot(labels, num_classes=classes).float()
    return target.mul(1.0 - smoothing).add(smoothing / classes)


def mixed_targets_and_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes: int,
    values: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    smoothing = float(values.get("label_smoothing", 0.0))
    target = _one_hot(labels, classes, smoothing)
    mixup_alpha = float(values.get("mixup_alpha", 0.0))
    cutmix_alpha = float(values.get("cutmix_alpha", 0.0))
    probability = float(values.get("batch_mix_probability", 1.0))
    if images.shape[0] < 2 or random.random() >= probability or max(mixup_alpha, cutmix_alpha) <= 0:
        return images, target
    permutation = torch.randperm(images.shape[0], device=images.device)
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = random.betavariate(alpha, alpha)
    if use_cutmix:
        height, width = images.shape[-2:]
        ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        center_y, center_x = random.randrange(height), random.randrange(width)
        y0, y1 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
        x0, x1 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
        mixed = images.clone()
        mixed[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
        lam = 1.0 - ((y1 - y0) * (x1 - x0) / (height * width))
        images = mixed
    else:
        images = images.mul(lam).add(images[permutation], alpha=1.0 - lam)
    return images, target.mul(lam).add(target[permutation], alpha=1.0 - lam)


def _reduce(values: torch.Tensor, context: Context) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def topk_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    predictions = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    matches = predictions.eq(labels.view(-1, 1))
    return float(matches[:, :1].sum()), float(matches.any(dim=1).sum())


def accumulation_window_size(batch_index: int, total_batches: int, accumulation: int) -> int:
    """Return the true divisor, including a short final accumulation window."""

    if accumulation <= 0 or not 1 <= batch_index <= total_batches:
        raise ValueError("Invalid accumulation window arguments")
    window_start = ((batch_index - 1) // accumulation) * accumulation + 1
    return min(accumulation, total_batches - window_start + 1)


def amp_optimizer_step_succeeded(scale_before: float, scale_after: float) -> bool:
    """GradScaler lowers its scale when it skipped an overflowing optimizer step."""

    return float(scale_after) >= float(scale_before)


def phase_gradient_report(model: nn.Module) -> dict[str, Any]:
    rows = []
    for index, parameter in enumerate(model.phase_parameters()):
        gradient = parameter.grad
        rows.append(
            {
                "stage": index + 1,
                "present": gradient is not None,
                "finite": bool(torch.isfinite(gradient).all()) if gradient is not None else False,
                "norm": float(gradient.float().norm().detach().cpu()) if gradient is not None else 0.0,
            }
        )
    return {
        "stages": rows,
        "all_present_finite": all(row["present"] and row["finite"] for row in rows),
        "nonzero_stages": sum(row["norm"] > 0 for row in rows),
    }


def train_epoch(
    wrapped: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    ema: TrainableEMA,
    config: Mapping[str, Any],
    context: Context,
    *,
    epoch: int,
    global_step: int,
):
    wrapped.train()
    core = unwrap(wrapped)
    training = config["training"]
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    maximum = training.get("max_train_batches")
    maximum = min(len(loader), int(maximum)) if maximum is not None else len(loader)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)
    started = time.monotonic()
    last_gradient = None
    skipped_optimizer_steps = 0
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > maximum:
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True).long()
        images, targets = mixed_targets_and_images(
            images,
            labels,
            classes=int(config["model"]["num_classes"]),
            values=config["loss"],
        )
        window_size = accumulation_window_size(batch_index, maximum, accumulation)
        window_start = ((batch_index - 1) // accumulation) * accumulation + 1
        window_end = window_start + window_size - 1
        step_now = batch_index == window_end
        sync_scope = nullcontext()
        if isinstance(wrapped, DistributedDataParallel) and not step_now:
            sync_scope = wrapped.no_sync()
        with sync_scope:
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.float16,
                enabled=bool(training.get("use_amp", True)) and context.device.type == "cuda",
            ):
                logits = wrapped(images)
                loss = soft_target_cross_entropy(logits, targets) / window_size
            scaler.scale(loss).backward()
        raw_loss = float(loss.detach()) * window_size
        top1, top5 = topk_counts(logits.detach(), labels)
        totals += torch.tensor(
            [raw_loss * labels.numel(), top1, top5, labels.numel()],
            dtype=torch.float64,
            device=context.device,
        )
        if step_now:
            scaler.unscale_(optimizer)
            last_gradient = phase_gradient_report(core)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in core.parameters() if parameter.requires_grad],
                float(config["optimizer"].get("gradient_clip_norm", 5.0)),
            )
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            if amp_optimizer_step_succeeded(scale_before, scale_after):
                scheduler.step()
                ema.update(core)
                global_step += 1
            else:
                skipped_optimizer_steps += 1
        interval = int(training.get("log_interval_batches", 100))
        if context.is_main and (batch_index % interval == 0 or batch_index == maximum):
            print(
                f"[train] epoch={epoch} batch={batch_index}/{maximum} "
                f"loss={raw_loss:.5f} updates={global_step}",
                flush=True,
            )
    totals = _reduce(totals, context)
    count = max(float(totals[3]), 1.0)
    metrics = {
        "loss": float(totals[0]) / count,
        "hard_label_top1_accuracy": float(totals[1]) / count,
        "hard_label_top5_accuracy": float(totals[2]) / count,
        "samples": int(totals[3]),
        "batches_per_rank": maximum,
        "elapsed_seconds": time.monotonic() - started,
        "amp_overflow_skipped_optimizer_steps_per_rank": skipped_optimizer_steps,
    }
    return metrics, last_gradient, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, config: Mapping[str, Any], context: Context):
    model.eval()
    maximum = config.get("evaluation", {}).get("max_batches")
    maximum = min(len(loader), int(maximum)) if maximum is not None else len(loader)
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > maximum:
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True).long()
        with torch.autocast(
            device_type=context.device.type,
            dtype=torch.float16,
            enabled=bool(config["training"].get("use_amp", True)) and context.device.type == "cuda",
        ):
            logits = model(images)
        loss = F.cross_entropy(logits.float(), labels)
        top1, top5 = topk_counts(logits, labels)
        totals += torch.tensor(
            [float(loss) * labels.numel(), top1, top5, labels.numel()],
            dtype=torch.float64,
            device=context.device,
        )
    totals = _reduce(totals, context)
    count = max(float(totals[3]), 1.0)
    return {
        "cross_entropy": float(totals[0]) / count,
        "top1_accuracy": float(totals[1]) / count,
        "top5_accuracy": float(totals[2]) / count,
        "samples": int(totals[3]),
    }


def _checkpoint_payload(
    *,
    role: str,
    model: nn.Module,
    ema: TrainableEMA,
    optimizer,
    optimizer_schema,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    history,
    config,
    initialization,
    implementation,
    dataset_identity,
    initial_phase_sha256: str,
    rng_states,
    world_size: int,
    selection: Mapping[str, Any],
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
        "history": list(history),
        "selection": dict(selection),
        "config_digest": config["_config_digest"],
        "model_config": dict(config["model"]),
        "initialization": dict(initialization),
        "implementation_manifest": dict(implementation),
        "dataset_identity": dict(dataset_identity),
        "initial_phase_sha256": initial_phase_sha256,
        "rng_states": list(rng_states),
        "world_size": int(world_size),
        "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
    }


def export_backbone(
    path: Path,
    *,
    model: nn.Module,
    state_variant: str,
    epoch: int,
    config: Mapping[str, Any],
    initialization: Mapping[str, Any],
    implementation: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    selection_semantics: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "format": BACKBONE_FORMAT,
            "backbone": model.backbone_state_dict(),
            "state_variant": state_variant,
            "epoch": int(epoch),
            "selection_semantics": selection_semantics,
            "config_digest": config["_config_digest"],
            "model_config": dict(config["model"]),
            "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
            "initialization": dict(initialization),
            "implementation_manifest": dict(implementation),
            "dataset_identity": dict(dataset_identity),
            "temporary_large_vocabulary_readout_exported": False,
        },
    )


def run(config: dict[str, Any], context: Context, *, resume: bool) -> None:
    check = preflight(config, verify_large_files=False)
    expected_world = config["training"].get("expected_world_size")
    if expected_world is not None and context.world_size != int(expected_world):
        raise RuntimeError(
            f"Recipe expects world_size={int(expected_world)}, got {context.world_size}"
        )
    seed = int(config["training"].get("seed", 2026))
    # Identical seed on all ranks makes the new head identity deterministic
    # before DDP broadcasts it and makes strict resume reconstruction possible.
    seed_all(seed)
    output = resolve_path(config["output_dir"])
    last_path = output / "checkpoints" / "last.pt"
    if resume and not last_path.is_file():
        raise FileNotFoundError("--resume requires checkpoints/last.pt")
    if not resume and any((output / name).exists() for name in ("manifest.json", "result.json", "checkpoints")):
        raise RuntimeError("--fresh refuses an output directory containing run artifacts")

    implementation = implementation_manifest()
    train_dataset, validation_dataset, dataset_identity = build_datasets(config)
    if dataset_identity["publishable_result"] != check["dataset"]["publishable_result"]:
        raise RuntimeError("Preflight/runtime dataset publication identity differs")
    if config["dataset"]["mode"] == "indexed_class_folder" and dataset_identity != check["dataset"]:
        raise RuntimeError("Preflight/runtime indexed dataset identity differs")
    train_sampler = GlobalAffineDistributedSampler(
        train_dataset,
        rank=context.rank,
        world_size=context.world_size,
        seed=seed,
        shuffle=True,
        pad_to_even=True,
    )
    validation_sampler = None
    if validation_dataset is not None:
        validation_sampler = GlobalAffineDistributedSampler(
            validation_dataset,
            rank=context.rank,
            world_size=context.world_size,
            seed=seed + 1,
            shuffle=False,
            pad_to_even=False,
        )
    train_loader = make_loader(train_dataset, train_sampler, config["training"], train=True)
    validation_loader = (
        make_loader(validation_dataset, validation_sampler, config["training"], train=False)
        if validation_dataset is not None
        else None
    )

    model = construct_large_vocabulary_model(
        stem_checkpoint=resolve_path(config["stem_checkpoint"]),
        model_config=config["model"],
    )
    initialization = initialize_from_frozen_p11_backbone(
        model,
        backbone_checkpoint=resolve_path(config["initialization"]["backbone_checkpoint"]),
        expected_backbone_sha256=config["initialization"]["expected_backbone_sha256"],
        expected_stem_sha256=config["initialization"]["expected_stem_sha256"],
    )
    initial_phases = model.phase_snapshot()
    initial_phase_sha256 = state_dict_sha256({"phases": initial_phases})
    resume_payload = None
    if resume:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        locks = {
            "format": CHECKPOINT_FORMAT,
            "checkpoint_role": "last",
            "config_digest": config["_config_digest"],
            "implementation_manifest": implementation,
            "dataset_identity": dataset_identity,
            "initialization": initialization,
            "initial_phase_sha256": initial_phase_sha256,
            "world_size": context.world_size,
        }
        for key, expected in locks.items():
            if resume_payload.get(key) != expected:
                raise RuntimeError(f"Exact resume identity mismatch: {key}")
        model.load_state_dict(resume_payload["model"], strict=True)

    model.to(context.device)
    core = model
    wrapped: nn.Module = core
    if context.distributed:
        wrapped = DistributedDataParallel(
            core,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    # Model/new-head construction used the same seed on every rank and DDP has
    # now broadcast rank 0.  Only now switch fresh data augmentation, Mixup and
    # CutMix to rank-specific RNG streams.  Resume restores the saved per-rank
    # streams below instead of reseeding them.
    if resume_payload is None:
        seed_all(training_rank_seed(seed, context.rank))
    optimizer, optimizer_schema = build_layerwise_optimizer(core, config)
    training = config["training"]
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    batches = len(train_loader)
    if training.get("max_train_batches") is not None:
        batches = min(batches, int(training["max_train_batches"]))
    updates_per_epoch = math.ceil(batches / accumulation)
    scheduler = build_scheduler(optimizer, config, updates_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training.get("use_amp", True)) and context.device.type == "cuda",
        init_scale=float(training.get("amp_initial_scale", 256.0)),
        growth_interval=int(training.get("amp_growth_interval", 100_000)),
    )
    ema = TrainableEMA(
        core,
        decay=float(config.get("ema", {}).get("decay", 0.9999)),
        warmup_updates=int(config.get("ema", {}).get("warmup_updates", 1_000)),
    )
    start_epoch, global_step = 1, 0
    history: list[dict[str, Any]] = []
    selection: dict[str, Any] = {
        "evaluation_enabled": validation_loader is not None,
        "best_raw_top1": None,
        "best_ema_top1": None,
        "best_raw_epoch": None,
        "best_ema_epoch": None,
    }
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        if list(resume_payload["optimizer_schema"]) != optimizer_schema:
            raise RuntimeError("Resume optimizer group schema mismatch")
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload["scaler"])
        ema.load_state_dict(resume_payload["ema"])
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload["global_optimizer_step"])
        history = list(resume_payload["history"])
        selection = dict(resume_payload["selection"])
        states = list(resume_payload.get("rng_states", []))
        if len(states) != context.world_size:
            raise RuntimeError("Resume checkpoint lacks exact per-rank RNG state")
        restore_rng_state(states[context.rank], context)

    effective_batch = int(training["batch_size"]) * accumulation * context.world_size
    expected_batch = training.get("expected_effective_global_batch")
    if expected_batch is not None and effective_batch != int(expected_batch):
        raise RuntimeError(
            f"Effective global batch is {effective_batch}; config locks {int(expected_batch)}"
        )
    backbone_trainable = sum(
        parameter.numel()
        for name, parameter in core.named_parameters()
        if parameter.requires_grad and not name.startswith("readout.")
    )
    optical = sum(parameter.numel() for parameter in core.phase_parameters())
    optical_fraction = optical / backbone_trainable
    required_optical_fraction = float(
        config["model"].get("minimum_optical_parameter_fraction", 0.50)
    )
    if optical_fraction < required_optical_fraction:
        raise RuntimeError(
            f"Trainable-backbone optical fraction {optical_fraction:.6f} is below "
            f"the locked minimum {required_optical_fraction:.6f}"
        )
    initial_gates = core.optical_gates()
    if not initial_gates or min(initial_gates) < 0.50 - 1.0e-6:
        raise RuntimeError(
            f"Optical gate constraint is violated before training: {initial_gates}"
        )
    manifest = {
        "experiment": "P11 eight-stage large-vocabulary supervised pretraining",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config_path": config["_config_path"],
        "config_digest": config["_config_digest"],
        "implementation_manifest": implementation,
        "dataset_identity": dataset_identity,
        "publishable_result": dataset_identity["publishable_result"],
        "publishable_performance_metric": (
            bool(dataset_identity["publishable_result"]) and validation_loader is not None
        ),
        "non_performance_label": (
            None
            if dataset_identity["publishable_result"]
            else "PLUMBING ONLY: ImageNet-1K input with an untrained 21841-way head"
        ),
        "evaluation_enabled": validation_loader is not None,
        "no_validation_selection_semantics": (
            None
            if validation_loader is not None
            else "last raw and last EMA only; no checkpoint may be called best"
        ),
        "initialization": initialization,
        "initial_phase_sha256": initial_phase_sha256,
        "optimizer_groups": optimizer_schema,
        "effective_global_batch": effective_batch,
        "world_size": context.world_size,
        "model_initialization_seed_shared_across_ranks": seed,
        "training_rng_seed_formula": "seed + rank * 1000003 (fresh only)",
        "rank_specific_training_rng": True,
        "persistent_workers": bool(training.get("persistent_workers", False)),
        "epoch_boundary_worker_restart_for_resume": not bool(
            training.get("persistent_workers", False)
        ),
        "updates_per_epoch": updates_per_epoch,
        "temporary_readout_classes": int(config["model"]["num_classes"]),
        "temporary_readout_excluded_from_backbone_optical_fraction": True,
        "optical_fraction_of_trainable_backbone": optical_fraction,
        "minimum_optical_parameter_fraction_required": required_optical_fraction,
        "initial_optical_gates": initial_gates,
    }
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), flush=True)
    context.barrier()

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_sampler.set_epoch(epoch - 1)
        train_metrics, gradients, global_step = train_epoch(
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
        raw_metrics = None
        ema_metrics = None
        improved_raw = improved_ema = False
        if validation_loader is not None:
            assert validation_sampler is not None
            validation_sampler.set_epoch(0)
            raw_metrics = evaluate(wrapped, validation_loader, config, context)
            with ema.apply(core):
                ema_metrics = evaluate(wrapped, validation_loader, config, context)
            raw_top1 = float(raw_metrics["top1_accuracy"])
            ema_top1 = float(ema_metrics["top1_accuracy"])
            improved_raw = selection["best_raw_top1"] is None or raw_top1 > selection["best_raw_top1"]
            improved_ema = selection["best_ema_top1"] is None or ema_top1 > selection["best_ema_top1"]
            if improved_raw:
                selection.update(best_raw_top1=raw_top1, best_raw_epoch=epoch)
            if improved_ema:
                selection.update(best_ema_top1=ema_top1, best_ema_epoch=epoch)
        optical_gates = core.optical_gates()
        if not optical_gates or min(optical_gates) < 0.50 - 1.0e-6:
            raise RuntimeError(
                f"Optical gate constraint was violated at epoch {epoch}: {optical_gates}"
            )
        row = {
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "train": train_metrics,
            "validation_raw": raw_metrics,
            "validation_ema": ema_metrics,
            "phase_gradients": gradients,
            "phase_motion": core.phase_motion(initial_phases),
            "optical_gates": optical_gates,
            "minimum_optical_gate": min(optical_gates),
            "selection": dict(selection),
        }
        history.append(row)
        states = gather_rng_states(context)
        common = dict(
            model=core,
            ema=ema,
            optimizer=optimizer,
            optimizer_schema=optimizer_schema,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            history=history,
            config=config,
            initialization=initialization,
            implementation=implementation,
            dataset_identity=dataset_identity,
            initial_phase_sha256=initial_phase_sha256,
            rng_states=states,
            world_size=context.world_size,
            selection=selection,
        )
        if context.is_main:
            write_json(output / "metrics" / "history.json", history)
            write_json(output / "metrics" / "latest.json", row)
            atomic_torch_save(last_path, _checkpoint_payload(role="last", **common))
            if validation_loader is not None and improved_raw:
                atomic_torch_save(
                    output / "checkpoints" / "best_raw.pt",
                    _checkpoint_payload(role="best_raw", **common),
                )
            if validation_loader is not None and improved_ema:
                atomic_torch_save(
                    output / "checkpoints" / "best_ema.pt",
                    _checkpoint_payload(role="best_ema", **common),
                )
            interval = int(training.get("checkpoint_interval_epochs", 5))
            if interval > 0 and epoch % interval == 0:
                atomic_torch_save(
                    output / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    _checkpoint_payload(role="periodic", **common),
                )
            print(
                f"[epoch] {epoch}/{training['epochs']} train_loss={train_metrics['loss']:.6f} "
                f"validation={'enabled' if raw_metrics else 'disabled'}",
                flush=True,
            )
        context.barrier()

    final_epoch = int(training["epochs"])
    if validation_loader is None:
        raw_path = output / "checkpoints" / "backbone_last_raw.pt"
        ema_path = output / "checkpoints" / "backbone_last_ema.pt"
        if context.is_main:
            export_backbone(
                raw_path,
                model=core,
                state_variant="last_raw",
                epoch=final_epoch,
                config=config,
                initialization=initialization,
                implementation=implementation,
                dataset_identity=dataset_identity,
                selection_semantics="last epoch; evaluation disabled; not best",
            )
        with ema.apply(core):
            if context.is_main:
                export_backbone(
                    ema_path,
                    model=core,
                    state_variant="last_ema",
                    epoch=final_epoch,
                    config=config,
                    initialization=initialization,
                    implementation=implementation,
                    dataset_identity=dataset_identity,
                    selection_semantics="EMA at last epoch; evaluation disabled; not best",
                )
        exports = [str(raw_path), str(ema_path)]
        selected = None
    else:
        candidates = [
            (float(selection["best_raw_top1"]), "best_raw", int(selection["best_raw_epoch"])),
            (float(selection["best_ema_top1"]), "best_ema", int(selection["best_ema_epoch"])),
        ]
        _, selected, selected_epoch = max(candidates)
        selected_path = output / "checkpoints" / f"{selected}.pt"
        payload = torch.load(selected_path, map_location=context.device, weights_only=False)
        core.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"])
        scope = ema.apply(core) if selected == "best_ema" else nullcontext()
        export_path = output / "checkpoints" / "backbone_best_validated.pt"
        with scope:
            if context.is_main:
                export_backbone(
                    export_path,
                    model=core,
                    state_variant=selected,
                    epoch=selected_epoch,
                    config=config,
                    initialization=initialization,
                    implementation=implementation,
                    dataset_identity=dataset_identity,
                    selection_semantics="highest declared validation Top-1 across raw/EMA",
                )
        exports = [str(export_path)]
    context.barrier()

    publishable = bool(dataset_identity["publishable_result"])
    result = {
        "status": (
            "complete" if publishable else "plumbing_smoke_complete_non_result"
        ),
        "publishable_result": publishable,
        "publishable_performance_metric": publishable and validation_loader is not None,
        "non_performance_label": (
            None
            if publishable
            else "PLUMBING ONLY: no ImageNet-21K/22K data were used; metrics are not results"
        ),
        "dataset_identity": dataset_identity,
        "evaluation_enabled": validation_loader is not None,
        "selection": selection if validation_loader is not None else None,
        "selected_state": selected,
        "backbone_exports": exports,
        "final_epoch": final_epoch,
        "history_rows": len(history),
    }
    if context.is_main:
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    context.barrier()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P11 on a manifest-locked large taxonomy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-large-index-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    # Always validate data/assets/disk before touching CUDA or NCCL.  The shell
    # launcher also runs this as a separate CPU process, but keeping the guard
    # here makes a direct ``python -m ...train`` invocation safe as well.
    checked = preflight(config, verify_large_files=args.verify_large_index_files)
    if args.preflight_only:
        # Preflight is intentionally completed before Context initializes NCCL
        # or selects a CUDA device, and before output directories are created.
        print(json.dumps(checked, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    context = Context()
    try:
        run(config, context, resume=args.resume)
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
