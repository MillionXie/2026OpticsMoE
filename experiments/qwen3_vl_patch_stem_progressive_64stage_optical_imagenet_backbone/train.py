from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

from . import migration as migration_api
from .model import QwenStemProgressiveOpticalImageNetBackbone


TRAINING_CHECKPOINT_FORMAT = "p13-progressive-imagenet-training-v1"
BACKBONE_EXPORT_FORMAT = "p13-progressive-imagenet-backbone-v1"
P13_PROGRESSIVE_TRANSITIONS = {(16, 32), (32, 64), (64, 100)}
IMPLEMENTATION_MANIFEST_FORMAT = "p13-training-implementation-v1"
OFFICIAL_P11_BACKBONE_SHA256 = (
    "c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2"
)
OFFICIAL_P11_TRAINING_SHA256 = (
    "a30d5c06b61a635bb3dc379aeaca4c371c1d27e6b862c5ffd49777ce738b33034"
)
OFFICIAL_P11_CONFIG_DIGEST = (
    "c588b9ead9661b5bc513f00349681895979729d8c46b08bba72a109b6d5c74fa"
)
FORMAL_TRAINING_IMPLEMENTATION_FILES = (
    "experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/train.py",
    "experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/model.py",
    "experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/migration.py",
    "experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/p11_matched_continue.py",
    "experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py",
    "experiments/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/model.py",
    "experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/model.py",
    "experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/stem.py",
    "experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/train.py",
    "experiments/d2nn_cifar10_high_performance_optical_backbone/optics.py",
    "experiments/d2nn_cifar10_high_performance_optical_backbone/general_backbone_pretraining.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/datasets.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/settings.py",
)
OFFICIAL_P11_MODEL_CONFIG = {
    "canvas_size": 224,
    "optical_channels": 3,
    "token_dim": 224,
    "num_classes": 1000,
    "head_hidden_dim": 448,
    "wavelength_m": 5.32e-7,
    "pixel_size_m": 1.6e-5,
    "propagation_distance_m": 0.05,
    "token_axis_propagation_distance_m": 0.05,
    "channel_axis_propagation_distance_m": 0.05,
    "phase_init_std": 0.10,
    "layernorm_eps": 1.0e-5,
    "optical_gate_init": 0.60,
    "optical_gate_min": 0.50,
    "mixer_width": 96,
    "mixer_expansion": 2.0,
    "mixer_kernel_size": 3,
    "mixer_dropout": 0.10,
    "mixer_spatial_gate_init": 0.10,
    "mixer_channel_gate_init": 0.10,
    "residual_scale_init": 0.10,
    "residual_scale_max": 0.25,
    "seed": 2026,
}
FreshInitializer = Callable[[nn.Module, Mapping[str, Any]], dict[str, Any]]


def unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    # A deliberately narrow testing hook lets unit tests verify no_sync without
    # constructing a multi-process DDP group. Production wrappers are DDP.
    if bool(getattr(model, "_p13_test_ddp_wrapper", False)):
        return model.module
    return model


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def training_implementation_manifest(
    *,
    repository_root: str | Path | None = None,
    relative_paths: Sequence[str] = FORMAL_TRAINING_IMPLEMENTATION_FILES,
) -> dict[str, Any]:
    """Hash the dirty-worktree implementation and numerical runtime exactly."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_relative in relative_paths:
        relative = Path(raw_relative).as_posix()
        if relative in seen:
            raise RuntimeError(f"Duplicate implementation file: {relative}")
        seen.add(relative)
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing training implementation file: {path}")
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    runtime = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": _installed_distribution_version("torchvision"),
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    identity = {
        "format": IMPLEMENTATION_MANIFEST_FORMAT,
        "files": files,
        "runtime": runtime,
    }
    return {**identity, "aggregate_sha256": canonical_sha256(identity)}


def assert_implementation_manifest_matches(
    saved: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> None:
    if not isinstance(saved, Mapping):
        raise RuntimeError("Checkpoint is missing the implementation manifest")
    if dict(saved) != dict(current):
        raise RuntimeError(
            "Training implementation/runtime differs from the checkpoint; "
            "same-depth resume is not exact"
        )


def _official_p11_config_guard(model_config: Mapping[str, Any]) -> None:
    mismatches = []
    for key, expected in OFFICIAL_P11_MODEL_CONFIG.items():
        actual = model_config.get(key, expected)
        if actual != expected:
            mismatches.append(f"{key}: expected={expected!r}, actual={actual!r}")
    if mismatches:
        raise RuntimeError(
            "Target config cannot reconstruct the official P11 source model: "
            + "; ".join(mismatches)
        )


def _locked_sha256(
    initialization: Mapping[str, Any],
    field: str,
    official: str,
) -> str:
    canonical_hex = set("0123456789abcdef")
    if len(official) != 64 or any(char not in canonical_hex for char in official):
        raise RuntimeError(f"Internal locked identity for {field} is not a SHA-256")
    value = initialization.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in canonical_hex for char in value.lower())
        or value.lower() != official
    ):
        raise RuntimeError(f"{field} must equal the locked official identity")
    return value.lower()


def effective_micro_batches(loader_length: int, configured_limit: Any) -> int:
    limit = int(configured_limit) if configured_limit is not None else int(loader_length)
    if limit <= 0:
        raise ValueError("max_train_batches must be positive when specified")
    return min(int(loader_length), limit)


def optimizer_updates_per_epoch(
    loader_length: int,
    configured_limit: Any,
    accumulation_steps: int,
) -> int:
    accumulation = int(accumulation_steps)
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    micro_batches = effective_micro_batches(loader_length, configured_limit)
    return math.ceil(micro_batches / accumulation)


def dataloader_generator_seed(
    base_seed: int,
    *,
    split: str,
    rank: int,
) -> int:
    if split not in {"train", "validation"}:
        raise ValueError(f"Unknown DataLoader split: {split}")
    payload = f"p13-loader-generator-v1:{int(base_seed)}:{split}:{int(rank)}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "little",
    ) % (2**63 - 1)


def attach_dataloader_generator(loader: Any, seed: int) -> torch.Generator:
    """Keep worker base-seed creation off the main training RNG stream."""

    if getattr(loader, "_iterator", None) is not None:
        raise RuntimeError("DataLoader generator must be attached before iteration")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader.generator = generator
    return generator


def _parameter_sequence(model: nn.Module, accessor: str) -> list[nn.Parameter]:
    method = getattr(model, accessor, None)
    if not callable(method):
        raise RuntimeError(f"Growth model is missing required accessor {accessor}()")
    parameters = list(method())
    if any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
        raise TypeError(f"{accessor}() returned a non-Parameter value")
    return parameters


def build_growth_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    """Build five disjoint old/new groups and prove trainable exhaustiveness."""

    values = config["optimizer"]
    weight_decay = float(values.get("weight_decay", 5.0e-4))
    definitions = (
        ("new_phase", "new_phase_parameters", "new_phase_learning_rate", 7.0e-3, 0.0),
        (
            "carried_phase",
            "carried_phase_parameters",
            "carried_phase_learning_rate",
            3.5e-3,
            0.0,
        ),
        (
            "new_electronic",
            "new_electronic_parameters",
            "new_electronic_learning_rate",
            3.5e-4,
            weight_decay,
        ),
        (
            "carried_electronic",
            "carried_electronic_parameters",
            "carried_electronic_learning_rate",
            2.0e-4,
            weight_decay,
        ),
        ("head", "head_parameters", "head_learning_rate", 5.0e-4, weight_decay),
    )
    groups: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    assigned: dict[int, str] = {}
    for name, accessor, learning_rate_key, default_lr, decay in definitions:
        parameters = _parameter_sequence(model, accessor)
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter in trainable:
            identity = id(parameter)
            if identity in assigned:
                raise RuntimeError(
                    f"Parameter occurs in both {assigned[identity]!r} and {name!r}"
                )
            assigned[identity] = name
        learning_rate = float(values.get(learning_rate_key, default_lr))
        if learning_rate <= 0.0:
            raise ValueError(f"{learning_rate_key} must be positive")
        schema.append(
            {
                "name": name,
                "parameter_tensors": len(trainable),
                "parameter_elements": sum(value.numel() for value in trainable),
                "learning_rate": learning_rate,
                "weight_decay": float(decay),
            }
        )
        if trainable:
            groups.append(
                {
                    "params": trainable,
                    "lr": learning_rate,
                    "weight_decay": float(decay),
                    "name": name,
                }
            )

    all_trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = [all_trainable[identity] for identity in all_trainable.keys() - assigned.keys()]
    extra = [identity for identity in assigned.keys() - all_trainable.keys()]
    if missing or extra:
        raise RuntimeError(
            "Old/new optimizer groups do not exactly partition trainable parameters: "
            f"missing={sorted(missing)}, extra_parameter_ids={extra}"
        )
    if not groups:
        raise RuntimeError("Growth optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in values.get("betas", [0.9, 0.999])),
        eps=float(values.get("eps", 1.0e-8)),
    )
    return optimizer, schema


def build_update_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    updates_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    values = config["optimizer"]
    total_updates = max(int(config["training"]["epochs"]) * updates_per_epoch, 1)
    warmup_updates = int(values.get("warmup_epochs", 1)) * updates_per_epoch
    minimum = float(values.get("minimum_learning_rate_ratio", 0.05))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must lie in [0,1]")

    def scale(update: int) -> float:
        if warmup_updates > 0 and update < warmup_updates:
            return max((update + 1) / warmup_updates, 1.0 / warmup_updates)
        progress = min(
            max((update - warmup_updates) / max(total_updates - warmup_updates, 1), 0.0),
            1.0,
        )
        return minimum + (1.0 - minimum) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def assert_scheduler_update_alignment(
    scheduler: Any,
    global_optimizer_step: int,
) -> None:
    last_epoch = int(getattr(scheduler, "last_epoch", -1))
    if last_epoch != int(global_optimizer_step):
        raise RuntimeError(
            "Scheduler/update mismatch: "
            f"scheduler.last_epoch={last_epoch}, "
            f"global_optimizer_step={int(global_optimizer_step)}"
        )


def capture_rng_state(context: Context) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "rank": int(context.rank),
    }
    if context.device.type == "cuda":
        state["torch_cuda_current_device"] = torch.cuda.get_rng_state(context.device)
    return state


def gather_rng_states(context: Context) -> list[dict[str, Any]]:
    local = capture_rng_state(context)
    if not torch.distributed.is_initialized():
        return [local]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    torch.distributed.all_gather_object(gathered, local)
    states = [dict(value) for value in gathered]
    if [int(value["rank"]) for value in states] != list(range(context.world_size)):
        raise RuntimeError("Distributed RNG states were not gathered in rank order")
    return states


def restore_rng_state(state: Mapping[str, Any], context: Context) -> None:
    if int(state.get("rank", context.rank)) != context.rank:
        raise RuntimeError("Checkpoint RNG state rank does not match the current rank")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if context.device.type == "cuda":
        cuda_state = state.get("torch_cuda_current_device")
        if cuda_state is None:
            raise RuntimeError("CUDA resume checkpoint has no per-rank CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state, context.device)


def _feedback_identity(
    manifest: Mapping[str, Any],
    *,
    include_dynamic_phase: bool,
) -> dict[str, Any]:
    method = str(manifest.get("method"))
    source = manifest.get("source")
    source_hash = (
        source.get("phase_sequence_sha256")
        if isinstance(source, Mapping)
        else manifest.get("source_phase_sequence_sha256")
    )
    connections = []
    for raw in manifest.get("connections", []):
        if not isinstance(raw, Mapping):
            raise RuntimeError("feedback manifest contains a malformed connection")
        item = {
            "index": raw.get("connector_index_zero_based"),
            "role": raw.get("connector_role"),
            "source_node": raw.get("source_node"),
            "target_optical_operator": raw.get("target_optical_operator"),
            "axis": raw.get("axis"),
            "is_p11_mixer_anchor": raw.get("is_p11_mixer_anchor"),
            "is_carried_from_parent": raw.get("is_carried_from_parent"),
            "is_newly_inserted": raw.get("is_newly_inserted"),
            "growth_parent_stage_zero_based": raw.get(
                "growth_parent_stage_zero_based"
            ),
            "frozen": raw.get("frozen"),
            "actual_stage_feedback_mode": raw.get("actual_stage_feedback_mode"),
            "source_phase_sha256": raw.get("source_phase_sha256"),
            "propagation_transfer_sha256": raw.get("propagation_transfer_sha256"),
            "random_substream_seed": raw.get("random_substream_seed"),
            "connector_scale_control": raw.get("connector_scale_control"),
            "real_amplitude_jacobian_spectrum_claimed_equal": raw.get(
                "real_amplitude_jacobian_spectrum_claimed_equal"
            ),
        }
        if include_dynamic_phase or method != "bp_current":
            item["feedback_phase_sha256"] = raw.get("feedback_phase_sha256")
        connections.append(item)
    identity = {
        "format": manifest.get("format"),
        "method": method,
        "depth": manifest.get("depth"),
        "connector_count": manifest.get("connector_count"),
        "random_base_seed": manifest.get("random_base_seed"),
        "source_phase_sequence_sha256": source_hash,
        "connections": connections,
    }
    if include_dynamic_phase or method != "bp_current":
        identity["feedback_phase_sequence_sha256"] = manifest.get(
            "feedback_phase_sequence_sha256"
        )
    return identity


def requested_feedback(config: Mapping[str, Any]) -> tuple[str, int | None]:
    values = config.get("feedback", {})
    method = str(values.get("method", "bp_current"))
    if method not in {"bp_current", "fa_source", "fa_random"}:
        raise ValueError(f"Unsupported feedback method: {method}")
    raw_seed = values.get("random_seed")
    if method == "fa_random":
        if raw_seed is None:
            raise ValueError("fa_random requires feedback.random_seed")
        return method, int(raw_seed)
    if raw_seed is not None:
        raise ValueError(f"feedback.random_seed must be null for {method}")
    return method, None


def assert_feedback_slot_modes(
    manifest: Mapping[str, Any],
    requested_method: str,
) -> None:
    expected = {
        "bp_current": "bp",
        "fa_source": "fa_pretrained",
        "fa_random": "fa_random",
    }.get(requested_method)
    if expected is None:
        raise RuntimeError(f"Unsupported feedback guard method: {requested_method}")
    connections = manifest.get("connections")
    if not isinstance(connections, list):
        raise RuntimeError("Feedback manifest has no per-slot connection list")
    expected_count = manifest.get("connector_count")
    if expected_count != len(connections):
        raise RuntimeError("Feedback manifest connector count is inconsistent")
    failures = []
    for index, connection in enumerate(connections):
        if not isinstance(connection, Mapping):
            raise RuntimeError("Feedback manifest contains a malformed connection")
        actual = connection.get("actual_stage_feedback_mode")
        if actual != expected:
            failures.append(
                {
                    "slot": index,
                    "expected": expected,
                    "actual": actual,
                }
            )
    if failures:
        raise RuntimeError(
            "One or more optical slots silently changed feedback mode: "
            f"{failures}"
        )


def configure_feedback_strict(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    saved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconfigure runtime feedback and reject any silent BP fallback."""

    method, seed = requested_feedback(config)
    configure = getattr(model, "configure_feedback", None)
    manifest_method = getattr(model, "feedback_manifest", None)
    if not callable(configure) or not callable(manifest_method):
        raise RuntimeError("Model lacks the required explicit feedback contract")
    if saved is not None:
        if str(saved.get("method")) != method or saved.get("random_seed") != seed:
            raise RuntimeError(
                "Resume feedback request differs from the checkpoint; refusing a "
                "silent training-method change"
            )
        saved_manifest = saved.get("manifest")
        if not isinstance(saved_manifest, Mapping):
            raise RuntimeError("Resume checkpoint has no complete feedback manifest")
        saved_manifest_sha = canonical_sha256(saved_manifest)
        if saved.get("manifest_sha256") != saved_manifest_sha:
            raise RuntimeError("Saved complete feedback manifest hash is inconsistent")
        # The source phases themselves are persistent tensors.  Their textual
        # capture provenance is deliberately a Python mapping, however, and the
        # model load hook asks the external checkpoint controller to restore it.
        # Do this before rebuilding the runtime connector manifest.
        source = saved_manifest.get("source")
        provenance = source.get("provenance") if isinstance(source, Mapping) else None
        if hasattr(model, "feedback_source_provenance"):
            if not isinstance(provenance, Mapping):
                raise RuntimeError(
                    "P13 saved feedback source has no restorable provenance"
                )
            model.feedback_source_provenance = dict(provenance)
    if method == "fa_random":
        configure(method, random_seed=int(seed))
    else:
        configure(method)
    if str(getattr(model, "feedback_method", "")) != method:
        raise RuntimeError("Model did not activate the requested feedback method")
    manifest = dict(manifest_method())
    if str(manifest.get("method")) != method:
        raise RuntimeError("Feedback manifest disagrees with the active model mode")
    assert_feedback_slot_modes(manifest, method)
    exact_sha = canonical_sha256(
        _feedback_identity(manifest, include_dynamic_phase=True)
    )
    runtime_sha = canonical_sha256(
        _feedback_identity(manifest, include_dynamic_phase=False)
    )
    if saved is not None:
        if canonical_sha256(manifest) != saved["manifest_sha256"]:
            raise RuntimeError(
                "Complete feedback manifest reconstructed after load does not "
                "hash-match the saved manifest"
            )
        if saved.get("exact_resume_sha256") != exact_sha:
            raise RuntimeError(
                "Runtime feedback reconstructed after load does not hash-match "
                "the saved connector state"
            )
        if saved.get("runtime_contract_sha256") != runtime_sha:
            raise RuntimeError("Saved feedback runtime contract hash is inconsistent")
    return {
        "method": method,
        "random_seed": seed,
        "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
        "exact_resume_sha256": exact_sha,
        "runtime_contract_sha256": runtime_sha,
    }


def assert_feedback_runtime(model: nn.Module, expected: Mapping[str, Any]) -> None:
    method = str(expected["method"])
    if str(getattr(model, "feedback_method", "")) != method:
        raise RuntimeError(
            f"Feedback changed silently from {method} to "
            f"{getattr(model, 'feedback_method', None)}"
        )
    manifest = dict(model.feedback_manifest())
    assert_feedback_slot_modes(manifest, method)
    runtime_sha = canonical_sha256(
        _feedback_identity(manifest, include_dynamic_phase=False)
    )
    if runtime_sha != expected["runtime_contract_sha256"]:
        raise RuntimeError("Active feedback connector contract changed during training")


def feedback_checkpoint_state(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    method, seed = requested_feedback(config)
    if str(getattr(model, "feedback_method", "")) != method:
        raise RuntimeError("Refusing to checkpoint a silently changed feedback mode")
    manifest = dict(model.feedback_manifest())
    assert_feedback_slot_modes(manifest, method)
    return {
        "method": method,
        "random_seed": seed,
        "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
        "exact_resume_sha256": canonical_sha256(
            _feedback_identity(manifest, include_dynamic_phase=True)
        ),
        "runtime_contract_sha256": canonical_sha256(
            _feedback_identity(manifest, include_dynamic_phase=False)
        ),
    }


def phase_gradient_report(model: nn.Module) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, accessor in (
        ("new_phase", "new_phase_parameters"),
        ("carried_phase", "carried_phase_parameters"),
    ):
        norms: list[float] = []
        finite: list[bool] = []
        present: list[bool] = []
        for parameter in _parameter_sequence(model, accessor):
            gradient = parameter.grad
            present.append(gradient is not None)
            if gradient is None:
                norms.append(0.0)
                finite.append(False)
            else:
                norms.append(float(gradient.float().norm().detach().cpu()))
                finite.append(bool(torch.isfinite(gradient).all()))
        report[name] = {
            "per_tensor_l2_norm": norms,
            "all_present": all(present),
            "all_finite": all(finite),
            "all_nonzero": all(value > 0.0 for value in norms),
        }
    return report


def _gradient_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    phase = _parameter_sequence(model, "new_phase_parameters") + _parameter_sequence(
        model, "carried_phase_parameters"
    )
    electronic = (
        _parameter_sequence(model, "new_electronic_parameters")
        + _parameter_sequence(model, "carried_electronic_parameters")
        + _parameter_sequence(model, "head_parameters")
    )
    return phase, electronic


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    context: Context,
    *,
    epoch: int,
    global_optimizer_step: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    model.train()
    core = unwrap(model)
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    training = config["training"]
    total_micro = effective_micro_batches(
        len(loader), training.get("max_train_batches")
    )
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    use_amp = bool(training.get("use_amp", True)) and context.device.type == "cuda"
    amp_name = str(training.get("amp_dtype", "float16"))
    if amp_name not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    amp_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16
    label_smoothing = float(config.get("loss", {}).get("label_smoothing", 0.1))
    phase_clip = float(config["optimizer"].get("phase_gradient_clip_norm", 2.0))
    electronic_clip = float(
        config["optimizer"].get("electronic_gradient_clip_norm", 5.0)
    )
    log_interval = int(training.get("log_interval_batches", 100))
    vector = torch.zeros(5, dtype=torch.float64, device=context.device)
    started = time.perf_counter()
    gradient_report: dict[str, Any] | None = None
    phase_parameters, electronic_parameters = _gradient_parameter_groups(core)
    optimizer.zero_grad(set_to_none=True)
    completed_updates = 0
    skipped_updates = 0

    for batch_index, batch in enumerate(loader, 1):
        if batch_index > total_micro:
            break
        window_start = ((batch_index - 1) // accumulation) * accumulation + 1
        window_end = min(window_start + accumulation - 1, total_micro)
        window_size = window_end - window_start + 1
        synchronize = batch_index == window_end
        no_sync = (
            not synchronize
            and context.world_size > 1
            and callable(getattr(model, "no_sync", None))
        )
        sync_context = model.no_sync() if no_sync else nullcontext()
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        images, labels_a, labels_b, lam, _ = mix_batch(images, labels, dict(config))
        with sync_context:
            with torch.autocast(
                device_type=context.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(images)
                loss_a = F.cross_entropy(
                    logits,
                    labels_a,
                    label_smoothing=label_smoothing,
                )
                loss_b = F.cross_entropy(
                    logits,
                    labels_b,
                    label_smoothing=label_smoothing,
                )
                loss = lam * loss_a + (1.0 - lam) * loss_b
            scaler.scale(loss / window_size).backward()

        count = labels.numel()
        correct1, correct5 = topk_counts(logits.detach(), labels)
        vector += torch.tensor(
            [float(loss.detach()) * count, correct1, correct5, count, 1],
            dtype=torch.float64,
            device=context.device,
        )

        if synchronize:
            scaler.unscale_(optimizer)
            if gradient_report is None:
                gradient_report = phase_gradient_report(core)
                if bool(training.get("require_all_phase_gradients", True)):
                    failures = []
                    for group_name, group_report in gradient_report.items():
                        if group_report["per_tensor_l2_norm"] and not (
                            group_report["all_present"]
                            and group_report["all_finite"]
                            and group_report["all_nonzero"]
                        ):
                            failures.append(group_name)
                    if failures:
                        raise RuntimeError(
                            "Phase gradient health check failed for groups: "
                            f"{failures}"
                        )
            torch.nn.utils.clip_grad_norm_(phase_parameters, phase_clip)
            torch.nn.utils.clip_grad_norm_(electronic_parameters, electronic_clip)
            old_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            skipped = bool(scaler.is_enabled()) and float(scaler.get_scale()) < old_scale
            if skipped:
                skipped_updates += 1
            else:
                scheduler.step()
                global_optimizer_step += 1
                completed_updates += 1
            optimizer.zero_grad(set_to_none=True)

        if context.is_main and (
            batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_micro
        ):
            rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            print(
                f"[train] epoch={epoch} micro={batch_index}/{total_micro} "
                f"optimizer_step={global_optimizer_step} loss={float(loss.detach()):.4f} "
                f"lr={rates}",
                flush=True,
            )

    metrics: dict[str, Any] = reduce_metrics(
        vector, time.perf_counter() - started
    )
    metrics.update(
        {
            "samples_per_second": metrics["samples"]
            / max(metrics["seconds"], 1.0e-9),
            "micro_batches": total_micro,
            "gradient_accumulation_steps": accumulation,
            "optimizer_updates": completed_updates,
            "skipped_optimizer_updates": skipped_updates,
            "global_optimizer_step": global_optimizer_step,
        }
    )
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
        metrics["peak_allocated_mib"] = float(peak[0].cpu()) / (1024.0**2)
        metrics["peak_reserved_mib"] = float(peak[1].cpu()) / (1024.0**2)
    return metrics, gradient_report, global_optimizer_step


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
    amp_name = str(config["training"].get("amp_dtype", "float16"))
    amp_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16
    for batch_index, batch in enumerate(loader, 1):
        if limit is not None and batch_index > int(limit):
            break
        images = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        with torch.autocast(
            device_type=context.device.type,
            dtype=amp_dtype,
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


def is_full_depth(model: nn.Module) -> bool:
    report = dict(model.depth_alpha_report())
    full = bool(report.get("all_full_depth", False))
    if full:
        minimum = float(report.get("minimum", 1.0))
        maximum = float(report.get("maximum", 1.0))
        if minimum != 1.0 or maximum != 1.0:
            raise RuntimeError("all_full_depth is true but alpha extrema are not exactly one")
    return full


def checkpoint_roles_for_epoch(
    *,
    improved_any: bool,
    improved_full: bool,
    full_depth: bool,
) -> list[str]:
    if improved_full and not full_depth:
        raise RuntimeError("alpha<1 state cannot be selected as best_full_depth")
    roles = ["last"]
    if improved_any:
        roles.append("best_any")
    if improved_full:
        roles.append("best_full_depth")
    return roles


def _verify_p11_epoch88_sources(
    initialization: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    _official_p11_config_guard(model_config)
    expected_epoch = int(initialization.get("expected_p11_best_epoch", 88))
    backbone = resolve_path(initialization["p11_backbone_checkpoint"])
    training = resolve_path(initialization["p11_training_checkpoint"])
    expected_backbone_sha = _locked_sha256(
        initialization,
        "expected_p11_backbone_sha256",
        OFFICIAL_P11_BACKBONE_SHA256,
    )
    expected_training_sha = _locked_sha256(
        initialization,
        "expected_p11_training_sha256",
        OFFICIAL_P11_TRAINING_SHA256,
    )
    expected_config_digest = _locked_sha256(
        initialization,
        "expected_p11_config_digest",
        OFFICIAL_P11_CONFIG_DIGEST,
    )
    backbone_sha = sha256_file(backbone)
    training_sha = sha256_file(training)
    if backbone_sha != expected_backbone_sha:
        raise RuntimeError("P11 backbone file SHA-256 is not the locked official value")
    if training_sha != expected_training_sha:
        raise RuntimeError("P11 training file SHA-256 is not the locked official value")
    backbone_payload = torch.load(backbone, map_location="cpu", weights_only=False)
    training_payload = torch.load(training, map_location="cpu", weights_only=False)
    if sha256_file(backbone) != expected_backbone_sha:
        raise RuntimeError("P11 backbone changed while its payload was being loaded")
    if sha256_file(training) != expected_training_sha:
        raise RuntimeError("P11 training checkpoint changed while it was being loaded")
    if int(backbone_payload.get("best_epoch", -1)) != expected_epoch:
        raise RuntimeError("P11 backbone source is not the required epoch-88 best export")
    if int(training_payload.get("epoch", -1)) != expected_epoch:
        raise RuntimeError("P11 training source is not the required epoch-88 best checkpoint")
    if backbone_payload.get("config_digest") != training_payload.get("config_digest"):
        raise RuntimeError("P11 backbone and training checkpoints have different config digests")
    if backbone_payload.get("config_digest") != expected_config_digest:
        raise RuntimeError("P11 checkpoint config digest is not the locked official value")
    return {
        "backbone_checkpoint": str(backbone),
        "backbone_sha256": backbone_sha,
        "training_checkpoint": str(training),
        "training_sha256": training_sha,
        "config_digest": expected_config_digest,
        "best_epoch": expected_epoch,
        "_backbone_payload": backbone_payload,
        "_training_payload": training_payload,
    }


def _validated_parent_feedback_provenance(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = payload.get("feedback")
    if not isinstance(feedback, Mapping):
        raise RuntimeError("Progressive parent has no formal feedback checkpoint state")
    method = feedback.get("method")
    if method not in {"bp_current", "fa_source", "fa_random"}:
        raise RuntimeError("Progressive parent feedback method is invalid")
    manifest = feedback.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Progressive parent has no complete feedback manifest")
    manifest_sha = canonical_sha256(manifest)
    if feedback.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Progressive parent feedback manifest SHA-256 is invalid")
    if manifest.get("method") != method:
        raise RuntimeError("Progressive parent feedback method disagrees with its manifest")
    assert_feedback_slot_modes(manifest, str(method))
    expected_exact = canonical_sha256(
        _feedback_identity(manifest, include_dynamic_phase=True)
    )
    expected_runtime = canonical_sha256(
        _feedback_identity(manifest, include_dynamic_phase=False)
    )
    if feedback.get("exact_resume_sha256") != expected_exact:
        raise RuntimeError("Progressive parent exact feedback hash is invalid")
    if feedback.get("runtime_contract_sha256") != expected_runtime:
        raise RuntimeError("Progressive parent runtime feedback hash is invalid")
    return {
        "method": method,
        "random_seed": feedback.get("random_seed"),
        "manifest_sha256": manifest_sha,
        "exact_resume_sha256": feedback.get("exact_resume_sha256"),
        "runtime_contract_sha256": feedback.get("runtime_contract_sha256"),
    }


def initialize_p13_fresh(
    model: nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    initialization = config["initialization"]
    mode = str(initialization.get("mode"))
    target_depth = int(config["model"]["num_stages"])
    if mode == "p11_to_16":
        if target_depth != 16:
            raise RuntimeError("p11_to_16 initialization requires model.num_stages=16")
        source_identity = _verify_p11_epoch88_sources(
            initialization,
            getattr(model, "config", config["model"]),
        )
        backbone = resolve_path(initialization["p11_backbone_checkpoint"])
        training = resolve_path(initialization["p11_training_checkpoint"])
        migration = migration_api.migrate_strict_p11_training_checkpoint(
            model,
            backbone,
            training,
        )
        if migration.get("source_checkpoint_sha256") != source_identity[
            "backbone_sha256"
        ] or migration.get("source_training_checkpoint_sha256") != source_identity[
            "training_sha256"
        ]:
            raise RuntimeError("P11 migration used a source outside the locked identity")
        return {
            "mode": mode,
            "source_depth": 8,
            "target_depth": 16,
            "p11_backbone_checkpoint": str(backbone),
            "p11_backbone_sha256": source_identity["backbone_sha256"],
            "p11_training_checkpoint": str(training),
            "p11_training_sha256": source_identity["training_sha256"],
            "p11_config_digest": source_identity["config_digest"],
            "expected_p11_best_epoch": int(
                initialization.get("expected_p11_best_epoch", 88)
            ),
            "migration": migration,
        }
    if mode == "progressive_growth":
        checkpoint = resolve_path(initialization["source_training_checkpoint"])
        expected_source_sha = initialization.get("expected_source_training_sha256")
        if not isinstance(expected_source_sha, str) or len(expected_source_sha) != 64:
            raise RuntimeError(
                "progressive_growth requires expected_source_training_sha256"
            )
        source_sha = sha256_file(checkpoint)
        if source_sha != expected_source_sha.lower():
            raise RuntimeError("Progressive source training checkpoint SHA-256 changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != TRAINING_CHECKPOINT_FORMAT:
            raise RuntimeError("Progressive source is not a formal P13 training checkpoint")
        if payload.get("checkpoint_role") != "best_full_depth":
            raise RuntimeError("Progressive growth must start from best_full_depth.pt")
        source_depth = int(payload.get("model_config", {}).get("num_stages", -1))
        if initialization.get("expected_source_depth") != source_depth:
            raise RuntimeError("Progressive source depth differs from the locked value")
        if (source_depth, target_depth) not in P13_PROGRESSIVE_TRANSITIONS:
            raise RuntimeError(
                f"Unsupported progressive growth transition {source_depth}->{target_depth}"
            )
        if not bool(payload.get("depth_alpha", {}).get("all_full_depth", False)):
            raise RuntimeError("Progressive source checkpoint is not at alpha=1")
        if initialization.get("expected_source_epoch") != payload.get("epoch"):
            raise RuntimeError("Progressive source epoch differs from the locked value")
        if initialization.get("expected_source_config_digest") != payload.get(
            "config_digest"
        ):
            raise RuntimeError(
                "Progressive source config digest differs from the locked value"
            )
        parent_feedback = _validated_parent_feedback_provenance(payload)
        if initialization.get("expected_source_feedback_method") != parent_feedback[
            "method"
        ]:
            raise RuntimeError(
                "Progressive source feedback method differs from the locked value"
            )
        if initialization.get(
            "expected_source_feedback_manifest_sha256"
        ) != parent_feedback["manifest_sha256"]:
            raise RuntimeError(
                "Progressive source feedback manifest differs from the locked value"
            )
        migration = migration_api.migrate_strict_progressive_checkpoint(
            model,
            checkpoint,
        )
        if sha256_file(checkpoint) != expected_source_sha.lower():
            raise RuntimeError("Progressive source changed during strict migration")
        migration = dict(migration)
        migration["parent_training_feedback"] = parent_feedback
        if not hasattr(model, "migration_manifest"):
            raise RuntimeError("P13 model has no migration_manifest provenance field")
        model.migration_manifest = migration
        return {
            "mode": mode,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "source_training_checkpoint": str(checkpoint),
            "source_training_sha256": source_sha,
            "parent_training_feedback": parent_feedback,
            "migration": migration,
        }
    raise ValueError(f"Unsupported initialization.mode: {mode}")


def validate_training_config(config: Mapping[str, Any]) -> None:
    required = {
        "output_dir",
        "imagenet_config",
        "stem_checkpoint",
        "model",
        "initialization",
        "feedback",
        "optimizer",
        "training",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Training config is missing fields: {missing}")
    training = config["training"]
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(training.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    requested_feedback(config)


def _dataset_loaders(
    config: Mapping[str, Any],
    context: Context,
) -> tuple[Any, Any, Any, Any, Any, list[int], list[int]]:
    training = config["training"]
    imagenet_settings = load_imagenet_settings(resolve_path(config["imagenet_config"]))
    bundle = load_imagenet(imagenet_settings)
    train_per_class = training.get("train_samples_per_class")
    train_indices = (
        list(range(bundle.train.base_sample_count))
        if train_per_class is None
        else stratified_base_indices(
            bundle.train.targets,
            int(train_per_class),
            int(training.get("seed", 2026)),
        )
    )
    validation_per_class = training.get("validation_samples_per_class")
    validation_indices = (
        list(range(bundle.validation.base_sample_count))
        if validation_per_class is None
        else stratified_base_indices(
            bundle.validation.targets,
            int(validation_per_class),
            int(training.get("seed", 2026)) + 1,
        )
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
    train_loader = make_loader(bundle.train, train_sampler, dict(config), train=True)
    validation_loader = make_loader(
        bundle.validation, validation_sampler, dict(config), train=False
    )
    base_seed = int(training.get("seed", 2026))
    attach_dataloader_generator(
        train_loader,
        dataloader_generator_seed(
            base_seed,
            split="train",
            rank=context.rank,
        ),
    )
    attach_dataloader_generator(
        validation_loader,
        dataloader_generator_seed(
            base_seed,
            split="validation",
            rank=context.rank,
        ),
    )
    return (
        bundle,
        train_loader,
        validation_loader,
        train_sampler,
        validation_sampler,
        train_indices,
        validation_indices,
    )


def fresh_run_artifacts(output: str | Path) -> list[Path]:
    root = Path(output)
    artifacts: list[Path] = []
    for name in (
        "manifest.json",
        "manifest.json.tmp",
        "initial_phases.pt",
        "initial_phases.pt.tmp",
        "result.json",
        "result.json.tmp",
    ):
        path = root / name
        if path.exists():
            artifacts.append(path)
    for directory_name in ("checkpoints", "metrics"):
        directory = root / directory_name
        if directory.is_dir():
            artifacts.extend(path for path in directory.iterdir())
    return sorted(artifacts, key=lambda value: str(value))


def _validate_resume_payload(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    checkpoint_format: str,
    world_size: int,
    implementation_manifest: Mapping[str, Any],
) -> None:
    if payload.get("format") != checkpoint_format:
        raise RuntimeError("Resume checkpoint format mismatch")
    if payload.get("checkpoint_role") != "last":
        raise RuntimeError("Same-depth resume must read last.pt")
    if payload.get("config_digest") != config["_config_digest"]:
        raise RuntimeError("Resume config digest mismatch")
    if dict(payload.get("model_config", {})) != dict(model_config):
        raise RuntimeError("Resume model config/depth mismatch")
    if int(payload.get("world_size", -1)) != int(world_size):
        raise RuntimeError("Exact resume requires the same DDP world size")
    if len(payload.get("rng_states", [])) != int(world_size):
        raise RuntimeError("Resume checkpoint has incomplete per-rank RNG state")
    assert_implementation_manifest_matches(
        payload.get("implementation_manifest"),
        implementation_manifest,
    )


def restore_migration_provenance(
    model: nn.Module,
    initialization_manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the non-state_dict migration audit trail after construction/load.

    ``migration_manifest`` is intentionally Python provenance rather than a tensor
    buffer.  A same-depth resume therefore has to restore it explicitly after the
    strict model state load.  The target identity is checked before assignment so
    that provenance from another depth or architecture cannot be attached to the
    live model.
    """

    core = unwrap(model)
    raw_manifest = initialization_manifest.get("migration")
    if not isinstance(raw_manifest, Mapping):
        raise RuntimeError(
            "Formal P13 initialization is missing its migration manifest"
        )
    manifest = dict(raw_manifest)
    configured_depth = model_config.get("num_stages")
    if not isinstance(configured_depth, int) or isinstance(configured_depth, bool):
        raise RuntimeError("P13 model_config num_stages must be an integer")
    live_depth = int(getattr(core, "num_stages", configured_depth))
    if live_depth != configured_depth:
        raise RuntimeError(
            "Live model depth differs from the configured P13 target depth"
        )
    wrapper_target_depth = initialization_manifest.get("target_depth")
    if wrapper_target_depth != configured_depth:
        raise RuntimeError(
            "Initialization wrapper target depth differs from the live P13 model"
        )
    manifest_target_depth = manifest.get("target_num_stages")
    if (
        not isinstance(manifest_target_depth, int)
        or isinstance(manifest_target_depth, bool)
        or manifest_target_depth != configured_depth
    ):
        raise RuntimeError(
            "Migration manifest target depth differs from the live P13 model"
        )

    expected_signature = (13, 1, 2, configured_depth)
    signature_buffer = getattr(core, "p13_progressive_architecture_signature", None)
    if not isinstance(signature_buffer, torch.Tensor):
        raise RuntimeError("P13 model is missing its architecture signature buffer")
    live_signature = tuple(
        int(value) for value in signature_buffer.detach().cpu().tolist()
    )
    if live_signature != expected_signature:
        raise RuntimeError(
            "Live P13 architecture signature differs from the configured depth"
        )
    raw_signature = manifest.get("target_architecture_signature")
    if (
        not isinstance(raw_signature, (list, tuple))
        or len(raw_signature) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_signature
        )
    ):
        raise RuntimeError("Migration manifest is missing its target signature")
    manifest_signature = tuple(raw_signature)
    if manifest_signature != live_signature:
        raise RuntimeError(
            "Migration manifest target signature differs from the live P13 model"
        )

    wrapper_source_depth = initialization_manifest.get("source_depth")
    if manifest.get("source_depth") != wrapper_source_depth:
        raise RuntimeError(
            "Initialization wrapper and migration manifest source depths differ"
        )
    if not hasattr(core, "migration_manifest"):
        raise RuntimeError("P13 model has no migration_manifest provenance field")
    core.migration_manifest = manifest
    return manifest


def _checkpoint_payload(
    *,
    role: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_schema: Sequence[Mapping[str, Any]],
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_optimizer_step: int,
    best_any_top1: float,
    best_full_depth_top1: float,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    initialization_manifest: Mapping[str, Any],
    rng_states: Sequence[Mapping[str, Any]],
    initial_phases: torch.Tensor,
    checkpoint_format: str,
    world_size: int,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    core = unwrap(model)
    assert_implementation_manifest_matches(
        implementation_manifest,
        training_implementation_manifest(),
    )
    feedback = feedback_checkpoint_state(core, config)
    migration_manifest = dict(initialization_manifest).get("migration")
    runtime_migration_manifest = getattr(core, "migration_manifest", None)
    if migration_manifest is not None and runtime_migration_manifest != migration_manifest:
        raise RuntimeError(
            "Live migration provenance differs from initialization_manifest"
        )
    return {
        "format": checkpoint_format,
        "checkpoint_role": role,
        "model": core.state_dict(),
        "model_config": dict(model_config),
        "model_report": core.parameter_report(),
        "stem_checkpoint_sha256": core.stem.checkpoint_sha256,
        "optimizer": optimizer.state_dict(),
        "optimizer_schema": list(optimizer_schema),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "best_any_top1": float(best_any_top1),
        "best_full_depth_top1": float(best_full_depth_top1),
        "history": list(history),
        "config_digest": config["_config_digest"],
        "initialization_manifest": dict(initialization_manifest),
        "migration_manifest": migration_manifest,
        "feedback": feedback,
        "depth_alpha": dict(core.depth_alpha_report()),
        "phase_motion": core.phase_motion(initial_phases),
        "initial_phase_sha256": sha256_tensor(initial_phases),
        "rng_states": list(rng_states),
        "world_size": int(world_size),
        "implementation_manifest": dict(implementation_manifest),
    }


def _save_roles(
    output: Path,
    roles: Sequence[str],
    **payload_arguments: Any,
) -> None:
    filenames = {
        "last": "last.pt",
        "best_any": "best_any.pt",
        "best_full_depth": "best_full_depth.pt",
    }
    for role in roles:
        payload = _checkpoint_payload(role=role, **payload_arguments)
        atomic_save(output / "checkpoints" / filenames[role], payload)


def run_training(
    config: dict[str, Any],
    context: Context,
    *,
    resume: bool,
    model_class: type[nn.Module] = QwenStemProgressiveOpticalImageNetBackbone,
    fresh_initializer: FreshInitializer = initialize_p13_fresh,
    experiment_name: str = "P13 progressive optical ImageNet growth",
    checkpoint_format: str = TRAINING_CHECKPOINT_FORMAT,
    export_format: str = BACKBONE_EXPORT_FORMAT,
) -> None:
    validate_training_config(config)
    training = config["training"]
    seed_all(int(training.get("seed", 2026)), context.rank)
    expected_world_size = training.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != context.world_size:
        raise RuntimeError(
            f"This controlled run requires world_size={int(expected_world_size)}, "
            f"got {context.world_size}"
        )
    output = resolve_path(config["output_dir"])
    implementation_manifest = training_implementation_manifest()
    last_path = output / "checkpoints" / "last.pt"
    if resume and not last_path.is_file():
        raise FileNotFoundError("--resume requires an existing checkpoints/last.pt")
    existing_artifacts = fresh_run_artifacts(output)
    if not resume and existing_artifacts:
        raise RuntimeError(
            "--fresh refuses an output directory containing prior run artifacts; "
            f"use --resume or choose a new output_dir: {existing_artifacts}"
        )

    (
        bundle,
        train_loader,
        validation_loader,
        train_sampler,
        validation_sampler,
        train_indices,
        validation_indices,
    ) = _dataset_loaders(config, context)
    model_config = dict(config["model"])
    model_config.setdefault("seed", int(training.get("seed", 2026)))
    model = model_class(resolve_path(config["stem_checkpoint"]), model_config)

    resume_payload: Mapping[str, Any] | None = None
    if resume:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_resume_payload(
            resume_payload,
            config,
            model_config,
            checkpoint_format=checkpoint_format,
            world_size=context.world_size,
            implementation_manifest=implementation_manifest,
        )
        if (
            resume_payload.get("stem_checkpoint_sha256")
            != model.stem.checkpoint_sha256
        ):
            raise RuntimeError(
                "Resume stem checkpoint identity differs from the live model"
            )
        model.load_state_dict(resume_payload["model"], strict=True)
        initialization_manifest = dict(resume_payload["initialization_manifest"])
        if checkpoint_format == TRAINING_CHECKPOINT_FORMAT:
            restored_migration = restore_migration_provenance(
                model,
                initialization_manifest,
                model_config,
            )
            if resume_payload.get("migration_manifest") != restored_migration:
                raise RuntimeError(
                    "Resume top-level migration manifest differs from initialization"
                )
            saved_report = resume_payload.get("model_report")
            if not isinstance(saved_report, Mapping) or saved_report.get(
                "migration_manifest"
            ) != restored_migration:
                raise RuntimeError(
                    "Resume model_report migration provenance is inconsistent"
                )
        feedback_runtime = configure_feedback_strict(
            model,
            config,
            saved=resume_payload["feedback"],
        )
        saved_depth_alpha = resume_payload.get("depth_alpha")
        current_depth_alpha = dict(model.depth_alpha_report())
        if not isinstance(saved_depth_alpha, Mapping) or dict(
            saved_depth_alpha
        ) != current_depth_alpha:
            raise RuntimeError(
                "Resume depth-alpha metadata differs from the loaded model state"
            )
        saved_model_report = resume_payload.get("model_report")
        if not isinstance(saved_model_report, Mapping) or dict(
            saved_model_report
        ) != dict(model.parameter_report()):
            raise RuntimeError(
                "Resume model_report differs from the reconstructed live model"
            )
        initial_phases = torch.load(
            output / "initial_phases.pt", map_location="cpu", weights_only=True
        )
        if sha256_tensor(initial_phases) != resume_payload["initial_phase_sha256"]:
            raise RuntimeError("Resume initial_phases.pt hash mismatch")
    else:
        initialization_manifest = fresh_initializer(model, config)
        if checkpoint_format == TRAINING_CHECKPOINT_FORMAT:
            restored_migration = restore_migration_provenance(
                model,
                initialization_manifest,
                model_config,
            )
            if getattr(model, "migration_manifest", None) != restored_migration:
                raise RuntimeError("Fresh migration provenance was not installed")
        model.apply_depth_ramp(0)
        feedback_runtime = configure_feedback_strict(model, config)
        initial_phases = model.phase_snapshot()

    report = model.parameter_report()
    minimum_fraction = float(
        model_config.get("minimum_optical_parameter_fraction", 0.50)
    )
    measured_fraction = float(report["optical_fraction_of_backbone_trainable"])
    if measured_fraction < minimum_fraction:
        raise RuntimeError(
            f"Optical backbone fraction {measured_fraction:.4f} is below "
            f"the required {minimum_fraction:.4f}"
        )
    if float(report["minimum_optical_gate"]) < 0.50:
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
    core = unwrap(model)
    optimizer, optimizer_schema = build_growth_optimizer(core, config)
    updates_per_epoch = optimizer_updates_per_epoch(
        len(train_loader),
        training.get("max_train_batches"),
        int(training.get("gradient_accumulation_steps", 1)),
    )
    scheduler = build_update_scheduler(optimizer, config, updates_per_epoch)
    amp_dtype = str(training.get("amp_dtype", "float16"))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            bool(training.get("use_amp", True))
            and context.device.type == "cuda"
            and amp_dtype == "float16"
        ),
        init_scale=float(training.get("amp_initial_scale", 256.0)),
        growth_interval=int(training.get("amp_growth_interval", 100000)),
    )

    start_epoch = 1
    global_optimizer_step = 0
    best_any_top1 = -math.inf
    best_full_depth_top1 = -math.inf
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        if list(resume_payload.get("optimizer_schema", [])) != optimizer_schema:
            raise RuntimeError("Resume old/new optimizer group schema mismatch")
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = int(resume_payload["epoch"]) + 1
        global_optimizer_step = int(resume_payload["global_optimizer_step"])
        best_any_top1 = float(resume_payload["best_any_top1"])
        best_full_depth_top1 = float(resume_payload["best_full_depth_top1"])
        history = list(resume_payload.get("history", []))
        restore_rng_state(resume_payload["rng_states"][context.rank], context)
    assert_scheduler_update_alignment(scheduler, global_optimizer_step)

    effective_global_batch = (
        int(training["batch_size"])
        * context.world_size
        * int(training.get("gradient_accumulation_steps", 1))
    )
    expected_global_batch = training.get("expected_effective_global_batch")
    if (
        expected_global_batch is not None
        and int(expected_global_batch) != effective_global_batch
    ):
        raise RuntimeError(
            f"Expected effective global batch {int(expected_global_batch)}, "
            f"got {effective_global_batch}"
        )
    manifest = {
        "experiment": experiment_name,
        "checkpoint_format": checkpoint_format,
        "config": config["_config_path"],
        "config_digest": config["_config_digest"],
        "world_size": context.world_size,
        "dataset_digest": bundle.digest,
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
        "full_resolution_imagenet_images": True,
        "online_frozen_qwen_stem": True,
        "hidden_state_cache_used": False,
        "initialization": initialization_manifest,
        "feedback": feedback_runtime,
        "optimizer_groups": optimizer_schema,
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 1)
        ),
        "optimizer_updates_per_epoch": updates_per_epoch,
        "effective_global_batch": effective_global_batch,
        "model": report,
        "implementation_manifest": implementation_manifest,
        "dataloader_generator_seeds": {
            "train_by_rank": [
                dataloader_generator_seed(
                    int(training.get("seed", 2026)),
                    split="train",
                    rank=rank,
                )
                for rank in range(context.world_size)
            ],
            "validation_by_rank": [
                dataloader_generator_seed(
                    int(training.get("seed", 2026)),
                    split="validation",
                    rank=rank,
                )
                for rank in range(context.world_size)
            ],
            "rank_specific": True,
            "isolated_from_main_training_rng": True,
        },
    }
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        if not resume:
            atomic_save(output / "initial_phases.pt", initial_phases)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)
    context.barrier()

    if not resume:
        validation_sampler.set_epoch(0)
        baseline = evaluate(model, validation_loader, config, context)
        best_any_top1 = float(baseline["top1_accuracy"])
        baseline_full = is_full_depth(core)
        if baseline_full:
            best_full_depth_top1 = best_any_top1
        rng_states = gather_rng_states(context)
        if context.is_main:
            write_json(output / "metrics" / "initial_baseline.json", baseline)
            roles = ["last", "best_any"]
            if baseline_full:
                roles.append("best_full_depth")
            _save_roles(
                output,
                roles,
                model=model,
                optimizer=optimizer,
                optimizer_schema=optimizer_schema,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                global_optimizer_step=global_optimizer_step,
                best_any_top1=best_any_top1,
                best_full_depth_top1=best_full_depth_top1,
                history=history,
                config=config,
                model_config=model_config,
                initialization_manifest=initialization_manifest,
                rng_states=rng_states,
                initial_phases=initial_phases,
                checkpoint_format=checkpoint_format,
                world_size=context.world_size,
                implementation_manifest=implementation_manifest,
            )
        context.barrier()

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        alpha = float(core.apply_depth_ramp(epoch))
        assert_feedback_runtime(core, feedback_runtime)
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        train_metrics, gradient_report, global_optimizer_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            config,
            context,
            epoch=epoch,
            global_optimizer_step=global_optimizer_step,
        )
        assert_scheduler_update_alignment(scheduler, global_optimizer_step)
        validation_metrics = evaluate(model, validation_loader, config, context)
        assert_feedback_runtime(core, feedback_runtime)
        depth_alpha = dict(core.depth_alpha_report())
        full_depth = is_full_depth(core)
        motion = core.phase_motion(initial_phases)
        top1 = float(validation_metrics["top1_accuracy"])
        improved_any = top1 > best_any_top1
        improved_full = full_depth and top1 > best_full_depth_top1
        if improved_any:
            best_any_top1 = top1
        if improved_full:
            best_full_depth_top1 = top1
        row = {
            "epoch": epoch,
            "depth_alpha": depth_alpha,
            "scheduled_new_stage_alpha": alpha,
            "full_depth_export_eligible": full_depth,
            "learning_rates": {
                group["name"]: group["lr"] for group in optimizer.param_groups
            },
            "global_optimizer_step": global_optimizer_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "phase_gradients": gradient_report,
            "phase_motion": motion,
            "optical_gates": core.optical_gates(),
        }
        rng_states = gather_rng_states(context)
        if context.is_main:
            history.append(row)
            write_json(output / "metrics" / "history.json", history)
            write_json(output / "metrics" / "latest.json", row)
            roles = checkpoint_roles_for_epoch(
                improved_any=improved_any,
                improved_full=improved_full,
                full_depth=full_depth,
            )
            _save_roles(
                output,
                roles,
                model=model,
                optimizer=optimizer,
                optimizer_schema=optimizer_schema,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_optimizer_step=global_optimizer_step,
                best_any_top1=best_any_top1,
                best_full_depth_top1=best_full_depth_top1,
                history=history,
                config=config,
                model_config=model_config,
                initialization_manifest=initialization_manifest,
                rng_states=rng_states,
                initial_phases=initial_phases,
                checkpoint_format=checkpoint_format,
                world_size=context.world_size,
                implementation_manifest=implementation_manifest,
            )
            interval = int(training.get("checkpoint_interval_epochs", 5))
            if epoch % interval == 0:
                interval_payload = _checkpoint_payload(
                    role="last",
                    model=model,
                    optimizer=optimizer,
                    optimizer_schema=optimizer_schema,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_optimizer_step=global_optimizer_step,
                    best_any_top1=best_any_top1,
                    best_full_depth_top1=best_full_depth_top1,
                    history=history,
                    config=config,
                    model_config=model_config,
                    initialization_manifest=initialization_manifest,
                    rng_states=rng_states,
                    initial_phases=initial_phases,
                    checkpoint_format=checkpoint_format,
                    world_size=context.world_size,
                    implementation_manifest=implementation_manifest,
                )
                atomic_save(
                    output / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    interval_payload,
                )
            print(
                f"[epoch] {epoch}/{training['epochs']} alpha={alpha:.6f} "
                f"full={full_depth} train_top1={train_metrics['top1_accuracy']:.4f} "
                f"val_top1={top1:.4f} best_any={best_any_top1:.4f} "
                f"best_full={best_full_depth_top1:.4f}",
                flush=True,
            )
        context.barrier()

    best_full_path = output / "checkpoints" / "best_full_depth.pt"
    if not best_full_path.is_file():
        raise RuntimeError(
            "No alpha=1 checkpoint exists; refusing backbone export. Extend the "
            "run or shorten the configured depth ramp."
        )
    assert_implementation_manifest_matches(
        implementation_manifest,
        training_implementation_manifest(),
    )
    best_payload = torch.load(best_full_path, map_location="cpu", weights_only=False)
    assert_implementation_manifest_matches(
        best_payload.get("implementation_manifest"),
        implementation_manifest,
    )
    if best_payload.get("checkpoint_role") != "best_full_depth":
        raise RuntimeError("best_full_depth.pt has the wrong checkpoint role")
    core.load_state_dict(best_payload["model"], strict=True)
    feedback_runtime = configure_feedback_strict(
        core,
        config,
        saved=best_payload["feedback"],
    )
    if not is_full_depth(core):
        raise RuntimeError("Refusing export from an alpha<1 checkpoint")
    validation_sampler.set_epoch(0)
    normal = evaluate(model, validation_loader, config, context)
    ablations: dict[str, Any] = {}
    if bool(training.get("run_final_ablations", True)):
        for name in ("optical_off", "phase_random", "electronic_skip_off"):
            ablations[name] = evaluate(
                model, validation_loader, config, context, ablation=name
            )
    if context.is_main:
        backbone_path = output / "checkpoints" / "backbone_full_depth.pt"
        atomic_save(
            backbone_path,
            {
                "format": export_format,
                "backbone": core.backbone_state_dict(),
                "best_epoch": int(best_payload["epoch"]),
                "source_training_checkpoint": str(best_full_path),
                "source_training_checkpoint_sha256": sha256_file(best_full_path),
                "config_digest": config["_config_digest"],
                "stem_checkpoint_sha256": core.stem.checkpoint_sha256,
                "model_config": model_config,
                "model_report": core.parameter_report(),
                "initialization_manifest": initialization_manifest,
                "feedback": feedback_checkpoint_state(core, config),
                "depth_alpha": dict(core.depth_alpha_report()),
                "feature_contract": {
                    "input": "CLIP-normalized RGB [B,3,224,224]",
                    "final": "three latent optical banks [B,3,224,224]",
                    "stages": f"{model_config['num_stages']} OEO feature maps",
                    "qwen_transformer_required": False,
                    "temporary_imagenet_readout_exported": False,
                },
                "implementation_manifest": implementation_manifest,
            },
        )
        result = {
            "status": "complete",
            "best_any_top1": best_any_top1,
            "best_full_depth_top1": best_full_depth_top1,
            "best_full_depth_epoch": int(best_payload["epoch"]),
            "backbone_checkpoint": str(backbone_path),
            "best_full_depth_validation": normal,
            "ablations": ablations,
            "depth_alpha": dict(core.depth_alpha_report()),
            "feedback": feedback_runtime,
            "model": core.parameter_report(),
            "phase_motion": core.phase_motion(initial_phases),
            "global_optimizer_step": int(best_payload["global_optimizer_step"]),
            "implementation_manifest": implementation_manifest,
        }
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    context.barrier()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a progressively grown P13 optical ImageNet backbone"
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
        run_training(
            load_config(args.config),
            context,
            resume=bool(args.resume),
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
