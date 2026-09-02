from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.settings import (
    load_settings as load_imagenet_settings,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    Context,
    atomic_save,
    load_config,
    resolve_path,
    seed_all,
    write_json,
)
from experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.train import (
    canonical_sha256,
    gather_rng_states,
    restore_rng_state,
    sha256_file,
    sha256_tensor,
    training_implementation_manifest,
)

from .large_scale_continue import (
    LargeScaleP11Model,
    TrainableEMA,
    _dataset_loaders,
    _null_scope,
    _run_has_artifacts,
    build_layerwise_optimizer,
    build_scheduler,
    evaluate,
    train_epoch,
)


CHECKPOINT_FORMAT = "p11-post-strong-pretrain-clean-recovery-v1"
BACKBONE_FORMAT = "p11-post-strong-pretrain-clean-recovery-backbone-v1"
EXPECTED_SOURCE_FORMAT = "p11-large-scale-supervised-continuation-v1"
EXPECTED_SOURCE_ROLE = "last"
EXPECTED_SOURCE_EPOCH = 5
EXPECTED_SOURCE_SHA256 = (
    "34175ba9e764b7eef5bd59b1e1d1dd7f602281d02bd709ebf12ec55c0338f681"
)
EXPECTED_SOURCE_CONFIG_DIGEST = (
    "8bea33ea2f6cccf25b499ff5949eb5462d232b11694e29bba9c1b8dccb8ba202"
)

# This manifest is deliberately independent of large_scale_continue.py's
# IMPLEMENTATION_FILES. Adding this entry point cannot change the identity of
# a large-scale continuation run that is already in progress.
IMPLEMENTATION_FILES = (
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/clean_recovery.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/large_scale_continue.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/stem.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/train.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/train.py",
    "FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/general_backbone_pretraining.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/datasets.py",
    "experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/settings.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/configs/imagenet1k_clean_recovery.json",
)


def validate_source_payload(payload: Mapping[str, Any]) -> None:
    """Reject every source except the completed high-phase five-epoch proxy."""

    checks = (
        ("format", payload.get("format"), EXPECTED_SOURCE_FORMAT),
        ("checkpoint_role", payload.get("checkpoint_role"), EXPECTED_SOURCE_ROLE),
        ("epoch", int(payload.get("epoch", -1)), EXPECTED_SOURCE_EPOCH),
        (
            "config_digest",
            payload.get("config_digest"),
            EXPECTED_SOURCE_CONFIG_DIGEST,
        ),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise RuntimeError(
                f"Clean-recovery source {name} mismatch: expected {expected!r}, "
                f"got {actual!r}"
            )
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("Clean-recovery source has no model state mapping")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise RuntimeError("Clean-recovery source has no valid EMA shadow mapping")


def select_source_state(
    payload: Mapping[str, Any], state_variant: str
) -> dict[str, Any]:
    """Return a complete model state for the requested RAW or EMA variant."""

    validate_source_payload(payload)
    raw = dict(payload["model"])
    if state_variant == "raw":
        return raw
    if state_variant != "ema":
        raise ValueError("initialization.source_state_variant must be raw or ema")
    shadow = dict(payload["ema"]["shadow"])
    unknown = sorted(shadow.keys() - raw.keys())
    if unknown:
        raise RuntimeError(f"EMA source contains unknown model tensors: {unknown}")
    missing_trainable = sorted(
        name
        for name in raw
        if name in shadow and tuple(raw[name].shape) != tuple(shadow[name].shape)
    )
    if missing_trainable:
        raise RuntimeError(
            f"EMA source tensor shapes differ from RAW: {missing_trainable}"
        )
    raw.update(shadow)
    return raw


def _compatible_model_config(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    # Drop-path is a parameter-free training behavior and is intentionally
    # changed from 0.05 in the source proxy to 0.0 for clean recovery.
    ignored = {"stage_drop_path_rate", "seed"}
    for key in sorted((source.keys() | target.keys()) - ignored):
        source_value = source.get(key)
        if key in ignored:
            continue
        if key not in source or key not in target or target[key] != source_value:
            raise RuntimeError(
                f"Source/clean model config differs at {key}: "
                f"source={source_value!r}, clean={target.get(key)!r}"
            )


def initialize_from_completed_proxy(
    model: LargeScaleP11Model,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    values = config["initialization"]
    source_path = resolve_path(values["source_checkpoint"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Clean-recovery source is missing: {source_path}")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Clean-recovery source SHA256 mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Clean-recovery source checkpoint is not a mapping")
    validate_source_payload(payload)
    state_variant = str(values.get("source_state_variant", "raw"))
    _compatible_model_config(payload.get("model_config", {}), config["model"])
    if payload.get("stem_checkpoint_sha256") != model.stem.checkpoint_sha256:
        raise RuntimeError("Clean-recovery source stem identity mismatch")
    if state_variant == "ema":
        expected_shadow = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        actual_shadow = set(payload["ema"]["shadow"])
        if actual_shadow != expected_shadow:
            raise RuntimeError(
                "EMA source does not cover the trainable model exactly: "
                f"missing={sorted(expected_shadow - actual_shadow)}, "
                f"extra={sorted(actual_shadow - expected_shadow)}"
            )
    model.load_state_dict(select_source_state(payload, state_variant), strict=True)
    return {
        "mode": "post_strong_pretrain_clean_recovery",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_format": EXPECTED_SOURCE_FORMAT,
        "source_checkpoint_role": EXPECTED_SOURCE_ROLE,
        "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
        "source_config_digest": EXPECTED_SOURCE_CONFIG_DIGEST,
        "source_state_variant": state_variant,
        "source_best_raw_top1": float(payload.get("best_raw_top1", math.nan)),
        "source_best_ema_top1": float(payload.get("best_ema_top1", math.nan)),
        "source_global_optimizer_step": int(
            payload.get("global_optimizer_step", -1)
        ),
        "source_initial_phases_sha256": payload.get("initial_phases_sha256"),
        "source_optimizer_reused": False,
        "source_scheduler_reused": False,
        "source_scaler_reused": False,
        "clean_optimizer_initialized_fresh": True,
    }


def validate_clean_config(config: Mapping[str, Any], context: Context) -> None:
    training = config["training"]
    if int(training.get("expected_world_size", 1)) != 1 or context.world_size != 1:
        raise RuntimeError("The clean-recovery probe is fixed to one GPU")
    if int(training.get("epochs", -1)) != 5:
        raise RuntimeError("The registered clean-recovery probe is fixed to 5 epochs")
    effective_batch = int(training["batch_size"]) * int(
        training.get("gradient_accumulation_steps", 1)
    )
    if effective_batch != 96 or int(
        training.get("expected_effective_global_batch", -1)
    ) != 96:
        raise RuntimeError("The clean-recovery probe is fixed to global batch 96")
    if str(config.get("objective", {}).get("mode")) != (
        "post_strong_pretrain_clean_recovery"
    ):
        raise ValueError("Unexpected clean-recovery objective mode")
    if float(config["model"].get("stage_drop_path_rate", -1.0)) != 0.0:
        raise ValueError("Clean recovery requires stage_drop_path_rate=0")

    loss = config.get("loss", {})
    expected_loss = {
        "mode": "cross_entropy",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "batch_mix_probability": 0.0,
    }
    for key, expected in expected_loss.items():
        if loss.get(key) != expected:
            raise ValueError(
                f"Clean recovery requires loss.{key}={expected!r}, "
                f"got {loss.get(key)!r}"
            )
    if float(
        config.get("augmentation", {}).get("random_erasing_probability", -1.0)
    ) != 0.0:
        raise ValueError("Clean recovery requires random erasing to be disabled")
    phase_lr = float(config["optimizer"]["phase_learning_rate"])
    if not 7.0e-4 <= phase_lr <= 1.0e-3:
        raise ValueError("Clean-recovery phase LR must lie in [7e-4, 1e-3]")
    for key in (
        "electronic_learning_rate",
        "adapter_learning_rate",
        "head_learning_rate",
    ):
        if float(config["optimizer"][key]) >= phase_lr / 4.0:
            raise ValueError(f"optimizer.{key} is not sufficiently below phase LR")

    settings = load_imagenet_settings(resolve_path(config["imagenet_config"]))
    if settings.clip.randaugment_enabled:
        raise ValueError("Clean recovery requires RandAugment to be disabled")
    if tuple(settings.clip.random_resized_crop_scale) != (0.08, 1.0):
        raise ValueError("Unexpected clean RRC scale")
    if tuple(settings.clip.random_resized_crop_ratio) != (
        0.75,
        1.3333333333333333,
    ):
        raise ValueError("Unexpected clean RRC aspect ratio")
    if float(settings.clip.horizontal_flip_probability) != 0.5:
        raise ValueError("Clean recovery requires horizontal flip probability 0.5")
    if int(settings.clip.views_per_train_image) != 5:
        raise ValueError("Five recovery epochs require five deterministic RRC views")


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
        "world_size": 1,
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_save(path, _checkpoint_payload(**kwargs))


def run(config: dict[str, Any], context: Context, *, resume: bool) -> None:
    validate_clean_config(config, context)
    training = config["training"]
    seed_all(int(training.get("seed", 2026)), context.rank)
    output = resolve_path(config["output_dir"])
    last_path = output / "checkpoints" / "last.pt"
    if resume and not last_path.is_file():
        raise FileNotFoundError("--resume requires checkpoints/last.pt")
    if not resume and _run_has_artifacts(output):
        raise RuntimeError("--fresh refuses an output directory containing run artifacts")

    implementation = training_implementation_manifest(
        relative_paths=IMPLEMENTATION_FILES
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
    dataset_identity = {
        "dataset_digest": bundle.digest,
        "train_indices_sha256": canonical_sha256(train_indices),
        "validation_indices_sha256": canonical_sha256(validation_indices),
        "train_base_samples": len(train_indices),
        "validation_base_samples": len(validation_indices),
        "imagenet_config_sha256": sha256_file(
            resolve_path(config["imagenet_config"])
        ),
        "train_transform": {
            "schema": "rrc_flip_clean_recovery_v1",
            "random_resized_crop_scale": [0.08, 1.0],
            "random_resized_crop_ratio": [0.75, 1.3333333333333333],
            "horizontal_flip_probability": 0.5,
            "randaugment_enabled": False,
            "deterministic_views_per_image": 5,
        },
    }

    model_config = dict(config["model"])
    model_config.setdefault("seed", int(training.get("seed", 2026)))
    model = LargeScaleP11Model(resolve_path(config["stem_checkpoint"]), model_config)
    initialization = initialize_from_completed_proxy(model, config)
    source_phases = model.phase_snapshot()
    initial_phases_path = output / "initial_phases.pt"
    resume_payload: Mapping[str, Any] | None = None
    if resume:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_payload.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint format mismatch")
        if resume_payload.get("checkpoint_role") != "last":
            raise RuntimeError("Resume checkpoint is not a last-state checkpoint")
        if resume_payload.get("config_digest") != config["_config_digest"]:
            raise RuntimeError("Resume config digest mismatch")
        if resume_payload.get("implementation_manifest") != implementation:
            raise RuntimeError("Resume implementation/runtime manifest mismatch")
        if resume_payload.get("dataset_identity") != dataset_identity:
            raise RuntimeError("Resume dataset/index identity mismatch")
        if resume_payload.get("initialization") != initialization:
            raise RuntimeError("Resume source initialization identity mismatch")
        if resume_payload.get("stem_checkpoint_sha256") != model.stem.checkpoint_sha256:
            raise RuntimeError("Resume stem checkpoint identity mismatch")
        model.load_state_dict(resume_payload["model"], strict=True)
        if not initial_phases_path.is_file():
            raise FileNotFoundError("Exact resume requires initial_phases.pt")
        initial_phases = torch.load(
            initial_phases_path, map_location="cpu", weights_only=False
        )
        if not isinstance(initial_phases, torch.Tensor):
            raise RuntimeError("Initial phase snapshot is not a tensor")
    else:
        initial_phases = source_phases
    if tuple(initial_phases.shape) != tuple(source_phases.shape):
        raise RuntimeError("Initial phase snapshot shape differs from source model")
    initial_phases_sha256 = sha256_tensor(initial_phases)
    if resume_payload is not None and resume_payload.get(
        "initial_phases_sha256"
    ) != initial_phases_sha256:
        raise RuntimeError("Resume initial phase identity mismatch")

    report = model.parameter_report()
    if any(probability != 0.0 for probability in model.stage_drop_probabilities()):
        raise RuntimeError("Clean-recovery model unexpectedly enables stochastic depth")
    if float(report["optical_fraction_of_backbone_trainable"]) < 0.50:
        raise RuntimeError("Optical trainable-parameter fraction fell below 0.5")
    if float(report["minimum_optical_gate"]) < 0.50:
        raise RuntimeError("Optical gate fell below 0.5 before clean recovery")

    model.to(context.device)
    optimizer, optimizer_schema = build_layerwise_optimizer(model, config)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    micro_batches = min(
        len(train_loader), int(training.get("max_train_batches") or len(train_loader))
    )
    updates_per_epoch = math.ceil(micro_batches / accumulation)
    scheduler = build_scheduler(optimizer, config, updates_per_epoch)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training.get("use_amp", True))
        and context.device.type == "cuda",
        init_scale=float(training.get("amp_initial_scale", 256.0)),
        growth_interval=int(training.get("amp_growth_interval", 100000)),
    )
    ema_values = config.get("ema", {})
    ema = TrainableEMA(
        model,
        decay=float(ema_values.get("decay", 0.999)),
        warmup_updates=int(ema_values.get("warmup_updates", 100)),
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
        if int(resume_payload.get("world_size", -1)) != 1:
            raise RuntimeError("Resume world-size mismatch")
        rng_states = list(resume_payload.get("rng_states", []))
        if len(rng_states) != 1:
            raise RuntimeError("Resume checkpoint has incomplete RNG state")
        restore_rng_state(rng_states[0], context)
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload["global_optimizer_step"])
        best_raw = float(resume_payload["best_raw_top1"])
        best_ema = float(resume_payload["best_ema_top1"])
        history = list(resume_payload.get("history", []))

    manifest = {
        "experiment": "P11 post-strong-pretrain clean recovery probe",
        "interpretation": "recovery stage, not a new FA method group",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config": config["_config_path"],
        "config_digest": config["_config_digest"],
        "dataset_identity": dataset_identity,
        "objective": "clean supervised ImageNet-1K recovery",
        "augmentation": "RandomResizedCrop + horizontal flip only",
        "mixup": False,
        "cutmix": False,
        "randaugment": False,
        "random_erasing": False,
        "label_smoothing": 0.0,
        "stochastic_depth": False,
        "optimizer_state_inherited": False,
        "effective_global_batch": 96,
        "updates_per_epoch": updates_per_epoch,
        "initialization": initialization,
        "initial_phases_sha256": initial_phases_sha256,
        "model": report,
        "optimizer_groups": optimizer_schema,
        "ema": dict(ema_values),
        "implementation_manifest": implementation,
    }
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(output / "manifest.json", manifest)
        atomic_save(initial_phases_path, initial_phases)
        validation_sampler.set_epoch(0)
        baseline = evaluate(model, validation_loader, config, context)
        best_raw = float(baseline["top1_accuracy"])
        best_ema = best_raw
        rng_states = gather_rng_states(context)
        common = dict(
            model=model,
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
        )
        write_json(output / "metrics" / "initial_baseline.json", baseline)
        _save_checkpoint(output / "checkpoints" / "last.pt", role="last", **common)
        _save_checkpoint(
            output / "checkpoints" / "best_raw.pt", role="best_raw", **common
        )
        _save_checkpoint(
            output / "checkpoints" / "best_ema.pt", role="best_ema", **common
        )
        print(
            f"[baseline] top1={best_raw:.6f} top5={baseline['top5_accuracy']:.6f}",
            flush=True,
        )

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_sampler.set_epoch(epoch - 1)
        validation_sampler.set_epoch(0)
        train_metrics, gradient_report, global_step = train_epoch(
            model,
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
        raw_metrics = evaluate(model, validation_loader, config, context)
        with ema.apply(model):
            ema_metrics = evaluate(model, validation_loader, config, context)
        raw_top1 = float(raw_metrics["top1_accuracy"])
        ema_top1 = float(ema_metrics["top1_accuracy"])
        improved_raw = raw_top1 > best_raw
        improved_ema = ema_top1 > best_ema
        best_raw = max(best_raw, raw_top1)
        best_ema = max(best_ema, ema_top1)
        row = {
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "learning_rates": {
                group["name"]: group["lr"] for group in optimizer.param_groups
            },
            "train": train_metrics,
            "validation_raw": raw_metrics,
            "validation_ema": ema_metrics,
            "phase_gradients": gradient_report,
            "phase_motion_from_recovery_start": model.phase_motion(initial_phases),
            "optical_gates": model.optical_gates(),
            "electronic_skip_gates": model.electronic_skip_gates(),
        }
        history.append(row)
        rng_states = gather_rng_states(context)
        common = dict(
            model=model,
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
        )
        write_json(output / "metrics" / "history.json", history)
        write_json(output / "metrics" / "latest.json", row)
        _save_checkpoint(output / "checkpoints" / "last.pt", role="last", **common)
        if improved_raw:
            _save_checkpoint(
                output / "checkpoints" / "best_raw.pt", role="best_raw", **common
            )
        if improved_ema:
            _save_checkpoint(
                output / "checkpoints" / "best_ema.pt", role="best_ema", **common
            )
        _save_checkpoint(
            output / "checkpoints" / f"epoch_{epoch:03d}.pt",
            role="last",
            **common,
        )
        print(
            f"[epoch] {epoch}/{training['epochs']} raw_top1={raw_top1:.6f} "
            f"ema_top1={ema_top1:.6f} best_raw={best_raw:.6f} "
            f"best_ema={best_ema:.6f}",
            flush=True,
        )

    best_role = "best_ema" if best_ema >= best_raw else "best_raw"
    best_path = output / "checkpoints" / f"{best_role}.pt"
    best_payload = torch.load(best_path, map_location=context.device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    ema.load_state_dict(best_payload["ema"])
    scope = ema.apply(model) if best_role == "best_ema" else _null_scope()
    with scope:
        validation_sampler.set_epoch(0)
        final_metrics = evaluate(model, validation_loader, config, context)
        backbone_path = output / "checkpoints" / "backbone_best.pt"
        atomic_save(
            backbone_path,
            {
                "format": BACKBONE_FORMAT,
                "backbone": model.backbone_state_dict(),
                "state_variant": best_role,
                "best_epoch": int(best_payload["epoch"]),
                "source_training_checkpoint": str(best_path),
                "source_training_checkpoint_sha256": sha256_file(best_path),
                "source_proxy_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
                "config_digest": config["_config_digest"],
                "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
                "model_config": model_config,
                "model_report": model.parameter_report(),
                "initialization": initialization,
                "implementation_manifest": implementation,
                "temporary_imagenet_readout_exported": False,
            },
        )
    result = {
        "status": "complete",
        "interpretation": "post-strong-pretrain clean recovery; not a new method group",
        "best_state_variant": best_role,
        "best_epoch": int(best_payload["epoch"]),
        "best_raw_top1": best_raw,
        "best_ema_top1": best_ema,
        "best_validation": final_metrics,
        "backbone_checkpoint": str(backbone_path),
        "backbone_sha256": sha256_file(backbone_path),
        "initialization": initialization,
        "model": model.parameter_report(),
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover P11 classification boundaries after strong pretraining"
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
