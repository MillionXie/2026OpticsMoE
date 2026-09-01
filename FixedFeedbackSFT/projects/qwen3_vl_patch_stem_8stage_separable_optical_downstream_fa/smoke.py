from __future__ import annotations

import argparse
import time

import torch

from .settings import add_settings_arguments, load_settings, load_settings_from_args
from .training import (
    CHECKPOINT_FORMAT,
    _load_common_start,
    _train_epoch,
    build_data,
    build_model,
    build_optimizer_scheduler,
    configure_runtime_feedback,
    evaluate,
    gradient_diagnostic,
    implementation_sha256,
    save_torch,
    seed_everything,
    write_json,
)


def main() -> None:
    parser = add_settings_arguments(
        argparse.ArgumentParser(description="One-full-batch P12 CUDA/gradient smoke")
    )
    args = parser.parse_args()
    settings = load_settings_from_args(args)
    formal_root = load_settings(
        args.config,
        task=settings.task,
        method=settings.method,
        seed=settings.seed,
    ).paths.output_root
    if settings.paths.output_root == formal_root:
        raise ValueError(
            "Smoke refuses the formal output root; pass an isolated --output-root"
        )
    settings.validate_runtime_paths()
    if settings.limits.max_train_batches != 1:
        raise ValueError("Smoke runs must pass --max-train-batches 1")
    if settings.limits.max_validation_batches != 1:
        raise ValueError("Smoke runs must pass --max-validation-batches 1")
    if not torch.cuda.is_available():
        raise RuntimeError("P12 smoke requires a CUDA GPU")

    seed_everything(settings.seed)
    _, bundle, loaders = build_data(settings)
    model = build_model(settings)
    synthetic_common_sha256: str | None = None
    if settings.updates_backbone:
        # Exercise the exact strict common-start loader in an isolated smoke
        # output root. This artifact is explicitly synthetic and is never a
        # formal NoFT result.
        save_torch(
            settings.paths.common_start_checkpoint,
            {
                "format": CHECKPOINT_FORMAT,
                "method": "noft",
                "task": settings.task,
                "seed": settings.seed,
                "epoch": 1,
                "selected_as_common_start": True,
                "completed_head_only_epochs": settings.training.head_only_epochs,
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "source_checkpoint_sha256": model.source_manifest["sha256"],
                "implementation_sha256": implementation_sha256(),
                "model": model.state_dict(),
                "synthetic_smoke_only": True,
            },
        )
        synthetic_common_sha256 = _load_common_start(
            model,
            settings,
            bundle.manifest_sha256,
            implementation_sha256(),
            require_completed_noft=False,
        )
    model.set_backbone_trainable(settings.updates_backbone)
    configure_runtime_feedback(model, settings)
    device = torch.device("cuda", 0)
    model.to(device)
    configure_runtime_feedback(model, settings)
    optimizer, scheduler = build_optimizer_scheduler(model, settings, len(loaders["train"]))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=settings.training.use_amp,
        init_scale=settings.training.amp_initial_scale,
        growth_interval=settings.training.amp_growth_interval,
    )
    validation_batch = next(iter(loaders["val"]))
    diagnostic = gradient_diagnostic(model, validation_batch, settings, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    train = _train_epoch(
        model, loaders["train"], settings, device, optimizer, scheduler, scaler, epoch=1
    )
    validation = evaluate(
        model,
        loaders["val"],
        settings,
        device,
        max_batches=1,
    )
    torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "configured_train_batch_size": settings.train_batch_size,
        "configured_evaluation_batch_size": settings.evaluation_batch_size,
        "train_samples_in_step": train["samples"],
        "train": train,
        "validation": validation,
        "gradient_diagnostic": diagnostic,
        "model_report": model.parameter_report(),
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "synthetic_common_start_sha256": synthetic_common_sha256,
        "synthetic_common_start_only": settings.updates_backbone,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.time() - started,
    }
    output = settings.output_dir / "smoke_result.json"
    write_json(output, result)
    print(
        f"[P12 smoke] passed task={settings.task} method={settings.method} "
        f"batch={train['samples']} peak_gib={result['peak_cuda_memory_bytes'] / 2**30:.3f} "
        f"result={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
