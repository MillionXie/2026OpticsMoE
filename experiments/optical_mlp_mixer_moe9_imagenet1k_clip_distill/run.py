from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path

import torch

from .datasets import load_imagenet, write_dataset_report
from .io_utils import ensure_output_directories, environment_report, write_json
from .settings import load_settings
from .teacher_cache import build_all_clip_caches
from .training import (
    barrier,
    final_evaluation,
    finalize_distributed,
    initialize_distributed,
    seed_everything,
    train,
)


PHASES = ("prepare_data", "clip_cache", "train", "evaluate", "all")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Train the seven-block OpticalMixerMoE9 on ImageNet-1K with a "
            "frozen OpenAI CLIP ViT-B/16 teacher."
        )
    )
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--phase", choices=PHASES, default="all")
    result.add_argument("--device", type=str)
    result.add_argument("--epochs", type=int)
    result.add_argument("--output-dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.device:
        settings.training.device = args.device
    if args.epochs is not None:
        settings.training.epochs = int(args.epochs)
    if args.output_dir is not None:
        settings.training.output_dir = args.output_dir.expanduser().resolve()
    settings.validate()
    context = initialize_distributed(settings.training.device)
    seed_everything(settings.training.seed, context.rank)
    try:
        if context.is_main:
            ensure_output_directories(settings.training.output_dir)
            write_json(
                settings.training.output_dir / "config_resolved.json",
                settings.to_dict(),
            )
            write_json(
                settings.training.output_dir / "environment.json",
                environment_report(),
            )
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"phase={args.phase} rank={context.rank}/{context.world_size} "
                f"device={context.device}",
                flush=True,
            )
        barrier()
        bundle = load_imagenet(settings)
        if context.is_main:
            write_dataset_report(
                bundle, settings, settings.training.output_dir / "dataset.json"
            )
            write_json(
                settings.training.output_dir / "optical_parameter_formula.json",
                settings.optical_parameter_formula,
            )
            print(
                f"ImageNet-1K train={bundle.train.base_sample_count:,} "
                f"validation={bundle.validation.base_sample_count:,} "
                f"classes={len(bundle.class_names)}",
                flush=True,
            )
            print(
                "Optical phase parameters="
                f"{settings.optical_parameter_formula['optical_phase_parameters_total']:,}",
                flush=True,
            )
        barrier()
        if args.phase == "prepare_data":
            return 0
        if args.phase in {"clip_cache", "all"}:
            if context.is_main:
                reports = build_all_clip_caches(bundle, settings, context.device)
                write_json(
                    settings.training.output_dir / "metrics" / "clip_cache.json",
                    reports,
                )
            barrier()
            if args.phase == "clip_cache":
                return 0
        if args.phase in {"train", "all"}:
            train(bundle, settings, context)
            barrier()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.phase == "train":
                return 0
        if args.phase in {"evaluate", "all"}:
            final_evaluation(bundle, settings, context)
        return 0
    finally:
        finalize_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
