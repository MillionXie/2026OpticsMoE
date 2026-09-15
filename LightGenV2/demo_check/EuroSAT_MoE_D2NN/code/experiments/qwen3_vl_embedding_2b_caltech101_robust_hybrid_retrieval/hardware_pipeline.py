from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval import (
    hardware_pipeline as shared,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    load_checkpoint,
)
from experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval.modeling import (
    build_optical_student,
    load_backbone,
)

from .prepare_caltech101_retrieval import prepare_caltech101_subset
from .settings import load_settings


def build_runtime(hardware: shared.HardwareConfig) -> shared.Runtime:
    settings = load_settings(hardware.model_config)
    bundle = prepare_caltech101_subset(settings, persist=True)
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_optical_student(loaded, settings)
    checkpoint = load_checkpoint(hardware.checkpoint, replacement, readout)
    if (
        len(replacement.vision_surrogate.core.expert_layers) != 1
        or len(replacement.language_surrogate.core.expert_layers) != 1
    ):
        raise RuntimeError(
            "Hardware pipeline requires one expert plane plus one global plane"
        )
    replacement.set_phase_dropout_active(False)
    loaded.model.eval()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()
    write_json(
        hardware.output_dir / "00_manifest" / "checkpoint_metadata.json",
        checkpoint.get("metadata", {}),
    )
    return shared.Runtime(
        hardware, settings, bundle, loaded, replacement, readout
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four-plane SLM/CCD pipeline for Caltech101 target-10 retrieval"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(shared.PHASES))
    parser.add_argument("--use-simulation", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--queries-per-sku", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=("selected_test", "test_only", "full_dataset"),
        default=None,
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--artifact-profile", choices=("minimal", "full"), default=None
    )
    args = parser.parse_args()
    hardware = shared.load_hardware_config(args.config)
    requested_profile = args.artifact_profile
    if requested_profile is None and args.phase == "prepare":
        requested_profile = "minimal"
    if requested_profile is not None:
        hardware = replace(
            hardware, minimal_artifacts=requested_profile == "minimal"
        )
    if hardware.minimal_artifacts and args.phase != "prepare":
        raise RuntimeError(
            "The minimal artifact profile only supports prepare; use full for CCD replay"
        )
    if args.output_dir is not None:
        hardware = replace(
            hardware, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    if args.checkpoint is not None:
        hardware = replace(
            hardware, checkpoint=Path(args.checkpoint).expanduser().resolve()
        )
    if args.queries_per_sku is not None:
        if args.queries_per_sku <= 0:
            raise ValueError("--queries-per-sku must be positive")
        hardware = replace(hardware, queries_per_sku=args.queries_per_sku)
    if args.selection_mode is not None:
        hardware = replace(hardware, selection_mode=args.selection_mode)
    if args.sample_limit is not None:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        hardware = replace(hardware, sample_limit=args.sample_limit)

    seed_everything(42)
    runtime = build_runtime(hardware)
    try:
        if args.phase == "prepare":
            report = shared.prepare(runtime)
            print(
                f"Prepared {report['sample_count']} Caltech target-10 samples "
                f"and four shared masks under {hardware.output_dir}"
            )
        elif args.phase == "process_vision_expert":
            shared.process_vision_expert(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_vision_global":
            shared.process_vision_global(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_language_expert":
            shared.process_language_expert(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_language_global":
            metrics = shared.process_language_global(
                runtime, use_simulation=args.use_simulation
            )
            print(
                f"Hardware replay Top-1={metrics['top1_retrieval_accuracy']:.4f} "
                f"Top-3={metrics['top3_retrieval_accuracy']:.4f} "
                f"MRR={metrics['mrr']:.4f}"
            )
        else:
            shared.prepare(runtime)
            shared.process_vision_expert(runtime, use_simulation=True)
            shared.process_vision_global(runtime, use_simulation=True)
            shared.process_language_expert(runtime, use_simulation=True)
            metrics = shared.process_language_global(runtime, use_simulation=True)
            print(
                f"Simulation replay complete: "
                f"Top-1={metrics['top1_retrieval_accuracy']:.4f}"
            )
    finally:
        shared.close_runtime(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
