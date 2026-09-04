from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.evaluate_retrieval import (
    evaluate_all_systems,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    environment_report,
    seed_everything,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    train_optical_retrieval,
)

from .modeling import build_student, load_backbone
from .sampling import CyclicBalancedPKBatchSampler
from .settings import load_settings, save_resolved_config


PHASES = {"prepare_data", "train", "evaluate", "all"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caltech-101 controlled multiplane optical retrieval ablations"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_caltech101_subset(settings, persist=True)
    if args.phase == "prepare_data":
        print(bundle.metadata["counts"])
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_student(loaded, settings)
    report = replacement.student_architecture_report()
    phase_parameters = {
        id(parameter): parameter
        for values in replacement.phase_parameter_groups().values()
        for parameter in values
        if parameter.requires_grad
    }
    router_parameters = {
        id(parameter): parameter
        for parameter in replacement.router_parameters()
        if parameter.requires_grad
    }
    all_trainable = {
        id(parameter): parameter
        for module in (
            replacement.vision_surrogate,
            replacement.language_surrogate,
            readout,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    trainable_total = sum(parameter.numel() for parameter in all_trainable.values())
    optical_phase_total = sum(
        parameter.numel() for parameter in phase_parameters.values()
    )
    router_total = sum(parameter.numel() for parameter in router_parameters.values())
    electronic_nonrouter_total = trainable_total - optical_phase_total - router_total
    report["parameter_counts"] = {
        "vision": sum(p.numel() for p in replacement.vision_surrogate.parameters()),
        "language": sum(p.numel() for p in replacement.language_surrogate.parameters()),
        "retrieval_readout": sum(p.numel() for p in readout.parameters()),
        "trainable_total": trainable_total,
        "optical_phase": optical_phase_total,
        "electronic_router": router_total,
        "electronic_nonrouter_interfaces_and_readout": electronic_nonrouter_total,
        "optical_phase_fraction_of_trainable": (
            optical_phase_total / trainable_total if trainable_total else 0.0
        ),
    }
    write_json(settings.output_dir / "architecture_report.json", report)
    print(report, flush=True)

    shared_training = importlib.import_module(
        "experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval."
        "train_optical_retrieval"
    )
    original_sampler = shared_training.PKBatchSampler
    if settings.multiplane_sampling_mode == "cyclic_balanced":
        # Patch only the defining module used by the imported shared training
        # function.  Existing experiments and files remain unchanged.
        shared_training.PKBatchSampler = CyclicBalancedPKBatchSampler
    try:
        if args.phase in {"train", "all"}:
            train_optical_retrieval(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
                resume_checkpoint=(
                    Path(args.resume_checkpoint).expanduser().resolve()
                    if args.resume_checkpoint
                    else None
                ),
            )
            if args.phase == "train":
                return 0
        if args.phase in {"evaluate", "all"}:
            replacement.vision_surrogate.core.reset_analysis_accumulators()
            replacement.language_surrogate.core.reset_analysis_accumulators()
            evaluate_all_systems(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else None,
            )
            write_json(
                settings.output_dir / "metrics" / "multiplane_diagnostics.json",
                {
                    "schema_version": 1,
                    "variant": settings.multiplane_variant,
                    "vision": replacement.vision_surrogate.core.analysis_summary(),
                    "language": replacement.language_surrogate.core.analysis_summary(),
                },
            )
    finally:
        shared_training.PKBatchSampler = original_sampler
        replacement.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
