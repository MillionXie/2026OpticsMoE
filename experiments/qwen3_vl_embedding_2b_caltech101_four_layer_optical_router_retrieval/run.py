from __future__ import annotations

import argparse
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
    save_checkpoint,
    train_optical_retrieval,
)

from .modeling import (
    build_hybrid_student,
    load_backbone,
    load_warmstart5_initialization,
)
from .settings import load_settings, save_resolved_config


PHASES = {
    "prepare_data",
    "materialize_initialization",
    "train",
    "evaluate",
}


def _device(settings: object) -> torch.device:
    requested = str(settings.device)
    return torch.device(
        requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Caltech101 warmstart5 electronic/optical router and top-k ablation"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    # Keep the dataset split and PK batches paired through settings.random_seed,
    # while allowing independent repeated Router/perturbation seeds.
    seed_everything(settings.router_optimization_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_caltech101_subset(settings, persist=True)
    if args.phase == "prepare_data":
        print(bundle.metadata["counts"])
        return 0

    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("--phase evaluate requires an explicit --checkpoint")
    if args.phase != "evaluate" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid with --phase evaluate")
    if args.resume_checkpoint and args.phase != "train":
        parser.error("--resume-checkpoint is only valid with --phase train")

    loaded = load_backbone(settings, _device(settings))
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    try:
        write_json(
            settings.output_dir / "student_architecture.json",
            replacement.student_architecture_report(),
        )
        if args.phase == "train":
            if args.resume_checkpoint is None:
                report = load_warmstart5_initialization(
                    settings, replacement, readout
                )
                write_json(
                    settings.output_dir / "router_initialization_report.json",
                    report,
                )
                print(
                    f"[router_init] backend={settings.router_backend} "
                    f"top_k={settings.top_k} source_sha={report['sha256']}",
                    flush=True,
                )
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
            return 0

        if args.phase == "materialize_initialization":
            report = load_warmstart5_initialization(settings, replacement, readout)
            trainable = [
                parameter
                for module in (
                    replacement.vision_surrogate,
                    replacement.language_surrogate,
                    readout,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate)
            output = (
                settings.output_dir
                / "converted_warmstart5_initialization_checkpoint.pt"
            )
            save_checkpoint(
                output,
                replacement,
                readout,
                optimizer,
                report["source_epoch"],
                report["source_train_loss"],
                settings,
                weight_variant="ema",
            )
            write_json(
                settings.output_dir / "router_initialization_report.json", report
            )
            print(output, flush=True)
            return 0

        print(
            "[sealed_test] exactly one explicit evaluation; do not use this "
            "test result to select another epoch or router variant",
            flush=True,
        )
        evaluate_all_systems(
            loaded,
            replacement,
            readout,
            bundle,
            None,
            settings,
            Path(args.checkpoint).expanduser().resolve(),
        )
    finally:
        replacement.close()
    return 0


__all__ = ["main"]
