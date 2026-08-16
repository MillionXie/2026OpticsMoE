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
    train_optical_retrieval,
)

from .modeling import (
    build_hybrid_student,
    load_backbone,
    load_electronic_initialization,
)
from .settings import load_settings, save_resolved_config


PHASES = {"prepare_data", "train", "evaluate", "all"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caltech101 Vision-2D electronic + Language block-2 optical residual"
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
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    try:
        if args.phase in {"train", "all"}:
            if args.resume_checkpoint is None:
                report = load_electronic_initialization(
                    settings.initial_electronic_checkpoint, replacement, readout
                )
                write_json(settings.output_dir / "electronic_initialization.json", report)
                print(
                    "[initialization] loaded frozen 2D electronic baseline "
                    f"from epoch={report['source_epoch']}",
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
            if args.phase == "train":
                return 0
        if args.phase in {"evaluate", "all"}:
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
    finally:
        replacement.close()
    return 0

