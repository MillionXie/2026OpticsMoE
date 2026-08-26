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
    load_dual_initialization,
    load_stage_a_initialization,
)
from .settings import load_settings, save_resolved_config


PHASES = {"prepare_data", "train", "evaluate"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sealed-test 5%-floor Caltech101 four-layer warm-start hybrid"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
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

    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("sealed-test evaluate requires an explicit --checkpoint")
    if args.phase == "train" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid with --phase evaluate")
    if args.resume_checkpoint and args.phase != "train":
        parser.error("--resume-checkpoint is only valid with --phase train")

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    try:
        if args.phase == "train":
            if args.resume_checkpoint is None:
                initialization = (
                    load_dual_initialization(settings, replacement, readout)
                    if settings.warmstart_stage == "optical_calibration"
                    else load_stage_a_initialization(settings, replacement, readout)
                )
                write_json(
                    settings.output_dir / "warmstart_initialization_report.json",
                    initialization,
                )
                print(
                    f"[warmstart] stage={settings.warmstart_stage} "
                    f"mode={initialization['mode']}",
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

        checkpoint = Path(args.checkpoint).expanduser().resolve()
        print(
            "[sealed_test] one explicit evaluation; this result must not be used "
            "to select another epoch from the same run",
            flush=True,
        )
        evaluate_all_systems(
            loaded,
            replacement,
            readout,
            bundle,
            None,
            settings,
            checkpoint,
        )
    finally:
        replacement.close()
    return 0


__all__ = ["main"]
