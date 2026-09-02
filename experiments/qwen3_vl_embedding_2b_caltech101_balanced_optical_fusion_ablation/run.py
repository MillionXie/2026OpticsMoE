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
    ABLATION_MODES,
    build_hybrid_student,
    load_backbone,
    load_fair_initialization,
)
from .settings import load_settings, save_resolved_config


PHASES = {"prepare_data", "train", "evaluate"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caltech101 scale-matched convex optical-fusion ablation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_MODES),
        default="none",
        help="Evaluation-only counterfactual; remove_optical physically skips optics.",
    )
    parser.add_argument(
        "--evaluation-output-dir",
        default=None,
        help="Keep full/remove-optical/remove-electronic metrics in separate folders.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.evaluation_output_dir:
        if args.phase != "evaluate":
            parser.error("--evaluation-output-dir is only valid for evaluate")
        settings.output_dir = Path(args.evaluation_output_dir).expanduser().resolve()
    if args.phase != "evaluate" and args.ablation != "none":
        parser.error("--ablation is evaluation-only")
    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires an explicit --checkpoint")

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
    replacement, readout = build_hybrid_student(loaded, settings)
    try:
        if args.phase == "train":
            initialization = load_fair_initialization(
                settings, replacement, readout
            )
            write_json(
                settings.output_dir / "fair_initialization_report.json",
                initialization,
            )
            print(
                "[fair_initialization] same warmstart5 weights loaded; "
                f"four gates reset to alpha={settings.fusion_alpha_initial:.4f}",
                flush=True,
            )
            train_optical_retrieval(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
            )
        else:
            replacement.set_fusion_ablation(args.ablation)
            evaluate_all_systems(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
                Path(args.checkpoint).expanduser().resolve(),
            )
        write_json(
            settings.output_dir / "fusion_diagnostics_last_batch.json",
            replacement.fusion_diagnostics(),
        )
    finally:
        replacement.close()
    return 0


__all__ = ["main"]
