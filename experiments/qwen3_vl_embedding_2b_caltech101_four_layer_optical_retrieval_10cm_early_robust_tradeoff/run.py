from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.modeling import (
    sha256_file,
)
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

from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings, save_resolved_config


PHASES = {"prepare_data", "train", "evaluate"}


def _validate_source(settings: object) -> str:
    checkpoint = settings.continuation_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != settings.continuation_sha256:
        raise RuntimeError(
            "Warm-start SHA-256 mismatch: "
            f"expected={settings.continuation_sha256} actual={digest}"
        )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Early robust Caltech101 fusion/zero-order trade-off study"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--checkpoint", default=None)
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
    if args.phase == "train" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid with --phase evaluate")
    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("--phase evaluate requires an explicit --checkpoint")

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
            digest = _validate_source(settings)
            write_json(
                settings.output_dir / "continuation_initialization_report.json",
                {
                    "mode": "strict_stage_a_resume_weights_only",
                    "path": str(settings.continuation_checkpoint),
                    "sha256": digest,
                    "optimizer_state": "reset",
                    "tradeoff_variant": settings.tradeoff_variant,
                },
            )
            train_optical_retrieval(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
                resume_checkpoint=settings.continuation_checkpoint,
            )
            return 0

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
