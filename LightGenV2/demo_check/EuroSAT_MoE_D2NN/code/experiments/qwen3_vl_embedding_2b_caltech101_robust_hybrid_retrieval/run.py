from __future__ import annotations

import argparse
from pathlib import Path

import torch

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
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.visualize_retrieval_results import (
    visualize_saved_results,
)
from experiments.qwen3_vl_embedding_2b_grocery10_robust_hybrid_retrieval.modeling import (
    build_optical_student,
    load_backbone,
)

from .prepare_caltech101_retrieval import prepare_caltech101_subset
from .settings import load_settings, save_resolved_config
from .teacher_cache import TeacherEmbeddingStore, build_teacher_embedding_cache


PHASES = {
    "prepare_data",
    "cache_teacher_embeddings",
    "train",
    "evaluate",
    "visualize",
    "all",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robust hybrid Qwen3-VL optical Caltech101 retrieval"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--force-teacher-cache", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_caltech101_subset(settings, persist=True)
    if args.phase == "prepare_data":
        counts = bundle.metadata["counts"]
        print(
            f"Prepared Caltech101 retrieval: train={counts['train']} "
            f"test={counts['test']} gallery={counts['gallery']}"
        )
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    save_resolved_config(settings)
    if args.phase in {"cache_teacher_embeddings", "all"}:
        build_teacher_embedding_cache(
            loaded, bundle, settings, force=args.force_teacher_cache
        )
        if args.phase == "cache_teacher_embeddings":
            return 0
    teacher_store = TeacherEmbeddingStore(
        settings.teacher_cache_path, bundle, settings
    )
    replacement, readout = build_optical_student(loaded, settings)
    try:
        if args.phase in {"train", "all"}:
            train_optical_retrieval(
                loaded,
                replacement,
                readout,
                bundle,
                teacher_store,
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
            checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
            evaluate_all_systems(
                loaded,
                replacement,
                readout,
                bundle,
                teacher_store,
                settings,
                checkpoint,
            )
            if args.phase == "evaluate":
                return 0
        if args.phase in {"visualize", "all"}:
            visualize_saved_results(settings)
    finally:
        replacement.close()
    return 0
