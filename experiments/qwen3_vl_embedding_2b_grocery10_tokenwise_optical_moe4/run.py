from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    TeacherEmbeddingStore,
    build_teacher_embedding_cache,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    environment_report,
    seed_everything,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    prepare_grocery_subset,
)

from .modeling import build_student, load_backbone
from .settings import load_settings, save_resolved_config
from .training import evaluate, train


PHASES = {"prepare_data", "cache_teacher_embeddings", "train", "evaluate", "all"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapter-free per-token Optical MoE4 Qwen3-VL Grocery retrieval"
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
    write_json(settings.output_dir / "architecture_static.json", {
        "token_hidden": [settings.max_tokens, settings.hidden_size],
        "token_field": [settings.max_tokens, settings.token_feature_side, settings.token_feature_side],
        "experts_per_token": settings.num_experts,
        "top_k_per_token": settings.top_k,
        "expert_group_size": [settings.expert_group_height, settings.expert_group_width],
        "active_panel_size": [settings.active_height, settings.active_width],
        "propagation_canvas": [settings.canvas_size, settings.canvas_size],
        "second_plane_mode": settings.second_plane_mode,
        "vision_input_adapter": None,
        "vision_output_adapter": None,
        "language_stack": settings.student_language_mode,
        "language_input_adapter": (
            [2048, 1024] if settings.student_language_mode == "optical_moe" else None
        ),
        "language_output_adapter": (
            [1024, 2048] if settings.student_language_mode == "optical_moe" else None
        ),
        "deepstack_enabled": settings.student_deepstack_enabled,
        "visual_text_merge_count": 1 if not settings.student_deepstack_enabled else 4,
    })
    bundle = prepare_grocery_subset(settings, persist=True)
    if args.phase == "prepare_data":
        print(
            f"prepared train={len(bundle.train_samples)} gallery={len(bundle.gallery_samples)} "
            f"test={len(bundle.test_samples)}"
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
    teacher_store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    replacement = build_student(loaded, settings)
    try:
        write_json(
            settings.output_dir / "architecture_runtime.json",
            replacement.architecture_report(),
        )
        if args.phase in {"train", "all"}:
            train(
                loaded,
                replacement,
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
            evaluate(
                loaded,
                replacement,
                bundle,
                teacher_store,
                settings,
                checkpoint=(
                    Path(args.checkpoint).expanduser().resolve()
                    if args.checkpoint
                    else None
                ),
            )
    finally:
        replacement.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
