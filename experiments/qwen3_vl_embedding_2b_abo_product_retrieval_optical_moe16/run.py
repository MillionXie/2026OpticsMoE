from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    load_backbone,
)

from .datasets import prepare_abo
from .evaluation import evaluate_all
from .io_utils import environment_report, seed_everything, write_json
from .modeling import build_encoder, load_vision_backbone
from .settings import load_settings, save_resolved_config
from .teacher_adapter import train_teacher_adapter
from .teacher_cache import TeacherEmbeddingStore, build_teacher_cache
from .training import train_stage1, train_stage2
from .visualize import visualize_results


PHASES = {
    "prepare_data",
    "cache_teacher",
    "train_teacher_adapter",
    "train_stage1",
    "train_stage2",
    "evaluate",
    "visualize",
    "all",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ABO two-stage Qwen3-VL -> Optical MoE16 product retrieval"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--force-teacher-cache", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    bundle = prepare_abo(settings, persist=True)
    if args.phase == "prepare_data":
        return 0
    if args.phase == "visualize":
        visualize_results(bundle, settings)
        return 0

    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    if args.phase in {"cache_teacher", "all"}:
        loaded_teacher = load_backbone(settings, device)
        settings.resolve_architecture(loaded_teacher.model)
        save_resolved_config(settings)
        build_teacher_cache(
            loaded_teacher,
            bundle,
            settings,
            force=args.force_teacher_cache,
        )
        del loaded_teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.phase == "cache_teacher":
            return 0

    if args.phase in {"train_teacher_adapter", "all"}:
        if settings.teacher_embedding_mode != "adapted_head":
            if args.phase == "train_teacher_adapter":
                raise RuntimeError(
                    "train_teacher_adapter requires "
                    "retrieval.teacher_embedding_mode=adapted_head"
                )
        else:
            train_teacher_adapter(bundle, settings, device)
        if args.phase == "train_teacher_adapter":
            return 0

    teacher_store = TeacherEmbeddingStore(bundle, settings)
    loaded = (
        load_backbone(settings, device)
        if settings.student_architecture == "multimodal_optical"
        else load_vision_backbone(settings, device)
    )
    encoder = build_encoder(loaded, settings)
    save_resolved_config(settings)
    try:
        if args.phase in {"train_stage1", "all"}:
            train_stage1(
                loaded, encoder, bundle, teacher_store, settings
            )
            if args.phase == "train_stage1":
                return 0
        if args.phase in {"train_stage2", "all"}:
            stage1_checkpoint = (
                Path(args.stage1_checkpoint).expanduser().resolve()
                if args.stage1_checkpoint
                else None
            )
            train_stage2(
                loaded,
                encoder,
                bundle,
                teacher_store,
                settings,
                stage1_checkpoint,
            )
            if args.phase == "train_stage2":
                return 0
        if args.phase in {"evaluate", "all"}:
            checkpoint = (
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else None
            )
            metrics = evaluate_all(
                loaded,
                encoder,
                bundle,
                teacher_store,
                settings,
                checkpoint,
            )
            teacher = metrics["systems"]["teacher"]
            student = metrics["systems"]["student"]
            aligned = metrics["systems"]["student_query_teacher_gallery"]
            print(
                "Teacher retrieval: "
                f"Top-1={teacher['top1_accuracy']:.4f} "
                f"R@5={teacher['recall_at_5']:.4f} "
                f"R@10={teacher['recall_at_10']:.4f} "
                f"mAP={teacher['mean_average_precision']:.4f}"
            )
            print(
                "Optical Student retrieval: "
                f"Top-1={student['top1_accuracy']:.4f} "
                f"R@5={student['recall_at_5']:.4f} "
                f"R@10={student['recall_at_10']:.4f} "
                f"mAP={student['mean_average_precision']:.4f}"
            )
            print(
                "Diagnostic Student-query / Teacher-gallery: "
                f"Top-1={aligned['top1_accuracy']:.4f} "
                f"R@5={aligned['recall_at_5']:.4f} "
                f"R@10={aligned['recall_at_10']:.4f} "
                f"mAP={aligned['mean_average_precision']:.4f}"
            )
            retention = metrics["comparison"]["student_teacher_retention"]
            print(
                "Student/Teacher retention: "
                f"Top-1={_ratio_text(retention['top1'])} "
                f"R@5={_ratio_text(retention['recall_at_5'])} "
                f"R@10={_ratio_text(retention['recall_at_10'])} "
                f"mAP={_ratio_text(retention['mean_average_precision'])}"
            )
            if args.phase == "evaluate":
                return 0
        if args.phase in {"visualize", "all"}:
            visualize_results(bundle, settings)
    finally:
        encoder.restore_native()
    return 0


def _ratio_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
