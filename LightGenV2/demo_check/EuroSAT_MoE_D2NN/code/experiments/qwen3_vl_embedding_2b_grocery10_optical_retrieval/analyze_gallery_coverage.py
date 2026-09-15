from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .io_utils import seed_everything, write_csv, write_json
from .modeling import build_optical_student, load_backbone
from .prepare_grocery_retrieval_subset import (
    GrocerySample,
    prepare_grocery_subset,
)
from .retrieval_metrics import evaluate_embeddings
from .settings import load_settings
from .train_optical_retrieval import encode_student_samples, load_checkpoint


def select_additional_gallery_samples(
    train_samples: Sequence[GrocerySample],
    train_teacher_embeddings: torch.Tensor,
    official_gallery_samples: Sequence[GrocerySample],
    official_gallery_teacher_embeddings: torch.Tensor,
    *,
    per_sku: int,
) -> tuple[tuple[GrocerySample, ...], list[dict[str, Any]]]:
    """Choose train-only views nearest to each official gallery in Teacher space."""

    if per_sku <= 0:
        raise ValueError("per_sku must be positive")
    if len(train_samples) != len(train_teacher_embeddings):
        raise ValueError("Train samples and embeddings must have equal length")
    if len(official_gallery_samples) != len(official_gallery_teacher_embeddings):
        raise ValueError("Gallery samples and embeddings must have equal length")
    train_values = F.normalize(train_teacher_embeddings.float(), dim=-1)
    gallery_values = F.normalize(
        official_gallery_teacher_embeddings.float(), dim=-1
    )
    class_indexes = sorted({int(sample.sku_index) for sample in official_gallery_samples})
    selected: list[GrocerySample] = []
    records: list[dict[str, Any]] = []
    for class_index in class_indexes:
        candidate_indexes = [
            index
            for index, sample in enumerate(train_samples)
            if int(sample.sku_index) == class_index
        ]
        gallery_indexes = [
            index
            for index, sample in enumerate(official_gallery_samples)
            if int(sample.sku_index) == class_index
        ]
        if not candidate_indexes or not gallery_indexes:
            raise RuntimeError(
                f"SKU index {class_index} lacks train or official gallery samples"
            )
        prototype = F.normalize(
            gallery_values[gallery_indexes].mean(0, keepdim=True), dim=-1
        )[0]
        ranked = sorted(
            (
                (
                    float(torch.dot(train_values[index], prototype)),
                    str(train_samples[index].sample_id),
                    index,
                )
                for index in candidate_indexes
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if len(ranked) < per_sku:
            raise RuntimeError(
                f"SKU index {class_index} has only {len(ranked)} train views; "
                f"cannot select {per_sku}"
            )
        for rank, (similarity, _, index) in enumerate(ranked[:per_sku], 1):
            sample = train_samples[index]
            selected.append(sample)
            records.append(
                {
                    "sku_index": class_index,
                    "sku_name": sample.sku_name,
                    "rank": rank,
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "teacher_similarity_to_official_gallery": similarity,
                    "selection_source": "train_only",
                    "test_used_for_selection": False,
                }
            )
    return tuple(selected), records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether one iconic gallery view limits Grocery retrieval"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--additional-gallery-per-sku", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    bundle = prepare_grocery_subset(settings, persist=True)
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    teacher_store = TeacherEmbeddingStore(
        settings.teacher_cache_path, bundle, settings
    )
    replacement, readout = build_optical_student(loaded, settings)
    try:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        payload = load_checkpoint(checkpoint, replacement, readout)
        teacher_train = teacher_store.lookup(bundle.train_samples)
        teacher_official_gallery = teacher_store.lookup(bundle.gallery_samples)
        selected, selection_rows = select_additional_gallery_samples(
            bundle.train_samples,
            teacher_train,
            bundle.gallery_samples,
            teacher_official_gallery,
            per_sku=args.additional_gallery_per_sku,
        )
        student_query = encode_student_samples(
            loaded, replacement, readout, bundle.test_samples, settings
        )
        student_official_gallery = encode_student_samples(
            loaded, replacement, readout, bundle.gallery_samples, settings
        )
        student_additional_gallery = encode_student_samples(
            loaded, replacement, readout, selected, settings
        )
        expanded_samples = tuple(bundle.gallery_samples) + selected
        expanded_student = torch.cat(
            (student_official_gallery, student_additional_gallery), dim=0
        )
        expanded_teacher = teacher_store.lookup(expanded_samples)
        teacher_query = teacher_store.lookup(bundle.test_samples)

        evaluations = {
            "student_official_iconic_mean": evaluate_embeddings(
                student_query,
                bundle.test_samples,
                student_official_gallery,
                bundle.gallery_samples,
                bundle.class_names,
                "mean_prototype",
                system_name="student_official_iconic_mean",
            ).metrics,
            "student_expanded_three_view_mean": evaluate_embeddings(
                student_query,
                bundle.test_samples,
                expanded_student,
                expanded_samples,
                bundle.class_names,
                "mean_prototype",
                system_name="student_expanded_three_view_mean",
            ).metrics,
            "student_expanded_three_view_max": evaluate_embeddings(
                student_query,
                bundle.test_samples,
                expanded_student,
                expanded_samples,
                bundle.class_names,
                "max_similarity",
                system_name="student_expanded_three_view_max",
            ).metrics,
            "teacher_official_iconic_mean": evaluate_embeddings(
                teacher_query,
                bundle.test_samples,
                teacher_official_gallery,
                bundle.gallery_samples,
                bundle.class_names,
                "mean_prototype",
                system_name="teacher_official_iconic_mean",
            ).metrics,
            "teacher_expanded_three_view_mean": evaluate_embeddings(
                teacher_query,
                bundle.test_samples,
                expanded_teacher,
                expanded_samples,
                bundle.class_names,
                "mean_prototype",
                system_name="teacher_expanded_three_view_mean",
            ).metrics,
            "teacher_expanded_three_view_max": evaluate_embeddings(
                teacher_query,
                bundle.test_samples,
                expanded_teacher,
                expanded_samples,
                bundle.class_names,
                "max_similarity",
                system_name="teacher_expanded_three_view_max",
            ).metrics,
        }
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else settings.output_dir
            / "metrics"
            / "gallery_coverage_diagnostic.json"
        )
        write_json(
            output,
            {
                "diagnostic_only": True,
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(payload["epoch"]),
                "official_gallery_views_per_sku": settings.gallery_images_per_sku,
                "additional_train_views_per_sku": args.additional_gallery_per_sku,
                "selection_method": (
                    "nearest train-only image to official gallery in frozen "
                    "Teacher embedding space"
                ),
                "test_used_for_gallery_selection": False,
                "deployment_note": (
                    "Expanded results diagnose gallery coverage. They are not "
                    "the one-iconic-image primary result."
                ),
                "metrics": evaluations,
            },
        )
        write_csv(
            output.with_name("gallery_coverage_selected_views.csv"),
            selection_rows,
            list(selection_rows[0]),
        )
        for name, values in evaluations.items():
            print(
                f"{name}: Top-1={values['top1_retrieval_accuracy']:.4f} "
                f"Top-3={values['top3_retrieval_accuracy']:.4f} "
                f"MRR={values['mrr']:.4f}",
                flush=True,
            )
    finally:
        replacement.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
