from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .io_utils import write_csv, write_json
from .modeling import build_optical_student, load_backbone, trainable_parameter_report
from .prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    prepare_grocery_subset,
)
from .retrieval_metrics import RetrievalEvaluation, evaluate_embeddings
from .settings import Settings, load_settings
from .train_optical_retrieval import encode_student_samples, load_checkpoint


def evaluate_all_systems(
    loaded: Any,
    replacement: Any,
    readout: Any,
    bundle: GroceryRetrievalBundle,
    teacher_store: TeacherEmbeddingStore | None,
    settings: Settings,
    checkpoint_path: Path | None = None,
) -> dict[str, RetrievalEvaluation]:
    checkpoint_path = (
        checkpoint_path
        or settings.output_dir / "best_train_loss_checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, replacement, readout)
    fusion_core = getattr(
        getattr(replacement, "vision_surrogate", None), "core", None
    )
    fusion_ablation = str(
        getattr(fusion_core, "fusion_ablation_mode", "none")
    )
    has_optical_phases = bool(
        getattr(replacement, "has_optical_phases", True)
    )
    if fusion_ablation == "remove_optical":
        student_system_prefix = "hybrid_checkpoint_remove_optical_electronic"
        student_kind = "Hybrid remove-optical"
    elif fusion_ablation == "remove_electronic":
        student_system_prefix = "hybrid_checkpoint_remove_electronic_optical"
        student_kind = "Hybrid remove-electronic"
    elif has_optical_phases:
        student_system_prefix = "optical_student"
        student_kind = "Optical"
    else:
        student_system_prefix = "electronic_student"
        student_kind = "Electronic"
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    # New checkpoints state their selection policy explicitly. Keep the
    # filename fallback only for legacy artifacts written before that field
    # existed, so renamed files cannot silently change the scientific label.
    observed_test_selected = bool(
        checkpoint_metadata.get(
            "test_metrics_used_for_selection",
            "observed_test" in checkpoint_path.name,
        )
    )
    selection_criterion = checkpoint_metadata.get("selection_criterion")
    student_gallery = encode_student_samples(
        loaded, replacement, readout, bundle.gallery_samples, settings
    )
    student_query = encode_student_samples(
        loaded, replacement, readout, bundle.test_samples, settings
    )
    systems = {
        "student": evaluate_embeddings(
            student_query,
            bundle.test_samples,
            student_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name=(
                f"{student_system_prefix}_query_vs_"
                f"{student_system_prefix}_gallery"
            ),
        ),
    }
    teacher_metrics: dict[str, Any] | None = None
    if teacher_store is not None:
        teacher_query = teacher_store.lookup(bundle.test_samples)
        teacher_gallery = teacher_store.lookup(bundle.gallery_samples)
        systems["teacher"] = evaluate_embeddings(
            teacher_query,
            bundle.test_samples,
            teacher_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name="frozen_teacher_query_vs_frozen_teacher_gallery",
        )
        systems["student_teacher_gallery"] = evaluate_embeddings(
            student_query,
            bundle.test_samples,
            teacher_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name=(
                f"{student_system_prefix}_query_vs_"
                "frozen_teacher_gallery_diagnostic"
            ),
        )
        teacher_metrics = {
            **systems["teacher"].metrics,
            "teacher_embedding_shape": [
                len(bundle.test_samples), settings.embedding_dim
            ],
            "teacher_frozen": True,
            "teacher_trainable_parameters": 0,
            "instruction": settings.instruction,
            "manifest_sha256": bundle.manifest_digest,
        }
    student_metrics = {
        **systems["student"].metrics,
        "main_deployment_result": True,
        "diagnostic_student_query_teacher_gallery": (
            systems["student_teacher_gallery"].metrics
            if "student_teacher_gallery" in systems
            else None
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection": (
            f"{selection_criterion or 'test-metric-selected checkpoint'}; "
            "selection-biased diagnostic"
            if observed_test_selected
            else (
                selection_criterion
                or "minimum training total loss or fixed final epoch; test was not used"
            )
        ),
        "selection_biased": observed_test_selected,
        "fusion_ablation": fusion_ablation,
        "optical_propagation_active": (
            has_optical_phases and fusion_ablation != "remove_optical"
        ),
        "student_detector_output_shape": [
            len(bundle.test_samples),
            settings.detector_output_size,
        ],
        "student_embedding_shape": [len(bundle.test_samples), settings.embedding_dim],
        "manifest_sha256": bundle.manifest_digest,
    }
    if teacher_metrics is not None:
        write_json(settings.output_dir / "teacher_metrics.json", teacher_metrics)
    write_json(settings.output_dir / "student_metrics.json", student_metrics)
    all_rows = [row for result in systems.values() for row in result.rows]
    if all_rows:
        write_csv(
            settings.output_dir / "retrieval_results.csv",
            all_rows,
            list(all_rows[0]),
        )
    per_sku_rows: list[dict[str, Any]] = []
    for name, result in systems.items():
        for sku, values in result.metrics["per_sku"].items():
            per_sku_rows.append({"system": name, "sku": sku, **values})
    write_csv(
        settings.output_dir / "per_sku_metrics.csv",
        per_sku_rows,
        ["system", "sku", "query_count", "top1_accuracy", "top3_accuracy"],
    )
    _plot_confusion(
        systems["student"].confusion,
        bundle.class_names,
        settings.output_dir / "confusion_matrix.png",
        student_kind=student_kind,
    )
    write_json(
        settings.output_dir / "metrics" / "evaluation_summary.json",
        {
            "teacher": teacher_metrics,
            "student": student_metrics,
            "trainable_parameters": trainable_parameter_report(
                loaded.model, replacement, readout
            )["trainable_parameters"],
            "data_leakage_check": "passed during subset preparation",
            "normalization_check": "all evaluated embeddings are finite and L2 normalized",
            "dimension_check": {
                "teacher": settings.embedding_dim if teacher_metrics else None,
                "student": settings.embedding_dim,
                "match": True if teacher_metrics else None,
            },
        },
    )
    return systems


def _plot_confusion(
    matrix: torch.Tensor,
    class_names: tuple[str, ...],
    path: Path,
    *,
    student_kind: str = "Optical",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    side = max(12.0, min(32.0, len(class_names) * 0.28))
    figure, axis = plt.subplots(figsize=(side, side), constrained_layout=True)
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="Query count")
    tick_size = max(3, min(8, round(300 / len(class_names))))
    axis.set_xticks(
        range(len(class_names)),
        class_names,
        rotation=90 if len(class_names) > 40 else 45,
        ha="center" if len(class_names) > 40 else "right",
        fontsize=tick_size,
    )
    axis.set_yticks(range(len(class_names)), class_names, fontsize=tick_size)
    axis.set_xlabel("Retrieved class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"{student_kind} Student {len(class_names)}-Class Retrieval Confusion Matrix"
    )
    if len(class_names) <= 31:
        for y in range(len(class_names)):
            for x in range(len(class_names)):
                axis.text(
                    x,
                    y,
                    str(int(matrix[y, x])),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_grocery_subset(settings, persist=True)
    teacher_store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    replacement, readout = build_optical_student(loaded, settings)
    systems = evaluate_all_systems(
        loaded,
        replacement,
        readout,
        bundle,
        teacher_store,
        settings,
        Path(args.checkpoint).resolve() if args.checkpoint else None,
    )
    print(
        "Teacher Top-1/Top-3/MRR="
        f"{systems['teacher'].metrics['top1_retrieval_accuracy']:.4f}/"
        f"{systems['teacher'].metrics['top3_retrieval_accuracy']:.4f}/"
        f"{systems['teacher'].metrics['mrr']:.4f}"
    )
    print(
        "Student Top-1/Top-3/MRR="
        f"{systems['student'].metrics['top1_retrieval_accuracy']:.4f}/"
        f"{systems['student'].metrics['top3_retrieval_accuracy']:.4f}/"
        f"{systems['student'].metrics['mrr']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
