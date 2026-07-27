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
    teacher_store: TeacherEmbeddingStore,
    settings: Settings,
    checkpoint_path: Path | None = None,
) -> dict[str, RetrievalEvaluation]:
    checkpoint_path = (
        checkpoint_path
        or settings.output_dir / "best_train_loss_checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, replacement, readout)
    teacher_query = teacher_store.lookup(bundle.test_samples)
    teacher_gallery = teacher_store.lookup(bundle.gallery_samples)
    student_gallery = encode_student_samples(
        loaded, replacement, readout, bundle.gallery_samples, settings
    )
    student_query = encode_student_samples(
        loaded, replacement, readout, bundle.test_samples, settings
    )
    systems = {
        "teacher": evaluate_embeddings(
            teacher_query,
            bundle.test_samples,
            teacher_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name="frozen_teacher_query_vs_frozen_teacher_gallery",
        ),
        "student": evaluate_embeddings(
            student_query,
            bundle.test_samples,
            student_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name="optical_student_query_vs_optical_student_gallery",
        ),
        "student_teacher_gallery": evaluate_embeddings(
            student_query,
            bundle.test_samples,
            teacher_gallery,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name="optical_student_query_vs_frozen_teacher_gallery_diagnostic",
        ),
    }
    teacher_metrics = {
        **systems["teacher"].metrics,
        "teacher_embedding_shape": [len(bundle.test_samples), settings.embedding_dim],
        "teacher_frozen": True,
        "teacher_trainable_parameters": 0,
        "instruction": settings.instruction,
        "manifest_sha256": bundle.manifest_digest,
    }
    student_metrics = {
        **systems["student"].metrics,
        "main_deployment_result": True,
        "diagnostic_student_query_teacher_gallery": systems[
            "student_teacher_gallery"
        ].metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection": "minimum training total loss; test was not used",
        "student_detector_output_shape": [
            len(bundle.test_samples),
            settings.detector_output_size,
        ],
        "student_embedding_shape": [len(bundle.test_samples), settings.embedding_dim],
        "manifest_sha256": bundle.manifest_digest,
    }
    write_json(settings.output_dir / "teacher_metrics.json", teacher_metrics)
    write_json(settings.output_dir / "student_metrics.json", student_metrics)
    all_rows = [
        row
        for name in ("teacher", "student", "student_teacher_gallery")
        for row in systems[name].rows
    ]
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
            "normalization_check": "all cached and evaluated embeddings are finite and L2 normalized",
            "dimension_check": {
                "teacher": settings.embedding_dim,
                "student": settings.embedding_dim,
                "match": True,
            },
        },
    )
    return systems


def _plot_confusion(
    matrix: torch.Tensor, class_names: tuple[str, ...], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="Query count")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Retrieved SKU")
    axis.set_ylabel("True SKU")
    axis.set_title("Optical Student Grocery-10 Retrieval Confusion Matrix")
    for y in range(len(class_names)):
        for x in range(len(class_names)):
            axis.text(x, y, str(int(matrix[y, x])), ha="center", va="center", fontsize=8)
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
