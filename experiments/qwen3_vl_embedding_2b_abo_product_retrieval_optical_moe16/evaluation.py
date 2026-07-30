from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .datasets import ABOBundle, ABORetrievalDataset, collate_abo
from .io_utils import write_json
from .modeling import (
    LoadedVisionBackbone,
    VisionOpticalRetrievalEncoder,
    preprocess_vision,
)
from .teacher_cache import TeacherEmbeddingStore
from .training import load_encoder_checkpoint


@torch.no_grad()
def encode_student_dataset(
    loaded: LoadedVisionBackbone,
    encoder: VisionOpticalRetrievalEncoder,
    dataset: ABORetrievalDataset,
    settings: Any,
) -> tuple[torch.Tensor, list[str], list[str], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_abo,
    )
    encoder.eval()
    values: list[torch.Tensor] = []
    image_ids: list[str] = []
    item_ids: list[str] = []
    paths: list[str] = []
    for batch in loader:
        inputs = preprocess_vision(
            loaded.processor, batch["images"], loaded.device
        )
        embedding = encoder(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        values.append(embedding.detach().cpu())
        image_ids.extend(batch["image_ids"])
        item_ids.extend(batch["item_ids"])
        paths.extend(batch["image_paths"])
    return torch.cat(values), image_ids, item_ids, paths


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    query_item_ids: Sequence[str],
    gallery_embeddings: torch.Tensor,
    gallery_item_ids: Sequence[str],
    *,
    aggregation: str,
    query_image_ids: Sequence[str] | None = None,
    query_paths: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(query_embeddings) != len(query_item_ids):
        raise ValueError("Query embeddings/item IDs differ in length")
    if len(gallery_embeddings) != len(gallery_item_ids):
        raise ValueError("Gallery embeddings/item IDs differ in length")
    query_embeddings = F.normalize(query_embeddings.float(), p=2, dim=-1)
    gallery_embeddings = F.normalize(gallery_embeddings.float(), p=2, dim=-1)
    unique_items = sorted(set(gallery_item_ids))
    if set(query_item_ids) != set(unique_items):
        raise RuntimeError(
            "Query and Gallery item identity sets differ; retrieval protocol is invalid"
        )
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item_id in enumerate(gallery_item_ids):
        groups[str(item_id)].append(index)
    if aggregation == "mean_prototype":
        candidates = torch.stack(
            [
                F.normalize(
                    gallery_embeddings[groups[item_id]].mean(dim=0),
                    p=2,
                    dim=-1,
                )
                for item_id in unique_items
            ]
        )
        similarities = query_embeddings @ candidates.T
    elif aggregation == "max_similarity":
        columns = [
            (query_embeddings @ gallery_embeddings[groups[item_id]].T).amax(
                dim=1
            )
            for item_id in unique_items
        ]
        similarities = torch.stack(columns, dim=1)
    else:
        raise ValueError(f"Unknown gallery aggregation {aggregation!r}")
    target_index = {item_id: index for index, item_id in enumerate(unique_items)}
    ranked = similarities.argsort(dim=1, descending=True)
    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    for query_index, true_item in enumerate(query_item_ids):
        target = target_index[str(true_item)]
        position = int(
            torch.nonzero(ranked[query_index].eq(target), as_tuple=False)[0]
        )
        rank = position + 1
        ranks.append(rank)
        top_indexes = ranked[query_index, : min(10, len(unique_items))].tolist()
        top_items = [unique_items[index] for index in top_indexes]
        top_scores = [
            float(similarities[query_index, index]) for index in top_indexes
        ]
        rows.append(
            {
                "query_image_id": (
                    query_image_ids[query_index]
                    if query_image_ids is not None
                    else query_index
                ),
                "query_image_path": (
                    query_paths[query_index] if query_paths is not None else ""
                ),
                "true_item_id": true_item,
                "rank": rank,
                "correct_top1": int(rank == 1),
                "top10_item_ids": "|".join(top_items),
                "top10_similarities": "|".join(
                    f"{score:.8f}" for score in top_scores
                ),
                "correct_item_similarity": float(
                    similarities[query_index, target]
                ),
                "max_wrong_similarity": float(
                    similarities[query_index]
                    .masked_fill(
                        torch.arange(
                            len(unique_items), device=similarities.device
                        ).eq(target),
                        -torch.inf,
                    )
                    .max()
                ),
            }
        )
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    per_item: dict[str, list[int]] = defaultdict(list)
    for item_id, rank in zip(query_item_ids, ranks):
        per_item[str(item_id)].append(rank)
    metrics = {
        "query_count": len(ranks),
        "gallery_image_count": len(gallery_item_ids),
        "gallery_item_count": len(unique_items),
        "top1_accuracy": float((ranks_tensor <= 1).float().mean()),
        "recall_at_5": float((ranks_tensor <= 5).float().mean()),
        "recall_at_10": float((ranks_tensor <= 10).float().mean()),
        # Item-level gallery aggregation creates one relevant ranked item per
        # query, so AP=1/rank and mAP is numerically equal to MRR.
        "mean_average_precision": float((1.0 / ranks_tensor).mean()),
        "mrr": float((1.0 / ranks_tensor).mean()),
        "median_rank": float(ranks_tensor.median()),
        "gallery_aggregation": aggregation,
        "per_item": {
            item_id: {
                "queries": len(values),
                "top1_accuracy": sum(rank == 1 for rank in values) / len(values),
                "recall_at_5": sum(rank <= 5 for rank in values) / len(values),
                "mean_reciprocal_rank": sum(1.0 / rank for rank in values)
                / len(values),
            }
            for item_id, values in sorted(per_item.items())
        },
    }
    return metrics, rows


@torch.no_grad()
def evaluate_all(
    loaded: LoadedVisionBackbone,
    encoder: VisionOpticalRetrievalEncoder,
    bundle: ABOBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint or (
        settings.output_dir / "checkpoints" / "deployment_encoder.pt"
    )
    load_encoder_checkpoint(
        checkpoint,
        encoder,
        expected_manifest_digest=bundle.manifest_digest,
    )
    (
        student_gallery,
        gallery_image_ids,
        gallery_item_ids,
        gallery_paths,
    ) = encode_student_dataset(loaded, encoder, bundle.gallery, settings)
    (
        student_query,
        query_image_ids,
        query_item_ids,
        query_paths,
    ) = encode_student_dataset(loaded, encoder, bundle.query, settings)
    teacher_gallery = teacher_store.get(gallery_image_ids, torch.device("cpu"))
    teacher_query = teacher_store.get(query_image_ids, torch.device("cpu"))

    systems = {
        "teacher": (teacher_query, teacher_gallery),
        "student": (student_query, student_gallery),
        "student_query_teacher_gallery": (student_query, teacher_gallery),
    }
    result: dict[str, Any] = {
        "manifest_sha256": bundle.manifest_digest,
        "checkpoint": str(checkpoint),
        "test_used_for_checkpoint_selection": False,
        "systems": {},
    }
    for name, (query_values, gallery_values) in systems.items():
        metrics, rows = evaluate_retrieval(
            query_values,
            query_item_ids,
            gallery_values,
            gallery_item_ids,
            aggregation=settings.gallery_aggregation,
            query_image_ids=query_image_ids,
            query_paths=query_paths,
        )
        result["systems"][name] = metrics
        path = settings.output_dir / "metrics" / f"{name}_retrieval_results.csv"
        _write_csv(path, rows)
        if name == "student":
            _write_csv(settings.output_dir / "retrieval_results.csv", rows)
            _write_per_item(
                settings.output_dir / "per_item_metrics.csv", metrics["per_item"]
            )
    teacher = result["systems"]["teacher"]
    student = result["systems"]["student"]
    aligned = result["systems"]["student_query_teacher_gallery"]
    gallery_items = int(student["gallery_item_count"])
    result["comparison"] = {
        "gallery_item_count": gallery_items,
        "random_top1_accuracy": 1.0 / gallery_items,
        "random_recall_at_5": min(1.0, 5.0 / gallery_items),
        "random_recall_at_10": min(1.0, 10.0 / gallery_items),
        "student_teacher_retention": {
            "top1": _safe_ratio(
                student["top1_accuracy"], teacher["top1_accuracy"]
            ),
            "recall_at_5": _safe_ratio(
                student["recall_at_5"], teacher["recall_at_5"]
            ),
            "recall_at_10": _safe_ratio(
                student["recall_at_10"], teacher["recall_at_10"]
            ),
            "mean_average_precision": _safe_ratio(
                student["mean_average_precision"],
                teacher["mean_average_precision"],
            ),
        },
        "student_query_teacher_gallery": {
            "top1_accuracy": aligned["top1_accuracy"],
            "recall_at_5": aligned["recall_at_5"],
            "recall_at_10": aligned["recall_at_10"],
            "mean_average_precision": aligned["mean_average_precision"],
            "diagnostic": (
                "If this is much higher than student-vs-student, query embeddings "
                "are partly aligned but the student gallery/prototype geometry is "
                "unstable. If it is similarly low, the main bottleneck is teacher-"
                "space alignment of the optical encoder."
            ),
        },
    }
    write_json(settings.output_dir / "retrieval_metrics.json", result)
    write_json(
        settings.output_dir / "teacher_metrics.json",
        result["systems"]["teacher"],
    )
    write_json(
        settings.output_dir / "student_metrics.json",
        result["systems"]["student"],
    )
    return result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if float(denominator) == 0.0 else float(numerator) / float(denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty retrieval result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_per_item(path: Path, metrics: dict[str, Any]) -> None:
    rows = [{"item_id": item_id, **values} for item_id, values in metrics.items()]
    _write_csv(path, rows)
