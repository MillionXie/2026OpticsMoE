from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .io_utils import write_json
from .prepare_grocery_retrieval_subset import (
    GrocerySample,
    prepare_grocery_subset,
)
from .retrieval_metrics import evaluate_embeddings
from .settings import load_settings


def screen_replacements(
    all_sku_config: str | Path,
    target_config: str | Path,
    dropped_skus: Sequence[str],
    *,
    selection_source_split: str = "val",
    top_n: int = 20,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rank replacement combinations using frozen Teacher validation retrieval.

    The official test split is deliberately never consulted. This avoids
    selecting easier SKU replacements on the same samples later reported as
    final test results.
    """
    all_settings = load_settings(all_sku_config)
    target_settings = load_settings(target_config)
    bundle = prepare_grocery_subset(all_settings, persist=True)
    store = TeacherEmbeddingStore(all_settings.teacher_cache_path, bundle, all_settings)

    dropped = tuple(str(value) for value in dropped_skus)
    if not dropped:
        raise ValueError("At least one --drop-sku value is required")
    missing_drop = sorted(set(dropped) - set(target_settings.selected_skus))
    if missing_drop:
        raise ValueError(f"Dropped SKUs are absent from target config: {missing_drop}")
    retained = [
        name for name in target_settings.selected_skus if name not in set(dropped)
    ]
    candidates = [
        name
        for name in all_settings.selected_skus
        if name not in retained and name not in dropped
    ]
    if len(candidates) < len(dropped):
        raise RuntimeError("There are not enough candidate SKUs to replace the dropped set")

    validation_by_name: dict[str, list[GrocerySample]] = {}
    gallery_by_name: dict[str, list[GrocerySample]] = {}
    for name in all_settings.selected_skus:
        validation_by_name[name] = [
            sample
            for sample in bundle.train_samples
            if sample.sku_name == name
            and sample.source_split == selection_source_split
        ]
        gallery_by_name[name] = [
            sample for sample in bundle.gallery_samples if sample.sku_name == name
        ]
    missing_validation = [
        name
        for name in all_settings.selected_skus
        if not validation_by_name[name]
    ]
    if missing_validation:
        raise RuntimeError(
            f"Official source split {selection_source_split!r} has no samples for: "
            f"{missing_validation}. Test data will not be used as a fallback."
        )

    results: list[dict[str, Any]] = []
    for replacements in itertools.combinations(candidates, len(dropped)):
        class_names = tuple(retained) + tuple(replacements)
        remap = {name: index for index, name in enumerate(class_names)}
        queries = [
            replace(sample, sku_index=remap[name])
            for name in class_names
            for sample in validation_by_name[name]
        ]
        galleries = [
            replace(sample, sku_index=remap[name])
            for name in class_names
            for sample in gallery_by_name[name]
        ]
        result = evaluate_embeddings(
            store.lookup(queries),
            queries,
            store.lookup(galleries),
            galleries,
            class_names,
            all_settings.gallery_aggregation,
            system_name="teacher_validation_replacement_screen",
        )
        per_class = result.metrics["per_sku"]
        per_class_top1 = [
            float(per_class[name]["top1_accuracy"]) for name in class_names
        ]
        results.append(
            {
                "replacements": list(replacements),
                "selected_skus": list(class_names),
                "validation_query_count": len(queries),
                "validation_top1": result.metrics[
                    "top1_retrieval_accuracy"
                ],
                "validation_top3": result.metrics[
                    "top3_retrieval_accuracy"
                ],
                "validation_mrr": result.metrics["mrr"],
                "validation_macro_sku_top1": sum(per_class_top1)
                / len(per_class_top1),
                "validation_min_sku_top1": min(per_class_top1),
                "per_sku": per_class,
            }
        )
    results.sort(
        key=lambda item: (
            item["validation_macro_sku_top1"],
            item["validation_min_sku_top1"],
            item["validation_top1"],
            item["validation_mrr"],
        ),
        reverse=True,
    )
    payload: dict[str, Any] = {
        "selection_policy": (
            "maximize Teacher macro per-SKU Top-1, then minimum per-SKU Top-1, "
            "micro Top-1, and MRR"
        ),
        "selection_data": f"official {selection_source_split} source split only",
        "test_split_used_for_selection": False,
        "all_sku_config": str(Path(all_sku_config).resolve()),
        "target_config": str(Path(target_config).resolve()),
        "teacher_cache": str(all_settings.teacher_cache_path),
        "dropped_skus": list(dropped),
        "retained_skus": retained,
        "candidate_count": len(candidates),
        "combination_count": len(results),
        "recommended": results[0],
        "top_candidates": results[:top_n],
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else all_settings.output_dir
        / "metrics"
        / "replacement_screen_official_val.json"
    )
    write_json(destination, payload)
    print(
        f"Recommended replacements={results[0]['replacements']} "
        f"macro_top1={results[0]['validation_macro_sku_top1']:.4f} "
        f"min_sku_top1={results[0]['validation_min_sku_top1']:.4f} "
        f"report={destination}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-sku-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--drop-sku", action="append", required=True)
    parser.add_argument("--selection-source-split", default="val")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    screen_replacements(
        args.all_sku_config,
        args.target_config,
        args.drop_sku,
        selection_source_split=args.selection_source_split,
        top_n=args.top_n,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
