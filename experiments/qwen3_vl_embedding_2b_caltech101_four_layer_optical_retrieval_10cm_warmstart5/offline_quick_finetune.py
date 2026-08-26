"""Warmstart5 entry point for the Qwen-free Language-global offline tail.

The strict data loader and optimizer are shared with the audited 10 cm
implementation. This thin 5% runner additionally preserves query-level
evaluation evidence immediately before fine-tuning and after restoring the
train-loss-selected tail. Neither gallery nor test metrics participate in
checkpoint selection.

Importing this module loads PyTorch only; it does not import Qwen,
Transformers, or the full optical simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    offline_quick_finetune as _implementation,
)


EXPECTED_CHECKPOINT_ARCHITECTURE = (
    "vision2_language2_moe4_10cm_warmstart5_stage_b_v1"
)
if _implementation.SUPPORTED_CHECKPOINT_ARCHITECTURES.get(
    EXPECTED_CHECKPOINT_ARCHITECTURE
) != 0.05:
    raise RuntimeError("Shared offline runner lacks the warmstart5 5% contract")

OfflineQuickData = _implementation.OfflineQuickData


def _require_warmstart5(data: OfflineQuickData) -> None:
    architecture = str(data.contract.get("checkpoint_architecture", ""))
    if architecture != EXPECTED_CHECKPOINT_ARCHITECTURE:
        raise RuntimeError(
            "Warmstart5 offline runner requires checkpoint architecture "
            f"{EXPECTED_CHECKPOINT_ARCHITECTURE!r}, got {architecture!r}"
        )


def load_offline_quick_data(
    session_dir: str | Path,
    *,
    ccd_dir: str | Path | None = None,
) -> OfflineQuickData:
    data = _implementation.load_offline_quick_data(session_dir, ccd_dir=ccd_dir)
    _require_warmstart5(data)
    return data


def load_offline_tail(
    data: OfflineQuickData, device: torch.device
) -> _implementation.LanguageGlobalOfflineTail:
    _require_warmstart5(data)
    return _implementation.load_offline_tail(data, device)


def _evaluate_with_predictions(
    tail: _implementation.LanguageGlobalOfflineTail,
    data: OfflineQuickData,
    device: torch.device,
    batch_size: int,
    *,
    system: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the fixed gallery/test split and retain one row per query."""

    gallery_indexes = data.indexes_for_split("gallery")
    test_indexes = data.indexes_for_split("test")
    gallery = F.normalize(
        _implementation._embeddings(
            tail, data, gallery_indexes, device, batch_size
        ).float(),
        dim=-1,
    )
    query = F.normalize(
        _implementation._embeddings(
            tail, data, test_indexes, device, batch_size
        ).float(),
        dim=-1,
    )
    gallery_labels = data.labels.index_select(0, gallery_indexes)
    truth = data.labels.index_select(0, test_indexes)
    class_names = [str(value) for value in data.contract["class_names"]]
    classes = torch.arange(len(class_names), dtype=torch.long)
    missing = [
        int(label) for label in classes if not gallery_labels.eq(label).any()
    ]
    if missing:
        raise RuntimeError(f"Offline gallery is missing class prototypes {missing}")
    prototypes = torch.stack(
        [
            F.normalize(gallery[gallery_labels.eq(label)].mean(dim=0), dim=0)
            for label in classes
        ]
    )
    scores = query @ prototypes.T
    ranking = scores.argsort(dim=1, descending=True)
    top1 = ranking[:, 0]
    top3 = ranking[:, : min(3, len(class_names))]
    correct1 = top1.eq(truth)
    correct3 = top3.eq(truth[:, None]).any(dim=1)
    ranks = torch.stack(
        [
            torch.nonzero(
                ranking[index].eq(truth[index]), as_tuple=False
            )[0, 0]
            + 1
            for index in range(len(truth))
        ]
    ).to(torch.long)

    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.long
    )
    for true, predicted in zip(truth, top1):
        confusion[int(true), int(predicted)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label, name in enumerate(class_names):
        selected = truth.eq(label)
        per_class[name] = {
            "query_count": int(selected.sum()),
            "top1_accuracy": float(correct1[selected].float().mean()),
            "top3_accuracy": float(correct3[selected].float().mean()),
        }

    prediction_rows: list[dict[str, Any]] = []
    for query_offset, data_index_tensor in enumerate(test_indexes):
        data_index = int(data_index_tensor)
        manifest = data.rows[data_index]
        key = data.keys[data_index]
        if str(manifest["key"]) != key:
            raise RuntimeError(
                "Offline manifest key drifted from its validated cache order"
            )
        true_index = int(truth[query_offset])
        predicted_index = int(top1[query_offset])
        wrong_scores = scores[query_offset].clone()
        wrong_scores[true_index] = -torch.inf
        true_score = float(scores[query_offset, true_index])
        best_wrong_score = float(wrong_scores.max())
        prediction_rows.append(
            {
                "system": system,
                "sample_id": str(manifest.get("sample_id") or key),
                "key": key,
                "split": str(manifest["split"]),
                "true_label": class_names[true_index],
                "true_label_index": true_index,
                "predicted_label": class_names[predicted_index],
                "predicted_label_index": predicted_index,
                "top1_correct": bool(correct1[query_offset]),
                "similarity_margin": true_score - best_wrong_score,
                "rank": int(ranks[query_offset]),
                "true_prototype_score": true_score,
                "best_wrong_prototype_score": best_wrong_score,
            }
        )

    metrics = {
        "system": system,
        "query_count": int(len(test_indexes)),
        "gallery_image_count": int(len(gallery_indexes)),
        "class_count": len(class_names),
        "gallery_aggregation": "mean_prototype",
        "top1_retrieval_accuracy": float(correct1.float().mean()),
        "top3_retrieval_accuracy": float(correct3.float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    return metrics, prediction_rows


def _result_directory(
    data: OfflineQuickData, output_dir: str | Path | None
) -> Path:
    return (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else data.stage_dir / "offline_results"
    )


def finetune_offline_quick(
    *,
    session_dir: str | Path,
    ccd_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str = "auto",
    epochs: int | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Run the shared optimizer and write paired pre/post test evidence."""

    data = load_offline_quick_data(session_dir, ccd_dir=ccd_dir)
    if validate_only:
        return _implementation.finetune_offline_quick(
            session_dir=session_dir,
            ccd_dir=ccd_dir,
            output_dir=output_dir,
            device_name=device_name,
            epochs=epochs,
            validate_only=True,
        )

    device = _implementation._select_device(device_name)
    training = data.contract["training_contract"]
    batch_size = int(training["pk_classes"]) * int(
        training["pk_images_per_class"]
    )
    initial_tail = load_offline_tail(data, device)
    pre_metrics, pre_rows = _evaluate_with_predictions(
        initial_tail,
        data,
        device,
        batch_size,
        system="quick210_0_pre_finetune",
    )

    # The shared implementation selects the tail strictly by training loss.
    report = _implementation.finetune_offline_quick(
        session_dir=session_dir,
        ccd_dir=ccd_dir,
        output_dir=output_dir,
        device_name=device_name,
        epochs=epochs,
        validate_only=False,
    )
    result_dir = _result_directory(data, output_dir)
    state_path = result_dir / "best_offline_tail_state.pt"
    best_state = torch.load(state_path, map_location="cpu", weights_only=True)
    initial_tail.load_state_dict(best_state, strict=True)
    post_metrics, post_rows = _evaluate_with_predictions(
        initial_tail,
        data,
        device,
        batch_size,
        system="quick210_1_post_finetune",
    )

    for field in ("top1_retrieval_accuracy", "top3_retrieval_accuracy", "mrr"):
        if abs(float(report[field]) - float(post_metrics[field])) > 1.0e-10:
            raise RuntimeError(
                f"Post-finetune evidence mismatch for {field}: "
                f"shared={report[field]}, evidence={post_metrics[field]}"
            )

    provenance = {
        "source_checkpoint_sha256": data.contract["source_checkpoint_sha256"],
        "contract_sha256": _implementation._sha256(data.contract_path),
        "ccd_set_sha256": report["ccd_set_sha256"],
        "test_selection_rule": "best checkpoint selected only by train loss",
        "gallery_or_test_used_for_epoch_selection": False,
    }
    pre_payload = {
        **pre_metrics,
        **provenance,
        "evaluation_point": "before_offline_finetune",
        "tail_state_source": "../offline_downstream/downstream_state.pt",
        "tail_state_sha256": data.contract["state_sha256"],
    }
    post_payload = {
        **report,
        **post_metrics,
        **provenance,
        "evaluation_point": "after_train_loss_selected_offline_finetune",
        "tail_state_source": "best_offline_tail_state.pt",
    }
    _implementation._write_json(
        result_dir / "pre_finetune_metrics.json", pre_payload
    )
    _implementation._write_json(
        result_dir / "post_finetune_metrics.json", post_payload
    )

    # Interleave systems so every test query has adjacent pre/post rows.
    post_by_key = {str(row["key"]): row for row in post_rows}
    paired_rows: list[dict[str, Any]] = []
    for pre_row in pre_rows:
        key = str(pre_row["key"])
        post_row = post_by_key.pop(key, None)
        if post_row is None:
            raise RuntimeError(f"Post-finetune prediction missing manifest key {key}")
        paired_rows.extend((pre_row, post_row))
    if post_by_key:
        raise RuntimeError(
            "Post-finetune predictions contain unmatched manifest keys "
            f"{sorted(post_by_key)}"
        )
    _implementation._write_csv(result_dir / "predictions.csv", paired_rows)

    report.update(
        {
            "evaluation_point": "after_train_loss_selected_offline_finetune",
            "pre_finetune_metrics": "pre_finetune_metrics.json",
            "post_finetune_metrics": "post_finetune_metrics.json",
            "query_predictions": "predictions.csv",
            "query_prediction_rows": len(paired_rows),
            "paired_test_queries": len(pre_rows),
            "gallery_or_test_used_for_epoch_selection": False,
        }
    )
    _implementation._write_json(result_dir / "metrics.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the warmstart5 Language-global tail without loading Qwen "
            "and preserve paired pre/post query evidence"
        )
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--ccd-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = finetune_offline_quick(
        session_dir=args.session_dir,
        ccd_dir=args.ccd_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        validate_only=args.validate_only,
    )
    if args.validate_only:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
