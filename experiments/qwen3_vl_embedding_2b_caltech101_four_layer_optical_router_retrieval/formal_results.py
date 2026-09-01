from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "formal_results"
DEFAULT_MARKDOWN = PROJECT_ROOT / "FORMAL_RESULTS.md"
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_RESAMPLES = 10_000

METRICS = (
    "top1_retrieval_accuracy",
    "top3_retrieval_accuracy",
    "mrr",
)
METRIC_LABELS = {
    "top1_retrieval_accuracy": "Top-1",
    "top3_retrieval_accuracy": "Top-3",
    "mrr": "MRR",
}


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    directory: str
    router: str
    top_k: int
    formal: bool = True


RUN_SPECS = (
    RunSpec("legacy", 42, "electronic_legacy_topk2_anchor", "electronic", 2, False),
    RunSpec("E1", 42, "electronic_power_topk1", "electronic", 1),
    RunSpec("E1", 43, "electronic_power_topk1_seed43", "electronic", 1),
    RunSpec("E1", 44, "electronic_power_topk1_seed44", "electronic", 1),
    RunSpec("E2", 42, "electronic_power_topk2", "electronic", 2),
    RunSpec("E2", 43, "electronic_power_topk2_seed43", "electronic", 2),
    RunSpec("E2", 44, "electronic_power_topk2_seed44", "electronic", 2),
    RunSpec("E4", 42, "electronic_power_topk4", "electronic", 4),
    RunSpec("E4", 43, "electronic_power_topk4_seed43", "electronic", 4),
    RunSpec("E4", 44, "electronic_power_topk4_seed44", "electronic", 4),
    RunSpec("O2", 42, "optical_power_topk2", "optical", 2),
    RunSpec("O2", 43, "optical_power_topk2_seed43", "optical", 2),
    RunSpec("O2", 44, "optical_power_topk2_seed44", "optical", 2),
)

# The first three are the pre-registered matched ablations.  Comparisons to the
# historical anchor are also useful, but the anchor is one fixed run and is not
# a three-seed estimate.
MCNEMAR_PAIRS = (
    ("E1", "E2", "electronic_k1_vs_k2"),
    ("E4", "E2", "electronic_k4_vs_k2"),
    ("O2", "E2", "optical_vs_electronic_top2"),
    ("E1", "legacy", "E1_vs_legacy_anchor"),
    ("E2", "legacy", "E2_vs_legacy_anchor"),
    ("E4", "legacy", "E4_vs_legacy_anchor"),
    ("O2", "legacy", "O2_vs_legacy_anchor"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value {value!r}")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int, key: str
) -> tuple[float, float]:
    if not values:
        raise ValueError("Bootstrap requires at least one value")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    key_seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"
    )
    generator = random.Random(seed ^ key_seed)
    count = len(values)
    estimates = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _summary(
    values: Sequence[float], *, samples: int, seed: int, key: str
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "bootstrap_95_ci": [None, None]}
    low, high = _bootstrap_mean_ci(values, samples=samples, seed=seed, key=key)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "bootstrap_95_ci": [low, high],
        "values": list(values),
    }


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    # mmap avoids eagerly copying checkpoint tensor storage while we inspect the
    # small Python metadata object.  The fallback supports older PyTorch builds.
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return value


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = _checkpoint_payload(path)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "checkpoint_version": payload.get("checkpoint_version"),
        "epoch": payload.get("epoch"),
        "train_loss": payload.get("train_loss"),
        "weight_variant": metadata.get("weight_variant"),
        "selection_criterion": metadata.get("selection_criterion"),
        "test_metrics_used_for_selection": metadata.get(
            "test_metrics_used_for_selection"
        ),
        "optical_architecture": metadata.get("optical_architecture"),
        "model_id": metadata.get("model_id"),
        "embedding_dim": metadata.get("embedding_dim"),
        "detector_dim": metadata.get("detector_dim"),
        "learning_rate": metadata.get("learning_rate"),
        "router_learning_rate": metadata.get("router_learning_rate"),
        "phase_learning_rate": metadata.get("phase_learning_rate"),
        "ema_decay": metadata.get("ema_decay"),
        "training_objective": metadata.get("training_objective"),
    }


def _training_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("train_log.csv has no rows")
    parsed: list[tuple[int, float, Mapping[str, Any]]] = []
    for row in rows:
        epoch = int(row["epoch"])
        loss = _finite_float(row.get("total_loss"))
        if loss is None:
            raise ValueError(f"Non-finite total_loss at epoch {epoch}")
        parsed.append((epoch, loss, row))
    selected_epoch, minimum_loss, _ = min(parsed, key=lambda item: (item[1], item[0]))
    test_columns = (
        "test_top1",
        "test_top3",
        "test_mrr",
        "ema_test_top1",
        "ema_test_top3",
        "ema_test_mrr",
    )
    finite_test_values = [
        {"epoch": int(row["epoch"]), "column": column, "value": float(value)}
        for row in rows
        for column in test_columns
        if (value := _finite_float(row.get(column))) is not None
    ]
    last = parsed[-1][2]
    return {
        "epochs_logged": len(rows),
        "first_epoch": min(epoch for epoch, _, _ in parsed),
        "last_epoch": max(epoch for epoch, _, _ in parsed),
        "minimum_training_loss_epoch": selected_epoch,
        "minimum_training_total_loss": minimum_loss,
        "last_training_total_loss": parsed[-1][1],
        "last_train_top1": _finite_float(last.get("train_top1")),
        "total_epoch_time_sec": sum(
            _finite_float(row.get("epoch_time_sec")) or 0.0 for row in rows
        ),
        "finite_per_epoch_test_metric_count": len(finite_test_values),
        "finite_per_epoch_test_metrics": finite_test_values,
        "test_used_during_training": bool(finite_test_values),
    }


def _student_rows(path: Path, system_name: str | None) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    if system_name:
        matching = [row for row in rows if row.get("system") == system_name]
        if matching:
            rows = matching
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        result[sample_id] = {
            "correct": _as_bool(row.get("top1_correct")),
            "top3_correct": (
                _as_bool(row["top3_correct"])
                if row.get("top3_correct") not in (None, "")
                else None
            ),
            "reciprocal_rank": _finite_float(row.get("reciprocal_rank")),
            "prediction": row.get("predicted_sku", row.get("predicted_sku_index")),
            "truth": row.get("true_sku", row.get("true_sku_index")),
        }
    if not result:
        raise ValueError(f"No student sample predictions found in {path}")
    return result


def _inspect_run(spec: RunSpec, runs_dir: Path) -> dict[str, Any]:
    run_dir = runs_dir / spec.directory
    checkpoint_name = (
        "converted_warmstart5_initialization_checkpoint.pt"
        if spec.variant == "legacy"
        else "ema_best_train_loss_checkpoint.pt"
    )
    paths = {
        "run_dir": run_dir,
        "metrics": run_dir / "student_metrics.json",
        "train_log": run_dir / "train_log.csv",
        "checkpoint": run_dir / checkpoint_name,
        "predictions": run_dir / "retrieval_results.csv",
    }
    required = ["metrics", "checkpoint"] if spec.variant == "legacy" else [
        "metrics",
        "train_log",
        "checkpoint",
    ]
    missing = [name for name in required if not paths[name].is_file()]
    record: dict[str, Any] = {
        "variant": spec.variant,
        "seed": spec.seed,
        "directory": spec.directory,
        "router": spec.router,
        "top_k": spec.top_k,
        "formal_repeated_seed": spec.formal,
        "status": "pending" if missing else "complete",
        "missing": missing,
        "paths": {key: str(value) for key, value in paths.items()},
        "predictions_available": paths["predictions"].is_file(),
        "errors": [],
    }
    if missing:
        return record
    try:
        metrics = _read_json(paths["metrics"])
        for metric in METRICS:
            value = _finite_float(metrics.get(metric))
            if value is None:
                raise ValueError(f"student_metrics.json is missing finite {metric}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric}={value} is outside [0,1]")
        record["metrics"] = {metric: float(metrics[metric]) for metric in METRICS}
        record["query_count"] = int(metrics["query_count"])
        record["gallery_image_count"] = int(metrics["gallery_image_count"])
        record["class_count"] = int(metrics.get("sku_count", len(metrics.get("per_sku", {}))))
        record["manifest_sha256"] = metrics.get("manifest_sha256")
        if (
            not isinstance(record["manifest_sha256"], str)
            or len(record["manifest_sha256"]) != 64
        ):
            raise ValueError("student_metrics has no valid manifest_sha256")
        record["metric_system"] = metrics.get("system")
        per_class = metrics.get("per_sku", {})
        if not isinstance(per_class, dict) or len(per_class) != record["class_count"]:
            raise ValueError(
                "student_metrics per_sku does not contain exactly class_count entries"
            )
        for class_name, class_metrics in per_class.items():
            if not isinstance(class_metrics, dict):
                raise ValueError(f"per_sku[{class_name!r}] is not a mapping")
            if int(class_metrics.get("query_count", 0)) <= 0:
                raise ValueError(f"per_sku[{class_name!r}] has no queries")
            for metric in ("top1_accuracy", "top3_accuracy"):
                value = _finite_float(class_metrics.get(metric))
                if value is None or not 0.0 <= value <= 1.0:
                    raise ValueError(f"per_sku[{class_name!r}].{metric} is invalid")
        per_class_query_count = sum(
            int(class_metrics["query_count"]) for class_metrics in per_class.values()
        )
        if per_class_query_count != record["query_count"]:
            raise ValueError("Sum of per_sku query counts disagrees with query_count")
        for class_metric, overall_metric in (
            ("top1_accuracy", "top1_retrieval_accuracy"),
            ("top3_accuracy", "top3_retrieval_accuracy"),
        ):
            weighted = sum(
                int(class_metrics["query_count"])
                * float(class_metrics[class_metric])
                for class_metrics in per_class.values()
            ) / per_class_query_count
            if not math.isclose(
                weighted,
                record["metrics"][overall_metric],
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ):
                raise ValueError(
                    f"Weighted per_sku {class_metric} disagrees with {overall_metric}"
                )
        record["per_class"] = per_class
        record["evaluation_checkpoint"] = metrics.get("checkpoint")
        record["evaluation_checkpoint_epoch"] = metrics.get("checkpoint_epoch")
        record["evaluation_selection_biased"] = metrics.get("selection_biased")
        record["checkpoint"] = _checkpoint_metadata(paths["checkpoint"])
        if int(record["evaluation_checkpoint_epoch"]) != int(
            record["checkpoint"]["epoch"]
        ):
            raise ValueError(
                "student_metrics checkpoint_epoch disagrees with checkpoint metadata"
            )
        if record["evaluation_selection_biased"] is not False:
            raise ValueError("student_metrics marks evaluation as selection-biased")
        evaluated_name = Path(str(record["evaluation_checkpoint"])).name
        if evaluated_name != checkpoint_name:
            raise ValueError(
                f"student_metrics evaluated {evaluated_name!r}, expected {checkpoint_name!r}"
            )
        if record["checkpoint"].get("weight_variant") != "ema":
            raise ValueError("Evaluated checkpoint metadata is not weight_variant=ema")
        if record["checkpoint"].get("test_metrics_used_for_selection") is not False:
            raise ValueError(
                "Checkpoint does not explicitly state test_metrics_used_for_selection=false"
            )
        if spec.variant != "legacy":
            training = _training_summary(_read_csv(paths["train_log"]))
            record["training"] = training
            checkpoint_epoch = int(record["checkpoint"]["epoch"])
            if checkpoint_epoch != training["minimum_training_loss_epoch"]:
                raise ValueError(
                    "EMA checkpoint epoch does not match the minimum training-loss "
                    f"epoch ({checkpoint_epoch} vs {training['minimum_training_loss_epoch']})"
                )
            if not math.isclose(
                float(record["checkpoint"]["train_loss"]),
                float(training["minimum_training_total_loss"]),
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "EMA checkpoint train_loss does not match train_log minimum"
                )
            if training["test_used_during_training"]:
                raise ValueError("Finite test metrics occur in train_log.csv")
            if record["checkpoint"].get("selection_criterion") != "minimum_training_total_loss":
                raise ValueError("Checkpoint was not selected by minimum_training_total_loss")
        if paths["predictions"].is_file():
            record["predictions"] = _student_rows(
                paths["predictions"], record.get("metric_system")
            )
            predictions = list(record["predictions"].values())
            if len(predictions) != record["query_count"]:
                raise ValueError(
                    "retrieval_results query count disagrees with student_metrics"
                )
            observed_top1 = statistics.fmean(
                float(row["correct"]) for row in predictions
            )
            if not math.isclose(
                observed_top1,
                record["metrics"]["top1_retrieval_accuracy"],
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    "retrieval_results Top-1 disagrees with student_metrics"
                )
            if all(row["top3_correct"] is not None for row in predictions):
                observed_top3 = statistics.fmean(
                    float(row["top3_correct"]) for row in predictions
                )
                if not math.isclose(
                    observed_top3,
                    record["metrics"]["top3_retrieval_accuracy"],
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                ):
                    raise ValueError(
                        "retrieval_results Top-3 disagrees with student_metrics"
                    )
            if all(row["reciprocal_rank"] is not None for row in predictions):
                observed_mrr = statistics.fmean(
                    float(row["reciprocal_rank"]) for row in predictions
                )
                if not math.isclose(
                    observed_mrr,
                    record["metrics"]["mrr"],
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                ):
                    raise ValueError("retrieval_results MRR disagrees with student_metrics")
    except Exception as exc:  # Preserve other completed runs in a partial report.
        record["status"] = "error"
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def _group_completed(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record["status"] == "complete":
            grouped.setdefault(str(record["variant"]), []).append(record)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["seed"]))
    return grouped


def _cross_run_contract_errors(records: Sequence[Mapping[str, Any]]) -> list[str]:
    completed = [record for record in records if record["status"] == "complete"]
    if len(completed) <= 1:
        return []
    errors: list[str] = []
    for field in ("manifest_sha256", "query_count", "gallery_image_count", "class_count"):
        values = {str(record.get(field)) for record in completed}
        if len(values) != 1:
            errors.append(f"Completed runs disagree on {field}: {sorted(values)}")
    class_sets = {tuple(sorted(record.get("per_class", {}))) for record in completed}
    if len(class_sets) != 1:
        errors.append("Completed runs disagree on the evaluated class names")
    return errors


def _aggregate_variants(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    anchor = grouped.get("legacy", [])
    anchor_metrics = anchor[0]["metrics"] if len(anchor) == 1 else None
    output: dict[str, Any] = {}
    for variant in ("E1", "E2", "E4", "O2"):
        runs = list(grouped.get(variant, []))
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [float(run["metrics"][metric]) for run in runs]
            item = _summary(
                values,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"{variant}:{metric}",
            )
            anchor_value = (
                float(anchor_metrics[metric]) if anchor_metrics is not None else None
            )
            item["legacy_anchor"] = anchor_value
            item["absolute_delta_vs_legacy"] = (
                item["mean"] - anchor_value
                if item["mean"] is not None and anchor_value is not None
                else None
            )
            item["relative_delta_vs_legacy_percent"] = (
                100.0 * (item["mean"] - anchor_value) / anchor_value
                if item["mean"] is not None and anchor_value not in (None, 0.0)
                else None
            )
            if item["n"]:
                delta_values = [value - anchor_value for value in values] if anchor_value is not None else []
                item["absolute_delta_vs_legacy_bootstrap_95_ci"] = (
                    list(
                        _bootstrap_mean_ci(
                            delta_values,
                            samples=bootstrap_samples,
                            seed=bootstrap_seed,
                            key=f"{variant}:{metric}:legacy_delta",
                        )
                    )
                    if delta_values
                    else [None, None]
                )
            metrics[metric] = item
        output[variant] = {
            "status": "complete" if len(runs) == 3 else "pending",
            "completed_seeds": [int(run["seed"]) for run in runs],
            "expected_seeds": [42, 43, 44],
            "metrics": metrics,
        }
    return output


def _aggregate_per_class(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for variant in ("E1", "E2", "E4", "O2"):
        runs = list(grouped.get(variant, []))
        if not runs:
            continue
        class_sets = [set(run.get("per_class", {})) for run in runs]
        common = set.intersection(*class_sets) if class_sets else set()
        nested[variant] = {}
        for class_name in sorted(common):
            nested[variant][class_name] = {}
            for metric in ("top1_accuracy", "top3_accuracy"):
                values = [
                    float(run["per_class"][class_name][metric]) for run in runs
                ]
                summary = _summary(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                    key=f"per_class:{variant}:{class_name}:{metric}",
                )
                nested[variant][class_name][metric] = summary
                rows.append(
                    {
                        "variant": variant,
                        "class": class_name,
                        "metric": metric,
                        "n_seeds": summary["n"],
                        "mean": summary["mean"],
                        "sample_std": summary["sample_std"],
                        "bootstrap_95_ci_low": summary["bootstrap_95_ci"][0],
                        "bootstrap_95_ci_high": summary["bootstrap_95_ci"][1],
                    }
                )
            counts = [int(run["per_class"][class_name]["query_count"]) for run in runs]
            nested[variant][class_name]["query_count_per_seed"] = counts
    return rows, nested


def _exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _mcnemar(
    left: Mapping[str, Any], right: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    left_predictions = left["predictions"]
    right_predictions = right["predictions"]
    common = sorted(set(left_predictions) & set(right_predictions))
    if not common:
        raise ValueError("No common sample_id values")
    if set(left_predictions) != set(right_predictions):
        raise ValueError("Prediction files do not contain identical sample_id sets")
    for sample_id in common:
        if left_predictions[sample_id]["truth"] != right_predictions[sample_id]["truth"]:
            raise ValueError(f"Truth label mismatch for sample_id={sample_id}")
    both_correct = sum(
        left_predictions[key]["correct"] and right_predictions[key]["correct"]
        for key in common
    )
    left_only = sum(
        left_predictions[key]["correct"] and not right_predictions[key]["correct"]
        for key in common
    )
    right_only = sum(
        not left_predictions[key]["correct"] and right_predictions[key]["correct"]
        for key in common
    )
    both_wrong = len(common) - both_correct - left_only - right_only
    prediction_disagreements = sum(
        left_predictions[key]["prediction"] != right_predictions[key]["prediction"]
        for key in common
    )
    return {
        "family": family,
        "left_variant": left["variant"],
        "right_variant": right["variant"],
        "seed": int(left["seed"]),
        "right_seed": int(right["seed"]),
        "query_count": len(common),
        "both_correct": both_correct,
        "left_only_correct_b": left_only,
        "right_only_correct_c": right_only,
        "both_wrong": both_wrong,
        "left_minus_right_top1": (left_only - right_only) / len(common),
        "prediction_disagreement_count": prediction_disagreements,
        "prediction_disagreement_fraction": prediction_disagreements / len(common),
        "discordant_count": left_only + right_only,
        "odds_ratio_haldane": (left_only + 0.5) / (right_only + 0.5),
        "p_exact_two_sided": _exact_mcnemar_p(left_only, right_only),
        "p_holm_all_reported": None,
        "method": "exact paired McNemar/binomial test on query Top-1 correctness",
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_exact_two_sided"])
    running = 0.0
    count = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row["p_exact_two_sided"]))
        running = max(running, adjusted)
        rows[index]["p_holm_all_reported"] = running


def _paired_tests(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    by_variant_seed = {
        (str(run["variant"]), int(run["seed"])): run
        for runs in grouped.values()
        for run in runs
        if "predictions" in run
    }
    rows: list[dict[str, Any]] = []
    for left_variant, right_variant, family in MCNEMAR_PAIRS:
        for seed in (42, 43, 44):
            left = by_variant_seed.get((left_variant, seed))
            right_seed = 42 if right_variant == "legacy" else seed
            right = by_variant_seed.get((right_variant, right_seed))
            if left is None or right is None:
                continue
            rows.append(_mcnemar(left, right, family=family))
    _holm_adjust(rows)
    return rows


def _format_mean_std(item: Mapping[str, Any]) -> str:
    if item.get("mean") is None:
        return "pending"
    mean = 100.0 * float(item["mean"])
    std = item.get("sample_std")
    low, high = item["bootstrap_95_ci"]
    std_text = "n/a" if std is None else f"{100.0 * float(std):.2f}"
    return f"{mean:.2f} ± {std_text} [{100*low:.2f}, {100*high:.2f}]"


def _markdown(report: Mapping[str, Any]) -> str:
    complete = report["status"] == "complete"
    lines = [
        "# Formal Router Results",
        "",
        f"> Status: **{'COMPLETE' if complete else 'PENDING'}**. "
        + (
            "All pre-registered E1/E2/E4/O2 seeds and the legacy anchor passed artifact checks."
            if complete
            else "Missing or invalid runs are listed below; partial numbers are not formal conclusions."
        ),
        "",
        "## Reporting contract",
        "",
        "- E1/E2/E4/O2 each use optimization seeds 42, 43, and 44; the Caltech split and PK batch order remain fixed.",
        "- Each test evaluation is performed on the pre-selected `ema_best_train_loss_checkpoint.pt`; the checkpoint is selected only by minimum training total loss.",
        "- The training log must contain no finite per-epoch test metric. The held-out test is evaluated once after selection under this run protocol.",
        "- The historical Caltech test has been examined in earlier project work, so it is **not globally unseen**. The defensible statement is: this adaptation run did not use test results for epoch selection.",
        f"- Mean ± sample SD is across three optimization seeds. Brackets are a 95% percentile bootstrap CI of the seed mean ({report['protocol']['bootstrap_resamples']:,} resamples; fixed seed {report['protocol']['bootstrap_seed']}). With only three seeds, the CI is descriptive and coarse.",
        "",
        "## Aggregate retrieval results",
        "",
        "All values below are percentage points. `Δ legacy` compares the three-seed mean with the single fixed warmstart5 anchor; it is not a paired three-seed estimate.",
        "",
        "| Variant | Router | k | Seeds | Top-1 mean ± SD [95% CI] | Top-3 mean ± SD [95% CI] | MRR mean ± SD [95% CI] | Top-1 Δ legacy |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    variant_meta = {
        "E1": ("electronic", 1),
        "E2": ("electronic", 2),
        "E4": ("electronic", 4),
        "O2": ("optical", 2),
    }
    for variant, (router, top_k) in variant_meta.items():
        item = report["variants"][variant]
        metrics = item["metrics"]
        delta = metrics["top1_retrieval_accuracy"].get("absolute_delta_vs_legacy")
        lines.append(
            f"| {variant} | {router} | {top_k} | "
            f"{','.join(str(seed) for seed in item['completed_seeds']) or 'pending'} | "
            f"{_format_mean_std(metrics['top1_retrieval_accuracy'])} | "
            f"{_format_mean_std(metrics['top3_retrieval_accuracy'])} | "
            f"{_format_mean_std(metrics['mrr'])} | "
            f"{'pending' if delta is None else f'{100*delta:+.2f}'} |"
        )
    anchor = report.get("legacy_anchor")
    lines.extend(["", "## Legacy anchor", ""])
    if anchor:
        lines.append(
            "Fixed warmstart5 anchor (one run, no optimizer adaptation): "
            f"Top-1 {100*anchor['metrics']['top1_retrieval_accuracy']:.2f}%, "
            f"Top-3 {100*anchor['metrics']['top3_retrieval_accuracy']:.2f}%, "
            f"MRR {100*anchor['metrics']['mrr']:.2f}%."
        )
    else:
        lines.append("Pending: legacy anchor metrics are not available.")
    lines.extend([
        "",
        "## Query-level paired tests",
        "",
        "McNemar tests are computed separately for each paired optimization seed on identical `sample_id` values. Anchor comparisons reuse the same fixed anchor predictions and are labeled accordingly. No rows are pooled across seeds, avoiding pseudo-replication. `p_Holm` adjusts across all tests emitted in this report.",
        "",
        "| Comparison | Seed | n | left-only correct | right-only correct | Δ Top-1 | exact p | p_Holm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if report["paired_mcnemar"]:
        for row in report["paired_mcnemar"]:
            lines.append(
                f"| {row['left_variant']} vs {row['right_variant']} | {row['seed']} | "
                f"{row['query_count']} | {row['left_only_correct_b']} | "
                f"{row['right_only_correct_c']} | "
                f"{100*row['left_minus_right_top1']:+.2f} | "
                f"{row['p_exact_two_sided']:.4g} | {row['p_holm_all_reported']:.4g} |"
            )
    else:
        lines.append("| pending | — | — | — | — | — | — | — |")
    if report.get("cross_run_contract_errors"):
        lines.extend(["", "### Cross-run contract errors", ""])
        lines.extend(
            f"- {message}" for message in report["cross_run_contract_errors"]
        )
    lines.extend(["", "## Run inventory", ""])
    lines.extend([
        "| Run | Variant | Seed | Status | Selected epoch | Missing / error |",
        "|---|---|---:|---|---:|---|",
    ])
    for run in report["runs"]:
        selected = run.get("checkpoint", {}).get("epoch", "—")
        issue = ", ".join(run.get("missing", []) + run.get("errors", [])) or "—"
        lines.append(
            f"| `{run['directory']}` | {run['variant']} | {run['seed']} | "
            f"{run['status']} | {selected} | {issue} |"
        )
    lines.extend([
        "",
        "## Machine-readable handoff",
        "",
        "- `formal_results/aggregate_results.json`: complete provenance, run audit, aggregate metrics, per-class metrics, and paired tests.",
        "- `formal_results/aggregate_metrics.csv`: one row per variant and metric.",
        "- `formal_results/per_class_metrics.csv`: class-resolved mean, sample SD, and bootstrap CI.",
        "- `formal_results/paired_mcnemar.csv`: per-seed exact paired tests.",
        "- `formal_results/run_inventory.csv`: checkpoint and training-selection audit.",
        "",
        "Regenerate from the repository root:",
        "",
        "```text",
        "python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.formal_results",
        "```",
        "",
    ])
    return "\n".join(lines)


def generate_report(
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    markdown_path: Path = DEFAULT_MARKDOWN,
    bootstrap_samples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    records = [_inspect_run(spec, runs_dir) for spec in RUN_SPECS]
    cross_run_errors = _cross_run_contract_errors(records)
    grouped = _group_completed(records)
    variants = _aggregate_variants(
        grouped,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    per_class_rows, per_class = _aggregate_per_class(
        grouped,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired = _paired_tests(grouped)
    legacy = grouped.get("legacy", [])
    all_formal_complete = all(
        variants[variant]["status"] == "complete" for variant in variants
    )
    no_errors = (
        not any(record["status"] == "error" for record in records)
        and not cross_run_errors
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete" if all_formal_complete and len(legacy) == 1 and no_errors else "pending"
        ),
        "protocol": {
            "formal_variants": ["E1", "E2", "E4", "O2"],
            "optimization_seeds": [42, 43, 44],
            "legacy_anchor_runs": 1,
            "checkpoint": "ema_best_train_loss_checkpoint.pt",
            "checkpoint_selection": "minimum_training_total_loss",
            "test_evaluation": "one explicit evaluation after checkpoint pre-selection",
            "test_used_for_epoch_selection": False,
            "historical_test_globally_unseen": False,
            "historical_test_note": (
                "The adaptation runs do not use test results for epoch selection, "
                "but this historical test set has been viewed in prior project work."
            ),
            "bootstrap_unit": "optimization seed",
            "bootstrap_method": "percentile bootstrap of the three-seed mean",
            "bootstrap_resamples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "standard_deviation": "sample standard deviation (ddof=1)",
        },
        "legacy_anchor": legacy[0] if len(legacy) == 1 else None,
        "variants": variants,
        "per_class": per_class,
        "paired_mcnemar": paired,
        "cross_run_contract_errors": cross_run_errors,
        "runs": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "aggregate_results.json", report)

    aggregate_rows: list[dict[str, Any]] = []
    for variant, variant_value in variants.items():
        for metric, item in variant_value["metrics"].items():
            aggregate_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "status": variant_value["status"],
                    "n_seeds": item["n"],
                    "seeds": ";".join(str(seed) for seed in variant_value["completed_seeds"]),
                    "mean": item["mean"],
                    "sample_std": item["sample_std"],
                    "bootstrap_95_ci_low": item["bootstrap_95_ci"][0],
                    "bootstrap_95_ci_high": item["bootstrap_95_ci"][1],
                    "legacy_anchor": item["legacy_anchor"],
                    "absolute_delta_vs_legacy": item["absolute_delta_vs_legacy"],
                    "relative_delta_vs_legacy_percent": item[
                        "relative_delta_vs_legacy_percent"
                    ],
                }
            )
    _write_csv(
        output_dir / "aggregate_metrics.csv",
        aggregate_rows,
        (
            "variant",
            "metric",
            "status",
            "n_seeds",
            "seeds",
            "mean",
            "sample_std",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
            "legacy_anchor",
            "absolute_delta_vs_legacy",
            "relative_delta_vs_legacy_percent",
        ),
    )
    _write_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        (
            "variant",
            "class",
            "metric",
            "n_seeds",
            "mean",
            "sample_std",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
        ),
    )
    _write_csv(
        output_dir / "paired_mcnemar.csv",
        paired,
        (
            "family",
            "left_variant",
            "right_variant",
            "seed",
            "right_seed",
            "query_count",
            "both_correct",
            "left_only_correct_b",
            "right_only_correct_c",
            "both_wrong",
            "left_minus_right_top1",
            "prediction_disagreement_count",
            "prediction_disagreement_fraction",
            "discordant_count",
            "odds_ratio_haldane",
            "p_exact_two_sided",
            "p_holm_all_reported",
            "method",
        ),
    )
    inventory_rows = []
    for record in records:
        checkpoint = record.get("checkpoint", {})
        training = record.get("training", {})
        inventory_rows.append(
            {
                "directory": record["directory"],
                "variant": record["variant"],
                "seed": record["seed"],
                "router": record["router"],
                "top_k": record["top_k"],
                "status": record["status"],
                "missing": ";".join(record.get("missing", [])),
                "errors": ";".join(record.get("errors", [])),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_train_loss": checkpoint.get("train_loss"),
                "checkpoint_weight_variant": checkpoint.get("weight_variant"),
                "checkpoint_selection_criterion": checkpoint.get("selection_criterion"),
                "test_metrics_used_for_selection": checkpoint.get(
                    "test_metrics_used_for_selection"
                ),
                "training_minimum_loss_epoch": training.get(
                    "minimum_training_loss_epoch"
                ),
                "finite_per_epoch_test_metric_count": training.get(
                    "finite_per_epoch_test_metric_count"
                ),
                "predictions_available": record["predictions_available"],
            }
        )
    _write_csv(
        output_dir / "run_inventory.csv",
        inventory_rows,
        (
            "directory",
            "variant",
            "seed",
            "router",
            "top_k",
            "status",
            "missing",
            "errors",
            "checkpoint_epoch",
            "checkpoint_train_loss",
            "checkpoint_weight_variant",
            "checkpoint_selection_criterion",
            "test_metrics_used_for_selection",
            "training_minimum_loss_epoch",
            "finite_per_epoch_test_metric_count",
            "predictions_available",
        ),
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the pre-registered three-seed router experiment"
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return a non-zero status when any formal seed or anchor is pending/invalid",
    )
    args = parser.parse_args()
    report = generate_report(
        runs_dir=args.runs_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        markdown_path=args.markdown.expanduser().resolve(),
        bootstrap_samples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    completed = sum(run["status"] == "complete" for run in report["runs"])
    pending = sum(run["status"] == "pending" for run in report["runs"])
    errors = sum(run["status"] == "error" for run in report["runs"])
    print(
        f"formal_results status={report['status']} complete={completed}/13 "
        f"pending={pending} errors={errors}"
    )
    print(f"JSON/CSV: {args.output_dir.expanduser().resolve()}")
    print(f"Markdown: {args.markdown.expanduser().resolve()}")
    return 2 if args.require_complete and report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
