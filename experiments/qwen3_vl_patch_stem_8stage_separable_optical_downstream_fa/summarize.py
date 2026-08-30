from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .queue import build_job_matrix, completion_reason, parse_int_list
    from .settings import METHODS, TASKS, load_settings
except ImportError:  # pragma: no cover
    from queue import build_job_matrix, completion_reason, parse_int_list  # type: ignore[no-redef]
    from settings import METHODS, TASKS, load_settings  # type: ignore[no-redef]


SUMMARY_FORMAT = "p12-downstream-summary-v2"
DEFAULT_BOOTSTRAP_SAMPLES = 100_000


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): number
        for key, item in value.items()
        if (number := _finite_number(item)) is not None
    }


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        number
        for item in value
        if (number := _finite_number(item)) is not None
    ]


def _summary(values: Iterable[Any]) -> dict[str, Any]:
    selected = [
        number
        for value in values
        if (number := _finite_number(value)) is not None
    ]
    return {
        "n": len(selected),
        "mean": statistics.fmean(selected) if selected else None,
        # A single seed has no sample variance; null avoids overstating certainty.
        "std": statistics.stdev(selected) if len(selected) > 1 else None,
        "min": min(selected) if selected else None,
        "max": max(selected) if selected else None,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, identity: str
) -> tuple[float | None, float | None]:
    """Deterministic percentile CI over paired seed-level observations."""

    if len(values) < 2:
        return None, None
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    seed_bytes = hashlib.sha256(identity.encode("utf-8")).digest()[:8]
    generator = random.Random(int.from_bytes(seed_bytes, "big"))
    count = len(values)
    means = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _value_by_seed(
    rows: Sequence[Mapping[str, Any]], task: str, method: str
) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        if row.get("task") != task or row.get("method") != method:
            continue
        value = _finite_number(row.get("test_primary"))
        if value is not None:
            values[int(row["seed"])] = value
    return values


def _paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    left_method: str,
    right_method: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    left = _value_by_seed(rows, task, left_method)
    right = _value_by_seed(rows, task, right_method)
    seeds = sorted(set(left) & set(right))
    pairs = [
        {
            "seed": seed,
            "left": left[seed],
            "right": right[seed],
            "delta_left_minus_right": left[seed] - right[seed],
        }
        for seed in seeds
    ]
    deltas = [float(pair["delta_left_minus_right"]) for pair in pairs]
    low, high = _bootstrap_mean_ci(
        deltas,
        samples=bootstrap_samples,
        identity=(
            f"P12|{task}|{left_method}|{right_method}|paired-contrast|"
            f"{bootstrap_samples}"
        ),
    )
    return {
        "definition": f"{left_method} - {right_method} on identical seeds",
        "paired_n": len(pairs),
        "paired_seeds": seeds,
        "raw_pairs": pairs,
        "mean": statistics.fmean(deltas) if deltas else None,
        "sample_std": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "bootstrap_samples": bootstrap_samples if len(deltas) >= 2 else 0,
        "bootstrap_unit": "paired_seed",
        "inference_note": (
            "descriptive_only_fewer_than_three_seeds" if len(deltas) < 3 else None
        ),
    }


def _bp_recovery_summary(
    rows: Sequence[Mapping[str, Any]], *, task: str, method: str
) -> dict[str, Any]:
    definition = "mean(method - NoFT) / mean(BP - NoFT), paired by seed"
    if method == "noft":
        return {
            "definition": definition,
            "paired_n": 0,
            "ratio_of_paired_mean_gains": None,
            "diagnostic": False,
            "non_diagnostic_reason": "undefined_for_noft_baseline",
        }
    noft = _value_by_seed(rows, task, "noft")
    bp = _value_by_seed(rows, task, "bp")
    candidate = _value_by_seed(rows, task, method)
    seeds = sorted(set(noft) & set(bp) & set(candidate))
    method_gains = [candidate[seed] - noft[seed] for seed in seeds]
    bp_gains = [bp[seed] - noft[seed] for seed in seeds]
    mean_method = statistics.fmean(method_gains) if method_gains else None
    mean_bp = statistics.fmean(bp_gains) if bp_gains else None
    numerical_epsilon = 1.0e-12
    ratio = (
        mean_method / mean_bp
        if mean_method is not None
        and mean_bp is not None
        and abs(mean_bp) > numerical_epsilon
        else None
    )
    nonzero_signs = {
        1 if gain > numerical_epsilon else -1
        for gain in bp_gains
        if abs(gain) > numerical_epsilon
    }
    reason = None
    if not seeds:
        reason = "no_three_way_paired_seeds"
    elif len(seeds) < 2:
        reason = "fewer_than_two_paired_seeds"
    elif mean_bp is None or abs(mean_bp) <= numerical_epsilon:
        reason = "mean_bp_gain_is_numerically_zero"
    elif len(nonzero_signs) > 1:
        reason = "bp_gain_changes_sign_across_seeds"
    return {
        "definition": definition,
        "paired_n": len(seeds),
        "paired_seeds": seeds,
        "method_gains": method_gains,
        "bp_gains": bp_gains,
        "method_mean_gain": mean_method,
        "bp_mean_gain": mean_bp,
        "ratio_of_paired_mean_gains": ratio,
        "diagnostic": reason is None,
        "non_diagnostic_reason": reason,
        "practical_epsilon_note": (
            "Only numerical zero is enforced here; scientific interpretation must "
            "also require a preregistered task-level practical BP-gain threshold."
        ),
    }


def _throughput_from_history(run_dir: Path) -> dict[str, Any] | None:
    history_path = run_dir / "metrics" / "history.json"
    history = _read_json(history_path)
    if not isinstance(history, list):
        return None
    epoch_rows: list[dict[str, float]] = []
    for row in history:
        train = row.get("train", {}) if isinstance(row, Mapping) else {}
        epoch = _finite_number(row.get("epoch")) if isinstance(row, Mapping) else None
        samples = _finite_number(train.get("samples"))
        seconds = _finite_number(train.get("seconds"))
        rate = _finite_number(train.get("images_per_second"))
        if epoch is None:
            continue
        epoch_rows.append(
            {
                "epoch": epoch,
                "samples": samples if samples is not None else math.nan,
                "seconds": seconds if seconds is not None else math.nan,
                "images_per_second": rate if rate is not None else math.nan,
            }
        )
    valid_timing = [
        row
        for row in epoch_rows
        if math.isfinite(row["samples"])
        and math.isfinite(row["seconds"])
        and row["seconds"] > 0
    ]
    total_samples = sum(row["samples"] for row in valid_timing) if valid_timing else None
    total_seconds = sum(row["seconds"] for row in valid_timing) if valid_timing else None
    warm = sorted(
        row["images_per_second"]
        for row in epoch_rows
        if row["epoch"] >= 2 and math.isfinite(row["images_per_second"])
    )
    process = _read_json(run_dir / "process.json")
    process = process if isinstance(process, Mapping) else {}
    return {
        "source": str(history_path),
        "epochs_recorded": len(history),
        "epochs_with_timing": len(valid_timing),
        "total_train_samples_processed": total_samples,
        "total_train_seconds": total_seconds,
        "overall_images_per_second": (
            total_samples / total_seconds
            if total_samples is not None and total_seconds is not None and total_seconds > 0
            else None
        ),
        "epoch_2_plus_images_per_second_median": statistics.median(warm) if warm else None,
        "epoch_2_plus_images_per_second_iqr": (
            _percentile(warm, 0.75) - _percentile(warm, 0.25) if warm else None
        ),
        "epoch_1_images_per_second": next(
            (
                row["images_per_second"]
                for row in epoch_rows
                if row["epoch"] == 1 and math.isfinite(row["images_per_second"])
            ),
            None,
        ),
        "gpu_index": process.get("gpu_index"),
        "gpu_uuid": process.get("gpu_uuid"),
    }


def _gradient_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    stages = value.get("per_stage")
    rows = stages if isinstance(stages, list) else []
    cosines: list[float] = []
    norm_ratios: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cosine = _finite_number(row.get("cosine_to_bp_current"))
        ratio = _finite_number(row.get("norm_ratio_to_bp_current"))
        if cosine is not None:
            cosines.append(cosine)
        if ratio is not None:
            norm_ratios.append(ratio)
    return {
        "mean_cosine_stages_1_to_7": _finite_number(
            value.get("mean_cosine_stages_1_to_7")
        ),
        "per_stage_cosine_to_bp_current": cosines,
        "per_stage_norm_ratio_to_bp_current": norm_ratios,
        "stage_8_local_gradient_expected_exact": value.get(
            "last_stage_expected_exact_local_gradient"
        ),
        "trainable_gradient_groups": value.get("trainable_gradient_groups"),
    }


def _gradient_trajectory(run_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path in sorted((run_dir / "diagnostics").glob("gradient_epoch_*.json")):
        payload = _read_json(path)
        summary = _gradient_summary(payload)
        if summary is not None:
            output[path.stem.removeprefix("gradient_epoch_")] = summary
    selected = _gradient_summary(_read_json(run_dir / "diagnostics" / "gradient_selected.json"))
    if selected is not None:
        output["selected"] = selected
    return output


def _electronic_gate_report(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, float]] = []
    for item in value:
        numeric = _numeric_mapping(item)
        if numeric:
            rows.append(numeric)
    return {"per_stage": rows} if rows else None


def _metric_summary_by_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    ablations = sorted(
        {
            ablation
            for row in rows
            for ablation in row.get("test_metrics_by_ablation", {})
        }
    )
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for ablation in ablations:
        names = sorted(
            {
                metric
                for row in rows
                for metric in row.get("test_metrics_by_ablation", {}).get(ablation, {})
            }
        )
        output[ablation] = {
            name: _summary(
                row.get("test_metrics_by_ablation", {}).get(ablation, {}).get(name)
                for row in rows
            )
            for name in names
        }
    return output


def _per_stage_summary(
    rows: Sequence[Mapping[str, Any]], path: Sequence[str]
) -> list[dict[str, Any]]:
    lists: list[list[float]] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        current = _numeric_list(value)
        if current:
            lists.append(current)
    count = max((len(values) for values in lists), default=0)
    return [
        {
            "stage": stage + 1,
            **_summary(values[stage] for values in lists if stage < len(values)),
        }
        for stage in range(count)
    ]


def _electronic_gate_stage_summary(
    rows: Sequence[Mapping[str, Any]], gate_name: str
) -> list[dict[str, Any]]:
    reports = [
        row.get("electronic_skip_gate_report", {}).get("per_stage", [])
        for row in rows
        if isinstance(row.get("electronic_skip_gate_report"), Mapping)
    ]
    count = max((len(report) for report in reports), default=0)
    return [
        {
            "stage": stage + 1,
            **_summary(
                report[stage].get(gate_name)
                for report in reports
                if stage < len(report) and isinstance(report[stage], Mapping)
            ),
        }
        for stage in range(count)
    ]


def _run_row(settings: Any, result: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    primary = settings.task_settings.primary_metric
    raw_test = result.get("test", {})
    tests = {
        str(ablation): _numeric_mapping(metrics)
        for ablation, metrics in raw_test.items()
        if isinstance(raw_test, Mapping) and isinstance(metrics, Mapping)
    }
    normal = tests.get("normal", {})
    normal_primary = _finite_number(normal.get(primary))
    ablation_values = {
        name: metrics[primary]
        for name, metrics in tests.items()
        if primary in metrics
    }
    dependency_drop = {
        name: normal_primary - value
        for name, value in ablation_values.items()
        if name != "normal" and normal_primary is not None
    }
    phase = result.get("phase") if isinstance(result.get("phase"), Mapping) else {}
    selected_gradient = _gradient_summary(result.get("selected_gradient_diagnostic"))
    initial_gradient = _gradient_summary(result.get("initial_gradient_diagnostic"))
    optical_gates = _numeric_list(result.get("optical_gates"))
    return {
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "status": "complete",
        "reason": "complete",
        "primary_metric": primary,
        "test_primary": normal_primary,
        "best_validation_metric": _finite_number(result.get("best_validation_metric")),
        "best_epoch": result.get("best_epoch"),
        "test_metrics_by_ablation": tests,
        "test_metadata_by_ablation": {
            str(name): {
                str(key): value
                for key, value in metrics.items()
                if _finite_number(value) is None
            }
            for name, metrics in raw_test.items()
            if isinstance(raw_test, Mapping) and isinstance(metrics, Mapping)
        },
        "normal_secondary_metrics": {
            name: value for name, value in normal.items() if name != primary
        },
        "ablation_primary": ablation_values,
        "ablation_dependency_drop": dependency_drop,
        "ablation_interpretation": "coadapted_destructive_dependency_not_standalone_model",
        "phase_report": phase,
        "phase_mean_absolute_rad": _finite_number(phase.get("mean_absolute_rad")),
        "phase_median_absolute_rad": _finite_number(phase.get("median_absolute_rad")),
        "phase_fraction_over_0p1_rad": _finite_number(
            phase.get("fraction_over_0p1_rad")
        ),
        "initial_gradient": initial_gradient,
        "selected_gradient": selected_gradient,
        "gradient_trajectory": _gradient_trajectory(result_path.parent),
        "gradient_cosine_stages_1_to_7": (
            selected_gradient.get("mean_cosine_stages_1_to_7")
            if selected_gradient is not None
            else None
        ),
        "optical_gates": optical_gates,
        "electronic_skip_gate_report": _electronic_gate_report(
            result.get("electronic_skip_gates")
        ),
        "gate_interpretation": "learned_mixture_controls_not_energy_or_compute_fraction",
        "throughput": _throughput_from_history(result_path.parent),
        "peak_cuda_memory_bytes": _finite_number(result.get("peak_cuda_memory_bytes")),
        "cuda_device_name": result.get("cuda_device_name"),
        "result_path": str(result_path),
    }


def _incomplete_row(settings: Any, result_path: Path, reason: str) -> dict[str, Any]:
    return {
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "status": "incomplete",
        "reason": reason,
        "primary_metric": settings.task_settings.primary_metric,
        "test_primary": None,
        "best_validation_metric": None,
        "best_epoch": None,
        "test_metrics_by_ablation": {},
        "test_metadata_by_ablation": {},
        "normal_secondary_metrics": {},
        "ablation_primary": {},
        "ablation_dependency_drop": {},
        "phase_report": None,
        "phase_mean_absolute_rad": None,
        "phase_median_absolute_rad": None,
        "phase_fraction_over_0p1_rad": None,
        "initial_gradient": None,
        "selected_gradient": None,
        "gradient_trajectory": {},
        "gradient_cosine_stages_1_to_7": None,
        "optical_gates": [],
        "electronic_skip_gate_report": None,
        "throughput": None,
        "peak_cuda_memory_bytes": None,
        "cuda_device_name": None,
        "result_path": str(result_path),
    }


def _throughput_by_device(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    devices = sorted(
        {
            str(row["cuda_device_name"])
            for row in rows
            if row.get("cuda_device_name") is not None
        }
    )
    return {
        device: {
            "run_count": sum(row.get("cuda_device_name") == device for row in rows),
            "overall_images_per_second": _summary(
                row.get("throughput", {}).get("overall_images_per_second")
                if isinstance(row.get("throughput"), Mapping)
                and row.get("cuda_device_name") == device
                else None
                for row in rows
            ),
            "epoch_2_plus_images_per_second_median": _summary(
                row.get("throughput", {}).get(
                    "epoch_2_plus_images_per_second_median"
                )
                if isinstance(row.get("throughput"), Mapping)
                and row.get("cuda_device_name") == device
                else None
                for row in rows
            ),
            "peak_cuda_memory_bytes": _summary(
                row.get("peak_cuda_memory_bytes")
                if row.get("cuda_device_name") == device
                else None
                for row in rows
            ),
        }
        for device in devices
    }


def _aggregate_group(
    all_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    method: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    primary_summary = _summary(row.get("test_primary") for row in group_rows)
    contrast = _paired_contrast(
        all_rows,
        task=task,
        left_method=method,
        right_method="noft",
        bootstrap_samples=bootstrap_samples,
    )
    ablations = sorted(
        {
            name
            for row in group_rows
            for name in row.get("ablation_dependency_drop", {})
        }
    )
    return {
        "task": task,
        "method": method,
        "completed_seeds": sum(row.get("status") == "complete" for row in group_rows),
        "test_primary": primary_summary,
        "mean_test_primary": primary_summary["mean"],
        "std_test_primary": primary_summary["std"],
        "mean_delta_vs_noft": contrast["mean"],
        "paired_delta_vs_noft": contrast,
        "bp_recovery": _bp_recovery_summary(all_rows, task=task, method=method),
        "test_metric_summary_by_ablation": _metric_summary_by_ablation(group_rows),
        "ablation_dependency_drop": {
            name: _summary(
                row.get("ablation_dependency_drop", {}).get(name)
                for row in group_rows
            )
            for name in ablations
        },
        "phase_mean_absolute_rad": _summary(
            row.get("phase_mean_absolute_rad") for row in group_rows
        ),
        "phase_median_absolute_rad": _summary(
            row.get("phase_median_absolute_rad") for row in group_rows
        ),
        "phase_fraction_over_0p1_rad": _summary(
            row.get("phase_fraction_over_0p1_rad") for row in group_rows
        ),
        "phase_per_stage_mean_absolute_rad": _per_stage_summary(
            group_rows, ("phase_report", "per_stage_mean_absolute_rad")
        ),
        "phase_per_stage_rms_rad": _per_stage_summary(
            group_rows, ("phase_report", "per_stage_rms_rad")
        ),
        "selected_gradient_cosine_stages_1_to_7": _summary(
            row.get("gradient_cosine_stages_1_to_7") for row in group_rows
        ),
        "selected_gradient_per_stage_cosine": _per_stage_summary(
            group_rows, ("selected_gradient", "per_stage_cosine_to_bp_current")
        ),
        "selected_gradient_per_stage_norm_ratio": _per_stage_summary(
            group_rows,
            ("selected_gradient", "per_stage_norm_ratio_to_bp_current"),
        ),
        "optical_gate_per_stage": _per_stage_summary(group_rows, ("optical_gates",)),
        "electronic_skip_gates_per_stage": {
            name: _electronic_gate_stage_summary(group_rows, name)
            for name in ("spatial", "channel", "output_scale")
        },
        # Deliberately stratified: 3090/4090 and frozen/adapting runs are not pooled.
        "throughput_by_cuda_device": _throughput_by_device(group_rows),
        "total_train_seconds": _summary(
            row.get("throughput", {}).get("total_train_seconds")
            if isinstance(row.get("throughput"), Mapping)
            else None
            for row in group_rows
        ),
    }


def collect_results(
    config: Path,
    seeds: Iterable[int],
    adaptation_seeds: Iterable[int] | None = None,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    normalized_adaptation = (
        normalized_seeds
        if adaptation_seeds is None
        else tuple(dict.fromkeys(int(seed) for seed in adaptation_seeds))
    )
    jobs = build_job_matrix(normalized_seeds, normalized_adaptation)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        settings = load_settings(config, task=job.task, method=job.method, seed=job.seed)
        complete, reason = completion_reason(settings)
        result_path = settings.output_dir / "result.json"
        result = _read_json(result_path) if complete else None
        rows.append(
            _run_row(settings, result, result_path)
            if isinstance(result, Mapping)
            else _incomplete_row(settings, result_path, reason)
        )

    noft = {
        (row["task"], row["seed"]): value
        for row in rows
        if row["method"] == "noft"
        and (value := _finite_number(row["test_primary"])) is not None
    }
    for row in rows:
        baseline = _finite_number(noft.get((row["task"], row["seed"])))
        value = _finite_number(row["test_primary"])
        row["delta_vs_noft"] = (
            value - baseline if value is not None and baseline is not None else None
        )

    aggregates = []
    contrasts: dict[str, Any] = {}
    for task in TASKS:
        contrasts[task] = {
            name: _paired_contrast(
                rows,
                task=task,
                left_method=left,
                right_method=right,
                bootstrap_samples=bootstrap_samples,
            )
            for name, left, right in (
                ("bp_minus_noft", "bp", "noft"),
                ("fa_pretrained_minus_noft", "fa_pretrained", "noft"),
                ("fa_random_minus_noft", "fa_random", "noft"),
                ("fa_pretrained_minus_fa_random", "fa_pretrained", "fa_random"),
                ("fa_pretrained_minus_bp", "fa_pretrained", "bp"),
            )
        }
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["task"] == task and row["method"] == method
            ]
            aggregates.append(
                _aggregate_group(
                    rows,
                    group,
                    task=task,
                    method=method,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return {
        "format": SUMMARY_FORMAT,
        "config": str(config.resolve()),
        "seeds": list(normalized_seeds),
        "adaptation_seeds": list(normalized_adaptation),
        "bootstrap_samples": bootstrap_samples,
        "statistical_note": (
            "All method contrasts are paired by task/seed. With three or fewer seeds, "
            "bootstrap intervals are descriptive rather than strong significance claims."
        ),
        "complete_runs": sum(row["status"] == "complete" for row in rows),
        "expected_runs": len(rows),
        "runs": rows,
        "paired_primary_contrasts": contrasts,
        "aggregates": aggregates,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True, ensure_ascii=False)
                            if isinstance(value, (dict, list, tuple))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate strict P12 result.json files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=(2026, 2027, 2028))
    parser.add_argument(
        "--adaptation-seeds",
        type=parse_int_list,
        help="subset of seeds expected to have BP/FA results",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="paired-seed bootstrap replicates for primary-metric contrasts",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    first = load_settings(args.config, task="caltech101", method="noft", seed=args.seeds[0])
    output = args.output or first.paths.output_root / "summary.json"
    summary = collect_results(
        args.config,
        args.seeds,
        args.adaptation_seeds,
        bootstrap_samples=args.bootstrap_samples,
    )
    _write_json_atomic(output, summary)
    write_csv(output.with_suffix(".csv"), summary["runs"])
    print(
        f"P12 results: {summary['complete_runs']}/{summary['expected_runs']} complete; "
        f"JSON={output}; CSV={output.with_suffix('.csv')}",
        flush=True,
    )
    for row in summary["aggregates"]:
        mean = row["mean_test_primary"]
        delta = row["mean_delta_vs_noft"]
        rendered = "pending" if mean is None else f"{mean:.6f}"
        delta_rendered = "pending" if delta is None else f"{delta:+.6f}"
        print(
            f"  {row['task']:10s} {row['method']:14s} "
            f"n={row['completed_seeds']} mean={rendered} paired_delta={delta_rendered}",
            flush=True,
        )
    return 0 if summary["complete_runs"] == summary["expected_runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
