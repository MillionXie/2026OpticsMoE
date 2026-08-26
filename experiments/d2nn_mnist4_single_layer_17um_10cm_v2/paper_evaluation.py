"""Paper-grade MNIST-4 metrics and figures from untouched four-ROI energies.

This module is evaluation-only.  It never opens a training checkpoint and
never modifies CCD values: predictions are recomputed as ``argmax`` of the four
raw ROI sums already written by :mod:`ccd_evaluate`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_IDS = (0, 1, 2, 3)
CLASS_NAMES = ("0", "1", "2", "3")
FORMAL_SAMPLES_PER_CLASS = 100
FORMAL_SAMPLE_COUNT = 400
QUICK_SAMPLES_PER_CLASS = 10
QUICK_SAMPLE_COUNT = 40
FIGURE_HEIGHT_CM = 5.0
CM_TO_INCH = 1.0 / 2.54
ENERGY_COLUMN_SETS = (
    tuple(f"raw_energy_{index}" for index in CLASS_IDS),
    tuple(f"detector_raw_energy_{index}" for index in CLASS_IDS),
    tuple(f"score_{index}" for index in CLASS_IDS),
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


@dataclass(frozen=True)
class PredictionRunSpec:
    mask_name: str
    predictions_path: Path
    metrics_path: Path | None = None
    profile_override: str | None = None
    suitable_override: bool | None = None
    phase_sha256_override: str | None = None


@dataclass
class LoadedRun:
    mask_name: str
    run_id: str
    profile: str
    reporting_status: str
    suitable_for_accuracy_reporting: bool
    eligible_for_formal_comparison: bool
    phase_sha256: str | None
    predictions_path: Path
    metrics_path: Path | None
    keys: list[str]
    labels: np.ndarray
    predictions: np.ndarray
    energies: np.ndarray
    correct: np.ndarray
    confusion: np.ndarray
    metrics: dict[str, Any]
    per_class: list[dict[str, Any]]
    sample_rows: list[dict[str, Any]]
    roi_fraction_matrix: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Prediction CSV is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    if not result:
        raise ValueError(f"Mask/profile name cannot be converted to a safe slug: {value!r}")
    return result[:100]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean string, got {value!r}")


def _resolve_prediction_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Prediction path is missing: {path}")
    candidates = [
        path / "hardware_predictions_raw.csv",
        path / "hardware_predictions.csv",
    ]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one hardware prediction CSV under {path}; got {matches}"
        )
    return matches[0]


def _discover_metrics_path(predictions_path: Path) -> Path | None:
    candidates = (
        predictions_path.with_name("hardware_metrics_raw.json"),
        predictions_path.with_name("hardware_metrics.json"),
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous metrics JSON beside {predictions_path}: {matches}")
    return matches[0] if matches else None


def _phase_sha256(metrics: dict[str, Any], metrics_path: Path | None) -> str | None:
    capture = metrics.get("capture_manifest")
    if isinstance(capture, dict):
        value = str(capture.get("phase_sha256") or "").strip().lower()
        if value:
            return value
    stage_value = metrics.get("stage_contract")
    candidates: list[Path] = []
    if stage_value:
        path = Path(str(stage_value)).expanduser()
        if not path.is_absolute() and metrics_path is not None:
            path = metrics_path.parent / path
        candidates.append(path.resolve())
    if metrics_path is not None:
        candidates.append((metrics_path.parent.parent / "stage_contract.json").resolve())
        candidates.append((metrics_path.parent / "stage_contract.json").resolve())
    for path in candidates:
        if not path.is_file():
            continue
        value = str(_read_json(path).get("phase_sha256") or "").strip().lower()
        if value:
            return value
    return None


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, confusion: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = int(len(labels))
    correct = int(np.trace(confusion))
    ci_low, ci_high = _wilson_interval(correct, total)
    per_class: list[dict[str, Any]] = []
    for class_id in CLASS_IDS:
        tp = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        predicted = int(confusion[:, class_id].sum())
        fn = support - tp
        fp = predicted - tp
        tn = total - tp - fn - fp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        class_ci_low, class_ci_high = _wilson_interval(tp, support)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "support": support,
                "predicted_count": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "recall_wilson95_low": class_ci_low,
                "recall_wilson95_high": class_ci_high,
            }
        )
    weights = np.asarray([row["support"] for row in per_class], dtype=np.float64)
    weight_sum = float(weights.sum())
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    weighted_precision = _safe_divide(
        float(sum(row["precision"] * row["support"] for row in per_class)),
        weight_sum,
    )
    weighted_recall = _safe_divide(
        float(sum(row["recall"] * row["support"] for row in per_class)),
        weight_sum,
    )
    weighted_f1 = _safe_divide(
        float(sum(row["f1"] * row["support"] for row in per_class)),
        weight_sum,
    )
    summary = {
        "samples": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
        "accuracy_wilson95_low": ci_low,
        "accuracy_wilson95_high": ci_high,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": _safe_divide(correct, total),
        "micro_recall": _safe_divide(correct, total),
        "micro_f1": _safe_divide(correct, total),
    }
    return summary, per_class


def _classify_profile(
    suitable: bool, labels: np.ndarray, profile: str
) -> tuple[str, bool]:
    supports = [int(np.sum(labels == class_id)) for class_id in CLASS_IDS]
    normalized_profile = profile.strip().lower()
    is_declared_quick = normalized_profile.startswith("quick40") or normalized_profile.startswith(
        "quick_"
    )
    # quick40 is an intentionally small fixed-random diagnostic.  Its stage
    # contract may conservatively set suitable_for_accuracy_reporting=false,
    # but that does not make it a simulation-selected demo.  It remains
    # non-formal and is never admitted to aggregate mask comparisons.
    if (
        is_declared_quick
        and len(labels) == QUICK_SAMPLE_COUNT
        and supports == [QUICK_SAMPLES_PER_CLASS] * 4
    ):
        return "quick40_diagnostic", False
    if not suitable:
        return "biased_demo_diagnostic", False
    if len(labels) == FORMAL_SAMPLE_COUNT and supports == [FORMAL_SAMPLES_PER_CLASS] * 4:
        return "formal400", True
    if len(labels) == QUICK_SAMPLE_COUNT and supports == [QUICK_SAMPLES_PER_CLASS] * 4:
        return "quick40_diagnostic", False
    return f"nonformal_diagnostic:{profile}", False


def _select_energy_columns(rows: list[dict[str, str]]) -> tuple[str, str, str, str]:
    headers = set(rows[0])
    for columns in ENERGY_COLUMN_SETS:
        if set(columns).issubset(headers):
            return columns
    raise RuntimeError(
        "Prediction CSV must contain one four-energy column set: "
        + ", ".join("/".join(columns) for columns in ENERGY_COLUMN_SETS)
    )


def _load_run(spec: PredictionRunSpec, allow_biased_diagnostic: bool) -> LoadedRun:
    predictions_path = _resolve_prediction_path(spec.predictions_path)
    metrics_path = (
        spec.metrics_path.expanduser().resolve()
        if spec.metrics_path is not None
        else _discover_metrics_path(predictions_path)
    )
    metrics_source = _read_json(metrics_path) if metrics_path is not None else {}
    profile = str(
        spec.profile_override
        if spec.profile_override is not None
        else metrics_source.get("profile", "unspecified")
    )
    suitable_value = (
        spec.suitable_override
        if spec.suitable_override is not None
        else metrics_source.get("suitable_for_accuracy_reporting")
    )
    if not isinstance(suitable_value, bool):
        raise RuntimeError(
            f"Run {spec.mask_name!r} needs metrics JSON or an explicit suitable override"
        )
    rows = _read_csv(predictions_path)
    energy_columns = _select_energy_columns(rows)
    keys: list[str] = []
    labels: list[int] = []
    recorded_predictions: list[int] = []
    energy_rows: list[list[float]] = []
    for index, row in enumerate(rows):
        key = str(row.get("key", "")).strip()
        if not key or key in keys:
            raise RuntimeError(f"Invalid/duplicate sample key at row {index}: {key!r}")
        label = int(row["label"])
        if label not in CLASS_IDS:
            raise RuntimeError(f"MNIST-4 label must be 0..3, got {label} for {key}")
        energies = [float(row[column]) for column in energy_columns]
        if not all(math.isfinite(value) and value >= 0.0 for value in energies):
            raise RuntimeError(f"ROI energies must be finite and nonnegative for {key}")
        derived_prediction = int(np.argmax(np.asarray(energies)))
        if row.get("prediction", "") != "" and int(row["prediction"]) != derived_prediction:
            raise RuntimeError(
                f"Recorded prediction does not equal raw-energy argmax for {key}: "
                f"recorded={row['prediction']} derived={derived_prediction}"
            )
        if row.get("correct", "") != "":
            recorded_correct = _parse_bool(row["correct"])
            if recorded_correct != (derived_prediction == label):
                raise RuntimeError(f"Recorded correct flag is inconsistent for {key}")
        keys.append(key)
        labels.append(label)
        recorded_predictions.append(derived_prediction)
        energy_rows.append(energies)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(recorded_predictions, dtype=np.int64)
    energies_array = np.asarray(energy_rows, dtype=np.float64)
    correct = predictions_array == labels_array
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (labels_array, predictions_array), 1)
    if "confusion_matrix" in metrics_source:
        source_confusion = np.asarray(metrics_source["confusion_matrix"], dtype=np.int64)
        if source_confusion.shape != (4, 4) or not np.array_equal(
            source_confusion, confusion
        ):
            raise RuntimeError(
                f"Source metrics confusion matrix disagrees with predictions for {spec.mask_name}"
            )
    if suitable_value and "accuracy" in metrics_source:
        observed_accuracy = float(np.mean(correct))
        if not math.isclose(
            float(metrics_source["accuracy"]), observed_accuracy, abs_tol=1.0e-12
        ):
            raise RuntimeError("Source accuracy disagrees with prediction CSV")

    reporting_status, eligible = _classify_profile(suitable_value, labels_array, profile)
    if reporting_status == "biased_demo_diagnostic" and not allow_biased_diagnostic:
        raise PermissionError(
            f"Run {spec.mask_name!r}/{profile} is a biased demo. Pass "
            "--allow-biased-diagnostic only for clearly labelled diagnostics."
        )
    classification, per_class = _classification_metrics(
        labels_array, predictions_array, confusion
    )
    total_energy = energies_array.sum(axis=1)
    if np.any(total_energy <= 0.0):
        raise RuntimeError("Every sample must contain positive total energy across four ROIs")
    fractions = energies_array / total_energy[:, None]
    target_energy = energies_array[np.arange(len(labels_array)), labels_array]
    target_fraction = fractions[np.arange(len(labels_array)), labels_array]
    masked = energies_array.copy()
    masked[np.arange(len(labels_array)), labels_array] = -np.inf
    max_wrong = masked.max(axis=1)
    raw_margin = target_energy - max_wrong
    normalized_margin = raw_margin / total_energy
    ratio = target_energy / np.maximum(max_wrong, np.finfo(np.float64).eps)
    sample_rows: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        sample_rows.append(
            {
                "key": key,
                "label": int(labels_array[index]),
                "prediction": int(predictions_array[index]),
                "correct": bool(correct[index]),
                "energy_0": float(energies_array[index, 0]),
                "energy_1": float(energies_array[index, 1]),
                "energy_2": float(energies_array[index, 2]),
                "energy_3": float(energies_array[index, 3]),
                "total_roi_energy": float(total_energy[index]),
                "target_roi_energy": float(target_energy[index]),
                "max_wrong_roi_energy": float(max_wrong[index]),
                "target_energy_fraction": float(target_fraction[index]),
                "raw_target_margin": float(raw_margin[index]),
                "normalized_target_margin": float(normalized_margin[index]),
                "target_to_max_wrong_ratio": float(ratio[index]),
            }
        )
    roi_fraction_matrix = np.stack(
        [fractions[labels_array == class_id].mean(axis=0) for class_id in CLASS_IDS]
    )
    classification.update(
        {
            "mean_target_energy_fraction": float(np.mean(target_fraction)),
            "median_target_energy_fraction": float(np.median(target_fraction)),
            "mean_normalized_target_margin": float(np.mean(normalized_margin)),
            "median_normalized_target_margin": float(np.median(normalized_margin)),
            "mean_target_to_max_wrong_ratio": float(np.mean(ratio)),
        }
    )
    for row in per_class:
        row["mask_name"] = spec.mask_name
        row["profile"] = profile
        row["reporting_status"] = reporting_status
        row["eligible_for_formal_comparison"] = eligible
    phase_sha = (
        spec.phase_sha256_override.strip().lower()
        if spec.phase_sha256_override
        else _phase_sha256(metrics_source, metrics_path)
    )
    run_id = f"{_slug(spec.mask_name)}__{_slug(profile)}"
    return LoadedRun(
        mask_name=spec.mask_name,
        run_id=run_id,
        profile=profile,
        reporting_status=reporting_status,
        suitable_for_accuracy_reporting=suitable_value,
        eligible_for_formal_comparison=eligible,
        phase_sha256=phase_sha,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        keys=keys,
        labels=labels_array,
        predictions=predictions_array,
        energies=energies_array,
        correct=correct,
        confusion=confusion,
        metrics=classification,
        per_class=per_class,
        sample_rows=sample_rows,
        roi_fraction_matrix=roi_fraction_matrix,
    )


def _mcnemar_exact_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    tail = sum(math.comb(total, value) for value in range(lower + 1)) / (2.0**total)
    return min(1.0, 2.0 * tail)


def _paired_comparisons(formal_runs: list[LoadedRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first, second in combinations(formal_runs, 2):
        first_by_key = {
            key: (int(label), bool(correct), sample)
            for key, label, correct, sample in zip(
                first.keys, first.labels, first.correct, first.sample_rows
            )
        }
        second_by_key = {
            key: (int(label), bool(correct), sample)
            for key, label, correct, sample in zip(
                second.keys, second.labels, second.correct, second.sample_rows
            )
        }
        if set(first_by_key) != set(second_by_key):
            raise RuntimeError(
                f"Formal masks {first.mask_name!r} and {second.mask_name!r} do not "
                "use the same fixed 400 samples"
            )
        keys = sorted(first_by_key)
        if any(first_by_key[key][0] != second_by_key[key][0] for key in keys):
            raise RuntimeError("Paired formal masks disagree on ground-truth labels")
        first_correct = np.asarray([first_by_key[key][1] for key in keys], dtype=bool)
        second_correct = np.asarray([second_by_key[key][1] for key in keys], dtype=bool)
        first_only = int(np.sum(first_correct & ~second_correct))
        second_only = int(np.sum(~first_correct & second_correct))
        first_margin = np.asarray(
            [first_by_key[key][2]["normalized_target_margin"] for key in keys]
        )
        second_margin = np.asarray(
            [second_by_key[key][2]["normalized_target_margin"] for key in keys]
        )
        rows.append(
            {
                "mask_a": first.mask_name,
                "mask_b": second.mask_name,
                "samples": len(keys),
                "accuracy_a": first.metrics["accuracy"],
                "accuracy_b": second.metrics["accuracy"],
                "accuracy_b_minus_a": second.metrics["accuracy"]
                - first.metrics["accuracy"],
                "macro_f1_a": first.metrics["macro_f1"],
                "macro_f1_b": second.metrics["macro_f1"],
                "macro_f1_b_minus_a": second.metrics["macro_f1"]
                - first.metrics["macro_f1"],
                "a_correct_b_wrong": first_only,
                "a_wrong_b_correct": second_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact_p(first_only, second_only),
                "mean_paired_normalized_margin_b_minus_a": float(
                    np.mean(second_margin - first_margin)
                ),
            }
        )
    return rows


def _configure_matplotlib() -> tuple[Any, str | None]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    try:
        resolved = font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        resolved = None
    return plt, resolved


def _save_figure(
    fig: Any,
    output_dir: Path,
    basename: str,
    *,
    width_cm: float,
    height_cm: float,
    description: str,
    eligible_for_paper: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for suffix, dpi in (("pdf", 600), ("svg", 600), ("png", 600)):
        path = output_dir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
        rows.append(
            {
                "figure": basename,
                "format": suffix,
                "relative_path": path.relative_to(output_dir.parent).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "font_family_requested": "Arial",
                "font_size_pt": 7.0,
                "eligible_for_paper": eligible_for_paper,
                "description": description,
            }
        )
    return rows


def _plot_figures(runs: list[LoadedRun], output_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    plt, resolved_font = _configure_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("optical_blue", ("#FFFFFF", "#0072B2"))
    figure_rows: list[dict[str, Any]] = []
    figures_dir = output_dir / "figures"
    for run in runs:
        title_suffix = "formal400" if run.eligible_for_formal_comparison else run.reporting_status
        width_cm = 5.6
        fig, axis = plt.subplots(
            figsize=(width_cm * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
            constrained_layout=True,
        )
        row_totals = run.confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            run.confusion,
            row_totals,
            out=np.zeros_like(run.confusion, dtype=np.float64),
            where=row_totals > 0,
        )
        axis.imshow(normalized, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        for true_id in CLASS_IDS:
            for predicted_id in CLASS_IDS:
                value = normalized[true_id, predicted_id]
                axis.text(
                    predicted_id,
                    true_id,
                    f"{run.confusion[true_id, predicted_id]}\n{100.0 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value >= 0.55 else "black",
                )
        axis.set_xticks(CLASS_IDS, CLASS_NAMES)
        axis.set_yticks(CLASS_IDS, CLASS_NAMES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"{run.mask_name} ({title_suffix}, n={len(run.labels)})")
        figure_rows.extend(
            _save_figure(
                fig,
                figures_dir,
                f"confusion_{run.run_id}",
                width_cm=width_cm,
                height_cm=FIGURE_HEIGHT_CM,
                description="Row-normalized confusion matrix; cells show count and row percentage.",
                eligible_for_paper=run.eligible_for_formal_comparison,
            )
        )
        plt.close(fig)

        fig, axis = plt.subplots(
            figsize=(width_cm * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
            constrained_layout=True,
        )
        axis.imshow(
            run.roi_fraction_matrix,
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.5, float(run.roi_fraction_matrix.max())),
            interpolation="nearest",
        )
        for true_id in CLASS_IDS:
            for roi_id in CLASS_IDS:
                value = float(run.roi_fraction_matrix[true_id, roi_id])
                axis.text(
                    roi_id,
                    true_id,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value >= 0.40 else "black",
                )
        axis.set_xticks(CLASS_IDS, [f"ROI {value}" for value in CLASS_NAMES])
        axis.set_yticks(CLASS_IDS, CLASS_NAMES)
        axis.set_xlabel("Detector region")
        axis.set_ylabel("True class")
        axis.set_title(f"Mean four-ROI energy fraction: {run.mask_name}")
        figure_rows.extend(
            _save_figure(
                fig,
                figures_dir,
                f"roi_energy_{run.run_id}",
                width_cm=width_cm,
                height_cm=FIGURE_HEIGHT_CM,
                description="Mean fraction of four-ROI energy by true class and detector ROI.",
                eligible_for_paper=run.eligible_for_formal_comparison,
            )
        )
        plt.close(fig)

    formal_runs = [run for run in runs if run.eligible_for_formal_comparison]
    if formal_runs:
        if len({run.mask_name for run in formal_runs}) != len(formal_runs):
            raise RuntimeError("Each mask may have at most one formal400 run")
        width_cm = min(17.8, max(8.9, 2.1 * len(formal_runs) + 4.5))
        fig, axis = plt.subplots(
            figsize=(width_cm * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
            constrained_layout=True,
        )
        x = np.arange(len(formal_runs), dtype=np.float64)
        accuracies = np.asarray([run.metrics["accuracy"] for run in formal_runs])
        lower = accuracies - np.asarray(
            [run.metrics["accuracy_wilson95_low"] for run in formal_runs]
        )
        upper = np.asarray(
            [run.metrics["accuracy_wilson95_high"] for run in formal_runs]
        ) - accuracies
        axis.errorbar(
            x,
            accuracies,
            yerr=np.stack((lower, upper)),
            fmt="o",
            color=COLORS[0],
            capsize=2.0,
            label="Accuracy (95% Wilson CI)",
        )
        axis.plot(
            x,
            [run.metrics["macro_f1"] for run in formal_runs],
            "s",
            color=COLORS[1],
            linestyle="none",
            label="Macro F1",
        )
        axis.plot(
            x,
            [run.metrics["balanced_accuracy"] for run in formal_runs],
            "^",
            color=COLORS[2],
            linestyle="none",
            label="Balanced accuracy",
        )
        axis.set_xticks(x, [run.mask_name for run in formal_runs], rotation=20, ha="right")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.03)
        axis.set_title("Formal400 mask comparison")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower right")
        figure_rows.extend(
            _save_figure(
                fig,
                figures_dir,
                "formal400_mask_comparison",
                width_cm=width_cm,
                height_cm=FIGURE_HEIGHT_CM,
                description="Formal fixed-400 accuracy with Wilson CI, macro F1, and balanced accuracy.",
                eligible_for_paper=True,
            )
        )
        plt.close(fig)

        fig, axis = plt.subplots(
            figsize=(8.9 * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
            constrained_layout=True,
        )
        for index, run in enumerate(formal_runs):
            values = [row["f1"] for row in run.per_class]
            axis.plot(
                CLASS_IDS,
                values,
                marker="o",
                color=COLORS[index % len(COLORS)],
                label=run.mask_name,
            )
        axis.set_xticks(CLASS_IDS, CLASS_NAMES)
        axis.set_xlabel("MNIST class")
        axis.set_ylabel("F1 score")
        axis.set_ylim(0.0, 1.03)
        axis.set_title("Formal400 class-wise F1")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower right")
        figure_rows.extend(
            _save_figure(
                fig,
                figures_dir,
                "formal400_per_class_f1",
                width_cm=8.9,
                height_cm=FIGURE_HEIGHT_CM,
                description="Per-class F1 for formal fixed-400 evaluations only.",
                eligible_for_paper=True,
            )
        )
        plt.close(fig)

        fig, axis = plt.subplots(
            figsize=(8.9 * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
            constrained_layout=True,
        )
        margins = [
            [sample["normalized_target_margin"] for sample in run.sample_rows]
            for run in formal_runs
        ]
        boxplot_options = {
            "showfliers": False,
            "widths": 0.55,
            "patch_artist": True,
            "medianprops": {"color": "black", "linewidth": 0.8},
            "whiskerprops": {"linewidth": 0.7},
            "capprops": {"linewidth": 0.7},
        }
        try:
            # Matplotlib >=3.9 renamed labels to tick_labels and >=3.11
            # removed the legacy spelling.
            box = axis.boxplot(
                margins,
                tick_labels=[run.mask_name for run in formal_runs],
                **boxplot_options,
            )
        except TypeError:  # Matplotlib 3.7/3.8 laboratory environments
            box = axis.boxplot(
                margins,
                labels=[run.mask_name for run in formal_runs],
                **boxplot_options,
            )
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(COLORS[index % len(COLORS)])
            patch.set_alpha(0.65)
            patch.set_linewidth(0.6)
        axis.axhline(0.0, color="#666666", linewidth=0.6, linestyle="--")
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_ylabel("Normalized target margin")
        axis.set_title("Formal400 raw four-ROI separation")
        axis.spines[["top", "right"]].set_visible(False)
        figure_rows.extend(
            _save_figure(
                fig,
                figures_dir,
                "formal400_energy_margin",
                width_cm=8.9,
                height_cm=FIGURE_HEIGHT_CM,
                description="Target-minus-strongest-wrong ROI energy divided by total four-ROI energy.",
                eligible_for_paper=True,
            )
        )
        plt.close(fig)
    return figure_rows, resolved_font


def evaluate_prediction_runs(
    *,
    runs: list[PredictionRunSpec],
    output_dir: str | Path,
    allow_biased_diagnostic: bool = False,
    make_plots: bool = True,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one prediction run is required")
    if len({run.mask_name for run in runs}) == 0:
        raise ValueError("Mask names cannot be empty")
    loaded = [_load_run(run, allow_biased_diagnostic) for run in runs]
    run_ids = [run.run_id for run in loaded]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("Mask/profile pairs must be unique")
    formal_runs = [run for run in loaded if run.eligible_for_formal_comparison]
    paired = _paired_comparisons(formal_runs)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    run_json: dict[str, Any] = {}
    for run in loaded:
        row = {
            "run_id": run.run_id,
            "mask_name": run.mask_name,
            "profile": run.profile,
            "reporting_status": run.reporting_status,
            "suitable_for_accuracy_reporting": run.suitable_for_accuracy_reporting,
            "eligible_for_formal_comparison": run.eligible_for_formal_comparison,
            "phase_sha256": run.phase_sha256,
            **run.metrics,
            "predictions_csv": str(run.predictions_path),
            "predictions_sha256": _sha256(run.predictions_path),
            "source_metrics_json": None
            if run.metrics_path is None
            else str(run.metrics_path),
            "source_metrics_sha256": None
            if run.metrics_path is None
            else _sha256(run.metrics_path),
        }
        # Only formal400 values are admitted to the paper comparison table.
        row["formal_accuracy"] = (
            run.metrics["accuracy"] if run.eligible_for_formal_comparison else None
        )
        summary_rows.append(row)
        per_class_rows.extend(
            {"run_id": run.run_id, **class_row} for class_row in run.per_class
        )
        row_totals = run.confusion.sum(axis=1)
        for true_id in CLASS_IDS:
            for predicted_id in CLASS_IDS:
                count = int(run.confusion[true_id, predicted_id])
                confusion_rows.append(
                    {
                        "run_id": run.run_id,
                        "mask_name": run.mask_name,
                        "profile": run.profile,
                        "reporting_status": run.reporting_status,
                        "true_class": true_id,
                        "predicted_class": predicted_id,
                        "count": count,
                        "row_fraction": _safe_divide(count, int(row_totals[true_id])),
                        "total_fraction": _safe_divide(count, len(run.labels)),
                    }
                )
        sample_rows.extend(
            {
                "run_id": run.run_id,
                "mask_name": run.mask_name,
                "profile": run.profile,
                "reporting_status": run.reporting_status,
                **sample,
            }
            for sample in run.sample_rows
        )
        run_json[run.run_id] = {
            **row,
            "confusion_matrix": run.confusion.tolist(),
            "per_class": run.per_class,
            "mean_roi_energy_fraction_by_true_class": run.roi_fraction_matrix.tolist(),
        }

    summary_fields = list(summary_rows[0])
    _write_csv(destination / "run_summary.csv", summary_rows, summary_fields)
    formal_rows = [row for row in summary_rows if row["eligible_for_formal_comparison"]]
    _write_csv(
        destination / "formal400_mask_summary.csv",
        formal_rows,
        summary_fields,
    )
    _write_csv(
        destination / "per_class_metrics.csv",
        per_class_rows,
        list(per_class_rows[0]),
    )
    _write_csv(
        destination / "confusion_matrix_long.csv",
        confusion_rows,
        list(confusion_rows[0]),
    )
    _write_csv(
        destination / "sample_energy_metrics.csv",
        sample_rows,
        list(sample_rows[0]),
    )
    paired_fields = (
        list(paired[0])
        if paired
        else [
            "mask_a",
            "mask_b",
            "samples",
            "accuracy_a",
            "accuracy_b",
            "accuracy_b_minus_a",
            "macro_f1_a",
            "macro_f1_b",
            "macro_f1_b_minus_a",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "mcnemar_exact_two_sided_p",
            "mean_paired_normalized_margin_b_minus_a",
        ]
    )
    _write_csv(destination / "paired_formal400_mask_comparison.csv", paired, paired_fields)

    figure_rows: list[dict[str, Any]] = []
    resolved_font: str | None = None
    if make_plots:
        figure_rows, resolved_font = _plot_figures(loaded, destination)
        _write_csv(
            destination / "figure_manifest.csv",
            figure_rows,
            list(figure_rows[0]) if figure_rows else ("figure", "format"),
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "MNIST-4 raw four-ROI optical classification",
        "class_order": list(CLASS_IDS),
        "run_count": len(loaded),
        "formal400_run_count": len(formal_runs),
        "formal_comparison_policy": (
            "only suitable fixed-random runs with exactly 100 samples per class "
            "(400 total) enter aggregate mask comparisons"
        ),
        "quick40_policy": (
            "10 samples per class (40 total) are labelled quick diagnostics and "
            "never enter formal mask comparisons"
        ),
        "prediction_rule": "argmax of four untouched raw ROI sums",
        "ccd_postprocessing_added_by_this_module": False,
        "metrics": {
            "accuracy_interval": "two-sided 95% Wilson score interval",
            "per_class": "one-vs-rest precision, recall and F1",
            "macro_average": "unweighted arithmetic mean over classes 0,1,2,3",
            "weighted_average": "support-weighted mean over classes",
            "paired_comparison": "exact two-sided McNemar test on fixed formal400 keys",
        },
        "figure_style": {
            "font_family_requested": "Arial",
            "font_resolved_path": resolved_font,
            "font_size_pt": 7.0,
            "height_cm": FIGURE_HEIGHT_CM,
            "formats": ["PDF", "SVG", "PNG"],
            "png_dpi": 600,
            "note": (
                "If font_resolved_path is null, install Arial before final paper export; "
                "the SVG still requests Arial explicitly."
            ),
        },
        "runs": run_json,
        "paired_formal400_comparisons": paired,
        "figures": figure_rows,
    }
    _write_json(destination / "paper_metrics.json", report)
    output_inventory: list[dict[str, Any]] = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "output_inventory.json":
            output_inventory.append(
                {
                    "relative_path": path.relative_to(destination).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        destination / "output_inventory.json",
        {
            "schema_version": 1,
            "files": output_inventory,
            "file_count": len(output_inventory),
        },
    )
    return report


def _parse_run_argument(value: str) -> PredictionRunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be MASK_NAME=prediction_csv_or_directory")
    mask_name, raw_path = value.split("=", 1)
    if not mask_name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run requires non-empty mask name and path")
    return PredictionRunSpec(mask_name=mask_name.strip(), predictions_path=Path(raw_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_argument,
        required=True,
        help="Repeatable MASK_NAME=hardware_evaluation_directory_or_predictions.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-biased-diagnostic",
        action="store_true",
        help="Permit demo_topk only as explicitly labelled, non-paper diagnostics",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_prediction_runs(
        runs=args.run,
        output_dir=args.output_dir,
        allow_biased_diagnostic=args.allow_biased_diagnostic,
        make_plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "run_count": report["run_count"],
                "formal400_run_count": report["formal400_run_count"],
                "figure_files": len(report["figures"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
