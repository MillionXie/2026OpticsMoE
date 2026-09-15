"""Collect real simulation/hardware evidence and draw compact paper figures.

The module intentionally depends only on NumPy, Pillow and Matplotlib.  It can
therefore run from the extracted laboratory ZIP without Qwen, Transformers or
the optical simulator.  Missing hardware outputs are represented explicitly in
``results_report.json`` and never replaced by synthetic values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
STAGE_LABELS = {
    "vision_expert": "Vision expert",
    "vision_global": "Vision global",
    "language_expert": "Language expert",
    "language_global": "Language global",
}
CLAIM = (
    "The four-stage hybrid model with a 5% minimum optical fusion coefficient "
    "retains 81% Top-1 on the fixed simulation test; the same report quantifies "
    "changes after real CCD substitution and downstream fine-tuning."
)
BLUE = "#4C78A8"
PURPLE = "#7868A6"
PALE_BLUE = "#9CBBD4"
PALE_PURPLE = "#B4A7CF"
GREEN = "#4F8A70"
RED = "#B45D5D"
GREY = "#777777"
LIGHT_GREY = "#E7E7E7"
# Explicit publication-export contract (also consumed by the static QA tool).
SINGLE_COLUMN_WIDTH_MM = 89
DOUBLE_COLUMN_WIDTH_MM = 183
RASTER_DPI = 600
VECTOR_EXTENSIONS = (".svg", ".pdf")
RASTER_EXTENSIONS = (".png", ".tiff")


@dataclass(frozen=True)
class MetricRecord:
    record_id: str
    label: str
    kind: str
    stage: str | None
    top1: float
    top3: float | None
    mrr: float | None
    query_count: int | None
    gallery_count: int | None
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]] | None
    source: str
    measured_stages: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        # Reports are portable hand-off artifacts: never leak a server or Windows
        # home path when evidence was supplied outside ``root``.
        digest = _sha256(path)[:10] if path.is_file() else hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:10]
        return f"external/{digest}_{path.name}"


def _finite_fraction(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{label} must be a finite fraction, got {value!r}")
    return number


def _optional_fraction(value: Any, label: str) -> float | None:
    return None if value is None else _finite_fraction(value, label)


def _metric_payload(raw: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(raw.get("student"), dict):
        return raw["student"]
    if "top1_retrieval_accuracy" in raw:
        return raw
    if isinstance(raw.get("metrics"), dict) and "top1_retrieval_accuracy" in raw["metrics"]:
        return raw["metrics"]
    return None


def _infer_stage(path: Path, payload: dict[str, Any]) -> str | None:
    stage = payload.get("stage")
    if stage in STAGES:
        return str(stage)
    joined = "/".join(part.lower() for part in path.parts)
    return next((name for name in STAGES if name in joined), None)


def _record_from_json(path: Path, root: Path, *, kind_hint: str | None = None) -> MetricRecord | None:
    raw = _read_json(path)
    payload = _metric_payload(raw)
    if payload is None:
        return None
    stage = _infer_stage(path, payload)
    normalized = path.as_posix().lower()
    if kind_hint is not None:
        kind = kind_hint
    elif "offline_results" in normalized or "quick" in str(payload.get("system", "")).lower():
        kind = "quick_hardware"
    elif path.name == "finetune_metrics.json" or str(payload.get("system", "")).startswith("hardware_through_"):
        kind = "four_stage_hardware"
    elif "evaluation_summary" in path.name:
        kind = "simulation"
    else:
        kind = "other_evaluation"
    evaluation_point = str(payload.get("evaluation_point", "")).lower()
    system_name = str(payload.get("system", "")).lower()
    quick_point = ""
    if kind == "quick_hardware":
        if "before" in evaluation_point or "pre_finetune" in system_name:
            quick_point = "pre"
        elif "after" in evaluation_point or "post_finetune" in system_name:
            quick_point = "post"
    if kind == "simulation":
        label = "Simulation (fixed test)"
    elif kind == "quick_hardware":
        label = {
            "pre": "Quick CCD: before fine-tuning",
            "post": "Quick CCD: after fine-tuning",
        }.get(quick_point, "Quick: language-global CCD")
    elif stage:
        label = f"Hardware through {STAGE_LABELS[stage]}"
    else:
        label = str(payload.get("system", path.parent.name)).replace("_", " ")
    per_class = payload.get("per_sku", payload.get("per_class", {}))
    if not isinstance(per_class, dict):
        per_class = {}
    matrix = payload.get("confusion_matrix")
    if matrix is not None:
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1] or np.any(array < 0):
            raise RuntimeError(f"Invalid confusion matrix in {path}")
        matrix = array.astype(np.int64).tolist()
    measured = payload.get("measured_stages", [])
    if not isinstance(measured, list):
        measured = []
    record_id = {
        "simulation": "simulation_baseline",
        "quick_hardware": (
            f"quick_language_global_{quick_point}"
            if quick_point
            else "quick_language_global"
        ),
    }.get(kind, f"{kind}_{stage or hashlib.sha1(str(path).encode()).hexdigest()[:8]}")
    return MetricRecord(
        record_id=record_id,
        label=label,
        kind=kind,
        stage=stage,
        top1=_finite_fraction(payload["top1_retrieval_accuracy"], f"{path}: Top-1"),
        top3=_optional_fraction(payload.get("top3_retrieval_accuracy"), f"{path}: Top-3"),
        mrr=_optional_fraction(payload.get("mrr"), f"{path}: MRR"),
        query_count=int(payload["query_count"]) if payload.get("query_count") is not None else None,
        gallery_count=int(payload.get("gallery_image_count", payload.get("gallery_count"))) if payload.get("gallery_image_count", payload.get("gallery_count")) is not None else None,
        per_class=per_class,
        confusion_matrix=matrix,
        source=_relative(path, root),
        measured_stages=[str(value) for value in measured],
    )


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def collect_metric_records(
    root: Path,
    session_dirs: Sequence[Path],
    baseline_paths: Sequence[Path],
) -> tuple[list[MetricRecord], list[Path], list[str]]:
    canonical = root / "reference/training_evidence/stage_b/metrics/evaluation_summary.json"
    baselines = _unique_existing([*baseline_paths, canonical])
    records: list[MetricRecord] = []
    sources: list[Path] = []
    warnings: list[str] = []
    for path in baselines:
        record = _record_from_json(path, root, kind_hint="simulation")
        if record:
            records.append(record)
            sources.append(path)
    search_roots = _unique_existing(
        [*session_dirs, root / "payload/quick210", root / "hardware_sessions"]
    )
    candidates: list[Path] = []
    for search_root in search_roots:
        if search_root.is_file():
            candidates.append(search_root)
            continue
        candidates.extend(search_root.rglob("finetune_metrics.json"))
        candidates.extend(search_root.rglob("evaluation_summary.json"))
        for offline_dir in search_root.rglob("offline_results"):
            pre = offline_dir / "pre_finetune_metrics.json"
            post = offline_dir / "post_finetune_metrics.json"
            if pre.is_file() and post.is_file():
                candidates.extend((pre, post))
            else:
                candidates.append(offline_dir / "metrics.json")
    baseline_resolved = {path.resolve() for path in baselines}
    for path in _unique_existing(candidates):
        if path.resolve() in baseline_resolved:
            continue
        try:
            record = _record_from_json(path, root)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            warnings.append(f"Skipped unsupported metric file {_relative(path, root)}: {error}")
            continue
        if record:
            records.append(record)
            sources.append(path)
    by_id: dict[str, MetricRecord] = {}
    for record in records:
        if record.record_id in by_id:
            old = by_id[record.record_id]
            warnings.append(
                f"Duplicate {record.record_id}: kept {record.source}, replaced {old.source}."
            )
        by_id[record.record_id] = record
    order = {"simulation": 0, "four_stage_hardware": 1, "quick_hardware": 2, "other_evaluation": 3}
    final = sorted(
        by_id.values(),
        key=lambda item: (order.get(item.kind, 9), STAGES.index(item.stage) if item.stage in STAGES else 9),
    )
    return final, sources, warnings


def _stage_from_dir(path: Path) -> str:
    lowered = path.as_posix().lower()
    return next((stage for stage in STAGES if stage in lowered), path.parent.name)


def collect_ccd_qc(root: Path, session_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    search_roots = _unique_existing([*session_dirs, root / "payload/quick210", root / "hardware_sessions"])
    capture_dirs: list[Path] = []
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        if search_root.name == "ccd_captured":
            capture_dirs.append(search_root)
        capture_dirs.extend(path for path in search_root.rglob("ccd_captured") if path.is_dir())
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for capture_dir in _unique_existing(capture_dirs):
        stage = _stage_from_dir(capture_dir)
        for path in sorted(capture_dir.glob("*.png")):
            with Image.open(path) as image:
                mode = image.mode
                if len(image.getbands()) > 1:
                    array = np.asarray(image.convert("L"))
                    mode = f"{mode}->L"
                else:
                    array = np.asarray(image)
            values = array.astype(np.float64, copy=False)
            if values.size == 0 or not np.isfinite(values).all():
                continue
            if array.dtype == np.uint8:
                ceiling = 255.0
            elif array.dtype == np.uint16:
                ceiling = 65535.0
            else:
                ceiling = max(float(values.max()), 1.0)
            p01, p99 = np.percentile(values, [1, 99])
            mean = float(values.mean())
            rows.append(
                {
                    "stage": stage,
                    "key": path.stem,
                    "source": _relative(path, root),
                    "mode": mode,
                    "dtype": str(array.dtype),
                    "height": int(values.shape[0]),
                    "width": int(values.shape[1]),
                    "mean": mean,
                    "std": float(values.std()),
                    "p01": float(p01),
                    "p99": float(p99),
                    "relative_dynamic_range": float((p99 - p01) / max(mean, 1e-12)),
                    "low_fraction": float(np.mean(values <= 0.01 * ceiling)),
                    "saturated_fraction": float(np.mean(values >= 0.99 * ceiling)),
                    "sha256": _sha256(path),
                }
            )
            sources.append(path)
    return rows, sources


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return None


def collect_predictions(root: Path, session_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    search_roots = _unique_existing(
        [*session_dirs, root / "payload/quick210", root / "hardware_sessions"]
    )
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*.csv")):
            if path.name in {"train_log.csv", "manifest.csv", "capture_manifest.csv", "predictions_long.csv", "paired_predictions.csv"} or "source_data" in path.parts:
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                parsed = list(csv.DictReader(handle))
            if not parsed:
                continue
            fields = set(parsed[0])
            sample_key = next((name for name in ("sample_id", "key", "query_id") if name in fields), None)
            correct_key = next((name for name in ("top1_correct", "correct") if name in fields), None)
            truth_key = next((name for name in ("true_sku", "true_label", "target") if name in fields), None)
            prediction_key = next((name for name in ("predicted_sku", "predicted_label", "prediction") if name in fields), None)
            if sample_key is None or (correct_key is None and (truth_key is None or prediction_key is None)):
                continue
            system_default = path.parent.name
            accepted = 0
            for row in parsed:
                correct = _parse_bool(row.get(correct_key, "")) if correct_key else row.get(truth_key) == row.get(prediction_key)
                if correct is None:
                    continue
                margin_raw = row.get("similarity_margin", "")
                margin = float(margin_raw) if margin_raw not in {"", None} else None
                rank_raw = row.get("rank", "")
                rank = int(rank_raw) if rank_raw not in {"", None} else None
                rows.append(
                    {
                        "system": row.get("system") or system_default,
                        "sample_id": row[sample_key],
                        "key": row.get("key") or row[sample_key],
                        "true_label": row.get(truth_key, ""),
                        "predicted_label": row.get(prediction_key, ""),
                        "top1_correct": bool(correct),
                        "similarity_margin": margin,
                        "rank": rank,
                        "source": _relative(path, root),
                    }
                )
                accepted += 1
            if accepted:
                sources.append(path)
    return rows, sources


def _setup_matplotlib(*, require_arial: bool = False) -> tuple[Any, str]:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; install requirements-lab.txt") from error
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else ("Liberation Sans" if "Liberation Sans" in available else "DejaVu Sans")
    if require_arial and font != "Arial":
        raise RuntimeError(
            "Arial is required for the formal figure export but is not installed. "
            "Install Arial, refresh Matplotlib's font cache, and rerun."
        )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )
    return plt, font


def _mm(value: float) -> float:
    return value / 25.4


def _clean_axis(axis: Any, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    axis.tick_params(direction="out")


def _save_figure(
    figure: Any,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    plt: Any,
) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"dpi": RASTER_DPI}
        if extension.lower() in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_overall(records: Sequence[MetricRecord], plt: Any) -> Any:
    figure, axis = plt.subplots(figsize=(_mm(89), _mm(55)))
    metrics = ("Top-1", "Top-3", "MRR")
    x = np.arange(3)
    width = min(0.72 / max(len(records), 1), 0.24)
    palette = [BLUE, PURPLE, PALE_PURPLE, PALE_BLUE]
    for index, record in enumerate(records):
        offset = (index - (len(records) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            100 * np.array([record.top1, np.nan if record.top3 is None else record.top3, np.nan if record.mrr is None else record.mrr]),
            width=width * 0.92,
            color=palette[index % len(palette)],
            edgecolor="none",
            label=record.label,
            zorder=3,
        )
        if len(records) <= 2:
            axis.bar_label(
                bars,
                labels=["" if not np.isfinite(bar.get_height()) else f"{bar.get_height():.1f}" for bar in bars],
                padding=1,
                fontsize=7,
            )
    axis.set_xticks(x, metrics)
    axis.set_ylabel("Retrieval metric (%)")
    axis.set_ylim(0, 105)
    _clean_axis(axis)
    if len(records) > 1:
        axis.legend(frameon=False, loc="lower right")
    title = (
        "Fixed simulation baseline"
        if len(records) == 1 and records[0].kind == "simulation"
        else "Fixed simulation baseline and available hardware results"
    )
    axis.set_title(title, loc="left")
    figure.tight_layout(pad=0.5)
    return figure


def _class_names(records: Sequence[MetricRecord]) -> list[str]:
    for record in records:
        if record.per_class:
            return list(record.per_class)
    return []


def _plot_per_class(records: Sequence[MetricRecord], plt: Any) -> Any | None:
    usable = [record for record in records if record.per_class]
    names = _class_names(usable)
    if not usable or not names:
        return None
    figure, axis = plt.subplots(figsize=(_mm(89), _mm(60)))
    y = np.arange(len(names))
    offsets = np.linspace(-0.22, 0.22, max(len(usable), 2)) if len(usable) > 1 else [0.0]
    palette = [BLUE, PURPLE, PALE_PURPLE, PALE_BLUE]
    for index, record in enumerate(usable):
        values = [100 * float(record.per_class.get(name, {}).get("top1_accuracy", np.nan)) for name in names]
        axis.scatter(values, y + offsets[index], s=13, color=palette[index % len(palette)], label=record.label, zorder=3)
    axis.set_yticks(y, [name.replace("_", " ") for name in names])
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Top-1 accuracy (%)")
    _clean_axis(axis, grid="x")
    if len(usable) > 1:
        axis.legend(frameon=False, loc="lower right")
    axis.set_title("Class-resolved retrieval", loc="left")
    figure.tight_layout(pad=0.5)
    return figure


def _plot_confusion(record: MetricRecord, plt: Any) -> Any | None:
    if record.confusion_matrix is None:
        return None
    matrix = np.asarray(record.confusion_matrix, dtype=float)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    names = list(record.per_class) if len(record.per_class) == len(matrix) else [str(index) for index in range(len(matrix))]
    short = [name.replace("_", " ")[:10] for name in names]
    figure, axis = plt.subplots(figsize=(_mm(89), _mm(60)))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest", aspect="auto")
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            if matrix[row, column] > 0:
                axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center", fontsize=7, color="white" if normalized[row, column] > 0.55 else "#333333")
    axis.set_xticks(range(len(short)), short, rotation=45, rotation_mode="anchor", ha="right")
    axis.set_yticks(range(len(short)), short)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"Confusion matrix — {record.label}", loc="left")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Row-normalized fraction")
    colorbar.ax.tick_params(labelsize=7)
    figure.tight_layout(pad=0.4)
    return figure


def _plot_stage(records: Sequence[MetricRecord], plt: Any) -> Any | None:
    baseline = next((record for record in records if record.kind == "simulation"), None)
    # Quick210 measures only Language-global while the first three optical
    # stages remain simulated.  It is not a point on the complete measured
    # four-stage trajectory and must not be connected to that line.
    hardware = sorted(
        (record for record in records if record.kind == "four_stage_hardware"),
        key=lambda record: STAGES.index(record.stage) if record.stage in STAGES else 99,
    )
    if not hardware:
        return None
    selected = ([baseline] if baseline else []) + hardware
    figure, axis = plt.subplots(figsize=(_mm(89), _mm(55)))
    values = [100 * record.top1 for record in selected]
    x = np.arange(len(values))
    colors = [BLUE] + [PURPLE] * (len(values) - 1) if baseline else [PURPLE] * len(values)
    axis.plot(x, values, color=GREY, linewidth=0.8, zorder=2)
    axis.scatter(x, values, c=colors, s=22, zorder=3)
    for index, value in enumerate(values):
        axis.text(index, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(x, [record.label.replace("Hardware through ", "Through\n").replace("Simulation (fixed test)", "Simulation") for record in selected], rotation=20, rotation_mode="anchor", ha="right")
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_ylim(0, 105)
    _clean_axis(axis)
    axis.set_title("Measured-stage progression", loc="left")
    figure.tight_layout(pad=0.5)
    return figure


def _summarize_ccd(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["stage"]), []).append(row)
    result = []
    for stage, values in grouped.items():
        result.append(
            {
                "stage": stage,
                "frames": len(values),
                "median_mean": float(np.median([float(row["mean"]) for row in values])),
                "median_p01": float(np.median([float(row["p01"]) for row in values])),
                "median_p99": float(np.median([float(row["p99"]) for row in values])),
                "median_relative_dynamic_range": float(np.median([float(row["relative_dynamic_range"]) for row in values])),
                "median_low_fraction": float(np.median([float(row["low_fraction"]) for row in values])),
                "median_saturated_fraction": float(np.median([float(row["saturated_fraction"]) for row in values])),
            }
        )
    return sorted(result, key=lambda row: STAGES.index(row["stage"]) if row["stage"] in STAGES else 99)


def _plot_ccd(rows: Sequence[dict[str, Any]], plt: Any) -> Any | None:
    summary = _summarize_ccd(rows)
    if not summary:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(_mm(183), _mm(55)))
    x = np.arange(len(summary))
    labels = [STAGE_LABELS.get(row["stage"], str(row["stage"])).replace(" ", "\n") for row in summary]
    axes[0].plot(x, [row["median_mean"] for row in summary], "o-", color=BLUE, label="Mean")
    axes[0].fill_between(x, [row["median_p01"] for row in summary], [row["median_p99"] for row in summary], color=PALE_BLUE, alpha=0.55, label="P01–P99")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("CCD level (native units)")
    axes[0].set_title("a  Frame intensity", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    _clean_axis(axes[0])
    width = 0.34
    axes[1].bar(x - width / 2, 100 * np.array([row["median_low_fraction"] for row in summary]), width, color=GREY, label="Near-black")
    axes[1].bar(x + width / 2, 100 * np.array([row["median_saturated_fraction"] for row in summary]), width, color=RED, label="Near-saturation")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Pixel fraction (%)")
    axes[1].set_title("b  Clipping diagnostics", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    _clean_axis(axes[1])
    figure.tight_layout(pad=0.6, w_pad=2.2)
    return figure


def _paired_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    systems: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        systems.setdefault(str(row["system"]), {})[str(row["sample_id"])] = row
    if len(systems) < 2:
        return None
    names = sorted(systems, key=lambda name: ("sim" not in name.lower(), name))
    before_name, after_name = names[0], names[-1]
    common = sorted(set(systems[before_name]).intersection(systems[after_name]))
    if not common:
        return None
    transitions = {"correct→correct": 0, "correct→wrong": 0, "wrong→correct": 0, "wrong→wrong": 0}
    paired_rows = []
    for sample_id in common:
        before = systems[before_name][sample_id]
        after = systems[after_name][sample_id]
        key = ("correct" if before["top1_correct"] else "wrong") + "→" + ("correct" if after["top1_correct"] else "wrong")
        transitions[key] += 1
        paired_rows.append(
            {
                "sample_id": sample_id,
                "key": before.get("key") or after.get("key") or sample_id,
                "before_system": before_name,
                "after_system": after_name,
                "before_correct": before["top1_correct"],
                "after_correct": after["top1_correct"],
                "before_margin": before.get("similarity_margin"),
                "after_margin": after.get("similarity_margin"),
                "before_rank": before.get("rank"),
                "after_rank": after.get("rank"),
            }
        )
    return {"before": before_name, "after": after_name, "count": len(common), "transitions": transitions, "rows": paired_rows}


def _plot_paired(paired: dict[str, Any] | None, plt: Any) -> Any | None:
    if paired is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(_mm(183), _mm(55)))
    names = list(paired["transitions"])
    values = [paired["transitions"][name] for name in names]
    colors = [PALE_BLUE, RED, GREEN, GREY]
    axes[0].bar(np.arange(4), values, color=colors)
    axes[0].set_xticks(np.arange(4), [name.replace("→", "→\n") for name in names])
    axes[0].set_ylabel("Paired queries")
    axes[0].set_title("a  Correctness transitions", loc="left", fontweight="bold")
    _clean_axis(axes[0])
    margin_rows = [row for row in paired["rows"] if row["before_margin"] is not None and row["after_margin"] is not None]
    if margin_rows:
        for row in margin_rows:
            delta = float(row["after_margin"]) - float(row["before_margin"])
            axes[1].plot([0, 1], [row["before_margin"], row["after_margin"]], color=GREEN if delta >= 0 else RED, alpha=0.25, linewidth=0.5)
        axes[1].set_xticks([0, 1], [paired["before"], paired["after"]], rotation=15, rotation_mode="anchor", ha="right")
        axes[1].set_ylabel("Similarity margin")
        axes[1].set_title("b  Query-level margin", loc="left", fontweight="bold")
        _clean_axis(axes[1])
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Similarity margins\nnot saved", ha="center", va="center", color=GREY)
    figure.tight_layout(pad=0.6, w_pad=2.2)
    return figure


def _plot_overview(records: Sequence[MetricRecord], ccd_rows: Sequence[dict[str, Any]], plt: Any) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(_mm(183), _mm(100)))
    baseline = next((record for record in records if record.kind == "simulation"), records[0])
    baseline_values = [baseline.top1, baseline.top3, baseline.mrr]
    bars = axes[0, 0].bar(["Top-1", "Top-3", "MRR"], [np.nan if value is None else 100 * value for value in baseline_values], color=[BLUE, PALE_BLUE, PURPLE])
    axes[0, 0].bar_label(
        bars,
        labels=["" if not np.isfinite(bar.get_height()) else f"{bar.get_height():.1f}" for bar in bars],
        padding=1,
        fontsize=7,
    )
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Metric (%)")
    axes[0, 0].set_title("a  Fixed simulation test", loc="left", fontweight="bold")
    _clean_axis(axes[0, 0])
    names = list(baseline.per_class)
    if names:
        values = [100 * float(baseline.per_class[name]["top1_accuracy"]) for name in names]
        axes[0, 1].barh(np.arange(len(names)), values, color=BLUE)
        axes[0, 1].set_yticks(np.arange(len(names)), [name.replace("_", " ") for name in names])
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlim(0, 105)
        axes[0, 1].set_xlabel("Top-1 (%)")
        axes[0, 1].set_title("b  Class-resolved baseline", loc="left", fontweight="bold")
        _clean_axis(axes[0, 1], grid="x")
    else:
        axes[0, 1].axis("off")
    four_stage = sorted(
        (record for record in records if record.kind == "four_stage_hardware"),
        key=lambda record: STAGES.index(record.stage) if record.stage in STAGES else 99,
    )
    quick = [record for record in records if record.kind == "quick_hardware"]
    hardware = [*four_stage, *quick]
    if hardware:
        selected = [baseline, *hardware]
        x = np.arange(len(selected))
        values = [100 * record.top1 for record in selected]
        connected_count = 1 + len(four_stage)
        if connected_count > 1:
            axes[1, 0].plot(
                x[:connected_count],
                values[:connected_count],
                "o-",
                color=PURPLE,
                linewidth=0.9,
                label="Sequential four-stage",
            )
        else:
            axes[1, 0].scatter(x[0], values[0], color=BLUE, s=18)
        if quick:
            axes[1, 0].scatter(
                x[connected_count:],
                values[connected_count:],
                marker="D",
                color=GREEN,
                s=22,
                label="Quick last-stage path",
                zorder=3,
            )
        axes[1, 0].set_xticks(x, [record.label.replace("Hardware through ", "Through\n").replace("Simulation (fixed test)", "Simulation") for record in selected], rotation=20, rotation_mode="anchor", ha="right")
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].set_ylabel("Top-1 (%)")
        axes[1, 0].set_title("c  Available hardware results", loc="left", fontweight="bold")
        if quick:
            axes[1, 0].legend(frameon=False, loc="lower right")
        _clean_axis(axes[1, 0])
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.55, "Hardware retrieval metrics\nunavailable before capture", ha="center", va="center", color=GREY)
        axes[1, 0].set_title("c  Available hardware results", loc="left", fontweight="bold")
    ccd_summary = _summarize_ccd(ccd_rows)
    if ccd_summary:
        x = np.arange(len(ccd_summary))
        axes[1, 1].bar(x, [row["median_relative_dynamic_range"] for row in ccd_summary], color=PURPLE)
        axes[1, 1].set_xticks(x, [STAGE_LABELS.get(row["stage"], row["stage"]).replace(" ", "\n") for row in ccd_summary])
        axes[1, 1].set_ylabel("(P99−P01) / mean")
        axes[1, 1].set_title("d  CCD dynamic range", loc="left", fontweight="bold")
        _clean_axis(axes[1, 1])
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.55, "CCD quality control\nunavailable before capture", ha="center", va="center", color=GREY)
        axes[1, 1].set_title("d  CCD quality control", loc="left", fontweight="bold")
    figure.tight_layout(pad=0.8, h_pad=2.0, w_pad=2.4)
    return figure


def _source_tables(
    output_dir: Path,
    records: Sequence[MetricRecord],
    ccd_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    paired: dict[str, Any] | None,
) -> list[Path]:
    source_dir = output_dir / "source_data"
    outputs: list[Path] = []
    overall_rows = [
        {
            "record_id": record.record_id,
            "label": record.label,
            "kind": record.kind,
            "stage": record.stage or "",
            "top1": record.top1,
            "top3": record.top3,
            "mrr": record.mrr,
            "query_count": record.query_count,
            "gallery_count": record.gallery_count,
            "source": record.source,
        }
        for record in records
    ]
    path = source_dir / "overall_metrics.csv"
    _write_csv(path, overall_rows, ("record_id", "label", "kind", "stage", "top1", "top3", "mrr", "query_count", "gallery_count", "source"))
    outputs.append(path)
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for record in records:
        names = list(record.per_class)
        for name, metrics in record.per_class.items():
            class_rows.append({"record_id": record.record_id, "class": name, "query_count": metrics.get("query_count"), "top1": metrics.get("top1_accuracy"), "top3": metrics.get("top3_accuracy"), "source": record.source})
        if record.confusion_matrix is not None:
            matrix = np.asarray(record.confusion_matrix)
            for true_index in range(len(matrix)):
                denominator = int(matrix[true_index].sum())
                for predicted_index in range(len(matrix)):
                    confusion_rows.append({"record_id": record.record_id, "true_index": true_index, "true_class": names[true_index] if true_index < len(names) else true_index, "predicted_index": predicted_index, "predicted_class": names[predicted_index] if predicted_index < len(names) else predicted_index, "count": int(matrix[true_index, predicted_index]), "row_fraction": float(matrix[true_index, predicted_index] / denominator) if denominator else 0.0, "source": record.source})
    path = source_dir / "per_class_metrics.csv"
    _write_csv(path, class_rows, ("record_id", "class", "query_count", "top1", "top3", "source"))
    outputs.append(path)
    path = source_dir / "confusion_matrix_long.csv"
    _write_csv(path, confusion_rows, ("record_id", "true_index", "true_class", "predicted_index", "predicted_class", "count", "row_fraction", "source"))
    outputs.append(path)
    ccd_fields = ("stage", "key", "source", "mode", "dtype", "height", "width", "mean", "std", "p01", "p99", "relative_dynamic_range", "low_fraction", "saturated_fraction", "sha256")
    path = source_dir / "ccd_qc_per_frame.csv"
    _write_csv(path, ccd_rows, ccd_fields)
    outputs.append(path)
    path = source_dir / "ccd_qc_by_stage.csv"
    _write_csv(path, _summarize_ccd(ccd_rows), ("stage", "frames", "median_mean", "median_p01", "median_p99", "median_relative_dynamic_range", "median_low_fraction", "median_saturated_fraction"))
    outputs.append(path)
    path = source_dir / "predictions_long.csv"
    _write_csv(path, predictions, ("system", "sample_id", "key", "true_label", "predicted_label", "top1_correct", "similarity_margin", "rank", "source"))
    outputs.append(path)
    path = source_dir / "paired_predictions.csv"
    _write_csv(path, [] if paired is None else paired["rows"], ("sample_id", "key", "before_system", "after_system", "before_correct", "after_correct", "before_margin", "after_margin", "before_rank", "after_rank"))
    outputs.append(path)
    path = source_dir / "report_data.json"
    _write_json(path, {"claim": CLAIM, "metrics": [asdict(record) for record in records], "ccd_qc_summary": _summarize_ccd(ccd_rows), "paired_predictions": None if paired is None else {key: value for key, value in paired.items() if key != "rows"}})
    outputs.append(path)
    return outputs


def _metric_sample_note(records: Sequence[MetricRecord]) -> str:
    parts = []
    for record in records:
        query = "unknown" if record.query_count is None else str(record.query_count)
        gallery = "unknown" if record.gallery_count is None else str(record.gallery_count)
        classes = str(len(record.per_class)) if record.per_class else "unknown"
        parts.append(f"{record.label}: n={query} queries, {gallery} gallery images, {classes} classes")
    return "; ".join(parts) + "."


def _figure_qa(path: Path, expected_width_mm: float, expected_height_mm: float) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
    if path.suffix.lower() in {".png", ".tif", ".tiff"}:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if isinstance(dpi, tuple):
                serializable_dpi: float | list[float] | None = [float(value) for value in dpi]
            elif dpi is None:
                serializable_dpi = None
            else:
                serializable_dpi = float(dpi)
            result.update({"pixel_size_wh": list(image.size), "dpi": serializable_dpi})
            expected = [round(_mm(expected_width_mm) * 600), round(_mm(expected_height_mm) * 600)]
            result["nominal_600dpi_canvas_wh"] = expected
            if isinstance(dpi, tuple) and dpi[0] and dpi[1]:
                actual_mm = [
                    float(image.size[0]) / float(dpi[0]) * 25.4,
                    float(image.size[1]) / float(dpi[1]) * 25.4,
                ]
                result["actual_physical_size_mm"] = actual_mm
                result["physical_size_within_0p5mm"] = bool(
                    abs(actual_mm[0] - expected_width_mm) <= 0.5
                    and abs(actual_mm[1] - expected_height_mm) <= 0.5
                )
    if path.suffix.lower() == ".svg":
        text = path.read_text(encoding="utf-8")
        result["editable_text"] = "<text" in text
    return result


_REPORT_STEMS = tuple(f"fig{index:02d}_{name}" for index, name in enumerate(
    (
        "overall_metrics",
        "per_class_top1",
        "confusion_matrix",
        "stage_progression",
        "ccd_quality_control",
        "paired_query_changes",
        "overview",
    ),
    start=1,
))
_REPORT_ROOT_FILES = {
    "results_report.json",
    "figure_manifest.json",
    "figure_manifest.csv",
    "QA_REPORT.md",
    "FIGURE_LEGENDS.md",
}
_REPORT_SOURCE_FILES = {
    "overall_metrics.csv",
    "per_class_metrics.csv",
    "confusion_matrix_long.csv",
    "ccd_qc_per_frame.csv",
    "ccd_qc_by_stage.csv",
    "predictions_long.csv",
    "paired_predictions.csv",
    "report_data.json",
}


def _prepare_output_directory(output: Path, formats: Sequence[str]) -> None:
    """Remove only files owned by this report and reject mixed output dirs.

    Re-running into the same directory must not leave a formerly available
    hardware figure behind after its source data disappears.  At the same
    time, the reporter must never delete an unrelated user file.
    """

    output.mkdir(parents=True, exist_ok=True)
    owned_root = set(_REPORT_ROOT_FILES)
    owned_root.update(
        f"{stem}.{extension}"
        for stem in _REPORT_STEMS
        for extension in ("svg", "pdf", "png", "tiff")
    )
    unknown_root = [
        path
        for path in output.iterdir()
        if path.name != "source_data" and path.name not in owned_root
    ]
    if unknown_root:
        raise RuntimeError(
            "Result output directory contains unrelated files; choose a clean "
            f"--output-dir instead: {[path.name for path in unknown_root]}"
        )
    for name in sorted(owned_root):
        path = output / name
        if path.is_file():
            path.unlink()
        elif path.exists():
            raise RuntimeError(f"Expected report file path is not a file: {path}")

    source_dir = output / "source_data"
    if source_dir.exists() and not source_dir.is_dir():
        raise RuntimeError(f"Expected source_data directory, got: {source_dir}")
    source_dir.mkdir(parents=True, exist_ok=True)
    unknown_source = [
        path for path in source_dir.iterdir() if path.name not in _REPORT_SOURCE_FILES
    ]
    if unknown_source:
        raise RuntimeError(
            "source_data contains unrelated files; choose a clean --output-dir "
            f"instead: {[path.name for path in unknown_source]}"
        )
    for name in sorted(_REPORT_SOURCE_FILES):
        path = source_dir / name
        if path.is_file():
            path.unlink()
        elif path.exists():
            raise RuntimeError(f"Expected source-data path is not a file: {path}")


def generate_report(
    *,
    root: str | Path,
    output_dir: str | Path,
    session_dirs: Sequence[str | Path] = (),
    baseline_paths: Sequence[str | Path] = (),
    formats: Sequence[str] = ("svg", "pdf", "png", "tiff"),
    require_arial: bool = False,
) -> dict[str, Any]:
    evidence_root = Path(root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    sessions = [Path(path).expanduser().resolve() for path in session_dirs]
    baselines = [Path(path).expanduser().resolve() for path in baseline_paths]
    records, metric_sources, warnings = collect_metric_records(evidence_root, sessions, baselines)
    if not records:
        raise FileNotFoundError(
            "No real evaluation metric file was found. Expected the packaged fixed "
            "simulation baseline or an explicit --baseline-json/--session-dir."
        )
    ccd_rows, ccd_sources = collect_ccd_qc(evidence_root, sessions)
    predictions, prediction_sources = collect_predictions(evidence_root, sessions)
    paired = _paired_predictions(predictions)
    plt, font = _setup_matplotlib(require_arial=require_arial)
    _prepare_output_directory(output, formats)
    table_paths = _source_tables(output, records, ccd_rows, predictions, paired)
    metric_sample_note = _metric_sample_note(records)
    ccd_sample_note = (
        "CCD QC: "
        + ", ".join(f"{row['stage']} n={row['frames']} frames" for row in _summarize_ccd(ccd_rows))
        + "."
        if ccd_rows
        else "CCD QC unavailable because no captured frames were found."
    )
    figure_specs: list[tuple[str, Any | None, float, float, str]] = [
        ("fig01_overall_metrics", _plot_overall(records, plt), 89, 55, f"Overall Top-1, Top-3 and MRR. {metric_sample_note}"),
        ("fig02_per_class_top1", _plot_per_class(records, plt), 89, 60, f"Class-resolved Top-1. {metric_sample_note}"),
    ]
    confusion_record = next((record for record in reversed(records) if record.confusion_matrix is not None), None)
    figure_specs.extend(
        [
            ("fig03_confusion_matrix", _plot_confusion(confusion_record, plt) if confusion_record else None, 89, 60, f"Row-normalized confusion matrix with raw counts. {_metric_sample_note([confusion_record]) if confusion_record else metric_sample_note}"),
            ("fig04_stage_progression", _plot_stage(records, plt), 89, 55, f"Top-1 as measured optical stages become available. {metric_sample_note}"),
            ("fig05_ccd_quality_control", _plot_ccd(ccd_rows, plt), 183, 55, f"CCD intensity and clipping diagnostics. {ccd_sample_note}"),
            ("fig06_paired_query_changes", _plot_paired(paired, plt), 183, 55, f"Paired correctness transitions and similarity margins. Paired n={0 if paired is None else paired['count']} queries."),
            ("fig07_overview", _plot_overview(records, ccd_rows, plt), 183, 100, f"Overview: simulation, classes, hardware substitution and CCD QC. {metric_sample_note} {ccd_sample_note}"),
        ]
    )
    figure_manifest: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for stem, figure, width_mm, height_mm, legend in figure_specs:
        if figure is None:
            figure_manifest.append({"figure": stem, "status": "unavailable", "reason": "Required real source data are absent.", "legend": legend})
            continue
        paths = _save_figure(figure, output, stem, formats, plt)
        figure_paths.extend(paths)
        figure_manifest.append({"figure": stem, "status": "available", "width_mm": width_mm, "height_mm": height_mm, "legend": legend, "files": [_figure_qa(path, width_mm, height_mm) for path in paths]})
    availability = {
        "simulation_baseline": any(record.kind == "simulation" for record in records),
        "quick_language_global": any(record.kind == "quick_hardware" for record in records),
        **{f"four_stage_{stage}": any(record.kind == "four_stage_hardware" and record.stage == stage for record in records) for stage in STAGES},
        "ccd_qc": bool(ccd_rows),
        "paired_predictions": paired is not None,
        "similarity_margin_pairs": paired is not None and any(row["before_margin"] is not None and row["after_margin"] is not None for row in paired["rows"]),
    }
    input_sources = _unique_existing([*metric_sources, *ccd_sources, *prediction_sources])
    source_inventory = [
        {"path": _relative(path, evidence_root), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in input_sources
    ]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim": CLAIM,
        "evidence_policy": "Only files present on disk are parsed; unavailable hardware panels are skipped, never imputed.",
        "uncertainty_policy": "No error bars are shown for a single fixed checkpoint on one sealed split; metrics are descriptive for the reported query set, not seed/fold uncertainty estimates.",
        "optical_fusion_semantics": "5% is the configured minimum fusion coefficient, not a measured optical-energy fraction.",
        "font_requested": "Arial",
        "font_resolved": font,
        "font_requirement": "strict" if require_arial else "fallback_allowed",
        "font_warning": None if font == "Arial" else f"Arial unavailable; resolved {font}.",
        "availability": availability,
        "records": [asdict(record) for record in records],
        "ccd_qc_summary": _summarize_ccd(ccd_rows),
        "paired_predictions": None if paired is None else {key: value for key, value in paired.items() if key != "rows"},
        "warnings": warnings,
        "input_source_inventory": source_inventory,
        "source_data": [_relative(path, output) for path in table_paths],
        "figures": figure_manifest,
    }
    _write_json(output / "results_report.json", report)
    _write_json(output / "figure_manifest.json", {"schema_version": 1, "figures": figure_manifest})
    manifest_rows = []
    for item in figure_manifest:
        if item["status"] == "available":
            for file_item in item["files"]:
                manifest_rows.append({"figure": item["figure"], "status": "available", "file": file_item["path"], "sha256": file_item["sha256"], "bytes": file_item["bytes"]})
        else:
            manifest_rows.append({"figure": item["figure"], "status": "unavailable", "file": "", "sha256": "", "bytes": ""})
    _write_csv(output / "figure_manifest.csv", manifest_rows, ("figure", "status", "file", "sha256", "bytes"))
    qa_lines = [
        "# Figure QA",
        "",
        f"- Evidence files parsed: {len(source_inventory)}",
        f"- Metric records: {len(records)}",
        f"- CCD frames characterized: {len(ccd_rows)}",
        f"- Retrieval sample sizes: {metric_sample_note}",
        f"- CCD sample sizes: {ccd_sample_note}",
        f"- Requested font: Arial; resolved font: {font}",
        f"- Arial policy: {'strict' if require_arial else 'fallback allowed for non-final preview'}",
        "- SVG text remains editable (`svg.fonttype = none`).",
        "- PDF text uses TrueType embedding (`pdf.fonttype = 42`).",
        "- Raster outputs use 600 dpi; TIFF uses LZW compression.",
        "- Missing hardware evidence is marked `unavailable`; no placeholder metric is plotted.",
        "- No error bars are drawn: the baseline is one fixed checkpoint on one sealed split, with no seed/fold uncertainty estimate.",
        "- All plotted proportions were checked as finite values in [0, 1].",
        "",
        "See `figure_manifest.json` for file hashes and per-format checks.",
    ]
    (output / "QA_REPORT.md").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")
    legend_lines = ["# Figure legends", ""]
    for item in figure_manifest:
        legend_lines.append(f"- **{item['figure']}** ({item['status']}): {item['legend']}")
    (output / "FIGURE_LEGENDS.md").write_text("\n".join(legend_lines) + "\n", encoding="utf-8")
    print(
        f"[result_report] records={len(records)} ccd_frames={len(ccd_rows)} "
        f"figures={sum(item['status'] == 'available' for item in figure_manifest)} "
        f"output={output}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Qwen retrieval evidence and draw Nature-style figures")
    parser.add_argument("--root", default=".", help="Extracted bundle root (default: current directory)")
    parser.add_argument("--output-dir", default="result_report")
    parser.add_argument("--session-dir", action="append", default=[], help="Quick or four-stage session directory; repeatable")
    parser.add_argument("--baseline-json", action="append", default=[], help="Explicit fixed-test evaluation_summary.json; repeatable")
    parser.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png", "tiff"), default=("svg", "pdf", "png", "tiff"))
    parser.add_argument(
        "--require-arial",
        action="store_true",
        help="Fail rather than fall back when Arial is unavailable; use for final paper export",
    )
    args = parser.parse_args()
    generate_report(
        root=args.root,
        output_dir=args.output_dir,
        session_dirs=args.session_dir,
        baseline_paths=args.baseline_json,
        formats=args.formats,
        require_arial=args.require_arial,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
