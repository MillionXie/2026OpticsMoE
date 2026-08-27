"""Nature-style figures for the warmstart5 sim-to-real CCD agreement audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .agreement_common import read_csv, read_json, sha256_file, stage_directory, write_csv, write_json
from .agreement_evaluate import _load_ccd, _load_reference


BLUE = "#3B6FB6"
PURPLE = "#7A5195"
ORANGE = "#E07A3F"
GREY = "#777777"
LIGHT_GREY = "#E7E7E7"
RASTER_DPI = 600
SINGLE_COLUMN_MM = 89
# Static final width declaration is intentionally explicit for publication QA.
width_mm = 183
DOUBLE_COLUMN_MM = width_mm
VECTOR_EXTENSIONS = (".svg", ".pdf")
RASTER_EXTENSIONS = (".png", ".tiff")


def _mm(value: float) -> float:
    return value / 25.4


def _setup_matplotlib(require_arial: bool) -> tuple[Any, str]:
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as error:
        raise RuntimeError("Agreement plotting requires matplotlib") from error
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else (
        "Liberation Sans" if "Liberation Sans" in available else "DejaVu Sans"
    )
    if require_arial and font != "Arial":
        raise RuntimeError("Arial is required for formal agreement figures")
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
        }
    )
    return plt, font


def _clean_axis(axis: Any, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT_GREY, linewidth=0.5, zorder=0)
    axis.tick_params(direction="out")


def _number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in ("", None, "None"):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _save_figure(
    figure: Any,
    output: Path,
    stem: str,
    formats: Iterable[str],
    plt: Any,
) -> list[Path]:
    paths: list[Path] = []
    for extension in formats:
        path = output / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"dpi": RASTER_DPI}
        if extension in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _metric_distributions(rows: list[dict[str, str]], plt: Any) -> Any:
    selected = [row for row in rows if row["role"] == "evaluation"]
    figure, axes = plt.subplots(1, 3, figsize=(_mm(DOUBLE_COLUMN_MM), _mm(55)))
    metrics = (
        ("pcc_full", "PCC"),
        ("ssim", "SSIM"),
        ("shape_nrmse", "Shape NRMSE"),
    )
    groups = [
        ("transport_quantized", "linear", "Transport\nlinear", BLUE),
        ("ideal_model_fp32", "linear", "Ideal\nlinear", PURPLE),
        ("transport_quantized", "network_input", "Transport\nnetwork", ORANGE),
    ]
    for axis, (metric, label) in zip(axes, metrics):
        box_values: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []
        for reference, domain, group_label, color in groups:
            values = [
                value
                for row in selected
                if row["reference_kind"] == reference and row["domain"] == domain
                for value in [_number(row, metric)]
                if value is not None
            ]
            if values:
                box_values.append(values)
                labels.append(group_label)
                colors.append(color)
        positions = np.arange(1, len(box_values) + 1)
        boxes = axis.boxplot(
            box_values,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.6},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for position, values, color in zip(positions, box_values, colors):
            jitter = (
                np.linspace(-0.12, 0.12, len(values))
                if len(values) > 1
                else np.zeros(1)
            )
            axis.scatter(position + jitter, values, s=9, color=color, alpha=0.8, zorder=3)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(label)
        axis.set_title(label, loc="left")
        _clean_axis(axis)
        if metric in {"pcc_full", "ssim"}:
            axis.set_ylim(-0.05, 1.05)
    figure.tight_layout(pad=0.7)
    return figure


def _physical_diagnostics(rows: list[dict[str, str]], plt: Any) -> Any:
    selected = [
        row
        for row in rows
        if row["role"] == "evaluation"
        and row["reference_kind"] == "transport_quantized"
        and row["domain"] == "linear"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(_mm(DOUBLE_COLUMN_MM), _mm(55)))
    panels = (
        ("energy_ratio_calibrated", "Calibrated energy ratio", 1.0),
        ("centroid_distance_px", "Centroid error (px)", 0.0),
        ("outside_energy_fraction", "Energy outside sim support", 0.0),
    )
    for axis, (key, label, reference) in zip(axes, panels):
        values = [_number(row, key) for row in selected]
        values = [value for value in values if value is not None]
        x = np.arange(len(values))
        axis.scatter(x, values, color=BLUE, s=10, zorder=3)
        axis.axhline(reference, color=GREY, linewidth=0.7, linestyle="--")
        axis.set_xlabel("Independent probe")
        axis.set_ylabel(label)
        axis.set_title(label, loc="left")
        _clean_axis(axis)
        if key == "outside_energy_fraction":
            axis.set_ylim(bottom=0)
    figure.tight_layout(pad=0.7)
    return figure


def _selected_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["role"] == "evaluation"
        and row["reference_kind"] == "transport_quantized"
        and row["domain"] == "linear"
        and _number(row, "pcc_full") is not None
    ]
    if not selected:
        return []
    selected = sorted(selected, key=lambda row: row["canonical_key"])
    fixed = selected[0]
    ranked = sorted(selected, key=lambda row: float(row["pcc_full"]))
    worst = ranked[0]
    median = ranked[len(ranked) // 2]
    result: list[dict[str, str]] = []
    for row in (fixed, median, worst):
        if row["canonical_key"] not in {value["canonical_key"] for value in result}:
            result.append(row)
    return result


def _paired_images(
    session_dir: Path,
    rows: list[dict[str, str]],
    pairing_rows: list[dict[str, str]],
    plt: Any,
) -> Any | None:
    selected = _selected_examples(rows)
    if not selected:
        return None
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(_mm(DOUBLE_COLUMN_MM), _mm(48 * len(selected))), squeeze=False
    )
    labels = ("Measured CCD", "Transport simulation", "Mean-normalized difference")
    verified_capture: dict[tuple[str, str], dict[str, str]] = {}
    verified_reference: dict[tuple[str, str, str], dict[str, str]] = {}
    for audit in pairing_rows:
        if audit.get("status") != "verified":
            continue
        key = (audit["stage"], audit["capture_key"])
        previous = verified_capture.get(key)
        if previous is not None and (
            previous.get("ccd_file") != audit.get("ccd_file")
            or previous.get("ccd_sha256") != audit.get("ccd_sha256")
        ):
            raise RuntimeError(f"Pairing audit disagrees for {key}")
        verified_capture[key] = audit
        reference_key = (
            audit["stage"],
            audit["capture_key"],
            audit["reference_kind"],
        )
        if reference_key in verified_reference:
            raise RuntimeError(f"Duplicate pairing-audit reference row for {reference_key}")
        verified_reference[reference_key] = audit
    for row_index, row in enumerate(selected):
        stage_dir = stage_directory(session_dir, row["stage"])
        probe_rows = read_csv(stage_dir / "probe_manifest.csv")
        replicates = [
            value
            for value in probe_rows
            if value["canonical_key"] == row["canonical_key"]
        ]
        if len(replicates) != int(row["replicate_count"]):
            raise RuntimeError(
                f"Replicate count changed for {row['canonical_key']}: "
                f"metrics={row['replicate_count']} manifest={len(replicates)}"
            )
        measured_arrays: list[np.ndarray] = []
        for replicate in replicates:
            key = (row["stage"], replicate["capture_key"])
            audit = verified_capture.get(key)
            if audit is None:
                raise RuntimeError(f"No verified pairing-audit row for {key}")
            relative = Path(audit["ccd_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe pairing-audit CCD path: {relative}")
            measured_path = (stage_dir / relative).resolve()
            try:
                measured_path.relative_to(stage_dir.resolve())
            except ValueError as error:
                raise RuntimeError(
                    f"Pairing-audit CCD escapes stage directory: {relative}"
                ) from error
            if sha256_file(measured_path) != audit["ccd_sha256"]:
                raise RuntimeError(
                    f"Verified CCD changed after evaluation: {measured_path.name}"
                )
            measured_arrays.append(_load_ccd(measured_path, (478, 478))[0])
        measured = np.mean(np.stack(measured_arrays), axis=0)
        probe = replicates[0]
        reference_audit = verified_reference.get(
            (row["stage"], probe["capture_key"], "transport_quantized")
        )
        if reference_audit is None:
            raise RuntimeError(
                f"No verified transport reference for {row['stage']}/{probe['capture_key']}"
            )
        reference_relative = Path(reference_audit["reference_file"])
        if reference_relative.is_absolute() or ".." in reference_relative.parts:
            raise RuntimeError(
                f"Unsafe pairing-audit reference path: {reference_relative}"
            )
        reference_path = (stage_dir / reference_relative).resolve()
        try:
            reference_path.relative_to(stage_dir.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"Pairing-audit reference escapes stage directory: {reference_relative}"
            ) from error
        if sha256_file(reference_path) != reference_audit["reference_sha256"]:
            raise RuntimeError(
                f"Verified theoretical reference changed after evaluation: "
                f"{reference_path.name}"
            )
        theory = _load_reference(reference_path, (478, 478))
        measured_norm = measured / max(float(measured.mean()), 1.0e-12)
        theory_norm = theory / max(float(theory.mean()), 1.0e-12)
        values = (measured_norm, theory_norm, measured_norm - theory_norm)
        cmaps = ("viridis", "viridis", "coolwarm")
        limit = max(float(np.quantile(measured_norm, 0.995)), float(np.quantile(theory_norm, 0.995)))
        diff_limit = max(float(np.quantile(np.abs(values[2]), 0.995)), 1.0e-6)
        for column, (axis, value, label, cmap) in enumerate(zip(axes[row_index], values, labels, cmaps)):
            if column < 2:
                shown = axis.imshow(value, cmap=cmap, vmin=0, vmax=limit)
            else:
                shown = axis.imshow(value, cmap=cmap, vmin=-diff_limit, vmax=diff_limit)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(
                    f"{row['canonical_key'][:18]}\nPCC={float(row['pcc_full']):.3f}, "
                    f"n={int(row['replicate_count'])}"
                )
            figure.colorbar(shown, ax=axis, fraction=0.045, pad=0.02)
    figure.suptitle(
        "Fixed-key, median and worst held-out probes (selection rule declared)",
        x=0.01,
        ha="left",
    )
    figure.tight_layout(pad=0.6)
    return figure


def _repeat_orientation(
    repeat_rows: list[dict[str, str]],
    orientation_rows: list[dict[str, str]],
    plt: Any,
) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(_mm(DOUBLE_COLUMN_MM), _mm(55)))
    repeats = [
        value
        for row in repeat_rows
        if row.get("domain") == "linear"
        for value in [_number(row, "pcc_full")]
        if value is not None
    ]
    if repeats:
        axes[0].scatter(np.arange(len(repeats)), repeats, color=PURPLE, s=12)
        axes[0].axhline(np.median(repeats), color="black", linewidth=0.7)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("Replicate PCC")
        axes[0].set_xlabel("Replicate pair")
    else:
        axes[0].text(0.5, 0.5, "Repeat captures unavailable", ha="center", va="center")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    axes[0].set_title("Hardware repeatability ceiling", loc="left")
    _clean_axis(axes[0])

    orientations = [row for row in orientation_rows if row.get("stage")]
    if orientations:
        labels = [row["orientation"].replace("_", "\n") for row in orientations]
        values = [_number(row, "median_pcc") or 0.0 for row in orientations]
        colors = [ORANGE if row.get("diagnostic_best_candidate", "").lower() == "true" else BLUE for row in orientations]
        axes[1].bar(np.arange(len(values)), values, color=colors)
        axes[1].set_xticks(np.arange(len(values)), labels)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_ylabel("Calibration-probe median PCC")
    else:
        axes[1].text(0.5, 0.5, "Orientation diagnostic unavailable", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Orientation diagnostic only", loc="left")
    _clean_axis(axes[1])
    figure.tight_layout(pad=0.7)
    return figure


def build_report(
    evaluation_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    formats: Iterable[str] = ("svg", "pdf", "png", "tiff"),
    require_arial: bool = False,
) -> dict[str, Any]:
    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    manifest_path = evaluation_dir / "evaluation_manifest.json"
    evaluation = read_json(manifest_path)
    session_dir = Path(evaluation["session_dir"]).expanduser().resolve()
    declared_sources = evaluation.get("source_files")
    if not isinstance(declared_sources, list) or not declared_sources:
        raise RuntimeError("Evaluation manifest has no bound source_files")
    for item in declared_sources:
        if not isinstance(item, dict):
            raise RuntimeError("Evaluation source entry must be a mapping")
        source_path = Path(str(item.get("path", ""))).expanduser().resolve()
        try:
            source_path.relative_to(session_dir)
        except ValueError as error:
            raise RuntimeError(
                f"Evaluation source escapes session directory: {source_path}"
            ) from error
        declared_sha = str(item.get("sha256", "")).strip().lower()
        if len(declared_sha) != 64 or sha256_file(source_path) != declared_sha:
            raise RuntimeError(
                f"Evaluation source SHA-256 mismatch: {source_path.name}"
            )
    declared_evidence = evaluation.get("evidence_files")
    if not isinstance(declared_evidence, list) or not declared_evidence:
        raise RuntimeError("Evaluation manifest has no bound evidence_files")
    for item in declared_evidence:
        if not isinstance(item, dict):
            raise RuntimeError("Evaluation evidence entry must be a mapping")
        name = str(item.get("path", ""))
        relative = Path(name)
        if relative.name != name or relative.is_absolute():
            raise RuntimeError(f"Unsafe evaluation evidence path: {name!r}")
        evidence_path = evaluation_dir / relative
        declared_sha = str(item.get("sha256", "")).strip().lower()
        if len(declared_sha) != 64 or sha256_file(evidence_path) != declared_sha:
            raise RuntimeError(
                f"Evaluation evidence SHA-256 mismatch: {evidence_path.name}"
            )
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else evaluation_dir / "report"
    )
    output.mkdir(parents=True, exist_ok=True)
    formats = tuple(str(value).lower().lstrip(".") for value in formats)
    allowed = {"svg", "pdf", "png", "tif", "tiff"}
    if not formats or any(value not in allowed for value in formats):
        raise ValueError(f"Figure formats must be selected from {sorted(allowed)}")
    plt, font = _setup_matplotlib(require_arial)
    metric_rows = read_csv(evaluation_dir / "metrics_per_probe.csv")
    pairing_rows = read_csv(evaluation_dir / "pairing_audit.csv")
    repeat_rows = read_csv(evaluation_dir / "repeatability.csv")
    orientation_rows = read_csv(evaluation_dir / "orientation_summary.csv")

    figures = [
        ("fig01_agreement_distributions", _metric_distributions(metric_rows, plt), "All independent held-out probes; points are not replicate frames."),
        ("fig02_physical_diagnostics", _physical_diagnostics(metric_rows, plt), "Transport-matched linear-domain photometric and geometric diagnostics."),
        ("fig03_paired_ccd_examples", _paired_images(session_dir, metric_rows, pairing_rows, plt), "Fixed-key, median and worst examples; measured panels use the same verified replicate averages as their annotated metrics."),
        ("fig04_repeatability_orientation", _repeat_orientation(repeat_rows, orientation_rows, plt), "Repeatability and flip candidates are diagnostics; orientation candidates never alter primary metrics."),
    ]
    figure_manifest: list[dict[str, Any]] = []
    for stem, figure, legend in figures:
        if figure is None:
            figure_manifest.append({"figure": stem, "status": "unavailable", "legend": legend})
            continue
        paths = _save_figure(figure, output, stem, formats, plt)
        figure_manifest.append(
            {
                "figure": stem,
                "status": "available",
                "legend": legend,
                "files": [
                    {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in paths
                ],
            }
        )
    source_paths = [
        manifest_path,
        evaluation_dir / "pairing_audit.csv",
        evaluation_dir / "metrics_per_probe.csv",
        evaluation_dir / "summary_by_stage.csv",
        evaluation_dir / "repeatability.csv",
        evaluation_dir / "orientation_summary.csv",
    ]
    report = {
        "schema_version": 1,
        "type": "qwen_warmstart5_sim_to_real_agreement_report",
        "font": font,
        "font_size_pt": 7,
        "raster_dpi": RASTER_DPI,
        "session_dir": str(session_dir),
        "evidence_policy": (
            "Only verified on-disk CCD/reference pairs are plotted. Primary metrics "
            "use the predeclared canonical orientation; diagnostic flip scores are "
            "never substituted."
        ),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ],
        "figures": figure_manifest,
    }
    write_json(output / "report_manifest.json", report)
    flat_rows: list[dict[str, Any]] = []
    for item in figure_manifest:
        if item["status"] == "available":
            for file_item in item["files"]:
                flat_rows.append({"figure": item["figure"], "status": "available", **file_item})
        else:
            flat_rows.append({"figure": item["figure"], "status": "unavailable", "path": "", "bytes": "", "sha256": ""})
    write_csv(output / "figure_manifest.csv", flat_rows, ("figure", "status", "path", "bytes", "sha256"))
    (output / "README.md").write_text(
        "# Sim-to-real agreement report\n\n"
        "Primary metrics use the fixed canonical transform. Orientation candidates "
        "are calibration-only diagnostics and are never selected per frame. Missing "
        "hardware evidence is not imputed. See `report_manifest.json` for hashes.\n",
        encoding="utf-8",
    )
    print(
        f"[agreement_report] figures={sum(item['status'] == 'available' for item in figure_manifest)} "
        f"font={font} output={output}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Qwen warmstart5 sim-to-real agreement")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--require-arial", action="store_true")
    args = parser.parse_args()
    build_report(
        args.evaluation_dir,
        output_dir=args.output_dir,
        formats=[value.strip() for value in args.formats.split(",") if value.strip()],
        require_arial=args.require_arial,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
