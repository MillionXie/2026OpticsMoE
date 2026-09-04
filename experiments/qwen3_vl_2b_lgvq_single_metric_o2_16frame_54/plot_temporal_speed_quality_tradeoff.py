"""Plot audited Temporal quality, ideal optical speedup, and router usage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


OPTICAL_SIX_PASS_MS = 9.084


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Router-channel calibration reports retain the evaluated metrics under
    # this key so that the calibration provenance and the figure input can be
    # the same auditable JSON file.
    if "normal_optical_electronic" in payload:
        payload = payload["normal_optical_electronic"]
    if "router_diagnostics" not in payload:
        raise ValueError(f"No evaluated router diagnostics found in {path}")
    return payload


def _effective_count(shares: list[float]) -> float:
    values = np.asarray(shares, dtype=float)
    values = values / max(values.sum(), 1.0e-12)
    return float(np.exp(-(values * np.log(np.maximum(values, 1.0e-12))).sum()))


def plot(
    metrics16: Path,
    metrics36: Path,
    output_dir: Path,
    qwen16_ms: float,
    qwen36_ms: float,
) -> dict[str, Any]:
    payloads = [_load(metrics16), _load(metrics36)]
    labels = ["16 frames", "36 frames"]
    qwen_times = [qwen16_ms, qwen36_ms]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for label, qwen_time, payload in zip(labels, qwen_times, payloads):
        vision = list(map(float, payload["router_diagnostics"]["vision"]["selected_share"]))
        language = list(map(float, payload["router_diagnostics"]["language"]["selected_share"]))
        rows.append(
            {
                "scheme": label,
                "temporal_srcc": float(payload["srcc"]),
                "temporal_plcc": float(payload["plcc"]),
                "temporal_krcc": float(payload["krcc"]),
                "qwen_time_ms": qwen_time,
                "six_pass_optical_time_ms": OPTICAL_SIX_PASS_MS,
                "ideal_compute_speedup_x": qwen_time / OPTICAL_SIX_PASS_MS,
                "vision_effective_expert_count": _effective_count(vision),
                "sequence_effective_expert_count": _effective_count(language),
                **{f"vision_expert_{i + 1}_share": value for i, value in enumerate(vision)},
                **{f"sequence_expert_{i + 1}_share": value for i, value in enumerate(language)},
            }
        )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), constrained_layout=True)
    x = np.arange(2)
    srcc = [row["temporal_srcc"] for row in rows]
    axes[0].bar(x, srcc, width=0.58, color=("#2878B5", "#9AC9DB"))
    axes[0].axhline(0.83, color="#C82423", linestyle="--", linewidth=0.8, label="16f target")
    axes[0].axhline(0.80, color="#F8AC8C", linestyle=":", linewidth=0.9, label="36f target")
    axes[0].set_ylim(0.70, 0.87)
    axes[0].set_ylabel("Temporal SRCC")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("a  Quality")
    axes[0].legend(frameon=False, loc="lower left")

    speedup = [row["ideal_compute_speedup_x"] for row in rows]
    axes[1].bar(x, speedup, width=0.58, color=("#4DAF4A", "#A6D96A"))
    axes[1].axhline(50.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[1].axhline(100.0, color="0.45", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("Ideal compute speedup (×)")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("b  Six-pass optical compute")

    width = 0.18
    colors = ("#2878B5", "#9AC9DB", "#F8AC8C", "#C82423")
    positions = np.arange(4)
    for scheme_index, row in enumerate(rows):
        offset = (scheme_index - 0.5) * width
        values = [row[f"vision_expert_{i + 1}_share"] for i in range(4)]
        axes[2].bar(
            positions + offset,
            values,
            width=width,
            color=colors,
            alpha=1.0 if scheme_index == 0 else 0.55,
            edgecolor="none",
            label=labels[scheme_index],
        )
    axes[2].axhline(0.25, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].set_xticks(positions, ("E1", "E2", "E3", "E4"))
    axes[2].set_ylim(0.0, 0.55)
    axes[2].set_ylabel("Vision router selected share")
    axes[2].set_title("c  Expert use")
    axes[2].legend(frameon=False, ncol=1)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.5)
        axis.set_axisbelow(True)
    png = output_dir / "temporal_16_36_speed_quality_router.png"
    pdf = output_dir / "temporal_16_36_speed_quality_router.pdf"
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)

    csv_path = output_dir / "temporal_16_36_speed_quality_router.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "rows": rows,
        "figure_png": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
        "data_csv": str(csv_path.resolve()),
        "speedup_scope": (
            "Qwen compute time divided by 9.084 ms for six optical propagations; "
            "excludes SLM refresh, CCD exposure/readout, transfer, and electronic tail"
        ),
    }
    (output_dir / "temporal_16_36_tradeoff_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics16", required=True, type=Path)
    parser.add_argument("--metrics36", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qwen16-ms", type=float, default=476.591)
    parser.add_argument("--qwen36-ms", type=float, default=1133.494)
    args = parser.parse_args()
    report = plot(
        args.metrics16,
        args.metrics36,
        args.output_dir,
        args.qwen16_ms,
        args.qwen36_ms,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
