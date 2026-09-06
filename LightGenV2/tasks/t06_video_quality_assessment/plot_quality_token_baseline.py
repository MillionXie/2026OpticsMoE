"""Create compact paper figures from the audited 448px dataset-once evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .project import REPO_ROOT, sha256


FRAMES = (4, 9, 16)
SCHEMES = (
    "scheme1_scalar_linear",
    "scheme2_five_quality_tokens",
)
LABELS = {
    SCHEMES[0]: "Scheme 1: scalar linear",
    SCHEMES[1]: "Scheme 2: quality tokens",
}
COLORS = {SCHEMES[0]: "#4477AA", SCHEMES[1]: "#CC6677"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve()
    evidence_root = summary_path.parent
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = _json(summary_path)
    rows = {
        (int(row["frames"]), row["scheme"]): row
        for row in summary["rows"]
        if int(row["frames"]) in FRAMES and row["scheme"] in SCHEMES
    }
    samples = {
        (count, scheme): _csv(
            evidence_root
            / f"frames{count}"
            / scheme
            / "per_video_predictions_and_timing.csv"
        )
        for count in FRAMES
        for scheme in SCHEMES
    }
    if len(rows) != 6 or any(len(value) != 558 for value in samples.values()):
        raise RuntimeError("Expected six complete 558-video evidence groups")
    _style()

    fig, axes = plt.subplots(
        1, 3, figsize=(18 / 2.54, 5.2 / 2.54), constrained_layout=True
    )
    frame_x = np.asarray(FRAMES)
    for scheme in SCHEMES:
        mean = [rows[count, scheme]["model_mean_ms"] for count in FRAMES]
        p95 = [rows[count, scheme]["model_p95_ms"] for count in FRAMES]
        axes[0].plot(
            frame_x,
            mean,
            "o-",
            color=COLORS[scheme],
            linewidth=1.2,
            label=LABELS[scheme],
        )
        axes[0].plot(frame_x, p95, "--", color=COLORS[scheme], alpha=0.55, linewidth=0.8)
    axes[0].set_xticks(FRAMES)
    axes[0].set_xlabel("sampled frames / video")
    axes[0].set_ylabel("model latency (ms/video)")
    axes[0].set_title("a  Mean and P95 latency", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    width = 0.18
    positions = np.arange(len(FRAMES))
    for scheme_index, scheme in enumerate(SCHEMES):
        offset = (-1.5 + 2 * scheme_index) * width
        srcc = [rows[count, scheme]["srcc"] for count in FRAMES]
        plcc = [rows[count, scheme]["plcc"] for count in FRAMES]
        axes[1].bar(
            positions + offset,
            srcc,
            width,
            color=COLORS[scheme],
            label=("S1" if scheme == SCHEMES[0] else "S2") + " SRCC",
        )
        axes[1].bar(
            positions + offset + width,
            plcc,
            width,
            facecolor="none",
            edgecolor=COLORS[scheme],
            linewidth=0.9,
            label=("S1" if scheme == SCHEMES[0] else "S2") + " PLCC",
        )
    axes[1].set_xticks(positions, tuple(str(value) for value in FRAMES))
    axes[1].set_xlabel("sampled frames / video")
    axes[1].set_ylabel("correlation")
    axes[1].set_ylim(0.70, 0.82)
    axes[1].set_title("b  Temporal quality", loc="left", fontweight="bold")
    axes[1].legend(
        frameon=False,
        fontsize=5.7,
        ncol=2,
        loc="lower right",
        columnspacing=0.7,
        handlelength=1.2,
    )

    scheme = SCHEMES[1]
    model_mean = [rows[count, scheme]["model_mean_ms"] for count in FRAMES]
    preprocessing = [rows[count, scheme]["preprocess_mean_ms"] for count in FRAMES]
    axes[2].bar(positions - 0.17, model_mean, 0.34, color="#228833", label="model core")
    axes[2].bar(positions + 0.17, preprocessing, 0.34, color="#AA3377", label="decode + preprocess")
    axes[2].set_xticks(positions, tuple(str(value) for value in FRAMES))
    axes[2].set_xlabel("sampled frames / video")
    axes[2].set_ylabel("mean time (ms/video)")
    axes[2].set_yscale("log")
    axes[2].set_title("c  Scheme 2 time boundary", loc="left", fontweight="bold")
    axes[2].legend(frameon=False)
    _save(fig, output / "framecount_latency_performance")

    fig, axes = plt.subplots(
        1, 3, figsize=(18 / 2.54, 5.2 / 2.54), constrained_layout=True
    )
    for index, (axis, count) in enumerate(zip(axes, FRAMES)):
        evidence = samples[count, SCHEMES[1]]
        target = np.asarray([float(row["target_temporal_mos"]) for row in evidence])
        prediction = np.asarray([float(row["prediction"]) for row in evidence])
        lower = float(min(target.min(), prediction.min()))
        upper = float(max(target.max(), prediction.max()))
        axis.scatter(target, prediction, s=6, alpha=0.40, color=COLORS[SCHEMES[1]], edgecolors="none")
        axis.plot((lower, upper), (lower, upper), color="0.35", linestyle="--", linewidth=0.8)
        coefficients = np.polyfit(target, prediction, 1)
        grid = np.linspace(lower, upper, 100)
        axis.plot(grid, np.polyval(coefficients, grid), color=COLORS[SCHEMES[0]], linewidth=1.0)
        row = rows[count, SCHEMES[1]]
        axis.text(
            0.03,
            0.97,
            f"SRCC={row['srcc']:.3f}\nPLCC={row['plcc']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_xlabel("target Temporal MOS")
        axis.set_title(f"{chr(97 + index)}  {count} frames", loc="left", fontweight="bold")
    axes[0].set_ylabel("predicted Temporal MOS")
    _save(fig, output / "scheme2_prediction_scatter")

    fig, axes = plt.subplots(
        1, 3, figsize=(18 / 2.54, 5.2 / 2.54), constrained_layout=True
    )
    for index, (axis, count) in enumerate(zip(axes, FRAMES)):
        for scheme in SCHEMES:
            values = np.sort(
                np.asarray(
                    [float(row["model_internal_cuda_ms"]) for row in samples[count, scheme]]
                )
            )
            probability = np.arange(1, len(values) + 1) / len(values)
            axis.plot(
                values,
                probability,
                color=COLORS[scheme],
                linewidth=1.1,
                label=LABELS[scheme],
            )
        axis.set_xlabel("model latency (ms/video)")
        axis.set_xscale("log")
        axis.set_ylim(0, 1.01)
        axis.set_title(f"{chr(97 + index)}  {count} frames", loc="left", fontweight="bold")
    axes[0].set_ylabel("empirical CDF")
    axes[0].legend(frameon=False)
    _save(fig, output / "latency_ecdf")

    visual_geometry = {
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "image_size": 448,
        "visual_block0_patch_tokens": {"4": 1568, "9": 3920, "16": 6272},
        "merged_visual_tokens": {"4": 392, "9": 980, "16": 1568},
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "input_image_size_wh": [448, 448],
        "test_videos_per_group": 558,
        "protocol": summary["protocol"],
        "timing_scope": summary["timing_scope"],
        "visual_geometry": visual_geometry,
        "rows": [rows[count, scheme] for count in FRAMES for scheme in SCHEMES],
        "source_summary": str(summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_summary_sha256": sha256(summary_path),
        "figures": [
            "framecount_latency_performance.png",
            "framecount_latency_performance.pdf",
            "scheme2_prediction_scatter.png",
            "scheme2_prediction_scatter.pdf",
            "latency_ecdf.png",
            "latency_ecdf.pdf",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "figures": result["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
