"""Build the audited ABO optical-MoE versus frozen-Qwen comparison figure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
OPTICAL_PERFORMANCE = HERE.parent / "optical_router_moe_20260907" / "balanced" / "final_report.json"
OPTICAL_PROFILE = HERE / "optical_moe" / "optical_moe_electronics_5090d.json"
QWEN_REPORT = HERE / "qwen_baseline" / "baseline_report.json"
PROFILER_SOURCE = REPO_ROOT / "LightGenV2" / "scripts" / "profile_optical_electronics_5090d.py"
BASELINE_SOURCE = REPO_ROOT / "LightGenV2" / "tasks" / "t08_abo_image_text_retrieval" / "baseline_5090d.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    optical_result = _load(OPTICAL_PERFORMANCE)
    optical_profile = _load(OPTICAL_PROFILE)
    qwen = _load(QWEN_REPORT)
    optical_metrics = optical_result["student"]
    qwen_metrics = qwen["performance"]
    critical = optical_profile["tasks"]["t08"]["critical_path"]
    optical_power = optical_profile["power_measurement"]

    summary = {
        "task": "ABO easy100 image-query-to-fixed-title retrieval",
        "test_queries": 2400,
        "candidate_titles": 100,
        "optical_moe": {
            "method": "strong-balance optical-router Top-2 MoE",
            "performance": optical_metrics,
            "latency_estimated_ms_per_query": critical["estimated_wall_ms_per_call"],
            "electronic_compute_ms_per_query": critical["electronic_compute_wall_ms_per_call"],
            "optical_rig_power_w": optical_profile["constants"]["optical_power_w"],
            "gpu_idle_mean_w": optical_power["idle_mean_w"],
            "gpu_component_active_mean_w": optical_power["active_mean_w"],
            "gpu_component_active_peak_w": optical_power["active_peak_w"],
            "hybrid_average_power_proxy_w": critical["hybrid_average_power_proxy_w"],
            "physical_only_optical_energy_j_per_query": critical["physical_only_optical_energy_j_per_call"],
            "optical_rig_wall_energy_proxy_j_per_query": critical["optical_rig_wall_energy_proxy_j_per_call"],
            "gpu_board_critical_path_energy_proxy_j_per_query": critical["gpu_board_critical_path_energy_proxy_j_per_call"],
            "hybrid_energy_proxy_j_per_query": critical["hybrid_optical_plus_gpu_energy_proxy_j_per_call"],
            "hybrid_absolute_rated_upper_j_per_query": critical["hybrid_absolute_rated_upper_j_per_call"],
        },
        "frozen_qwen": {
            "model": qwen["model_family"],
            "parameters": qwen["model_parameters"],
            "trainable_parameters": qwen["trainable_parameters"],
            "performance": qwen_metrics,
            "latency_cuda_ms": qwen["latency_cuda_ms"],
            "power": qwen["power"],
        },
    }
    summary["comparison"] = {
        "optical_r1_gain_percentage_points": 100.0 * (
            optical_metrics["recall_at_1"] - qwen_metrics["recall_at_1"]
        ),
        "qwen_to_optical_latency_ratio": (
            qwen["latency_cuda_ms"]["mean"] / critical["estimated_wall_ms_per_call"]
        ),
        "qwen_measured_to_optical_hybrid_energy_ratio": (
            qwen["power"]["measured_active_energy_j_per_sample"]
            / critical["hybrid_optical_plus_gpu_energy_proxy_j_per_call"]
        ),
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        {
            "method": "Optical MoE strong balance",
            "r_at_1": optical_metrics["recall_at_1"],
            "r_at_5": optical_metrics["recall_at_5"],
            "r_at_10": optical_metrics["recall_at_10"],
            "mrr": optical_metrics["mrr"],
            "latency_ms_per_query": critical["estimated_wall_ms_per_call"],
            "power_w": critical["hybrid_average_power_proxy_w"],
            "energy_j_per_query": critical["hybrid_optical_plus_gpu_energy_proxy_j_per_call"],
            "rated_upper_energy_j_per_query": critical["hybrid_absolute_rated_upper_j_per_call"],
            "measurement_kind": "composed physical-plus-measured-electronics proxy",
        },
        {
            "method": "Frozen Qwen3-VL-Embedding-2B",
            "r_at_1": qwen_metrics["recall_at_1"],
            "r_at_5": qwen_metrics["recall_at_5"],
            "r_at_10": qwen_metrics["recall_at_10"],
            "mrr": qwen_metrics["mrr"],
            "latency_ms_per_query": qwen["latency_cuda_ms"]["mean"],
            "power_w": qwen["power"]["active_mean_w"],
            "energy_j_per_query": qwen["power"]["measured_active_energy_j_per_sample"],
            "rated_upper_energy_j_per_query": qwen["power"]["rated_upper_bound_energy_j_per_sample"],
            "measurement_kind": "direct 5090D model-active measurement",
        },
    ]
    with (HERE / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    blue, orange = "#2878B5", "#D95319"
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), constrained_layout=True)

    names = ["R@1", "R@5", "R@10", "MRR"]
    optical_values = [optical_metrics["recall_at_1"], optical_metrics["recall_at_5"], optical_metrics["recall_at_10"], optical_metrics["mrr"]]
    qwen_values = [qwen_metrics["recall_at_1"], qwen_metrics["recall_at_5"], qwen_metrics["recall_at_10"], qwen_metrics["mrr"]]
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, optical_values, 0.36, color=blue, label="Optical MoE")
    axes[0].bar(x + 0.18, qwen_values, 0.36, color=orange, label="Frozen Qwen")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.65, 1.01)
    axes[0].set_ylabel("Retrieval score")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    axes[0].set_title("a  Full-test performance", loc="left", fontweight="bold")

    latencies = [critical["estimated_wall_ms_per_call"], qwen["latency_cuda_ms"]["mean"]]
    axes[1].bar([0, 1], latencies, color=[blue, orange], width=0.58)
    axes[1].set_xticks([0, 1], ["Optical\nMoE", "Frozen\nQwen"])
    axes[1].set_ylabel("Latency (ms/query)")
    axes[1].set_title("b  Model-core latency", loc="left", fontweight="bold")
    axes[1].text(0.5, max(latencies) * 0.92, f"{summary['comparison']['qwen_to_optical_latency_ratio']:.2f}×", ha="center", va="top")

    energies = [critical["hybrid_optical_plus_gpu_energy_proxy_j_per_call"], qwen["power"]["measured_active_energy_j_per_sample"]]
    uppers = [critical["hybrid_absolute_rated_upper_j_per_call"], qwen["power"]["rated_upper_bound_energy_j_per_sample"]]
    axes[2].bar([0, 1], energies, color=[blue, orange], width=0.58, label="Measured/proxy")
    axes[2].scatter([0, 1], uppers, marker="_", s=180, color="#333333", linewidths=1.2, label="Rated upper")
    axes[2].set_xticks([0, 1], ["Optical\nMoE", "Frozen\nQwen"])
    axes[2].set_ylabel("Energy (J/query)")
    axes[2].set_title("c  Energy", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=6, loc="upper left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, zorder=0)
        axis.set_axisbelow(True)

    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"abo_5090d_comparison.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    evidence = [
        OPTICAL_PERFORMANCE,
        OPTICAL_PROFILE,
        HERE / "optical_moe" / "optical_moe_electronics_5090d.csv",
        HERE / "optical_moe" / "power_samples.csv",
        QWEN_REPORT,
        HERE / "qwen_baseline" / "timing_per_sample.csv",
        HERE / "qwen_baseline" / "power_samples.csv",
        HERE / "qwen_baseline" / "retrieval_predictions.csv",
        HERE / "summary.json",
        HERE / "comparison_summary.csv",
        HERE / "abo_5090d_comparison.png",
        HERE / "abo_5090d_comparison.pdf",
        PROFILER_SOURCE,
        BASELINE_SOURCE,
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in evidence
        ],
    }
    (HERE / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
