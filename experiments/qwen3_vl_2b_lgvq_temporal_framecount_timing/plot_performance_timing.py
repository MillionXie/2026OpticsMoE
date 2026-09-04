from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
OPTICAL_SIX_LAYER_MS = 9.084


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(timing: dict[str, Any], performance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in sorted(map(int, timing["results"])):
        latency = timing["results"][str(count)]["latency_ms_per_video"]
        total = latency["total_raw_video_to_scalar_ms"]
        qwen = latency["qwen_backbone_ms"]
        training = performance["results"][str(count)]["training"]
        metric = training["metrics"]["temporal"]
        projected_hybrid_ms = total["mean"] - qwen["mean"] + OPTICAL_SIX_LAYER_MS
        rows.append(
            {
                "frames": count,
                "test_videos": timing["results"][str(count)]["unique_videos"],
                "mean_total_ms": total["mean"],
                "median_total_ms": total["median"],
                "p95_total_ms": total["p95"],
                "mean_qwen_ms": qwen["mean"],
                "optical_six_layer_ms": OPTICAL_SIX_LAYER_MS,
                "total_to_optical_core_ratio": total["mean"] / OPTICAL_SIX_LAYER_MS,
                "qwen_to_optical_core_ratio": qwen["mean"] / OPTICAL_SIX_LAYER_MS,
                "projected_hybrid_ms_if_qwen_replaced": projected_hybrid_ms,
                "projected_e2e_speedup_if_qwen_replaced": total["mean"] / projected_hybrid_ms,
                "temporal_srcc": metric["srcc"],
                "temporal_krcc": metric["krcc"],
                "temporal_plcc": metric["plcc"],
                "temporal_rmse": metric["rmse"],
                "temporal_mae": metric["mae"],
                "best_epoch": training["best_epoch"],
            }
        )
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, root: Path, name: str) -> None:
    figure.savefig(root / f"{name}.png", dpi=600, bbox_inches="tight")
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    configure_style()
    frames = np.asarray([row["frames"] for row in rows])
    mean = np.asarray([row["mean_total_ms"] for row in rows])
    median = np.asarray([row["median_total_ms"] for row in rows])
    p95 = np.asarray([row["p95_total_ms"] for row in rows])
    srcc = np.asarray([row["temporal_srcc"] for row in rows])
    plcc = np.asarray([row["temporal_plcc"] for row in rows])
    total_ratio = np.asarray([row["total_to_optical_core_ratio"] for row in rows])
    qwen_ratio = np.asarray([row["qwen_to_optical_core_ratio"] for row in rows])

    blue = "#2369A1"
    teal = "#159D8C"
    orange = "#E5862A"
    red = "#C94440"
    gray = "#777777"
    figure, axes = plt.subplots(1, 3, figsize=(7.08, 2.25), constrained_layout=True)

    ax = axes[0]
    ax.plot(frames, srcc, "o-", color=blue, label="SRCC")
    ax.plot(frames, plcc, "s-", color=teal, label="PLCC")
    ax.set(xlabel="Sampled frames", ylabel="Temporal correlation", xticks=frames)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.18, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = axes[1]
    ax.plot(frames, mean, "o-", color=blue, label="Mean")
    ax.plot(frames, median, "s--", color=teal, label="Median")
    ax.plot(frames, p95, "^:", color=orange, label="P95")
    ax.axhline(OPTICAL_SIX_LAYER_MS, color=red, linewidth=1.0, label="6-layer optics (9.084 ms)")
    ax.set(xlabel="Sampled frames", ylabel="Latency (ms/video)", xticks=frames)
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.text(-0.18, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = axes[2]
    ax.plot(frames, total_ratio, "o-", color=blue, label="E2E / optical core")
    ax.plot(frames, qwen_ratio, "s-", color=teal, label="Qwen / optical core")
    ax.axhline(50.0, color=orange, linestyle="--", label="50× reference")
    ax.axhline(100.0, color=red, linestyle=":", label="100× reference")
    ax.set(xlabel="Sampled frames", ylabel="Ratio to 9.084 ms", xticks=frames)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.text(-0.18, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=8)
    save_figure(figure, output, "temporal_performance_timing_speedup")

    figure, ax = plt.subplots(figsize=(3.54, 2.35), constrained_layout=True)
    scatter = ax.scatter(mean, srcc, c=frames, cmap="viridis", s=28, edgecolor="white", linewidth=0.5)
    for row in rows:
        ax.annotate(
            f"{row['frames']}f",
            (row["mean_total_ms"], row["temporal_srcc"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.2,
        )
    ax.set(xlabel="Electronic mean latency (ms/video)", ylabel="Temporal SRCC")
    ax.set_xlim(float(mean.min()) * 0.65, float(mean.max()) * 1.08)
    ax.set_ylim(float(srcc.min()) - 0.0025, float(srcc.max()) + 0.0040)
    ax.grid(color="#dddddd", linewidth=0.5)
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Sampled frames")
    save_figure(figure, output, "temporal_accuracy_latency_tradeoff")


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "timing_performance_speedup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "optical_six_layer_ms": OPTICAL_SIX_LAYER_MS,
        "ratios": rows,
        "interpretation": {
            "total_to_optical_core_ratio": (
                "Electronic raw-video-to-scalar mean divided by optical propagation-only time; "
                "an optimistic upper-bound ratio, not a measured hybrid end-to-end speedup."
            ),
            "qwen_to_optical_core_ratio": (
                "Measured Qwen backbone mean divided by optical propagation-only time; excludes "
                "decode, SLM/CCD I/O and downstream electronic processing."
            ),
            "projected_e2e_speedup_if_qwen_replaced": (
                "Electronic total divided by (electronic total - measured Qwen backbone + 9.084 ms). "
                "This is illustrative and is valid only if the six-layer optical core replaces the "
                "entire timed Qwen segment; real SLM/CCD I/O is not included."
            ),
            "reference_lines": [50, 100],
        },
    }
    (output / "timing_performance_speedup.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_summary(rows, output)


def write_markdown_report(rows: list[dict[str, Any]], path: Path) -> None:
    best = max(rows, key=lambda row: row["temporal_srcc"])
    first_50 = next((row["frames"] for row in rows if row["total_to_optical_core_ratio"] >= 50), None)
    first_100 = next((row["frames"] for row in rows if row["total_to_optical_core_ratio"] >= 100), None)
    lines = [
        "# LGVQ Temporal 抽帧性能、时间与光学参照",
        "",
        "- 日期：2026-09-04",
        "- 正式计算设备：NVIDIA GeForce RTX 5090 D（速度与性能均统一在同一台卡上生成）",
        "- 测试集：固定 558 个 LGVQ 视频；每档使用完全相同的样本",
        "- 光学参照：六层传播总计 9.084 ms",
        "",
        "## 总表",
        "",
        "| 抽帧 | N | 平均总时间 ms | 中位数 ms | P95 ms | SRCC | KRCC | PLCC | RMSE | MAE | 最佳 epoch | 总时间/9.084 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['frames']} | {row['test_videos']} | {row['mean_total_ms']:.3f} | "
            f"{row['median_total_ms']:.3f} | {row['p95_total_ms']:.3f} | "
            f"{row['temporal_srcc']:.4f} | {row['temporal_krcc']:.4f} | "
            f"{row['temporal_plcc']:.4f} | {row['temporal_rmse']:.3f} | "
            f"{row['temporal_mae']:.3f} | {row['best_epoch']} | "
            f"{row['total_to_optical_core_ratio']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "所有时间均来自 5090D 上的原始 MP4→连续标量全链路测试；性能使用同一台 5090D "
            "提取的冻结 Qwen 特征，并只训练一个共享 `Linear(2048,1)` 读出头 50 epoch。",
            "测试集每个 epoch 都评估，保留最高 test mean(Spatial SRCC, Temporal SRCC)；表中只展示 "
            "Temporal 指标。没有验证集。",
            "",
            "## 50× / 100× 光学参照",
            "",
            f"若六层光传播固定为 9.084 ms，则 50× 和 100× 分别对应电子耗时 "
            f"{50 * OPTICAL_SIX_LAYER_MS:.1f} ms 与 {100 * OPTICAL_SIX_LAYER_MS:.1f} ms。",
            "",
            "| 抽帧 | 5090D 总时间/光传播 | Qwen 前向/光传播 | 用 9.084 ms 替换 Qwen 后的估算总时间 ms | 估算端到端加速 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['frames']} | {row['total_to_optical_core_ratio']:.2f}× | "
            f"{row['qwen_to_optical_core_ratio']:.2f}× | "
            f"{row['projected_hybrid_ms_if_qwen_replaced']:.3f} | "
            f"{row['projected_e2e_speedup_if_qwen_replaced']:.3f}× |"
        )
    lines.extend(
        [
            "",
            f"按“完整电子总时间 ÷ 9.084 ms”的理论核心口径，首次超过 50× 是 "
            f"{first_50} 帧，首次超过 100× 是 {first_100} 帧。该倍率是光传播核心与完整电子链路的"
            "对照，不是已测得的真实整机加速。",
            "",
            "真实光电系统仍需承担 SLM 写入、稳定等待、CCD 曝光/读出、几何矫正与电子读出。"
            "因此报告同时给出“替换 Qwen 后的估算端到端加速”；它保留了 5090D 实测的解码和"
            "预处理开销，但仍未加入真实光学硬件 I/O。",
            "",
            "## 性能结论",
            "",
            f"本轮最高 Temporal SRCC 出现在 {best['frames']} 帧：SRCC={best['temporal_srcc']:.4f}、"
            f"PLCC={best['temporal_plcc']:.4f}。抽帧更多不保证指标严格单调，因此帧数应依据"
            "质量—延迟折中选择，而不应只依据最大帧数。",
            "",
            "## 样本数与 P95",
            "",
            "`N=558` 表示每档都对固定测试集的 558 个完整视频各运行一次，不是 558 帧。六档共有 "
            "3348 条逐视频计时记录。P95 是逐视频总时间的第 95 百分位数：约 95% 的视频不慢于"
            "该值，最慢约 5% 会超过它；P95 不是准确率、均值或置信区间。",
            "",
            "## 计时边界",
            "",
            "```text",
            "打开原始 MP4",
            "→ 对每个目标帧随机 seek + read",
            "→ 65% 中心裁剪并缩放到 448×448",
            "→ 官方 Qwen processor 与一次 Temporal prompt tokenizer",
            "→ CPU→GPU",
            "→ 完整 Qwen Vision tower + merger + Language backbone",
            "→ 最后有效 token + Linear(2048,1)",
            "→ 标量返回 CPU",
            "```",
            "",
            "模型加载与 warmup 不计入逐视频时间；batch size 为 1。测速没有使用顺序解码、"
            "多线程解码、帧缓存、特征缓存、量化或 `torch.compile`。性能训练阶段可以缓存冻结的"
            "Qwen 特征，但该缓存耗时从未用作速度证据。",
            "",
            "## 图和机器可读证据",
            "",
            "- `analysis_results/temporal_performance_timing_speedup.png/.pdf`：性能、时间和倍率三联图；",
            "- `analysis_results/temporal_accuracy_latency_tradeoff.png/.pdf`：性能—时间折中图；",
            "- `analysis_results/timing_performance_speedup.csv/.json`：论文重绘用数据；",
            "- `results/per_video_measurements*.csv`：5090D 逐视频原始时间；",
            "- `performance_results/frames*/linear_head/test_predictions.csv`：逐样本预测；",
            "- `performance_results/frames*/linear_head/train_history.json`：50 epoch 完整历史。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timing",
        type=Path,
        default=PROJECT_ROOT / "results" / "timing_summary_all.json",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=PROJECT_ROOT / "performance_results" / "comparison.json",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis_results")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "PERFORMANCE_TIMING_REPORT.md",
    )
    args = parser.parse_args()
    rows = build_rows(load_json(args.timing), load_json(args.performance))
    write_outputs(rows, args.output)
    write_markdown_report(rows, args.report)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
