"""Consolidate the latest A100 Qwen baseline evidence into a paper-ready report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def fresh_row(report_path: Path, workload: str) -> dict[str, Any]:
    report = read_json(report_path)
    wall = report["timing"]["synchronized_wall_ms"]
    power = report["power"]["active_power_w"]
    utilization = report["power"]["active_gpu_utilization_percent"]
    return {
        "task_id": report["task_id"],
        "task": report["task"],
        "workload": workload,
        "metric": report["performance"]["metric"],
        "performance": report["performance"]["value"],
        "performance_samples": report["performance"]["samples"],
        "timing_calls": report["timing"]["samples"],
        "batch_size": 1,
        "frames_per_video": "",
        "wall_mean_ms_per_call": wall["mean"],
        "wall_median_ms_per_call": wall["median"],
        "wall_p95_ms_per_call": wall["p95"],
        "active_power_mean_w": power["mean"],
        "active_power_median_w": power["median"],
        "active_power_p95_w": power["p95"],
        "active_power_peak_w": power["maximum"],
        "active_gpu_utilization_mean_percent": utilization["mean"],
        "energy_j_per_call": report["power"]["measured_board_energy_j_per_inference"],
        "equivalent_16_video_calls": "",
        "equivalent_16_video_wall_ms": "",
        "equivalent_16_video_energy_j": "",
        "rated_250w_upper_energy_j_per_call": report["power"]["rated_250w_upper_energy_j_per_inference"],
        "timing_clock": "Synchronized Wall",
        "timing_boundary": "first native Vision Transformer block -> final task readout",
        "measurement_source": "fresh empty-card A100 run (2026-09-14)",
        "report": str(report_path),
    }


def temporal_row(formal_root: Path, batch_size: int) -> dict[str, Any]:
    directory = formal_root / f"batch_{batch_size:02d}"
    report = read_json(directory / "report.json")
    calls = [
        row
        for row in read_csv(directory / "batch_timing.csv")
        if int(row["batch_size_videos"]) == batch_size
    ]
    wall = stats(
        [float(row["legacy_vision_block0_to_score_synchronized_wall_ms"]) for row in calls]
    )
    telemetry_rows = [
        row
        for row in read_csv(directory / "telemetry.csv")
        if row["phase"].startswith("active:")
        and row["phase"].endswith(f":size{batch_size}")
    ]
    power_values = [float(row["watts"]) for row in telemetry_rows]
    utilization_values = [float(row["utilization_percent"]) for row in telemetry_rows]
    power = stats(power_values)
    calls_for_16 = math.ceil(16 / batch_size)
    energy = power["mean"] * wall["mean"] / 1000.0
    return {
        "task_id": f"T06-temporal-b{batch_size}",
        "task": f"LGVQ temporal quality, batch={batch_size}",
        "workload": f"{batch_size} video(s)/call, 4 frames/video; converted to 16 videos",
        "metric": "SRCC",
        "performance": report["performance"]["srcc"],
        "performance_samples": report["test_videos"],
        "timing_calls": len(calls),
        "batch_size": batch_size,
        "frames_per_video": 4,
        "wall_mean_ms_per_call": wall["mean"],
        "wall_median_ms_per_call": wall["median"],
        "wall_p95_ms_per_call": wall["p95"],
        "active_power_mean_w": power["mean"],
        "active_power_median_w": power["median"],
        "active_power_p95_w": power["p95"],
        "active_power_peak_w": power["maximum"],
        "active_gpu_utilization_mean_percent": statistics.fmean(utilization_values),
        "energy_j_per_call": energy,
        "equivalent_16_video_calls": calls_for_16,
        "equivalent_16_video_wall_ms": wall["mean"] * calls_for_16,
        "equivalent_16_video_energy_j": energy * calls_for_16,
        "rated_250w_upper_energy_j_per_call": 250.0 * wall["mean"] / 1000.0,
        "timing_clock": "Synchronized Wall",
        "timing_boundary": "first native Vision Transformer block -> five-quality-token score",
        "measurement_source": "existing exact-model A100 full-test run (2026-09-08)",
        "report": str(directory / "report.json"),
    }


def spatial_row(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    wall = report["model_internal_synchronized_wall_ms_all_558_first_included"]
    power = report["power"]
    return {
        "task_id": "T06-spatial",
        "task": "LGVQ spatial quality",
        "workload": "1 video/call, 4 frames/video",
        "metric": "SRCC",
        "performance": report["performance"]["srcc"],
        "performance_samples": report["test_videos"],
        "timing_calls": report["complete_qwen_forwards"],
        "batch_size": 1,
        "frames_per_video": 4,
        "wall_mean_ms_per_call": wall["mean"],
        "wall_median_ms_per_call": wall["median"],
        "wall_p95_ms_per_call": wall["p95"],
        "active_power_mean_w": power["active_mean_w"],
        "active_power_median_w": "",
        "active_power_p95_w": "",
        "active_power_peak_w": power["active_peak_w"],
        "active_gpu_utilization_mean_percent": "not recorded in this preserved run",
        "energy_j_per_call": power["measured_active_energy_j_per_sample"],
        "equivalent_16_video_calls": "",
        "equivalent_16_video_wall_ms": "",
        "equivalent_16_video_energy_j": "",
        "rated_250w_upper_energy_j_per_call": power["rated_upper_bound_energy_j_per_sample"],
        "timing_clock": "Synchronized Wall",
        "timing_boundary": "first native Vision Transformer block -> five-quality-token score",
        "measurement_source": "existing exact-model A100 full-test run (2026-09-09)",
        "report": str(report_path),
    }


def copy_evidence(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {target}")
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 任务 | 性能 | 单次/批次 Wall mean | 平均功率 | 单次/批次能耗 | 等效16视频时间 | 等效16视频能耗 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        equivalent_time = row["equivalent_16_video_wall_ms"]
        equivalent_energy = row["equivalent_16_video_energy_j"]
        performance = Decimal(str(row["performance"])).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        lines.append(
            "| {task} | {metric}={performance} | {wall:.3f} ms | {power:.3f} W | "
            "{energy:.3f} J | {equivalent_time} | {equivalent_energy} |".format(
                task=row["task"],
                metric=row["metric"],
                performance=performance,
                wall=float(row["wall_mean_ms_per_call"]),
                power=float(row["active_power_mean_w"]),
                energy=float(row["energy_j_per_call"]),
                equivalent_time=(
                    "—" if equivalent_time == "" else f"{float(equivalent_time):.3f} ms"
                ),
                equivalent_energy=(
                    "—" if equivalent_energy == "" else f"{float(equivalent_energy):.3f} J"
                ),
            )
        )
    return "\n".join(lines)


def plot(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Temporal b1", "Temporal b2", "Spatial", "ABO I2T", "ABO I2I", "LSP", "SALICON", "OpenMoji"
    ]
    times = [float(row["wall_mean_ms_per_call"]) for row in rows]
    powers = [float(row["active_power_mean_w"]) for row in rows]
    energies = [float(row["energy_j_per_call"]) for row in rows]
    colors = ["#2677b8", "#4c9bd3", "#7146a6", "#e17c26", "#efaa3c", "#379a70", "#c84f58", "#a63d70"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for axis, values, title, ylabel in zip(
        axes,
        (times, powers, energies),
        ("a  First-block-to-readout latency", "b  Active A100 board power", "c  Measured board energy"),
        ("Synchronized Wall mean (ms)", "Power (W)", "Energy per call (J)"),
    ):
        axis.bar(range(len(values)), values, color=colors)
        axis.set_xticks(range(len(labels)), labels, rotation=42, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "qwen_a100_latency_power_energy.png", dpi=240)
    figure.savefig(output / "qwen_a100_latency_power_energy.svg")
    plt.close(figure)

    temporal = rows[:2]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    values = [float(row["equivalent_16_video_wall_ms"]) for row in temporal]
    axes[0].bar(["batch=1", "batch=2"], values, color=colors[:2])
    axes[0].set_ylabel("16-video Wall time (ms)")
    axes[0].set_title("a  Equivalent 16-video latency", loc="left", fontweight="bold")
    values = [float(row["equivalent_16_video_energy_j"]) for row in temporal]
    axes[1].bar(["batch=1", "batch=2"], values, color=colors[:2])
    axes[1].set_ylabel("16-video board energy (J)")
    axes[1].set_title("b  Equivalent 16-video energy", loc="left", fontweight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "lgvq_temporal_batch1_batch2_equivalent16.png", dpi=240)
    figure.savefig(output / "lgvq_temporal_batch1_batch2_equivalent16.svg")
    plt.close(figure)


def sha256_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--temporal-formal", type=Path, required=True)
    parser.add_argument("--spatial-evidence", type=Path, required=True)
    parser.add_argument("--existing-a100-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fresh = output / "fresh"
    rows = [
        temporal_row(args.temporal_formal.resolve(), 1),
        temporal_row(args.temporal_formal.resolve(), 2),
        spatial_row(args.spatial_evidence.resolve() / "report.json"),
        fresh_row(fresh / "T08_abo_image_text/report.json", "1 image; 100 precomputed titles"),
        fresh_row(fresh / "T07_abo_image_image/report.json", "1 image; 1600 enrolled gallery images"),
        fresh_row(fresh / "T02_lsp/report.json", "1 RGB 224x224 image"),
        fresh_row(fresh / "T03_salicon/report.json", "1 RGB 224x224 image"),
        fresh_row(fresh / "T04_openmoji/report.json", "1 RGB 224x224 image + one editing instruction"),
    ]
    write_csv(output / "formal_summary.csv", rows)
    write_json(
        output / "formal_summary.json",
        {
            "schema_version": 1,
            "gpu": "NVIDIA A100-PCIE-40GB",
            "rated_power_limit_w": 250.0,
            "formal_clock": "Synchronized Wall",
            "timing_boundary": "first native Vision Transformer block to final task readout",
            "energy_formula": "active A100 board power mean * synchronized-wall mean / 1000",
            "rows": rows,
        },
    )

    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    copy_evidence(args.temporal_formal.resolve(), evidence / "T06_temporal_batch1_batch2")
    copy_evidence(args.spatial_evidence.resolve(), evidence / "T06_spatial")
    provenance = evidence / "performance_sources"
    provenance.mkdir(exist_ok=True)
    source_reports = {
        "T02_lsp_baseline_report.json": args.existing_a100_root / "LightGenV2/tasks/t02_keypoint_detection/runs/simulation/qwen_frozen_a100_20260907/baseline_report.json",
        "T08_abo_image_text_baseline_report.json": args.existing_a100_root / "LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/qwen_frozen_a100_easy100_controlled_20260907/baseline_report.json",
        "T03_salicon_selected_checkpoint_test.json": args.repo / "LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_50_20260912_seed42/selected_checkpoint_test_evaluation.json",
        "T04_openmoji_selected_checkpoint_test.json": args.repo / "LightGenV2/tasks/t04_semantic_interaction/runs/simulation/qwen_shared_s73/selected_checkpoint_test_evaluation.json",
        "T07_abo_image_image_final_report.json": output / "work/abo_qwen_reproduction/evidence/enrolled/final_report.json",
    }
    for name, source in source_reports.items():
        copy_evidence(source.resolve(), provenance / name)
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir(exist_ok=True)
    current_profiler = args.repo / "LightGenV2/scripts/profile_latest_qwen_changed_a100.py"
    shutil.copy2(current_profiler, source_snapshot / current_profiler.name)
    shutil.copy2(Path(__file__).resolve(), source_snapshot / Path(__file__).name)

    readme = f"""# 最新 Qwen3-VL baseline：A100 正式性能、速度、功率与能耗

## 主结果

{markdown_table(rows)}

LGVQ 时间质量按相同的 **16 个视频、每视频 4 帧** 工作量换算：batch=1 需要16次调用，batch=2需要8次调用。换算不是把单视频延迟除以batch，而是 `每批实测均值 × ceil(16/batch)`；能耗同理。

## 统一计时边界

正式时间均使用 **Synchronized Wall mean**：在第一个原生 Vision Transformer block 的 pre-hook 开始，到任务最终结果已经在GPU上生成并完成CUDA同步为止。包含所有后续必须执行的Vision/Language block及任务读出；不包含模型加载、文件I/O、图像/视频解码、processor/tokenizer、H2D，以及第一个block之前的patch embedding。

计时不是整段数据加载耗时，也不是只看CUDA Event。每次调用的CUDA Event仍保留在原始CSV中作审计。

## 功率与能耗

单次/批次能耗按 `E = P_active_mean × T_wall_mean / 1000` 计算，单位J。功率是A100整卡 `power.draw`，不是仅任务头功率，也不是250 W额定上界。额定上界能耗另存于CSV。

2026-09-14新测的T02/T03/T04/T07/T08采用独立连续推理功率段，10 ms采样，同时保存GPU利用率、显存和SM时钟；正式计时段不运行遥测采样器。每个新测报告的process_audit均证明加载前没有其他计算进程，加载后只有测量脚本本身。

T06 temporal使用已有的同一模型A100完整558视频原始记录（含batch=1/2功率、利用率和逐批时间）；T06 spatial使用同一模型558视频原始记录。它们的性能值与当前表格的0.7663/0.6908完全对应，未用旧checkpoint冒充。

## 输入与性能口径

- LGVQ：4帧/视频，每帧448×448；时间/空间分别使用对应五质量词baseline。时间质量batch=1为558批，batch=2为279个完整批次。
- ABO图搜文：RGB 224×224，冻结Qwen3-VL-Embedding-2B，2048维查询，对100个离线标题向量排序。
- ABO图搜图：新 enrolled-SKU 协议，800 query、1600 gallery、每query有8个同SKU正样本；native aspect、50176像素约束、64维前缀，对1600项完整稳定排序。它不是旧的120类中心协议。
- LSP：RGB 224×224，输出14×56×56热图；性能为1000张测试图的PCK@0.2。
- SALICON：RGB 224×224，冻结Vision主干，加197,184参数adapter与85,412参数density decoder；性能为5000张测试图的CC。
- OpenMoji：RGB 224×224 + 原始编辑指令；完整冻结Qwen Vision/Language，训练读出共972,952参数（image adapter 197,184、text adapter 393,792、shared grid readout 381,976），输出6×6类别与编辑网格；性能为1000样本changed-cell accuracy。

## 原始数据

- `fresh/`：五个2026-09-14空卡复测，每项含逐样本时间、10 ms功率/利用率/显存/时钟、进程审计、命令和JSON报告。
- `evidence/T06_temporal_batch1_batch2/`：batch=1/2的558视频逐批时间、预测、预处理轨迹和遥测。
- `evidence/T06_spatial/`：558视频逐视频时间、预测和功率。
- `evidence/performance_sources/`：表中最新版性能的原始评估报告副本。
- `source_snapshot/`：本次计时与汇总脚本；`SHA256SUMS.txt`固定全部证据字节。

图：`qwen_a100_latency_power_energy.png` 与 `lgvq_temporal_batch1_batch2_equivalent16.png`。
"""
    (output / "README_CN.md").write_text(readme, encoding="utf-8")
    plot(output, rows)
    sha256_manifest(output)
    print(markdown_table(rows))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
