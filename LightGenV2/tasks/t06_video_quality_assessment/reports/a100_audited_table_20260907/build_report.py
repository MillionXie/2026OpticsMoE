"""Rebuild the audited A100 comparison table and figures from archived evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
OPTICAL_POWER_W = 80.388


def load(name: str) -> dict[str, Any]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen_single(report: dict[str, Any]) -> dict[str, float]:
    latency = report.get("latency_cuda_ms")
    if latency is None:
        latency = report["model_internal_cuda_ms_all_558_first_included"]
    power = report["power"]
    return {
        "qwen_latency_mean_ms": float(latency["mean"]),
        "qwen_latency_median_ms": float(latency["median"]),
        "qwen_latency_p95_ms": float(latency["p95"]),
        "qwen_active_power_mean_w": float(power["active_mean_w"]),
        "qwen_active_power_peak_w": float(power["active_peak_w"]),
        "qwen_energy_measured_j": float(power["measured_active_energy_j_per_sample"]),
        "qwen_energy_idle_subtracted_j": float(
            power["idle_subtracted_energy_j_per_sample"]
        ),
        "qwen_energy_rated_upper_j": float(power["rated_upper_bound_energy_j_per_sample"]),
    }


def add_optical(row: dict[str, Any], profile: dict[str, Any]) -> None:
    row["ours_latency_ms"] = float(profile["estimated_wall_ms_per_call"])
    row["ours_physical_time_ms"] = float(profile["physical_time_ms_per_call"])
    row["ours_serial_electronic_ms"] = float(
        profile["serial_electronic_wall_ms_per_call"]
    )
    row["ours_energy_proxy_j"] = OPTICAL_POWER_W * row["ours_latency_ms"] / 1000.0
    row["speedup_qwen_over_ours"] = row["qwen_latency_mean_ms"] / row["ours_latency_ms"]
    row["energy_ratio_measured_qwen_over_ours"] = (
        row["qwen_energy_measured_j"] / row["ours_energy_proxy_j"]
    )
    row["energy_ratio_rated_upper_qwen_over_ours"] = (
        row["qwen_energy_rated_upper_j"] / row["ours_energy_proxy_j"]
    )


def f(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    temporal = load("t06_temporal_batch16_formal_a100.json")
    sweep = load("t06_temporal_batch_sweep_a100.json")
    temporal_old = load("t06_temporal_batch1_a100.json")
    spatial = load("t06_spatial_qwen_a100.json")
    t08 = load("t08_abo_qwen_a100.json")
    t07 = load("t07_abo_similarity10_qwen_a100.json")
    lsp = load("t02_lsp_qwen_a100.json")
    salicon = load("t03_salicon_qwen_a100.json")
    optical = load("optical_moe_electronics_a100.json")
    optical_spatial = load("optical_moe_spatial_electronics_a100.json")

    batch16_sweep = next(
        item for item in sweep["results"] if item["batch_size_videos"] == 16
    )
    steady_power = float(batch16_sweep["telemetry"]["active_mean_w"])
    steady_peak = float(batch16_sweep["telemetry"]["active_peak_w"])
    temporal_latency = temporal["model_boundary_full_batch_cuda_ms"]
    temporal_mean_ms = float(temporal_latency["mean"])
    temporal_idle = float(batch16_sweep["telemetry"]["idle_mean_w"])

    rows: list[dict[str, Any]] = []
    temporal_row = {
        "task_id": "T06-temporal",
        "task": "LGVQ temporal quality",
        "metric": "SRCC",
        "ours_performance": 0.8044,
        "qwen_performance": float(temporal["performance"]["srcc"]),
        "workload": "16 videos × 4 frames",
        "test_samples": 558,
        "qwen_latency_mean_ms": temporal_mean_ms,
        "qwen_latency_median_ms": float(temporal_latency["median"]),
        "qwen_latency_p95_ms": float(temporal_latency["p95"]),
        "qwen_active_power_mean_w": steady_power,
        "qwen_active_power_peak_w": steady_peak,
        "qwen_energy_measured_j": steady_power * temporal_mean_ms / 1000.0,
        "qwen_energy_idle_subtracted_j": max(0.0, steady_power - temporal_idle)
        * temporal_mean_ms
        / 1000.0,
        "qwen_energy_rated_upper_j": 250.0 * temporal_mean_ms / 1000.0,
        "qwen_energy_power_basis": "batch16 steady-compute active mean from sweep",
    }
    add_optical(temporal_row, optical["tasks"]["t06"]["paper_serial_path"])
    rows.append(temporal_row)

    for base, performance, profile, task_id, task, metric, ours, workload, samples in [
        (
            spatial,
            spatial["performance"]["srcc"],
            optical_spatial["tasks"]["t06_spatial"]["paper_serial_path"],
            "T06-spatial",
            "LGVQ spatial quality",
            "SRCC",
            0.6393237619075796,
            "1 video × 4 frames",
            558,
        ),
        (
            t08,
            t08["performance"]["recall_at_1"],
            optical["tasks"]["t08"]["paper_serial_path"],
            "T08",
            "ABO image-to-title retrieval",
            "R@1",
            0.7983333333333333,
            "1 image query × 100 fixed titles",
            2400,
        ),
        (
            lsp,
            lsp["performance"]["pck_at_0.2_torso"],
            optical["tasks"]["t02"]["paper_serial_path"],
            "T02",
            "LSP keypoint detection",
            "PCK@0.2",
            0.5773,
            "1 image",
            1000,
        ),
        (
            salicon,
            salicon["performance"]["cc"],
            optical["tasks"]["t03"]["paper_serial_path"],
            "T03",
            "SALICON saliency",
            "CC",
            0.8291,
            "1 image",
            5000,
        ),
    ]:
        row = {
            "task_id": task_id,
            "task": task,
            "metric": metric,
            "ours_performance": float(ours),
            "qwen_performance": float(performance),
            "workload": workload,
            "test_samples": samples,
            **qwen_single(base),
            "qwen_energy_power_basis": "model-active board-power samples",
        }
        add_optical(row, profile)
        rows.append(row)

    t07_row = {
        "task_id": "T07",
        "task": "ABO similarity-10 image-to-image retrieval",
        "metric": "R@1",
        "ours_performance": None,
        "qwen_performance": float(t07["performance"]["r_at_1"]),
        "workload": "1 image query × 120 train-product centroids",
        "test_samples": 480,
        **qwen_single(t07),
        "qwen_energy_power_basis": "model-active board-power samples",
        "ours_latency_ms": None,
        "ours_physical_time_ms": None,
        "ours_serial_electronic_ms": None,
        "ours_energy_proxy_j": None,
        "speedup_qwen_over_ours": None,
        "energy_ratio_measured_qwen_over_ours": None,
        "energy_ratio_rated_upper_qwen_over_ours": None,
    }
    rows.insert(3, t07_row)

    output = {
        "schema_version": 1,
        "status": "complete",
        "generated_from_archived_evidence": True,
        "gpu": "NVIDIA A100-PCIE-40GB",
        "physical_gpu_id": 6,
        "rated_power_limit_w": 250.0,
        "optical_power_proxy_w": OPTICAL_POWER_W,
        "rows": rows,
        "temporal_batch_selection": {
            "selected_batch": 16,
            "near_full_power_first_reached_at_batch": 8,
            "selection_reason": (
                "batch 8 reached 95.56% mean rated power; batch 16 stayed on the same "
                "power plateau and increased throughput only 4.60%, while exactly matching "
                "the 16-video optical workload"
            ),
            "sweep": sweep["results"],
        },
        "temporal_performance": {
            "ours": {"srcc": 0.8044, "plcc": 0.8180, "rmse": 7.991, "mae": 5.992},
            "qwen_batch16": temporal["performance"],
            "qwen_old_batch1": temporal_old["performance"],
        },
        "performance_detail": {
            "lgvq_spatial": {
                "ours": {
                    "srcc": 0.6393237619075796,
                    "plcc": 0.6743486500135264,
                    "rmse": 8.451935768127441,
                    "mae": 6.646285533905029,
                },
                "qwen": spatial["performance"],
            },
            "abo_image_to_title_t08": {
                "ours": {
                    "recall_at_1": 0.7983333333333333,
                    "recall_at_5": 0.95375,
                    "recall_at_10": 0.9858333333333333,
                    "mrr": 0.867616268422718,
                },
                "qwen": t08["performance"],
            },
            "abo_similarity10_t07": {"ours": None, "qwen": t07["performance"]},
            "lsp": {
                "ours": {"pck_at_0.2_torso": 0.5773, "pckh_at_0.5_head": 0.7363},
                "qwen": lsp["performance"],
            },
            "salicon": {
                "ours": {
                    "cc": 0.8291,
                    "sim": 0.8063,
                    "nss": 0.9283,
                    "auc_judd": 0.7631,
                    "kld": 0.1330,
                    "mae": 0.0890,
                },
                "qwen": salicon["performance"],
            },
        },
        "timing_contract": {
            "primary_start": "pre-hook at the first native Vision Transformer block input",
            "primary_end": "task score/retrieval result ready on GPU",
            "primary_excludes": [
                "model and processor loading",
                "disk I/O / MP4 random-seek decode",
                "crop and resize",
                "processor/tokenizer",
                "host-to-device copy",
                "vision patch embedding before block 0",
            ],
            "temporal_formal_protocol": (
                "preprocess all 558 videos on CPU, then execute 34 full batch-16 calls plus "
                "one batch-14 call consecutively; zero explicit warm-up and include first batch"
            ),
            "temporal_power_protocol": (
                "use batch-16 steady-compute sweep power for energy at the formal batch-16 "
                "mean latency; also archive the distinct-batch active-window reading separately"
            ),
        },
        "audit_corrections": [
            {
                "incorrect_or_mixed": "1046.928 ms / 16 videos",
                "correct": temporal_mean_ms,
                "reason": "1046.928 ms was a 5090D-derived sequential value, not this A100 batch-16 run",
            },
            {
                "incorrect_or_mixed": "1200.053 ms / 16 videos",
                "correct": temporal_mean_ms,
                "reason": "1200.053 ms is A100 batch-1 mean ×16; valid historical evidence, but not the new near-full-power batch-16 protocol",
            },
            {
                "incorrect_or_mixed": "9.384 ms and 0.754 J for temporal optical",
                "correct": {"latency_ms": 20.3822885, "energy_j": 1.6384914079380002},
                "reason": "9.384 ms omitted measured post-CCD fusion, frame-to-video bridge, and final temporal head",
            },
            {
                "incorrect_or_mixed": "LSP baseline 73.71%",
                "correct": float(lsp["performance"]["pck_at_0.2_torso"]),
                "reason": "73.71% belonged to a different ABO row; full LSP test gives PCK@0.2=72.257%",
            },
            {
                "incorrect_or_mixed": "ABO similarity-10 labelled text-to-image",
                "correct": "image query to train-product image centroids",
                "reason": "the delivered archive and formal baseline contain images/categories but no text-query contract",
            },
            {
                "incorrect_or_mixed": "one fixed 0.730 J optical energy for every task",
                "correct": "task-specific 80.388 W × complete task critical-path latency",
                "reason": "tasks use different optical pass counts and different measured serial electronic tails",
            },
        ],
    }

    (ROOT / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with (ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        path.relative_to(EVIDENCE).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(EVIDENCE.rglob("*"))
        if path.is_file()
    }
    (ROOT / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    sweep_rows = [item for item in sweep["results"] if item["status"] == "complete"]
    batches = [item["batch_size_videos"] for item in sweep_rows]
    power = [item["telemetry"]["active_mean_w"] for item in sweep_rows]
    throughput = [item["throughput_videos_per_second"] for item in sweep_rows]
    latency = [item["model_boundary_batch_cuda_ms"]["mean"] for item in sweep_rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    axes[0].plot(batches, power, "o-", color="#1f77b4")
    axes[0].axhline(250, color="0.4", linestyle="--", label="250 W limit")
    axes[0].set(xlabel="Batch (videos)", ylabel="Active mean power (W)", title="a  Power")
    axes[0].legend(frameon=False)
    axes[1].plot(batches, throughput, "o-", color="#d95f02")
    axes[1].set(
        xlabel="Batch (videos)", ylabel="Throughput (videos/s)", title="b  Throughput"
    )
    axes[2].plot(batches, latency, "o-", color="#1b9e77")
    axes[2].set(xlabel="Batch (videos)", ylabel="Batch latency (ms)", title="c  Latency")
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"a100_temporal_batch_sweep.{suffix}", dpi=300)
    plt.close(fig)

    comparable = [row for row in rows if row["ours_latency_ms"] is not None]
    x = np.arange(len(comparable))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), constrained_layout=True)
    axes[0].bar(x - width / 2, [row["ours_latency_ms"] for row in comparable], width, label="Optical MoE")
    axes[0].bar(x + width / 2, [row["qwen_latency_mean_ms"] for row in comparable], width, label="Qwen A100")
    axes[0].set_yscale("log")
    axes[0].set(ylabel="Matched-workload latency (ms)", title="a  Latency")
    axes[0].set_xticks(x, [row["task_id"] for row in comparable], rotation=25, ha="right")
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, [row["ours_energy_proxy_j"] for row in comparable], width, label="Optical proxy")
    axes[1].bar(x + width / 2, [row["qwen_energy_measured_j"] for row in comparable], width, label="Qwen active")
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Matched-workload energy (J)", title="b  Energy")
    axes[1].set_xticks(x, [row["task_id"] for row in comparable], rotation=25, ha="right")
    axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"a100_audited_latency_energy.{suffix}", dpi=300)
    plt.close(fig)

    table_lines = [
        "| 任务 | 负载 | 指标 | 光学仿真 | Qwen A100 | Ours (ms) | Qwen (ms) | 加速 | Ours (J) | Qwen active (J) | 能效比 | 250 W 上界 (J) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {task_id} | {workload} | {metric} | {ours_perf} | {qwen_perf} | {ours_ms} | {qwen_ms} | {speedup} | {ours_j} | {qwen_j} | {energy_ratio} | {upper_j} |".format(
                **row,
                ours_perf=f(row["ours_performance"], 4),
                qwen_perf=f(row["qwen_performance"], 4),
                ours_ms=f(row["ours_latency_ms"]),
                qwen_ms=f(row["qwen_latency_mean_ms"]),
                speedup="—" if row["speedup_qwen_over_ours"] is None else f"{row['speedup_qwen_over_ours']:.2f}×",
                ours_j=f(row["ours_energy_proxy_j"]),
                qwen_j=f(row["qwen_energy_measured_j"]),
                energy_ratio="—"
                if row["energy_ratio_measured_qwen_over_ours"] is None
                else f"{row['energy_ratio_measured_qwen_over_ours']:.2f}×",
                upper_j=f(row["qwen_energy_rated_upper_j"]),
            )
        )

    batch_lines = [
        "| Batch | 平均功率 (W) | 峰值 (W) | 功率上限占比 | GPU util. | 吞吐 (video/s) | 整批 mean (ms) | 每视频能耗 (J) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sweep_rows:
        telemetry = item["telemetry"]
        batch_lines.append(
            f"| {item['batch_size_videos']} | {telemetry['active_mean_w']:.2f} | {telemetry['active_peak_w']:.2f} | "
            f"{100*telemetry['active_mean_fraction_of_power_limit']:.1f}% | "
            f"{telemetry['active_mean_gpu_utilization_percent']:.1f}% | "
            f"{item['throughput_videos_per_second']:.2f} | "
            f"{item['model_boundary_batch_cuda_ms']['mean']:.3f} | "
            f"{telemetry['measured_active_energy_j_per_video']:.3f} |"
        )

    main_table = "\n".join(table_lines)
    batch_table = "\n".join(batch_lines)
    readme = f"""# A100 正式数据总表与复核（2026-09-07）

## 结论

时间一致性 baseline 在 batch 8 已达到 A100 额定功率的 **95.6%**；batch 16 的平均功率保持在 **238.18 W（95.3%）**，吞吐仅比 batch 8 再提高约 **4.6%**，说明已进入平台。正式表选择 **batch 16**，因为它与光学方法一次并行 16 个视频的负载完全一致。

batch-16 完整 558 视频复测得到：SRCC **{temporal['performance']['srcc']:.4f}**、PLCC **{temporal['performance']['plcc']:.4f}**；16 视频整批模型边界 mean/median/P95 为 **{temporal_mean_ms:.3f}/{temporal_latency['median']:.3f}/{temporal_latency['p95']:.3f} ms**。按稳态平均功率 238.18 W 计算 active 能耗为 **{steady_power * temporal_mean_ms / 1000:.3f} J/16 视频**；250 W 严格上界为 **{250 * temporal_mean_ms / 1000:.3f} J**。

## 审计后的主表

{main_table}

T07 目前只有 frozen-Qwen baseline，没有同一数据合同下的光学 MoE，因此 Ours、加速比和能效比留空，不能借用 T08 的数值。

## 完整性能数据

| 任务 | 方法 | 主要与次要指标 |
|---|---|---|
| LGVQ 时间 | Ours | SRCC 0.8044；PLCC 0.8180；RMSE 7.991；MAE 5.992 |
| LGVQ 时间 | Qwen batch-16 | SRCC {temporal['performance']['srcc']:.4f}；KRCC {temporal['performance']['krcc']:.4f}；PLCC {temporal['performance']['plcc']:.4f}；RMSE {temporal['performance']['rmse']:.3f}；MAE {temporal['performance']['mae']:.3f} |
| LGVQ 空间 | Ours | SRCC 0.6393；PLCC 0.6743；RMSE 8.452；MAE 6.646 |
| LGVQ 空间 | Qwen | SRCC {spatial['performance']['srcc']:.4f}；KRCC {spatial['performance']['krcc']:.4f}；PLCC {spatial['performance']['plcc']:.4f}；RMSE {spatial['performance']['rmse']:.3f}；MAE {spatial['performance']['mae']:.3f} |
| ABO 图搜文 T08 | Ours | R@1/5/10 0.7983/0.9538/0.9858；MRR 0.8676 |
| ABO 图搜文 T08 | Qwen | R@1/5/10 {t08['performance']['recall_at_1']:.4f}/{t08['performance']['recall_at_5']:.4f}/{t08['performance']['recall_at_10']:.4f}；MRR {t08['performance']['mrr']:.4f} |
| ABO similarity-10 T07 | Qwen | R@1/5/10 {t07['performance']['r_at_1']:.4f}/{t07['performance']['r_at_5']:.4f}/{t07['performance']['r_at_10']:.4f}；mAP@10 {t07['performance']['map_at_10']:.4f}；NDCG@10 {t07['performance']['ndcg_at_10']:.4f} |
| LSP | Ours / Qwen | PCK@0.2 0.5773 / {lsp['performance']['pck_at_0.2_torso']:.4f}；PCKh@0.5 0.7363 / {lsp['performance']['pckh_at_0.5_head']:.4f} |
| SALICON | Ours | CC/SIM/NSS/AUC-Judd 0.8291/0.8063/0.9283/0.7631；KLD/MAE 0.1330/0.0890 |
| SALICON | Qwen | CC/SIM/NSS/AUC-Judd {salicon['performance']['cc']:.4f}/{salicon['performance']['sim']:.4f}/{salicon['performance']['nss']:.4f}/{salicon['performance']['auc_judd']:.4f}；KLD/MAE {salicon['performance']['kld']:.4f}/{salicon['performance']['mae']:.4f} |

## 测量样本量与协议

| 任务 | 性能样本 | 计时样本/批 | Batch | 显式 warm-up | active mean / peak (W) |
|---|---:|---:|---:|---:|---:|
| LGVQ 时间 | 558 视频 | 34 个 batch-16（另 1 个 batch-14） | 16 | 0 | {steady_power:.2f} / {steady_peak:.2f}（稳态 sweep） |
| LGVQ 空间 | 558 视频 | 558 视频 | 1 | 0 | {spatial['power']['active_mean_w']:.2f} / {spatial['power']['active_peak_w']:.2f} |
| ABO 图搜文 T08 | 2400 query | 200 query | 1 | 50 | {t08['power']['active_mean_w']:.2f} / {t08['power']['active_peak_w']:.2f} |
| ABO similarity-10 T07 | 480 query | 200 query | 1 | 50 | {t07['power']['active_mean_w']:.2f} / {t07['power']['active_peak_w']:.2f} |
| LSP | 1000 图 | 200 图 | 1 | 50 | {lsp['power']['active_mean_w']:.2f} / {lsp['power']['active_peak_w']:.2f} |
| SALICON | 5000 图 | 200 图 | 1 | 50 | {salicon['power']['active_mean_w']:.2f} / {salicon['power']['active_peak_w']:.2f} |

## 时间一致性 batch sweep

{batch_table}

功率来自物理 GPU 6 `NVIDIA A100-PCIE-40GB`，额定上限 250 W。`nvidia-smi power.draw` 的瞬时峰值可能短时超过额定平均功率限制，所以论文能耗同时保留 active mean 和 250 W 上界，不能用瞬时 peak 乘时延。

## 到底从哪里开始、到哪里结束

主时延起点是 **Qwen 原生 Vision Transformer 第 0 个 block 的输入 pre-hook**；终点是所有原生 Vision/Language blocks 与任务读出完成、最终标量分数或检索排序已在 GPU 上得到。主时延不含模型/processor 加载、磁盘读取、MP4 seek/解码、中心裁剪与缩放、processor/tokenizer、CPU→GPU 搬运，以及 Vision block 0 前的 patch embedding。

时间一致性正式运行先对 558 个视频完成 CPU 解码和 processor，再连续执行 34 个完整 batch-16 和最后 1 个 batch-14。**没有显式 warm-up，第一批冷 forward 纳入统计。** CPU 预处理完整阶段耗时 {temporal['preprocessing_phase_wall_seconds']:.3f} s，连续 GPU 推理阶段耗时 {temporal['continuous_inference_phase_wall_seconds']:.3f} s；完整 batch 的 CPU 预处理 mean/median/P95 为 {temporal['preprocessing_full_batch_ms']['mean']:.1f}/{temporal['preprocessing_full_batch_ms']['median']:.1f}/{temporal['preprocessing_full_batch_ms']['p95']:.1f} ms，单独报告，不混入主模型时延。

功率 sweep 的每个 batch 都由不同测试视频组成；3 次 warm-up 和 30 次重复只用于稳态 batch/功率选择。性能值来自完整 558 视频正式运行。正式 distinct-batch active-window 采样均值为 {temporal['telemetry']['active_mean_w']:.2f} W；由于每批之间仍有约 {temporal['host_to_device_full_batch_ms']['mean']:.1f} ms 的 H2D 间隙，20 Hz `nvidia-smi` 窗口会低估纯计算稳态功率，所以主能耗采用同一 batch-16 sweep 的 238.18 W，并把正式 active-window 原值完整归档。

## 发现并纠正的表格问题

1. `1046.928 ms/16视频` 来自 5090D 旧口径，不是 A100；`1200.053 ms` 是 A100 batch-1 单视频均值乘 16。新公平口径是 A100 原生 **batch-16：{temporal_mean_ms:.3f} ms/16视频**。
2. 时间一致性 Ours 的 `9.384 ms / 0.754 J` 不完整，漏掉了实测 CCD 后归一化/融合、frame-to-video bridge 和最终时序头。完整论文串行边界为 **20.382 ms / 1.638 J**。
3. 时间一致性 Qwen 不能再沿用 5090D 的 `122.384 J`；本次 A100 batch-16 是 **{steady_power * temporal_mean_ms / 1000:.3f} J active**，额定上界 **{250 * temporal_mean_ms / 1000:.3f} J**。
4. LSP baseline 不是 `73.71%`；那是 ABO 行误复制。完整 1000 图测试为 **PCK@0.2 = {lsp['performance']['pck_at_0.2_torso']:.4f}**。
5. `abo_similarity10_data_only.zip` 的正式合同是**图片 query → 120 个 train 商品图片 centroid**，没有文本 query，不能标成“图文搜图”。其 Qwen R@1 是 **{t07['performance']['r_at_1']:.4f}**。
6. Ours 能耗不能每行都写固定 `0.730 J`；应按各任务的完整串行关键路径计算。当前分别为时间 {rows[0]['ours_energy_proxy_j']:.3f}、空间 {rows[1]['ours_energy_proxy_j']:.3f}、T08 {rows[2]['ours_energy_proxy_j']:.3f}、LSP {rows[4]['ours_energy_proxy_j']:.3f}、SALICON {rows[5]['ours_energy_proxy_j']:.3f} J。
7. OpenMoji 按当前决定不进入正式表；没有用低可信 baseline 填数。Caltech101 也不在用户最新这张 A100 任务表中，未混入本报告。

## 可复核文件

- `summary.csv/json`：主表、全部次指标、batch 选择和纠错记录。
- `a100_temporal_batch_sweep.png/pdf`：功率、吞吐和整批时延。
- `a100_audited_latency_energy.png/pdf`：同负载时延与能耗。
- `evidence/`：原始正式 JSON；`evidence_manifest.json` 给出 SHA256。
- `build_report.py`：从证据一键重建全部表和图。
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
