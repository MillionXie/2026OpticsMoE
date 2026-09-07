"""Build the formal A100 cross-task summary from archived JSON evidence."""

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


def baseline_values(report: dict[str, Any], multiplier: int = 1) -> dict[str, float]:
    latency = report.get("model_internal_cuda_ms_all_558_first_included")
    if latency is None:
        latency = report["latency_cuda_ms"]
    power = report["power"]
    return {
        "qwen_latency_mean_ms": float(latency["mean"]) * multiplier,
        "qwen_latency_median_ms": float(latency["median"]) * multiplier,
        "qwen_latency_p95_ms": float(latency["p95"]) * multiplier,
        "qwen_active_power_mean_w": float(power["active_mean_w"]),
        "qwen_active_power_peak_w": float(power["active_peak_w"]),
        "qwen_energy_measured_j": float(
            power["measured_active_energy_j_per_sample"]
        )
        * multiplier,
        "qwen_energy_idle_subtracted_j": float(
            power["idle_subtracted_energy_j_per_sample"]
        )
        * multiplier,
        "qwen_energy_rated_upper_j": float(
            power["rated_upper_bound_energy_j_per_sample"]
        )
        * multiplier,
        "a100_power_limit_w": float(power["rated_power_limit_w"]),
    }


def main() -> int:
    temporal = load("t06_temporal_qwen_a100.json")
    spatial = load("t06_spatial_qwen_a100.json")
    abo = load("t08_abo_qwen_a100.json")
    lsp = load("t02_lsp_qwen_a100.json")
    salicon = load("t03_salicon_qwen_a100.json")
    optical = load("optical_moe_electronics_a100.json")
    optical_spatial = load("optical_moe_spatial_electronics_a100.json")

    definitions = [
        {
            "task_id": "T06-temporal",
            "task": "LGVQ temporal quality",
            "metric": "SRCC",
            "ours_performance": 0.8044,
            "qwen_performance": temporal["performance"]["srcc"],
            "ours_profile": optical["tasks"]["t06"]["paper_serial_path"],
            "qwen": temporal,
            "qwen_multiplier": 16,
            "ours_workload": "16 videos x 4 frames in one optical field",
            "qwen_workload": "16 videos sequential; each video has 4 frames",
            "test_samples": 558,
        },
        {
            "task_id": "T06-spatial",
            "task": "LGVQ spatial quality",
            "metric": "SRCC",
            "ours_performance": 0.6393237619075796,
            "qwen_performance": spatial["performance"]["srcc"],
            "ours_profile": optical_spatial["tasks"]["t06_spatial"]["paper_serial_path"],
            "qwen": spatial,
            "qwen_multiplier": 1,
            "ours_workload": "1 video x 4 frames in a 2x2 optical field",
            "qwen_workload": "1 video x 4 frames",
            "test_samples": 558,
        },
        {
            "task_id": "T08",
            "task": "ABO image-to-title retrieval",
            "metric": "R@1",
            "ours_performance": 0.7983333333333333,
            "qwen_performance": abo["performance"]["recall_at_1"],
            "ours_profile": optical["tasks"]["t08"]["paper_serial_path"],
            "qwen": abo,
            "qwen_multiplier": 1,
            "ours_workload": "1 image query against 100 titles",
            "qwen_workload": "1 image query against 100 precomputed titles",
            "test_samples": 2400,
        },
        {
            "task_id": "T02",
            "task": "LSP keypoint detection",
            "metric": "PCK@0.2",
            "ours_performance": 0.5773,
            "qwen_performance": lsp["performance"]["pck_at_0.2_torso"],
            "ours_profile": optical["tasks"]["t02"]["paper_serial_path"],
            "qwen": lsp,
            "qwen_multiplier": 1,
            "ours_workload": "1 image",
            "qwen_workload": "1 image",
            "test_samples": 1000,
        },
        {
            "task_id": "T03",
            "task": "SALICON saliency",
            "metric": "CC",
            "ours_performance": 0.8291,
            "qwen_performance": salicon["performance"]["cc"],
            "ours_profile": optical["tasks"]["t03"]["paper_serial_path"],
            "qwen": salicon,
            "qwen_multiplier": 1,
            "ours_workload": "1 image",
            "qwen_workload": "1 image",
            "test_samples": 5000,
        },
    ]

    rows: list[dict[str, Any]] = []
    for item in definitions:
        profile = item.pop("ours_profile")
        qwen = item.pop("qwen")
        multiplier = int(item.pop("qwen_multiplier"))
        row = dict(item)
        row.update(baseline_values(qwen, multiplier))
        row["ours_latency_ms"] = float(profile["estimated_wall_ms_per_call"])
        row["ours_serial_electronic_ms"] = float(
            profile["serial_electronic_wall_ms_per_call"]
        )
        row["ours_physical_passes"] = int(profile["physical_passes"])
        row["ours_physical_time_ms"] = float(profile["physical_time_ms_per_call"])
        row["ours_power_proxy_w"] = OPTICAL_POWER_W
        row["ours_energy_proxy_j"] = float(profile["optical_rig_energy_proxy_j_per_call"])
        row["speedup_qwen_over_ours"] = (
            row["qwen_latency_mean_ms"] / row["ours_latency_ms"]
        )
        row["energy_ratio_measured_qwen_over_ours"] = (
            row["qwen_energy_measured_j"] / row["ours_energy_proxy_j"]
        )
        row["energy_ratio_rated_upper_qwen_over_ours"] = (
            row["qwen_energy_rated_upper_j"] / row["ours_energy_proxy_j"]
        )
        rows.append(row)

    fields = list(rows[0])
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    detail = {
        "schema_version": 1,
        "status": "complete",
        "gpu": "NVIDIA A100-PCIE-40GB",
        "physical_gpu_index": 6,
        "a100_rated_power_limit_w": 250.0,
        "optical_rig_power_proxy_w": OPTICAL_POWER_W,
        "rows": rows,
        "performance_detail": {
            "lgvq_temporal": {
                "ours": {"srcc": 0.8044, "plcc": 0.8180, "rmse": 7.991, "mae": 5.992},
                "qwen": temporal["performance"],
            },
            "lgvq_spatial": {
                "ours": {
                    "srcc": 0.6393237619075796,
                    "plcc": 0.6743486500135264,
                    "rmse": 8.451935768127441,
                    "mae": 6.646285533905029,
                },
                "qwen": spatial["performance"],
            },
            "abo_image_to_title": {
                "ours": {
                    "recall_at_1": 0.7983333333333333,
                    "recall_at_5": 0.95375,
                    "recall_at_10": 0.9858333333333333,
                    "mrr": 0.867616268422718,
                },
                "qwen": abo["performance"],
            },
            "lsp": {
                "ours": {"pck_at_0.2_torso": 0.5773, "pckh_at_0.5_head": 0.7363},
                "qwen": lsp["performance"],
            },
            "salicon": {
                "ours": {
                    "cc": 0.8291,
                    "kld": 0.1330,
                    "sim": 0.8063,
                    "nss": 0.9283,
                    "auc_judd": 0.7631,
                    "mae": 0.0890,
                },
                "qwen": salicon["performance"],
            },
        },
        "contracts": {
            "qwen_latency": "first native Transformer block input through task output on GPU",
            "lgvq_protocol": "one process/model load, zero explicit warmup, all 558 test videos, first video included",
            "other_qwen_timing": "50 explicit warmup forwards followed by 200 deterministic timing samples",
            "ours_latency": "physical-pass constants plus measured A100 serial post-CCD electronics and task head; router post, next-SLM reconstruction, and parallel residual excluded",
            "ours_energy": "80.388 W optical-rig proxy multiplied by the composed ours latency; not a wired end-to-end energy measurement",
            "qwen_energy": "measured A100 board active power multiplied by mean model-boundary latency; rated upper bound uses 250 W",
        },
    }
    (ROOT / "summary.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evidence_manifest = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(EVIDENCE.glob("*"))
        if path.is_file()
    }
    (ROOT / "evidence_manifest.json").write_text(
        json.dumps(evidence_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = [row["task_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    axes[0].bar(x - width / 2, [r["ours_latency_ms"] for r in rows], width, label="Optical MoE")
    axes[0].bar(x + width / 2, [r["qwen_latency_mean_ms"] for r in rows], width, label="Qwen baseline")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Workload latency (ms, log scale)")
    axes[0].set_title("a  A100 latency", loc="left", fontweight="bold")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, [r["ours_energy_proxy_j"] for r in rows], width, label="Optical proxy")
    axes[1].bar(x + width / 2, [r["qwen_energy_measured_j"] for r in rows], width, label="A100 measured")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Workload energy (J, log scale)")
    axes[1].set_title("b  Energy", loc="left", fontweight="bold")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"a100_latency_energy.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.9), constrained_layout=True)
    ax.bar(x - width / 2, [r["ours_performance"] for r in rows], width, label="Optical MoE")
    ax.bar(x + width / 2, [r["qwen_performance"] for r in rows], width, label="Qwen baseline")
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Primary metric (task-specific)")
    ax.set_title("A100 comparison: performance", loc="left", fontweight="bold")
    ax.set_xticks(x, [f"{r['task_id']}\n{r['metric']}" for r in rows])
    ax.legend(frameon=False, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"a100_performance.{suffix}", dpi=300)
    plt.close(fig)

    table = [
        "| 任务 | 指标 | 光学仿真 | Qwen A100 | Ours 时延 | Qwen 同负载时延 | 加速比 | Ours 能耗 | Qwen 实测能耗 | 能效比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {task_id} | {metric} | {ours_performance:.4f} | {qwen_performance:.4f} | "
            "{ours_latency_ms:.3f} | {qwen_latency_mean_ms:.3f} | {speedup_qwen_over_ours:.2f}x | "
            "{ours_energy_proxy_j:.3f} | {qwen_energy_measured_j:.3f} | "
            "{energy_ratio_measured_qwen_over_ours:.2f}x |".format(**row)
        )
    readme = """# A100 跨任务正式结果（2026-09-07）

本目录把此前 RTX 5090D 上的横向表全部换成同一台物理 A100 的复测结果。所有
Qwen 时延与板卡功率均来自物理 GPU 6：`NVIDIA A100-PCIE-40GB`（40 GB，功率上限
250 W）。按当前决定，OpenMoji 不进入本表。

## 主表

{table}

其中 LGVQ 时间质量的比较负载严格统一：光学 MoE 在一幅光场中并行承载
`16 视频 × 4 帧`，Qwen baseline 使用 batch 1 顺序处理 16 个四帧视频，所以 Qwen
时延和能耗均乘 16。其余任务都是单样本负载。

## 测量边界与样本量

| 任务 | 性能测试量 | 时延采样 | 显式 warm-up | Qwen mean / median / P95 (ms) | active mean / peak (W) |
|---|---:|---:|---:|---:|---:|
| LGVQ 时间 | 558 视频 | 558 视频 | 0 | {tmean:.3f} / {tmedian:.3f} / {tp95:.3f}（单视频） | {tpower:.2f} / {tpeak:.2f} |
| LGVQ 空间 | 558 视频 | 558 视频 | 0 | {smean:.3f} / {smedian:.3f} / {sp95:.3f} | {spower:.2f} / {speak:.2f} |
| ABO 图搜文 | 2400 图 | 200 | 50 | {amean:.3f} / {amedian:.3f} / {ap95:.3f} | {apower:.2f} / {apeak:.2f} |
| LSP | 1000 图 | 200 | 50 | {lmean:.3f} / {lmedian:.3f} / {lp95:.3f} | {lpower:.2f} / {lpeak:.2f} |
| SALICON | 5000 图 | 200 | 50 | {cmean:.3f} / {cmedian:.3f} / {cp95:.3f} | {cpower:.2f} / {cpeak:.2f} |

Qwen 核心计时从第一个原生 Transformer block 的输入开始，到任务输出结束；不含
视频/图片读取、解码、processor 和模型加载。LGVQ 是单次进程和单次模型加载，零显式
warm-up，完整顺序计时 558 个测试视频，第一条冷样本也保留。LSP、SALICON 和 ABO
在 50 次 warm-up 后做 200 次固定样本计时；性能仍分别在完整 1000、5000、2400 条
测试集上计算。

## 光学时间与能耗口径

- 单次物理光路固定为 `0.714 + 0.100 + 0.500 = 1.314 ms`；LGVQ/ABO 为 6 次，
  LSP/SALICON 为 3 次。
- Ours 的论文时间 = 物理光路 + 实测的 CCD 后串行归一化/融合/必要模态 bridge/
  任务头。router 后处理、下一张 SLM 图重建及可与光传播并行的电残差不计入临界路径。
- 电残差仍被独立测量，各任务残差均小于 1.314 ms，因此可以被单层光传播覆盖。
- Ours 能耗是 `80.388 W × 上述组合时延` 的实验台功率代理，不是插座端整机实测。
- Qwen measured 是 A100 active mean 板卡功率乘模型边界均值时延；`summary.csv` 还保存
  idle-subtracted 能耗和严格的 250 W 额定功率上界，三种口径不能混写为同一列。

## 可复核文件

- `summary.csv`、`summary.json`：正式主表和全部次指标。
- `a100_latency_energy.png/.pdf`：同负载时延、能耗图。
- `a100_performance.png/.pdf`：各任务主指标图；不同任务的指标名称不同，不能横向解释数值大小。
- `evidence/`：精简后的原始 JSON 报告，不包含权重、数据集、缓存或逐样本预测。
- `evidence_manifest.json`：每份证据的 SHA256 与字节数。

在本目录执行 `python build_report.py` 可由证据重新生成全部表格和图。
""".format(
        table="\n".join(table),
        tmean=rows[0]["qwen_latency_mean_ms"] / 16,
        tmedian=rows[0]["qwen_latency_median_ms"] / 16,
        tp95=rows[0]["qwen_latency_p95_ms"] / 16,
        tpower=rows[0]["qwen_active_power_mean_w"],
        tpeak=rows[0]["qwen_active_power_peak_w"],
        smean=rows[1]["qwen_latency_mean_ms"],
        smedian=rows[1]["qwen_latency_median_ms"],
        sp95=rows[1]["qwen_latency_p95_ms"],
        spower=rows[1]["qwen_active_power_mean_w"],
        speak=rows[1]["qwen_active_power_peak_w"],
        amean=rows[2]["qwen_latency_mean_ms"],
        amedian=rows[2]["qwen_latency_median_ms"],
        ap95=rows[2]["qwen_latency_p95_ms"],
        apower=rows[2]["qwen_active_power_mean_w"],
        apeak=rows[2]["qwen_active_power_peak_w"],
        lmean=rows[3]["qwen_latency_mean_ms"],
        lmedian=rows[3]["qwen_latency_median_ms"],
        lp95=rows[3]["qwen_latency_p95_ms"],
        lpower=rows[3]["qwen_active_power_mean_w"],
        lpeak=rows[3]["qwen_active_power_peak_w"],
        cmean=rows[4]["qwen_latency_mean_ms"],
        cmedian=rows[4]["qwen_latency_median_ms"],
        cp95=rows[4]["qwen_latency_p95_ms"],
        cpower=rows[4]["qwen_active_power_mean_w"],
        cpeak=rows[4]["qwen_active_power_peak_w"],
    )
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
