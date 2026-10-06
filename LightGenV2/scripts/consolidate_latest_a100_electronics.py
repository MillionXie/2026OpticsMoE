"""Consolidate repeated A100 optical-electronics audits without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - point) + ordered[hi] * (point - lo)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "std": statistics.pstdev(values), "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95), "minimum": min(values), "maximum": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def occurrences(task: str, language: bool) -> dict[str, int]:
    if task == "lgvq_temporal":
        return {
            "frame_router_ccd_to_weights": 1, "frame_ccd_to_fusion": 2,
            "frame_to_video_bridge": 1, "video_router_ccd_to_weights": 1,
            "video_ccd_to_fusion": 2, "task_head": 1,
            "frame_parallel_residual": 2, "video_parallel_residual": 2,
        }
    result = {
        "router_ccd_to_weights": 2, "vision_ccd_to_fusion": 2,
        "vision_parallel_residual": 2, "task_head": 1,
    }
    if language:
        result.update({"language_ccd_to_fusion": 2, "language_parallel_residual": 2})
    if task in {"lgvq_spatial", "openmoji"}:
        result["language_to_vision_bridge"] = 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-trials", nargs="+", type=Path, required=True)
    parser.add_argument("--power-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-script", type=Path)
    parser.add_argument("--latest-source-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in args.latency_trials]
    if len({report["environment"]["device"] for report in reports}) != 1:
        raise RuntimeError("Latency trials used different devices")
    if any("A100" not in report["environment"]["device"] for report in reports):
        raise RuntimeError("Only A100 latency trials may be consolidated")
    power_report = json.loads((args.power_run / "report.json").read_text(encoding="utf-8"))
    power = power_report["power_measurement"]
    if power is None:
        raise RuntimeError("Power run has no power samples")

    raw_rows: list[dict[str, Any]] = []
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for trial_index, directory in enumerate(args.latency_trials, 1):
        with (directory / "per_call_timings.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                row = dict(row)
                row["trial"] = trial_index
                raw_rows.append(row)
                key = (row["task"], row["component"])
                bucket = values.setdefault(key, {"event": [], "wall": []})
                bucket["event"].append(float(row["cuda_event_ms"]))
                bucket["wall"].append(float(row["synchronized_wall_ms"]))
    # Put trial first while retaining every original column.
    raw_rows = [{"trial": row.pop("trial"), **row} for row in raw_rows]
    write_csv(output / "all_per_call_timings.csv", raw_rows)

    component_rows: list[dict[str, Any]] = []
    component_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(values):
        event, wall = summarize(values[key]["event"]), summarize(values[key]["wall"])
        component_stats[key] = {"cuda_event_ms": event, "synchronized_wall_ms": wall}
        component_rows.append({
            "task": key[0], "component": key[1], "trials": len(args.latency_trials),
            "calls": len(values[key]["event"]),
            **{f"cuda_{name}_ms": value for name, value in event.items()},
            **{f"wall_{name}_ms": value for name, value in wall.items()},
        })
    write_csv(output / "pooled_component_summary.csv", component_rows)

    constants = reports[-1]["constants"]
    physical_pass = float(constants["physical_pass_ms"])
    optical_power = float(constants["optical_rig_power_w"])
    rated_power = float(constants["a100_rated_power_w"])
    idle_power = float(power["idle_mean_w"])
    phase_power = power["per_component"]
    task_rows: list[dict[str, Any]] = []
    task_payloads: dict[str, Any] = {}
    for task, task_report in reports[-1]["tasks"].items():
        spec = task_report["specification"]
        counts = occurrences(task, bool(spec["language"]))
        event_medians = {component: component_stats[(task, component)]["cuda_event_ms"]["median"] for component in counts}
        serial_names = [name for name in counts if "parallel_residual" not in name]
        residual_names = [name for name in counts if "parallel_residual" in name]
        serial_ms = sum(event_medians[name] * counts[name] for name in serial_names)
        residual_ms = sum(event_medians[name] * counts[name] for name in residual_names)
        passes = int(spec["feature_passes"]) + int(spec["router_passes"])
        physical_ms = passes * physical_pass
        hybrid_ms = physical_ms + serial_ms
        electronic_active_j = 0.0
        component_energy: dict[str, float] = {}
        for component, count in counts.items():
            tag = f"{task}:{component}"
            watts = float(phase_power.get(tag, {}).get("mean_w", power["active_mean_w"]))
            energy = watts * event_medians[component] * count / 1000.0
            component_energy[component] = energy
            electronic_active_j += energy
        optical_physics_j = optical_power * physical_ms / 1000.0
        optical_always_on_j = optical_power * hybrid_ms / 1000.0
        # The board remains powered during physical waits.  This conservative
        # proxy charges idle board power only to time not already represented
        # by measured active electronic kernels.
        active_kernel_ms = serial_ms + residual_ms
        gpu_idle_wait_ms = max(0.0, hybrid_ms - active_kernel_ms)
        gpu_board_proxy_j = electronic_active_j + idle_power * gpu_idle_wait_ms / 1000.0
        task_payloads[task] = {
            "specification": spec, "performance_identity": reports[-1]["metric_identities"][task],
            "component_occurrences": counts, "pooled_component_medians_cuda_event_ms": event_medians,
            "serialized_electronic_ms": serial_ms, "parallel_residual_sum_ms": residual_ms,
            "physical_passes": passes, "physical_only_ms": physical_ms,
            "formal_hybrid_latency_ms": hybrid_ms,
            "formal_latency_scope": "physical passes + serialized neural-only electronics; excludes SLM layout/reload; residual executes in parallel and is energy-counted",
            "all_individual_residuals_covered": all(event_medians[name] <= physical_pass for name in residual_names),
            "component_active_gpu_energy_j": component_energy,
            "measured_active_gpu_kernel_energy_j": electronic_active_j,
            "gpu_board_with_idle_wait_energy_proxy_j": gpu_board_proxy_j,
            "optical_rig_physics_window_energy_j": optical_physics_j,
            "optical_rig_always_on_hybrid_window_energy_j": optical_always_on_j,
            "hybrid_energy_proxy_j": optical_always_on_j + gpu_board_proxy_j,
            "hybrid_rated_gpu_upper_j": optical_always_on_j + rated_power * (serial_ms + residual_ms) / 1000.0,
        }
        identity = reports[-1]["metric_identities"][task]
        router_ms = sum(
            event_medians[name] * counts[name]
            for name in counts if "router" in name and name.endswith("weights")
        )
        ccd_fusion_ms = sum(
            event_medians[name] * counts[name]
            for name in counts if name.endswith("ccd_to_fusion")
        )
        bridge_ms = sum(
            event_medians[name] * counts[name]
            for name in counts if name.endswith("bridge")
        )
        task_rows.append({
            "task": task, "metric": identity["metric"], "performance": identity["value"],
            "logical_samples_per_call": spec["logical_samples_per_call"], "physical_passes": passes,
            "physical_only_ms": physical_ms, "serialized_electronic_cuda_ms": serial_ms,
            "parallel_residual_cuda_ms": residual_ms,
            "router_weights_cuda_ms": router_ms, "ccd_readout_nonlinearity_fusion_cuda_ms": ccd_fusion_ms,
            "bridge_cuda_ms": bridge_ms,
            "task_head_cuda_ms": event_medians["task_head"], "formal_hybrid_latency_ms": hybrid_ms,
            "formal_hybrid_latency_ms_per_logical_sample": hybrid_ms / int(spec["logical_samples_per_call"]),
            "residual_covered": task_payloads[task]["all_individual_residuals_covered"],
            "measured_active_gpu_kernel_energy_j": electronic_active_j,
            "hybrid_energy_proxy_j": optical_always_on_j + gpu_board_proxy_j,
            "hybrid_rated_gpu_upper_j": task_payloads[task]["hybrid_rated_gpu_upper_j"],
        })
    write_csv(output / "paper_summary.csv", task_rows)

    consolidated = {
        "schema_version": 1,
        "selection_policy": "No fastest-trial selection. Pool every per-call sample from both independent latency-only trials and take the pooled median per component.",
        "latency_trials": [str(path.resolve()) for path in args.latency_trials],
        "power_run": str(args.power_run.resolve()),
        "latency_device": reports[-1]["environment"],
        "constants": constants,
        "power_measurement": power,
        "total_raw_latency_rows": len(raw_rows),
        "calls_per_component": 1000 * len(args.latency_trials),
        "tasks": task_payloads,
        "disclosures": [
            "CUDA Event is the formal optical-electronics timing basis; synchronized wall is retained in pooled_component_summary.csv.",
            "Power and latency were measured in separate runs to avoid conflating sampling overhead with latency.",
            "ABO image-to-text 0.8058 is historical 1934/2400; current-environment replay differs by 3 samples.",
            "LSP bound artifact is 0.7347857143, not exactly the screenshot's 0.7353; this discrepancy remains unresolved.",
        ],
    }
    (output / "consolidated_report.json").write_text(json.dumps(consolidated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.power_run / "power_samples.csv", output / "power_samples.csv")
    source_runs = output / "source_runs"
    for label, directory in [
        *[(f"latency_trial_{index}", path) for index, path in enumerate(args.latency_trials, 1)],
        ("power_trial", args.power_run),
    ]:
        target = source_runs / label
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "report.json", "component_summary.csv", "command.txt", "environment.txt",
            "nvidia_smi_before.txt", "nvidia_smi_after.txt", "run.log", "git_status.txt",
            "SHA256SUMS.txt",
        ):
            source = directory / name
            if source.is_file():
                shutil.copy2(source, target / name)
    if args.measurement_script:
        shutil.copy2(args.measurement_script, output / args.measurement_script.name)
    if args.latest_source_dir:
        target = output / "latest_lgvq_source_snapshot"
        target.mkdir(exist_ok=True)
        for name in ("settings.py", "modeling.py"):
            shutil.copy2(args.latest_source_dir / name, target / name)
    readme = f"""# A100 最新光学 MoE 电子部分计时（2026-09-14）

## 正式口径

- GPU：{reports[-1]['environment']['device']}；FP32 eager，未使用 `torch.compile`。
- 每个组件每次实验先预热 50 次，再测 1000 次；两次独立纯延迟实验共保留 {len(raw_rows):,} 条逐调用记录。
- 论文表采用两次实验全部样本的 pooled CUDA Event median，不挑选较快的一次。
- 正式电子时间只包括 CCD 读出/非线性/融合、光路由权重计算、必要 bridge 和任务头。
- SLM 画布排版、scatter、文件 I/O 不计入神经网络推理；这些操作仍保存在各原始运行的 full-reload audit 中。
- 电残差与光过程并行，因此不加到延迟；其时间与能量均单独保留。
- 每次物理光过程为 0.714 + 0.300 + 0.0307 = {physical_pass:.4f} ms。

## 文件

- `all_per_call_timings.csv`：两次纯延迟实验的全部逐调用 Event/Wall 数据。
- `pooled_component_summary.csv`：各组件 2000 次调用的 mean/median/std/P05/P95/min/max。
- `paper_summary.csv`：可直接填表的性能绑定、电子明细、物理时间、总时间和能耗代理。
- `consolidated_report.json`：完整机器可读合同、指标来源、能耗定义和组件次数。
- `power_samples.csv`：独立功率实验的 10 ms 板卡功率原始采样。
- `source_runs/`：两次延迟实验与一次功率实验各自的原报告、命令、环境和日志。
- `latest_lgvq_source_snapshot/`：0.6710 空间版本实际测速采用的最新源码快照。

## 必须披露的指标差异

- ABO 图搜文 0.8058 是固定 epoch-25 EMA 的历史 1934/2400；当前环境完整复核少 3 张，不能写成完全复现。
- LSP 当前绑定的 `final_report.json` 是 0.7347857143，截图的 0.7353 尚未找到逐样本/权重证据，论文定稿前需确认来源。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    checks = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checks.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output), "raw_rows": len(raw_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
