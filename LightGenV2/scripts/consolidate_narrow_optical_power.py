"""Join uncontaminated narrow latency with a separate sustained-power run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


OPTICAL_RIG_POWER_W = 80.388
A100_RATED_POWER_W = 250.0
PHYSICAL_PASS_MS = 1.0447
PASSES = {
    "lgvq_temporal": 6,
    "lgvq_spatial": 6,
    "abo_image_to_text": 6,
    "abo_image_to_image": 6,
    "lsp": 3,
    "salicon": 3,
    "openmoji": 6,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(component: str) -> str:
    if "router_core" in component:
        return "router"
    if "ccd_nonlinearity_readout_nn" in component:
        return "layer_readout"
    if component == "task_head_nn":
        return "task_head"
    if component == "bridge_nn_only":
        return "bridge"
    if "parallel_residual_nn" in component:
        return "parallel_residual"
    raise KeyError(component)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--power-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    latency_dir, power_dir, output = (
        args.latency_dir.resolve(), args.power_dir.resolve(), args.output_dir.resolve()
    )
    # Historical measurements are immutable inputs. Never overwrite a prior
    # consolidation, including partial output from an interrupted attempt.
    output.mkdir(parents=True, exist_ok=False)

    latency_report = json.loads((latency_dir / "report.json").read_text(encoding="utf-8"))
    latency_rows = read_csv(latency_dir / "component_summary.csv")
    formal_rows = {row["task"]: row for row in read_csv(latency_dir / "paper_narrow_summary.csv")}
    power_report = json.loads((power_dir / "power_summary.json").read_text(encoding="utf-8"))
    occurrences = latency_report["occurrence_contracts"]
    median_ms = {(row["task"], row["component"]): float(row["cuda_median_ms"]) for row in latency_rows}
    active_power = {
        tuple(key.split(":", 1)): float(value["mean_w"])
        for key, value in power_report["per_component"].items()
    }
    active_utilization = {
        tuple(key.split(":", 1)): float(value.get("gpu_utilization_mean_percent", 0.0))
        for key, value in power_report["per_component"].items()
    }
    idle_power = float(power_report["idle_mean_w"])

    component_energy_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    for task, contract in occurrences.items():
        by_category_energy = {name: 0.0 for name in ("router", "layer_readout", "task_head", "bridge", "parallel_residual")}
        by_category_time = dict.fromkeys(by_category_energy, 0.0)
        by_category_utilization_time = dict.fromkeys(by_category_energy, 0.0)
        for component, count in contract.items():
            duration = median_ms[task, component]
            watts = active_power[task, component]
            energy = watts * duration * count / 1000.0
            group = category(component)
            by_category_energy[group] += energy
            by_category_time[group] += duration * count
            by_category_utilization_time[group] += active_utilization[task, component] * duration * count
            component_energy_rows.append({
                "task": task,
                "component": component,
                "category": group,
                "occurrences": count,
                "formal_cuda_median_ms_per_call": duration,
                "measured_sustained_board_power_mean_w": watts,
                "energy_j": energy,
            })

        main_no_bridge_ms = float(formal_rows[task]["main_electronic_excluding_bridge_cuda_ms"])
        bridge_ms = float(formal_rows[task]["bridge_nn_cuda_ms_separate"])
        serial_ms = main_no_bridge_ms + bridge_ms
        residual_ms = float(formal_rows[task]["parallel_residual_nn_cuda_ms_separate"])
        main_energy = by_category_energy["router"] + by_category_energy["layer_readout"] + by_category_energy["task_head"]
        bridge_energy = by_category_energy["bridge"]
        residual_energy = by_category_energy["parallel_residual"]
        active_gpu_energy = main_energy + bridge_energy + residual_energy
        active_gpu_ms = serial_ms + residual_ms
        physical_ms = PASSES[task] * PHYSICAL_PASS_MS
        hybrid_ms = physical_ms + serial_ms
        idle_gpu_ms = max(0.0, hybrid_ms - active_gpu_ms)
        gpu_idle_energy = idle_power * idle_gpu_ms / 1000.0
        gpu_board_proxy = active_gpu_energy + gpu_idle_energy
        optical_physics_energy = OPTICAL_RIG_POWER_W * physical_ms / 1000.0
        optical_always_on_energy = OPTICAL_RIG_POWER_W * hybrid_ms / 1000.0
        system_energy_proxy = gpu_board_proxy + optical_always_on_energy
        system_average_power = system_energy_proxy / (hybrid_ms / 1000.0)
        active_weighted_power = active_gpu_energy / (active_gpu_ms / 1000.0)
        main_active_power = main_energy / (main_no_bridge_ms / 1000.0)
        bridge_active_power = bridge_energy / (bridge_ms / 1000.0) if bridge_ms else 0.0
        residual_active_power = residual_energy / (residual_ms / 1000.0)
        serial_active_power = (main_energy + bridge_energy) / (serial_ms / 1000.0)
        gpu_board_average_power = gpu_board_proxy / (hybrid_ms / 1000.0)
        serial_utilization = (
            by_category_utilization_time["router"]
            + by_category_utilization_time["layer_readout"]
            + by_category_utilization_time["task_head"]
            + by_category_utilization_time["bridge"]
        ) / serial_ms
        all_active_utilization = sum(by_category_utilization_time.values()) / active_gpu_ms
        rated_system_upper_energy = (OPTICAL_RIG_POWER_W + A100_RATED_POWER_W) * hybrid_ms / 1000.0
        paper_rows.append({
            "task": task,
            "physical_passes": PASSES[task],
            "physical_optics_ms": physical_ms,
            "main_electronic_ms_excluding_bridge": main_no_bridge_ms,
            "bridge_ms_separate": bridge_ms,
            "serial_electronic_ms_including_bridge": serial_ms,
            "parallel_residual_ms_separate": residual_ms,
            "hybrid_latency_ms": hybrid_ms,
            "main_electronic_measured_energy_j_excluding_bridge": main_energy,
            "main_electronic_measured_active_mean_power_w": main_active_power,
            "bridge_measured_energy_j": bridge_energy,
            "bridge_measured_active_mean_power_w": bridge_active_power,
            "serial_electronic_measured_active_mean_power_w": serial_active_power,
            "serial_electronic_measured_active_mean_gpu_utilization_percent": serial_utilization,
            "parallel_residual_measured_energy_j": residual_energy,
            "parallel_residual_measured_active_mean_power_w": residual_active_power,
            "all_active_gpu_kernel_energy_j": active_gpu_energy,
            "all_active_gpu_weighted_mean_power_w": active_weighted_power,
            "all_active_gpu_weighted_mean_utilization_percent": all_active_utilization,
            "idle_gpu_power_mean_w": idle_power,
            "idle_gpu_time_within_hybrid_proxy_ms": idle_gpu_ms,
            "gpu_board_energy_proxy_j": gpu_board_proxy,
            "gpu_board_average_power_over_hybrid_w": gpu_board_average_power,
            "optical_physics_only_energy_j": optical_physics_energy,
            "optical_rig_always_on_hybrid_energy_j": optical_always_on_energy,
            "hybrid_system_energy_proxy_j": system_energy_proxy,
            "hybrid_system_average_power_w": system_average_power,
            "hybrid_system_rated_upper_energy_j": rated_system_upper_energy,
            "hybrid_system_rated_upper_power_w": OPTICAL_RIG_POWER_W + A100_RATED_POWER_W,
        })

    for name, rows in (("component_energy.csv", component_energy_rows), ("paper_energy_summary.csv", paper_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    shutil.copy2(latency_dir / "all_per_call_timings.csv", output / "all_per_call_timings.csv")
    shutil.copy2(power_dir / "power_samples.csv", output / "power_samples.csv")
    shutil.copy2(latency_dir / "component_summary.csv", output / "latency_component_summary.csv")
    shutil.copy2(power_dir / "power_summary.json", output / "power_summary.json")
    shutil.copy2(Path(__file__).resolve(), output / "consolidate_narrow_optical_power.py")
    shutil.copy2(
        Path(__file__).with_name("profile_narrow_optical_electronics_a100.py").resolve(),
        output / "profile_narrow_optical_electronics_a100.py",
    )
    (output / "report.json").write_text(json.dumps({
        "schema_version": 1,
        "latency_source": str(latency_dir),
        "power_source": str(power_dir),
        "constants": {
            "optical_rig_power_w": OPTICAL_RIG_POWER_W,
            "a100_rated_power_w": A100_RATED_POWER_W,
            "physical_pass_ms": PHYSICAL_PASS_MS,
        },
        "measurement_policy": {
            "latency": "two trials, pooled CUDA-event median, power sampler disabled",
            "power": "separate 10 ms nvidia-smi board-power run with sustained exact component kernels",
            "main_scope": "router core from four energies + CCD nonlinearity/readout NN + task head; bridge separate; no fusion/transfer/layout/I/O",
            "residual": "energy counted separately; not added to latency because it overlaps physical optics",
            "system_energy_proxy": "optical rig at 80.388 W over hybrid window + measured active A100 component energy + measured idle A100 power over remaining hybrid window",
            "rated_upper": "(80.388 W optical rig + 250 W A100 TDP) over the complete hybrid latency",
        },
        "paper_rows": paper_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Narrow A100 optical-MoE time and power evidence\n\n"
        "Latency and power were measured in separate runs. Formal latency uses pooled CUDA-event medians. "
        "Power is 10 ms nvidia-smi A100 board power while each exact narrow component is repeated to steady state. "
        "See report.json for all inclusion and energy definitions.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(paper_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
