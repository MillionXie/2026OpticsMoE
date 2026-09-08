"""Build the batch-3/4 audit and recompute hybrid optical+A100 energy."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
OPTICAL_POWER_W = 80.388
PHYSICAL_PASS_MS = 1.314
GPU_RATED_POWER_W = 250.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


fusion = read_json(EVIDENCE / "ccd_fusion_only_a100.json")
heads = read_json(EVIDENCE / "task_heads_a100.json")
common_power = read_json(EVIDENCE / "optical_moe_electronics_a100.json")
spatial_power = read_json(EVIDENCE / "optical_moe_spatial_electronics_a100.json")
openmoji_power = read_json(EVIDENCE / "t04_optical_electronics_power_a100.json")
batch4 = read_json(EVIDENCE / "formal_batch4" / "report.json")
batch2 = read_json(EVIDENCE / "formal_batch2" / "report.json")
batch3 = read_json(EVIDENCE / "sweep_batch3" / "result.json")
batch_sweep = read_json(EVIDENCE / "t06_temporal_batch_sweep_a100.json")
batch16 = read_json(EVIDENCE / "t06_temporal_batch16_formal_a100.json")


head_rows = {(row["task"], row["component"]): row for row in heads["components"]}


TASKS = {
    "lgvq_temporal": {
        "label": "LGVQ temporal, 16 videos parallel",
        "profile": "t06",
        "power_source": common_power,
        "physical_passes": 6,
        "fusion": [
            ("frame_neural_ccd_to_fusion", "frame_ccd_to_fusion", 2),
            ("video_neural_ccd_to_fusion", "video_ccd_to_fusion", 2),
        ],
        "serial_extra": [
            ("LGVQ temporal, 16 videos parallel", "required_frame_to_video_bridge", "frame_to_video_bridge"),
            ("LGVQ temporal, 16 videos parallel", "task_head", "task_head"),
        ],
        "residual": [("frame_parallel_residual", 2), ("video_parallel_residual", 2)],
    },
    "lgvq_spatial": {
        "label": "LGVQ spatial, one video with 4 frames",
        "profile": "t06_spatial",
        "power_source": spatial_power,
        "physical_passes": 6,
        "fusion": [
            ("spatial_vision_neural_ccd_to_fusion", "spatial_vision_ccd_to_fusion", 2),
            ("spatial_language_neural_ccd_to_fusion", "spatial_language_ccd_to_fusion", 2),
        ],
        "serial_extra": [
            ("LGVQ spatial, single video 4 frames", "required_frame_to_sequence_bridge", "spatial_frame_to_sequence_bridge"),
            ("LGVQ spatial, single video 4 frames", "task_head", "spatial_task_head"),
        ],
        "residual": [
            ("spatial_vision_parallel_residual", 2),
            ("spatial_language_parallel_residual", 2),
        ],
    },
    "abo_image_to_text": {
        "label": "ABO image-to-title T08",
        "profile": "t08",
        "power_source": common_power,
        "physical_passes": 6,
        "fusion": [
            ("vision_neural_ccd_to_fusion", "vision_ccd_to_fusion", 2),
            ("language_neural_ccd_to_fusion", "language_ccd_to_fusion", 2),
        ],
        "serial_extra": [("ABO image-to-title T08", "task_head", "task_head")],
        "residual": [("vision_parallel_residual", 2), ("language_parallel_residual", 2)],
    },
    "lsp": {
        "label": "LSP keypoint",
        "profile": "t02",
        "power_source": common_power,
        "physical_passes": 3,
        "fusion": [("vision_neural_ccd_to_fusion", "vision_ccd_to_fusion", 2)],
        "serial_extra": [("LSP keypoint", "task_head", "task_head")],
        "residual": [("vision_parallel_residual", 2)],
    },
    "salicon": {
        "label": "SALICON saliency",
        "profile": "t03",
        "power_source": common_power,
        "physical_passes": 3,
        "fusion": [("vision_neural_ccd_to_fusion", "vision_ccd_to_fusion", 2)],
        "serial_extra": [("SALICON saliency", "task_head", "task_head")],
        "residual": [("vision_parallel_residual", 2)],
    },
    "openmoji": {
        "label": "OpenMoji interaction",
        "profile": "t04",
        "power_source": openmoji_power,
        "physical_passes": 6,
        "fusion": [
            ("vision_neural_ccd_to_fusion", "vision_ccd_to_fusion", 2),
            ("language_neural_ccd_to_fusion", "language_ccd_to_fusion", 2),
        ],
        "serial_extra": [
            ("OpenMoji interaction", "required_language_to_vision_bridge", "language_to_vision_bridge"),
            ("OpenMoji interaction", "task_head", "task_head"),
        ],
        "residual": [("vision_parallel_residual", 2), ("language_parallel_residual", 2)],
    },
}


def component(profile: str, name: str) -> dict[str, Any]:
    rows = fusion["tasks"][profile]["components"]
    return next(row for row in rows if row["component"] == name)


def source_component(source: dict[str, Any], profile: str, name: str) -> dict[str, Any]:
    rows = source["tasks"][profile]["components"]
    return next(row for row in rows if row["component"] == name)


def power(source: dict[str, Any], profile: str, name: str) -> float:
    return float(source["power_measurement"]["per_component"][f"{profile}:{name}"]["mean_w"])


energy_rows: list[dict[str, Any]] = []
for task_id, spec in TASKS.items():
    profile = spec["profile"]
    source = spec["power_source"]
    idle_w = float(source["power_measurement"]["idle_mean_w"])
    physical_ms = float(spec["physical_passes"]) * PHYSICAL_PASS_MS
    serial: list[dict[str, Any]] = []
    for measured_name, power_name, occurrences in spec["fusion"]:
        row = component(profile, measured_name)
        serial.append(
            {
                "name": power_name,
                "occurrences": occurrences,
                "wall_median_ms": float(row["synchronized_wall_ms"]["median"]),
                "active_mean_w": power(source, profile, power_name),
            }
        )
    for task_name, head_name, power_name in spec["serial_extra"]:
        row = head_rows[(task_name, head_name)]
        serial.append(
            {
                "name": power_name,
                "occurrences": 1,
                "wall_median_ms": float(row["wall_median_ms"]),
                "active_mean_w": power(source, profile, power_name),
            }
        )
    residual: list[dict[str, Any]] = []
    for name, occurrences in spec["residual"]:
        row = source_component(source, profile, name)
        residual.append(
            {
                "name": name,
                "occurrences": occurrences,
                "wall_median_ms": float(row["synchronized_wall_ms"]["median"]),
                "active_mean_w": power(source, profile, name),
            }
        )
    serial_ms = sum(row["wall_median_ms"] * row["occurrences"] for row in serial)
    wall_ms = physical_ms + serial_ms
    optical_j = OPTICAL_POWER_W * wall_ms / 1000.0
    gpu_idle_j = idle_w * wall_ms / 1000.0
    serial_increment_j = sum(
        max(0.0, row["active_mean_w"] - idle_w)
        * row["wall_median_ms"]
        * row["occurrences"]
        / 1000.0
        for row in serial
    )
    residual_increment_j = sum(
        max(0.0, row["active_mean_w"] - idle_w)
        * row["wall_median_ms"]
        * row["occurrences"]
        / 1000.0
        for row in residual
    )
    gpu_j = gpu_idle_j + serial_increment_j + residual_increment_j
    combined_j = optical_j + gpu_j
    energy_rows.append(
        {
            "task_id": task_id,
            "task": spec["label"],
            "physical_passes": spec["physical_passes"],
            "physical_ms": physical_ms,
            "serial_electronic_wall_ms": serial_ms,
            "composed_wall_ms": wall_ms,
            "optical_rig_power_w": OPTICAL_POWER_W,
            "optical_rig_energy_j": optical_j,
            "a100_idle_power_w": idle_w,
            "a100_idle_base_energy_j": gpu_idle_j,
            "a100_serial_incremental_energy_j": serial_increment_j,
            "a100_parallel_residual_incremental_energy_j": residual_increment_j,
            "a100_board_energy_proxy_j": gpu_j,
            "combined_optical_plus_a100_energy_j": combined_j,
            "combined_average_power_w": combined_j / (wall_ms / 1000.0),
            "combined_rated_upper_energy_j": optical_j
            + GPU_RATED_POWER_W * wall_ms / 1000.0,
            "serial_components": serial,
            "parallel_residual_components": residual,
            "excluded": "router post, next-SLM layout/rebuild, file/image processing",
        }
    )


qwen_reference = {
    "lgvq_temporal": {
        "time_ms": float(batch4["same_workload_16_video_model_ms"]),
        "energy_j": float(batch4["same_workload_16_video_measured_active_energy_j"]),
        "source": "formal batch=4, four measured calls for the same 16-video workload",
    },
    "lgvq_spatial": {"time_ms": 74.438, "energy_j": 6.242, "source": "audited A100 table"},
    "abo_image_to_text": {"time_ms": 43.963, "energy_j": 3.792, "source": "audited A100 table"},
    "lsp": {"time_ms": 18.016, "energy_j": 1.452, "source": "audited A100 table"},
    "salicon": {"time_ms": 19.141, "energy_j": 1.551, "source": "audited A100 table"},
    "openmoji": {"time_ms": 49.053, "energy_j": 4.476, "source": "audited A100 table"},
}
comparison_rows = []
for row in energy_rows:
    qwen = qwen_reference[row["task_id"]]
    comparison_rows.append(
        {
            "task_id": row["task_id"],
            "task": row["task"],
            "ours_time_ms": row["composed_wall_ms"],
            "qwen_time_ms": qwen["time_ms"],
            "speedup_x": qwen["time_ms"] / row["composed_wall_ms"],
            "ours_combined_energy_j": row["combined_optical_plus_a100_energy_j"],
            "qwen_energy_j": qwen["energy_j"],
            "energy_reduction_x": qwen["energy_j"] / row["combined_optical_plus_a100_energy_j"],
            "qwen_source": qwen["source"],
        }
    )


summary = {
    "schema_version": 1,
    "canonical_contract": {
        "ours_timing": "six/three physical 1.314ms passes + strict synchronized CCD-to-fusion + required bridge + complete task head",
        "ours_energy": "80.388W optical rig plus measured A100 board-power proxy over the same composed wall boundary",
        "qwen_formal": "performance, latency, and energy must come from the same full-558 formal run; sweep power is occupancy evidence only",
        "legacy_ours_times_ms_from_previous_table": {
            "lgvq_temporal": 10.637,
            "lgvq_spatial": 10.600,
            "abo_image_to_text": 9.941,
            "lsp": 5.861,
            "salicon": 6.125,
            "openmoji": 11.236,
        },
        "legacy_status": "superseded because those rows predate the retained strict full-task-head benchmark",
    },
    "batch2_formal": batch2,
    "batch4_formal": batch4,
    "batch3_sweep": batch3,
    "batch_sweep_reference": batch_sweep,
    "ours_energy_method": {
        "formula": "E_total = 80.388W*T_wall + P_idle_A100*T_wall + sum((P_component-P_idle)*T_component)",
        "parallel_residual": "time is covered by the physical pass, but its incremental A100 energy is included",
        "power_sources": "10ms nvidia-smi component loops on the same A100; strict wall-median component durations",
        "warning": "composed proxy, not a wired wall-plug measurement",
    },
    "ours_energy": energy_rows,
    "comparisons": comparison_rows,
    "source_files": {},
}
for path in sorted(EVIDENCE.rglob("*")):
    if path.is_file():
        summary["source_files"][str(path.relative_to(HERE)).replace("\\", "/")] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

(HERE / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
with (HERE / "ours_combined_energy.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in energy_rows
    ]
    writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
    writer.writeheader()
    writer.writerows(scalar_rows)
with (HERE / "comparisons.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)

formal_batch_rows = []
for report in (batch2, batch4, batch16):
    formal_batch_rows.append(
        {
            "batch_size": report["batch_size_videos"],
            "test_videos": report["test_videos"],
            "srcc": report["performance"]["srcc"],
            "plcc": report["performance"]["plcc"],
            "mean_ms_per_batch": report["model_boundary_full_batch_cuda_ms"]["mean"],
            "same_16_video_ms": report["same_workload_16_video_model_ms"],
            "formal_active_mean_w": report["telemetry"]["active_mean_w"],
            "same_16_video_energy_j": report["same_workload_16_video_measured_active_energy_j"],
            "ours_canonical_ms": energy_rows[0]["composed_wall_ms"],
            "ours_combined_energy_j": energy_rows[0]["combined_optical_plus_a100_energy_j"],
            "speedup_x": report["same_workload_16_video_model_ms"] / energy_rows[0]["composed_wall_ms"],
            "energy_reduction_x": report["same_workload_16_video_measured_active_energy_j"] / energy_rows[0]["combined_optical_plus_a100_energy_j"],
        }
    )
with (HERE / "temporal_formal_batch_comparison.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(formal_batch_rows[0]))
    writer.writeheader()
    writer.writerows(formal_batch_rows)

print(json.dumps({
    "batch4": {
        "performance": batch4["performance"],
        "mean_ms_per_4_videos": batch4["model_boundary_full_batch_cuda_ms"]["mean"],
        "active_mean_w": batch4["telemetry"]["active_mean_w"],
        "active_energy_j_per_4_videos": batch4["telemetry"]["measured_active_energy_j_per_batch"],
    },
    "batch3": {
        "mean_ms_per_3_videos": batch3["model_boundary_batch_cuda_ms"]["mean"],
        "active_mean_w": batch3["telemetry"]["active_mean_w"],
        "power_limit_fraction": batch3["telemetry"]["active_mean_fraction_of_power_limit"],
    },
    "ours_energy": [
        {"task": row["task"], "combined_j": row["combined_optical_plus_a100_energy_j"]}
        for row in energy_rows
    ],
}, ensure_ascii=False, indent=2))
