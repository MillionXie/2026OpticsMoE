from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_remote"
OPTICAL_DEVICES_W = 80.358
CONTROL_HOST_W = 41.388
OURS_A100_W = 62.842
BASELINE_HOST_W = 338.2
PASS_MS = 1.0447


ROWS = [
    ("01a", "LGVQ temporal quality", "SRCC", 0.8044, 0.7663428036522781, "lgvq_temporal", "01_lgvq_temporal", 16),
    ("01b", "LGVQ spatial quality", "SRCC", 0.6710, 0.6907725734336255, "lgvq_spatial", "01_lgvq_spatial", 1),
    ("02", "ABO image-to-text retrieval", "R@1", 0.8058333333333333, 0.7358333333333333, "abo_image_to_text", "02_abo_image_to_text", 1),
    ("03", "ABO image-to-image retrieval", "R@1", 0.83125, 0.85125, "abo_image_to_image", "03_abo_image_to_image", 1),
    ("05", "LSP keypoint detection", "PCK@0.2", 0.7347857143, 0.7225714285714285, "lsp", "05_lsp", 1),
    ("06", "SALICON saliency", "CC", 0.8624925082, 0.8748382970809937, "salicon", "06_salicon", 1),
    ("07", "OpenMoji semantic interaction", "Accuracy", 0.8765, 0.8120, "openmoji", "07_openmoji_new_layout", 1),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_report_path(folder: str) -> Path:
    base = RAW / "baseline" / folder
    reports = list(base.rglob("report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one report under {base}, found {reports}")
    return reports[0]


def narrow_rows(report: dict) -> dict[str, dict]:
    rows = report["formal_narrow_cuda_event"]
    return {str(row["task"]): row for row in rows}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ours_report_path = RAW / "ours" / "narrow_200_no_warmup" / "report.json"
    ours_report = read_json(ours_report_path)
    ours_narrow = narrow_rows(ours_report)
    ours_ops = read_json(RAW / "ours_ops" / "operation_counts.json")["tasks"]
    standard_ops_names = {
        "abo_image_to_text": "abo_image_text",
        "abo_image_to_image": "abo_image_image",
        "lsp": "lsp", "salicon": "salicon", "openmoji": "openmoji",
    }
    summary = []
    for number, label, metric, ours_perf, base_perf, task_name, folder, workload in ROWS:
        base_path = baseline_report_path(folder)
        base = read_json(base_path)
        if folder.startswith("01_"):
            base_ms_one = float(base["model_internal_synchronized_wall_ms_first_200_included"]["mean"])
            base_power = float(base["power"]["active_mean_w"])
        else:
            base_ms_one = float(base["timing"]["synchronized_wall_ms"]["mean"])
            base_power = float(base["power"]["active_power_w"]["mean"])
        passes = 3 if task_name in {"lsp", "salicon"} else 6
        ours_e_ms = float(ours_narrow[task_name]["main_electronic_including_bridge_cuda_ms"])
        ours_total_ms = ours_e_ms + passes * PASS_MS
        # Temporal Ours handles 16 videos in one optical field. Qwen handles one
        # video per call, so its formal matched-workload time and energy use x16.
        base_total_ms = base_ms_one * workload
        optical_ms = passes * PASS_MS
        ours_optical_energy = optical_ms / 1000 * (OPTICAL_DEVICES_W + CONTROL_HOST_W)
        ours_electronic_energy = ours_e_ms / 1000 * (CONTROL_HOST_W + OURS_A100_W)
        ours_energy = ours_optical_energy + ours_electronic_energy
        base_energy_one = base_ms_one / 1000 * (BASELINE_HOST_W + base_power)
        base_energy = base_energy_one * workload
        ours_top = float(ours_ops[task_name]["total_effective_top"])
        if task_name in {"lgvq_temporal", "lgvq_spatial"}:
            op_report = read_json(RAW / "baseline_ops" / task_name / "operation_count_and_topsw.json")
            base_top_one = float(op_report["qwen3vl"]["ops_per_video_1mac_eq_2op"]) / 1e12
        else:
            op_report = read_json(RAW / "baseline_ops" / f"{standard_ops_names[task_name]}.json")
            base_top_one = float(op_report["top"])
        base_top = base_top_one * workload
        summary.append({
            "number": number,
            "task": label,
            "metric": metric,
            "ours_performance_full_test": ours_perf,
            "baseline_performance_full_test": base_perf,
            "workload_units": workload,
            "physical_passes": passes,
            "optical_time_ms": optical_ms,
            "ours_electronic_cuda_event_mean_ms": ours_e_ms,
            "ours_total_time_ms": ours_total_ms,
            "baseline_synchronized_wall_mean_ms_per_unit": base_ms_one,
            "baseline_total_time_ms_matched_workload": base_total_ms,
            "ours_optical_plus_host_power_w": OPTICAL_DEVICES_W + CONTROL_HOST_W,
            "ours_electronic_host_plus_a100_power_w": CONTROL_HOST_W + OURS_A100_W,
            "baseline_a100_active_mean_power_w": base_power,
            "baseline_host_plus_a100_power_w": BASELINE_HOST_W + base_power,
            "ours_optical_energy_j": ours_optical_energy,
            "ours_electronic_energy_j": ours_electronic_energy,
            "ours_total_energy_j": ours_energy,
            "baseline_energy_j_per_unit": base_energy_one,
            "baseline_total_energy_j_matched_workload": base_energy,
            "speedup_x": base_total_ms / ours_total_ms,
            "energy_efficiency_gain_x": base_energy / ours_energy,
            "energy_reduction_fraction": 1 - ours_energy / base_energy,
            "ours_units_per_j": workload / ours_energy,
            "baseline_units_per_j": workload / base_energy,
            "ours_effective_top": ours_top,
            "baseline_achieved_top_matched_workload": base_top,
            "ours_effective_tops": ours_top / (ours_total_ms / 1000),
            "baseline_achieved_tops": base_top / (base_total_ms / 1000),
            "ours_effective_tops_per_w": ours_top / ours_energy,
            "baseline_achieved_tops_per_w": base_top / base_energy,
            "tops_per_w_gain_x": (ours_top / ours_energy) / (base_top / base_energy),
            "ours_report": str(ours_report_path.relative_to(ROOT)),
            "baseline_report": str(base_path.relative_to(ROOT)),
        })

    output_json = ROOT / "calculated_summary.json"
    output_csv = ROOT / "calculated_summary.csv"
    output_json.write_text(json.dumps({
        "schema_version": 1,
        "constants": {
            "optical_devices_w": OPTICAL_DEVICES_W,
            "control_host_w": CONTROL_HOST_W,
            "ours_a100_w": OURS_A100_W,
            "baseline_host_w": BASELINE_HOST_W,
            "physical_pass_ms": PASS_MS,
            "timing_policy": "baseline: first 200 test items; Ours: 200 consecutive shape-matched GPU kernel invocations; zero explicit warm-up; call/item 1 retained; means",
        },
        "rows": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "build_summary_stdout.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # Keep the requested numbered Ours/Baseline layout in addition to the
    # untouched remote tree. Copies are small and make paper review easier.
    for number, _label, _metric, _op, _bp, task_name, folder, _workload in ROWS:
        base_dst = ROOT / "baseline" / folder / "raw_200"
        ours_dst = ROOT / "ours" / folder.replace("01_lgvq_temporal", "01_lgvq_temporal").replace("01_lgvq_spatial", "01_lgvq_spatial") / "raw_200"
        shutil.copytree((RAW / "baseline" / folder), base_dst, dirs_exist_ok=True)
        ours_dst.mkdir(parents=True, exist_ok=True)
        for obsolete in (
            "per_call_timings.csv", "source_and_artifact_sha256.csv", "command.txt",
            "environment.txt", "shared_components_report.json",
        ):
            obsolete_path = ours_dst / obsolete
            if obsolete_path.exists():
                obsolete_path.unlink()
        shutil.copy2(ours_report_path, ours_dst / "shared_components_report.json")
        for name in ("all_per_call_timings.csv", "component_summary.csv", "paper_narrow_summary.csv", "SHA256SUMS.txt", "README.md"):
            source = ours_report_path.parent / name
            if source.exists():
                shutil.copy2(source, ours_dst / name)

    checks = []
    for current, _dirs, names in os.walk(ROOT, onerror=lambda _error: None):
        for name in sorted(names):
            path = Path(current) / name
            if name == "SHA256SUMS.txt":
                continue
            try:
                checks.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
            except (FileNotFoundError, OSError):
                # OneDrive may evict unrelated historical snapshot leaves while
                # walking. The formal raw/result files are local and retained.
                continue
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
