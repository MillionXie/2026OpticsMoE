"""Build the compact baseline plotting handoff from audited run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "LightGenV2" / "reports" / "baseline_plotting_20260922"
RAW = REPORT / "_server_raw"
OUT = REPORT / "delivery"
QWEN_A100 = ROOT / "LightGenV2" / "reports" / "20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def metric_rows(scope: str, values: dict, units: dict[str, str] | None = None, note: str = "") -> list[dict]:
    units = units or {}
    rows = []
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(
                {
                    "scope": scope,
                    "metric": key,
                    "value": value,
                    "unit": units.get(key, "dimensionless"),
                    "note": note,
                }
            )
    return rows


def copy_source_report(source: Path, task_dir: Path) -> None:
    shutil.copy2(source, task_dir / "source_report.json")


def build_lgvq(target: str) -> dict:
    task_id = f"lgvq_{target}"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    if target == "temporal":
        source_dir = QWEN_A100 / "evidence" / "T06_temporal_batch1_batch2" / "batch_01"
        report_path = source_dir / "report.json"
        report = read_json(report_path)
        predictions = read_csv(source_dir / "predictions.csv")
        timing = {row["batch_index"]: row for row in read_csv(source_dir / "batch_timing.csv")}
        rows = []
        for row in predictions:
            joined = dict(row)
            joined["signed_error_mos"] = float(row["prediction"]) - float(row["target_mos"])
            joined["absolute_error_mos"] = abs(joined["signed_error_mos"])
            joined.update(timing.get(row["batch_index"], {}))
            joined["target_unit"] = "MOS score"
            joined["latency_unit"] = "ms/video"
            rows.append(joined)
    else:
        source_dir = QWEN_A100 / "evidence" / "T06_spatial"
        report_path = source_dir / "report.json"
        report = read_json(report_path)
        rows = read_csv(source_dir / "per_video_predictions_and_timing.csv")
        for row in rows:
            row["signed_error_mos"] = float(row["prediction"]) - float(row["target_mos"])
            row["absolute_error_mos"] = abs(row["signed_error_mos"])
            row["target_unit"] = "MOS score"
            row["latency_unit"] = "ms/video"
    write_csv(task_dir / "data.csv", rows)
    units = {"rmse": "MOS score", "mae": "MOS score"}
    write_csv(task_dir / "metrics.csv", metric_rows("full_test", report["performance"], units))
    copy_source_report(report_path, task_dir)
    source = {
        "task": task_id,
        "code_commit": report.get("git_commit"),
        "code_entry": "LightGenV2/tasks/t06_video_quality_assessment/quality_token_dataset_once.py",
        "checkpoint_server_path": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "manifest_sha256": report.get("manifest_sha256"),
        "model": report.get("model"),
        "sample_rows": len(rows),
        "timing_boundary": report.get("timing_boundary"),
    }
    write_json(task_dir / "SOURCE.json", source)
    return {
        "task_id": task_id,
        "display_name": f"LGVQ {target}",
        "primary_metric": "SRCC",
        "primary_value": report["performance"]["srcc"],
        "unit": "correlation",
        "test_samples": len(rows),
        "data_file": f"{task_id}/data.csv",
        "status": "complete",
    }


def build_abo_image_to_text() -> dict:
    task_id = "abo_image_to_text"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path = RAW / task_id / "baseline_report.json"
    report = read_json(report_path)
    rows = read_csv(RAW / task_id / "retrieval_predictions.csv")
    timing = {r["sample_id"]: r for r in read_csv(RAW / task_id / "timing_per_sample.csv")}
    for row in rows:
        row["hit_at_1"] = int(row["true_rank"]) <= 1
        row["hit_at_5"] = int(row["true_rank"]) <= 5
        row["hit_at_10"] = int(row["true_rank"]) <= 10
        row["reciprocal_rank"] = 1.0 / int(row["true_rank"])
        measured = timing.get(row["sample_id"])
        row["cuda_ms"] = measured["cuda_ms"] if measured else ""
        row["synchronized_wall_ms"] = measured["host_ms"] if measured else ""
        row["timing_measured"] = bool(measured)
        row["latency_unit"] = "ms/query image"
    write_csv(task_dir / "data.csv", rows)
    write_csv(task_dir / "metrics.csv", metric_rows("full_test", report["performance"]))
    copy_source_report(report_path, task_dir)
    shutil.copy2(RAW / task_id / "per_product_metrics.json", task_dir / "per_product_metrics.json")
    shutil.copy2(RAW / task_id / "baseline_overview.png", task_dir / "sample_overview.png")
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": report.get("git_commit"),
            "code_entry": "LightGenV2/tasks/t08_abo_image_text_retrieval/baseline_5090d.py",
            "checkpoint": None,
            "frozen_model_sha256": report.get("model_safetensors_sha256"),
            "dataset_sha256": report.get("dataset_sha256"),
            "sample_rows": len(rows),
            "timing_rows": sum(bool(timing.get(row["sample_id"])) for row in rows),
            "timing_note": "Only the deterministic 200-query timing subset has latency values.",
        },
    )
    return {
        "task_id": task_id,
        "display_name": "ABO image-to-text",
        "primary_metric": "R@1",
        "primary_value": report["performance"]["recall_at_1"],
        "unit": "fraction",
        "test_samples": len(rows),
        "data_file": f"{task_id}/data.csv",
        "status": "complete",
    }


def load_timing_by_sample(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_csv(path):
        contract = json.loads(row["output_contract"])
        result[contract["sample_id"]] = row
    return result


def build_abo_image_to_image() -> dict:
    task_id = "abo_image_to_image"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path = RAW / task_id / "final_report.json"
    report = read_json(report_path)
    rows = read_csv(RAW / task_id / "normal_predictions.csv")
    timing_path = QWEN_A100 / "fresh" / "T07_abo_image_image" / "timing_per_sample.csv"
    timing = load_timing_by_sample(timing_path)
    for row in rows:
        measured = timing.get(row["sample_id"])
        row["cuda_ms"] = measured["cuda_event_ms"] if measured else ""
        row["synchronized_wall_ms"] = measured["synchronized_wall_ms"] if measured else ""
        row["timing_measured"] = bool(measured)
        row["latency_unit"] = "ms/query image"
    write_csv(task_dir / "data.csv", rows)
    metrics = report["metrics"]["normal"]
    write_csv(task_dir / "metrics.csv", metric_rows("full_test", metrics))
    copy_source_report(report_path, task_dir)
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": report.get("source_commit"),
            "code_entry": "LightGenV2/tasks/t07_abo_image_retrieval/standalone/retrieval_screen.py",
            "checkpoint": None,
            "frozen_model_config_sha256": report.get("model_config_sha256"),
            "manifest_sha256": report.get("manifest_sha256"),
            "sample_rows": len(rows),
            "timing_rows": sum(bool(timing.get(row["sample_id"])) for row in rows),
        },
    )
    return {
        "task_id": task_id,
        "display_name": "ABO image-to-image",
        "primary_metric": "R@1",
        "primary_value": metrics["r_at_1"],
        "unit": "fraction",
        "test_samples": len(rows),
        "data_file": f"{task_id}/data.csv",
        "status": "complete",
    }


def build_abo_text_to_image() -> dict:
    task_id = "abo_text_to_image"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path = RAW / task_id / "report.json"
    report = read_json(report_path)
    predictions = read_json(RAW / task_id / "dynamic_2048_predictions.json")
    rows = []
    for row in predictions:
        rank = int(row["first_positive_rank"])
        rows.append(
            {
                "product_id": row["product_id"],
                "title": row["title"],
                "first_positive_rank": rank,
                "hit_at_1": rank <= 1,
                "hit_at_5": rank <= 5,
                "hit_at_10": rank <= 10,
                "reciprocal_rank": 1.0 / rank,
                "top10_sample_ids": json.dumps(row["top10_sample_ids"], ensure_ascii=False),
                "preprocessing": "dynamic native aspect",
                "embedding_dimension": 2048,
            }
        )
    write_csv(task_dir / "data.csv", rows)
    metrics = report["metrics"]["dynamic_2048"]
    write_csv(task_dir / "metrics.csv", metric_rows("dynamic_2048", metrics))
    copy_source_report(report_path, task_dir)
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": report.get("source_commit"),
            "code_entry": "LightGenV2/tasks/t08_abo_image_text_retrieval/text_to_image_baseline.py",
            "code_location_note": "The exact source is at Git commit 1e218991a; it is not present on the current mixed branch.",
            "checkpoint": None,
            "frozen_model_weight_sha256": report.get("model_weight_sha256"),
            "dataset_sha256": report.get("data_sha256"),
            "query_rows": len(rows),
            "gallery_images": metrics["gallery_count"],
            "timing_status": "not measured in this run",
        },
    )
    return {
        "task_id": task_id,
        "display_name": "ABO text-to-image",
        "primary_metric": "Hit@1",
        "primary_value": metrics["hit_at_1"],
        "unit": "fraction",
        "test_samples": len(rows),
        "data_file": f"{task_id}/data.csv",
        "status": "complete; latency not measured",
    }


def build_lsp() -> dict:
    task_id = "lsp"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    exact_report_path = QWEN_A100 / "evidence" / "performance_sources" / "T02_lsp_baseline_report.json"
    exact_report = read_json(exact_report_path)
    joint_rows = read_csv(RAW / task_id / "teacher_inference_predictions.csv")
    write_csv(task_dir / "per_joint.csv", joint_rows)
    grouped = defaultdict(list)
    for row in joint_rows:
        grouped[row["sample_id"]].append(row)
    timing_path = QWEN_A100 / "fresh" / "T02_lsp" / "timing_per_sample.csv"
    timing = load_timing_by_sample(timing_path)
    image_rows = []
    for sample_id, joints in grouped.items():
        by_name = {r["joint_name"]: r for r in joints}
        point = lambda name: (float(by_name[name]["true_x"]), float(by_name[name]["true_y"]))
        dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        torso = (dist(point("right_shoulder"), point("left_hip")) + dist(point("left_shoulder"), point("right_hip"))) / 2
        head = 2 * dist(point("neck"), point("head_top"))
        errors = [float(r["pixel_error"]) for r in joints]
        measured = timing.get(sample_id)
        image_rows.append(
            {
                "sample_id": sample_id,
                "image_path": joints[0]["image_path"],
                "evaluated_joints": len(joints),
                "mean_pixel_error_px": sum(errors) / len(errors),
                "torso_scale_px": torso,
                "head_scale_px": head,
                "pck_at_0.2_torso": sum(e <= 0.2 * torso for e in errors) / len(errors),
                "pckh_at_0.5_head": sum(e <= 0.5 * head for e in errors) / len(errors),
                "normalized_mean_error_torso": sum(e / torso for e in errors) / len(errors),
                "cuda_ms": measured["cuda_event_ms"] if measured else "",
                "synchronized_wall_ms": measured["synchronized_wall_ms"] if measured else "",
                "timing_measured": bool(measured),
                "coordinate_unit": "pixel at 224x224 input",
                "latency_unit": "ms/image",
            }
        )
    write_csv(task_dir / "data.csv", image_rows)
    performance = exact_report["performance"]
    units = {"mean_pixel_error": "pixel at 224x224 input"}
    rows = metric_rows("A100_exact_aggregate", performance, units)
    per_joint = performance.get("per_joint_pck_at_0.2_torso", {})
    rows += metric_rows("per_joint_pck_at_0.2_torso", per_joint)
    write_csv(task_dir / "metrics.csv", rows)
    copy_source_report(exact_report_path, task_dir)
    sample_dir = task_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for source in sorted((RAW / task_id / "samples").glob("*.png"))[:6]:
        shutil.copy2(source, sample_dir / source.name)
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": exact_report.get("git_commit"),
            "code_entry": "LightGenV2/tasks/t02_keypoint_detection/baseline_5090d.py",
            "checkpoint_server_path": exact_report.get("checkpoint"),
            "checkpoint_sha256": exact_report.get("checkpoint_sha256"),
            "exact_a100_primary_value": performance["pck_at_0.2_torso"],
            "per_joint_source_value": read_json(RAW / task_id / "teacher_inference.json")["pck_at_0.2_torso"],
            "important_note": "The retained per-joint CSV is an earlier evaluation of the same epoch-37 teacher and recomputes to 0.7216; the requested 0.7226 is the later A100 aggregate. They are kept distinct, not silently mixed.",
            "sample_rows": len(image_rows),
        },
    )
    return {
        "task_id": task_id,
        "display_name": "LSP keypoint detection",
        "primary_metric": "PCK@0.2 torso",
        "primary_value": performance["pck_at_0.2_torso"],
        "unit": "fraction",
        "test_samples": len(image_rows),
        "data_file": f"{task_id}/data.csv",
        "status": "aggregate exact; per-joint source differs by 0.001",
    }


def build_salicon() -> dict:
    task_id = "salicon"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path = RAW / task_id / "selected_checkpoint_test_evaluation.json"
    report = read_json(report_path)
    rows = read_csv(RAW / task_id / "per_image_cc.csv")
    timing_path = QWEN_A100 / "fresh" / "T03_salicon" / "timing_per_sample.csv"
    timing = load_timing_by_sample(timing_path)
    timing_by_id = {"".join(c for c in key if c.isdigit())[-12:]: value for key, value in timing.items()}
    for row in rows:
        numeric_id = "".join(c for c in row["sample_id"] if c.isdigit())[-12:]
        measured = timing_by_id.get(numeric_id)
        row["cuda_ms"] = measured["cuda_event_ms"] if measured else ""
        row["synchronized_wall_ms"] = measured["synchronized_wall_ms"] if measured else ""
        row["timing_measured"] = bool(measured)
        row["latency_unit"] = "ms/image"
    write_csv(task_dir / "data.csv", rows)
    write_csv(task_dir / "metrics.csv", metric_rows("full_test", report["metrics"]))
    copy_source_report(report_path, task_dir)
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": report.get("git_commit"),
            "code_entry": "LightGenV2/tasks/t03_saliency/aligned_baseline.py",
            "checkpoint_server_path": "LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_50_20260912_seed42/best_checkpoint.pt",
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "sample_rows": len(rows),
            "per_sample_metric_limit": "The audited run retained per-image CC only; KLD/SIM/NSS/AUC-Judd/MAE are aggregate values in metrics.csv.",
        },
    )
    return {
        "task_id": task_id,
        "display_name": "SALICON saliency",
        "primary_metric": "CC",
        "primary_value": report["metrics"]["cc"],
        "unit": "correlation",
        "test_samples": len(rows),
        "data_file": f"{task_id}/data.csv",
        "status": "complete; only CC retained per image",
    }


def build_openmoji() -> dict:
    task_id = "openmoji"
    task_dir = OUT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    report_path = RAW / task_id / "selected_checkpoint_test_evaluation.json"
    report = read_json(report_path)
    predictions = [json.loads(line) for line in (RAW / task_id / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    write_csv(task_dir / "data.csv", predictions)
    rows = []
    for scope, values in report["metrics"].items():
        rows += metric_rows(scope, values)
    write_csv(task_dir / "metrics.csv", rows)
    copy_source_report(report_path, task_dir)
    sample_dir = task_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for source in sorted((RAW / task_id / "samples").glob("*.png")):
        shutil.copy2(source, sample_dir / source.name)
    manifest = read_json(RAW / task_id / "run_manifest.json")
    write_json(
        task_dir / "SOURCE.json",
        {
            "task": task_id,
            "code_commit": manifest.get("git_commit"),
            "code_entry": "LightGenV2/tasks/t04_semantic_interaction/run.py --profile layered_scene_qwen_shared",
            "checkpoint_server_path": report.get("checkpoint"),
            "checkpoint_sha256": "0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8",
            "selected_epoch": report.get("selected_epoch"),
            "sample_rows": len(predictions),
            "timing_status": "not measured for this exact layered-scene checkpoint",
            "warning": "Do not replace this 0.8120 run with the older qwen_shared_s73 0.8420 run from the mixed current branch.",
        },
    )
    return {
        "task_id": task_id,
        "display_name": "OpenMoji semantic interaction",
        "primary_metric": "changed-cell accuracy",
        "primary_value": report["metrics"]["overall"]["changed_cell_accuracy"],
        "unit": "fraction",
        "test_samples": len(predictions),
        "data_file": f"{task_id}/data.csv",
        "status": "complete; exact-checkpoint latency not measured",
    }


def build_readme(summary: list[dict]) -> None:
    table = ["| 任务 | 主指标 | 数值 | 样本数 | 状态 |", "|---|---:|---:|---:|---|"]
    for row in summary:
        table.append(
            f"| {row['display_name']} | {row['primary_metric']} | {float(row['primary_value']):.6f} | {row['test_samples']} | {row['status']} |"
        )
    text = f"""# Baseline 绘图数据包（2026-09-22）

这个目录是给绘图同学的最小交付：先看 `summary.csv`，需要散点、箱线、分组或失败案例时再进入相应任务目录。每个任务最多只有 `data.csv`、`metrics.csv`、`SOURCE.json`、`source_report.json` 和可选 `samples/`。

{chr(10).join(table)}

## 列与单位

- 比例、相关系数、Recall、Hit、MRR、mAP、IoU、F1 均为 0–1 的无量纲小数，绘图时需要百分数可乘 100。
- LGVQ `target_mos`、`prediction`、`*_error_mos` 是原始 MOS 分数，不是百分比。
- 所有 `*_ms` 是毫秒；LSP 坐标及误差是 224×224 输入平面上的像素。
- `timing_measured=false` 表示该样本没有进入独立的 200 条计时子集，不能以均值回填单点。
- 样本级文件保留代码实际落盘的指标；缺失项不从汇总值反推，也不造数据。

## 两个容易混淆的版本

1. OpenMoji 0.8120 对应 layered-scene baseline、epoch 30、checkpoint SHA256 `0be775...cf8`；当前混合分支中的旧 `qwen_shared_s73` 是 0.8420，不能互换。
2. ABO 文搜图 0.8200 是独立的“100 标题 query → 2400 TEST 图片 gallery”冻结 Qwen 2048D run，源码固定在 Git commit `1e218991a`，不是把图搜文矩阵转置。

## 已知边界

- LSP 请求值 0.722571 来自后续 A100 完整汇总；服务器保留的逐关节 CSV 属于同一 epoch-37 teacher 的较早评测，重算为 0.721571。两者在 `lsp/SOURCE.json` 明确分开。
- SALICON 的正式 run 只保留逐图 CC；KLD、SIM、NSS、AUC-Judd、MAE 只有完整 5000 图汇总。
- 文搜图 0.8200 与 OpenMoji 0.8120 的精确 run 没有单次推理计时，不能借用其他协议的延迟。
"""
    (OUT / "README_CN.md").write_text(text, encoding="utf-8")


def required_inputs(raw: Path, qwen: Path) -> list[Path]:
    """Required original files; optional example PNGs are not fabricated."""
    return [raw / relative for relative in (
        "abo_image_to_text/baseline_report.json",
        "abo_image_to_text/retrieval_predictions.csv",
        "abo_image_to_text/timing_per_sample.csv",
        "abo_image_to_text/per_product_metrics.json",
        "abo_image_to_text/baseline_overview.png",
        "abo_image_to_image/final_report.json",
        "abo_image_to_image/normal_predictions.csv",
        "abo_text_to_image/report.json",
        "abo_text_to_image/dynamic_2048_predictions.json",
        "lsp/teacher_inference_predictions.csv",
        "lsp/teacher_inference.json",
        "salicon/selected_checkpoint_test_evaluation.json",
        "salicon/per_image_cc.csv",
        "openmoji/selected_checkpoint_test_evaluation.json",
        "openmoji/test_predictions.jsonl",
        "openmoji/run_manifest.json",
    )] + [qwen / relative for relative in (
        "evidence/T06_temporal_batch1_batch2/batch_01/report.json",
        "evidence/T06_temporal_batch1_batch2/batch_01/predictions.csv",
        "evidence/T06_temporal_batch1_batch2/batch_01/batch_timing.csv",
        "evidence/T06_spatial/report.json",
        "evidence/T06_spatial/per_video_predictions_and_timing.csv",
        "fresh/T07_abo_image_image/timing_per_sample.csv",
        "evidence/performance_sources/T02_lsp_baseline_report.json",
        "fresh/T02_lsp/timing_per_sample.csv",
        "fresh/T03_salicon/timing_per_sample.csv",
    )]


def prepare_output(output: Path, inputs: list[Path], protected: list[Path]) -> Path:
    """Fail before writing if the output exists, overlaps inputs or lacks evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Existing output is protected: {output}")
    target = output.resolve()
    for source in protected:
        source = source.resolve()
        if target == source or source in target.parents:
            raise ValueError(f"Output overlaps original input tree: {source}")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing original evidence; output not created: " + "; ".join(missing))
    target.mkdir(parents=True, exist_ok=False)
    return target


def main(argv: list[str] | None = None) -> None:
    global OUT, RAW, QWEN_A100
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New derived delivery directory; existing paths are never removed or overwritten")
    parser.add_argument("--raw", type=Path, default=RAW, help="Original baseline raw evidence tree")
    parser.add_argument("--qwen-report", type=Path, default=QWEN_A100,
                        help="Original historical Qwen A100 report tree, not a new model's report")
    args = parser.parse_args(argv)
    # Assign only after validation; all old per-task metric/join rules remain unchanged.
    target = prepare_output(args.output, required_inputs(args.raw, args.qwen_report),
                            [args.raw, args.qwen_report])
    OUT, RAW, QWEN_A100 = target, args.raw, args.qwen_report
    summary = [
        build_lgvq("temporal"),
        build_lgvq("spatial"),
        build_abo_image_to_text(),
        build_abo_image_to_image(),
        build_abo_text_to_image(),
        build_lsp(),
        build_salicon(),
        build_openmoji(),
    ]
    write_csv(OUT / "summary.csv", summary)
    code_rows = []
    for row in summary:
        source = read_json(OUT / row["task_id"] / "SOURCE.json")
        code_rows.append(
            {
                "task_id": row["task_id"],
                "code_commit": source.get("code_commit", ""),
                "code_entry": source.get("code_entry", ""),
                "checkpoint_server_path": source.get("checkpoint_server_path", ""),
                "checkpoint_sha256": source.get("checkpoint_sha256", ""),
            }
        )
    write_csv(OUT / "CODE_AND_WEIGHTS.csv", code_rows)
    build_readme(summary)
    hashes = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.csv":
            continue
        hashes.append({"sha256": digest(path), "file": path.relative_to(OUT).as_posix()})
    write_csv(OUT / "SHA256SUMS.csv", hashes)
    print(json.dumps({"output": str(OUT), "tasks": len(summary), "files": len(hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
