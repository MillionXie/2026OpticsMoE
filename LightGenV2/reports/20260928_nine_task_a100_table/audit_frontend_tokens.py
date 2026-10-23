"""Read-only derivation of timing boundaries, input throughput and legacy OP costs.

Never infers an unrecorded embedding time from a different run or a missing value.
Writes only new processed evidence; the original measurements are not modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
BASELINE = ROOT / "LightGenV2/reports/20260927_demo_energy_efficiency_a100/raw_remote/baseline"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def throughput(counts: list[float], milliseconds: list[float]) -> float:
    if not counts or len(counts) != len(milliseconds):
        raise ValueError("Token counts and durations must describe the same calls")
    if any(n < 0 for n in counts) or any(t <= 0 for t in milliseconds):
        raise ValueError("Invalid token count or duration")
    return sum(counts) / (sum(milliseconds) / 1000)


def raw_token_audit(key: str) -> dict:
    if key in {"lgvq_temporal", "lgvq_spatial"}:
        kind = "temporal" if key.endswith("temporal") else "spatial"
        path = BASELINE / f"01_lgvq_{kind}/resolution448/frames4/scheme2_five_quality_tokens/per_video_predictions_and_timing.csv"
        rows = read_csv(path)
        times = [float(r["model_internal_synchronized_wall_ms"]) for r in rows]
        visual = [ast.literal_eval(r["vision_block0_input_shape"])[0] for r in rows]
        language = [int(r["sequence_length"]) for r in rows]
    elif key == "abo_text_to_image":
        path = REPORT / "raw_remote/formal_04_baseline/baseline_timing_per_sample.csv"
        rows = read_csv(path)
        times = [float(r["synchronized_wall_ms"]) for r in rows]
        visual = None
        language = [ast.literal_eval(r["first_block_input_shape"])[1] for r in rows]
    elif key == "text_to_image_generation":
        return {"status": "200条基线记录未保存逐输入token长度，不以最大长度或Ours样例长度代替", "visual_mean": None, "language_mean": None, "native_mean": None, "native_unit": "未保存", "native_rate": None}
    else:
        directories = {"abo_image_to_text": "02_abo_image_to_text", "abo_image_to_image": "03_abo_image_to_image", "lsp": "05_lsp", "salicon": "06_salicon", "openmoji": "07_openmoji_new_layout"}
        path = BASELINE / directories[key] / "timing_per_sample.csv"
        rows = read_csv(path)
        times = [float(r["synchronized_wall_ms"]) for r in rows]
        visual = [ast.literal_eval(r["first_block_input_shape"])[0] for r in rows]
        language = None
        if key == "abo_image_to_image":
            language = [int(json.loads(r["output_contract"])["sequence_length"]) for r in rows]
    if len(rows) != 200:
        raise ValueError(f"Expected exactly 200 historical calls: {path}, got {len(rows)}")
    native = language if language is not None else visual
    result = {"status": "原始200次输入形状计数；序列槽位不等于生成token", "source": relative(path), "sha256": sha(path), "calls": len(rows), "wall_total_ms": sum(times), "wall_mean_ms": statistics.fmean(times), "native_unit": "LLM输入序列token" if language is not None else "视觉patch token", "native_total": sum(native), "native_mean": statistics.fmean(native), "native_rate": throughput(native, times)}
    for name, counts in [("visual", visual), ("language", language)]:
        result[f"{name}_total"] = sum(counts) if counts is not None else None
        result[f"{name}_mean"] = statistics.fmean(counts) if counts is not None else None
        result[f"{name}_rate"] = throughput(counts, times) if counts is not None else None
    if key == "openmoji":
        result["status"] += "；只记录instruction子序列，完整LLM序列长度缺失，故本行只报视觉patch吞吐"
    return result


def paired_frontend_audit() -> list[dict]:
    output = []
    all_core, all_full, all_front = [], [], []
    for repeat in range(1, 4):
        path = ROOT / f"LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_a100_20261020/qwen_cap100_repeat0{repeat}/calls.csv"
        rows = read_csv(path)[:200]
        if len(rows) != 200:
            raise ValueError("Incomplete paired LGVQ cost subset")
        core = [(float(r["gpu_output"]) - float(r["first_block_start"])) * 1000 for r in rows]
        full = [(float(r["gpu_output"]) - float(r["model_call_start"])) * 1000 for r in rows]
        front = [(float(r["first_block_start"]) - float(r["model_call_start"])) * 1000 for r in rows]
        for c, f, d in zip(core, full, front):
            if not math.isclose(f, c + d, abs_tol=1e-6):
                raise ValueError("Paired timestamp difference failed to reconcile")
        output.append({"repeat": repeat, "calls": 200, "core_wall_ms": statistics.fmean(core), "prepared_input_wall_ms": statistics.fmean(full), "preblock_host_difference_ms": statistics.fmean(front), "source": relative(path), "sha256": sha(path)})
        all_core.extend(core)
        all_full.extend(full)
        all_front.extend(front)
    output.append({"repeat": "三次合并", "calls": 600, "core_wall_ms": statistics.fmean(all_core), "prepared_input_wall_ms": statistics.fmean(all_full), "preblock_host_difference_ms": statistics.fmean(all_front), "source": "以上三个同源CSV各前200行；非9月旧表", "sha256": "见逐次记录"})
    return output


def build() -> dict:
    previous_path = REPORT / "version_audit_20261023/version_and_cost_audit_20261023.json"
    previous = read_json(previous_path)
    ops_path = ROOT / "LightGenV2/reports/20260927_demo_energy_efficiency_a100/raw_remote/ours_ops/operation_counts.json"
    ops = read_json(ops_path)["tasks"]
    top_path = REPORT / "latest_nine_task_summary_TOPS_20261008.csv"
    tops = {r["number"]: r for r in read_csv(top_path)}
    counts = {"lgvq_temporal": 3136, "lgvq_spatial": 784, "abo_image_to_text": 196, "abo_image_to_image": 196, "abo_text_to_image": None, "lsp": 196, "salicon": 196, "openmoji": 196, "text_to_image_generation": 32}
    bridges = {
        "lgvq_temporal": ("已测局部桥接", "frame_merger、serial_field、router_width、router_frames；前端/编码adapter仍未独立完整计时"),
        "lgvq_spatial": ("已测局部桥接", "已测frame_merger；不代表已计patch/position和所有input/outputadapter"),
        "abo_image_to_text": ("未注册计时，不能按0解释", "旧add_standard未登记merger；真实替换Qwen保留主merger，需补计"),
        "abo_image_to_image": ("未注册计时，确定漏merger", "standalone/model.py:212执行frontend.merge，旧formal_03报告明确排除merger/layout"),
        "abo_text_to_image": ("在线纯文本无视觉merger", "采用cc977b83标题query只走language三次；旧六次为通用多模态参考，不是该query路径"),
        "lsp": ("无语言桥接，不适用", "vision-only无vision→language桥接；patch/position和适配器另测"),
        "salicon": ("无语言桥接，不适用", "vision-only；兼容外壳可能执行无预测作用merger，实际执行路径仍需测，不当作不存在"),
        "openmoji": ("已测局部桥接", "position summary/pool及prompt_to_vision；不是全部GPU前端"),
        "text_to_image_generation": ("已测局部桥接", "text_to_spatial局部量；前端/所有输入编码另测"),
    }
    rows = []
    for old in previous["rows"]:
        key = old["key"]
        tokens = raw_token_audit(key)
        if "wall_mean_ms" in tokens and not math.isclose(tokens["wall_mean_ms"], old["baseline_per_unit_ms"], abs_tol=1e-6):
            raise ValueError(f"Token source does not match old timing: {key}")
        ours_top = ops[key]["total_effective_top"] if key in ops else None
        baseline_top = float(tops[old["id"]]["baseline_achieved_top"]) if tops[old["id"]].get("baseline_achieved_top") else None
        op_note = "历史MAC主项/光等效形状估算，非当前PT完整全图；1实数MAC=2OP；排并行残差"
        if key == "abo_image_to_image":
            op_note += "；旧算量77语言槽，旧统一计时71语言槽，二者不严格同图"
        if key not in {"lgvq_temporal", "lgvq_spatial"}:
            op_note += "；基线算量只取首样本，变长任务不代表200样本平均"
        row = {**old, "qwen_tokens": tokens, "ours_input_count_reference": counts[key], "ours_input_unit": "光学特征输入槽位", "ours_input_count_basis": "旧固定形状组件合同，一次输入计一次，不跨光层累加；不是200个独立输入的valid mask统计", "ours_top_reference": ours_top, "baseline_top_reference": baseline_top, "operation_basis": op_note, "operation_source": relative(ops_path), "operation_source_sha256": sha(ops_path), "bridge_status": bridges[key][0], "bridge_explanation": bridges[key][1], "frontend_increment_ms": None, "additional_nn_increment_ms": None, "full_gpu_frontend_remeasured": False, "expanded_passes": 3 if key == "abo_text_to_image" else old["passes"]}
        if key == "abo_text_to_image":
            row["ours_input_unit"] = "在线标题token未保存"
        if key == "text_to_image_generation":
            row["ours_input_unit"] = "固定样例prompt输入token"
            row["ours_input_count_basis"] = "formal_08_fullaudit/ours_report.json中该固定样例32个prompt tokens；不是200不同prompt均值，不可套给基线"
        rows.append(row)
    return {"date": "2026-10-23", "source_audit": relative(previous_path), "source_audit_sha256": sha(previous_path), "scope": "两个口径的分离核查；旧核心口径可追溯参考；含GPU前端版缺测项不填零", "powers": previous["powers"], "per_pass_ms": previous["optical_input"]["per_pass_ms"], "rows": rows, "paired_lgvq": paired_frontend_audit(), "definitions": {"input_throughput": "sum(input_tokens_i) / (sum(duration_ms_i)/1000)，不是mean(tokens_i/time_i)", "token_unit": "分别标注LLM输入序列、视觉patch和Ours特征槽位；不计层内重复token，不宣称生成token/s", "cross_architecture": "原生粒度不同，原生Tokens/s不能直接用于跨模型加速倍率；任务/s和任务/J用于匹配工作量比较", "tokenizer": "当前实现的CPU processor/tokenizer不计GPU前端；H2D和解码另保留", "core": "旧Ours为CUDA Event局部分量之和；Qwen为首native block起同步wall；两者均非完整前端", "expanded": "GPU输入已就绪后加入patch/position、embedding、必要adapter/merger/神经编码，Ours CUDA Event，Qwen synchronized wall；不重复计旧边界内的merger", "energy_ours": "E=(121.746*t_optical_ms+104.230*t_electronic_ms)/1000", "energy_qwen": "E=(338.2+P_A100_task)*t_wall_ms/1000", "tops": "TOPS=TOP/(t_ms/1000)，TOPS/W=TOP/E_J；新增前端必须同边界补计OP，不能只改时间分母", "embedding_mac": "Embedding查表通常没有Linear/Conv式MAC；按MAC主项统计并不意味着查表没有时间或能耗", "zero": "缺失、未注册、无该分支、不影响预测但仍执行，是四种不同状态"}, "limitations": ["本轮没有GPU补测；A100有其他用户进程，不能产生独占能耗证据。", "旧性能权重和旧计时/算量不全同版本；沿用上次逐行版本警告。", "新版缺GPU前端独立时间、部分必要adapter/merger及匹配OP，故完整新版速度/能耗/倍数暂不能给数值。", "LGVQ同批前端差是host时间戳区间，含前端调度，不能冒称纯embedding CUDA时间或加到9月旧数据。", "原生Tokens/s计的是已记录序列/形状槽位，缺mask时不冒称有效非padding token/s。"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT / "frontend_tokens_20261023")
    parser.add_argument("--manifest-only", action="store_true", help="Hash completed private artifacts after workbook verification")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        files = ["frontend_tokens_audit_20261023.json", "qwen_input_tokens_from_raw_200.csv", "runtime_frontend_source_and_gpu_audit.json", "九任务_核心与含前端_Tokens_20261023.xlsx"]
        code = [Path(__file__), REPORT / "test_frontend_tokens.py", args.output / "build_workbook.mjs", REPORT / "README.md"]
        payload = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "method": "No new GPU inference; raw per-call files preserved; new frontend measurements missing", "artifacts": [{"file": name, "bytes": (args.output / name).stat().st_size, "sha256": sha(args.output / name)} for name in files], "source_files": [{"file": relative(path), "sha256": sha(path)} for path in code]}
        dest = args.output / "delivery_manifest_20261023.json"
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(dest), "artifacts": len(files)}))
        return 0
    audit = build()
    path = args.output / "frontend_tokens_audit_20261023.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "qwen_input_tokens_from_raw_200.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "task", "calls", "visual_total", "language_total", "visual_rate", "language_rate", "native_unit", "native_rate", "source", "sha256", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit["rows"]:
            writer.writerow({k: row.get(k, row["qwen_tokens"].get(k, "")) for k in fields})
    print(json.dumps({"output": str(path), "tasks": len(audit["rows"]), "qwen_raw_calls_with_token_shapes": sum(r["qwen_tokens"].get("calls", 0) for r in audit["rows"]), "paired_frontend_calls": 600}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
