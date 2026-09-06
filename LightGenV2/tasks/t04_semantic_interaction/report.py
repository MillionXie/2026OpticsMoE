"""Build the paper-facing OpenMoji main/D2NN comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(method: str, run_dir: Path) -> dict[str, Any]:
    result = _json(run_dir / "selected_checkpoint_test_evaluation.json")
    metrics = result["metrics"]["overall"]
    architecture = _json(run_dir / "student_architecture.json")
    return {
        "method": method,
        "status": "measured_simulation",
        "test_samples": result["test_samples"],
        "selected_epoch": result["selected_epoch"],
        "changed_cell_accuracy": metrics["changed_cell_accuracy"],
        "foreground_category_accuracy": metrics["foreground_category_accuracy"],
        "edit_grid_iou": metrics["edit_grid_iou"],
        "object_f1": metrics["object_f1"],
        "scene_exact_match": metrics["scene_exact_match"],
        "router": architecture["router"]["backend"],
        "top_k": architecture["router"].get("top_k"),
        "router_audit": result.get("router_audit"),
        "best_checkpoint_sha256": _sha(run_dir / "best_checkpoint.pt"),
        "speed_ms_5090d": None,
        "power_w_5090d": None,
    }


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    metrics = (("changed_cell_accuracy", "Changed"), ("foreground_category_accuracy", "Category"), ("edit_grid_iou", "Edit IoU"), ("object_f1", "Object F1"), ("scene_exact_match", "Exact"))
    figure, axis = plt.subplots(figsize=(5.2, 2.1), constrained_layout=True)
    x = list(range(len(metrics)))
    width = 0.34
    colors = ["#2F7E79", "#777777"]
    for index, row in enumerate(rows):
        values = [100.0 * float(row[key]) for key, _ in metrics]
        axis.bar([value + (index - 0.5) * width for value in x], values, width, label=row["method"], color=colors[index])
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Test performance (%)")
    axis.set_ylim(0, 105)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.set_axisbelow(True)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--d2nn", required=True)
    parser.add_argument("--qwen-pending", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "reports" / "dc20_comparison"))
    args = parser.parse_args()
    main_dir, d2nn_dir = Path(args.main), Path(args.d2nn)
    measured = [_row("Optical Router Top-2", main_dir), _row("Matched D2NN", d2nn_dir)]
    pending = _json(Path(args.qwen_pending))
    qwen = {"method": "Frozen Qwen", "status": pending["status"], "changed_cell_accuracy": None, "foreground_category_accuracy": None, "edit_grid_iou": None, "object_f1": None, "scene_exact_match": None, "router": "none", "top_k": None, "router_audit": None, "speed_ms_5090d": None, "power_w_5090d": None}
    rows = [*measured, qwen]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps({"schema_version": 1, "no_validation": True, "test_used_for_selection": True, "runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plot(measured, output / "openmoji_performance")
    for key, run_dir in (("main", main_dir), ("d2nn", d2nn_dir)):
        source = run_dir / "best_visualization"
        if source.is_dir():
            shutil.copytree(source, output / f"{key}_best_visualization", dirs_exist_ok=True)
    lines = ["# OpenMoji 正式仿真比较", "", "无 validation；每 5 epoch 测 test 并按 changed-cell accuracy 选模。", "", "| 方法 | Router | Changed↑ | Category↑ | Edit IoU↑ | Object F1↑ | Exact↑ |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in measured:
        lines.append(f"| {row['method']} | {row['router']} | {row['changed_cell_accuracy']:.4f} | {row['foreground_category_accuracy']:.4f} | {row['edit_grid_iou']:.4f} | {row['object_f1']:.4f} | {row['scene_exact_match']:.4f} |")
    lines += ["| Frozen Qwen | none | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D |", "", "大模型性能、速度与功耗均留待 RTX 5090 D 按 `BASELINE_5090D_TODO.md` 实测。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
