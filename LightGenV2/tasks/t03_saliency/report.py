"""Build the paper-facing SALICON main/D2NN comparison."""

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
    metrics = result["metrics"]
    architecture = _json(run_dir / "student_architecture.json")
    return {
        "method": method,
        "status": "measured_simulation",
        "test_samples": result["test_samples"],
        "selected_epoch": result["selected_epoch"],
        "cc": metrics["cc"],
        "kld": metrics["kld"],
        "sim": metrics["sim"],
        "nss": metrics["nss"],
        "auc_judd": metrics["auc_judd"],
        "mae": metrics["mae"],
        "router": architecture["router"]["backend"],
        "top_k": architecture["router"].get("top_k"),
        "router_audit": result.get("router_audit"),
        "best_checkpoint_sha256": _sha(run_dir / "best_checkpoint.pt"),
        "speed_ms_5090d": None,
        "power_w_5090d": None,
    }


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(5.2, 2.0), constrained_layout=True)
    labels = [row["method"] for row in rows]
    colors = ["#2F7E79", "#777777"]
    x = list(range(len(rows)))
    width = 0.18
    for offset, (key, label) in enumerate((("cc", "CC"), ("sim", "SIM"), ("auc_judd", "AUC-J"))):
        values = [float(row[key]) for row in rows]
        axes[0].bar([value + (offset - 1) * width for value in x], values, width, label=label)
    axes[0].set_ylabel("Higher is better")
    axes[0].legend(frameon=False, ncol=3)
    for offset, (key, label) in enumerate((("kld", "KLD"), ("mae", "MAE"))):
        values = [float(row[key]) for row in rows]
        axes[1].bar([value + (offset - 0.5) * 0.26 for value in x], values, 0.26, label=label, color=colors[offset])
    axes[1].set_ylabel("Lower is better")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=10, ha="right")
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
    qwen = {
        "method": "Frozen Qwen",
        "status": pending["status"],
        "cc": None, "kld": None, "sim": None, "nss": None,
        "auc_judd": None, "mae": None, "router": "none", "top_k": None,
        "router_audit": None,
        "speed_ms_5090d": None, "power_w_5090d": None,
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [*measured, qwen]
    (output / "comparison.json").write_text(json.dumps({"schema_version": 1, "no_validation": True, "test_used_for_selection": True, "runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plot(measured, output / "salicon_performance")
    for key, run_dir in (("main", main_dir), ("d2nn", d2nn_dir)):
        source = run_dir / "best_visualization"
        if source.is_dir():
            shutil.copytree(source, output / f"{key}_best_visualization", dirs_exist_ok=True)
    lines = [
        "# SALICON 正式仿真比较", "",
        "无 validation；SALICON 官方 val2014 作为 public test，并用于每 5 epoch 选模。", "",
        "| 方法 | Router | CC↑ | KLD↓ | SIM↑ | NSS↑ | AUC-J↑ | MAE↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in measured:
        lines.append(f"| {row['method']} | {row['router']} | {row['cc']:.4f} | {row['kld']:.4f} | {row['sim']:.4f} | {row['nss']:.4f} | {row['auc_judd']:.4f} | {row['mae']:.4f} |")
    lines += ["| Frozen Qwen | none | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D |", "", "大模型速度与功耗也必须留待 RTX 5090 D 按 `BASELINE_5090D_TODO.md` 实测。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
