"""Create the compact paper-facing T02 main-versus-D2NN report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def _row(label: str, path: Path) -> dict[str, Any]:
    report = _json(path / "selected_checkpoint_test_evaluation.json")
    metrics = report["metrics"]
    architecture = _json(path / "student_architecture.json")
    return {
        "method": label,
        "run_dir": str(path.resolve()),
        "selected_epoch": report["selected_epoch"],
        "test_samples": report["test_samples"],
        "pck_at_0.2_torso": metrics["pck_at_0.2_torso"],
        "pckh_at_0.5_head": metrics["pckh_at_0.5_head"],
        "normalized_mean_error_torso": metrics["normalized_mean_error_torso"],
        "mean_pixel_error": metrics["mean_pixel_error"],
        "best_checkpoint_sha256": _sha(path / "best_checkpoint.pt"),
        "last_checkpoint_sha256": _sha(path / "last_checkpoint.pt"),
        "phase_parameters": architecture["optics"]["phase_parameters"],
        "router_backend": architecture["router"]["backend"],
    }


def _plot(rows: list[dict[str, Any]], destination: Path) -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(4.72, 1.93), constrained_layout=True)
    labels = [row["method"] for row in rows]
    colors = ["#2F7E79", "#7A7A7A"]
    x = range(len(rows))
    pck = [100 * float(row["pck_at_0.2_torso"]) for row in rows]
    pckh = [100 * float(row["pckh_at_0.5_head"]) for row in rows]
    width = 0.34
    axes[0].bar([value - width / 2 for value in x], pck, width, label="PCK@0.2", color=colors[0])
    axes[0].bar([value + width / 2 for value in x], pckh, width, label="PCKh@0.5", color="#D49A3A")
    axes[0].set_ylabel("Correct keypoints (%)")
    axes[0].legend(frameon=False)
    nme = [float(row["normalized_mean_error_torso"]) for row in rows]
    axes[1].bar(x, nme, color=colors)
    axes[1].set_ylabel("Torso-normalized mean error")
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=12, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.set_axisbelow(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--d2nn", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports" / "dc20_comparison"),
    )
    args = parser.parse_args()
    rows = [
        _row("Optical Router + MoE", Path(args.main)),
        _row("Matched D2NN", Path(args.d2nn)),
    ]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps({"schema_version": 1, "runs": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output / "lsp_pose_comparison")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
