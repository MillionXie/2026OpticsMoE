"""Build fixed-sample comparison plates with per-image perceptual scores.

Creates new scientific figures from the archived PNGs; it does not alter any model
output or choose images based on score. Figure indices come from the original audit.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import io
import json
import textwrap
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


MODEL_KEY = {
    "Small SIM": "small_sim", "Small EXP*": "small_exp_tuned",
    "Large SIM": "large_sim", "Qwen28 SIM": "qwen28_sim",
    "pix2pix-Turbo SIM": "pix2pix_turbo_sim",
}


def load_csv(path: Path, keys: tuple[str, ...]) -> dict[tuple, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {tuple(row[key] for key in keys): row for row in csv.DictReader(handle)}


def png(label: str, index: int) -> Image.Image:
    archive, prefix = IMAGE_SOURCE[label]
    return Image.open(io.BytesIO(archive.read(f"{prefix}test_{index:05d}.png"))).convert("RGB")


def score_text(model: str, index: int, mode: str) -> str:
    key = (MODEL_KEY[model], str(index))
    l, d = LPIPS[key], DISTS[key]
    if mode == "object":
        return (f"edit PSNR {float(l['edit_region_psnr_db']):.1f} dB\n"
                f"ROI LPIPS {float(l['lpips_alex_product_roi']):.3f}  DISTS {float(d['dists_product_roi']):.3f}")
    return (f"PSNR {float(l['psnr_db']):.1f} dB\n"
            f"LPIPS {float(l['lpips_alex_full']):.3f}  DISTS {float(d['dists_full']):.3f}")


def title_text(label: str, mode: str) -> str:
    if label not in MODEL_KEY:
        return label
    key = (MODEL_KEY[label], mode)
    l, d = SUMMARY_L[key], SUMMARY_D[key]
    if mode == "object":
        return (f"{label}\nmean ROI LPIPS {float(l['lpips_alex_product_roi']):.3f}"
                f" / DISTS {float(d['dists_product_roi']):.3f}")
    return (f"{label}\nmean LPIPS {float(l['lpips_alex_full']):.3f}"
            f" / DISTS {float(d['dists_full']):.3f}")


def make(mode: str, include_exp: bool) -> Path:
    offset = {"background": 0, "object": 4, "joint": 8}[mode]
    indices = [i for i in INDICES if i % 12 == offset]
    if len(indices) != 6:
        raise ValueError(f"Expected six fixed {mode} examples, got {indices}")
    columns = ["Input", "GT", "Small SIM"]
    if include_exp:
        columns.append("Small EXP*")
    columns += ["Large SIM", "Qwen28 SIM", "pix2pix-Turbo SIM"]
    fig = plt.figure(figsize=(3.0 * len(columns) + 2.7, 18.7), dpi=120)
    grid = fig.add_gridspec(6, len(columns) + 1,
                           width_ratios=[1.20] + [1.0] * len(columns),
                           left=0.025, right=0.995, top=0.915, bottom=0.055,
                           wspace=0.08, hspace=0.35)
    fig.suptitle(f"T12 {mode} | fixed TEST samples | saved 256px RGB outputs", fontsize=17, y=0.989)
    for row, index in enumerate(indices):
        sample = SAMPLES[(str(index),)]
        label = fig.add_subplot(grid[row, 0])
        label.axis("off")
        label.text(0.0, 0.96,
                   f"test_{index:05d}  {sample['category']}\n\n"
                   + textwrap.fill(sample["prompt"], width=31),
                   ha="left", va="top", fontsize=9, transform=label.transAxes)
        for col, name in enumerate(columns, start=1):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(png(name, index))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title_text(name, mode), fontsize=9, pad=14)
            if name in MODEL_KEY:
                ax.text(0.5, -0.055, score_text(name, index, mode),
                        ha="center", va="top", fontsize=8, transform=ax.transAxes)
    note = ("Object edit ROI = union of source and target product masks dilated 12 px; ROI features use its bounding crop. "
            "Other modes use full-image scores. Lower LPIPS/DISTS is better. Metrics are recomputed from saved PNGs.")
    if include_exp:
        note += "  *Small EXP is physical output from a DIFFERENT electronically tuned weight; not a matched clean-SIM baseline."
    fig.text(0.025, 0.018, note, fontsize=9, ha="left", va="bottom", wrap=True)
    output = OUT / (f"annotated_{mode}_{'with_exp' if include_exp else 'clean_sim'}.png")
    with output.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=120)
    plt.close(fig)
    print(output)
    return output

def main():
    global OUT, IMAGE_SOURCE, LPIPS, DISTS, SUMMARY_L, SUMMARY_D, SAMPLES, INDICES
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "metrics", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    if OUT.exists():
        parser.error("--output must be a new directory")
    four = args.repo / "handoffs/t12_four_group_summary_20260927"
    LPIPS = load_csv(args.metrics / "per_image_lpips.csv", ("model", "index"))
    DISTS = load_csv(args.metrics / "per_image_dists.csv", ("model", "index"))
    SUMMARY_L = load_csv(args.metrics / "summary_lpips.csv", ("model", "mode"))
    SUMMARY_D = load_csv(args.metrics / "summary_dists.csv", ("model", "mode"))
    SAMPLES = load_csv(four / "small_sim/sample_metrics.csv", ("test_index",))
    INDICES = json.loads((four / "figure_indices.json").read_text(encoding="utf-8"))["indices"]
    with ExitStack() as stack:
        def archive(path):
            return stack.enter_context(zipfile.ZipFile(path))
        small = archive(args.repo / "handoffs/t12_lab_robust17m_20260927/paper_bundle.zip")
        exp = archive(args.repo / "handoffs/t12_lab_robust17m_20260927/decoder20_comparison/paper_bundle_full.zip")
        large = archive(four / "large_baseline_export/paper_bundle.zip")
        pix = archive(args.repo / "handoffs/t12_pix2pix_turbo_20260927/pix2pix_test_export.zip")
        IMAGE_SOURCE = {
            "Input": (small, "images/reference/"), "GT": (small, "images/target/"),
            "Small SIM": (small, "images/simulation/"), "Small EXP*": (exp, "images/physical_tuned/"),
            "Large SIM": (large, "images/large_sim/"), "Qwen28 SIM": (large, "images/qwen_baseline/"),
            "pix2pix-Turbo SIM": (pix, "images/generated/"),
        }
        # Exclusive directory creation protects all original figures and archives.
        OUT.mkdir(parents=True, exist_ok=False)
        for task in ("background", "object", "joint"):
            make(task, include_exp=False)
            make(task, include_exp=True)


if __name__ == "__main__":
    main()
