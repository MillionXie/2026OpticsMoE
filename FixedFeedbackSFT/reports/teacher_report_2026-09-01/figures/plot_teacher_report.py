"""Rebuild the two advisor-report figures from committed source-data tables.

Outputs are editable SVG plus PDF, 600-dpi TIFF and 300-dpi PNG previews.
The script performs no statistical test: dots are individual downstream seeds,
and error bars are mean +/- sample standard deviation (n=3).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "source_data"

COLORS = {
    "optical": "#0072B2",
    "electronic": "#E69F00",
    "source": "#CC79A7",
    "random": "#D55E00",
    "bp": "#2F2F2F",
    "pretrained": "#009E73",
    "scratch": "#56B4E9",
    "muted": "#7A7A7A",
    "grid": "#D8D8D8",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    svg_path = HERE / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    # Matplotlib writes path commands with trailing spaces. Strip those spaces so
    # the committed editable SVG also passes Git's whitespace preflight.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        HERE / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    face_alpha: float = 0.12,
    fontsize: float = 6.7,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, face_alpha),
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    dashed: bool = False,
    curve: float = 0.0,
    linewidth: float = 1.1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def architecture_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a")

    draw_box(ax, (0.015, 0.37), 0.105, 0.27, "Static\nimage", COLORS["muted"])
    draw_box(
        ax,
        (0.155, 0.37),
        0.145,
        0.27,
        "Frozen Qwen\npatch + position stem\n196 tokens x 1024",
        COLORS["pretrained"],
        fontsize=6.2,
    )
    draw_box(ax, (0.335, 0.37), 0.105, 0.27, "Trainable\n1024 -> 224\nadapter", COLORS["electronic"])
    draw_box(
        ax,
        (0.485, 0.31),
        0.245,
        0.39,
        "8-stage optoelectronic token body\n[token-axis -> feature-axis] x 4\n3 latent optical banks (not RGB)",
        COLORS["optical"],
        fontsize=6.6,
    )
    draw_box(ax, (0.775, 0.37), 0.105, 0.27, "Task\nreadout", COLORS["electronic"])
    draw_box(ax, (0.915, 0.37), 0.07, 0.27, "Loss", COLORS["muted"])

    for x0, x1 in [(0.12, 0.155), (0.30, 0.335), (0.44, 0.485), (0.73, 0.775), (0.88, 0.915)]:
        add_arrow(ax, (x0, 0.505), (x1, 0.505), COLORS["bp"])

    ax.text(0.04, 0.86, "Black arrows: current physical forward in every method", color=COLORS["bp"], fontsize=6.7)
    ax.text(0.50, 0.78, "Exact BP: current inter-stage optical connector", color=COLORS["optical"], fontsize=6.7)
    add_arrow(ax, (0.72, 0.77), (0.49, 0.77), COLORS["optical"], curve=0.0)
    ax.text(0.52, 0.14, "FA-source: fixed source operator for inter-stage optical error transport", color=COLORS["source"], fontsize=6.7)
    add_arrow(ax, (0.72, 0.24), (0.49, 0.24), COLORS["source"], dashed=True, curve=0.0)
    ax.text(0.756, 0.73, "Electronic residuals, gates, norms and head: exact BP", color=COLORS["electronic"], fontsize=6.3, ha="center")


def grouped_values(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[int, float]]:
    out: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["body_regime"], row["task"], row["method"])
        out.setdefault(key, {})[int(row["seed"])] = float(row["value"]) * 100.0
    return out


def dot_summary(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    marker: str,
    *,
    label: str | None = None,
) -> None:
    jitter = np.linspace(-0.055, 0.055, len(values))
    ax.scatter(
        np.full(len(values), x) + jitter,
        values,
        s=16,
        color=mpl.colors.to_rgba(color, 0.58),
        marker=marker,
        linewidths=0.45,
        edgecolors=color,
        zorder=3,
    )
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ax.errorbar(
        x,
        mean,
        yerr=sd,
        fmt=marker,
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=5.2,
        capsize=2.4,
        elinewidth=1.0,
        label=label,
        zorder=4,
    )


def build_figure1() -> None:
    rows = read_csv("downstream_runs.csv")
    values = grouped_values(rows)
    tasks = ["caltech101", "isic2016", "lsp"]
    task_labels = ["Caltech-101\nTop-1", "ISIC2016\nmIoU", "LSP\nPCK@0.2 torso"]
    seeds = [2026, 2027, 2028]

    fig = plt.figure(figsize=(7.2, 6.15), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.18], hspace=0.42, wspace=0.42)
    architecture_panel(fig.add_subplot(grid[0, :]))

    ax_b = fig.add_subplot(grid[1, 0])
    panel_label(ax_b, "b")
    x = np.arange(len(tasks), dtype=float)
    for idx, task in enumerate(tasks):
        pretrained = np.array([values[("imagenet_pretrained", task, "noft")][s] for s in seeds])
        no_imagenet = np.array([values[("no_imagenet_body", task, "noft")][s] for s in seeds])
        delta = pretrained - no_imagenet
        dot_summary(
            ax_b,
            x[idx],
            delta,
            COLORS["pretrained"],
            "o",
            label="Pretrained - no-ImageNet body" if idx == 0 else None,
        )
    ax_b.axhline(0, color=COLORS["muted"], lw=0.7)
    ax_b.set_xticks(x, task_labels)
    ax_b.set_ylabel("Head-only pretraining gain (pp)")
    ax_b.set_title("ImageNet body improves\nhead-only transfer", pad=6)
    ax_b.set_ylim(0, 64)
    ax_b.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax_b.text(0.02, 0.97, "n=3 downstream seeds", transform=ax_b.transAxes, va="top", color=COLORS["muted"], fontsize=6.2)

    ax_c = fig.add_subplot(grid[1, 1])
    panel_label(ax_c, "c")
    offsets = {"imagenet_pretrained": -0.13, "no_imagenet_body": 0.13}
    regime_styles = {
        "imagenet_pretrained": (COLORS["pretrained"], "o", "ImageNet-pretrained body"),
        "no_imagenet_body": (COLORS["scratch"], "s", "No-ImageNet body"),
    }
    for regime, offset in offsets.items():
        color, marker, label = regime_styles[regime]
        for idx, task in enumerate(tasks):
            delta = np.array(
                [values[(regime, task, "fa_source")][s] - values[(regime, task, "bp_current")][s] for s in seeds]
            )
            dot_summary(ax_c, x[idx] + offset, delta, color, marker, label=label if idx == 0 else None)
    ax_c.axhline(0, color=COLORS["bp"], lw=0.8)
    ax_c.set_xticks(x, task_labels)
    ax_c.set_ylabel("FA-source - exact BP (pp)")
    ax_c.set_title("Fixed source feedback\ntracks exact BP", pad=6)
    ax_c.set_ylim(-2.7, 2.7)
    ax_c.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax_c.legend(frameon=False, loc="lower left", handletextpad=0.4)

    ax_d = fig.add_subplot(grid[1, 2])
    panel_label(ax_d, "d")
    for regime, offset in offsets.items():
        color, marker, label = regime_styles[regime]
        for idx, task in enumerate(tasks):
            delta = np.array(
                [values[(regime, task, "fa_random")][s] - values[(regime, task, "fa_source")][s] for s in seeds]
            )
            dot_summary(ax_d, x[idx] + offset, delta, color, marker, label=label if idx == 0 else None)
    ax_d.axhline(0, color=COLORS["bp"], lw=0.8)
    ax_d.set_xticks(x, task_labels)
    ax_d.set_ylabel("FA-random - FA-source (pp)")
    ax_d.set_title("Random feedback weakens\nwithout body pretraining", pad=6)
    ax_d.set_ylim(-10.5, 2.0)
    ax_d.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax_d.text(
        0.02,
        0.04,
        "Same random body init;\nseeds vary downstream training",
        transform=ax_d.transAxes,
        fontsize=5.9,
        color=COLORS["muted"],
        va="bottom",
    )

    fig.suptitle(
        "Reusable optical feedback supports downstream adaptation across three vision tasks",
        x=0.5,
        y=0.995,
        fontsize=10.2,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.94, bottom=0.10, left=0.08, right=0.985)
    save_figure(fig, "fig1_fixed_feedback_evidence")
    plt.close(fig)


def build_figure2() -> None:
    growth = read_csv("p13_growth_history.csv")
    scale = read_csv("scale_audit.csv")
    budget_rows: dict[int, dict[str, str]] = {}
    for row in scale:
        budget_rows.setdefault(int(row["depth"]), row)
    depths = np.array(sorted(budget_rows))
    phase_m = np.array([float(budget_rows[d]["optical_phase_params"]) / 1e6 for d in depths])
    optical_fraction = np.array([float(budget_rows[d]["optical_body_param_fraction"]) * 100 for d in depths])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    panel_label(ax_a, "a")
    bars = ax_a.bar(depths.astype(str), phase_m, color=mpl.colors.to_rgba(COLORS["optical"], 0.74), width=0.64)
    ax_a.set_ylabel("Trainable optical phase (million)")
    ax_a.set_xlabel("OEO stage depth")
    ax_a.set_title("Optical parameters scale;\nelectronics stay near 0.965 M")
    ax_a.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    for bar, value in zip(bars, phase_m):
        ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.32, f"{value:.2f}", ha="center", va="bottom", fontsize=5.9)
    twin_a = ax_a.twinx()
    twin_a.spines["top"].set_visible(False)
    twin_a.plot(depths.astype(str), optical_fraction, color=COLORS["source"], marker="o", lw=1.2, ms=3.8)
    twin_a.set_ylabel("Optical share of trainable body params (%)", color=COLORS["source"])
    twin_a.tick_params(axis="y", colors=COLORS["source"])
    twin_a.set_ylim(45, 100)

    panel_label(ax_b, "b")
    epochs = np.array([int(row["epoch"]) for row in growth])
    top1 = np.array([float(row["val_top1"]) * 100 for row in growth])
    alpha = np.array([float(row["alpha"]) for row in growth])
    ax_b.plot(epochs, top1, color=COLORS["optical"], marker="o", ms=4, lw=1.35, label="Validation Top-1")
    ax_b.axhline(top1[0], color=COLORS["muted"], linestyle="--", lw=0.8, label="P11 8-stage source")
    ax_b.set_xlabel("Growth epoch (0 = source)")
    ax_b.set_ylabel("ImageNet validation Top-1 (%)")
    ax_b.set_title("Interim 16-stage growth;\nfull depth begins at alpha=1")
    ax_b.axvline(10, color=COLORS["random"], linestyle=":", lw=0.9)
    ax_b.text(
        9.82,
        48.62,
        "alpha=1",
        color=COLORS["random"],
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="bottom",
        fontsize=6.2,
    )
    ax_b.set_xlim(-0.35, 10.35)
    ax_b.set_xticks([0, 2, 4, 6, 8, 10])
    ax_b.set_ylim(48.5, 52.1)
    ax_b.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    twin_b = ax_b.twinx()
    twin_b.spines["top"].set_visible(False)
    twin_b.plot(epochs, alpha, color=COLORS["electronic"], marker="s", ms=3.3, lw=1.0, label="Growth alpha")
    twin_b.set_ylabel("New-stage alpha", color=COLORS["electronic"])
    twin_b.tick_params(axis="y", colors=COLORS["electronic"])
    twin_b.set_ylim(-0.04, 1.04)
    lines, labels = ax_b.get_legend_handles_labels()
    lines2, labels2 = twin_b.get_legend_handles_labels()
    ax_b.legend(lines + lines2, labels + labels2, frameon=False, loc="lower right")

    panel_label(ax_c, "c")
    trained_rows = growth[1:]
    tr_epochs = np.array([int(row["epoch"]) for row in trained_rows])
    motion = np.array([float(row["phase_motion_mean_abs_rad"]) for row in trained_rows])
    ax_c.plot(tr_epochs, motion, color=COLORS["source"], marker="o", lw=1.35, ms=4)
    ax_c.fill_between(tr_epochs, 0, motion, color=mpl.colors.to_rgba(COLORS["source"], 0.09))
    ax_c.set_xlabel("Growth epoch")
    ax_c.set_ylabel("Mean absolute phase motion (rad)")
    ax_c.set_title("Inserted optical stages are learning,\nnot frozen")
    ax_c.set_xticks(tr_epochs)
    ax_c.set_ylim(0, 0.68)
    ax_c.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax_c.text(
        0.04,
        0.93,
        "Every reported epoch: 8/8 new + 8/8 carried\nphase tensors have finite, non-zero gradients",
        transform=ax_c.transAxes,
        va="top",
        fontsize=6.2,
        color=COLORS["muted"],
    )

    panel_label(ax_d, "d")
    audit = [row for row in scale if row["feedback_mode"]]
    modes = ["bp_current", "fa_source", "fa_random"]
    mode_labels = ["Exact BP", "FA-source", "FA-random"]
    mode_colors = [COLORS["bp"], COLORS["source"], COLORS["random"]]
    x_depth = np.arange(2, dtype=float)
    width = 0.22
    for mode_idx, (mode, label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        vals = []
        for depth in (64, 100):
            row = next(r for r in audit if int(r["depth"]) == depth and r["feedback_mode"] == mode)
            vals.append(float(row["mean_step_s"]))
        ax_d.bar(x_depth + (mode_idx - 1) * width, vals, width=width, color=mpl.colors.to_rgba(color, 0.78), label=label)
    ax_d.set_xticks(x_depth, ["64 stages\n64/64 grads", "100 stages\n100/100 grads"])
    ax_d.set_ylabel("Synthetic audit step time (s)")
    ax_d.set_title("Full-depth CUDA gradients\npass at alpha=1")
    ax_d.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax_d.legend(frameon=False, ncol=1, loc="upper left")
    ax_d.text(
        0.98,
        0.04,
        "Engineering audit only;\nnot semantic performance",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=COLORS["random"],
    )

    fig.suptitle(
        "Function-preserving growth reaches engineering scale, but semantic validation remains incomplete",
        y=0.995,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.09, right=0.93, hspace=0.42, wspace=0.45)
    save_figure(fig, "fig2_depth_growth_status")
    plt.close(fig)


def print_numeric_summary() -> None:
    rows = read_csv("downstream_runs.csv")
    values = grouped_values(rows)
    tasks = ["caltech101", "isic2016", "lsp"]
    seeds = [2026, 2027, 2028]
    for regime in ("imagenet_pretrained", "no_imagenet_body"):
        print(regime)
        for task in tasks:
            parts = []
            for method in ("noft", "bp_current", "fa_source", "fa_random"):
                arr = np.array([values[(regime, task, method)][s] for s in seeds])
                parts.append(f"{method}={arr.mean():.3f}+/-{arr.std(ddof=1):.3f}")
            print(f"  {task}: " + "; ".join(parts))


if __name__ == "__main__":
    configure_style()
    print_numeric_summary()
    build_figure1()
    build_figure2()
    print(f"Figures written to {HERE}")
