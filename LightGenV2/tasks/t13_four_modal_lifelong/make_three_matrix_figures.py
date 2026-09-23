"""Render the three formal four-modal comparison matrices for presentations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "figures"
TASKS = ("eurosat", "clevr", "speech", "physical")
COL_LABELS = (
    "EuroSAT\nRGB / SAR",
    "CLEVR\nImage–Text",
    "Speech\nAudio–Text",
    "Physical\nVideo–Text",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def triangular_matrix(payload: dict) -> np.ndarray:
    matrix = np.full((4, 4), np.nan, dtype=float)
    for row, scores in enumerate(payload["test"]):
        for col, task in enumerate(TASKS):
            if task in scores:
                matrix[row, col] = 100.0 * float(scores[task])
    return matrix


def full_matrix(payload: dict) -> np.ndarray:
    rows = payload["score_matrix"]
    return np.array(
        [[100.0 * float(rows[source][target]) for target in TASKS] for source in TASKS],
        dtype=float,
    )


def text_color(value: float) -> str:
    return "white" if value < 24 or value > 64 else "#102235"


def draw_matrix(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    subtitle: str,
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...] = COL_LABELS,
    xlabel: str = "Evaluation task",
    ylabel: str = "Model state / source",
) -> mpl.image.AxesImage:
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f1f3f5")
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(4), labels=col_labels)
    ax.set_yticks(range(4), labels=row_labels)
    ax.tick_params(axis="x", labelrotation=0, length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=27)
    ax.text(
        0.0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#52606d",
    )

    for row in range(4):
        for col in range(4):
            value = data[row, col]
            if np.isnan(value):
                label = "—"
                color = "#9aa5b1"
                weight = "normal"
            else:
                label = f"{value:.2f}"
                color = text_color(value)
                weight = "semibold" if row == col else "normal"
            ax.text(col, row, label, ha="center", va="center", color=color,
                    fontsize=11, fontweight=weight)

    for edge in np.arange(-0.5, 4.5, 1.0):
        ax.axhline(edge, color="white", linewidth=1.5)
        ax.axvline(edge, color="white", linewidth=1.5)
    for index in range(4):
        ax.add_patch(
            mpl.patches.Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#102235",
                linewidth=1.8,
            )
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#263442",
            "xtick.color": "#263442",
            "ytick.color": "#263442",
            "text.color": "#102235",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    moe = triangular_matrix(read_json(REPORTS / "moe_replay_continual_matrix_s17.json"))
    frozen = full_matrix(
        read_json(REPORTS / "fair_986_inference_only_4x4_clevr12_s17" / "matrix.json")
    )
    d2nn = triangular_matrix(read_json(REPORTS / "d2nn_no_replay_continual_matrix_s17.json"))

    entries = (
        (
            moe,
            "A  Optical MoE + full replay",
            "Expandable 4→8→12→16 experts · lower-triangle mean 72.06%",
            tuple(f"After {name}" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_1_moe_full_replay",
        ),
        (
            frozen,
            "B  Independent frozen D2NN inference",
            "Row optics + target task's original Linear head · zero per-cell adaptation",
            tuple(f"{name} optics" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_2_frozen_d2nn_cross_modal",
        ),
        (
            d2nn,
            "C  Reconfigurable D2NN, no replay",
            "One evolving optical state · lower-triangle mean 64.58%",
            tuple(f"After {name}" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_3_d2nn_no_replay",
        ),
    )

    for data, title, subtitle, rows, stem in entries:
        fig, ax = plt.subplots(figsize=(7.1, 6.0), constrained_layout=True)
        image = draw_matrix(ax, data, title, subtitle, rows)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label("Balanced accuracy (%)")
        colorbar.outline.set_visible(False)
        save_figure(fig, stem)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18.3, 6.2), constrained_layout=True)
    images = []
    for ax, (data, title, subtitle, rows, _) in zip(axes, entries):
        images.append(draw_matrix(ax, data, title, subtitle, rows))
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Balanced accuracy (%)")
    colorbar.outline.set_visible(False)
    fig.suptitle(
        "Four-modal optical lifelong learning — three formal matrices",
        fontsize=16,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.015,
        "Full test sets · shared 986×986 effective phase aperture · one Linear(784, C) electronic readout · higher is better",
        ha="center",
        color="#52606d",
        fontsize=10,
    )
    save_figure(fig, "three_formal_matrices")
    plt.close(fig)

    chinese_columns = (
        "EuroSAT\nRGB / SAR",
        "CLEVR\n图文",
        "Speech\n音文",
        "Physical\n视频文",
    )
    chinese_entries = (
        (
            moe,
            "A  光学 MoE + 全量回放",
            "固定 16 槽，专家逐步激活 4→8→12→16 · 下三角均值 72.06%",
            tuple(f"学完 {name}" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_1_moe_full_replay_zh",
        ),
        (
            frozen,
            "B  独立 D2NN 纯推理",
            "行光学权重 + 列任务原始单层 Linear · 每个单元零微调",
            tuple(f"{name} 光学" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_2_frozen_d2nn_cross_modal_zh",
        ),
        (
            d2nn,
            "C  可重构 D2NN，无回放",
            "单一光学权重顺序更新 · 下三角均值 64.58%",
            tuple(f"学完 {name}" for name in ("EuroSAT", "CLEVR", "Speech", "Physical")),
            "matrix_3_d2nn_no_replay_zh",
        ),
    )
    mpl.rcParams["font.family"] = "Microsoft YaHei"
    for data, title, subtitle, rows, stem in chinese_entries:
        fig, ax = plt.subplots(figsize=(7.1, 6.0), constrained_layout=True)
        image = draw_matrix(
            ax,
            data,
            title,
            subtitle,
            rows,
            col_labels=chinese_columns,
            xlabel="测试任务",
            ylabel="学习阶段 / 光学来源",
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label("平衡准确率 (%)")
        colorbar.outline.set_visible(False)
        save_figure(fig, stem)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18.3, 6.2), constrained_layout=True)
    images = []
    for ax, (data, title, subtitle, rows, _) in zip(axes, chinese_entries):
        images.append(
            draw_matrix(
                ax,
                data,
                title,
                subtitle,
                rows,
                col_labels=chinese_columns,
                xlabel="测试任务",
                ylabel="学习阶段 / 光学来源",
            )
        )
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("平衡准确率 (%)")
    colorbar.outline.set_visible(False)
    fig.suptitle("四模态光学终身学习：三张正式矩阵", fontsize=16, fontweight="semibold")
    fig.text(
        0.5,
        -0.015,
        "完整测试集 · 统一 986×986 有效相位口径 · 电子读出仅一层 Linear(784, C) · 数值越高越好",
        ha="center",
        color="#52606d",
        fontsize=10,
    )
    save_figure(fig, "three_formal_matrices_zh")
    plt.close(fig)


if __name__ == "__main__":
    main()
