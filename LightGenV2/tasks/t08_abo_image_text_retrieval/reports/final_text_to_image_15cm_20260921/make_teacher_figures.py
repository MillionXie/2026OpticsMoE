from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    }
)


def _finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def performance() -> None:
    labels = ["冻结Qwen\n2048D", "最终光电模型\n64D", "同权重去光\n64D"]
    values = [82, 88, 74]
    colors = ["#778899", "#1677b8", "#d98e32"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylim(0, 100)
    ax.set_ylabel("文搜图 Hit@1 (%)")
    ax.set_title("ABO easy100 文搜图：最终结果与消融", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=15,
            weight="bold",
        )
    ax.annotate(
        "光支路贡献：+14个百分点",
        xy=(2, 74),
        xytext=(1.52, 57),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        ha="center",
    )
    ax.text(
        0.5,
        -0.22,
        "Baseline使用完整2048维冻结Qwen；最终光电模型输出64维。二者维度不同，主表需明确标注。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.28)
    _finish(fig, "performance_comparison.png")


def compression() -> None:
    params = [3.177750, 2.882070, 2.734230]
    hit1 = [90, 90, 88]
    labels = ["原版\n384隐层", "第一档\n192隐层", "最终版\n96隐层"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(params, hit1, "o-", color="#1677b8", linewidth=2.5, markersize=9)
    for x, y, label in zip(params, hit1, labels):
        ax.annotate(
            f"{label}\n{x:.3f}M / {y}%",
            (x, y),
            xytext=(0, 15 if y == 90 else -42),
            textcoords="offset points",
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#bbbbbb"},
        )
    ax.set_xlim(2.64, 3.27)
    ax.set_ylim(84, 93)
    ax.invert_xaxis()
    ax.set_xlabel("Checkpoint张量元素（百万；向右表示更小）")
    ax.set_ylabel("Hit@1 (%)")
    ax.set_title("电子残差压缩—性能折中", weight="bold")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        -0.2,
        "最终版相对原版减少13.96%总张量；非相位部分减少19.12%。",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    fig.subplots_adjust(bottom=0.25)
    _finish(fig, "compression_tradeoff.png")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.2,
                edgecolor=color,
                facecolor=color + "18",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#555555",
            )
        )

    ax.text(7, 7.65, "15 cm 光Router Top-2文搜图推理架构（最终版）", ha="center", fontsize=17, weight="bold")
    ax.text(0.25, 5.95, "图库图像（离线编码）", weight="bold", color="#155b8a")
    ax.text(0.25, 2.55, "标题Query（在线编码）", weight="bold", color="#8a4d12")

    box(0.3, 4.75, 1.5, 0.85, "商品图像\n+ 图像Prompt", "#1677b8")
    box(2.15, 4.75, 1.7, 0.85, "冻结Qwen前端\nProcessor / Patch Embed", "#667788")
    box(4.2, 4.55, 2.05, 1.25, "Vision Stage 1\n光Router→Top-2专家\n+ 96隐层电子残差\n同尺度融合 α≈0.398", "#1677b8")
    box(6.6, 4.55, 2.0, 1.25, "Vision Stage 2\nGlobal相位传播\n+ 96隐层电子残差\n同尺度融合 α≈0.398", "#1677b8")
    box(0.3, 1.35, 1.5, 0.85, "英文商品标题\n+ 文本Prompt", "#d98e32")
    box(2.15, 1.35, 1.7, 0.85, "Tokenizer\n+ 冻结Embed Tokens", "#667788")
    box(9.0, 3.55, 2.0, 1.45, "Language Stage 1\n光Router→Top-2专家\n+ 96隐层因果残差\n同尺度融合 α≈0.392", "#d98e32")
    box(11.35, 3.55, 1.95, 1.45, "Language Stage 2\nGlobal相位传播\n+ 96隐层因果残差\n同尺度融合 α≈0.392", "#d98e32")
    box(9.0, 1.15, 2.0, 1.15, "Detector特征\nMean Pool + Max Pool\n192+192=384", "#6f5aa8")
    box(11.35, 1.15, 1.95, 1.15, "轻量读出头\nLN(384)→Linear(64)\nL2 Normalize", "#6f5aa8")
    box(5.85, 0.05, 2.25, 0.85, "余弦相似度\n100标题 × 2400图像", "#39754a")

    arrow(1.8, 5.18, 2.15, 5.18)
    arrow(3.85, 5.18, 4.2, 5.18)
    arrow(6.25, 5.18, 6.6, 5.18)
    arrow(8.6, 5.18, 9.0, 4.65)
    arrow(1.8, 1.78, 2.15, 1.78)
    arrow(3.85, 1.78, 9.0, 3.9)
    arrow(11.0, 4.28, 11.35, 4.28)
    arrow(12.32, 3.55, 10.0, 2.3)
    arrow(11.0, 1.72, 11.35, 1.72)
    arrow(11.35, 1.35, 8.1, 0.48)

    ax.text(
        7,
        6.75,
        "原Qwen Vision/Language Transformer块在学生推理中被旁路；不启用Attention。\n"
        "标题库/图库特征可预计算，在线查询只运行标题的Language两阶段。",
        ha="center",
        va="center",
        color="#333333",
    )
    _finish(fig, "architecture_overview.png")


if __name__ == "__main__":
    performance()
    compression()
    architecture()
