from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


MODEL_PALETTE = {
    "general_d2nn": "#0072B2",
    "fixed_route_moe": "#E69F00",
    "learnable_route_moe": "#009E73",
    "lenet5": "#CC79A7",
}

DATASET_PALETTE = {
    "mnist": "#0072B2",
    "fashionmnist": "#D55E00",
    "kmnist": "#009E73",
    "emnist": "#CC79A7",
    "cifar10": "#E69F00",
}


def _font_available(name):
    names = {font.name for font in font_manager.fontManager.ttflist}
    return name in names


def set_paper_style(font="Times New Roman", font_size=10):
    chosen_font = font
    if not _font_available(font):
        chosen_font = "DejaVu Serif"
        print(f"warning: font '{font}' not found; using '{chosen_font}'")
    plt.rcParams.update(
        {
            "font.family": chosen_font,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#9aa0a6",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def get_model_palette():
    return dict(MODEL_PALETTE)


def get_dataset_palette():
    return dict(DATASET_PALETTE)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def save_figure(fig, out_path_base, formats=("png", "pdf", "svg"), dpi=300):
    out_path_base = Path(out_path_base)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_path_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")

