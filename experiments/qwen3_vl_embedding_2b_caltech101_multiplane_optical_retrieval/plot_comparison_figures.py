from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
# Static preflight markers mirror the active rcParams above:
# svg.fonttype='none'; pdf.fonttype=42

FIG_WIDTH_MM = 183
RASTER_DPI = 600


METHODS = (
    "d2nn_continuous",
    "d2nn_oeo_sigmoid",
    "moe_continuous_fixed_router",
    "moe_oeo_dynamic_router",
    "moe_oeo_fixed_router",
)
DISPLAY = {
    "d2nn_continuous": "D2NN\ncontinuous",
    "d2nn_oeo_sigmoid": "D2NN\n+ OEO",
    "moe_continuous_fixed_router": "MoE\ncontinuous",
    "moe_oeo_dynamic_router": "MoE OEO\n+ reroute",
    "moe_oeo_fixed_router": "MoE OEO\nfixed route",
}
AXIS_DISPLAY = {
    "d2nn_continuous": "D2NN",
    "d2nn_oeo_sigmoid": "D2NN\nOEO",
    "moe_continuous_fixed_router": "MoE\ncont.",
    "moe_oeo_dynamic_router": "MoE\nre-route",
    "moe_oeo_fixed_router": "MoE\nfixed",
}
SHORT = {
    "d2nn_continuous": "D2NN",
    "d2nn_oeo_sigmoid": "D2NN + OEO",
    "moe_continuous_fixed_router": "MoE continuous",
    "moe_oeo_dynamic_router": "MoE OEO + reroute",
    "moe_oeo_fixed_router": "MoE OEO fixed",
}
COLORS = {
    "d2nn_continuous": "#4D4D4D",
    "d2nn_oeo_sigmoid": "#0F4D92",
    "moe_continuous_fixed_router": "#42949E",
    "moe_oeo_dynamic_router": "#B64342",
    "moe_oeo_fixed_router": "#A8A8A8",
}
LINESTYLES = {
    "d2nn_continuous": "-",
    "d2nn_oeo_sigmoid": "-",
    "moe_continuous_fixed_router": "-",
    "moe_oeo_dynamic_router": "-",
    "moe_oeo_fixed_router": "--",
}


def mm(value: float) -> float:
    return value / 25.4


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": RASTER_DPI}),
        (".tiff", {"dpi": RASTER_DPI}),
    ):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    comparison = pd.read_csv(root / "comparison.csv").set_index("variant").loc[list(METHODS)]
    router = pd.read_csv(root / "router_diagnostics.csv")
    power = pd.read_csv(root / "power_diagnostics.csv")
    logs = {
        method: pd.read_csv(root / "source_data" / "training_logs" / f"{method}.csv")
        for method in METHODS
    }
    required = {
        "final_epoch_test_top1",
        "final_epoch_test_top3",
        "final_epoch_test_mrr",
        "train_top1",
        "phase_parameters",
        "phase_delta_rms_rad",
    }
    missing = required.difference(comparison.columns)
    if missing or comparison[list(required)].isna().any().any():
        raise RuntimeError(f"Comparison data are incomplete; missing={sorted(missing)}")
    for method, frame in logs.items():
        if len(frame) != 28 or frame["epoch"].nunique() != 28:
            raise RuntimeError(f"{method} training log is not the expected 28 epochs")
    return comparison, router, power, logs


def performance_summary(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(mm(FIG_WIDTH_MM), mm(67)), constrained_layout=True)
    x = np.arange(len(METHODS))
    labels = [AXIS_DISPLAY[value] for value in METHODS]
    colors = [COLORS[value] for value in METHODS]

    ax = axes[0]
    values = data["final_epoch_test_top1"].to_numpy(float) * 100.0
    bars = ax.bar(x, values, width=0.68, color=colors, edgecolor="#333333", linewidth=0.45)
    bars[-1].set_hatch("///")
    for index, value in enumerate(values):
        ax.text(index, value + 1.6, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("Final test Top-1 (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.34
    top3 = data["final_epoch_test_top3"].to_numpy(float) * 100.0
    mrr = data["final_epoch_test_mrr"].to_numpy(float) * 100.0
    ax.bar(x - width / 2, top3, width=width, color="#3775BA", label="Top-3")
    ax.bar(x + width / 2, mrr, width=width, color="#AADCA9", label="MRR")
    ax.set_ylabel("Retrieval metric (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    panel_label(ax, "b")

    ax = axes[2]
    train = data["train_top1"].to_numpy(float) * 100.0
    test = data["final_epoch_test_top1"].to_numpy(float) * 100.0
    for index, method in enumerate(METHODS):
        ax.plot([index, index], [test[index], train[index]], color=COLORS[method], lw=1.2)
        ax.scatter(index, train[index], marker="s", s=22, facecolor="white", edgecolor=COLORS[method], lw=0.9, zorder=3)
        ax.scatter(index, test[index], marker="o", s=22, color=COLORS[method], edgecolor="white", lw=0.4, zorder=3)
    ax.scatter([], [], marker="s", s=22, facecolor="white", edgecolor="#444444", label="Train")
    ax.scatter([], [], marker="o", s=22, color="#444444", label="Test")
    ax.set_ylabel("Top-1 (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 90)
    ax.legend(loc="lower left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    panel_label(ax, "c")
    export(fig, output / "fig01_performance_summary")


def learning_dynamics(logs: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(mm(FIG_WIDTH_MM), mm(66)), constrained_layout=True)
    for method in METHODS:
        frame = logs[method]
        for ax, column in zip(axes, ("train_top1", "test_top1")):
            ax.plot(
                frame["epoch"],
                frame[column] * 100.0,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.25,
                alpha=0.95,
                label=SHORT[method],
            )
    axes[0].set_ylabel("Train Top-1 (%)")
    axes[1].set_ylabel("Test Top-1 (%)")
    for label, ax in zip(("a", "b"), axes):
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 28)
        ax.set_ylim(0, 90)
        ax.grid(color="#E6E6E6", linewidth=0.5)
        panel_label(ax, label)
    axes[1].legend(loc="lower right", ncol=1)
    export(fig, output / "fig02_learning_dynamics")


def parameter_and_phase_audit(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(mm(FIG_WIDTH_MM), mm(68)), constrained_layout=True)
    final = data["final_epoch_test_top1"].to_numpy(float) * 100.0
    phase_millions = data["phase_parameters"].to_numpy(float) / 1.0e6
    phase_motion = data["phase_delta_rms_rad"].to_numpy(float) * 1000.0
    offsets = {
        "d2nn_continuous": (4, -9),
        "d2nn_oeo_sigmoid": (4, 5),
        "moe_continuous_fixed_router": (4, 4),
        "moe_oeo_dynamic_router": (4, -9),
        "moe_oeo_fixed_router": (-55, 5),
    }
    for index, method in enumerate(METHODS):
        marker = "o" if method.startswith("d2nn") else "D"
        axes[0].scatter(
            phase_millions[index], final[index], s=42, marker=marker,
            color=COLORS[method], edgecolor="white", linewidth=0.5, zorder=3,
        )
        axes[1].scatter(
            phase_motion[index], final[index], s=42, marker=marker,
            color=COLORS[method], edgecolor="white", linewidth=0.5, zorder=3,
        )
        for ax, x_value in ((axes[0], phase_millions[index]), (axes[1], phase_motion[index])):
            dx, dy = offsets[method]
            ax.annotate(
                SHORT[method], (x_value, final[index]), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.2,
                ha="left" if dx >= 0 else "right", va="center",
            )
    axes[0].set_xlabel("Trainable phase parameters (million)")
    axes[0].set_ylabel("Final test Top-1 (%)")
    axes[1].set_xlabel("Phase displacement from initialization (mrad RMS)")
    axes[1].set_ylabel("Final test Top-1 (%)")
    for label, ax in zip(("a", "b"), axes):
        ax.set_ylim(20, 85)
        ax.grid(color="#E6E6E6", linewidth=0.5)
        panel_label(ax, label)
    export(fig, output / "fig03_parameter_and_phase_audit")


def _router_matrix(frame: pd.DataFrame, variant: str, modality: str) -> np.ndarray:
    values = frame[(frame["variant"] == variant) & (frame["modality"] == modality)]
    pivot = values.pivot(index="router_stage", columns="expert", values="selection_rate")
    return pivot.sort_index().sort_index(axis=1).to_numpy(float) * 100.0


def router_heatmaps(router: pd.DataFrame, output: Path) -> None:
    panels = (
        ("moe_continuous_fixed_router", "vision", "Continuous MoE · Vision"),
        ("moe_continuous_fixed_router", "language", "Continuous MoE · Language"),
        ("moe_oeo_dynamic_router", "vision", "Rerouted OEO MoE · Vision"),
        ("moe_oeo_dynamic_router", "language", "Rerouted OEO MoE · Language"),
        ("moe_oeo_fixed_router", "vision", "Fixed-route OEO MoE · Vision"),
        ("moe_oeo_fixed_router", "language", "Fixed-route OEO MoE · Language"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(mm(FIG_WIDTH_MM), mm(88)), constrained_layout=True)
    image = None
    for label, ax, (variant, modality, title) in zip("abcdef", axes.flat, panels):
        matrix = _router_matrix(router, variant, modality)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_title(title, pad=3)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Router stage")
        ax.set_xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1))
        ax.set_yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if matrix[row, col] >= 55 else "#222222"
                ax.text(col, row, f"{matrix[row, col]:.0f}", ha="center", va="center", fontsize=6, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, label)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Selection rate (%)")
    export(fig, output / "fig04_router_selection_heatmaps")


def power_flow(power: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(mm(FIG_WIDTH_MM), mm(66)), constrained_layout=True)
    for ax, modality, label in zip(axes, ("vision", "language"), ("a", "b")):
        for method in METHODS:
            frame = power[(power["variant"] == method) & (power["modality"] == modality)].sort_values("optical_stage")
            if frame.empty:
                raise RuntimeError(f"Missing power diagnostics for {method}/{modality}")
            reference = float(frame.iloc[0]["mean_input_power"])
            relative = frame["mean_output_or_reload_power"].to_numpy(float) / reference * 100.0
            ax.plot(
                frame["optical_stage"], relative,
                color=COLORS[method], linestyle=LINESTYLES[method], linewidth=1.25,
                marker="o", markersize=2.8, label=SHORT[method],
            )
        ax.set_title(modality.capitalize())
        ax.set_xlabel("Optical stage")
        ax.set_ylabel("Output or reload power (% of stage-1 input)")
        ax.set_xticks(np.arange(1, 6))
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0, 105)
        ax.grid(color="#E6E6E6", linewidth=0.5)
        panel_label(ax, label)
    axes[1].legend(loc="lower right")
    export(fig, output / "fig05_normalized_power_flow")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render publication-style multiplane comparison figures")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "comparison",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.comparison_dir.expanduser().resolve()
    output = (args.output_dir or (root / "figures")).expanduser().resolve()
    apply_style()
    data, router, power, logs = load_inputs(root)
    performance_summary(data, output)
    learning_dynamics(logs, output)
    parameter_and_phase_audit(data, output)
    router_heatmaps(router, output)
    power_flow(power, output)
    print(f"Exported five figure bundles to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
