"""Create paper-ready data, geometry, and Temporal-9 result figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache_qwen_front import decode_frames, frame_fractions
from .data import read_manifest
from .settings import Geometry, load_settings


CM = 1.0 / 2.54


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save(fig: Any, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")


def _geometry_variants() -> list[tuple[str, Geometry]]:
    return [
        (
            "4 frames / 2×2\nexpert 109, gap 14",
            Geometry(
                lane_grid=2,
                lane_size=232,
                lane_pitch=246,
                lane_offset=0,
                parallel_expert_size=109,
                parallel_expert_pitch=123,
            ),
        ),
        (
            "9 frames / 3×3\nexpert 77, gap 2",
            Geometry(
                lane_grid=3,
                lane_size=156,
                lane_pitch=160,
                lane_offset=1,
                parallel_expert_size=77,
                parallel_expert_pitch=79,
            ),
        ),
        (
            "16 frames / 4×4\nexpert 54, gap 6",
            Geometry(),
        ),
    ]


def plot_geometry(output: Path) -> None:
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(17.8 * CM, 5.7 * CM))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for axis, (title, geometry) in zip(axes, _geometry_variants()):
        geometry.validate(formal=True)
        axis.add_patch(
            Rectangle((0, 0), 478, 478, facecolor="#F4F4F4", edgecolor="black", lw=0.8)
        )
        for frame_index, (top, left) in enumerate(geometry.lane_origins):
            axis.add_patch(
                Rectangle(
                    (left, top),
                    geometry.lane_size,
                    geometry.lane_size,
                    fill=False,
                    edgecolor="#888888",
                    lw=0.35,
                )
            )
            for expert_index, (dy, dx) in enumerate(geometry.parallel_expert_origins):
                axis.add_patch(
                    Rectangle(
                        (left + dx, top + dy),
                        geometry.parallel_expert_size,
                        geometry.parallel_expert_size,
                        facecolor=colors[expert_index],
                        edgecolor="white",
                        lw=0.2,
                        alpha=0.72,
                    )
                )
            axis.text(
                left + geometry.lane_size / 2,
                top + geometry.lane_size / 2,
                str(frame_index + 1),
                ha="center",
                va="center",
                color="white",
                fontsize=5.2,
                weight="bold",
            )
        axis.set(xlim=(0, 478), ylim=(478, 0), aspect="equal", title=title)
        axis.set_xticks((0, 239, 478))
        axis.set_yticks((0, 239, 478))
        axis.set_xlabel("logical x (17 µm pixels)")
    axes[0].set_ylabel("logical y (17 µm pixels)")
    for label, axis in zip("abc", axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=8)
    fig.suptitle("Fixed 478×478 active field: frame packing and four experts per lane", y=1.02)
    fig.tight_layout(pad=0.6, w_pad=0.8)
    _save(fig, output, "temporal_frame_packing_geometry")
    plt.close(fig)


def plot_dataset(settings_path: Path, output: Path) -> dict[str, Any]:
    plt = _pyplot()
    settings = load_settings(settings_path)
    if settings.manifest_path is None:
        raise RuntimeError("Config has no manifest")
    rows = read_manifest(settings.manifest_path)
    train = np.asarray([row.temporal for row in rows if row.split == "train"])
    test = np.asarray([row.temporal for row in rows if row.split == "test"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0 * CM, 5.0 * CM))
    bins = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), 22)
    axes[0].hist(train, bins=bins, density=True, alpha=0.62, color="#0072B2", label=f"train (n={len(train)})")
    axes[0].hist(test, bins=bins, density=True, alpha=0.62, color="#E69F00", label=f"test (n={len(test)})")
    axes[0].set(xlabel="Temporal MOS", ylabel="Density", title="Target distribution")
    axes[0].legend(frameon=False)
    styles = {4: ("#999999", "2×2"), 9: ("#D55E00", "3×3 proposed"), 16: ("#0072B2", "4×4")}
    for count in (4, 9, 16):
        color, label = styles[count]
        fractions = np.asarray(frame_fractions(count))
        axes[1].scatter(fractions, np.full(count, count), s=10, color=color, label=label, zorder=3)
        axes[1].plot(fractions, np.full(count, count), color=color, alpha=0.55)
    axes[1].set(
        xlim=(0.06, 0.94),
        yticks=(4, 9, 16),
        xlabel="Normalized video time",
        ylabel="Sampled frames",
        title="Central 10%–90% sampling",
    )
    axes[1].legend(frameon=False, loc="lower right")
    for label, axis in zip("ab", axes):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.7, w_pad=1.2)
    _save(fig, output, "temporal_dataset_and_sampling")
    plt.close(fig)
    summary = {
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_mean": float(train.mean()),
        "train_std": float(train.std()),
        "test_mean": float(test.mean()),
        "test_std": float(test.std()),
        "temporal9_sampling_fractions": list(frame_fractions(9)),
    }
    (output / "temporal_dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def plot_contact_sheet(settings_path: Path, output: Path) -> None:
    plt = _pyplot()
    settings = load_settings(settings_path)
    assert settings.manifest_path is not None
    rows = sorted(
        (row for row in read_manifest(settings.manifest_path) if row.split == "test"),
        key=lambda row: row.temporal,
    )
    chosen = [rows[round((len(rows) - 1) * q)] for q in (0.10, 0.50, 0.90)]
    fig, axes = plt.subplots(3, 9, figsize=(17.8 * CM, 6.0 * CM))
    for row_index, row in enumerate(chosen):
        frames = decode_frames(Path(row.video_path), 9)
        for frame_index, (axis, frame) in enumerate(zip(axes[row_index], frames)):
            axis.imshow(frame)
            axis.set_axis_off()
            if row_index == 0:
                axis.set_title(f"t={frame_fractions(9)[frame_index]:.1f}", fontsize=5.5, pad=1)
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"MOS {row.temporal:.1f}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=6,
        )
    fig.suptitle("Representative LGVQ videos: exact nine-frame input to the 3×3 optical layout", y=0.995)
    fig.subplots_adjust(left=0.045, right=0.997, top=0.88, bottom=0.01, wspace=0.025, hspace=0.05)
    _save(fig, output, "temporal9_representative_frame_sequences")
    plt.close(fig)


def plot_phase_planes(
    settings_path: Path, snapshot_path: Path | None, output: Path
) -> None:
    if snapshot_path is None or not snapshot_path.is_file():
        return
    import torch

    plt = _pyplot()
    settings = load_settings(settings_path)
    geometry = settings.geometry
    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    planes = payload["planes"]

    def phase(name: str) -> np.ndarray:
        return planes[name]["phase_rad"].detach().float().numpy()

    def empty() -> np.ndarray:
        return np.full((geometry.active_size, geometry.active_size), np.nan)

    parallel_experts = empty()
    values = phase("parallel_optics.raw_expert_phase")
    for lane_index, (lane_y, lane_x) in enumerate(geometry.lane_origins):
        for expert_index, (dy, dx) in enumerate(geometry.parallel_expert_origins):
            size = geometry.parallel_expert_size
            parallel_experts[
                lane_y + dy : lane_y + dy + size,
                lane_x + dx : lane_x + dx + size,
            ] = values[lane_index * 4 + expert_index]

    parallel_router = empty()
    values = phase("parallel_router.raw_router_phase")
    offset = (geometry.lane_size - geometry.parallel_expert_size) // 2
    for lane_index, (lane_y, lane_x) in enumerate(geometry.lane_origins):
        size = geometry.parallel_expert_size
        parallel_router[
            lane_y + offset : lane_y + offset + size,
            lane_x + offset : lane_x + offset + size,
        ] = values[lane_index]

    serial_experts = empty()
    values = phase("serial_optics.raw_expert_phase")
    for expert_index, (top, left) in enumerate(geometry.serial_expert_origins):
        size = geometry.serial_expert_size
        serial_experts[top : top + size, left : left + size] = values[expert_index]

    serial_router = empty()
    values = phase("serial_router.raw_router_phase")
    offset = (geometry.active_size - geometry.serial_expert_size) // 2
    size = geometry.serial_expert_size
    serial_router[offset : offset + size, offset : offset + size] = values

    arrays = (
        parallel_experts,
        phase("parallel_optics.raw_global_phase"),
        parallel_router,
        serial_experts,
        phase("serial_optics.raw_global_phase"),
        serial_router,
    )
    titles = (
        "Vision experts\n9 frames x 4",
        "Vision global",
        "Vision router\n9 optical masks",
        "Language experts\n4 masks",
        "Language global",
        "Language router",
    )
    fig, axes_grid = plt.subplots(2, 3, figsize=(13.0 * CM, 6.0 * CM))
    axes = axes_grid.ravel()
    cmap = plt.get_cmap("twilight").copy()
    cmap.set_bad("white")
    image = None
    for label, axis, array, title in zip("abcdef", axes, arrays, titles):
        image = axis.imshow(array, cmap=cmap, vmin=0.0, vmax=2.0 * np.pi)
        axis.set_title(title, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(-0.08, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=8)
    assert image is not None
    colorbar_axis = fig.add_axes((0.915, 0.16, 0.018, 0.63))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks((0.0, np.pi, 2.0 * np.pi))
    colorbar.set_ticklabels(("0", r"$\pi$", r"$2\pi$"))
    colorbar.set_label("Phase (rad)")
    fig.suptitle(
        f"Temporal-9 learned phase planes at selected epoch {payload['epoch']}",
        y=1.02,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.88,
        top=0.82,
        bottom=0.02,
        wspace=0.16,
        hspace=0.38,
    )
    _save(fig, output, "temporal9_selected_phase_planes")
    plt.close(fig)


def _load_report(path: Path | None) -> dict[str, Any] | None:
    return None if path is None or not path.is_file() else json.loads(path.read_text(encoding="utf-8"))


def plot_results(
    baseline_path: Path | None,
    accuracy_path: Path | None,
    candidate_path: Path | None,
    history_path: Path | None,
    output: Path,
) -> None:
    baseline = _load_report(baseline_path)
    accuracy = _load_report(accuracy_path)
    candidate = _load_report(candidate_path)
    if candidate is None:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(17.8 * CM, 5.2 * CM))
    reports = [] if baseline is None else [("16f optical", baseline["normal_optical_electronic"])]
    if accuracy is not None:
        reports.append(("9f max accuracy", accuracy["normal_optical_electronic"]))
    reports.extend(
        [
            ("9f balanced", candidate["normal_optical_electronic"]),
            ("9f balanced\noptics off", candidate["same_checkpoint_optics_bypassed"]),
        ]
    )
    x = np.arange(len(reports))
    width = 0.34
    for offset, metric, color in ((-width / 2, "srcc", "#0072B2"), (width / 2, "plcc", "#E69F00")):
        axes[0].bar(x + offset, [r[1][metric] for r in reports], width, label=metric.upper(), color=color)
    axes[0].set_xticks(x, [r[0] for r in reports], rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Correlation")
    axes[0].legend(frameon=False)
    axes[0].set_title("Accuracy and optics-off control")

    routing = candidate["normal_optical_electronic"].get("router_diagnostics", {})
    route_names = [name for name in ("vision", "language") if name in routing]
    for index, name in enumerate(route_names):
        shares = routing[name]["selected_share"]
        axes[1].plot(np.arange(1, 5), shares, marker="o", label=name)
    axes[1].axhline(0.25, color="#777777", ls="--", lw=0.8, label="uniform reference")
    axes[1].set(xticks=(1, 2, 3, 4), ylim=(0, 0.55), xlabel="Expert", ylabel="Selected share", title="Hard Top-2 usage")
    axes[1].legend(frameon=False)

    history = _load_report(history_path)
    if isinstance(history, list):
        evaluated = [row for row in history if row.get("test_evaluated")]
        axes[2].plot(
            [row["epoch"] for row in evaluated],
            [row["test_optical_on"]["srcc"] for row in evaluated],
            marker="o",
            ms=2.5,
            color="#009E73",
        )
    axes[2].set(xlabel="Epoch", ylabel="Test SRCC", title="Periodic test selection")
    for label, axis in zip("abc", axes):
        axis.text(-0.15, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.7, w_pad=1.0)
    _save(fig, output, "temporal9_metrics_router_training")
    plt.close(fig)

    with (output / "temporal9_metrics_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "srcc", "krcc", "plcc", "rmse", "mae"))
        for name, metrics in reports:
            writer.writerow((name, *(metrics[key] for key in ("srcc", "krcc", "plcc", "rmse", "mae"))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--accuracy-report", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--phase-snapshot", type=Path)
    parser.add_argument("--skip-contact-sheet", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    plot_geometry(output)
    plot_dataset(args.config, output)
    if not args.skip_contact_sheet:
        plot_contact_sheet(args.config, output)
    plot_phase_planes(args.config, args.phase_snapshot, output)
    plot_results(
        args.baseline_report,
        args.accuracy_report,
        args.candidate_report,
        args.history,
        output,
    )
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
