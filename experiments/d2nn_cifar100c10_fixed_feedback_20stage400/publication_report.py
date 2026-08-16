from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .settings import Settings


METHODS = ("bp", "fa_pretrained", "fa_random")
DISPLAY = {
    "bp": "BP",
    "fa_pretrained": "FA-pretrained",
    "fa_random": "FA-random",
    "no_finetune": "No fine-tuning",
}
COLORS = {
    "bp": "#3B6FB6",
    "fa_pretrained": "#D9902F",
    "fa_random": "#8A6AA6",
    "no_finetune": "#555555",
}
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 76.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _sample_summary(values: Iterable[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    return mean(finite), stdev(finite) if len(finite) > 1 else 0.0


def _flatten_delta(parameters: dict[str, Any], initial: dict[str, Any]) -> Any:
    import torch

    return torch.cat(
        [
            (parameters[name].detach().float().cpu() - initial[name].detach().float().cpu()).reshape(-1)
            for name in sorted(initial)
        ]
    )


def _checkpoint_path(run_dir: Path, epoch: int, final_epoch: int) -> Path:
    return run_dir / "checkpoints" / ("last.pt" if epoch == final_epoch else f"epoch_{epoch:03d}.pt")


def _collect_source_data(settings: Settings, results_root: Path) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    run_root = settings.output_dir
    source_dir = results_root / "source_data"
    initial_path = run_root / "pretrain" / "checkpoints" / "pretrained_best_validation.pt"
    initial_checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
    initial_parameters = initial_checkpoint["parameters"]
    initial_flat = torch.cat(
        [initial_parameters[name].detach().float().cpu().reshape(-1) for name in sorted(initial_parameters)]
    )

    trajectory_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in settings.training.finetune_seeds:
            run_dir = run_root / "finetune" / method / f"seed_{seed}"
            history = _read_csv(run_dir / "training_history.csv")
            for row in history:
                trajectory_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "epoch": int(row["epoch"]),
                        "train_accuracy": float(row["train_accuracy"]),
                        "validation_accuracy": float(row["validation_accuracy"]),
                        "test_accuracy": float(row["test_accuracy"]),
                        "relative_parameter_drift": float(row["relative_parameter_drift"]),
                        "phase_circular_rms_rad": float(row["phase_circular_rms_rad"]),
                        "phase_operator_coherence": float(row["phase_operator_coherence"]),
                    }
                )
            last = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
            best = torch.load(run_dir / "checkpoints" / "best_validation.pt", map_location="cpu", weights_only=False)
            for policy, checkpoint in (("fixed_epoch", last), ("validation_selected", best)):
                metrics = checkpoint["metrics"]
                item = {
                    "policy": policy,
                    "method": method,
                    "seed": seed,
                    "selected_epoch": int(checkpoint["epoch"]),
                    "train_accuracy": float(metrics["train_accuracy"]),
                    "validation_accuracy": float(metrics["validation_accuracy"]),
                    "test_accuracy": float(metrics["test_accuracy"]),
                    "generalization_gap_train_minus_test": float(metrics["train_accuracy"])
                    - float(metrics["test_accuracy"]),
                }
                checkpoint_rows.append(item)
                (fixed_rows if policy == "fixed_epoch" else best_rows).append(item)

    noft = json.loads((run_root / "finetune" / "no_finetune" / "summary.json").read_text(encoding="utf-8"))
    checkpoint_rows.append(
        {
            "policy": "fixed_model",
            "method": "no_finetune",
            "seed": "",
            "selected_epoch": 0,
            "train_accuracy": "",
            "validation_accuracy": "",
            "test_accuracy": float(noft["test_accuracy"]),
            "generalization_gap_train_minus_test": "",
        }
    )

    matched_epochs = sorted({10, settings.training.finetune_epochs})
    geometry_rows: list[dict[str, Any]] = []
    for epoch in matched_epochs:
        for seed in settings.training.finetune_seeds:
            deltas: dict[str, torch.Tensor] = {}
            checkpoints: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                run_dir = run_root / "finetune" / method / f"seed_{seed}"
                checkpoint = torch.load(
                    _checkpoint_path(run_dir, epoch, settings.training.finetune_epochs),
                    map_location="cpu",
                    weights_only=False,
                )
                checkpoints[method] = checkpoint
                deltas[method] = _flatten_delta(checkpoint["parameters"], initial_parameters)
            bp_norm = deltas["bp"].norm().clamp_min(1e-12)
            for method in METHODS:
                delta = deltas[method]
                cosine = float(F.cosine_similarity(delta, deltas["bp"], dim=0, eps=1e-12).clamp(-1.0, 1.0))
                metrics = checkpoints[method]["metrics"]
                geometry_rows.append(
                    {
                        "matched_epoch": epoch,
                        "method": method,
                        "seed": seed,
                        "relative_parameter_drift": float(delta.norm() / initial_flat.norm().clamp_min(1e-12)),
                        "drift_ratio_to_matched_bp": float(delta.norm() / bp_norm),
                        "endpoint_cosine_to_matched_bp": cosine,
                        "phase_circular_rms_rad": float(metrics["phase_circular_rms_rad"]),
                        "phase_operator_coherence": float(metrics["phase_operator_coherence"]),
                    }
                )

    _write_csv(source_dir / "training_trajectories.csv", trajectory_rows)
    _write_csv(source_dir / "checkpoint_performance.csv", checkpoint_rows)
    _write_csv(source_dir / "endpoint_geometry.csv", geometry_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for policy, rows in (("fixed_epoch", fixed_rows), ("validation_selected", best_rows)):
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            test_mean, test_sd = _sample_summary(row["test_accuracy"] for row in selected)
            epoch_mean, epoch_sd = _sample_summary(row["selected_epoch"] for row in selected)
            gap_mean, gap_sd = _sample_summary(row["generalization_gap_train_minus_test"] for row in selected)
            aggregate_rows.append(
                {
                    "policy": policy,
                    "method": method,
                    "n_seeds": len(selected),
                    "test_accuracy_mean": test_mean,
                    "test_accuracy_sample_sd": test_sd,
                    "selected_epoch_mean": epoch_mean,
                    "selected_epoch_sample_sd": epoch_sd,
                    "train_minus_test_gap_mean": gap_mean,
                    "train_minus_test_gap_sample_sd": gap_sd,
                }
            )
    aggregate_rows.append(
        {
            "policy": "fixed_model",
            "method": "no_finetune",
            "n_seeds": 1,
            "test_accuracy_mean": float(noft["test_accuracy"]),
            "test_accuracy_sample_sd": "",
            "selected_epoch_mean": 0,
            "selected_epoch_sample_sd": "",
            "train_minus_test_gap_mean": "",
            "train_minus_test_gap_sample_sd": "",
        }
    )
    _write_csv(source_dir / "aggregate_performance.csv", aggregate_rows)

    geometry_aggregate: list[dict[str, Any]] = []
    for epoch in matched_epochs:
        for method in METHODS:
            rows = [row for row in geometry_rows if row["matched_epoch"] == epoch and row["method"] == method]
            drift_mean, drift_sd = _sample_summary(row["relative_parameter_drift"] for row in rows)
            ratio_mean, ratio_sd = _sample_summary(row["drift_ratio_to_matched_bp"] for row in rows)
            cosine_mean, cosine_sd = _sample_summary(row["endpoint_cosine_to_matched_bp"] for row in rows)
            geometry_aggregate.append(
                {
                    "matched_epoch": epoch,
                    "method": method,
                    "n_seeds": len(rows),
                    "relative_parameter_drift_mean": drift_mean,
                    "relative_parameter_drift_sample_sd": drift_sd,
                    "drift_ratio_to_matched_bp_mean": ratio_mean,
                    "drift_ratio_to_matched_bp_sample_sd": ratio_sd,
                    "endpoint_cosine_to_matched_bp_mean": cosine_mean,
                    "endpoint_cosine_to_matched_bp_sample_sd": cosine_sd,
                }
            )
    _write_csv(source_dir / "aggregate_geometry.csv", geometry_aggregate)
    return {
        "noft": noft,
        "aggregate_performance": aggregate_rows,
        "aggregate_geometry": geometry_aggregate,
        "matched_epochs": matched_epochs,
        "pretrained_epoch": int(initial_checkpoint.get("epoch", -1)),
        "pretrained_validation_accuracy": float(initial_checkpoint["metrics"]["validation_accuracy"]),
    }


def _load_numeric_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        converted: dict[str, Any] = dict(row)
        for key, value in row.items():
            if key == "method" or value == "":
                continue
            try:
                converted[key] = int(value) if key in {"seed", "epoch", "matched_epoch", "selected_epoch"} else float(value)
            except ValueError:
                pass
        rows.append(converted)
    return rows


def _configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_publication_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300)


def _group_by_method_epoch(rows: list[dict[str, Any]], field: str) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["method"])][int(row["epoch"])].append(float(row[field]))
    return grouped


def _plot_task_performance(results_root: Path) -> None:
    import numpy as np

    plt = _configure_matplotlib()
    trajectories = _load_numeric_rows(results_root / "source_data" / "training_trajectories.csv")
    checkpoints = _load_numeric_rows(results_root / "source_data" / "checkpoint_performance.csv")
    noft = next(row for row in checkpoints if row["method"] == "no_finetune")
    grouped = _group_by_method_epoch(trajectories, "test_accuracy")
    width_in = FIGURE_WIDTH_MM / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_in, FIGURE_HEIGHT_MM / 25.4), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    for method in METHODS:
        epochs = sorted(grouped[method])
        matrix = np.asarray([grouped[method][epoch] for epoch in epochs], dtype=float)
        center = matrix.mean(axis=1)
        spread = matrix.std(axis=1, ddof=1)
        for seed in sorted({int(row["seed"]) for row in trajectories if row["method"] == method}):
            seed_rows = sorted(
                (row for row in trajectories if row["method"] == method and int(row["seed"]) == seed),
                key=lambda row: int(row["epoch"]),
            )
            ax.plot(
                [row["epoch"] for row in seed_rows],
                [row["test_accuracy"] for row in seed_rows],
                color=COLORS[method],
                alpha=0.18,
                linewidth=0.65,
            )
        ax.fill_between(epochs, center - spread, center + spread, color=COLORS[method], alpha=0.13, linewidth=0)
        ax.plot(epochs, center, color=COLORS[method], label=DISPLAY[method])
    ax.axhline(float(noft["test_accuracy"]), color=COLORS["no_finetune"], linestyle="--", linewidth=1.0)
    ax.text(49.2, float(noft["test_accuracy"]) + 0.004, "No fine-tuning", ha="right", va="bottom", color=COLORS["no_finetune"])
    ax.set(xlabel="Fine-tuning epoch", ylabel="Test accuracy", xlim=(1, 50), ylim=(0.30, 0.47))
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(loc="lower left", ncol=1)

    ax = axes[1]
    order = (*METHODS, "no_finetune")
    for x, method in enumerate(order):
        if method == "no_finetune":
            values = [float(noft["test_accuracy"])]
            ax.scatter([x], values, marker="D", s=26, color=COLORS[method], edgecolor="white", linewidth=0.5, zorder=4)
        else:
            values = [
                float(row["test_accuracy"])
                for row in checkpoints
                if row["method"] == method and row["policy"] == "validation_selected"
            ]
            offsets = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(x + offsets, values, s=19, color=COLORS[method], alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
            ax.hlines(mean(values), x - 0.19, x + 0.19, color="#202020", linewidth=1.0, zorder=4)
    ax.set_xticks(range(len(order)), [DISPLAY[method].replace("-", "-\n", 1) for method in order])
    ax.set(ylabel="Test accuracy at validation-selected checkpoint", ylim=(0.36, 0.45))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[1].text(-0.17, 1.04, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.95, wspace=0.32)
    _save_publication_figure(fig, results_root / "figures" / "task_performance")
    plt.close(fig)


def _plot_endpoint_geometry(results_root: Path) -> None:
    import numpy as np

    plt = _configure_matplotlib()
    trajectories = _load_numeric_rows(results_root / "source_data" / "training_trajectories.csv")
    geometry = _load_numeric_rows(results_root / "source_data" / "endpoint_geometry.csv")
    grouped = _group_by_method_epoch(trajectories, "relative_parameter_drift")
    matched_epochs = sorted({int(row["matched_epoch"]) for row in geometry})
    width_in = FIGURE_WIDTH_MM / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_in, FIGURE_HEIGHT_MM / 25.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    for method in METHODS:
        epochs = sorted(grouped[method])
        matrix = np.asarray([grouped[method][epoch] for epoch in epochs], dtype=float)
        center = matrix.mean(axis=1)
        spread = matrix.std(axis=1, ddof=1)
        ax.fill_between(epochs, center - spread, center + spread, color=COLORS[method], alpha=0.13, linewidth=0)
        ax.plot(epochs, center, color=COLORS[method], label=DISPLAY[method])
    ax.set(xlabel="Fine-tuning epoch", ylabel="Relative parameter drift from pretraining", xlim=(1, 50), ylim=(0, None))
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(loc="upper left")

    ax = axes[1]
    markers = {matched_epochs[0]: "o", matched_epochs[-1]: "s"}
    for method in METHODS:
        means: list[tuple[float, float]] = []
        for epoch in matched_epochs:
            rows = [row for row in geometry if row["method"] == method and int(row["matched_epoch"]) == epoch]
            xs = [float(row["drift_ratio_to_matched_bp"]) for row in rows]
            ys = [float(row["endpoint_cosine_to_matched_bp"]) for row in rows]
            ax.scatter(xs, ys, s=14 if epoch == matched_epochs[0] else 22, marker=markers[epoch], color=COLORS[method], alpha=0.34, linewidth=0)
            means.append((mean(xs), mean(ys)))
            ax.scatter(
                [mean(xs)],
                [mean(ys)],
                s=27 if epoch == matched_epochs[0] else 37,
                marker=markers[epoch],
                facecolor=COLORS[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
        if len(means) == 2:
            ax.annotate("", xy=means[1], xytext=means[0], arrowprops={"arrowstyle": "->", "color": COLORS[method], "lw": 0.9})
    ax.axvline(1.0, color="#BDBDBD", linestyle=":", linewidth=0.7)
    ax.axhline(1.0, color="#BDBDBD", linestyle=":", linewidth=0.7)
    ax.set(
        xlabel="Drift magnitude / matched BP drift",
        ylabel="Cosine similarity to matched BP update",
        xlim=(0.93, 1.36),
        ylim=(0.30, 1.025),
    )
    ax.grid(color="#E5E5E5", linewidth=0.45)
    legend_handles = [
        plt.Line2D([], [], marker=markers[epoch], linestyle="none", color="#555555", markersize=4.2, label=f"Epoch {epoch}")
        for epoch in matched_epochs
    ]
    ax.legend(handles=legend_handles, loc="lower left")
    axes[0].text(-0.15, 1.04, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[1].text(-0.18, 1.04, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.95, wspace=0.34)
    _save_publication_figure(fig, results_root / "figures" / "endpoint_geometry")
    plt.close(fig)


def render_from_source(results_root: Path, legacy_comparison_dir: Path | None = None) -> None:
    _plot_task_performance(results_root)
    _plot_endpoint_geometry(results_root)
    if legacy_comparison_dir is not None:
        legacy_comparison_dir.mkdir(parents=True, exist_ok=True)
        for name in ("task_performance", "endpoint_geometry"):
            for suffix in (".png", ".pdf", ".svg", ".tiff"):
                shutil.copy2(results_root / "figures" / f"{name}{suffix}", legacy_comparison_dir / f"{name}{suffix}")
        shutil.copy2(results_root / "figures" / "task_performance.png", legacy_comparison_dir / "method_comparison.png")


def _find(rows: list[dict[str, Any]], policy: str, method: str) -> dict[str, Any]:
    return next(row for row in rows if row["policy"] == policy and row["method"] == method)


def _format_pm(row: dict[str, Any], key: str, sd_key: str, scale: float = 100.0) -> str:
    return f"{scale * float(row[key]):.2f} ± {scale * float(row[sd_key]):.2f}"


def _write_report(settings: Settings, results_root: Path, summary: dict[str, Any]) -> None:
    perf = summary["aggregate_performance"]
    geometry = summary["aggregate_geometry"]
    noft = _find(perf, "fixed_model", "no_finetune")
    last_bp = _find(perf, "fixed_epoch", "bp")
    best_bp = _find(perf, "validation_selected", "bp")
    final_epoch = settings.training.finetune_epochs
    final_geometry = [row for row in geometry if int(row["matched_epoch"]) == final_epoch]
    geo_bp = next(row for row in final_geometry if row["method"] == "bp")
    geo_pre = next(row for row in final_geometry if row["method"] == "fa_pretrained")
    geo_rand = next(row for row in final_geometry if row["method"] == "fa_random")
    lines = [
        "# Fixed-feedback optical fine-tuning: main-result record",
        "",
        "## Executive conclusion",
        "",
        (
            f"At the prespecified epoch-{final_epoch} endpoint, no fine-tuning reached "
            f"{100 * float(noft['test_accuracy_mean']):.2f}% test accuracy, above BP "
            f"({_format_pm(last_bp, 'test_accuracy_mean', 'test_accuracy_sample_sd')}%). "
            "This is not evidence that fine-tuning is intrinsically harmful: the downstream set is a "
            "10-class CIFAR-100-C subset whose classes and 100-way output coordinates were already learned during "
            "CIFAR-100 pretraining, while the 50-epoch full-parameter adaptation strongly overfits the small corrupted-domain split."
        ),
        "",
        (
            "The geometric result is nevertheless internally consistent with the fixed-feedback hypothesis: "
            f"FA-pretrained ends with {float(geo_pre['drift_ratio_to_matched_bp_mean']):.3f}× the BP drift and "
            f"cosine {float(geo_pre['endpoint_cosine_to_matched_bp_mean']):.3f} to the matched BP update, whereas "
            f"FA-random moves {float(geo_rand['drift_ratio_to_matched_bp_mean']):.3f}× as far with cosine "
            f"{float(geo_rand['endpoint_cosine_to_matched_bp_mean']):.3f}. The separation is qualitatively similar "
            "to the reference paper, but substantially weaker for the random-feedback baseline."
        ),
        "",
        "## What each metric means",
        "",
        "- `relative_parameter_drift = ||θ_T − θ_pre||₂ / ||θ_pre||₂`. It is measured from the shared pretrained checkpoint.",
        "- `endpoint_cosine_to_matched_bp` is the cosine between `θ_method,T − θ_pre` and the same-seed BP update `θ_BP,T − θ_pre` at the same epoch.",
        "- BP is therefore exactly 1 up to floating-point precision. No fine-tuning has a zero update, so its cosine is undefined (N/A), not 0.",
        "- The task-performance checkpoint policy and the geometric matched-endpoint policy answer different questions and must not be mixed.",
        "",
        "## Checkpoint policy and fairness",
        "",
        f"The original comparison used `last.pt` at epoch {final_epoch} for BP/FA and the fixed pretrained-best model for No Fine-Tuning. "
        "This is a valid fixed-budget endpoint comparison because all trained methods receive the same budget, but it is not an “each method's best result” comparison.",
        "",
        "For reporting, use both of the following without using test data for model selection:",
        "",
        f"1. **Primary fixed-budget endpoint:** epoch {final_epoch} for BP, FA-pretrained and FA-random; fixed pretrained model for NoFT.",
        "2. **Secondary validation-selected result:** select the epoch with highest validation accuracy independently per seed/method, then report that checkpoint's test accuracy. NoFT remains fixed.",
        "",
        "Do not compare geometry at independently selected best epochs. The endpoint vectors would correspond to different training times. The geometric analysis uses matched epoch 10 and epoch 50 checkpoints.",
        "",
        "The reference paper itself uses mixed rules: GSM8K reports the best scheduled test checkpoint; FA on SAMSum uses validation selection; BP on SAMSum reports the best scheduled test checkpoint. It explicitly labels the test-selected values as best-observed scheduled checkpoints rather than independent held-out estimates. Its geometric analysis instead uses matched epoch-3 checkpoints for every method and seed. Our validation-selected secondary table is statistically cleaner than selecting on test.",
        "",
        "## Task performance",
        "",
        "| Policy | Method | Test accuracy, mean ± sample SD | Mean selected epoch | Train−test gap |",
        "|---|---|---:|---:|---:|",
    ]
    for policy in ("fixed_epoch", "validation_selected"):
        for method in METHODS:
            row = _find(perf, policy, method)
            lines.append(
                f"| {policy.replace('_', ' ')} | {DISPLAY[method]} | "
                f"{_format_pm(row, 'test_accuracy_mean', 'test_accuracy_sample_sd')}% | "
                f"{float(row['selected_epoch_mean']):.1f} | "
                f"{100 * float(row['train_minus_test_gap_mean']):.2f} pp |"
            )
    lines.append(f"| fixed model | No fine-tuning | {100 * float(noft['test_accuracy_mean']):.2f}% | 0 | N/A |")
    lines.extend(
        [
            "",
            f"Even after validation selection, BP reaches {_format_pm(best_bp, 'test_accuracy_mean', 'test_accuracy_sample_sd')}%, "
            f"still slightly below NoFT ({100 * float(noft['test_accuracy_mean']):.2f}%). Thus the result is not only a last-epoch artifact, although the late-epoch overfitting makes the fixed-endpoint gap larger.",
            "",
            "## Matched-endpoint geometry",
            "",
            "| Epoch | Method | Relative drift | Drift / BP | Cosine to BP |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in geometry:
        lines.append(
            f"| {int(row['matched_epoch'])} | {DISPLAY[row['method']]} | "
            f"{float(row['relative_parameter_drift_mean']):.4f} ± {float(row['relative_parameter_drift_sample_sd']):.4f} | "
            f"{float(row['drift_ratio_to_matched_bp_mean']):.3f} ± {float(row['drift_ratio_to_matched_bp_sample_sd']):.3f} | "
            f"{float(row['endpoint_cosine_to_matched_bp_mean']):.3f} ± {float(row['endpoint_cosine_to_matched_bp_sample_sd']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation relative to the paper",
            "",
            "The trend agrees qualitatively with the paper: FA-pretrained stays close to BP in both update magnitude and direction, while random fixed feedback deviates more. However, this run does **not** reproduce the paper's limited-drift regime quantitatively:",
            "",
            f"- BP relative drift is {float(geo_bp['relative_parameter_drift_mean']):.3f}, far larger than the approximately 0.004 scale reported for language-model SFT.",
            f"- FA-random moves only {float(geo_rand['drift_ratio_to_matched_bp_mean']):.2f}× as far as BP rather than approximately 2×.",
            f"- FA-random cosine is {float(geo_rand['endpoint_cosine_to_matched_bp_mean']):.3f}, lower than FA-pretrained but not near orthogonal.",
            "",
            "Therefore the present experiment supports a preliminary geometric analogy, not a close quantitative replication.",
            "",
            "## Why No Fine-Tuning is higher",
            "",
            "1. **The downstream classes are not unseen.** They are selected CIFAR-100 classes under CIFAR-100-C corruptions, and the same 100-way head coordinates are retained.",
            "2. **The downstream set is small.** Fine-tuning uses 1,800 training images against millions of trainable optical phase values plus electronic parameters.",
            "3. **The phase learning rate and horizon are aggressive.** Validation-selected epochs occur early, while training accuracy keeps rising and test accuracy falls.",
            "4. **The experiment violates the intended small-drift premise.** A relative drift near 0.40 means the frozen pretrained feedback is being tested far from its initialization.",
            "5. **NoFT already solves much of the task.** Fine-tuning has little headroom and can easily overwrite robust pretrained features.",
            "",
            "## Recommended next experiment",
            "",
            "- Keep these completed runs unchanged as the fixed-budget baseline.",
            "- Add a genuinely distinct downstream task (actual CIFAR-10 with a newly initialized 10-way head, or a disjoint CIFAR-100 class split). This makes NoFT an honest transfer baseline.",
            "- Use a shorter 10–15 epoch adaptation window, validation selection, and a lower phase learning rate or warmup/cosine decay to target a much smaller drift regime.",
            "- Report drift during training and predefine a drift budget; do not select a checkpoint using test accuracy.",
            "- If the goal is to test feedback rather than readout mismatch, first train the new readout with the optical backbone frozen, then jointly fine-tune all methods from that same checkpoint.",
            "- For the current same-class corruption task, consider partial freezing or an explicit penalty to the pretrained optical operator; the objective should preserve robust features rather than relearn the labels.",
            "",
            "## Figures and source data",
            "",
            "- `figures/task_performance.*`: test trajectories and validation-selected test performance.",
            "- `figures/endpoint_geometry.*`: drift trajectories and matched-endpoint update geometry.",
            "- `source_data/training_trajectories.csv`: all epochs, methods and seeds.",
            "- `source_data/checkpoint_performance.csv`: fixed endpoint and validation-selected checkpoints.",
            "- `source_data/endpoint_geometry.csv`: matched epoch-10 and epoch-50 geometry.",
            "",
            "All reported dispersion values are sample standard deviations over three matched seeds. NoFT is one deterministic checkpoint and therefore has no seed SD.",
        ]
    )
    (results_root / "RESULTS_AND_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure_contract(results_root: Path) -> None:
    content = """# Figure contract

- Core conclusion: fixed pretrained feedback follows BP geometrically, but all fine-tuning methods overfit the current same-class corruption transfer and finish below the pretrained model.
- Evidence chain: task-performance trajectories establish overfitting; validation-selected seed points separate selection from endpoint effects; drift trajectories establish update magnitude; the drift-ratio/cosine plane establishes matched-direction geometry.
- Archetype: quantitative grid with one hero trajectory panel per figure.
- Backend: Python/matplotlib only.
- Export: 183 mm × 76 mm, Arial 7 pt, editable SVG/PDF, 600 dpi TIFF, 300 dpi PNG preview.
- Replicates: three matched random seeds; lines and bands show mean ± sample SD; raw seeds are retained where useful.
- Reviewer risks: NoFT is deterministic; downstream labels overlap pretraining; test is evaluated every epoch but is not used by the validation-selected policy; geometry must use matched epochs; NoFT cosine is undefined.
"""
    (results_root / "FIGURE_CONTRACT.md").write_text(content, encoding="utf-8")


def generate_publication_report(settings: Settings, legacy_comparison_dir: Path | None = None) -> Path:
    package_root = Path(__file__).resolve().parent
    results_root = package_root / "results" / "main"
    results_root.mkdir(parents=True, exist_ok=True)
    summary = _collect_source_data(settings, results_root)
    _write_report(settings, results_root, summary)
    shutil.copy2(results_root / "RESULTS_AND_ANALYSIS.md", package_root / "RESULTS_AND_ANALYSIS.md")
    _write_figure_contract(results_root)
    render_from_source(results_root, legacy_comparison_dir=legacy_comparison_dir)
    return results_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the fixed-feedback result record and Nature-style figures.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        if args.results_root is None:
            raise SystemExit("--plot-only requires --results-root")
        render_from_source(args.results_root.resolve())
        return 0
    if args.config is None:
        raise SystemExit("--config is required unless --plot-only is used")
    from .settings import load_settings

    settings = load_settings(args.config)
    generate_publication_report(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
