"""Freeze summaries and draw compact Nature-style figures for four hybrid tasks.

The script reads only the evidence copied into this document directory.  It
does not read checkpoints or live run directories, so a later rerun cannot
silently change a reported figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
FIGURES = ROOT / "figures"

# Nature double-column width, with the requested 4--6 cm height.
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 54.0
FIGSIZE = (FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4)

COLORS = {
    "blue": "#3569A8",
    "teal": "#3E8E88",
    "gold": "#C68A24",
    "red": "#B24A4A",
    "grey": "#737373",
    "light_grey": "#D9D9D9",
    "grid": "#E5E5E5",
    "text": "#222222",
}

SOURCE_ROOT = "/DATA/DATA1/guest3/2026OpticsMoE"
SOURCE_PATHS = {
    "salicon/student_history.csv": "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/metrics/student_history.csv",
    "salicon/dataset.json": "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/dataset.json",
    "salicon/resolved_config.json": "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/resolved_config.json",
    "salicon/student_model.json": "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/student_model.json",
    "salicon/environment.json": "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/environment.json",
    "isic2016/training_history.csv": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/metrics/training_history.csv",
    "isic2016/test_metrics.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/metrics/test_metrics.json",
    "isic2016/test_predictions.csv": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/metrics/test_predictions.csv",
    "isic2016/initial_test_observation.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/metrics/initial_test_observation.json",
    "isic2016/optimizer_joint_end_to_end.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/metrics/optimizer_joint_end_to_end.json",
    "isic2016/dataset.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/dataset.json",
    "isic2016/resolved_config.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/resolved_config.json",
    "isic2016/model.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/model.json",
    "isic2016/qwen_vision.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/qwen_vision.json",
    "isic2016/environment.json": "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/environment.json",
    "lsp/student_training_history.csv": "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/metrics/student_training_history.csv",
    "lsp/student_model.json": "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/metrics/student_model.json",
    "lsp/dataset.json": "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/dataset.json",
    "lsp/resolved_config.yaml": "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/resolved_config.yaml",
    "caltech101/train_log.csv": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/train_log.csv",
    "caltech101/config.yaml": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/config.yaml",
    "caltech101/dataset.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/dataset.json",
    "caltech101/model.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/model.json",
    "caltech101/environment.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/environment.json",
    "caltech101/metrics_training_latest.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/metrics/training_latest.json",
    "caltech101/metrics_best_train_loss.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/metrics/best_train_loss.json",
    "caltech101/metrics_best_observed_test.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/metrics/best_observed_test.json",
    "caltech101/metrics_ema_best_observed_test.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/metrics/ema_best_observed_test.json",
    "caltech101/metrics_phase_training_latest.json": "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/metrics/phase_training_latest.json",
}


def configure_plotting() -> None:
    """Require Arial instead of silently producing a mislabeled fallback."""
    font_manager.findfont("Arial", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(EVIDENCE / relative)


def validate_history(name: str, frame: pd.DataFrame, epochs: int) -> None:
    if len(frame) != epochs:
        raise RuntimeError(f"{name}: expected {epochs} rows, found {len(frame)}")
    expected = np.arange(1, epochs + 1)
    if not np.array_equal(frame["epoch"].to_numpy(dtype=int), expected):
        raise RuntimeError(f"{name}: epoch sequence is incomplete or reordered")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{name}: non-finite value found in numeric history")


def selected_rows() -> dict[str, Any]:
    salicon = load_csv("salicon/student_history.csv")
    isic = load_csv("isic2016/training_history.csv")
    lsp = load_csv("lsp/student_training_history.csv")
    caltech = load_csv("caltech101/train_log.csv")
    validate_history("SALICON", salicon, 60)
    validate_history("ISIC 2016", isic, 100)
    validate_history("LSP", lsp, 150)
    validate_history("Caltech101", caltech, 60)

    salicon_selected = salicon.loc[salicon["validation_cc"].idxmax()]
    isic_selected = isic.loc[isic["train_loss"].idxmin()]
    lsp_selected = lsp.loc[lsp["train_loss"].idxmin()]
    lsp_observed = lsp.loc[lsp["test_pck_at_0.2_torso"].idxmax()]
    caltech_selected = caltech.loc[caltech["total_loss"].idxmin()]
    caltech_observed = caltech.loc[caltech["test_top1"].idxmax()]
    caltech_best_train = load_json("caltech101/metrics_best_train_loss.json")
    caltech_dataset = load_json("caltech101/dataset.json")
    isic_test = load_json("isic2016/test_metrics.json")
    if int(isic_test["checkpoint_epoch"]) != int(isic_selected["epoch"]):
        raise RuntimeError("ISIC test result does not match min-train-loss checkpoint")
    if bool(isic_test["test_used_for_selection"]):
        raise RuntimeError("ISIC test set unexpectedly participated in selection")
    if int(caltech_best_train["epoch"]) != int(caltech_selected["epoch"]):
        raise RuntimeError("Caltech101 best-train-loss record does not match history")
    if not bool(caltech_best_train["test_was_not_used_for_selection"]):
        raise RuntimeError("Caltech101 test unexpectedly participated in formal selection")
    if not np.allclose(caltech["phase_learning_rate"].to_numpy(dtype=float), 0.0):
        raise RuntimeError("Caltech101 frozen evidence no longer has fixed phase masks")
    if not np.allclose(caltech["phase_delta_run_rms_rad"].to_numpy(dtype=float), 0.0):
        raise RuntimeError("Caltech101 phase masks moved despite the fixed-phase record")

    return {
        "frames": {
            "salicon": salicon,
            "isic2016": isic,
            "lsp": lsp,
            "caltech101": caltech,
        },
        "salicon_selected": salicon_selected,
        "isic_selected": isic_selected,
        "isic_test": isic_test,
        "lsp_selected": lsp_selected,
        "lsp_observed": lsp_observed,
        "caltech_selected": caltech_selected,
        "caltech_observed": caltech_observed,
        "caltech_best_train": caltech_best_train,
        "caltech_dataset": caltech_dataset,
    }


def scalar(value: Any) -> float | int | str | bool | None:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_summaries(rows: dict[str, Any]) -> None:
    salicon = rows["salicon_selected"]
    isic_train = rows["isic_selected"]
    isic_test = rows["isic_test"]
    lsp = rows["lsp_selected"]
    lsp_peak = rows["lsp_observed"]
    caltech = rows["caltech_selected"]
    caltech_peak = rows["caltech_observed"]
    caltech_dataset = rows["caltech_dataset"]
    summary = {
        "generated_from_frozen_evidence": True,
        "common_architecture": {
            "name": "Vision2 hybrid dense MoE4",
            "qwen_stem": "frozen patch embedding and position processing",
            "native_qwen_vision_blocks_executed": 0,
            "deepstack_enabled": False,
            "latent_width": 192,
            "electronic_blocks": 2,
            "electronic_token_mixer": "3x3 depthwise Conv2D + pointwise projection",
            "optical_stages": ["MoE4 expert stage (top-2)", "global phase stage"],
            "fusion": "electronic + sigmoid(gate) * optical_delta at each stage",
            "joint_training_from_epoch_1": True,
            "teacher_or_kd_used": False,
            "attention_used": False,
        },
        "caltech101_architecture": {
            "name": "Vision2 + Language2 four-stage hybrid retrieval",
            "evaluation_mode": "simulation only",
            "qwen_backbone": "pretrained and frozen",
            "deepstack_enabled": False,
            "latent_width": 192,
            "vision_electronic_blocks": "2x 3x3 depthwise Conv2D residual MLP",
            "language_electronic_blocks": "2x causal depthwise Conv1D residual MLP",
            "optical_stages": [
                "vision MoE4 expert",
                "vision global",
                "language MoE4 expert",
                "language global",
            ],
            "fusion": "electronic + sigmoid(gate) * optical_delta at each stage",
            "teacher_or_kd_used": False,
            "attention_used": False,
            "phase_masks_optimized_in_this_run": False,
            "phase_learning_rate": 0.0,
        },
        "tasks": {
            "salicon": {
                "epochs": 60,
                "selection": "maximum validation CC",
                "selected_epoch": int(salicon["epoch"]),
                "split": "official public validation",
                "samples": 5000,
                "metrics": {
                    key.removeprefix("validation_"): float(salicon[key])
                    for key in [
                        "validation_kld",
                        "validation_cc",
                        "validation_sim",
                        "validation_nss",
                        "validation_auc_judd",
                        "validation_mae",
                    ]
                },
            },
            "isic2016": {
                "epochs": 100,
                "selection": "minimum training loss; test evaluated once after selection",
                "selected_epoch": int(isic_train["epoch"]),
                "split": "official test",
                "samples": int(isic_test["samples"]),
                "metrics": {
                    key: scalar(isic_test[key])
                    for key in [
                        "loss",
                        "mean_iou",
                        "mean_dice",
                        "mae",
                        "pixel_accuracy",
                        "sensitivity",
                        "specificity",
                        "balanced_pixel_accuracy",
                    ]
                },
            },
            "lsp": {
                "epochs": 150,
                "selection": "minimum training loss",
                "selected_epoch": int(lsp["epoch"]),
                "split": "fixed LSP test; monitored every epoch",
                "samples": int(lsp["test_samples"]),
                "selected_checkpoint_metrics": {
                    "pck_at_0.2_torso": float(lsp["test_pck_at_0.2_torso"]),
                    "pckh_at_0.5_head": float(lsp["test_pckh_at_0.5_head"]),
                    "mean_pixel_error": float(lsp["test_mean_pixel_error"]),
                    "normalized_mean_error_torso": float(
                        lsp["test_normalized_mean_error_torso"]
                    ),
                },
                "observed_test_peak_not_for_formal_selection": {
                    "epoch": int(lsp_peak["epoch"]),
                    "pck_at_0.2_torso": float(lsp_peak["test_pck_at_0.2_torso"]),
                    "pckh_at_0.5_head": float(lsp_peak["test_pckh_at_0.5_head"]),
                },
            },
            "caltech101": {
                "epochs": 60,
                "selection": "minimum training total loss; test monitored every epoch",
                "selected_epoch": int(caltech["epoch"]),
                "split": "fixed class-balanced test query set",
                "samples": int(caltech_dataset["counts"]["test"]),
                "gallery_samples": int(caltech_dataset["counts"]["gallery"]),
                "train_samples": int(caltech_dataset["counts"]["train"]),
                "classes": len(caltech_dataset["selected_categories"]),
                "simulation_only": True,
                "phase_masks_optimized": False,
                "phase_learning_rate": float(caltech["phase_learning_rate"]),
                "selected_checkpoint_metrics": {
                    "top1": float(caltech["test_top1"]),
                    "top3": float(caltech["test_top3"]),
                    "mrr": float(caltech["test_mrr"]),
                    "ema_top1": float(caltech["ema_test_top1"]),
                    "ema_top3": float(caltech["ema_test_top3"]),
                    "ema_mrr": float(caltech["ema_test_mrr"]),
                },
                "observed_test_peak_not_for_formal_selection": {
                    "epoch": int(caltech_peak["epoch"]),
                    "top1": float(caltech_peak["test_top1"]),
                    "selection_biased": True,
                },
                "selected_checkpoint_diagnostics": {
                    "phase_grad_rms": float(caltech["phase_grad_rms"]),
                    "phase_delta_run_rms_rad": float(
                        caltech["phase_delta_run_rms_rad"]
                    ),
                    "vision_router_entropy": float(caltech["vision_router_entropy"]),
                    "language_router_entropy": float(
                        caltech["language_router_entropy"]
                    ),
                    "vision_router_max_importance": float(
                        caltech["vision_router_max_importance"]
                    ),
                    "language_router_max_importance": float(
                        caltech["language_router_max_importance"]
                    ),
                    "vision_router_unselected_experts": int(
                        caltech["vision_router_unselected_experts"]
                    ),
                    "language_router_unselected_experts": int(
                        caltech["language_router_unselected_experts"]
                    ),
                },
            },
        },
    }
    (EVIDENCE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        {
            "task": "SALICON",
            "selected_epoch": int(salicon["epoch"]),
            "selection_rule": "max validation CC",
            "evaluation_split": "official public validation",
            "evaluation_samples": 5000,
            "primary_metric": "CC",
            "primary_value": float(salicon["validation_cc"]),
            "secondary_metric": "AUC-Judd",
            "secondary_value": float(salicon["validation_auc_judd"]),
            "status": "single run",
        },
        {
            "task": "ISIC 2016",
            "selected_epoch": int(isic_train["epoch"]),
            "selection_rule": "min training loss",
            "evaluation_split": "official test",
            "evaluation_samples": int(isic_test["samples"]),
            "primary_metric": "mean IoU",
            "primary_value": float(isic_test["mean_iou"]),
            "secondary_metric": "mean Dice",
            "secondary_value": float(isic_test["mean_dice"]),
            "status": "single run; test evaluated once after selection",
        },
        {
            "task": "LSP",
            "selected_epoch": int(lsp["epoch"]),
            "selection_rule": "min training loss",
            "evaluation_split": "fixed LSP test monitored each epoch",
            "evaluation_samples": int(lsp["test_samples"]),
            "primary_metric": "PCK@0.2 torso",
            "primary_value": float(lsp["test_pck_at_0.2_torso"]),
            "secondary_metric": "PCKh@0.5 head",
            "secondary_value": float(lsp["test_pckh_at_0.5_head"]),
            "status": "single run; selected without test metric",
        },
        {
            "task": "Caltech101-10",
            "selected_epoch": int(caltech["epoch"]),
            "selection_rule": "min training total loss",
            "evaluation_split": "fixed test queries monitored each epoch",
            "evaluation_samples": int(caltech_dataset["counts"]["test"]),
            "primary_metric": "Top-1",
            "primary_value": float(caltech["test_top1"]),
            "secondary_metric": "EMA Top-1",
            "secondary_value": float(caltech["ema_test_top1"]),
            "status": "simulation only; phase LR 0; single run",
        },
    ]
    with (EVIDENCE / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def write_manifest() -> None:
    records = []
    for relative, source in SOURCE_PATHS.items():
        path = EVIDENCE / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "local_path": f"evidence/{relative}",
                "server_source": f"{SOURCE_ROOT}/{source}",
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    with (ROOT / "SOURCE_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def finish_axes(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45, zorder=0)
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_note(figure: plt.Figure, text: str) -> None:
    figure.text(0.01, 0.025, text, ha="left", va="bottom", fontsize=7)


def export(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / name
    svg_path = stem.with_suffix(".svg")
    figure.savefig(svg_path)
    # Matplotlib writes trailing spaces inside multiline SVG path data.  Strip
    # them so repository whitespace checks stay useful and deterministic.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def salicon_figure(rows: dict[str, Any]) -> None:
    frame = rows["frames"]["salicon"]
    selected = rows["salicon_selected"]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axis = axes[0]
    for key, label, color in [
        ("validation_cc", "CC", COLORS["blue"]),
        ("validation_sim", "SIM", COLORS["teal"]),
        ("validation_auc_judd", "AUC-Judd", COLORS["gold"]),
    ]:
        axis.plot(frame["epoch"], frame[key], label=label, color=color)
    axis.axvline(int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8, linestyle="--")
    axis.scatter(
        [selected["epoch"]], [selected["validation_cc"]],
        color=COLORS["blue"], edgecolor="white", linewidth=0.5, zorder=5,
    )
    axis.set(xlabel="Epoch", ylabel="Validation score", title="SALICON score trajectory")
    axis.set_ylim(0.73, 0.88)
    axis.legend(loc="lower right", ncol=3, columnspacing=0.9, handlelength=1.5)
    finish_axes(axis)
    panel_label(axis, "a")

    axis = axes[1]
    axis.plot(frame["epoch"], frame["validation_kld"], label="KLD", color=COLORS["red"])
    axis.plot(frame["epoch"], frame["validation_mae"], label="MAE", color=COLORS["grey"])
    axis.axvline(int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8, linestyle="--")
    axis.set(xlabel="Epoch", ylabel="Validation error (lower is better)", title="Density-map error")
    axis.set_ylim(0.06, 0.20)
    axis.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.5)
    finish_axes(axis)
    panel_label(axis, "b")
    add_note(
        figure,
        f"Official public validation n=5,000; checkpoint epoch {int(selected['epoch'])} selected by maximum CC; single run.",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.29, wspace=0.30)
    export(figure, "salicon_vision2_hybrid")


def isic_figure(rows: dict[str, Any]) -> None:
    frame = rows["frames"]["isic2016"]
    selected = rows["isic_selected"]
    test = rows["isic_test"]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1.0]})
    axis = axes[0]
    axis.plot(frame["epoch"], frame["train_mean_iou"], label="Train IoU", color=COLORS["blue"])
    axis.plot(frame["epoch"], frame["train_mean_dice"], label="Train Dice", color=COLORS["teal"])
    axis.axvline(int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8, linestyle="--")
    axis.set(xlabel="Epoch", ylabel="Training score", title="ISIC 2016 optimization")
    axis.set_ylim(0.55, 0.96)
    axis.legend(loc="lower right", ncol=2, columnspacing=1.0, handlelength=1.5)
    finish_axes(axis)
    panel_label(axis, "a")

    labels = ["IoU", "Dice", "Sensitivity", "Specificity", "Balanced acc."]
    values = [
        test["mean_iou"],
        test["mean_dice"],
        test["sensitivity"],
        test["specificity"],
        test["balanced_pixel_accuracy"],
    ]
    axis = axes[1]
    y = np.arange(len(labels))[::-1]
    axis.hlines(y, 0.78, values, color=COLORS["light_grey"], linewidth=2.0, zorder=1)
    axis.scatter(values, y, color=COLORS["blue"], s=20, zorder=3)
    for value, position in zip(values, y):
        axis.text(value + 0.004, position, f"{value:.3f}", va="center", ha="left")
    axis.set_yticks(y, labels)
    axis.set(xlabel="Official-test score", title="Frozen checkpoint evaluation")
    axis.set_xlim(0.78, 0.985)
    finish_axes(axis)
    panel_label(axis, "b")
    add_note(
        figure,
        f"Official test n=379, evaluated once after selecting epoch {int(selected['epoch'])} by minimum training loss; single run.",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.29, wspace=0.34)
    export(figure, "isic2016_vision2_hybrid")


def lsp_figure(rows: dict[str, Any]) -> None:
    frame = rows["frames"]["lsp"]
    selected = rows["lsp_selected"]
    peak = rows["lsp_observed"]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axis = axes[0]
    axis.plot(frame["epoch"], frame["test_pck_at_0.2_torso"], label="PCK@0.2", color=COLORS["blue"])
    axis.plot(frame["epoch"], frame["test_pckh_at_0.5_head"], label="PCKh@0.5", color=COLORS["teal"])
    axis.axvline(int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8, linestyle="--", label="Selected")
    axis.scatter(
        [peak["epoch"]], [peak["test_pck_at_0.2_torso"]],
        marker="x", s=24, linewidth=1.0, color=COLORS["red"], zorder=5,
    )
    axis.set(xlabel="Epoch", ylabel="Monitored-test score", title="LSP pose trajectory")
    axis.set_ylim(0.25, 0.88)
    axis.legend(loc="lower right", ncol=3, columnspacing=0.8, handlelength=1.4)
    finish_axes(axis)
    panel_label(axis, "a")

    axis = axes[1]
    axis.plot(
        frame["epoch"], frame["test_normalized_mean_error_torso"],
        label="Test NME", color=COLORS["red"],
    )
    axis.plot(
        frame["epoch"], frame["train_router_entropy"],
        label="Router entropy", color=COLORS["gold"],
    )
    axis.axvline(int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8, linestyle="--")
    axis.set(xlabel="Epoch", ylabel="Diagnostic value", title="Error and routing diagnostic")
    axis.set_ylim(-0.02, 0.62)
    axis.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.5)
    finish_axes(axis)
    panel_label(axis, "b")
    add_note(
        figure,
        f"Fixed LSP test n=1,000; epoch {int(selected['epoch'])} selected by training loss; red × is diagnostic test peak at epoch {int(peak['epoch'])}.",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.29, wspace=0.30)
    export(figure, "lsp_vision2_hybrid")


def caltech101_figure(rows: dict[str, Any]) -> None:
    frame = rows["frames"]["caltech101"]
    selected = rows["caltech_selected"]
    peak = rows["caltech_observed"]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    axis = axes[0]
    axis.plot(frame["epoch"], frame["test_top1"], label="Raw Top-1", color=COLORS["blue"])
    axis.plot(
        frame["epoch"], frame["ema_test_top1"],
        label="EMA Top-1", color=COLORS["teal"],
    )
    axis.axvline(
        int(selected["epoch"]), color=COLORS["grey"], linewidth=0.8,
        linestyle="--", label="Selected",
    )
    axis.scatter(
        [peak["epoch"]], [peak["test_top1"]], marker="x", s=24,
        linewidth=1.0, color=COLORS["red"], zorder=5,
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Monitored-test Top-1",
        title="Caltech101-10 retrieval trajectory",
    )
    axis.set_ylim(0.40, 0.93)
    axis.legend(loc="lower right", ncol=3, columnspacing=0.8, handlelength=1.4)
    finish_axes(axis)
    panel_label(axis, "a")

    axis = axes[1]
    labels = ["Top-1", "Top-3", "MRR"]
    raw = np.array([selected["test_top1"], selected["test_top3"], selected["test_mrr"]])
    ema = np.array(
        [selected["ema_test_top1"], selected["ema_test_top3"], selected["ema_test_mrr"]]
    )
    x = np.arange(len(labels))
    width = 0.32
    axis.bar(x - width / 2, raw, width, label="Raw", color=COLORS["blue"])
    axis.bar(x + width / 2, ema, width, label="EMA", color=COLORS["teal"])
    for positions, values in [(x - width / 2, raw), (x + width / 2, ema)]:
        for position, value in zip(positions, values):
            axis.text(position, value + 0.004, f"{value:.3f}", ha="center", va="bottom")
    axis.set_xticks(x, labels)
    axis.set(
        ylabel="Selected-checkpoint score",
        title=f"Epoch {int(selected['epoch'])} retrieval metrics",
    )
    axis.set_ylim(0.84, 1.005)
    axis.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.5)
    finish_axes(axis)
    panel_label(axis, "b")
    add_note(
        figure,
        "Simulation only; 10 classes, 200 test queries and 30 gallery images; "
        f"epoch {int(selected['epoch'])} selected by training loss; phase LR=0 (fixed masks); single run.",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.29, wspace=0.30)
    export(figure, "caltech101_four_layer_hybrid_simulation")


def main() -> None:
    configure_plotting()
    rows = selected_rows()
    write_summaries(rows)
    write_manifest()
    salicon_figure(rows)
    isic_figure(rows)
    lsp_figure(rows)
    caltech101_figure(rows)
    print(
        f"Wrote summaries and 4 figure sets at {FIGURE_WIDTH_MM:.0f} x "
        f"{FIGURE_HEIGHT_MM:.0f} mm with Arial 7 pt."
    )


if __name__ == "__main__":
    main()
