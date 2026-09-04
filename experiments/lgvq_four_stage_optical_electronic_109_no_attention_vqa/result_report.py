"""Generate the simulation-only result report and publication figures.

The script deliberately reads only the immutable JSON evidence copied from the
recommended run.  It does not load a model, checkpoint, video, or hardware
driver, so regenerating the report cannot change an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = PROJECT_DIR / "evidence" / "recommended"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "deployment" / "simulation_report"
DEFAULT_REPORT = PROJECT_DIR / "SIMULATION_REPORT.md"

CM = 1.0 / 2.54
COLORS = {
    "hybrid": "#0072B2",
    "bypassed": "#D55E00",
    "stage1": "#009E73",
    "stage2": "#56B4E9",
    "stage3": "#CC79A7",
    "stage4": "#E69F00",
    "neutral": "#6E6E6E",
    "alert": "#B2182B",
}


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.18,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _despine(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8, zorder=0)


def _save_pair(fig: Any, output_dir: Path, stem: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / (stem + ".png"), output_dir / (stem + ".pdf")]
    fig.savefig(str(paths[0]), dpi=600)
    fig.savefig(str(paths[1]))
    plt.close(fig)
    return paths


def _bar_values(axis: Any, bars: Iterable[Any], values: Sequence[float], fmt: str) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=6,
        )


def plot_optical_contribution(
    optical_on: Mapping[str, Any], optical_off: Mapping[str, Any], output_dir: Path
) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(17.8 * CM, 5.4 * CM), constrained_layout=True)
    targets = ("spatial", "temporal")
    target_labels = ("Spatial", "Temporal")
    x = np.arange(len(targets), dtype=float)
    width = 0.34
    panels = (("srcc", "SRCC", (0.0, 1.0)), ("plcc", "PLCC", (0.0, 1.0)), ("rmse", "RMSE", (0.0, 25.5)))
    for panel_index, (axis, (metric, title, ylim)) in enumerate(zip(axes, panels)):
        on_values = [float(optical_on[target][metric]) for target in targets]
        off_values = [float(optical_off[target][metric]) for target in targets]
        left = axis.bar(x - width / 2.0, on_values, width, color=COLORS["hybrid"], label="Hybrid", zorder=3)
        right = axis.bar(x + width / 2.0, off_values, width, color=COLORS["bypassed"], label="Optics bypassed", zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels(target_labels)
        axis.set_ylabel(title)
        axis.set_ylim(*ylim)
        axis.set_title(title, fontweight="bold", pad=3)
        _bar_values(axis, left, on_values, "{:.3f}" if metric != "rmse" else "{:.2f}")
        _bar_values(axis, right, off_values, "{:.3f}" if metric != "rmse" else "{:.2f}")
        _despine(axis)
        _panel_label(axis, chr(ord("a") + panel_index))
    axes[0].legend(frameon=False, loc="upper left")
    return _save_pair(fig, output_dir, "fig01_same_checkpoint_optical_contribution")


def _evaluated_rows(history: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in history if bool(row.get("test_evaluated")) and "test_optical_on" in row]


def plot_alpha_trajectory(
    history: Sequence[Mapping[str, Any]], best_epoch: int, alpha_minimum: float, output_dir: Path
) -> List[Path]:
    rows = _evaluated_rows(history)
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=float)
    fig, axis = plt.subplots(figsize=(9.0 * CM, 5.0 * CM), constrained_layout=True)
    for stage_index in range(1, 5):
        stage = "stage{}".format(stage_index)
        values = [float(row["test_optical_on"]["fusion_diagnostics"][stage]["alpha"]) for row in rows]
        axis.plot(
            epochs,
            values,
            marker="o",
            markevery=max(1, len(values) // 8),
            color=COLORS[stage],
            label="Stage {}".format(stage_index),
        )
    axis.axhline(alpha_minimum, color=COLORS["neutral"], linestyle="--", linewidth=0.8, label=r"$\alpha$ floor")
    axis.axvline(best_epoch, color="#222222", linestyle=":", linewidth=0.8)
    axis.text(best_epoch + 1.5, axis.get_ylim()[1], "selected epoch {}".format(best_epoch), va="top", fontsize=6)
    axis.set_xlabel("Epoch (periodic test only)")
    axis.set_ylabel(r"Normalized fusion coefficient $\alpha$")
    axis.set_title("Fusion coefficients during training", fontweight="bold", pad=3)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    _despine(axis)
    _panel_label(axis, "a")
    return _save_pair(fig, output_dir, "fig02_fusion_alpha_trajectory")


def plot_router_diagnostics(router: Mapping[str, Any], output_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(17.8 * CM, 5.4 * CM), constrained_layout=True)
    experts = np.arange(4, dtype=float)
    width = 0.34
    stage1 = router["stage1"]
    stage3 = router["stage3"]
    stages = (stage1, stage3)
    stage_labels = ("Stage 1", "Stage 3")
    stage_colors = (COLORS["stage1"], COLORS["stage3"])

    for axis, field, title, panel in (
        (axes[0], "mean_probability", "Mean router probability", "a"),
        (axes[1], "selected_share", "Top-2 selection share", "b"),
    ):
        for offset, values, label, color in zip((-width / 2.0, width / 2.0), stages, stage_labels, stage_colors):
            series = [float(value) for value in values[field]]
            bars = axis.bar(experts + offset, series, width, color=color, label=label, zorder=3)
            if field == "selected_share" and label == "Stage 3":
                _bar_values(axis, bars, series, "{:.2f}")
        axis.set_xticks(experts)
        axis.set_xticklabels(("E1", "E2", "E3", "E4"))
        axis.set_ylim(0.0, 0.68 if field == "selected_share" else 0.62)
        axis.set_ylabel("Fraction")
        axis.set_title(title, fontweight="bold", pad=3)
        _despine(axis)
        _panel_label(axis, panel)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].text(
        1.5,
        0.645,
        "Stage 3 collapse: fixed E2 + E4",
        color=COLORS["alert"],
        fontsize=6,
        ha="center",
        va="top",
    )

    capture = [float(stage1["capture_fraction_mean"]), float(stage3["capture_fraction_mean"])]
    bars = axes[2].bar(np.arange(2), capture, color=stage_colors, width=0.55, zorder=3)
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels(stage_labels)
    axes[2].set_ylim(0.0, 0.5)
    axes[2].set_ylabel("Captured / active energy")
    axes[2].set_title("Detector-window capture", fontweight="bold", pad=3)
    _bar_values(axes[2], bars, capture, "{:.3f}")
    _despine(axes[2])
    _panel_label(axes[2], "c")
    return _save_pair(fig, output_dir, "fig03_optical_router_diagnostics")


def plot_phase_diagnostics(phase: Mapping[str, Any], output_dir: Path) -> List[Path]:
    ordered = (
        ("parallel_optics.raw_expert_phase", "S1 expert"),
        ("parallel_optics.raw_global_phase", "S2 global"),
        ("serial_optics.raw_expert_phase", "S3 expert"),
        ("serial_optics.raw_global_phase", "S4 global"),
        ("parallel_router.raw_router_phase", "S1 router"),
        ("serial_router.raw_router_phase", "S3 router"),
    )
    planes = phase["planes"]
    x = np.arange(len(ordered), dtype=float)
    labels = [label for _, label in ordered]
    delta = [float(planes[name]["wrapped_delta_rad_rms"]) for name, _ in ordered]
    changed = [float(planes[name]["fraction_changed_over_0p05_rad"]) for name, _ in ordered]
    colors = [COLORS["stage1"], COLORS["stage2"], COLORS["stage3"], COLORS["stage4"], "#3264A8", "#8C4A90"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2 * CM, 5.4 * CM), constrained_layout=True)
    for axis, values, title, ylabel, panel, fmt in (
        (axes[0], delta, "Wrapped phase displacement", "RMS displacement (rad)", "a", "{:.3f}"),
        (axes[1], changed, "Pixels changed > 0.05 rad", "Fraction of phase pixels", "b", "{:.2f}"),
    ):
        bars = axis.bar(x, values, color=colors, width=0.68, zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=38, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", pad=3)
        axis.set_ylim(0.0, max(values) * 1.28)
        _bar_values(axis, bars, values, fmt)
        _despine(axis)
        _panel_label(axis, panel)
    return _save_pair(fig, output_dir, "fig04_phase_training_diagnostics")


def _fmt(values: Sequence[float], digits: int = 4) -> str:
    return " / ".join(("{:.%df}" % digits).format(float(value)) for value in values)


def _report_text(
    contribution: Mapping[str, Any],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    phase: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> str:
    on = contribution["normal_optical_electronic"]
    off = contribution["same_checkpoint_optics_bypassed"]
    delta = contribution["on_minus_off"]
    fusion = on["fusion_diagnostics"]
    router = on["router_diagnostics"]
    alphas = [fusion["stage{}".format(index)]["alpha"] for index in range(1, 5)]
    stage1_probability = router["stage1"]["mean_probability"]
    stage1_selection = router["stage1"]["selected_share"]
    stage3_probability = router["stage3"]["mean_probability"]
    stage3_selection = router["stage3"]["selected_share"]
    phase_rows = []
    phase_labels = {
        "parallel_optics.raw_expert_phase": "Stage 1 expert",
        "parallel_optics.raw_global_phase": "Stage 2 global",
        "serial_optics.raw_expert_phase": "Stage 3 expert",
        "serial_optics.raw_global_phase": "Stage 4 global",
        "parallel_router.raw_router_phase": "Stage 1 router",
        "serial_router.raw_router_phase": "Stage 3 router",
    }
    for name, label in phase_labels.items():
        item = phase["planes"][name]
        phase_rows.append(
            "| {} | {:,} | {:.4f} | {:.4f} | {:.2%} |".format(
                label,
                int(item["parameters"]),
                float(item["phase_rad_std_final"]),
                float(item["wrapped_delta_rad_rms"]),
                float(item["fraction_changed_over_0p05_rad"]),
            )
        )
    relative_gain = (float(on["selection_mean_srcc"]) / float(off["selection_mean_srcc"]) - 1.0) * 100.0
    lines = [
        "# LGVQ 四层光电融合模型：仿真结果报告",
        "",
        "> 本报告只描述仿真测试。没有包含或暗示任何实际光路结果。所有数值均由 `evidence/recommended` 中的固定 JSON 证据自动生成。",
        "",
        "## 1. 正式对象与选模口径",
        "",
        "- 正式配置：`{}`。".format(Path(str(resolved["config_path"])).name),
        "- 数据划分：2250 个训练视频、558 个测试视频，不设 validation。",
        "- 训练共 100 epoch；epoch 1 以及此后每 5 epoch 在 test 上评估一次。",
        "- 选模指标：Spatial SRCC 与 Temporal SRCC 的算术平均；最高值出现在 epoch {}，为 `{:.4f}`。".format(int(summary["best_epoch"]), float(summary["best_optical_on_test_mean_srcc"])),
        "- checkpoint SHA256：`{}`。".format(provenance["checkpoint_sha256"]),
        "- **test 被用于周期性选模**，因此这些数值应准确称为“best observed test”，不能当作从未参与决策的无偏最终测试估计。",
        "- 训练期使用二维标量软目标（权重 {:.1f}），但部署推理不加载教师模型或软目标。".format(float(summary["soft_target_weight"])),
        "",
        "## 2. 光电融合与同 checkpoint 去光对照",
        "",
        "![同 checkpoint 的光学贡献](deployment/simulation_report/fig01_same_checkpoint_optical_contribution.png)",
        "",
        "| 模式 | 目标 | SRCC | PLCC | RMSE |",
        "|---|---|---:|---:|---:|",
        "| 正常光电融合 | Spatial | {:.4f} | {:.4f} | {:.4f} |".format(float(on["spatial"]["srcc"]), float(on["spatial"]["plcc"]), float(on["spatial"]["rmse"])),
        "| 同 checkpoint 旁路光学 | Spatial | {:.4f} | {:.4f} | {:.4f} |".format(float(off["spatial"]["srcc"]), float(off["spatial"]["plcc"]), float(off["spatial"]["rmse"])),
        "| 正常光电融合 | Temporal | {:.4f} | {:.4f} | {:.4f} |".format(float(on["temporal"]["srcc"]), float(on["temporal"]["plcc"]), float(on["temporal"]["rmse"])),
        "| 同 checkpoint 旁路光学 | Temporal | {:.4f} | {:.4f} | {:.4f} |".format(float(off["temporal"]["srcc"]), float(off["temporal"]["plcc"]), float(off["temporal"]["rmse"])),
        "",
        "开光相对旁路光学：Spatial SRCC `{:+.4f}`、PLCC `{:+.4f}`、RMSE `{:+.4f}`；Temporal SRCC `{:+.4f}`、PLCC `{:+.4f}`、RMSE `{:+.4f}`。平均 SRCC 从 `{:.4f}` 提高到 `{:.4f}`，绝对增益 `{:+.4f}`，相对增益 `{:.1f}%`。".format(
            float(delta["spatial"]["srcc"]), float(delta["spatial"]["plcc"]), float(delta["spatial"]["rmse"]),
            float(delta["temporal"]["srcc"]), float(delta["temporal"]["plcc"]), float(delta["temporal"]["rmse"]),
            float(off["selection_mean_srcc"]), float(on["selection_mean_srcc"]), float(on["selection_mean_srcc"]) - float(off["selection_mean_srcc"]), relative_gain,
        ),
        "",
        "这里的对照只回答：**同一个已训练光电模型在推理时失去光学分支会下降多少**。本工程没有单独训练、微调或选模一个纯电子模型，因此不能据此声称光电模型优于“重新训练到最优的纯电子基线”。",
        "",
        "## 3. 融合系数与数值尺度",
        "",
        "![融合 alpha 轨迹](deployment/simulation_report/fig02_fusion_alpha_trajectory.png)",
        "",
        "最佳 checkpoint 的 Stage 1–4 alpha 为 `{}`，均高于配置下限 `{:.2f}`。每层融合前分别计算电子与光学特征 RMS，将两路归一到相同尺度后执行 `(1-alpha)E + alpha O`，再把输出 RMS 恢复到电子分支尺度；四层 `output_to_electronic_rms` 均为 `1.0000`。因此 alpha 是**尺度配平后的混合系数**，不能直接解释为探测器原始光功率占比。".format(_fmt(alphas), float(resolved["alpha_min"])),
        "",
        "| Stage | alpha | 电子 RMS（配平前） | 光学 RMS（配平前） | 输出/电子 RMS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index in range(1, 5):
        item = fusion["stage{}".format(index)]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                index,
                float(item["alpha"]),
                float(item["electronic_rms"]),
                float(item["optical_rms"]),
                float(item["output_to_electronic_rms"]),
            )
        )
    lines.extend(
        [
            "",
            "## 4. 光路由诊断",
            "",
            "![光路由诊断](deployment/simulation_report/fig03_optical_router_diagnostics.png)",
            "",
            "| Router | 决策数 | E1–E4 平均概率 | E1–E4 Top-2 选择份额 | 捕获率 |",
            "|---|---:|---|---|---:|",
            "| Stage 1（并行） | {:,} | {} | {} | {:.2%} |".format(int(router["stage1"]["decision_count"]), _fmt(stage1_probability), _fmt(stage1_selection), float(router["stage1"]["capture_fraction_mean"])),
            "| Stage 3（串行） | {:,} | {} | {} | {:.2%} |".format(int(router["stage3"]["decision_count"]), _fmt(stage3_probability), _fmt(stage3_selection), float(router["stage3"]["capture_fraction_mean"])),
            "",
            "`selected_share` 以所有被选中的 Top-2 槽位为分母，四项和为 1。Stage 1 的四个专家均被使用；Stage 3 的选择份额为 `0 / 0.5 / 0 / 0.5`。由于每次必须选两个专家，这意味着 **558 次 Stage 3 决策全部只选择 E2 与 E4，E1 与 E3 从未进入硬 Top-2**。虽然 soft probability 在 E1/E3 上仍非零，但硬路由组合没有样本间变化，因此应明确报告为 **Stage 3 expert-selection collapse**。",
            "",
            "捕获率表示四个路由探测窗口能量总和占对应 active detector 能量的比例；Stage 1 为 `{:.2%}`，Stage 3 为 `{:.2%}`。它衡量能量是否进入定义的路由窗口，不是分类/回归准确率。".format(float(router["stage1"]["capture_fraction_mean"]), float(router["stage3"]["capture_fraction_mean"])),
            "",
            "## 5. 相位是否实际训练",
            "",
            "![相位训练诊断](deployment/simulation_report/fig04_phase_training_diagnostics.png)",
            "",
            "| 相位面 | 参数数 | 最终相位 std (rad) | 相对初始化 wrapped RMS (rad) | 变化 >0.05 rad 的像素 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(phase_rows)
    lines.extend(
        [
            "",
            "六个相位面的平均 wrapped RMS 位移为 `{:.4f} rad`，说明整体并非停留在初始化。Stage 4 global 的位移最小（`{:.4f} rad`，仅 `{:.2%}` 像素变化超过 0.05 rad），应视为当前最弱的相位学习环节；这与 Stage 3 路由塌缩是两个不同问题。".format(
                float(phase["mean_wrapped_delta_rad_rms"]),
                float(phase["planes"]["serial_optics.raw_global_phase"]["wrapped_delta_rad_rms"]),
                float(phase["planes"]["serial_optics.raw_global_phase"]["fraction_changed_over_0p05_rad"]),
            ),
            "",
            "## 6. 科学结论边界",
            "",
            "1. 可以陈述：在选中的光电 checkpoint 上，旁路所有光学计算会显著降低两项任务的 SRCC/PLCC 并提高 RMSE，模型对光学分支存在明确的推理依赖。",
            "2. 不可以陈述：该结果证明光电模型优于一个独立充分训练的纯电子模型；这样的基线没有运行。",
            "3. 不可以陈述：Stage 3 的四专家实现了有效动态分工；硬 Top-2 已塌缩为固定 E2+E4。",
            "4. 当前数值是 test 参与选模后的 best-observed 结果，论文中应如实注明该口径。",
            "5. 本报告是仿真证据汇总，不代表硬件复现结果。",
            "",
            "## 7. 复现报告",
            "",
            "从仓库根目录执行：",
            "",
            "```powershell",
            "python experiments\\lgvq_four_stage_optical_electronic_109_no_attention_vqa\\result_report.py",
            "```",
            "",
            "输出目录为 `deployment/simulation_report`。所有 PNG 使用 600 dpi；PNG/PDF 图中文字统一为 Arial 7 pt，画布高度为 5.0–5.4 cm。",
            "",
        ]
    )
    return "\n".join(lines)


def generate(evidence_dir: Path, output_dir: Path, report_path: Path) -> Dict[str, Any]:
    required = {
        "on": evidence_dir / "test_metrics_optical_on.json",
        "off": evidence_dir / "test_metrics_optical_off.json",
        "contribution": evidence_dir / "optical_contribution_same_checkpoint.json",
        "history": evidence_dir / "train_history.json",
        "summary": evidence_dir / "training_summary.json",
        "phase": evidence_dir / "phase_training_diagnostics.json",
        "provenance": evidence_dir / "checkpoint_provenance.json",
        "resolved": evidence_dir / "resolved_config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report evidence: {}".format(missing))
    optical_on = _json(required["on"])
    optical_off = _json(required["off"])
    contribution = _json(required["contribution"])
    history = _json(required["history"])
    summary = _json(required["summary"])
    phase = _json(required["phase"])
    provenance = _json(required["provenance"])
    resolved = _json(required["resolved"])

    if contribution.get("separately_trained_electronic_baseline") is not False:
        raise RuntimeError("Evidence no longer describes a same-checkpoint-only ablation")
    if provenance.get("same_checkpoint_used_for_both_evaluations") is not True:
        raise RuntimeError("Optical on/off evaluations do not share a checkpoint")
    if optical_on != contribution["normal_optical_electronic"]:
        raise RuntimeError("Optical-on evidence disagrees with contribution report")
    if optical_off != contribution["same_checkpoint_optics_bypassed"]:
        raise RuntimeError("Optical-off evidence disagrees with contribution report")
    if int(summary["best_epoch"]) != int(provenance["checkpoint_epoch"]):
        raise RuntimeError("Best epoch and checkpoint provenance disagree")
    stage3_share = optical_on["router_diagnostics"]["stage3"]["selected_share"]
    if not (float(stage3_share[0]) == 0.0 and float(stage3_share[2]) == 0.0):
        raise RuntimeError("Expected Stage 3 collapse signature is absent; update report wording")

    _style()
    artifacts = []
    artifacts.extend(plot_optical_contribution(optical_on, optical_off, output_dir))
    artifacts.extend(plot_alpha_trajectory(history, int(summary["best_epoch"]), float(resolved["alpha_min"]), output_dir))
    artifacts.extend(plot_router_diagnostics(optical_on["router_diagnostics"], output_dir))
    artifacts.extend(plot_phase_diagnostics(phase, output_dir))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_text(contribution, summary, provenance, phase, resolved), encoding="utf-8")
    artifacts.append(report_path)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "simulation_only",
        "source_evidence": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in required.items()},
        "checkpoint_epoch": int(provenance["checkpoint_epoch"]),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "test_selection_policy": "highest periodically observed optical-on test mean SRCC; no validation split",
        "same_checkpoint_optics_bypassed": True,
        "separately_trained_electronic_baseline": False,
        "stage3_expert_selection_collapse": True,
        "figure_style": {"font": "Arial", "font_size_pt": 7, "png_dpi": 600, "height_cm": [5.0, 5.4]},
        "artifacts": [{"path": str(path), "sha256": _sha256(path)} for path in artifacts],
    }
    manifest_path = output_dir / "report_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    result = generate(args.evidence_dir.resolve(), args.output_dir.resolve(), args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
