import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in [SRC_ROOT, SCRIPTS_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opticalmoe.optics.four_expert_geometry import FourExpertLayout
from opticalmoe.optics.microlens_prompt import MicrolensArrayPrompt
from test_four_expert_prompt_geometry import (
    apply_identity_expert,
    choose_device,
    compute_plane_metrics,
    core_centroid,
    edge_energy_ratio,
    intensity_2d,
    make_input_field,
    make_propagator,
    peak_location,
    plot_phase,
    plot_scalar_map,
    quadrant_masks,
    save_json,
    weighted_centroid,
    write_csv,
)


EPS = 1e-12
WAVELENGTH_M = 532e-9
PIXEL_SIZE_M = 8e-6
INTER_LAYER_M = 0.05
LAYER5_TO_FC_M = 0.05
FC_TO_DETECTOR_M = 0.05
AMPLITUDE_PATTERNS = {
    "all_on": [1.0, 1.0, 1.0, 1.0],
    "onehot_E0": [1.0, 0.0, 0.0, 0.0],
    "onehot_E1": [0.0, 1.0, 0.0, 0.0],
    "onehot_E2": [0.0, 0.0, 1.0, 0.0],
    "onehot_E3": [0.0, 0.0, 0.0, 1.0],
}


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Sweep four-expert microlens prompt cell sizes without training."
    )
    parser.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"])
    parser.add_argument(
        "--out-dir",
        default=f"runs/four_expert_prompt_cell_size_sweep_{timestamp}",
    )
    parser.add_argument(
        "--cell-sizes",
        nargs="+",
        type=int,
        default=[200, 220, 240, 260, 280, 300],
    )
    parser.add_argument(
        "--input-types",
        nargs="+",
        default=["centered_delta", "centered_square", "mnist_sample"],
        choices=["centered_delta", "centered_square", "mnist_sample"],
    )
    parser.add_argument("--include-negative-controls", action="store_true")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--mnist-index", type=int, default=5)
    parser.add_argument("--aperture-mode", default="hard", choices=["hard", "transparent"])
    parser.add_argument("--distance", type=float, default=0.20)
    parser.add_argument("--input-to-prompt", type=float, default=None)
    parser.add_argument("--prompt-to-expert", type=float, default=None)
    parser.add_argument("--focal-length", type=float, default=0.10)
    parser.add_argument("--plot-dpi", type=int, default=120)
    parser.add_argument("--square-size", type=int, default=100)
    parser.add_argument("--edge-border", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_distances(args) -> Tuple[float, float, float]:
    input_to_prompt_m = (
        float(args.distance)
        if args.input_to_prompt is None
        else float(args.input_to_prompt)
    )
    prompt_to_expert_m = (
        float(args.distance)
        if args.prompt_to_expert is None
        else float(args.prompt_to_expert)
    )
    focal_length_m = float(args.focal_length)
    if min(input_to_prompt_m, prompt_to_expert_m, focal_length_m) <= 0.0:
        raise ValueError("Distances and focal length must be positive.")
    return input_to_prompt_m, prompt_to_expert_m, focal_length_m


def target_index(amplitude_mode: str) -> Optional[int]:
    if amplitude_mode.startswith("onehot_E"):
        return int(amplitude_mode[-1])
    return None


def finite_mean(values: Sequence[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan")


def finite_max(values: Sequence[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.max(selected)) if selected else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / (float(denominator) + EPS)


def normalized_correlation(image: torch.Tensor, expected: torch.Tensor) -> float:
    image = image.float().flatten()
    expected = expected.float().flatten()
    image = image - image.mean()
    expected = expected - expected.mean()
    denominator = torch.linalg.vector_norm(image) * torch.linalg.vector_norm(expected)
    if float(denominator.item()) <= EPS:
        return float("nan")
    return float(torch.dot(image, expected).div(denominator).item())


def crop_correlations(
    field: torch.Tensor,
    input_field: torch.Tensor,
    layout: FourExpertLayout,
    active_experts: Sequence[int],
) -> List[float]:
    image = intensity_2d(field)
    input_intensity = intensity_2d(input_field)
    input_aperture = layout.input_aperture
    expected = input_intensity[
        input_aperture.y0 : input_aperture.y1,
        input_aperture.x0 : input_aperture.x1,
    ]
    # A unit-magnification thin-lens image is inverted in both axes.
    expected = torch.flip(expected, dims=(-2, -1))
    correlations = []
    for index in active_experts:
        aperture = layout.experts[index]
        crop = image[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
        if tuple(crop.shape) != tuple(expected.shape):
            crop = F.interpolate(
                crop.unsqueeze(0).unsqueeze(0),
                size=expected.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        correlations.append(normalized_correlation(crop, expected))
    return correlations


def sampling_metrics(
    layout: FourExpertLayout,
    prompt: MicrolensArrayPrompt,
    prompt_to_expert_m: float,
) -> Dict:
    reports = prompt.report(prompt_to_expert_m=prompt_to_expert_m)
    grating_periods = []
    for report in reports:
        for axis in ["x", "y"]:
            period = float(report[f"grating_period_{axis}_px"])
            if math.isfinite(period):
                grating_periods.append(period)
    min_grating_period_px = min(grating_periods)

    # Use the largest Cartesian lens slope at a cell edge. Sampling occurs on
    # x/y pixels, so this per-axis period is more relevant than radial arc length.
    half_extent_px = float(layout.prompt_cell_size) / 2.0
    min_lens_period_px = (
        WAVELENGTH_M
        * prompt.focal_length_m
        / (half_extent_px * PIXEL_SIZE_M ** 2)
    )
    max_lens_cycles_per_px = 1.0 / min_lens_period_px
    max_grating_cycles_per_px = 1.0 / min_grating_period_px
    min_total_phase_period_px = 1.0 / (
        max_lens_cycles_per_px + max_grating_cycles_per_px
    )
    return {
        "min_grating_period_px": min_grating_period_px,
        "min_lens_period_px": min_lens_period_px,
        "min_total_phase_period_px": min_total_phase_period_px,
        "warn_if_min_period_below_8px": bool(
            min_grating_period_px < 8.0 or min_lens_period_px < 8.0
        ),
    }


def active_expert_indices(amplitudes: Sequence[float]) -> List[int]:
    return [index for index, value in enumerate(amplitudes) if float(value) > 0.0]


def layer_centroids(
    field: torch.Tensor,
    layout: FourExpertLayout,
    active_experts: Sequence[int],
) -> Dict[int, Tuple[float, float]]:
    image = intensity_2d(field)
    masks = layout.expert_masks(device=image.device)
    return {
        index: weighted_centroid(image, masks[index])
        for index in active_experts
    }


def summarize_drift(
    traces: Sequence[Dict[int, Tuple[float, float]]],
    active_experts: Sequence[int],
) -> Tuple[float, float]:
    drifts = []
    for previous, current in zip(traces[:-1], traces[1:]):
        for index in active_experts:
            y0, x0 = previous[index]
            y1, x1 = current[index]
            if all(math.isfinite(value) for value in [y0, x0, y1, x1]):
                drifts.append(math.hypot(y1 - y0, x1 - x0))
    return finite_mean(drifts), finite_max(drifts)


def run_condition(
    input_type: str,
    input_field: torch.Tensor,
    after_input_to_prompt: torch.Tensor,
    prompt: MicrolensArrayPrompt,
    prompt_mode: str,
    amplitude_mode: str,
    amplitudes: Sequence[float],
    layout: FourExpertLayout,
    propagators: Dict,
    sampling: Dict,
    aperture_mode: str,
    edge_border: int,
) -> Tuple[Dict, Dict]:
    prompt.set_amplitudes(amplitudes)
    active = active_expert_indices(amplitudes)
    after_prompt = prompt(after_input_to_prompt, mode=prompt_mode)
    expert_plane = propagators["prompt_to_expert"](after_prompt)

    expert_masks = layout.expert_masks(device=expert_plane.device)
    copy_masks = quadrant_masks(layout, expert_plane.device)
    metric, _ = compute_plane_metrics(
        expert_plane,
        plane_name="expert1_plane",
        case_name=prompt_mode,
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        edge_border=edge_border,
        amplitudes=amplitudes,
    )

    centroid_errors = [
        metric[f"copy{index}_centroid_error_px"] for index in active
    ]
    core_errors = [
        metric[f"copy{index}_core_centroid_error_px"] for index in active
    ]
    target = target_index(amplitude_mode)
    expert_ratios = [
        float(metric[f"E{index}_energy_ratio"]) for index in range(4)
    ]
    if target is None:
        target_ratio = float("nan")
        max_wrong = float("nan")
        onehot_ratio = float("nan")
        expected_y = float("nan")
        expected_x = float("nan")
    else:
        target_ratio = expert_ratios[target]
        max_wrong = max(
            expert_ratios[index] for index in range(4) if index != target
        )
        onehot_ratio = safe_ratio(target_ratio, max_wrong)
        expected_y, expected_x = layout.experts[target].center

    stack_traces = []
    layer_fields = []
    field = expert_plane
    union_mask = layout.expert_union_mask(device=field.device)
    for layer_index in range(1, 6):
        field = apply_identity_expert(field, aperture_mode, union_mask)
        layer_fields.append(field)
        stack_traces.append(layer_centroids(field, layout, active))
        if layer_index < 5:
            field = propagators["inter_layer"](field)

    after_global_fc = propagators["layer5_to_fc"](field)
    detector_plane = propagators["fc_to_detector"](after_global_fc)
    final_image = intensity_2d(detector_plane)
    final_total = final_image.sum()
    final_expert_energy = torch.einsum(
        "hw,khw->k", final_image, expert_masks
    ).sum()
    final_outside_ratio = float(
        ((final_total - final_expert_energy).clamp_min(0.0) / (final_total + EPS)).item()
    )
    mean_drift, max_drift = summarize_drift(stack_traces, active)
    peak_y, peak_x = peak_location(intensity_2d(expert_plane))
    correlations = (
        crop_correlations(expert_plane, input_field, layout, active)
        if input_type != "centered_delta"
        else []
    )

    input_to_prompt_energy = float(
        (torch.abs(after_input_to_prompt) ** 2).sum().item()
    )
    transmitted_energy = float((torch.abs(after_prompt) ** 2).sum().item())
    expert_total_energy = float((torch.abs(expert_plane) ** 2).sum().item())

    row = {
        "prompt_cell_size": int(layout.prompt_cell_size),
        "prompt_fill_factor": (
            4.0
            * float(layout.prompt_cell_size) ** 2
            / float(layout.canvas_height * layout.canvas_width)
        ),
        "input_type": input_type,
        "amplitude_mode": amplitude_mode,
        "prompt_mode": prompt_mode,
        "input_to_prompt_m": prompt.input_to_prompt_m,
        "prompt_to_expert_m": propagators["prompt_to_expert"].distance_m,
        "focal_length_m": prompt.focal_length_m,
        **sampling,
        "expert_energy_E0": metric["E0_energy"],
        "expert_energy_E1": metric["E1_energy"],
        "expert_energy_E2": metric["E2_energy"],
        "expert_energy_E3": metric["E3_energy"],
        "expert_energy_ratio_E0": expert_ratios[0],
        "expert_energy_ratio_E1": expert_ratios[1],
        "expert_energy_ratio_E2": expert_ratios[2],
        "expert_energy_ratio_E3": expert_ratios[3],
        "outside_energy_ratio": metric["outside_energy_ratio"],
        "total_energy_after_input_to_prompt": input_to_prompt_energy,
        "total_transmitted_energy_after_prompt": transmitted_energy,
        "transmitted_energy_ratio": safe_ratio(
            transmitted_energy, input_to_prompt_energy
        ),
        "total_energy_at_expert_plane": expert_total_energy,
        "target_expert_energy_ratio": target_ratio,
        "max_wrong_expert_energy_ratio": max_wrong,
        "target_to_max_wrong_ratio": onehot_ratio,
        "pass_onehot_ratio_criterion": bool(
            math.isfinite(onehot_ratio) and onehot_ratio >= 10.0
        ),
        "point_source_centroid_error_px": (
            finite_mean(centroid_errors)
            if input_type == "centered_delta"
            else float("nan")
        ),
        "expert_core_centroid_error_px": (
            finite_mean(core_errors)
            if input_type != "centered_delta"
            else float("nan")
        ),
        "peak_location_y": peak_y,
        "peak_location_x": peak_x,
        "expected_expert_center_y": expected_y,
        "expected_expert_center_x": expected_x,
        "identity_stack_mean_drift_px": mean_drift,
        "identity_stack_max_drift_px": max_drift,
        "final_plane_outside_energy_ratio": final_outside_ratio,
        "edge_energy_ratio": edge_energy_ratio(final_image, edge_border),
        "normalized_correlation_with_expected_input": finite_mean(correlations),
        "crop_energy_concentration": (
            safe_ratio(target_ratio, sum(expert_ratios))
            if target is not None
            else 1.0 - metric["outside_energy_ratio"]
        ),
    }
    fields = {
        "expert_plane": expert_plane,
        "detector_plane": detector_plane,
        "layer_fields": layer_fields,
        "stack_traces": stack_traces,
        "metric": metric,
    }
    return row, fields


def add_sweep_overlay(ax, layout: FourExpertLayout, metric: Optional[Dict] = None):
    input_aperture = layout.input_aperture
    ax.add_patch(
        Rectangle(
            (input_aperture.x0, input_aperture.y0),
            input_aperture.width,
            input_aperture.height,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.0,
        )
    )
    for index, cell in enumerate(layout.prompt_cells):
        ax.add_patch(
            Rectangle(
                (cell.x0, cell.y0),
                cell.width,
                cell.height,
                fill=False,
                edgecolor="deepskyblue",
                linestyle=":",
                linewidth=1.1,
            )
        )
        expert = layout.experts[index]
        ax.add_patch(
            Rectangle(
                (expert.x0, expert.y0),
                expert.width,
                expert.height,
                fill=False,
                edgecolor="lime",
                linewidth=1.2,
            )
        )
        ax.scatter(
            [expert.center[1]], [expert.center[0]], c="lime", marker="x", s=32
        )
        if metric is not None:
            cy = metric.get(f"copy{index}_centroid_y")
            cx = metric.get(f"copy{index}_centroid_x")
            if cy is not None and cx is not None and math.isfinite(cy) and math.isfinite(cx):
                ax.scatter([cx], [cy], c="magenta", marker="+", s=38)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)


def plot_layout_comparison(layout: FourExpertLayout, path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.zeros(layout.canvas_shape), cmap="gray", vmin=0.0, vmax=1.0)
    add_sweep_overlay(ax, layout)
    ax.set_title(
        f"Prompt cells {layout.prompt_cell_size}px (blue dotted), "
        "experts 200px (green)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_field(
    field: torch.Tensor,
    layout: FourExpertLayout,
    metric: Dict,
    path: Path,
    title: str,
    dpi: int,
):
    image = intensity_2d(field).detach().cpu().numpy()
    display = np.log10(image / (image.max() + EPS) + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(display, cmap="inferno", origin="upper")
    add_sweep_overlay(ax, layout, metric)
    ax.set_title(
        f"{title}\nout={metric['outside_energy_ratio']:.3f}",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log10(I/Imax)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_prompt_map(
    value: torch.Tensor,
    layout: FourExpertLayout,
    path: Path,
    title: str,
    cmap: str,
    dpi: int,
    phase: bool = False,
    mask: Optional[torch.Tensor] = None,
):
    array = value.detach().cpu().numpy()
    if phase:
        array = np.remainder(array, 2.0 * math.pi)
    if mask is not None:
        array = np.ma.array(
            array,
            mask=mask.detach().cpu().numpy() <= 0.0,
        )
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        array,
        cmap=cmap,
        origin="upper",
        vmin=0.0 if phase else None,
        vmax=2.0 * math.pi if phase else None,
    )
    add_sweep_overlay(ax, layout)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_layer_trace(
    traces: Sequence[Dict[int, Tuple[float, float]]],
    layout: FourExpertLayout,
    path: Path,
    dpi: int,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    layers = np.arange(1, len(traces) + 1)
    for index in range(4):
        axes[0].plot(
            layers,
            [trace[index][0] for trace in traces],
            marker="o",
            label=f"E{index}",
        )
        axes[1].plot(
            layers,
            [trace[index][1] for trace in traces],
            marker="o",
            label=f"E{index}",
        )
        axes[0].axhline(layout.experts[index].center[0], color="gray", alpha=0.15)
        axes[1].axhline(layout.experts[index].center[1], color="gray", alpha=0.15)
    axes[0].set_ylabel("Centroid y [px]")
    axes[1].set_ylabel("Centroid x [px]")
    axes[1].set_xlabel("Identity expert layer")
    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"Layer centroid trace, prompt cell {layout.prompt_cell_size}px")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def aggregate_ranking(rows: Sequence[Dict], cell_sizes: Sequence[int]) -> List[Dict]:
    ranking = []
    for size in cell_sizes:
        subset = [
            row
            for row in rows
            if int(row["prompt_cell_size"]) == int(size)
            and row["prompt_mode"] == "lens_plus_grating"
        ]
        all_on = [row for row in subset if row["amplitude_mode"] == "all_on"]
        onehot = [
            row for row in subset if row["amplitude_mode"].startswith("onehot_E")
        ]
        delta_all_on = next(
            (
                row
                for row in all_on
                if row["input_type"] == "centered_delta"
            ),
            None,
        )
        if delta_all_on is None:
            raise RuntimeError(
                "centered_delta/all_on is required for alignment ranking."
            )
        onehot_ratios = [
            row["target_to_max_wrong_ratio"]
            for row in onehot
            if math.isfinite(float(row["target_to_max_wrong_ratio"]))
        ]
        mean_transmission = finite_mean(
            [row["transmitted_energy_ratio"] for row in all_on]
        )
        mean_outside = finite_mean([row["outside_energy_ratio"] for row in all_on])
        mean_final_outside = finite_mean(
            [row["final_plane_outside_energy_ratio"] for row in all_on]
        )
        min_onehot_ratio = min(onehot_ratios) if onehot_ratios else float("nan")
        point_error = float(delta_all_on["point_source_centroid_error_px"])
        drift = float(delta_all_on["identity_stack_mean_drift_px"])
        min_lens = float(delta_all_on["min_lens_period_px"])
        min_grating = float(delta_all_on["min_grating_period_px"])

        score_fill = float(delta_all_on["prompt_fill_factor"])
        score_transmission = min(max(mean_transmission, 0.0), 1.0)
        score_outside = 1.0 - min(max(mean_outside, 0.0), 1.0)
        score_alignment = math.exp(-max(point_error, 0.0) / 5.0)
        score_drift = math.exp(-max(drift, 0.0) / 10.0)
        score_routing = min(max(math.log10(max(min_onehot_ratio, 1.0)) / 2.0, 0.0), 1.0)
        score = (
            0.20 * score_fill
            + 0.20 * score_transmission
            + 0.20 * score_outside
            + 0.15 * score_alignment
            + 0.10 * score_drift
            + 0.15 * score_routing
        )
        hard_pass = bool(
            min_lens >= 5.0
            and min_onehot_ratio >= 10.0
            and point_error <= 5.0
            and drift <= 10.0
            and min_grating >= 8.0
        )
        ranking.append(
            {
                "prompt_cell_size": int(size),
                "prompt_fill_factor": score_fill,
                "mean_transmitted_energy_ratio_all_on": mean_transmission,
                "mean_expert_plane_outside_energy_ratio_all_on": mean_outside,
                "mean_final_plane_outside_energy_ratio_all_on": mean_final_outside,
                "point_source_centroid_error_px": point_error,
                "identity_stack_mean_drift_px": drift,
                "minimum_onehot_target_to_wrong_ratio": min_onehot_ratio,
                "min_lens_period_px": min_lens,
                "min_grating_period_px": min_grating,
                "sampling_warning_below_8px": bool(
                    min_lens < 8.0 or min_grating < 8.0
                ),
                "passes_hard_selection_rules": hard_pass,
                "score_fill": score_fill,
                "score_transmission": score_transmission,
                "score_low_outside": score_outside,
                "score_alignment": score_alignment,
                "score_drift": score_drift,
                "score_routing": score_routing,
                "weighted_score": score,
            }
        )

    baseline = next(row for row in ranking if row["prompt_cell_size"] == 200)
    for row in ranking:
        row["outside_reduced_vs_size_200"] = bool(
            row["mean_expert_plane_outside_energy_ratio_all_on"]
            <= baseline["mean_expert_plane_outside_energy_ratio_all_on"] + 1e-9
        )
        row["candidate_240_to_300"] = bool(
            240 <= row["prompt_cell_size"] <= 300
        )
        row["eligible_next_geometry"] = bool(
            row["candidate_240_to_300"]
            and row["passes_hard_selection_rules"]
            and row["outside_reduced_vs_size_200"]
        )
    return sorted(
        ranking,
        key=lambda row: (
            bool(row["eligible_next_geometry"]),
            bool(row["passes_hard_selection_rules"]),
            float(row["weighted_score"]),
        ),
        reverse=True,
    )


def recommendation_from_ranking(ranking: Sequence[Dict]) -> Dict:
    candidate_status = {
        str(size): next(
            (
                {
                    "eligible": bool(row["eligible_next_geometry"]),
                    "passes_hard_rules": bool(row["passes_hard_selection_rules"]),
                    "outside_reduced_vs_200": bool(row["outside_reduced_vs_size_200"]),
                    "weighted_score": row["weighted_score"],
                }
                for row in ranking
                if row["prompt_cell_size"] == size
            ),
            {"eligible": False, "not_tested": True},
        )
        for size in [240, 260, 280, 300]
    }
    eligible = [
        row
        for row in ranking
        if row["eligible_next_geometry"]
    ]
    if eligible:
        # The design rule prefers the largest passing aperture after requiring
        # it to reduce expert-plane outside energy relative to size 200.
        selected = max(eligible, key=lambda row: row["prompt_cell_size"])
        recommendation = int(selected["prompt_cell_size"])
        reason = (
            "Largest tested 240-300 cell that passes sampling, routing, "
            "alignment and drift checks while reducing outside energy versus 200."
        )
    else:
        recommendation = None
        reason = (
            "No tested size from 240 to 300 passes every hard rule and reduces "
            "outside energy versus the 200-pixel baseline."
        )
    return {
        "recommended_prompt_cell_size": recommendation,
        "reason": reason,
        "candidate_status_240_260_280_300": candidate_status,
    }


def plot_metric(
    ranking: Sequence[Dict],
    keys: Sequence[str],
    labels: Sequence[str],
    path: Path,
    ylabel: str,
    title: str,
    dpi: int,
    threshold: Optional[float] = None,
):
    ordered = sorted(ranking, key=lambda row: row["prompt_cell_size"])
    sizes = [row["prompt_cell_size"] for row in ordered]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in zip(keys, labels):
        ax.plot(sizes, [row[key] for row in ordered], marker="o", label=label)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prompt cell size [px]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(keys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_summary_markdown(
    path: Path,
    recommendation: Dict,
    ranking: Sequence[Dict],
    warnings: Sequence[str],
):
    lines = [
        "# Four-Expert Prompt Cell-Size Sweep",
        "",
        f"- recommended size: {recommendation['recommended_prompt_cell_size']}",
        f"- reason: {recommendation['reason']}",
        "",
        "## Candidate Status",
        "",
        "| size | hard rules | outside reduced | eligible | score |",
        "|---:|:---:|:---:|:---:|---:|",
    ]
    by_size = {row["prompt_cell_size"]: row for row in ranking}
    for size in [240, 260, 280, 300]:
        if size not in by_size:
            lines.append(f"| {size} | not tested | not tested | no | - |")
            continue
        row = by_size[size]
        lines.append(
            f"| {size} | {row['passes_hard_selection_rules']} | "
            f"{row['outside_reduced_vs_size_200']} | "
            f"{row['eligible_next_geometry']} | {row['weighted_score']:.4f} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_sizes = sorted(set(int(value) for value in args.cell_sizes))
    if 200 not in cell_sizes:
        raise ValueError("--cell-sizes must include 200 for baseline validation.")
    input_to_prompt_m, prompt_to_expert_m, focal_length_m = resolve_distances(args)

    base_layout = FourExpertLayout()
    base_layout.validate()
    propagators = {
        "input_to_prompt": make_propagator(
            base_layout,
            WAVELENGTH_M,
            PIXEL_SIZE_M,
            input_to_prompt_m,
            device,
        ),
        "prompt_to_expert": make_propagator(
            base_layout,
            WAVELENGTH_M,
            PIXEL_SIZE_M,
            prompt_to_expert_m,
            device,
        ),
        "inter_layer": make_propagator(
            base_layout,
            WAVELENGTH_M,
            PIXEL_SIZE_M,
            INTER_LAYER_M,
            device,
        ),
        "layer5_to_fc": make_propagator(
            base_layout,
            WAVELENGTH_M,
            PIXEL_SIZE_M,
            LAYER5_TO_FC_M,
            device,
        ),
        "fc_to_detector": make_propagator(
            base_layout,
            WAVELENGTH_M,
            PIXEL_SIZE_M,
            FC_TO_DETECTOR_M,
            device,
        ),
    }

    warnings = []
    inputs = {}
    for input_type in args.input_types:
        try:
            input_field = make_input_field(
                input_type=input_type,
                layout=base_layout,
                device=device,
                square_size=args.square_size,
                data_root=args.data_root,
                mnist_index=args.mnist_index,
            )
        except RuntimeError as exc:
            if input_type == "mnist_sample":
                warnings.append(f"Skipped mnist_sample: {exc}")
                continue
            raise
        inputs[input_type] = {
            "input_field": input_field,
            "after_input_to_prompt": propagators["input_to_prompt"](input_field),
        }
    if "centered_delta" not in inputs:
        raise ValueError("centered_delta must be included for alignment ranking.")
    if "centered_square" not in inputs:
        raise ValueError("centered_square must be included for representative plots.")

    all_rows = []
    reference_grating_period = None
    for size in cell_sizes:
        layout = FourExpertLayout(prompt_cell_size=size)
        layout.validate()
        prompt = MicrolensArrayPrompt(
            layout=layout,
            wavelength_m=WAVELENGTH_M,
            pixel_size_m=PIXEL_SIZE_M,
            focal_length_m=focal_length_m,
            input_to_prompt_m=input_to_prompt_m,
        ).to(device)
        sampling = sampling_metrics(layout, prompt, prompt_to_expert_m)
        if reference_grating_period is None:
            reference_grating_period = sampling["min_grating_period_px"]
        elif not math.isclose(
            sampling["min_grating_period_px"],
            reference_grating_period,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Grating period changed while only prompt_cell_size changed. "
                "Cell centers or steering geometry were modified unexpectedly."
            )
        if sampling["min_lens_period_px"] < 8.0:
            warnings.append(
                f"size {size}: min lens period "
                f"{sampling['min_lens_period_px']:.2f}px is below 8px."
            )
        if sampling["min_grating_period_px"] < 8.0:
            warnings.append(
                f"size {size}: min grating period "
                f"{sampling['min_grating_period_px']:.2f}px is below 8px."
            )

        if size in {200, 300}:
            plot_layout_comparison(
                layout,
                out_dir / f"prompt_cell_layout_size_{size}.png",
                args.plot_dpi,
            )
        prompt.set_amplitudes(AMPLITUDE_PATTERNS["all_on"])
        amplitude_map = prompt.amplitude_map()
        plot_prompt_map(
            amplitude_map,
            layout,
            out_dir / f"prompt_amplitude_size_{size}.png",
            f"Prompt amplitude, cell size {size}",
            "viridis",
            args.plot_dpi,
        )
        plot_prompt_map(
            prompt.phase_map("lens_plus_grating"),
            layout,
            out_dir / f"prompt_phase_wrapped_size_{size}.png",
            f"Lens + grating phase, cell size {size}",
            "twilight",
            args.plot_dpi,
            phase=True,
            mask=amplitude_map,
        )
        plot_prompt_map(
            prompt.lens_phase_map(),
            layout,
            out_dir / f"prompt_lens_phase_wrapped_size_{size}.png",
            f"Lens phase, cell size {size}",
            "twilight",
            args.plot_dpi,
            phase=True,
            mask=prompt.cell_masks.sum(dim=0),
        )
        plot_prompt_map(
            prompt.grating_phase_map(),
            layout,
            out_dir / f"prompt_grating_phase_wrapped_size_{size}.png",
            f"Grating phase, cell size {size}",
            "twilight",
            args.plot_dpi,
            phase=True,
            mask=prompt.cell_masks.sum(dim=0),
        )

        representative = {}
        prompt_modes = ["lens_plus_grating"]
        if args.include_negative_controls:
            prompt_modes.extend(["lens_only", "grating_only"])
        for input_type, input_payload in inputs.items():
            for prompt_mode in prompt_modes:
                amplitude_items = (
                    AMPLITUDE_PATTERNS.items()
                    if prompt_mode == "lens_plus_grating"
                    else [("all_on", AMPLITUDE_PATTERNS["all_on"])]
                )
                for amplitude_mode, amplitudes in amplitude_items:
                    row, fields = run_condition(
                        input_type=input_type,
                        input_field=input_payload["input_field"],
                        after_input_to_prompt=input_payload["after_input_to_prompt"],
                        prompt=prompt,
                        prompt_mode=prompt_mode,
                        amplitude_mode=amplitude_mode,
                        amplitudes=amplitudes,
                        layout=layout,
                        propagators=propagators,
                        sampling=sampling,
                        aperture_mode=args.aperture_mode,
                        edge_border=args.edge_border,
                    )
                    all_rows.append(row)
                    if (
                        input_type == "centered_square"
                        and prompt_mode == "lens_plus_grating"
                        and amplitude_mode in {"all_on", "onehot_E0"}
                    ):
                        representative[amplitude_mode] = fields

        all_on_fields = representative["all_on"]
        onehot_fields = representative["onehot_E0"]
        plot_field(
            all_on_fields["expert_plane"],
            layout,
            all_on_fields["metric"],
            out_dir / f"expert_plane_all_on_square_size_{size}.png",
            f"Expert plane all-on square, cell size {size}",
            args.plot_dpi,
        )
        plot_field(
            onehot_fields["expert_plane"],
            layout,
            onehot_fields["metric"],
            out_dir / f"expert_plane_onehot_E0_square_size_{size}.png",
            f"Expert plane onehot E0 square, cell size {size}",
            args.plot_dpi,
        )
        detector_metric, _ = compute_plane_metrics(
            all_on_fields["detector_plane"],
            plane_name="detector_plane",
            case_name="lens_plus_grating",
            layout=layout,
            expert_masks=layout.expert_masks(device=device),
            copy_masks=quadrant_masks(layout, device),
            edge_border=args.edge_border,
            amplitudes=AMPLITUDE_PATTERNS["all_on"],
        )
        plot_field(
            all_on_fields["detector_plane"],
            layout,
            detector_metric,
            out_dir / f"detector_plane_all_on_square_size_{size}.png",
            f"Detector plane all-on square, cell size {size}",
            args.plot_dpi,
        )
        plot_layer_trace(
            all_on_fields["stack_traces"],
            layout,
            out_dir / f"layer_stack_trace_size_{size}.png",
            args.plot_dpi,
        )
        print(
            f"size={size}: fill={layout.to_dict()['prompt_fill_factor']:.3f}, "
            f"lens_period={sampling['min_lens_period_px']:.2f}px, "
            f"grating_period={sampling['min_grating_period_px']:.2f}px"
        )

    ranking = aggregate_ranking(all_rows, cell_sizes)
    recommendation = recommendation_from_ranking(ranking)
    baseline = next(row for row in ranking if row["prompt_cell_size"] == 200)
    baseline_square_all_on = next(
        row
        for row in all_rows
        if row["prompt_cell_size"] == 200
        and row["input_type"] == "centered_square"
        and row["amplitude_mode"] == "all_on"
        and row["prompt_mode"] == "lens_plus_grating"
    )
    baseline_validation = {
        "point_source_error_below_5px": bool(
            baseline["point_source_centroid_error_px"] <= 5.0
        ),
        "point_source_error_matches_existing_test": bool(
            math.isclose(
                baseline["point_source_centroid_error_px"],
                1.2980509909666753,
                abs_tol=0.10,
            )
        ),
        "square_outside_energy_matches_existing_test": bool(
            math.isclose(
                baseline_square_all_on["outside_energy_ratio"],
                0.2585829794406891,
                abs_tol=0.02,
            )
        ),
        "onehot_ratio_above_10": bool(
            baseline["minimum_onehot_target_to_wrong_ratio"] >= 10.0
        ),
        "grating_period_matches_existing_geometry": bool(
            math.isclose(
                baseline["min_grating_period_px"],
                11.083532831537868,
                abs_tol=0.05,
            )
        ),
        "prompt_fill_factor_matches_200_baseline": bool(
            math.isclose(
                baseline["prompt_fill_factor"],
                4.0 * 200.0 ** 2 / 700.0 ** 2,
                abs_tol=1e-9,
            )
        ),
    }

    write_csv(out_dir / "sweep_summary.csv", all_rows)
    write_csv(out_dir / "ranking_table.csv", ranking)
    geometry_config = {
        "canvas_shape": [700, 700],
        "input_size": 200,
        "expert_size": 200,
        "expert_apertures": [item.to_dict() for item in base_layout.experts],
        "prompt_cell_sizes": cell_sizes,
        "prompt_cell_centers": [list(item.center) for item in base_layout.prompt_cells],
        "wavelength_m": WAVELENGTH_M,
        "pixel_size_m": PIXEL_SIZE_M,
        "distances_m": {
            "input_to_prompt": input_to_prompt_m,
            "prompt_to_expert": prompt_to_expert_m,
            "inter_layer": INTER_LAYER_M,
            "layer5_to_fc": LAYER5_TO_FC_M,
            "fc_to_detector": FC_TO_DETECTOR_M,
        },
        "focal_length_m": focal_length_m,
        "aperture_mode": args.aperture_mode,
        "gap_transmission": 0.0,
        "taper_enabled": False,
    }
    save_json(out_dir / "geometry_config.json", geometry_config)
    summary = {
        "geometry": geometry_config,
        "input_types_evaluated": list(inputs.keys()),
        "amplitude_modes": list(AMPLITUDE_PATTERNS.keys()),
        "prompt_modes": (
            ["lens_plus_grating", "lens_only", "grating_only"]
            if args.include_negative_controls
            else ["lens_plus_grating"]
        ),
        "baseline_size_200_validation": baseline_validation,
        "grating_period_invariance_check_passed": True,
        "score_definition": {
            "formula": (
                "0.20*fill + 0.20*transmission + 0.20*(1-outside) + "
                "0.15*exp(-point_error/5) + 0.10*exp(-drift/10) + "
                "0.15*clamp(log10(min_onehot_ratio)/2, 0, 1)"
            ),
            "hard_rejection_rules": {
                "min_lens_period_px": ">= 5",
                "min_grating_period_px": ">= 8",
                "minimum_onehot_target_to_wrong_ratio": ">= 10",
                "point_source_centroid_error_px": "<= 5",
                "identity_stack_mean_drift_px": "<= 10",
            },
            "selection_rule": (
                "Choose the largest 240-300 cell passing hard rules and "
                "reducing expert-plane outside energy versus size 200."
            ),
        },
        "ranking": ranking,
        "recommendation": recommendation,
        "warnings": warnings,
    }
    save_json(out_dir / "sweep_summary.json", summary)
    save_summary_markdown(
        out_dir / "sweep_summary.md",
        recommendation,
        ranking,
        warnings,
    )

    plot_metric(
        ranking,
        ["point_source_centroid_error_px", "identity_stack_mean_drift_px"],
        ["point-source error", "identity-stack drift"],
        out_dir / "metric_vs_cell_size.png",
        "Pixels",
        "Alignment and propagation stability",
        args.plot_dpi,
    )
    plot_metric(
        ranking,
        [
            "mean_expert_plane_outside_energy_ratio_all_on",
            "mean_final_plane_outside_energy_ratio_all_on",
        ],
        ["expert plane", "final plane"],
        out_dir / "outside_energy_vs_cell_size.png",
        "Outside energy ratio",
        "Outside energy versus prompt cell size",
        args.plot_dpi,
    )
    plot_metric(
        ranking,
        ["minimum_onehot_target_to_wrong_ratio"],
        ["minimum one-hot ratio"],
        out_dir / "onehot_ratio_vs_cell_size.png",
        "Target / max wrong",
        "Worst-case one-hot routing ratio",
        args.plot_dpi,
        threshold=10.0,
    )
    plot_metric(
        ranking,
        ["mean_transmitted_energy_ratio_all_on"],
        ["transmitted energy"],
        out_dir / "transmitted_energy_vs_cell_size.png",
        "After-prompt / before-prompt energy",
        "Prompt throughput versus cell size",
        args.plot_dpi,
    )
    plot_metric(
        ranking,
        ["min_lens_period_px"],
        ["lens period"],
        out_dir / "min_lens_period_vs_cell_size.png",
        "Pixels",
        "Minimum Cartesian lens-phase period",
        args.plot_dpi,
        threshold=8.0,
    )
    plot_metric(
        ranking,
        ["min_grating_period_px"],
        ["grating period"],
        out_dir / "min_grating_period_vs_cell_size.png",
        "Pixels",
        "Minimum grating period (must remain constant)",
        args.plot_dpi,
        threshold=8.0,
    )

    print("\nFour-expert prompt cell-size sweep complete")
    print(f"output directory: {out_dir}")
    print(f"evaluated inputs: {', '.join(inputs.keys())}")
    print(f"grating period invariant: {reference_grating_period:.2f}px")
    for size in [240, 260, 280, 300]:
        status = recommendation["candidate_status_240_260_280_300"][str(size)]
        print(f"candidate {size}: {status}")
    print(
        "recommended prompt cell size: "
        f"{recommendation['recommended_prompt_cell_size']}"
    )
    print(f"reason: {recommendation['reason']}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
