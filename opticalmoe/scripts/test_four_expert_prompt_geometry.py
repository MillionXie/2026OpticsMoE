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
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.optics.angular_spectrum import AngularSpectrumPropagator
from opticalmoe.optics.four_expert_geometry import Aperture, FourExpertLayout
from opticalmoe.optics.microlens_prompt import MicrolensArrayPrompt


EPS = 1e-12
PROMPT_MODES = [
    "lens_only",
    "lens_plus_grating",
    "grating_only",
    "identity_prompt",
]


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Standalone four-expert microlens prompt geometry verification."
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.20,
        help="Convenience value used for both object and image distance unless overridden.",
    )
    parser.add_argument("--input-to-prompt", type=float, default=None)
    parser.add_argument("--prompt-to-expert", type=float, default=None)
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Thin-lens focal length in meters. Defaults to s*s'/(s+s').",
    )
    parser.add_argument(
        "--input-type",
        default="centered_square",
        choices=["centered_square", "centered_delta", "mnist_sample"],
    )
    parser.add_argument("--amplitudes", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--phase-biases", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--aperture-mode", default="hard", choices=["hard", "transparent"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument(
        "--out-dir",
        default=f"runs/four_expert_prompt_geometry_{timestamp}",
    )
    parser.add_argument("--sweep-distances", action="store_true")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--mnist-index", type=int, default=0)
    parser.add_argument("--square-size", type=int, default=100)
    parser.add_argument("--edge-border", type=int, default=40)
    parser.add_argument("--outside-warning-threshold", type=float, default=0.35)
    parser.add_argument("--grating-period-warning-px", type=float, default=8.0)
    parser.add_argument("--plot-dpi", type=int, default=110)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


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
    if input_to_prompt_m <= 0.0 or prompt_to_expert_m <= 0.0:
        raise ValueError("Propagation distances must be positive.")
    focal_length_m = (
        input_to_prompt_m
        * prompt_to_expert_m
        / (input_to_prompt_m + prompt_to_expert_m)
        if args.focal_length is None
        else float(args.focal_length)
    )
    if focal_length_m <= 0.0:
        raise ValueError("focal_length must be positive.")
    return input_to_prompt_m, prompt_to_expert_m, focal_length_m


def make_propagator(
    layout: FourExpertLayout,
    wavelength_m: float,
    pixel_size_m: float,
    distance_m: float,
    device: torch.device,
) -> AngularSpectrumPropagator:
    return AngularSpectrumPropagator(
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
        grid_size=layout.canvas_shape,
        distance_m=float(distance_m),
        evanescent_mode="zero",
    ).to(device)


def make_input_field(
    input_type: str,
    layout: FourExpertLayout,
    device: torch.device,
    square_size: int,
    data_root: str,
    mnist_index: int,
) -> torch.Tensor:
    amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32)
    cy, cx = [int(value) for value in layout.canvas_center]

    if input_type == "centered_square":
        size = int(square_size)
        if size <= 0 or size > layout.input_size:
            raise ValueError("square_size must be within the 200 x 200 input aperture.")
        y0 = cy - size // 2
        x0 = cx - size // 2
        amplitude[y0 : y0 + size, x0 : x0 + size] = 1.0
    elif input_type == "centered_delta":
        amplitude[cy, cx] = 1.0
    elif input_type == "mnist_sample":
        try:
            from torchvision import datasets, transforms
            import torch.nn.functional as F

            dataset = datasets.MNIST(
                root=data_root,
                train=False,
                transform=transforms.ToTensor(),
                download=False,
            )
            image, _ = dataset[int(mnist_index) % len(dataset)]
            image = F.interpolate(
                image.unsqueeze(0),
                size=(layout.input_size, layout.input_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            aperture = layout.input_aperture
            amplitude[
                aperture.y0 : aperture.y1,
                aperture.x0 : aperture.x1,
            ] = image
        except Exception as exc:
            raise RuntimeError(
                "MNIST sample was requested but the dataset was not available locally. "
                "Download it with the existing dataset pipeline or use centered_square."
            ) from exc
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    return amplitude.unsqueeze(0).to(device=device, dtype=torch.complex64)


def intensity_2d(field_or_intensity: torch.Tensor) -> torch.Tensor:
    value = field_or_intensity
    if torch.is_complex(value):
        value = torch.abs(value.to(torch.complex64)) ** 2
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D intensity or [1,H,W] field, got {tuple(value.shape)}")
    return value.float()


def weighted_centroid(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    work = image if mask is None else image * mask
    total = work.sum()
    if float(total.item()) <= EPS:
        return float("nan"), float("nan")
    height, width = work.shape
    y = torch.arange(height, dtype=work.dtype, device=work.device).view(height, 1)
    x = torch.arange(width, dtype=work.dtype, device=work.device).view(1, width)
    cy = (work * y).sum() / total
    cx = (work * x).sum() / total
    return float(cy.item()), float(cx.item())


def core_centroid(
    image: torch.Tensor,
    mask: torch.Tensor,
    threshold_fraction: float = 0.20,
) -> Tuple[float, float]:
    """Centroid of the bright image core inside one spatial branch.

    A finite square or MNIST object creates coherent diffraction background
    across a quadrant. The ordinary energy centroid remains useful, but that
    background can pull it toward the canvas center even when the visible copy
    is correctly placed. This metric thresholds relative to the local peak and
    measures the position of the reconstructed image core.
    """

    masked = image * mask
    local_peak = masked.max()
    if float(local_peak.item()) <= EPS:
        return float("nan"), float("nan")
    bright_mask = mask * (masked >= local_peak * float(threshold_fraction)).to(mask.dtype)
    return weighted_centroid(image, bright_mask)


def peak_location(image: torch.Tensor) -> Tuple[int, int]:
    flat_index = int(torch.argmax(image).item())
    width = image.shape[1]
    return flat_index // width, flat_index % width


def edge_energy_ratio(image: torch.Tensor, border: int) -> float:
    border = max(1, min(int(border), min(image.shape) // 2))
    edge = torch.zeros_like(image, dtype=torch.bool)
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True
    return float((image[edge].sum() / (image.sum() + EPS)).item())


def quadrant_masks(
    layout: FourExpertLayout,
    device: torch.device,
) -> List[torch.Tensor]:
    cy, cx = [int(value) for value in layout.canvas_center]
    ranges = [
        (0, cy, 0, cx),
        (0, cy, cx, layout.canvas_width),
        (cy, layout.canvas_height, 0, cx),
        (cy, layout.canvas_height, cx, layout.canvas_width),
    ]
    masks = []
    for y0, y1, x0, x1 in ranges:
        mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        mask[y0:y1, x0:x1] = 1.0
        masks.append(mask)
    return masks


def compute_plane_metrics(
    field_or_intensity: torch.Tensor,
    plane_name: str,
    case_name: str,
    layout: FourExpertLayout,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    edge_border: int,
    amplitudes: Sequence[float],
) -> Tuple[Dict, List[Dict]]:
    image = intensity_2d(field_or_intensity)
    total = image.sum()
    expert_energies = torch.einsum("hw,khw->k", image, expert_masks)
    expert_sum = expert_energies.sum()
    outside = torch.clamp(total - expert_sum, min=0.0)
    centroid_y, centroid_x = weighted_centroid(image)
    peak_y, peak_x = peak_location(image)

    row = {
        "case_name": case_name,
        "plane_name": plane_name,
        "amplitudes": " ".join(f"{value:g}" for value in amplitudes),
        "total_energy": float(total.item()),
        "outside_energy": float(outside.item()),
        "outside_energy_ratio": float((outside / (total + EPS)).item()),
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "peak_y": peak_y,
        "peak_x": peak_x,
        "edge_energy_ratio": edge_energy_ratio(image, edge_border),
    }
    trace_rows = []
    for index, aperture in enumerate(layout.experts):
        ratio = float((expert_energies[index] / (total + EPS)).item())
        local_y, local_x = weighted_centroid(image, expert_masks[index])
        copy_y, copy_x = weighted_centroid(image, copy_masks[index])
        core_y, core_x = core_centroid(image, copy_masks[index])
        target_y, target_x = aperture.center
        local_error = euclidean_error(local_y, local_x, target_y, target_x)
        copy_error = euclidean_error(copy_y, copy_x, target_y, target_x)
        core_error = euclidean_error(core_y, core_x, target_y, target_x)
        row[f"E{index}_energy"] = float(expert_energies[index].item())
        row[f"E{index}_energy_ratio"] = ratio
        row[f"E{index}_centroid_y"] = local_y
        row[f"E{index}_centroid_x"] = local_x
        row[f"E{index}_centroid_error_px"] = local_error
        row[f"copy{index}_centroid_y"] = copy_y
        row[f"copy{index}_centroid_x"] = copy_x
        row[f"copy{index}_centroid_error_px"] = copy_error
        row[f"copy{index}_core_centroid_y"] = core_y
        row[f"copy{index}_core_centroid_x"] = core_x
        row[f"copy{index}_core_centroid_error_px"] = core_error
        trace_rows.append(
            {
                "case_name": case_name,
                "plane_name": plane_name,
                "expert": f"E{index}",
                "target_y": target_y,
                "target_x": target_x,
                "expert_centroid_y": local_y,
                "expert_centroid_x": local_x,
                "expert_centroid_error_px": local_error,
                "quadrant_copy_centroid_y": copy_y,
                "quadrant_copy_centroid_x": copy_x,
                "quadrant_copy_error_px": copy_error,
                "core_centroid_y": core_y,
                "core_centroid_x": core_x,
                "core_centroid_error_px": core_error,
                "expert_energy_ratio": ratio,
            }
        )
    return row, trace_rows


def euclidean_error(
    y: float,
    x: float,
    target_y: float,
    target_x: float,
) -> float:
    if not math.isfinite(y) or not math.isfinite(x):
        return float("nan")
    return math.sqrt((y - target_y) ** 2 + (x - target_x) ** 2)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def add_geometry_overlay(
    ax,
    layout: FourExpertLayout,
    metric: Optional[Dict] = None,
    show_prompt_labels: bool = False,
) -> None:
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
            label="input",
        )
    )
    for index, aperture in enumerate(layout.experts):
        ax.add_patch(
            Rectangle(
                (aperture.x0, aperture.y0),
                aperture.width,
                aperture.height,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )
        label = f"C{index}/E{index}" if show_prompt_labels else f"E{index}"
        ax.text(
            aperture.x0 + 5,
            aperture.y0 + 18,
            label,
            color="cyan",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
        )
        ax.scatter(
            [aperture.center[1]],
            [aperture.center[0]],
            marker="x",
            s=35,
            c="lime",
        )
        if metric is not None:
            cy = metric.get(f"copy{index}_centroid_y")
            cx = metric.get(f"copy{index}_centroid_x")
            if cy is not None and cx is not None and math.isfinite(cy) and math.isfinite(cx):
                ax.scatter([cx], [cy], marker="+", s=45, c="magenta")
    if metric is not None:
        ax.scatter(
            [metric["centroid_x"]],
            [metric["centroid_y"]],
            marker="+",
            s=70,
            c="red",
            label="global centroid",
        )
        ax.scatter(
            [metric["peak_x"]],
            [metric["peak_y"]],
            marker="x",
            s=55,
            c="yellow",
            label="peak",
        )
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)


def plot_intensity(
    field_or_intensity: torch.Tensor,
    path: Path,
    layout: FourExpertLayout,
    metric: Dict,
    title: str,
    dpi: int,
    overlay: bool = True,
) -> None:
    image = intensity_2d(field_or_intensity).detach().cpu().numpy()
    normalized = image / (image.max() + EPS)
    display = np.log10(normalized + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(display, cmap="inferno", origin="upper")
    if overlay:
        add_geometry_overlay(ax, layout, metric=metric, show_prompt_labels=True)
    ratios = ", ".join(
        f"E{index}={metric[f'E{index}_energy_ratio']:.3f}" for index in range(4)
    )
    ax.set_title(
        f"{title}\n{ratios}, outside={metric['outside_energy_ratio']:.3f}",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log10(I/Imax)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_phase(
    phase: torch.Tensor,
    amplitude_mask: torch.Tensor,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    wrapped = torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()
    mask = amplitude_mask.detach().cpu().numpy() <= 0.0
    display = np.ma.array(wrapped, mask=mask)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        display,
        cmap="twilight",
        origin="upper",
        vmin=0.0,
        vmax=2.0 * math.pi,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_scalar_map(
    value: torch.Tensor,
    path: Path,
    title: str,
    cmap: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(value.detach().cpu().numpy(), cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_layout(
    layout: FourExpertLayout,
    path: Path,
    prompt_cells: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.zeros(layout.canvas_shape), cmap="gray", vmin=0.0, vmax=1.0)
    input_aperture = layout.input_aperture
    ax.add_patch(
        Rectangle(
            (input_aperture.x0, input_aperture.y0),
            input_aperture.width,
            input_aperture.height,
            fill=False,
            edgecolor="yellow",
            linestyle="--",
            linewidth=1.4,
        )
    )
    ax.text(
        input_aperture.x0 + 5,
        input_aperture.y0 + 18,
        "input",
        color="yellow",
    )
    apertures = layout.prompt_cells if prompt_cells else layout.experts
    for index, aperture in enumerate(apertures):
        color = "cyan" if prompt_cells else "lime"
        ax.add_patch(
            Rectangle(
                (aperture.x0, aperture.y0),
                aperture.width,
                aperture.height,
                fill=False,
                edgecolor=color,
                linewidth=1.6,
            )
        )
        label = f"C{index}" if prompt_cells else f"E{index}"
        ax.text(aperture.x0 + 8, aperture.y0 + 22, label, color=color, fontsize=11)
        ax.scatter([aperture.center[1]], [aperture.center[0]], c=color, marker="x")
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_title("Prompt cell layout" if prompt_cells else "Expert aperture layout")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def find_metric(
    rows: Sequence[Dict],
    case_name: str,
    plane_name: str,
) -> Optional[Dict]:
    for row in rows:
        if row["case_name"] == case_name and row["plane_name"] == plane_name:
            return row
    return None


def copy_error_mean(row: Optional[Dict]) -> float:
    if row is None:
        return float("nan")
    values = [
        float(row[f"copy{index}_centroid_error_px"])
        for index in range(4)
        if math.isfinite(float(row[f"copy{index}_centroid_error_px"]))
    ]
    return float(np.mean(values)) if values else float("nan")


def core_error_mean(row: Optional[Dict]) -> float:
    if row is None:
        return float("nan")
    values = [
        float(row[f"copy{index}_core_centroid_error_px"])
        for index in range(4)
        if math.isfinite(float(row[f"copy{index}_core_centroid_error_px"]))
    ]
    return float(np.mean(values)) if values else float("nan")


def expert_energy_sum(row: Optional[Dict]) -> float:
    if row is None:
        return float("nan")
    return float(sum(float(row[f"E{index}_energy_ratio"]) for index in range(4)))


def run_prompt_cases(
    input_field: torch.Tensor,
    after_input_to_prompt: torch.Tensor,
    prompt: MicrolensArrayPrompt,
    prompt_to_expert: AngularSpectrumPropagator,
    layout: FourExpertLayout,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    args,
    out_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], List[Dict], List[Dict]]:
    fields = {}
    metric_rows = []
    trace_rows = []
    for mode in PROMPT_MODES:
        after_prompt = prompt(after_input_to_prompt, mode=mode)
        expert1 = prompt_to_expert(after_prompt)
        fields[f"{mode}_after_prompt"] = after_prompt
        fields[f"{mode}_expert1"] = expert1
        row, traces = compute_plane_metrics(
            expert1,
            plane_name="expert1_plane",
            case_name=mode,
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=args.amplitudes,
        )
        metric_rows.append(row)
        trace_rows.extend(traces)
        plot_intensity(
            expert1,
            out_dir / f"expert1_plane_intensity_{mode}.png",
            layout,
            row,
            f"Expert-1 plane: {mode}",
            args.plot_dpi,
        )
        if mode == "lens_plus_grating":
            plot_intensity(
                expert1,
                out_dir / "expert1_plane_overlay_lens_plus_grating.png",
                layout,
                row,
                "Lens + grating alignment overlay",
                args.plot_dpi,
            )
    return fields, metric_rows, trace_rows


def run_point_source_calibration(
    layout: FourExpertLayout,
    device: torch.device,
    prop_input_to_prompt: AngularSpectrumPropagator,
    prop_prompt_to_expert: AngularSpectrumPropagator,
    prompt: MicrolensArrayPrompt,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    args,
    out_dir: Path,
) -> Tuple[List[Dict], Dict]:
    """Calibrate the geometric mapping with a centered point source.

    This removes extended-object diffraction from the position measurement.
    The selected square/MNIST input is still used for all main visualizations
    and energy-routing checks.
    """

    point_input = make_input_field(
        input_type="centered_delta",
        layout=layout,
        device=device,
        square_size=args.square_size,
        data_root=args.data_root,
        mnist_index=args.mnist_index,
    )
    after_input = prop_input_to_prompt(point_input)
    rows = []
    for mode in ["lens_only", "lens_plus_grating", "grating_only"]:
        expert1 = prop_prompt_to_expert(prompt(after_input, mode=mode))
        row, _ = compute_plane_metrics(
            expert1,
            plane_name="expert1_plane",
            case_name=mode,
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=args.amplitudes,
        )
        rows.append(row)
        if mode == "lens_plus_grating":
            plot_intensity(
                expert1,
                out_dir / "point_source_calibration_lens_plus_grating.png",
                layout,
                row,
                "Point-source geometry calibration: lens + grating",
                args.plot_dpi,
            )

    write_csv(out_dir / "point_source_calibration.csv", rows)
    lens_only = find_metric(rows, "lens_only", "expert1_plane")
    lens_plus = find_metric(rows, "lens_plus_grating", "expert1_plane")
    grating_only = find_metric(rows, "grating_only", "expert1_plane")
    summary = {
        "lens_only_mean_error_px": copy_error_mean(lens_only),
        "lens_plus_grating_mean_error_px": copy_error_mean(lens_plus),
        "grating_only_mean_error_px": copy_error_mean(grating_only),
        "lens_plus_grating_expert_energy_ratio_sum": expert_energy_sum(lens_plus),
    }
    summary["passed"] = bool(
        summary["lens_plus_grating_mean_error_px"] < 10.0
        and summary["lens_only_mean_error_px"]
        > summary["lens_plus_grating_mean_error_px"]
        and summary["grating_only_mean_error_px"]
        > summary["lens_plus_grating_mean_error_px"]
    )
    return rows, summary


def apply_identity_expert(
    field: torch.Tensor,
    aperture_mode: str,
    union_mask: torch.Tensor,
) -> torch.Tensor:
    if aperture_mode == "hard":
        return field.to(torch.complex64) * union_mask.unsqueeze(0).to(torch.complex64)
    return field.to(torch.complex64)


def run_identity_stack(
    expert1_field: torch.Tensor,
    inter_layer: AngularSpectrumPropagator,
    layer5_to_fc: AngularSpectrumPropagator,
    fc_to_detector: AngularSpectrumPropagator,
    layout: FourExpertLayout,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    union_mask: torch.Tensor,
    args,
    out_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], List[Dict], List[Dict]]:
    fields = {}
    metric_rows = []
    trace_rows = []
    field = expert1_field

    for layer_index in range(1, 6):
        field = apply_identity_expert(field, args.aperture_mode, union_mask)
        plane_name = f"layer_{layer_index}_after_identity"
        fields[plane_name] = field
        row, traces = compute_plane_metrics(
            field,
            plane_name=plane_name,
            case_name="lens_plus_grating",
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=args.amplitudes,
        )
        metric_rows.append(row)
        trace_rows.extend(traces)
        plot_intensity(
            field,
            out_dir / f"layer_{layer_index}_after_identity_intensity.png",
            layout,
            row,
            f"Identity expert layer {layer_index}",
            args.plot_dpi,
        )
        if layer_index < 5:
            field = inter_layer(field)

    field = layer5_to_fc(field)
    after_fc_identity = field
    fields["after_global_fc_identity"] = after_fc_identity
    fc_row, fc_traces = compute_plane_metrics(
        after_fc_identity,
        plane_name="after_global_fc_identity",
        case_name="lens_plus_grating",
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        edge_border=args.edge_border,
        amplitudes=args.amplitudes,
    )
    metric_rows.append(fc_row)
    trace_rows.extend(fc_traces)
    plot_intensity(
        after_fc_identity,
        out_dir / "after_global_fc_identity_intensity.png",
        layout,
        fc_row,
        "After identity global FC mask",
        args.plot_dpi,
    )

    detector = fc_to_detector(after_fc_identity)
    fields["detector_plane"] = detector
    detector_row, detector_traces = compute_plane_metrics(
        detector,
        plane_name="detector_plane",
        case_name="lens_plus_grating",
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        edge_border=args.edge_border,
        amplitudes=args.amplitudes,
    )
    metric_rows.append(detector_row)
    trace_rows.extend(detector_traces)
    plot_intensity(
        detector,
        out_dir / "detector_plane_intensity.png",
        layout,
        detector_row,
        "Final detector plane",
        args.plot_dpi,
    )
    return fields, metric_rows, trace_rows


def amplitude_patterns(custom: Sequence[float]) -> Dict[str, List[float]]:
    return {
        "all_on": [1.0, 1.0, 1.0, 1.0],
        "onehot_E0": [1.0, 0.0, 0.0, 0.0],
        "onehot_E1": [0.0, 1.0, 0.0, 0.0],
        "onehot_E2": [0.0, 0.0, 1.0, 0.0],
        "onehot_E3": [0.0, 0.0, 0.0, 1.0],
        "custom": list(custom),
    }


def run_amplitude_routing(
    after_input_to_prompt: torch.Tensor,
    prompt: MicrolensArrayPrompt,
    prompt_to_expert: AngularSpectrumPropagator,
    layout: FourExpertLayout,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    args,
    out_dir: Path,
) -> Tuple[List[Dict], Dict]:
    original_amplitudes = prompt.amplitudes.detach().cpu().tolist()
    rows = []
    checks = {}
    patterns = amplitude_patterns(args.amplitudes)
    for pattern_name, values in patterns.items():
        prompt.set_amplitudes(values)
        field = prompt(after_input_to_prompt, mode="lens_plus_grating")
        expert1 = prompt_to_expert(field)
        row, _ = compute_plane_metrics(
            expert1,
            plane_name="expert1_plane",
            case_name=pattern_name,
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=values,
        )
        row["pattern"] = pattern_name
        rows.append(row)
        if pattern_name.startswith("onehot_E"):
            index = int(pattern_name[-1])
            target_ratio = row[f"E{index}_energy_ratio"]
            other_ratios = [
                row[f"E{other}_energy_ratio"]
                for other in range(4)
                if other != index
            ]
            checks[pattern_name] = {
                "target_ratio": target_ratio,
                "max_other_ratio": max(other_ratios),
                "passed": bool(target_ratio > max(other_ratios)),
            }
        elif pattern_name == "all_on":
            ratios = [row[f"E{index}_energy_ratio"] for index in range(4)]
            checks[pattern_name] = {
                "ratios": ratios,
                "passed": bool(all(value > 1e-5 for value in ratios)),
            }

    prompt.set_amplitudes(original_amplitudes)
    write_csv(out_dir / "amplitude_routing.csv", rows)
    plot_amplitude_routing(rows, out_dir / "amplitude_routing_energy_bar.png", args.plot_dpi)
    return rows, checks


def plot_amplitude_routing(rows: Sequence[Dict], path: Path, dpi: int) -> None:
    labels = [row["pattern"] for row in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))
    for index in range(4):
        values = [row[f"E{index}_energy_ratio"] for row in rows]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=f"E{index}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Energy / total")
    ax.set_title("Amplitude routing at expert-1 plane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_centroid_trace(
    trace_rows: Sequence[Dict],
    path: Path,
    dpi: int,
) -> None:
    layer_names = [
        "layer_1_after_identity",
        "layer_2_after_identity",
        "layer_3_after_identity",
        "layer_4_after_identity",
        "layer_5_after_identity",
        "after_global_fc_identity",
        "detector_plane",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for expert_index in range(4):
        expert_name = f"E{expert_index}"
        subset = {
            row["plane_name"]: row
            for row in trace_rows
            if row["case_name"] == "lens_plus_grating"
            and row["expert"] == expert_name
            and row["plane_name"] in layer_names
        }
        xs = []
        ys = []
        x_values = []
        for plane_index, plane_name in enumerate(layer_names):
            if plane_name in subset:
                xs.append(plane_index)
                ys.append(subset[plane_name]["expert_centroid_y"])
                x_values.append(subset[plane_name]["expert_centroid_x"])
        axes[0].plot(xs, ys, marker="o", label=expert_name)
        axes[1].plot(xs, x_values, marker="o", label=expert_name)
    axes[0].set_ylabel("centroid y [px]")
    axes[1].set_ylabel("centroid x [px]")
    axes[1].set_xticks(range(len(layer_names)))
    axes[1].set_xticklabels(layer_names, rotation=25, ha="right")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(ncol=4)
    fig.suptitle("Centroid trace: lens + grating")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_expert_energy_ratios(
    metric_rows: Sequence[Dict],
    path: Path,
    dpi: int,
) -> None:
    selected_planes = [
        "expert1_plane",
        "layer_1_after_identity",
        "layer_3_after_identity",
        "layer_5_after_identity",
        "detector_plane",
    ]
    rows = []
    for plane_name in selected_planes:
        row = find_metric(metric_rows, "lens_plus_grating", plane_name)
        if row is not None:
            rows.append(row)
    x = np.arange(len(rows))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))
    for expert_index in range(4):
        values = [row[f"E{expert_index}_energy_ratio"] for row in rows]
        ax.bar(
            x + (expert_index - 1.5) * width,
            values,
            width=width,
            label=f"E{expert_index}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([row["plane_name"] for row in rows], rotation=20)
    ax.set_ylabel("Energy / total")
    ax.set_title("Expert energy ratios through identity stack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def drift_summary(trace_rows: Sequence[Dict]) -> Dict:
    layer_names = [f"layer_{index}_after_identity" for index in range(1, 6)]
    per_expert = {}
    all_step_drifts = []
    for expert_index in range(4):
        expert_name = f"E{expert_index}"
        rows = {
            row["plane_name"]: row
            for row in trace_rows
            if row["case_name"] == "lens_plus_grating"
            and row["expert"] == expert_name
        }
        points = []
        for name in layer_names:
            if name in rows:
                points.append(
                    (
                        rows[name]["expert_centroid_y"],
                        rows[name]["expert_centroid_x"],
                    )
                )
        drifts = []
        for previous, current in zip(points[:-1], points[1:]):
            if all(math.isfinite(value) for value in previous + current):
                drifts.append(
                    math.sqrt(
                        (current[0] - previous[0]) ** 2
                        + (current[1] - previous[1]) ** 2
                    )
                )
        per_expert[expert_name] = {
            "per_step_drift_px": drifts,
            "mean_drift_px": float(np.mean(drifts)) if drifts else float("nan"),
            "max_drift_px": float(np.max(drifts)) if drifts else float("nan"),
        }
        all_step_drifts.extend(drifts)
    return {
        "per_expert": per_expert,
        "overall_mean_drift_px": (
            float(np.mean(all_step_drifts)) if all_step_drifts else float("nan")
        ),
        "overall_max_drift_px": (
            float(np.max(all_step_drifts)) if all_step_drifts else float("nan")
        ),
    }


def run_distance_sweep(
    layout: FourExpertLayout,
    wavelength_m: float,
    pixel_size_m: float,
    expert_masks: torch.Tensor,
    copy_masks: Sequence[torch.Tensor],
    device: torch.device,
    args,
    out_dir: Path,
) -> List[Dict]:
    # A point source gives an unambiguous geometric centroid at every distance.
    input_field = make_input_field(
        input_type="centered_delta",
        layout=layout,
        device=device,
        square_size=args.square_size,
        data_root=args.data_root,
        mnist_index=args.mnist_index,
    )
    rows = []
    for distance_m in [0.10, 0.15, 0.20]:
        focal_length_m = distance_m / 2.0
        prop_input = make_propagator(
            layout,
            wavelength_m,
            pixel_size_m,
            distance_m,
            device,
        )
        prop_expert = make_propagator(
            layout,
            wavelength_m,
            pixel_size_m,
            distance_m,
            device,
        )
        prompt = MicrolensArrayPrompt(
            layout=layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            focal_length_m=focal_length_m,
            input_to_prompt_m=distance_m,
            amplitudes=[1.0, 1.0, 1.0, 1.0],
        ).to(device)
        after_input = prop_input(input_field)
        expert1 = prop_expert(prompt(after_input, mode="lens_plus_grating"))
        metric, _ = compute_plane_metrics(
            expert1,
            plane_name="expert1_plane",
            case_name=f"distance_{distance_m:.2f}m",
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=[1.0, 1.0, 1.0, 1.0],
        )
        reports = prompt.report(prompt_to_expert_m=distance_m)
        for expert_index, report in enumerate(reports):
            rows.append(
                {
                    "distance_m": distance_m,
                    "focal_length_m": focal_length_m,
                    "expert": f"E{expert_index}",
                    "grating_period_x_px": report["grating_period_x_px"],
                    "grating_period_y_px": report["grating_period_y_px"],
                    "expert_energy_ratio": metric[f"E{expert_index}_energy_ratio"],
                    "copy_centroid_y": metric[f"copy{expert_index}_centroid_y"],
                    "copy_centroid_x": metric[f"copy{expert_index}_centroid_x"],
                    "copy_centroid_error_px": metric[
                        f"copy{expert_index}_centroid_error_px"
                    ],
                    "core_centroid_error_px": metric[
                        f"copy{expert_index}_core_centroid_error_px"
                    ],
                    "outside_energy_ratio": metric["outside_energy_ratio"],
                }
            )
    write_csv(out_dir / "distance_sweep.csv", rows)
    plot_distance_sweep(rows, out_dir / "distance_sweep.png", args.plot_dpi)
    return rows


def plot_distance_sweep(rows: Sequence[Dict], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for expert_index in range(4):
        expert = f"E{expert_index}"
        subset = [row for row in rows if row["expert"] == expert]
        distances = [row["distance_m"] for row in subset]
        errors = [row["copy_centroid_error_px"] for row in subset]
        ratios = [row["expert_energy_ratio"] for row in subset]
        axes[0].plot(distances, errors, marker="o", label=expert)
        axes[1].plot(distances, ratios, marker="o", label=expert)
    axes[0].set_xlabel("s = s' [m]")
    axes[0].set_ylabel("Copy centroid error [px]")
    axes[1].set_xlabel("s = s' [m]")
    axes[1].set_ylabel("Expert energy / total")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Distance sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def summarize_distance_sweep(
    rows: Sequence[Dict],
    warning_period_px: float,
    warning_outside_ratio: float,
) -> Tuple[List[Dict], List[str]]:
    summaries = []
    warnings = []
    distances = sorted({float(row["distance_m"]) for row in rows})
    for distance_m in distances:
        subset = [row for row in rows if float(row["distance_m"]) == distance_m]
        mean_error = float(np.mean([row["copy_centroid_error_px"] for row in subset]))
        mean_energy = float(np.mean([row["expert_energy_ratio"] for row in subset]))
        outside_ratio = float(np.mean([row["outside_energy_ratio"] for row in subset]))
        finite_periods = []
        for row in subset:
            for key in ["grating_period_x_px", "grating_period_y_px"]:
                period = float(row[key])
                if math.isfinite(period):
                    finite_periods.append(period)
        minimum_period = min(finite_periods) if finite_periods else float("inf")
        passed = bool(
            mean_error < 10.0
            and outside_ratio <= warning_outside_ratio
            and minimum_period >= warning_period_px
        )
        summaries.append(
            {
                "distance_m": distance_m,
                "focal_length_m": distance_m / 2.0,
                "mean_centroid_error_px": mean_error,
                "mean_expert_energy_ratio": mean_energy,
                "outside_energy_ratio": outside_ratio,
                "minimum_grating_period_px": minimum_period,
                "sampling_and_alignment_passed": passed,
            }
        )
        if minimum_period < warning_period_px:
            warnings.append(
                f"Distance sweep s=s'={distance_m:.2f}m uses a minimum grating "
                f"period of {minimum_period:.2f}px, below the "
                f"{warning_period_px:.1f}px sampling warning threshold."
            )
        if mean_error >= 10.0:
            warnings.append(
                f"Distance sweep s=s'={distance_m:.2f}m has mean point-source "
                f"centroid error {mean_error:.2f}px. This sampled ASM geometry "
                "should not be treated as aligned without further calibration."
            )
        if outside_ratio > warning_outside_ratio:
            warnings.append(
                f"Distance sweep s=s'={distance_m:.2f}m has outside energy "
                f"ratio {outside_ratio:.3f}, above {warning_outside_ratio:.3f}."
            )
    return summaries, warnings


def write_distance_sweep_markdown(path: Path, summaries: Sequence[Dict]) -> None:
    lines = [
        "# Distance Sweep",
        "",
        "The sweep uses a centered point source so centroid error measures geometry "
        "without extended-object diffraction background.",
        "",
        "| s=s' (m) | f (m) | min period (px) | mean error (px) | outside ratio | status |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        status = "PASS" if row["sampling_and_alignment_passed"] else "CHECK"
        lines.append(
            f"| {row['distance_m']:.2f} | {row['focal_length_m']:.3f} | "
            f"{row['minimum_grating_period_px']:.2f} | "
            f"{row['mean_centroid_error_px']:.2f} | "
            f"{row['outside_energy_ratio']:.3f} | {status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_acceptance_summary(
    metric_rows: Sequence[Dict],
    amplitude_checks: Dict,
    drift: Dict,
    point_source_calibration: Dict,
) -> Dict:
    lens_only = find_metric(metric_rows, "lens_only", "expert1_plane")
    lens_plus = find_metric(metric_rows, "lens_plus_grating", "expert1_plane")
    grating_only = find_metric(metric_rows, "grating_only", "expert1_plane")
    lens_only_error = copy_error_mean(lens_only)
    lens_plus_error = copy_error_mean(lens_plus)
    grating_only_error = copy_error_mean(grating_only)
    lens_plus_energy_sum = expert_energy_sum(lens_plus)
    all_on_nonzero = bool(amplitude_checks.get("all_on", {}).get("passed", False))
    onehot_pass = all(
        bool(amplitude_checks.get(f"onehot_E{index}", {}).get("passed", False))
        for index in range(4)
    )
    lens_plus_energy_nonzero = False
    if lens_plus is not None:
        lens_plus_energy_nonzero = all(
            lens_plus[f"E{index}_energy_ratio"] > 1e-5 for index in range(4)
        )
    checks = {
        "lens_only_mean_copy_error_px": lens_only_error,
        "lens_plus_grating_mean_copy_error_px": lens_plus_error,
        "grating_only_mean_copy_error_px": grating_only_error,
        "lens_plus_grating_expert_energy_ratio_sum": lens_plus_energy_sum,
        "point_source_lens_plus_grating_mean_error_px": point_source_calibration[
            "lens_plus_grating_mean_error_px"
        ],
        "point_source_geometry_passed": bool(point_source_calibration["passed"]),
        "lens_only_overshoots_more_than_corrected": bool(
            math.isfinite(lens_only_error)
            and math.isfinite(lens_plus_error)
            and lens_only_error > lens_plus_error
        ),
        "lens_plus_grating_improves_over_grating_only": bool(
            math.isfinite(grating_only_error)
            and math.isfinite(lens_plus_error)
            and lens_plus_error < grating_only_error
        ),
        "lens_plus_grating_all_experts_nonzero": lens_plus_energy_nonzero,
        "all_on_all_experts_nonzero": all_on_nonzero,
        "all_onehot_routes_dominant": onehot_pass,
        "identity_stack_mean_centroid_drift_px": drift["overall_mean_drift_px"],
        "identity_stack_max_centroid_drift_px": drift["overall_max_drift_px"],
        "identity_stack_drift_small": bool(
            math.isfinite(drift["overall_mean_drift_px"])
            and drift["overall_mean_drift_px"] < 10.0
        ),
    }
    checks["overall_passed"] = bool(
        checks["point_source_geometry_passed"]
        and checks["lens_only_overshoots_more_than_corrected"]
        and checks["lens_plus_grating_improves_over_grating_only"]
        and checks["lens_plus_grating_all_experts_nonzero"]
        and checks["all_on_all_experts_nonzero"]
        and checks["all_onehot_routes_dominant"]
        and checks["identity_stack_drift_small"]
    )
    return checks


def write_summary_markdown(
    path: Path,
    geometry: Dict,
    acceptance: Dict,
    warnings: Sequence[str],
) -> None:
    lines = [
        "# Four-Expert Prompt Geometry Verification",
        "",
        "## Geometry",
        "",
        f"- canvas: {geometry['layout']['canvas_shape']}",
        f"- input type: {geometry['input_type']}",
        f"- input-to-prompt: {geometry['distances_m']['input_to_prompt']:.3f} m",
        f"- prompt-to-expert: {geometry['distances_m']['prompt_to_expert1']:.3f} m",
        f"- focal length: {geometry['focal_length_m']:.3f} m",
        f"- magnification: {geometry['magnification']:.3f}",
        f"- aperture mode: {geometry['aperture_mode']}",
        "",
        "## Acceptance Checks",
        "",
        f"- overall status: {'PASS' if acceptance['overall_passed'] else 'FAIL'}",
        "- note: selected-input quadrant centroids are background-sensitive; "
        "the point-source error is the strict geometry pass/fail metric.",
    ]
    for key, value in acceptance.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- none")
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

    wavelength_m = 532e-9
    pixel_size_m = 8e-6
    inter_layer_m = 0.05
    layer5_to_fc_m = 0.05
    fc_to_detector_m = 0.05
    input_to_prompt_m, prompt_to_expert_m, focal_length_m = resolve_distances(args)

    layout = FourExpertLayout()
    layout.validate()
    expert_masks = layout.expert_masks(device=device)
    union_mask = layout.expert_union_mask(device=device)
    copy_masks = quadrant_masks(layout, device)

    input_field = make_input_field(
        input_type=args.input_type,
        layout=layout,
        device=device,
        square_size=args.square_size,
        data_root=args.data_root,
        mnist_index=args.mnist_index,
    )
    prompt = MicrolensArrayPrompt(
        layout=layout,
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
        focal_length_m=focal_length_m,
        input_to_prompt_m=input_to_prompt_m,
        amplitudes=args.amplitudes,
        phase_biases=args.phase_biases,
    ).to(device)

    prop_input_to_prompt = make_propagator(
        layout,
        wavelength_m,
        pixel_size_m,
        input_to_prompt_m,
        device,
    )
    prop_prompt_to_expert = make_propagator(
        layout,
        wavelength_m,
        pixel_size_m,
        prompt_to_expert_m,
        device,
    )
    prop_inter_layer = make_propagator(
        layout,
        wavelength_m,
        pixel_size_m,
        inter_layer_m,
        device,
    )
    prop_layer5_to_fc = make_propagator(
        layout,
        wavelength_m,
        pixel_size_m,
        layer5_to_fc_m,
        device,
    )
    prop_fc_to_detector = make_propagator(
        layout,
        wavelength_m,
        pixel_size_m,
        fc_to_detector_m,
        device,
    )

    after_input_to_prompt = prop_input_to_prompt(input_field)
    prompt_amplitude = prompt.amplitude_map()
    selected_phase = prompt.phase_map("lens_plus_grating")
    after_prompt_selected = prompt(after_input_to_prompt, mode="lens_plus_grating")

    plot_layout(
        layout,
        out_dir / "prompt_cell_layout.png",
        prompt_cells=True,
        dpi=args.plot_dpi,
    )
    plot_layout(
        layout,
        out_dir / "expert_aperture_layout.png",
        prompt_cells=False,
        dpi=args.plot_dpi,
    )
    plot_scalar_map(
        prompt_amplitude,
        out_dir / "prompt_amplitude.png",
        "Prompt scalar amplitude",
        "viridis",
        args.plot_dpi,
    )
    plot_phase(
        selected_phase,
        prompt_amplitude,
        out_dir / "prompt_phase_wrapped.png",
        "Lens + grating prompt phase [0, 2pi)",
        args.plot_dpi,
    )
    plot_phase(
        prompt.lens_phase_map(),
        prompt.cell_masks.sum(dim=0),
        out_dir / "prompt_lens_phase_wrapped.png",
        "Local thin-lens phase [0, 2pi)",
        args.plot_dpi,
    )
    plot_phase(
        prompt.grating_phase_map(),
        prompt.cell_masks.sum(dim=0),
        out_dir / "prompt_grating_phase_wrapped.png",
        "Local grating/prism phase [0, 2pi)",
        args.plot_dpi,
    )

    metric_rows = []
    trace_rows = []
    for plane_name, case_name, field, file_name, title in [
        (
            "input_plane",
            "input",
            input_field,
            "input_plane_intensity.png",
            "Input plane",
        ),
        (
            "after_input_to_prompt",
            "input",
            after_input_to_prompt,
            "after_input_to_prompt_intensity.png",
            "After input-to-prompt propagation",
        ),
        (
            "after_prompt",
            "lens_plus_grating",
            after_prompt_selected,
            "after_prompt_intensity.png",
            "After lens + grating prompt",
        ),
    ]:
        row, traces = compute_plane_metrics(
            field,
            plane_name=plane_name,
            case_name=case_name,
            layout=layout,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            edge_border=args.edge_border,
            amplitudes=args.amplitudes,
        )
        metric_rows.append(row)
        trace_rows.extend(traces)
        plot_intensity(
            field,
            out_dir / file_name,
            layout,
            row,
            title,
            args.plot_dpi,
        )

    case_fields, case_metrics, case_traces = run_prompt_cases(
        input_field=input_field,
        after_input_to_prompt=after_input_to_prompt,
        prompt=prompt,
        prompt_to_expert=prop_prompt_to_expert,
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        args=args,
        out_dir=out_dir,
    )
    metric_rows.extend(case_metrics)
    trace_rows.extend(case_traces)

    _, stack_metrics, stack_traces = run_identity_stack(
        expert1_field=case_fields["lens_plus_grating_expert1"],
        inter_layer=prop_inter_layer,
        layer5_to_fc=prop_layer5_to_fc,
        fc_to_detector=prop_fc_to_detector,
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        union_mask=union_mask,
        args=args,
        out_dir=out_dir,
    )
    metric_rows.extend(stack_metrics)
    trace_rows.extend(stack_traces)

    amplitude_rows, amplitude_checks = run_amplitude_routing(
        after_input_to_prompt=after_input_to_prompt,
        prompt=prompt,
        prompt_to_expert=prop_prompt_to_expert,
        layout=layout,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        args=args,
        out_dir=out_dir,
    )
    point_source_rows, point_source_calibration = run_point_source_calibration(
        layout=layout,
        device=device,
        prop_input_to_prompt=prop_input_to_prompt,
        prop_prompt_to_expert=prop_prompt_to_expert,
        prompt=prompt,
        expert_masks=expert_masks,
        copy_masks=copy_masks,
        args=args,
        out_dir=out_dir,
    )
    drift = drift_summary(trace_rows)
    acceptance = build_acceptance_summary(
        metric_rows,
        amplitude_checks,
        drift,
        point_source_calibration,
    )

    plot_centroid_trace(
        trace_rows,
        out_dir / "centroid_trace_lens_plus_grating.png",
        args.plot_dpi,
    )
    plot_expert_energy_ratios(
        metric_rows,
        out_dir / "expert_energy_ratios_bar.png",
        args.plot_dpi,
    )

    distance_sweep_rows = []
    distance_sweep_summary = []
    distance_sweep_warnings = []
    if args.sweep_distances:
        distance_sweep_rows = run_distance_sweep(
            layout=layout,
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            expert_masks=expert_masks,
            copy_masks=copy_masks,
            device=device,
            args=args,
            out_dir=out_dir,
        )
        distance_sweep_summary, distance_sweep_warnings = summarize_distance_sweep(
            distance_sweep_rows,
            warning_period_px=args.grating_period_warning_px,
            warning_outside_ratio=args.outside_warning_threshold,
        )
        write_distance_sweep_markdown(
            out_dir / "distance_sweep_summary.md",
            distance_sweep_summary,
        )

    write_csv(out_dir / "energy_ratios.csv", metric_rows)
    write_csv(out_dir / "centroid_trace.csv", trace_rows)

    cell_reports = prompt.report(prompt_to_expert_m=prompt_to_expert_m)
    warnings = []
    for report in cell_reports:
        for axis in ["x", "y"]:
            period = report[f"grating_period_{axis}_px"]
            if math.isfinite(period) and period < args.grating_period_warning_px:
                warnings.append(
                    f"{report['cell']} grating period {axis}={period:.2f}px is below "
                    f"{args.grating_period_warning_px:.1f}px; phase sampling may be unreliable."
                )

    expert1_lpg = find_metric(metric_rows, "lens_plus_grating", "expert1_plane")
    detector_metric = find_metric(metric_rows, "lens_plus_grating", "detector_plane")
    for label, row in [
        ("expert-1 plane", expert1_lpg),
        ("detector plane", detector_metric),
    ]:
        if row is not None and row["outside_energy_ratio"] > args.outside_warning_threshold:
            warnings.append(
                f"High outside energy at {label}: {row['outside_energy_ratio']:.3f} "
                f"> {args.outside_warning_threshold:.3f}."
            )
    warnings.extend(distance_sweep_warnings)

    magnification = prompt_to_expert_m / input_to_prompt_m
    geometry_summary = {
        "device": str(device),
        "input_type": args.input_type,
        "amplitudes": list(args.amplitudes),
        "phase_biases": list(args.phase_biases),
        "aperture_mode": args.aperture_mode,
        "wavelength_m": wavelength_m,
        "pixel_size_m": pixel_size_m,
        "layout": layout.to_dict(),
        "distances_m": {
            "input_to_prompt": input_to_prompt_m,
            "prompt_to_expert1": prompt_to_expert_m,
            "inter_layer": inter_layer_m,
            "layer5_to_fc": layer5_to_fc_m,
            "fc_to_detector": fc_to_detector_m,
        },
        "focal_length_m": focal_length_m,
        "magnification": magnification,
        "thin_lens_equation_residual": (
            1.0 / focal_length_m
            - 1.0 / input_to_prompt_m
            - 1.0 / prompt_to_expert_m
        ),
        "cell_reports": cell_reports,
        "amplitude_routing_checks": amplitude_checks,
        "point_source_calibration": point_source_calibration,
        "point_source_calibration_rows": point_source_rows,
        "identity_stack_drift": drift,
        "acceptance": acceptance,
        "warnings": warnings,
        "distance_sweep_enabled": bool(args.sweep_distances),
        "distance_sweep_rows": distance_sweep_rows,
        "distance_sweep_summary": distance_sweep_summary,
    }
    save_json(out_dir / "geometry_summary.json", geometry_summary)
    write_summary_markdown(
        out_dir / "summary.md",
        geometry_summary,
        acceptance,
        warnings,
    )

    print("\nFour-expert prompt geometry verification")
    print(f"output directory: {out_dir}")
    print(
        f"s={input_to_prompt_m:.3f} m, s'={prompt_to_expert_m:.3f} m, "
        f"f={focal_length_m:.3f} m, M={magnification:.3f}"
    )
    for report in cell_reports:
        print(
            f"{report['cell']}: theta_x={report['theta_x_deg']:.4f} deg, "
            f"theta_y={report['theta_y_deg']:.4f} deg, "
            f"period_x={report['grating_period_x_px']:.2f}px, "
            f"period_y={report['grating_period_y_px']:.2f}px"
        )
    print(
        "mean copy error [px] | "
        f"lens_only={acceptance['lens_only_mean_copy_error_px']:.2f}, "
        f"lens+grating={acceptance['lens_plus_grating_mean_copy_error_px']:.2f}, "
        f"grating_only={acceptance['grating_only_mean_copy_error_px']:.2f}"
    )
    print(
        "point-source lens+grating calibration error: "
        f"{acceptance['point_source_lens_plus_grating_mean_error_px']:.2f}px"
    )
    print(
        f"identity-stack mean centroid drift: "
        f"{acceptance['identity_stack_mean_centroid_drift_px']:.2f}px"
    )
    print(
        f"one-hot routing checks passed: "
        f"{acceptance['all_onehot_routes_dominant']}"
    )
    print(f"overall geometry status: {'PASS' if acceptance['overall_passed'] else 'FAIL'}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")


if __name__ == "__main__":
    main()
