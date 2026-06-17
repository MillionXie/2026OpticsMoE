import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from opticalmoe.utils.seed import set_seed
from opticalmoe.utils.units import cm_to_m, nm_to_m, um_to_m


EPS = 1e-12
EXPERT_IDS = [
    "E00",
    "E01",
    "E02",
    "E10",
    "E11",
    "E12",
    "E20",
    "E21",
    "E22",
]


@dataclass(frozen=True)
class Aperture:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.y0 + self.y1) / 2.0, (self.x0 + self.x1) / 2.0)

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def contains_with_margin(self, y: float, x: float, margin: float) -> bool:
        return (
            self.y0 - margin <= y <= self.y1 + margin
            and self.x0 - margin <= x <= self.x1 + margin
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "y0": self.y0,
            "y1": self.y1,
            "x0": self.x0,
            "x1": self.x1,
            "center": list(self.center),
            "height": self.height,
            "width": self.width,
        }


@dataclass
class Layout:
    canvas_height: int
    canvas_width: int
    input_size: int
    expert_size: int
    center_coords: List[int]
    experts: List[Aperture]
    input_aperture: Aperture

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (self.canvas_height, self.canvas_width)

    @property
    def canvas_center(self) -> Tuple[float, float]:
        return (self.canvas_height / 2.0, self.canvas_width / 2.0)

    def to_dict(self) -> Dict:
        baseline_area = 4 * 200 * 200
        nine_area = 9 * self.expert_size * self.expert_size
        return {
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": list(self.canvas_center),
            "input_size": self.input_size,
            "input_aperture": self.input_aperture.to_dict(),
            "num_experts": len(self.experts),
            "expert_size": self.expert_size,
            "center_coords": list(self.center_coords),
            "experts": [expert.to_dict() for expert in self.experts],
            "four_expert_baseline_area": baseline_area,
            "nine_expert_matched_area": nine_area,
            "relative_area_difference": (nine_area - baseline_area) / baseline_area,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone 9-expert free-space propagation geometry test."
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--canvas_height", type=int, default=700)
    parser.add_argument("--canvas_width", type=int, default=700)
    parser.add_argument("--input_size", type=int, default=200)
    parser.add_argument("--expert_size", type=int, default=134)
    parser.add_argument("--center_coords", default="167,350,533")

    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--input_to_prompt_cm", type=float, default=1.0)
    parser.add_argument("--prompt_to_first_layer_cm", type=float, default=24.0)
    parser.add_argument("--inter_layer_cm", type=float, default=5.0)
    parser.add_argument("--num_dummy_layers", type=int, default=5)

    parser.add_argument(
        "--input_types",
        default="gaussian,f_pattern,digit_like",
        help="Comma-separated input types.",
    )
    parser.add_argument(
        "--use_detilt",
        dest="use_detilt",
        action="store_true",
        default=True,
        help="Run the with-detilt case in addition to no-detilt.",
    )
    parser.add_argument(
        "--no_use_detilt",
        dest="use_detilt",
        action="store_false",
        help="Disable the with-detilt case and run no-detilt only.",
    )
    parser.add_argument(
        "--hard_aperture",
        dest="hard_aperture",
        action="store_true",
        default=True,
        help="Apply the union expert aperture at expert planes.",
    )
    parser.add_argument(
        "--no_hard_aperture",
        dest="hard_aperture",
        action="store_false",
        help="Disable hard aperture clipping at expert planes.",
    )
    parser.add_argument("--save_linear_intensity", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=120)
    parser.add_argument("--max_plot_dim", type=int, default=1400)

    parser.add_argument("--pass_centroid_px", type=float, default=25.0)
    parser.add_argument("--warn_centroid_px", type=float, default=45.0)
    parser.add_argument("--pass_ratio", type=float, default=2.0)
    parser.add_argument("--warn_ratio", type=float, default=1.2)
    parser.add_argument("--pass_drift_px", type=float, default=25.0)
    parser.add_argument("--warn_drift_px", type=float, default=45.0)
    parser.add_argument("--detector_margin_px", type=float, default=20.0)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def parse_csv_ints(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--center_coords must contain exactly three integers.")
    return values


def build_layout(args) -> Layout:
    centers = parse_csv_ints(args.center_coords)
    half_input = args.input_size // 2
    input_center_y = args.canvas_height // 2
    input_center_x = args.canvas_width // 2
    input_aperture = Aperture(
        "input",
        input_center_y - half_input,
        input_center_y + half_input,
        input_center_x - half_input,
        input_center_x + half_input,
    )
    half_expert = args.expert_size // 2
    experts = []
    for row, center_y in enumerate(centers):
        for col, center_x in enumerate(centers):
            experts.append(
                Aperture(
                    f"E{row}{col}",
                    center_y - half_expert,
                    center_y + half_expert,
                    center_x - half_expert,
                    center_x + half_expert,
                )
            )
    layout = Layout(
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        input_size=args.input_size,
        expert_size=args.expert_size,
        center_coords=centers,
        experts=experts,
        input_aperture=input_aperture,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: Layout) -> None:
    for aperture in [layout.input_aperture] + layout.experts:
        if aperture.height <= 0 or aperture.width <= 0:
            raise ValueError(f"{aperture.name} has invalid shape.")
        if aperture.y0 < 0 or aperture.x0 < 0:
            raise ValueError(f"{aperture.name} starts outside the canvas.")
        if aperture.y1 > layout.canvas_height or aperture.x1 > layout.canvas_width:
            raise ValueError(f"{aperture.name} ends outside the canvas.")
    masks = expert_masks(layout, device=torch.device("cpu"))
    if torch.any(masks.sum(dim=0) > 1.0):
        raise ValueError("Expert apertures overlap.")


def expert_masks(layout: Layout, device: torch.device) -> torch.Tensor:
    masks = []
    for expert in layout.experts:
        mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        mask[expert.y0 : expert.y1, expert.x0 : expert.x1] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)


def coordinate_grids(layout: Layout, device: torch.device):
    y = torch.arange(layout.canvas_height, dtype=torch.float32, device=device)
    x = torch.arange(layout.canvas_width, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    return y_grid, x_grid


def make_gaussian(layout: Layout, device: torch.device) -> torch.Tensor:
    y_grid, x_grid = coordinate_grids(layout, device)
    cy, cx = layout.canvas_center
    sigma = 45.0
    amplitude = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    mask = torch.zeros_like(amplitude)
    aperture = layout.input_aperture
    mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
    return amplitude * mask


def make_f_pattern(size: int, device: torch.device) -> torch.Tensor:
    pattern = torch.zeros((size, size), dtype=torch.float32, device=device)
    t = max(10, size // 10)
    pattern[20 : size - 20, 25 : 25 + t] = 1.0
    pattern[20 : 20 + t, 25 : size - 25] = 1.0
    pattern[size // 2 - t // 2 : size // 2 + t // 2, 25 : int(size * 0.68)] = 1.0
    pattern[int(size * 0.72) : int(size * 0.82), 25 : int(size * 0.45)] = 0.6
    return pattern


def make_digit_like(size: int, device: torch.device) -> torch.Tensor:
    pattern = torch.zeros((size, size), dtype=torch.float32, device=device)
    t = max(9, size // 12)
    pattern[20 : 20 + t, 35 : size - 35] = 1.0
    pattern[20 : size // 2, 35 : 35 + t] = 1.0
    pattern[size // 2 - t // 2 : size // 2 + t // 2, 35 : size - 35] = 1.0
    pattern[size // 2 : size - 25, size - 35 - t : size - 35] = 1.0
    pattern[size - 25 - t : size - 25, 35 : size - 35] = 1.0
    pattern[int(size * 0.67) : int(size * 0.75), int(size * 0.35) : int(size * 0.50)] = 0.55
    return pattern


def make_input_field(input_type: str, layout: Layout, device: torch.device) -> torch.Tensor:
    amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    if input_type == "gaussian":
        amplitude = make_gaussian(layout, device)
    elif input_type in {"f_pattern", "digit_like"}:
        local = (
            make_f_pattern(layout.input_size, device)
            if input_type == "f_pattern"
            else make_digit_like(layout.input_size, device)
        )
        aperture = layout.input_aperture
        amplitude[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = local
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    return amplitude.unsqueeze(0).to(torch.complex64)


def build_propagator(
    wavelength_m: float,
    pixel_size_m: float,
    grid_size: Tuple[int, int],
    distance_m: float,
    device: torch.device,
) -> AngularSpectrumPropagator:
    return AngularSpectrumPropagator(
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
        grid_size=grid_size,
        distance_m=distance_m,
    ).to(device)


def phase_increments(
    target: Aperture,
    layout: Layout,
    wavelength_m: float,
    pixel_size_m: float,
    distance_m: float,
) -> Dict[str, float]:
    center_y, center_x = layout.canvas_center
    target_y, target_x = target.center
    dx_m = (target_x - center_x) * pixel_size_m
    dy_m = (target_y - center_y) * pixel_size_m
    theta_x = math.atan(dx_m / distance_m)
    theta_y = math.atan(dy_m / distance_m)
    inc_x = 2.0 * math.pi * pixel_size_m * math.sin(theta_x) / wavelength_m
    inc_y = 2.0 * math.pi * pixel_size_m * math.sin(theta_y) / wavelength_m
    return {
        "dx_px": target_x - center_x,
        "dy_px": target_y - center_y,
        "theta_x_deg": math.degrees(theta_x),
        "theta_y_deg": math.degrees(theta_y),
        "phase_increment_x": inc_x,
        "phase_increment_y": inc_y,
    }


def linear_phase(
    layout: Layout,
    inc_x: float,
    inc_y: float,
    sign_x: int,
    sign_y: int,
    device: torch.device,
    origin: Tuple[float, float] = None,
) -> torch.Tensor:
    y_grid, x_grid = coordinate_grids(layout, device)
    if origin is None:
        origin_y, origin_x = layout.canvas_center
    else:
        origin_y, origin_x = origin
    return sign_x * inc_x * (x_grid - origin_x) + sign_y * inc_y * (y_grid - origin_y)


def build_prompt_phase(
    target: Aperture,
    layout: Layout,
    wavelength_m: float,
    pixel_size_m: float,
    distance_m: float,
    sign_x: int,
    sign_y: int,
    device: torch.device,
) -> torch.Tensor:
    increments = phase_increments(target, layout, wavelength_m, pixel_size_m, distance_m)
    if abs(increments["dx_px"]) < EPS and abs(increments["dy_px"]) < EPS:
        return torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    return linear_phase(
        layout,
        increments["phase_increment_x"],
        increments["phase_increment_y"],
        sign_x,
        sign_y,
        device,
    )


def build_detilt_phase(
    target: Aperture,
    layout: Layout,
    wavelength_m: float,
    pixel_size_m: float,
    distance_m: float,
    sign_x: int,
    sign_y: int,
    device: torch.device,
) -> torch.Tensor:
    increments = phase_increments(target, layout, wavelength_m, pixel_size_m, distance_m)
    phase = -linear_phase(
        layout,
        increments["phase_increment_x"],
        increments["phase_increment_y"],
        sign_x,
        sign_y,
        device,
        origin=target.center,
    )
    mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    mask[target.y0 : target.y1, target.x0 : target.x1] = 1.0
    return phase * mask


def centroid_from_intensity(intensity: torch.Tensor) -> Tuple[float, float]:
    if intensity.ndim == 3:
        intensity = intensity[0]
    total = float(intensity.sum().item())
    if total <= EPS:
        return float("nan"), float("nan")
    height, width = intensity.shape
    y = torch.arange(height, dtype=torch.float32, device=intensity.device)
    x = torch.arange(width, dtype=torch.float32, device=intensity.device)
    cy = float((intensity.sum(dim=1) * y).sum().item() / total)
    cx = float((intensity.sum(dim=0) * x).sum().item() / total)
    return cy, cx


def edge_energy_ratio(intensity: torch.Tensor, edge: int = 50) -> float:
    if intensity.ndim == 3:
        intensity = intensity[0]
    total = float(intensity.sum().item())
    if total <= EPS:
        return 0.0
    mask = torch.zeros_like(intensity)
    mask[:edge, :] = 1.0
    mask[-edge:, :] = 1.0
    mask[:, :edge] = 1.0
    mask[:, -edge:] = 1.0
    return float((intensity * mask).sum().item() / (total + EPS))


def compute_metrics(
    field_or_intensity: torch.Tensor,
    layout: Layout,
    masks: torch.Tensor,
    input_type: str,
    target_index: int,
    target: Aperture,
    plane_name: str,
    context: Dict,
) -> Dict:
    if torch.is_complex(field_or_intensity):
        intensity = torch.abs(field_or_intensity.to(torch.complex64)).square()
    else:
        intensity = field_or_intensity.float()
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    total_energy = float(intensity.sum().item())
    energies = torch.einsum("bhw,khw->bk", intensity, masks)[0]
    ratios = energies / (total_energy + EPS)
    outside = max(0.0, total_energy - float(energies.sum().item()))
    centroid_y, centroid_x = centroid_from_intensity(intensity)
    target_y, target_x = target.center
    centroid_error = math.sqrt((centroid_y - target_y) ** 2 + (centroid_x - target_x) ** 2)
    sorted_ratios = torch.sort(ratios, descending=True).values
    second = float(sorted_ratios[1].item()) if len(sorted_ratios) > 1 else 0.0
    target_ratio = float(ratios[target_index].item())
    row = {
        "plane_name": plane_name,
        "input_type": input_type,
        "case_name": context["case_name"],
        "target_expert_id": target.name,
        "target_expert_index": target_index,
        "target_center_y": target_y,
        "target_center_x": target_x,
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_error_px": centroid_error,
        "total_energy": total_energy,
        "target_expert_energy": float(energies[target_index].item()),
        "target_expert_energy_ratio": target_ratio,
        "outside_all_experts_energy": outside,
        "outside_all_experts_energy_ratio": outside / (total_energy + EPS),
        "second_largest_expert_energy_ratio": second,
        "target_to_second_ratio": target_ratio / (second + EPS),
        "edge_energy_ratio": edge_energy_ratio(intensity),
        "calibrated_sign_x": context["sign_x"],
        "calibrated_sign_y": context["sign_y"],
        "use_detilt": context["use_detilt"],
        "hard_aperture": context["hard_aperture"],
    }
    for index, expert_id in enumerate(EXPERT_IDS):
        row[f"{expert_id}_energy"] = float(energies[index].item())
        row[f"{expert_id}_energy_ratio"] = float(ratios[index].item())
    return row


def plot_intensity(
    intensity_or_field: torch.Tensor,
    layout: Layout,
    target: Aperture,
    centroid: Tuple[float, float],
    title: str,
    path: Path,
    plot_dpi: int,
    max_plot_dim: int,
    save_linear: bool,
) -> None:
    if torch.is_complex(intensity_or_field):
        intensity = torch.abs(intensity_or_field.to(torch.complex64)).square()
    else:
        intensity = intensity_or_field.float()
    if intensity.ndim == 3:
        intensity = intensity[0]
    array = intensity.detach().cpu().float().numpy()
    log_array = np.log10(array / (array.max() + EPS) + 1e-8)
    _draw_image(
        log_array,
        layout,
        target,
        centroid,
        title,
        path,
        plot_dpi,
        max_plot_dim,
        "log10(I/Imax+1e-8)",
    )
    if save_linear:
        linear_path = path.with_name(path.stem + "_linear.png")
        _draw_image(
            array,
            layout,
            target,
            centroid,
            title,
            linear_path,
            plot_dpi,
            max_plot_dim,
            "linear intensity",
        )


def _draw_image(
    array: np.ndarray,
    layout: Layout,
    target: Aperture,
    centroid: Tuple[float, float],
    title: str,
    path: Path,
    plot_dpi: int,
    max_plot_dim: int,
    colorbar_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = array
    stride = max(1, int(math.ceil(max(array.shape) / max(1, int(max_plot_dim)))))
    if stride > 1:
        display = array[::stride, ::stride]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        display,
        cmap="inferno",
        extent=(0, layout.canvas_width, layout.canvas_height, 0),
    )
    for expert in layout.experts:
        color = "cyan" if expert.name != target.name else "lime"
        linewidth = 1.0 if expert.name != target.name else 2.2
        ax.add_patch(
            Rectangle(
                (expert.x0, expert.y0),
                expert.width,
                expert.height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )
        cy, cx = expert.center
        ax.text(cx, cy, expert.name, color=color, ha="center", va="center", fontsize=8)
    ty, tx = target.center
    cy, cx = centroid
    ax.scatter([tx], [ty], c="lime", marker="+", s=70, linewidths=2.0, label="target")
    if math.isfinite(cy) and math.isfinite(cx):
        ax.scatter([cx], [cy], c="white", marker="x", s=60, linewidths=2.0, label="centroid")
    input_ap = layout.input_aperture
    ax.add_patch(
        Rectangle(
            (input_ap.x0, input_ap.y0),
            input_ap.width,
            input_ap.height,
            fill=False,
            edgecolor="orange",
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.set_title(title)
    ax.set_xlim(0, layout.canvas_width - 1)
    ax.set_ylim(layout.canvas_height - 1, 0)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def plot_phase(phase: torch.Tensor, path: Path, title: str, plot_dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(wrapped, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="wrapped phase [rad]")
    fig.tight_layout()
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def plot_layout(layout: Layout, path: Path, plot_dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("black")
    input_ap = layout.input_aperture
    ax.add_patch(
        Rectangle(
            (input_ap.x0, input_ap.y0),
            input_ap.width,
            input_ap.height,
            fill=False,
            edgecolor="orange",
            linestyle="--",
            linewidth=2.0,
            label="input",
        )
    )
    for expert in layout.experts:
        ax.add_patch(
            Rectangle(
                (expert.x0, expert.y0),
                expert.width,
                expert.height,
                fill=False,
                edgecolor="cyan",
                linewidth=1.8,
            )
        )
        cy, cx = expert.center
        ax.scatter([cx], [cy], c="white", s=18)
        ax.text(cx, cy, expert.name, color="white", ha="center", va="center")
    cy, cx = layout.canvas_center
    ax.axhline(cy, color="gray", linewidth=0.8)
    ax.axvline(cx, color="gray", linewidth=0.8)
    ax.set_title("9-Expert Parameter-Matched Layout")
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def plot_heatmap_3x3(
    values: Sequence[float],
    centers: Sequence[int],
    path: Path,
    title: str,
    colorbar_label: str,
    plot_dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(array, cmap="viridis")
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([str(value) for value in centers])
    ax.set_yticklabels([str(value) for value in centers])
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{array[row, col]:.3f}", ha="center", va="center", color="white")
    ax.set_xlabel("expert center x")
    ax.set_ylabel("expert center y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def plot_crosstalk(matrix: np.ndarray, path: Path, title: str, plot_dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels(EXPERT_IDS, rotation=45, ha="right")
    ax.set_yticklabels(EXPERT_IDS)
    ax.set_xlabel("measured expert aperture")
    ax.set_ylabel("target expert")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="energy ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def calibrate_signs(
    target: Aperture,
    layout: Layout,
    input_field: torch.Tensor,
    input_to_prompt,
    prompt_to_first,
    wavelength_m: float,
    pixel_size_m: float,
    distance_m: float,
    device: torch.device,
) -> Dict:
    increments = phase_increments(target, layout, wavelength_m, pixel_size_m, distance_m)
    if abs(increments["dx_px"]) < EPS and abs(increments["dy_px"]) < EPS:
        return {"sign_x": 1, "sign_y": 1, "calibration_error_px": 0.0}
    after_input = input_to_prompt(input_field)
    candidates = []
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            phase = build_prompt_phase(
                target,
                layout,
                wavelength_m,
                pixel_size_m,
                distance_m,
                sign_x,
                sign_y,
                device,
            )
            field = after_input * torch.exp(1j * phase).to(torch.complex64).unsqueeze(0)
            first = prompt_to_first(field)
            cy, cx = centroid_from_intensity(torch.abs(first).square())
            ty, tx = target.center
            error = math.sqrt((cy - ty) ** 2 + (cx - tx) ** 2)
            candidates.append((error, sign_x, sign_y))
    candidates.sort(key=lambda item: item[0])
    return {
        "sign_x": candidates[0][1],
        "sign_y": candidates[0][2],
        "calibration_error_px": candidates[0][0],
    }


def status_for_threshold(value: float, pass_value: float, warn_value: float) -> str:
    if value < pass_value:
        return "PASS"
    if value < warn_value:
        return "WARN"
    return "FAIL"


def status_for_ratio(value: float, pass_value: float, warn_value: float) -> str:
    if value > pass_value:
        return "PASS"
    if value > warn_value:
        return "WARN"
    return "FAIL"


def summarize_target_status(rows: List[Dict], target: Aperture, args) -> Dict:
    by_plane = {row["plane_name"]: row for row in rows}
    before = by_plane["first_layer_before_detilt"]
    after = by_plane.get("first_layer_after_detilt", before)
    detector = by_plane["detector_plane"]

    first_centroid_status = status_for_threshold(
        before["centroid_error_px"],
        args.pass_centroid_px,
        args.warn_centroid_px,
    )
    target_is_top = True
    target_ratio = before["target_expert_energy_ratio"]
    for expert_id in EXPERT_IDS:
        if before[f"{expert_id}_energy_ratio"] > target_ratio + 1e-9:
            target_is_top = False
            break
    ratio_status = status_for_ratio(
        before["target_to_second_ratio"],
        args.pass_ratio,
        args.warn_ratio,
    )

    reference_y = after["centroid_y"]
    reference_x = after["centroid_x"]
    layer_drifts = []
    for name, row in by_plane.items():
        if name.startswith("layer"):
            layer_drifts.append(
                math.sqrt(
                    (row["centroid_y"] - reference_y) ** 2
                    + (row["centroid_x"] - reference_x) ** 2
                )
            )
    max_drift = max(layer_drifts) if layer_drifts else 0.0
    drift_status = status_for_threshold(max_drift, args.pass_drift_px, args.warn_drift_px)
    detector_inside = target.contains_with_margin(
        detector["centroid_y"],
        detector["centroid_x"],
        args.detector_margin_px,
    )
    detector_status = "PASS" if detector_inside else "WARN"
    statuses = [first_centroid_status, ratio_status, drift_status, detector_status]
    if not target_is_top:
        statuses.append("FAIL")
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    return {
        "target_expert_id": target.name,
        "first_layer_centroid_error_px": before["centroid_error_px"],
        "first_layer_centroid_status": first_centroid_status,
        "first_layer_target_is_top": target_is_top,
        "first_layer_target_to_second_ratio": before["target_to_second_ratio"],
        "first_layer_ratio_status": ratio_status,
        "dummy_layer_max_drift_px": max_drift,
        "dummy_layer_drift_status": drift_status,
        "detector_centroid_inside_target_margin": detector_inside,
        "detector_status": detector_status,
        "overall_status": overall,
    }


def run_case(
    input_type: str,
    target_index: int,
    use_detilt: bool,
    layout: Layout,
    args,
    physical: Dict,
    propagators: Dict,
    masks: torch.Tensor,
    union_mask: torch.Tensor,
    signs: Dict,
    output_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict], Dict]:
    target = layout.experts[target_index]
    case_name = "with_detilt" if use_detilt else "no_detilt"
    case_dir = output_dir / "figures" / input_type / target.name / case_name
    input_field = make_input_field(input_type, layout, device)
    prompt_phase = build_prompt_phase(
        target,
        layout,
        physical["wavelength_m"],
        physical["pixel_size_m"],
        physical["prompt_to_first_layer_m"],
        signs["sign_x"],
        signs["sign_y"],
        device,
    )
    detilt_phase = build_detilt_phase(
        target,
        layout,
        physical["wavelength_m"],
        physical["pixel_size_m"],
        physical["prompt_to_first_layer_m"],
        signs["sign_x"],
        signs["sign_y"],
        device,
    )
    context = {
        "case_name": case_name,
        "sign_x": signs["sign_x"],
        "sign_y": signs["sign_y"],
        "use_detilt": use_detilt,
        "hard_aperture": bool(args.hard_aperture),
    }
    rows = []

    def record(plane_name: str, field, filename: str):
        row = compute_metrics(
            field,
            layout,
            masks,
            input_type,
            target_index,
            target,
            plane_name,
            context,
        )
        rows.append(row)
        plot_intensity(
            field,
            layout,
            target,
            (row["centroid_y"], row["centroid_x"]),
            (
                f"{input_type} {target.name} {case_name}: {plane_name}\n"
                f"centroid=({row['centroid_y']:.1f},{row['centroid_x']:.1f}), "
                f"target_ratio={row['target_expert_energy_ratio']:.3f}, "
                f"target/second={row['target_to_second_ratio']:.2f}"
            ),
            case_dir / filename,
            args.plot_dpi,
            args.max_plot_dim,
            args.save_linear_intensity,
        )

    record("input_plane", input_field, "input_intensity.png")
    after_input = propagators["input_to_prompt"](input_field)
    field = after_input * torch.exp(1j * prompt_phase).to(torch.complex64).unsqueeze(0)
    plot_phase(
        prompt_phase,
        case_dir / "prompt_phase_wrapped.png",
        f"{input_type} {target.name}: calibrated prompt phase",
        args.plot_dpi,
    )
    first = propagators["prompt_to_first"](field)
    record("first_layer_before_detilt", first, "first_layer_before_detilt_intensity.png")
    if args.hard_aperture:
        first = first * union_mask.unsqueeze(0).to(torch.complex64)
    if use_detilt:
        first = first * torch.exp(1j * detilt_phase).to(torch.complex64).unsqueeze(0)
    record("first_layer_after_detilt", first, "first_layer_after_detilt_intensity.png")

    field = first
    for layer_idx in range(2, int(args.num_dummy_layers) + 1):
        field = propagators["inter_layer"](field)
        if args.hard_aperture:
            field = field * union_mask.unsqueeze(0).to(torch.complex64)
        filename = f"layer{layer_idx}_intensity.png"
        record(f"layer{layer_idx}", field, filename)

    detector = propagators["inter_layer"](field)
    record("detector_plane", detector, "detector_plane_intensity.png")
    status = summarize_target_status(rows, target, args) if input_type == "gaussian" and use_detilt else {}
    return rows, status


def crosstalk_matrix(rows: List[Dict], input_type: str, case_name: str, plane_name: str) -> np.ndarray:
    matrix = np.zeros((9, 9), dtype=np.float32)
    selected = [
        row
        for row in rows
        if row["input_type"] == input_type
        and row["case_name"] == case_name
        and row["plane_name"] == plane_name
    ]
    for row in selected:
        target_index = int(row["target_expert_index"])
        for expert_index, expert_id in enumerate(EXPERT_IDS):
            matrix[target_index, expert_index] = float(row[f"{expert_id}_energy_ratio"])
    return matrix


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\measured"] + EXPERT_IDS)
        for row_index, expert_id in enumerate(EXPERT_IDS):
            writer.writerow([expert_id] + [float(value) for value in matrix[row_index]])


def save_summary_heatmaps(rows: List[Dict], layout: Layout, output_dir: Path, args) -> None:
    selected = [
        row
        for row in rows
        if row["input_type"] == "gaussian"
        and row["case_name"] == "with_detilt"
        and row["plane_name"] == "first_layer_before_detilt"
    ]
    selected = sorted(selected, key=lambda row: int(row["target_expert_index"]))
    if len(selected) == 9:
        plot_heatmap_3x3(
            [row["centroid_error_px"] for row in selected],
            layout.center_coords,
            output_dir / "figures" / "centroid_error_first_layer_3x3.png",
            "First-layer centroid error",
            "px",
            args.plot_dpi,
        )
        plot_heatmap_3x3(
            [row["target_expert_energy_ratio"] for row in selected],
            layout.center_coords,
            output_dir / "figures" / "target_energy_ratio_first_layer_3x3.png",
            "First-layer target energy ratio",
            "energy ratio",
            args.plot_dpi,
        )
        plot_heatmap_3x3(
            [row["target_to_second_ratio"] for row in selected],
            layout.center_coords,
            output_dir / "figures" / "target_to_second_ratio_first_layer_3x3.png",
            "First-layer target / second expert ratio",
            "ratio",
            args.plot_dpi,
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    layout = build_layout(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "runs" / "nine_expert_propagation_test" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength_m = nm_to_m(args.wavelength_nm)
    pixel_size_m = um_to_m(args.pixel_size_um)
    physical = {
        "wavelength_nm": args.wavelength_nm,
        "pixel_size_um": args.pixel_size_um,
        "wavelength_m": wavelength_m,
        "pixel_size_m": pixel_size_m,
        "input_to_prompt_cm": args.input_to_prompt_cm,
        "prompt_to_first_layer_cm": args.prompt_to_first_layer_cm,
        "inter_layer_cm": args.inter_layer_cm,
        "input_to_prompt_m": cm_to_m(args.input_to_prompt_cm),
        "prompt_to_first_layer_m": cm_to_m(args.prompt_to_first_layer_cm),
        "inter_layer_m": cm_to_m(args.inter_layer_cm),
        "num_dummy_layers": args.num_dummy_layers,
    }
    save_json(output_dir / "layout.json", layout.to_dict())
    save_json(output_dir / "physical_params.json", physical)

    baseline_area = 4 * 200 * 200
    nine_area = 9 * args.expert_size * args.expert_size
    relative_diff = (nine_area - baseline_area) / baseline_area
    print("9-expert propagation test")
    print(f"device: {device}")
    print(f"output_dir: {output_dir}")
    print(f"4-expert baseline expert area: 4 x 200 x 200 = {baseline_area}")
    print(
        f"9-expert matched expert area: 9 x {args.expert_size} x "
        f"{args.expert_size} = {nine_area}"
    )
    print(f"relative difference: {relative_diff * 100.0:+.2f}%")

    propagators = {
        "input_to_prompt": build_propagator(
            wavelength_m,
            pixel_size_m,
            layout.canvas_shape,
            physical["input_to_prompt_m"],
            device,
        ),
        "prompt_to_first": build_propagator(
            wavelength_m,
            pixel_size_m,
            layout.canvas_shape,
            physical["prompt_to_first_layer_m"],
            device,
        ),
        "inter_layer": build_propagator(
            wavelength_m,
            pixel_size_m,
            layout.canvas_shape,
            physical["inter_layer_m"],
            device,
        ),
    }
    masks = expert_masks(layout, device)
    union_mask = torch.clamp(masks.sum(dim=0), 0.0, 1.0)
    plot_layout(layout, output_dir / "figures" / "layout_overlay.png", args.plot_dpi)

    input_types = [item.strip() for item in args.input_types.split(",") if item.strip()]
    gaussian_field = make_input_field("gaussian", layout, device)
    signs_by_target = {}
    for target_index, target in enumerate(layout.experts):
        signs_by_target[target.name] = calibrate_signs(
            target,
            layout,
            gaussian_field,
            propagators["input_to_prompt"],
            propagators["prompt_to_first"],
            wavelength_m,
            pixel_size_m,
            physical["prompt_to_first_layer_m"],
            device,
        )
    save_json(output_dir / "calibrated_grating_signs.json", signs_by_target)

    all_rows = []
    statuses = []
    cases = [False]
    if args.use_detilt:
        cases.append(True)
    for input_type in input_types:
        for target_index, target in enumerate(layout.experts):
            for use_detilt in cases:
                rows, status = run_case(
                    input_type=input_type,
                    target_index=target_index,
                    use_detilt=use_detilt,
                    layout=layout,
                    args=args,
                    physical=physical,
                    propagators=propagators,
                    masks=masks,
                    union_mask=union_mask,
                    signs=signs_by_target[target.name],
                    output_dir=output_dir,
                    device=device,
                )
                all_rows.extend(rows)
                if status:
                    statuses.append(status)

    save_csv(output_dir / "metrics.csv", all_rows)
    save_json(
        output_dir / "status_summary.json",
        {
            "layout": layout.to_dict(),
            "physical_params": physical,
            "area_comparison": {
                "four_expert_baseline_area": 4 * 200 * 200,
                "nine_expert_matched_area": 9 * args.expert_size * args.expert_size,
                "relative_difference": (
                    9 * args.expert_size * args.expert_size - 4 * 200 * 200
                )
                / float(4 * 200 * 200),
            },
            "grating_signs": signs_by_target,
            "gaussian_with_detilt_status": statuses,
            "overall_status": (
                "FAIL"
                if any(item["overall_status"] == "FAIL" for item in statuses)
                else (
                    "WARN"
                    if any(item["overall_status"] == "WARN" for item in statuses)
                    else "PASS"
                )
            ),
            "diagnostic_notes": [
                "If corner experts fail first, inspect grating sign calibration and prompt_to_first_layer distance.",
                "If with-detilt drifts, check that de-tilt is applied only inside the target aperture.",
                "If target_to_second_ratio is low, expert_size may be too small or adjacent aperture crosstalk may be high.",
                "If edge_energy_ratio is high, check FFT wrap-around and canvas margins.",
            ],
        },
    )

    for case_name in ["no_detilt", "with_detilt"]:
        if case_name == "with_detilt" and not args.use_detilt:
            continue
        for plane_name, filename in [
            ("first_layer_before_detilt", "first_layer_before_detilt_crosstalk.csv"),
            ("first_layer_after_detilt", "first_layer_after_detilt_crosstalk.csv"),
            ("detector_plane", "detector_plane_crosstalk.csv"),
        ]:
            matrix = crosstalk_matrix(all_rows, "gaussian", case_name, plane_name)
            save_matrix_csv(output_dir / "crosstalk" / f"{case_name}_{filename}", matrix)
            if case_name == "with_detilt":
                save_matrix_csv(output_dir / filename, matrix)
                if plane_name == "first_layer_before_detilt":
                    plot_crosstalk(
                        matrix,
                        output_dir / "figures" / "crosstalk_matrix_first_layer.png",
                        "Gaussian first-layer crosstalk",
                        args.plot_dpi,
                    )
                if plane_name == "detector_plane":
                    plot_crosstalk(
                        matrix,
                        output_dir / "figures" / "crosstalk_matrix_detector.png",
                        "Gaussian detector-plane crosstalk",
                        args.plot_dpi,
                    )
    save_summary_heatmaps(all_rows, layout, output_dir, args)

    overall = json.loads((output_dir / "status_summary.json").read_text(encoding="utf-8"))[
        "overall_status"
    ]
    print(f"overall gaussian with-detilt status: {overall}")
    print(f"metrics: {output_dir / 'metrics.csv'}")
    print(f"figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
