import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class Aperture:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.y0 + self.y1) / 2.0, (self.x0 + self.x1) / 2.0)

    def shifted(self, dy: int = 0, dx: int = 0) -> "Aperture":
        return Aperture(self.y0 + dy, self.y1 + dy, self.x0 + dx, self.x1 + dx)


@dataclass
class Layout:
    canvas_shape: Tuple[int, int]
    left: Aperture
    right: Aperture
    center_y: float
    center_x: float


@dataclass
class PhysicalParams:
    wavelength_nm: float
    pixel_size_um: float
    input_to_prompt_cm: float
    prompt_to_first_layer_cm: float
    inter_layer_cm: float
    num_dummy_layers: int
    shift_pixels: float
    shift_mm: float
    steering_angle_deg: float
    grating_period_px: float
    phase_increment_rad_per_px: float
    drift_per_5cm_px: float


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt grating alignment test.")
    parser.add_argument("--run_name", default="prompt_grating_alignment_v1")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--canvas_height", type=int, default=800)
    parser.add_argument("--canvas_width", type=int, default=1600)
    parser.add_argument("--expert_size", type=int, default=600)
    parser.add_argument("--gap_pixels", type=int, default=200)
    parser.add_argument("--margin_x", type=int, default=100)
    parser.add_argument("--margin_y", type=int, default=100)

    parser.add_argument("--input_to_prompt_cm", type=float, default=1.0)
    parser.add_argument("--prompt_to_first_layer_cm", type=float, default=24.0)
    parser.add_argument("--inter_layer_cm", type=float, default=5.0)
    parser.add_argument("--num_dummy_layers", type=int, default=5)

    parser.add_argument(
        "--input_types",
        default="gaussian,f_pattern,digit_like",
        help="Comma-separated input types: gaussian,f_pattern,digit_like.",
    )
    parser.add_argument("--save_linear_intensity", action="store_true")
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=120,
        help="PNG dpi for diagnostic figures. Lower values save much faster.",
    )
    parser.add_argument(
        "--max_plot_dim",
        type=int,
        default=1600,
        help="Downsample figures for display when a canvas side is larger than this value. Metrics still use full resolution.",
    )
    parser.add_argument("--skip_sweeps", action="store_true")
    return parser.parse_args()


def build_layout(args) -> Layout:
    height = args.canvas_height
    width = args.canvas_width
    left = Aperture(
        args.margin_y,
        args.margin_y + args.expert_size,
        args.margin_x,
        args.margin_x + args.expert_size,
    )
    right_x0 = args.margin_x + args.expert_size + args.gap_pixels
    right = Aperture(args.margin_y, args.margin_y + args.expert_size, right_x0, right_x0 + args.expert_size)
    return Layout(
        canvas_shape=(height, width),
        left=left,
        right=right,
        center_y=height / 2.0,
        center_x=width / 2.0,
    )


def compute_physical_params(args, layout: Layout) -> PhysicalParams:
    wavelength_m = nm_to_m(args.wavelength_nm)
    pixel_size_m = um_to_m(args.pixel_size_um)
    z_m = cm_to_m(args.prompt_to_first_layer_cm)

    shift_pixels = layout.right.center[1] - layout.center_x
    shift_m = shift_pixels * pixel_size_m
    theta = math.atan(shift_m / z_m)
    period_px = wavelength_m / (pixel_size_m * math.sin(theta))
    phase_increment = 2.0 * math.pi / period_px
    drift_per_5cm_px = cm_to_m(args.inter_layer_cm) * math.tan(theta) / pixel_size_m

    return PhysicalParams(
        wavelength_nm=args.wavelength_nm,
        pixel_size_um=args.pixel_size_um,
        input_to_prompt_cm=args.input_to_prompt_cm,
        prompt_to_first_layer_cm=args.prompt_to_first_layer_cm,
        inter_layer_cm=args.inter_layer_cm,
        num_dummy_layers=args.num_dummy_layers,
        shift_pixels=shift_pixels,
        shift_mm=shift_m * 1e3,
        steering_angle_deg=math.degrees(theta),
        grating_period_px=period_px,
        phase_increment_rad_per_px=phase_increment,
        drift_per_5cm_px=drift_per_5cm_px,
    )


def make_propagator(args, distance_cm: float, device: torch.device) -> AngularSpectrumPropagator:
    prop = AngularSpectrumPropagator(
        wavelength_m=nm_to_m(args.wavelength_nm),
        pixel_size_m=um_to_m(args.pixel_size_um),
        grid_size=(args.canvas_height, args.canvas_width),
        distance_m=cm_to_m(distance_cm),
    )
    return prop.to(device)


def build_linear_grating_phase(
    canvas_shape: Tuple[int, int],
    period_px: float,
    direction: str,
    slope_sign: int = 1,
    multiplier: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if direction not in {"right", "left"}:
        raise ValueError("direction must be 'right' or 'left'")
    height, width = canvas_shape
    x = torch.arange(width, dtype=torch.float32, device=device).view(1, width)
    x = x - (width / 2.0)
    direction_sign = 1.0 if direction == "right" else -1.0
    phase_increment = 2.0 * math.pi / period_px
    phase = direction_sign * float(slope_sign) * float(multiplier) * phase_increment * x
    return phase.repeat(height, 1)


def build_detilt_phase_for_aperture(
    canvas_shape: Tuple[int, int],
    aperture: Aperture,
    period_px: float,
    direction: str,
    prompt_slope_sign: int,
    wrong_sign: bool = False,
    multiplier: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    prompt_phase = build_linear_grating_phase(
        canvas_shape,
        period_px,
        direction,
        slope_sign=prompt_slope_sign,
        multiplier=multiplier,
        device=device,
    )
    detilt_phase = prompt_phase if wrong_sign else -prompt_phase
    mask = aperture_mask(canvas_shape, aperture, device=device)
    return detilt_phase * mask


def aperture_mask(
    canvas_shape: Tuple[int, int],
    aperture: Aperture,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    height, width = canvas_shape
    mask = torch.zeros(height, width, dtype=torch.float32, device=device)
    y0 = max(0, aperture.y0)
    y1 = min(height, aperture.y1)
    x0 = max(0, aperture.x0)
    x1 = min(width, aperture.x1)
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def build_expert_aperture_mask(
    canvas_shape: Tuple[int, int],
    left_aperture: Aperture,
    right_aperture: Aperture,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    left = aperture_mask(canvas_shape, left_aperture, device=device)
    right = aperture_mask(canvas_shape, right_aperture, device=device)
    return torch.clamp(left + right, 0.0, 1.0)


def make_gaussian(layout: Layout, sigma_px: float, device: torch.device) -> torch.Tensor:
    height, width = layout.canvas_shape
    y = torch.arange(height, dtype=torch.float32, device=device).view(height, 1)
    x = torch.arange(width, dtype=torch.float32, device=device).view(1, width)
    amp = torch.exp(-(((y - layout.center_y) ** 2 + (x - layout.center_x) ** 2) / (2.0 * sigma_px**2)))
    return amp


def make_f_pattern(device: torch.device) -> torch.Tensor:
    pattern = torch.zeros(200, 200, dtype=torch.float32, device=device)
    pattern[25:175, 25:55] = 1.0
    pattern[25:55, 25:165] = 1.0
    pattern[85:115, 25:135] = 1.0
    pattern[130:145, 25:85] = 0.6
    pattern[55:85, 25:45] = 0.8
    return pattern


def make_digit_like(device: torch.device) -> torch.Tensor:
    pattern = torch.zeros(200, 200, dtype=torch.float32, device=device)
    pattern[25:55, 45:155] = 1.0
    pattern[55:95, 35:65] = 1.0
    pattern[90:120, 45:145] = 1.0
    pattern[115:165, 135:165] = 1.0
    pattern[155:180, 45:155] = 1.0
    pattern[35:75, 145:160] = 0.4
    return pattern


def make_input_field(input_type: str, layout: Layout, device: torch.device) -> torch.Tensor:
    height, width = layout.canvas_shape
    amplitude = torch.zeros(height, width, dtype=torch.float32, device=device)

    if input_type == "gaussian":
        amplitude = make_gaussian(layout, sigma_px=60.0, device=device)
    elif input_type in {"f_pattern", "digit_like"}:
        pattern = make_f_pattern(device) if input_type == "f_pattern" else make_digit_like(device)
        y0 = int(layout.center_y - pattern.shape[0] // 2)
        x0 = int(layout.center_x - pattern.shape[1] // 2)
        amplitude[y0 : y0 + pattern.shape[0], x0 : x0 + pattern.shape[1]] = pattern
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    return amplitude.unsqueeze(0).to(torch.complex64)


def intensity(field: torch.Tensor) -> torch.Tensor:
    return torch.abs(field.to(torch.complex64)) ** 2


def weighted_centroid(intensity_2d: torch.Tensor) -> Tuple[float, float]:
    total = intensity_2d.sum()
    if total.item() <= EPS:
        return float("nan"), float("nan")
    height, width = intensity_2d.shape
    y = torch.arange(height, dtype=torch.float32, device=intensity_2d.device).view(height, 1)
    x = torch.arange(width, dtype=torch.float32, device=intensity_2d.device).view(1, width)
    cy = (intensity_2d * y).sum() / total
    cx = (intensity_2d * x).sum() / total
    return float(cy.item()), float(cx.item())


def aperture_energy(intensity_2d: torch.Tensor, aperture: Aperture) -> torch.Tensor:
    height, width = intensity_2d.shape
    y0 = max(0, aperture.y0)
    y1 = min(height, aperture.y1)
    x0 = max(0, aperture.x0)
    x1 = min(width, aperture.x1)
    if y1 <= y0 or x1 <= x0:
        return torch.zeros((), dtype=intensity_2d.dtype, device=intensity_2d.device)
    return intensity_2d[y0:y1, x0:x1].sum()


def edge_energy_ratio(intensity_2d: torch.Tensor, border_px: int = 50) -> float:
    height, width = intensity_2d.shape
    edge = torch.zeros_like(intensity_2d, dtype=torch.bool)
    edge[:border_px, :] = True
    edge[-border_px:, :] = True
    edge[:, :border_px] = True
    edge[:, -border_px:] = True
    return float((intensity_2d[edge].sum() / (intensity_2d.sum() + EPS)).item())


def target_center(layout: Layout, target_side: Optional[str]) -> Tuple[float, float]:
    if target_side == "right":
        return layout.right.center
    if target_side == "left":
        return layout.left.center
    return layout.center_y, layout.center_x


def compute_metrics(
    field_or_intensity: torch.Tensor,
    plane_name: str,
    input_type: str,
    case_name: str,
    target_side: Optional[str],
    use_detilt: bool,
    prompt_slope_sign: int,
    args,
    layout: Layout,
    phys: PhysicalParams,
) -> Dict:
    image_intensity = field_or_intensity
    if torch.is_complex(image_intensity):
        image_intensity = intensity(image_intensity)
    if image_intensity.ndim == 3:
        image_intensity = image_intensity[0]

    total = image_intensity.sum()
    e_left = aperture_energy(image_intensity, layout.left)
    e_right = aperture_energy(image_intensity, layout.right)
    e_outside = total - e_left - e_right
    cy, cx = weighted_centroid(image_intensity)
    ty, tx = target_center(layout, target_side)
    err_y = cy - ty
    err_x = cx - tx
    err = math.sqrt(err_y**2 + err_x**2)

    if target_side == "right":
        other_to_target = float((e_left / (e_right + EPS)).item())
    elif target_side == "left":
        other_to_target = float((e_right / (e_left + EPS)).item())
    else:
        other_to_target = float("nan")

    return {
        "plane_name": plane_name,
        "input_type": input_type,
        "case_name": case_name,
        "target_side": target_side or "none",
        "use_detilt": bool(use_detilt),
        "prompt_slope_sign": int(prompt_slope_sign),
        "wavelength_nm": phys.wavelength_nm,
        "pixel_size_um": phys.pixel_size_um,
        "canvas_height": args.canvas_height,
        "canvas_width": args.canvas_width,
        "prompt_to_first_layer_cm": phys.prompt_to_first_layer_cm,
        "inter_layer_cm": phys.inter_layer_cm,
        "shift_pixels": phys.shift_pixels,
        "shift_mm": phys.shift_mm,
        "steering_angle_deg": phys.steering_angle_deg,
        "grating_period_px": phys.grating_period_px,
        "phase_increment_rad_per_px": phys.phase_increment_rad_per_px,
        "E_total": float(total.item()),
        "E_left": float(e_left.item()),
        "E_right": float(e_right.item()),
        "E_outside": float(e_outside.item()),
        "E_left_ratio": float((e_left / (total + EPS)).item()),
        "E_right_ratio": float((e_right / (total + EPS)).item()),
        "E_outside_ratio": float((e_outside / (total + EPS)).item()),
        "centroid_y": cy,
        "centroid_x": cx,
        "target_center_y": ty,
        "target_center_x": tx,
        "centroid_error_px": err,
        "centroid_error_y_px": err_y,
        "centroid_error_x_px": err_x,
        "edge_energy_ratio": edge_energy_ratio(image_intensity),
        "other_to_target_energy_ratio": other_to_target,
    }


METRIC_FIELDS = [
    "plane_name",
    "input_type",
    "case_name",
    "target_side",
    "use_detilt",
    "prompt_slope_sign",
    "wavelength_nm",
    "pixel_size_um",
    "canvas_height",
    "canvas_width",
    "prompt_to_first_layer_cm",
    "inter_layer_cm",
    "shift_pixels",
    "shift_mm",
    "steering_angle_deg",
    "grating_period_px",
    "phase_increment_rad_per_px",
    "E_total",
    "E_left",
    "E_right",
    "E_outside",
    "E_left_ratio",
    "E_right_ratio",
    "E_outside_ratio",
    "centroid_y",
    "centroid_x",
    "target_center_y",
    "target_center_x",
    "centroid_error_px",
    "centroid_error_y_px",
    "centroid_error_x_px",
    "edge_energy_ratio",
    "other_to_target_energy_ratio",
]


def write_metrics_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def downsample_for_plot(data: np.ndarray, max_plot_dim: int) -> np.ndarray:
    """Downsample only the displayed image; all metrics are computed before this."""
    if max_plot_dim <= 0:
        return data
    height, width = data.shape[-2], data.shape[-1]
    stride = int(math.ceil(max(height, width) / float(max_plot_dim)))
    if stride <= 1:
        return data
    return data[::stride, ::stride]


def plot_intensity(
    field_or_intensity: torch.Tensor,
    path: Path,
    metric: Dict,
    layout: Layout,
    case_name: str,
    plane_name: str,
    target_side: Optional[str],
    save_linear: bool = False,
    plot_dpi: int = 90,
    max_plot_dim: int = 1000,
) -> None:
    image_intensity = field_or_intensity
    if torch.is_complex(image_intensity):
        image_intensity = intensity(image_intensity)
    if image_intensity.ndim == 3:
        image_intensity = image_intensity[0]
    data = image_intensity.detach().cpu().float().numpy()
    norm = data / (data.max() + EPS)
    plot_data = np.log10(norm + 1e-8)

    def draw(data_to_show: np.ndarray, out_path: Path, title_suffix: str) -> None:
        display_data = downsample_for_plot(data_to_show, max_plot_dim)
        fig, ax = plt.subplots(figsize=(12, 6))
        # extent keeps overlays in full-resolution pixel coordinates even when the
        # rendered image is downsampled for faster PNG writing.
        height, width = layout.canvas_shape
        im = ax.imshow(display_data, cmap="inferno", origin="upper", extent=(0, width, height, 0))
        add_overlay(ax, layout, metric, target_side)
        ax.set_title(
            f"{case_name} | {plane_name} | {title_suffix}\n"
            f"centroid=({metric['centroid_y']:.1f}, {metric['centroid_x']:.1f}) | "
            f"E_left={metric['E_left_ratio']:.3f} | E_right={metric['E_right_ratio']:.3f}"
        )
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.subplots_adjust(left=0.04, right=0.92, bottom=0.06, top=0.86)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=plot_dpi)
        plt.close(fig)

    draw(plot_data, path, "log10(I/Imax+1e-8)")
    if save_linear:
        draw(norm, path.with_name(path.stem + "_linear.png"), "linear I/Imax")


def add_overlay(ax, layout: Layout, metric: Dict, target_side: Optional[str]) -> None:
    for aperture, color, label in [
        (layout.left, "cyan", "left"),
        (layout.right, "lime", "right"),
    ]:
        ax.add_patch(
            Rectangle(
                (aperture.x0, aperture.y0),
                aperture.x1 - aperture.x0,
                aperture.y1 - aperture.y0,
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                label=label,
            )
        )
    ax.axvline(layout.center_x, color="white", linestyle="--", linewidth=0.8)
    ty, tx = target_center(layout, target_side)
    ax.scatter([tx], [ty], marker="x", s=70, c="yellow", label="target")
    ax.scatter([metric["centroid_x"]], [metric["centroid_y"]], marker="+", s=90, c="red", label="centroid")
    ax.set_xlim(0, layout.canvas_shape[1])
    ax.set_ylim(layout.canvas_shape[0], 0)
    ax.legend(loc="upper right", fontsize=7)


def plot_phase(phase: torch.Tensor, path: Path, title: str, plot_dpi: int = 90, max_plot_dim: int = 1000) -> None:
    wrapped = torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()
    wrapped = downsample_for_plot(wrapped, max_plot_dim)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(wrapped, cmap="twilight", origin="upper", vmin=0.0, vmax=2.0 * math.pi)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.04, top=0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def save_target_crop(
    field_or_intensity: torch.Tensor,
    path: Path,
    layout: Layout,
    target_side: Optional[str],
    plot_dpi: int = 90,
) -> None:
    if target_side not in {"left", "right"}:
        return
    image_intensity = field_or_intensity
    if torch.is_complex(image_intensity):
        image_intensity = intensity(image_intensity)
    if image_intensity.ndim == 3:
        image_intensity = image_intensity[0]
    aperture = layout.right if target_side == "right" else layout.left
    crop = image_intensity[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
    crop_np = crop.detach().cpu().float().numpy()
    crop_np = np.log10(crop_np / (crop_np.max() + EPS) + 1e-8)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(crop_np, cmap="inferno", origin="upper")
    ax.set_title(f"Target crop: {target_side}")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plot_dpi)
    plt.close(fig)


def propagate_to_first_layer(
    field: torch.Tensor,
    prompt_phase: torch.Tensor,
    prop_input_to_prompt: AngularSpectrumPropagator,
    prop_prompt_to_first: AngularSpectrumPropagator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    after_input_to_prompt = prop_input_to_prompt(field)
    after_prompt = after_input_to_prompt * torch.exp(1j * prompt_phase).to(torch.complex64)
    first_layer = prop_prompt_to_first(after_prompt)
    return after_input_to_prompt, after_prompt, first_layer


def calibrate_prompt_slope_sign(args, layout: Layout, phys: PhysicalParams, device: torch.device, output_dir: Path) -> int:
    field = make_input_field("gaussian", layout, device)
    prop_input_to_prompt = make_propagator(args, args.input_to_prompt_cm, device)
    prop_prompt_to_first = make_propagator(args, args.prompt_to_first_layer_cm, device)
    results = []

    for sign in [1, -1]:
        phase = build_linear_grating_phase(
            layout.canvas_shape,
            phys.grating_period_px,
            direction="right",
            slope_sign=sign,
            device=device,
        )
        _, _, first_layer = propagate_to_first_layer(field, phase, prop_input_to_prompt, prop_prompt_to_first)
        c_y, c_x = weighted_centroid(intensity(first_layer)[0])
        results.append({"prompt_slope_sign": sign, "centroid_y": c_y, "centroid_x": c_x})

    right_x = layout.right.center[1]
    chosen = min(results, key=lambda row: abs(row["centroid_x"] - right_x))["prompt_slope_sign"]
    calibration_dir = output_dir / "slope_calibration"
    write_json(calibration_dir / "calibration.json", {"results": results, "chosen_right_slope_sign": chosen})
    return int(chosen)


def run_case(
    args,
    layout: Layout,
    phys: PhysicalParams,
    device: torch.device,
    input_type: str,
    target_side: Optional[str],
    prompt_slope_sign: int,
    use_detilt: bool,
    case_name: str,
    output_dir: Path,
    wrong_detilt: bool = False,
    prompt_multiplier: float = 1.0,
    detilt_multiplier: float = 1.0,
    actual_prompt_to_first_layer_cm: Optional[float] = None,
    left_aperture: Optional[Aperture] = None,
    right_aperture: Optional[Aperture] = None,
    save_outputs: bool = True,
) -> List[Dict]:
    left_aperture = left_aperture or layout.left
    right_aperture = right_aperture or layout.right
    prompt_to_first_cm = actual_prompt_to_first_layer_cm or args.prompt_to_first_layer_cm

    field = make_input_field(input_type, layout, device)
    if target_side in {"right", "left"}:
        prompt_phase = build_linear_grating_phase(
            layout.canvas_shape,
            phys.grating_period_px,
            direction=target_side,
            slope_sign=prompt_slope_sign,
            multiplier=prompt_multiplier,
            device=device,
        )
    else:
        prompt_phase = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        prompt_slope_sign = 0

    prop_input_to_prompt = make_propagator(args, args.input_to_prompt_cm, device)
    prop_prompt_to_first = make_propagator(args, prompt_to_first_cm, device)
    prop_inter = make_propagator(args, args.inter_layer_cm, device)

    case_dir = output_dir / input_type / case_name
    rows = []

    def record(plane_name: str, plane_field, file_name: str) -> Dict:
        row = compute_metrics(
            plane_field,
            plane_name,
            input_type,
            case_name,
            target_side,
            use_detilt,
            prompt_slope_sign,
            args,
            layout,
            phys,
        )
        rows.append(row)
        if save_outputs:
            plot_intensity(
                plane_field,
                case_dir / file_name,
                row,
                layout,
                case_name,
                plane_name,
                target_side,
                save_linear=args.save_linear_intensity,
                plot_dpi=args.plot_dpi,
                max_plot_dim=args.max_plot_dim,
            )
        return row

    record("input_intensity", field, "00_input_intensity.png")
    if save_outputs:
        plot_phase(
            prompt_phase,
            case_dir / "01_prompt_phase.png",
            f"{case_name} | prompt phase wrapped [0, 2pi)",
            plot_dpi=args.plot_dpi,
            max_plot_dim=args.max_plot_dim,
        )

    after_input_to_prompt, after_prompt, first_before = propagate_to_first_layer(
        field, prompt_phase, prop_input_to_prompt, prop_prompt_to_first
    )
    record("after_input_to_prompt", after_input_to_prompt, "02_after_input_to_prompt_intensity.png")
    record("first_layer_before_detilt", first_before, "03_first_layer_before_detilt_intensity.png")

    aperture = right_aperture if target_side == "right" else left_aperture
    aperture_union = build_expert_aperture_mask(layout.canvas_shape, left_aperture, right_aperture, device=device)
    field_after = first_before * aperture_union.to(torch.complex64)
    if use_detilt and target_side in {"right", "left"}:
        detilt_phase = build_detilt_phase_for_aperture(
            layout.canvas_shape,
            aperture,
            phys.grating_period_px,
            direction=target_side,
            prompt_slope_sign=prompt_slope_sign,
            wrong_sign=wrong_detilt,
            multiplier=detilt_multiplier,
            device=device,
        )
        field_after = field_after * torch.exp(1j * detilt_phase).to(torch.complex64)

    record("first_layer_after_detilt", field_after, "04_first_layer_after_detilt_intensity.png")

    current = field_after
    for layer_idx in range(2, args.num_dummy_layers + 1):
        current = prop_inter(current)
        current = current * aperture_union.to(torch.complex64)
        record(f"layer{layer_idx}_intensity", current, f"{layer_idx + 3:02d}_layer{layer_idx}_intensity.png")

    detector = prop_inter(current)
    record("detector_plane_intensity", detector, "09_detector_plane_intensity.png")

    if save_outputs:
        save_target_crop(first_before, case_dir / "10_first_layer_target_crop.png", layout, target_side, args.plot_dpi)
        save_target_crop(detector, case_dir / "11_detector_target_crop.png", layout, target_side, args.plot_dpi)
        write_metrics_csv(case_dir / "metrics.csv", rows)
        write_json(case_dir / "metrics.json", rows)

    return rows


def plane(rows: List[Dict], name: str) -> Optional[Dict]:
    for row in rows:
        if row["plane_name"] == name:
            return row
    return None


def evaluate_case_status(case_name: str, rows: List[Dict], layout: Layout, phys: PhysicalParams) -> Dict:
    first = plane(rows, "first_layer_before_detilt")
    after = plane(rows, "first_layer_after_detilt")
    layer2 = plane(rows, "layer2_intensity")
    layer3 = plane(rows, "layer3_intensity")
    layer4 = plane(rows, "layer4_intensity")
    layer5 = plane(rows, "layer5_intensity")
    final = plane(rows, "detector_plane_intensity")
    status = {"case_name": case_name, "passed": False, "details": ""}

    if case_name == "case0_no_grating":
        center_ok = abs(first["centroid_x"] - layout.center_x) < 20 and abs(first["centroid_y"] - layout.center_y) < 20
        expert_energy_ok = max(first["E_left_ratio"], first["E_right_ratio"]) < 0.1
        status["passed"] = bool(center_ok and expert_energy_ok)
        status["details"] = f"center_ok={center_ok}, expert_energy_ok={expert_energy_ok}"
    elif case_name == "case1_right_no_detilt":
        first_ok = abs(first["centroid_x"] - layout.right.center[1]) < 20
        drift_values = [
            layer2["centroid_x"] - after["centroid_x"],
            layer3["centroid_x"] - layer2["centroid_x"],
            layer4["centroid_x"] - layer3["centroid_x"],
            layer5["centroid_x"] - layer4["centroid_x"],
        ]
        drift_ok = drift_values[0] > 30 and max(drift_values[:2]) > 30
        status["passed"] = bool(first_ok and drift_ok)
        status["details"] = f"first_ok={first_ok}, drift_values={drift_values}, theory={phys.drift_per_5cm_px:.1f}px"
    elif case_name == "case2_right_with_detilt":
        first_ok = abs(first["centroid_x"] - layout.right.center[1]) < 20
        xs = [layer2["centroid_x"], layer3["centroid_x"], layer4["centroid_x"], layer5["centroid_x"]]
        residual = max(abs(x - after["centroid_x"]) for x in xs)
        final_in_aperture = layout.right.x0 <= final["centroid_x"] <= layout.right.x1
        ratio_ok = final["E_right"] / (final["E_left"] + EPS) > 10
        status["passed"] = bool(first_ok and residual < 20 and final_in_aperture and ratio_ok)
        status["details"] = f"first_ok={first_ok}, max_residual_drift={residual:.1f}, final_in_aperture={final_in_aperture}, ratio_ok={ratio_ok}"
    elif case_name == "case3_left_with_detilt":
        first_ok = abs(first["centroid_x"] - layout.left.center[1]) < 20
        xs = [layer2["centroid_x"], layer3["centroid_x"], layer4["centroid_x"], layer5["centroid_x"]]
        residual = max(abs(x - after["centroid_x"]) for x in xs)
        final_in_aperture = layout.left.x0 <= final["centroid_x"] <= layout.left.x1
        ratio_ok = final["E_left"] / (final["E_right"] + EPS) > 10
        status["passed"] = bool(first_ok and residual < 20 and final_in_aperture and ratio_ok)
        status["details"] = f"first_ok={first_ok}, max_residual_drift={residual:.1f}, final_in_aperture={final_in_aperture}, ratio_ok={ratio_ok}"
    elif case_name == "case4_right_wrong_detilt":
        xs = [layer2["centroid_x"], layer3["centroid_x"], layer4["centroid_x"], layer5["centroid_x"]]
        residual = max(abs(x - after["centroid_x"]) for x in xs)
        expected_fail = residual > 30 or not (layout.right.x0 <= final["centroid_x"] <= layout.right.x1)
        status["passed"] = bool(expected_fail)
        status["details"] = f"expected_fail={expected_fail}, residual={residual:.1f}, final_x={final['centroid_x']:.1f}"
    return status


def run_required_cases(args, layout: Layout, phys: PhysicalParams, device: torch.device, output_dir: Path, right_slope_sign: int) -> Tuple[List[Dict], List[Dict]]:
    all_metrics = []
    statuses = []
    input_types = [item.strip() for item in args.input_types.split(",") if item.strip()]

    cases = [
        ("case0_no_grating", None, 0, False, False),
        ("case1_right_no_detilt", "right", right_slope_sign, False, False),
        ("case2_right_with_detilt", "right", right_slope_sign, True, False),
        ("case3_left_with_detilt", "left", right_slope_sign, True, False),
        ("case4_right_wrong_detilt", "right", right_slope_sign, True, True),
    ]

    for input_type in input_types:
        for case_name, target_side, slope_sign, use_detilt, wrong_detilt in cases:
            print(f"[case] input={input_type} case={case_name} detilt={use_detilt}")
            rows = run_case(
                args,
                layout,
                phys,
                device,
                input_type=input_type,
                target_side=target_side,
                prompt_slope_sign=slope_sign,
                use_detilt=use_detilt,
                case_name=case_name,
                output_dir=output_dir,
                wrong_detilt=wrong_detilt,
                save_outputs=True,
            )
            all_metrics.extend(rows)
            if input_type == "gaussian":
                statuses.append(evaluate_case_status(case_name, rows, layout, phys))
            print(f"[case] finished input={input_type} case={case_name}")

    write_metrics_csv(output_dir / "metrics.csv", all_metrics)
    write_json(output_dir / "metrics.json", all_metrics)
    return all_metrics, statuses


def run_alignment_sweep(args, layout: Layout, phys: PhysicalParams, device: torch.device, output_dir: Path, right_slope_sign: int) -> List[Dict]:
    sweep_dir = output_dir / "alignment_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("[sweep] running distance, prompt-slope, de-tilt-slope, and aperture alignment sweeps")

    field = make_input_field("gaussian", layout, device)
    prop_input = make_propagator(args, args.input_to_prompt_cm, device)
    after_input = prop_input(field)

    def first_layer_centroid(prompt_multiplier: float = 1.0, z_cm: float = 24.0):
        prop_first = make_propagator(args, z_cm, device)
        phase = build_linear_grating_phase(
            layout.canvas_shape,
            phys.grating_period_px,
            "right",
            slope_sign=right_slope_sign,
            multiplier=prompt_multiplier,
            device=device,
        )
        first = prop_first(after_input * torch.exp(1j * phase).to(torch.complex64))
        cy, cx = weighted_centroid(intensity(first)[0])
        return cy, cx

    for z_cm in [23.0, 23.5, 24.0, 24.5, 25.0]:
        cy, cx = first_layer_centroid(z_cm=z_cm)
        rows.append({
            "sweep": "prompt_to_first_layer_distance_cm",
            "value": z_cm,
            "centroid_y": cy,
            "centroid_x": cx,
            "centroid_error_x_px": cx - layout.right.center[1],
            "centroid_error_px": math.sqrt((cy - layout.right.center[0]) ** 2 + (cx - layout.right.center[1]) ** 2),
        })

    for multiplier in [0.95, 0.98, 1.00, 1.02, 1.05]:
        cy, cx = first_layer_centroid(prompt_multiplier=multiplier, z_cm=args.prompt_to_first_layer_cm)
        rows.append({
            "sweep": "prompt_grating_slope_multiplier",
            "value": multiplier,
            "centroid_y": cy,
            "centroid_x": cx,
            "centroid_error_x_px": cx - layout.right.center[1],
            "centroid_error_px": math.sqrt((cy - layout.right.center[0]) ** 2 + (cx - layout.right.center[1]) ** 2),
        })

    for multiplier in [0.95, 0.98, 1.00, 1.02, 1.05]:
        rows_case = run_case(
            args,
            layout,
            phys,
            device,
            input_type="gaussian",
            target_side="right",
            prompt_slope_sign=right_slope_sign,
            use_detilt=True,
            case_name=f"sweep_detilt_{multiplier:.2f}",
            output_dir=sweep_dir,
            detilt_multiplier=multiplier,
            save_outputs=False,
        )
        after = plane(rows_case, "first_layer_after_detilt")
        layer5 = plane(rows_case, "layer5_intensity")
        rows.append({
            "sweep": "detilt_slope_multiplier",
            "value": multiplier,
            "centroid_y": layer5["centroid_y"],
            "centroid_x": layer5["centroid_x"],
            "centroid_error_x_px": layer5["centroid_x"] - after["centroid_x"],
            "centroid_error_px": abs(layer5["centroid_x"] - after["centroid_x"]),
        })

    # Aperture misalignment uses cached first-layer field and shifted masks. This checks clipping
    # without running hundreds of full FFT propagation paths.
    prop_first = make_propagator(args, args.prompt_to_first_layer_cm, device)
    phase = build_linear_grating_phase(
        layout.canvas_shape,
        phys.grating_period_px,
        "right",
        slope_sign=right_slope_sign,
        device=device,
    )
    first = prop_first(after_input * torch.exp(1j * phase).to(torch.complex64))
    first_i = intensity(first)[0]
    nominal_target_energy = aperture_energy(first_i, layout.right)
    for dy in [0, -5, 5, -10, 10, -20, 20, -50, 50]:
        for dx in [0, -5, 5, -10, 10, -20, 20, -50, 50]:
            shifted_right = layout.right.shifted(dy=dy, dx=dx)
            e_target = aperture_energy(first_i, shifted_right)
            masked = first_i * aperture_mask(layout.canvas_shape, shifted_right, device=device)
            cy, cx = weighted_centroid(masked)
            rows.append({
                "sweep": "aperture_misalignment",
                "value": f"dy={dy},dx={dx}",
                "centroid_y": cy,
                "centroid_x": cx,
                "centroid_error_x_px": cx - shifted_right.center[1],
                "centroid_error_px": math.sqrt((cy - shifted_right.center[0]) ** 2 + (cx - shifted_right.center[1]) ** 2),
                "E_target_ratio_vs_nominal": float((e_target / (nominal_target_energy + EPS)).item()),
                "aperture_clipped": bool((e_target / (nominal_target_energy + EPS)).item() < 0.9),
            })

    write_sweep_csv(sweep_dir / "alignment_sweep.csv", rows)
    write_json(sweep_dir / "alignment_sweep.json", rows)
    plot_sweep_summary(rows, sweep_dir / "alignment_sweep_summary.png")
    write_sweep_summary(rows, sweep_dir / "alignment_sweep_summary.md", phys)
    return rows


def write_sweep_csv(path: Path, rows: List[Dict]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_sweep_summary(rows: List[Dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.reshape(-1)
    sweeps = [
        "prompt_to_first_layer_distance_cm",
        "prompt_grating_slope_multiplier",
        "detilt_slope_multiplier",
        "aperture_misalignment",
    ]
    for ax, sweep in zip(axes, sweeps):
        subset = [row for row in rows if row["sweep"] == sweep]
        if not subset:
            ax.axis("off")
            continue
        if sweep == "aperture_misalignment":
            values = np.arange(len(subset))
            y = [row.get("E_target_ratio_vs_nominal", np.nan) for row in subset]
            ax.scatter(values, y, s=12)
            ax.set_ylabel("E target / nominal")
        else:
            values = [float(row["value"]) for row in subset]
            y = [float(row["centroid_error_x_px"]) for row in subset]
            ax.plot(values, y, marker="o")
            ax.set_ylabel("centroid error x (px)")
        ax.set_title(sweep)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_sweep_summary(rows: List[Dict], path: Path, phys: PhysicalParams) -> None:
    lines = [
        "# Alignment Sweep Summary",
        "",
        f"Theoretical 1 cm z-error shift is about {phys.drift_per_5cm_px / 5.0:.1f} px.",
        f"Theoretical 5 cm drift without de-tilt is {phys.drift_per_5cm_px:.1f} px.",
        "",
        "See `alignment_sweep.csv` for detailed values.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    output_dir: Path,
    args,
    phys: PhysicalParams,
    statuses: List[Dict],
    right_slope_sign: int,
    all_metrics: List[Dict],
) -> None:
    final_edge_warning = any(row["edge_energy_ratio"] > 0.05 for row in all_metrics)
    right_no_detilt = next((row for row in statuses if row["case_name"] == "case1_right_no_detilt"), None)
    right_detilt = next((row for row in statuses if row["case_name"] == "case2_right_with_detilt"), None)
    left_detilt = next((row for row in statuses if row["case_name"] == "case3_left_with_detilt"), None)

    summary = {
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "wavelength_nm": phys.wavelength_nm,
        "pixel_size_um": phys.pixel_size_um,
        "shift_pixels": phys.shift_pixels,
        "shift_mm": phys.shift_mm,
        "steering_angle_deg": phys.steering_angle_deg,
        "grating_period_px": phys.grating_period_px,
        "phase_increment_rad_per_px": phys.phase_increment_rad_per_px,
        "drift_per_5cm_px": phys.drift_per_5cm_px,
        "chosen_right_prompt_slope_sign": right_slope_sign,
        "statuses": statuses,
        "edge_energy_warning": final_edge_warning,
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Prompt Grating Alignment Test Summary",
        "",
        "## Test Parameters",
        "",
        f"- output directory: `{output_dir}`",
        f"- canvas: {args.canvas_height} x {args.canvas_width}",
        f"- wavelength: {phys.wavelength_nm} nm",
        f"- pixel size: {phys.pixel_size_um} um",
        f"- prompt-to-first-layer distance: {phys.prompt_to_first_layer_cm} cm",
        f"- inter-layer distance: {phys.inter_layer_cm} cm",
        "",
        "## Theory",
        "",
        f"- required shift: {phys.shift_pixels:.1f} px = {phys.shift_mm:.3f} mm",
        f"- steering angle: {phys.steering_angle_deg:.3f} deg",
        f"- grating period: {phys.grating_period_px:.3f} px",
        f"- phase increment: {phys.phase_increment_rad_per_px:.3f} rad/px",
        f"- expected no-detilt drift per 5 cm: {phys.drift_per_5cm_px:.1f} px",
        f"- chosen right prompt slope sign after calibration: {right_slope_sign}",
        "",
        "## Case Results",
        "",
    ]
    for status in statuses:
        verdict = "PASS" if status["passed"] else "FAIL"
        lines.append(f"- {status['case_name']}: {verdict} | {status['details']}")

    lines.extend(
        [
            "",
            "## F-pattern / Digit-like Orientation Check",
            "",
            "- Non-Gaussian inputs are saved with target crops for manual inspection.",
            "- The F-pattern should not be left-right mirrored, up-down flipped, or x/y transposed in the target crop.",
            "- These inputs intentionally do not use the Gaussian energy-ratio pass/fail rules as the only criterion.",
            "",
            "## Diagnostics",
            "",
            "- If right and left are swapped, the grating sign convention is wrong.",
            "- If the F pattern is mirrored or transposed, inspect x/y indexing and image origin.",
            "- If no-detilt does not drift, the prompt may be producing translation-like clipping rather than angular steering.",
            "- If with-detilt still drifts by about 83 px per 5 cm, the de-tilt sign is wrong or not applied in the aperture.",
            "- If edge energy is high, increase canvas padding or reduce propagation distance to avoid FFT wrap-around contamination.",
            f"- edge energy warning: {final_edge_warning}",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nPrompt grating alignment summary")
    print(f"output directory: {output_dir}")
    print(f"computed grating period: {phys.grating_period_px:.3f} px")
    print(f"steering angle: {phys.steering_angle_deg:.3f} deg")
    print(f"right no-detilt drift theory: {phys.drift_per_5cm_px:.1f} px / 5 cm")
    print(f"right no-detilt status: {right_no_detilt['passed'] if right_no_detilt else 'missing'}")
    print(f"right with-detilt pass/fail: {right_detilt['passed'] if right_detilt else 'missing'}")
    print(f"left with-detilt pass/fail: {left_detilt['passed'] if left_detilt else 'missing'}")
    print(f"edge energy warning: {final_edge_warning}")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = build_layout(args)
    phys = compute_physical_params(args, layout)
    right_slope_sign = calibrate_prompt_slope_sign(args, layout, phys, device, output_dir)

    all_metrics, statuses = run_required_cases(args, layout, phys, device, output_dir, right_slope_sign)
    if not args.skip_sweeps:
        run_alignment_sweep(args, layout, phys, device, output_dir, right_slope_sign)

    write_summary(output_dir, args, phys, statuses, right_slope_sign, all_metrics)


if __name__ == "__main__":
    main()
