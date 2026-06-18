import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
from opticalmoe.utils.seed import set_seed
from opticalmoe.utils.units import cm_to_m, nm_to_m, um_to_m


EPS = 1e-12
EXPERT_IDS = ["E00", "E01", "E02", "E10", "E11", "E12", "E20", "E21", "E22"]
CELL_IDS = ["C00", "C01", "C02", "C10", "C11", "C12", "C20", "C21", "C22"]


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
        return int(self.y1 - self.y0)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

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
class NineExpertLayout:
    canvas_height: int
    canvas_width: int
    input_size: int
    expert_size: int
    prompt_cell_size: int
    center_coords: List[int]
    input_aperture: Aperture
    expert_apertures: List[Aperture]
    prompt_cells: List[Aperture]

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
            "expert_size": self.expert_size,
            "prompt_cell_size": self.prompt_cell_size,
            "expert_center_coords": list(self.center_coords),
            "expert_apertures": [item.to_dict() for item in self.expert_apertures],
            "prompt_cells": [item.to_dict() for item in self.prompt_cells],
            "four_expert_baseline_area": baseline_area,
            "nine_expert_matched_area": nine_area,
            "relative_area_difference": (nine_area - baseline_area) / float(baseline_area),
        }


class NineCellMicrolensPrompt:
    """Spatially partitioned 9-cell microlens-array prompt.

    Each prompt pixel belongs to at most one cell. The final transmission is a
    masked sum of local thin-lens + local grating phases with user-specified
    scalar amplitudes. This is not a global grating fan-out.
    """

    def __init__(
        self,
        layout: NineExpertLayout,
        wavelength_m: float,
        pixel_size_m: float,
        focal_length_m: float,
        input_to_prompt_m: float,
        amplitudes: Sequence[float],
        phase_biases: Optional[Sequence[float]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if len(amplitudes) != 9:
            raise ValueError("amplitudes must contain exactly 9 values.")
        if any(float(value) < 0.0 for value in amplitudes):
            raise ValueError("amplitudes must be non-negative.")
        if phase_biases is None:
            phase_biases = [0.0] * 9
        if len(phase_biases) != 9:
            raise ValueError("phase_biases must contain exactly 9 values.")
        self.layout = layout
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.focal_length_m = float(focal_length_m)
        self.input_to_prompt_m = float(input_to_prompt_m)
        self.amplitudes = torch.tensor(amplitudes, dtype=torch.float32, device=device)
        self.phase_biases = torch.tensor(phase_biases, dtype=torch.float32, device=device)
        self.device = device

        y_grid, x_grid = physical_grids(layout, pixel_size_m, device)
        masks = prompt_cell_masks(layout, device)
        lens_phases = []
        grating_phases = []
        cell_reports = []
        for index, cell in enumerate(layout.prompt_cells):
            center_y_px, center_x_px = cell.center
            canvas_y, canvas_x = layout.canvas_center
            offset_y_m = (center_y_px - canvas_y) * pixel_size_m
            offset_x_m = (center_x_px - canvas_x) * pixel_size_m
            x_local = x_grid - offset_x_m
            y_local = y_grid - offset_y_m

            lens_phase = (
                -math.pi
                / (self.wavelength_m * self.focal_length_m)
                * (x_local ** 2 + y_local ** 2)
            )
            theta_x = math.atan(-offset_x_m / self.input_to_prompt_m)
            theta_y = math.atan(-offset_y_m / self.input_to_prompt_m)
            fx = math.sin(theta_x) / self.wavelength_m
            fy = math.sin(theta_y) / self.wavelength_m
            grating_phase = 2.0 * math.pi * (fx * x_local + fy * y_local)

            lens_phases.append(lens_phase * masks[index])
            grating_phases.append(grating_phase * masks[index])
            cell_reports.append(
                {
                    "cell_id": CELL_IDS[index],
                    "expert_id": EXPERT_IDS[index],
                    "center_y_px": center_y_px,
                    "center_x_px": center_x_px,
                    "offset_y_px": center_y_px - canvas_y,
                    "offset_x_px": center_x_px - canvas_x,
                    "theta_y_deg": math.degrees(theta_y),
                    "theta_x_deg": math.degrees(theta_x),
                    "grating_period_y_px": period_px(fy, pixel_size_m),
                    "grating_period_x_px": period_px(fx, pixel_size_m),
                }
            )
        self.masks = masks
        self.lens_phases = torch.stack(lens_phases, dim=0)
        self.grating_phases = torch.stack(grating_phases, dim=0)
        self.cell_reports = cell_reports

    def commanded_power(self) -> torch.Tensor:
        return self.amplitudes.square()

    def normalized_commanded_power(self) -> torch.Tensor:
        power = self.commanded_power()
        return power / (power.sum() + EPS)

    def amplitude_map(self) -> torch.Tensor:
        return torch.sum(self.masks * self.amplitudes.view(9, 1, 1), dim=0)

    def lens_phase_map(self) -> torch.Tensor:
        return torch.sum(self.lens_phases, dim=0)

    def grating_phase_map(self) -> torch.Tensor:
        return torch.sum(self.grating_phases, dim=0)

    def phase_map(self) -> torch.Tensor:
        phase = self.lens_phases + self.grating_phases
        phase = phase + self.masks * self.phase_biases.view(9, 1, 1)
        return torch.sum(phase, dim=0)

    def transmission(self) -> torch.Tensor:
        phase = self.lens_phases + self.grating_phases
        phase = phase + self.masks * self.phase_biases.view(9, 1, 1)
        cells = (
            self.masks
            * self.amplitudes.view(9, 1, 1)
            * torch.exp(1j * phase).to(torch.complex64)
        )
        return torch.sum(cells, dim=0).to(torch.complex64)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field.to(torch.complex64) * self.transmission().unsqueeze(0)


def period_px(spatial_frequency_per_m: float, pixel_size_m: float) -> float:
    if abs(spatial_frequency_per_m) < 1e-20:
        return float("inf")
    return 1.0 / (abs(spatial_frequency_per_m) * pixel_size_m)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Composite 9-cell microlens prompt propagation test."
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--canvas_height", type=int, default=700)
    parser.add_argument("--canvas_width", type=int, default=700)
    parser.add_argument("--input_size", type=int, default=200)
    parser.add_argument("--expert_size", type=int, default=134)
    parser.add_argument("--expert_center_coords", default="167,350,533")
    parser.add_argument("--prompt_cell_size", type=int, default=180)

    parser.add_argument(
        "--geometry_preset",
        default="strict_training_geometry",
        choices=["strict_training_geometry", "matched_magnification_134"],
    )
    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--input_to_prompt_cm", type=float, default=None)
    parser.add_argument("--prompt_to_expert_cm", type=float, default=None)
    parser.add_argument("--focal_length_cm", type=float, default=None)
    parser.add_argument("--inter_layer_cm", type=float, default=5.0)
    parser.add_argument("--num_dummy_layers", type=int, default=5)

    parser.add_argument("--input_types", default="gaussian,f_pattern,digit_like")
    parser.add_argument(
        "--amplitude_cases",
        default=(
            "uniform,center_only,corner_only_E00,corner_only_E22,top_row,"
            "left_col,diagonal,sparse_mix,random_seeded,task_like_mnist,"
            "task_like_fashion,task_like_emnist"
        ),
    )
    parser.add_argument("--custom_amplitudes", default=None)

    parser.add_argument("--hard_aperture", dest="hard_aperture", action="store_true", default=True)
    parser.add_argument("--no_hard_aperture", dest="hard_aperture", action="store_false")
    parser.add_argument("--save_linear_intensity", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=120)
    parser.add_argument("--max_plot_dim", type=int, default=1400)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("expert_center_coords must contain exactly three integers.")
    return values


def parse_float_list(text: str, expected: int) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError("Expected %d comma-separated values." % expected)
    return values


def resolve_geometry(args) -> Dict:
    if args.geometry_preset == "strict_training_geometry":
        input_to_prompt_cm = 20.0
        prompt_to_expert_cm = 20.0
        focal_length_cm = 10.0
    elif args.geometry_preset == "matched_magnification_134":
        input_to_prompt_cm = 20.0
        prompt_to_expert_cm = 13.4
        focal_length_cm = 8.02
    else:
        raise ValueError("Unsupported geometry preset.")
    if args.input_to_prompt_cm is not None:
        input_to_prompt_cm = float(args.input_to_prompt_cm)
    if args.prompt_to_expert_cm is not None:
        prompt_to_expert_cm = float(args.prompt_to_expert_cm)
    if args.focal_length_cm is not None:
        focal_length_cm = float(args.focal_length_cm)
    magnification = prompt_to_expert_cm / input_to_prompt_cm
    expected_image_size_px = args.input_size * magnification
    return {
        "geometry_preset": args.geometry_preset,
        "input_to_prompt_cm": input_to_prompt_cm,
        "prompt_to_expert_cm": prompt_to_expert_cm,
        "focal_length_cm": focal_length_cm,
        "inter_layer_cm": args.inter_layer_cm,
        "input_to_prompt_m": cm_to_m(input_to_prompt_cm),
        "prompt_to_expert_m": cm_to_m(prompt_to_expert_cm),
        "focal_length_m": cm_to_m(focal_length_cm),
        "inter_layer_m": cm_to_m(args.inter_layer_cm),
        "magnification": magnification,
        "expected_image_size_px": expected_image_size_px,
        "expected_image_to_expert_ratio": expected_image_size_px / float(args.expert_size),
        "num_dummy_layers": args.num_dummy_layers,
    }


def build_layout(args) -> NineExpertLayout:
    centers = parse_int_list(args.expert_center_coords)
    input_half = args.input_size // 2
    canvas_center_y = args.canvas_height // 2
    canvas_center_x = args.canvas_width // 2
    input_aperture = Aperture(
        "input",
        canvas_center_y - input_half,
        canvas_center_y + input_half,
        canvas_center_x - input_half,
        canvas_center_x + input_half,
    )
    expert_half = args.expert_size // 2
    cell_half = args.prompt_cell_size // 2
    experts = []
    cells = []
    for row, center_y in enumerate(centers):
        for col, center_x in enumerate(centers):
            name = "%d%d" % (row, col)
            experts.append(
                Aperture(
                    "E" + name,
                    center_y - expert_half,
                    center_y + expert_half,
                    center_x - expert_half,
                    center_x + expert_half,
                )
            )
            cells.append(
                Aperture(
                    "C" + name,
                    center_y - cell_half,
                    center_y + cell_half,
                    center_x - cell_half,
                    center_x + cell_half,
                )
            )
    layout = NineExpertLayout(
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        input_size=args.input_size,
        expert_size=args.expert_size,
        prompt_cell_size=args.prompt_cell_size,
        center_coords=centers,
        input_aperture=input_aperture,
        expert_apertures=experts,
        prompt_cells=cells,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: NineExpertLayout) -> None:
    for aperture in [layout.input_aperture] + layout.expert_apertures + layout.prompt_cells:
        if aperture.height <= 0 or aperture.width <= 0:
            raise ValueError("%s has invalid size." % aperture.name)
        if aperture.y0 < 0 or aperture.x0 < 0:
            raise ValueError("%s starts outside canvas." % aperture.name)
        if aperture.y1 > layout.canvas_height or aperture.x1 > layout.canvas_width:
            raise ValueError("%s ends outside canvas." % aperture.name)
    if torch.any(prompt_cell_masks(layout, torch.device("cpu")).sum(dim=0) > 1.0):
        raise ValueError("Prompt cells overlap. Reduce prompt_cell_size.")
    if torch.any(expert_masks(layout, torch.device("cpu")).sum(dim=0) > 1.0):
        raise ValueError("Expert apertures overlap.")


def aperture_masks(
    layout: NineExpertLayout,
    apertures: Sequence[Aperture],
    device: torch.device,
) -> torch.Tensor:
    masks = []
    for aperture in apertures:
        mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)


def expert_masks(layout: NineExpertLayout, device: torch.device) -> torch.Tensor:
    return aperture_masks(layout, layout.expert_apertures, device)


def prompt_cell_masks(layout: NineExpertLayout, device: torch.device) -> torch.Tensor:
    return aperture_masks(layout, layout.prompt_cells, device)


def physical_grids(
    layout: NineExpertLayout,
    pixel_size_m: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cy, cx = layout.canvas_center
    y = (torch.arange(layout.canvas_height, dtype=torch.float32, device=device) - cy) * pixel_size_m
    x = (torch.arange(layout.canvas_width, dtype=torch.float32, device=device) - cx) * pixel_size_m
    return torch.meshgrid(y, x, indexing="ij")


def pixel_grids(layout: NineExpertLayout, device: torch.device):
    y = torch.arange(layout.canvas_height, dtype=torch.float32, device=device)
    x = torch.arange(layout.canvas_width, dtype=torch.float32, device=device)
    return torch.meshgrid(y, x, indexing="ij")


def make_gaussian(layout: NineExpertLayout, device: torch.device) -> torch.Tensor:
    y_grid, x_grid = pixel_grids(layout, device)
    cy, cx = layout.canvas_center
    sigma = 45.0
    amplitude = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    ap = layout.input_aperture
    mask[ap.y0 : ap.y1, ap.x0 : ap.x1] = 1.0
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


def make_input_field(
    input_type: str,
    layout: NineExpertLayout,
    device: torch.device,
) -> torch.Tensor:
    amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    if input_type == "gaussian":
        amplitude = make_gaussian(layout, device)
    elif input_type in {"f_pattern", "digit_like"}:
        local = make_f_pattern(layout.input_size, device) if input_type == "f_pattern" else make_digit_like(layout.input_size, device)
        ap = layout.input_aperture
        amplitude[ap.y0 : ap.y1, ap.x0 : ap.x1] = local
    else:
        raise ValueError("Unsupported input type: %s" % input_type)
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


def amplitude_case_dict(seed: int, custom: Optional[str]) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    cases = {
        "uniform": [1.0] * 9,
        "center_only": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "corner_only_E00": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "corner_only_E22": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "top_row": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "left_col": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "diagonal": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "sparse_mix": [1.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.2, 0.0, 0.0],
        "random_seeded": [float(value) for value in rng.rand(9)],
        "task_like_mnist": [0.9, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.7],
        "task_like_fashion": [0.8, 0.8, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "task_like_emnist": [0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.7],
    }
    for index, expert_id in enumerate(EXPERT_IDS):
        values = [0.0] * 9
        values[index] = 1.0
        cases["onehot_" + expert_id] = values
    if custom:
        cases["custom"] = parse_float_list(custom, 9)
    return cases


def energy_ratios(intensity: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    total = float(intensity.sum().item())
    energies = torch.einsum("bhw,khw->bk", intensity, masks)[0]
    ratios = energies / (total + EPS)
    outside = max(0.0, total - float(energies.sum().item())) / (total + EPS)
    return ratios, outside, total


def centroid_and_radius(intensity: torch.Tensor) -> Tuple[float, float, float]:
    if intensity.ndim == 3:
        intensity = intensity[0]
    total = float(intensity.sum().item())
    if total <= EPS:
        return float("nan"), float("nan"), float("nan")
    h, w = intensity.shape
    y = torch.arange(h, dtype=torch.float32, device=intensity.device)
    x = torch.arange(w, dtype=torch.float32, device=intensity.device)
    cy = (intensity.sum(dim=1) * y).sum() / (total + EPS)
    cx = (intensity.sum(dim=0) * x).sum() / (total + EPS)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt((((yy - cy) ** 2 + (xx - cx) ** 2) * intensity).sum() / (total + EPS))
    return float(cy.item()), float(cx.item()), float(radius.item())


def expected_box_mask(
    layout: NineExpertLayout,
    expert_index: int,
    size_px: float,
    device: torch.device,
) -> torch.Tensor:
    center_y, center_x = layout.expert_apertures[expert_index].center
    half = float(size_px) / 2.0
    y0 = max(0, int(math.floor(center_y - half)))
    y1 = min(layout.canvas_height, int(math.ceil(center_y + half)))
    x0 = max(0, int(math.floor(center_x - half)))
    x1 = min(layout.canvas_width, int(math.ceil(center_x + half)))
    mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def vector_metrics(commanded, corrected, measured) -> Dict:
    commanded = np.asarray(commanded, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def pearson(a, b):
        if np.std(a) < EPS or np.std(b) < EPS:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "cosine_commanded_measured": cosine(commanded, measured),
        "cosine_corrected_measured": cosine(corrected, measured),
        "rmse_commanded_measured": rmse(commanded, measured),
        "rmse_corrected_measured": rmse(corrected, measured),
        "pearson_commanded_measured": pearson(commanded, measured),
        "pearson_corrected_measured": pearson(corrected, measured),
    }


def entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array / (array.sum() + EPS)
    return float(-(array * np.log(array + EPS)).sum())


def plot_layout(layout: NineExpertLayout, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("black")
    input_ap = layout.input_aperture
    ax.add_patch(Rectangle((input_ap.x0, input_ap.y0), input_ap.width, input_ap.height, fill=False, edgecolor="orange", linewidth=2.0, linestyle="--"))
    for cell in layout.prompt_cells:
        ax.add_patch(Rectangle((cell.x0, cell.y0), cell.width, cell.height, fill=False, edgecolor="violet", linewidth=1.1, linestyle="--"))
        cy, cx = cell.center
        ax.text(cx, cy - 16, cell.name, color="violet", ha="center", va="center", fontsize=8)
    for expert in layout.expert_apertures:
        ax.add_patch(Rectangle((expert.x0, expert.y0), expert.width, expert.height, fill=False, edgecolor="cyan", linewidth=1.8))
        cy, cx = expert.center
        ax.scatter([cx], [cy], c="white", s=14)
        ax.text(cx, cy + 14, expert.name, color="white", ha="center", va="center", fontsize=8)
    cy, cx = layout.canvas_center
    ax.axhline(cy, color="gray", linewidth=0.8)
    ax.axvline(cx, color="gray", linewidth=0.8)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_aspect("equal")
    ax.set_title("9-cell prompt / 9-expert layout")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def downsample_for_plot(array: np.ndarray, max_plot_dim: int) -> Tuple[np.ndarray, int]:
    stride = max(1, int(math.ceil(max(array.shape) / max(1, int(max_plot_dim)))))
    if stride > 1:
        return array[::stride, ::stride], stride
    return array, stride


def plot_intensity(
    intensity_or_field: torch.Tensor,
    layout: NineExpertLayout,
    path: Path,
    title: str,
    dpi: int,
    max_plot_dim: int,
    overlay_prompt: bool = False,
    overlay_experts: bool = True,
    save_linear: bool = False,
) -> None:
    if torch.is_complex(intensity_or_field):
        intensity = torch.abs(intensity_or_field.to(torch.complex64)).square()
    else:
        intensity = intensity_or_field.float()
    if intensity.ndim == 3:
        intensity = intensity[0]
    array = intensity.detach().cpu().float().numpy()
    log_array = np.log10(array / (array.max() + EPS) + 1e-8)
    _plot_array(log_array, layout, path, title, dpi, max_plot_dim, "log10(I/Imax+1e-8)", overlay_prompt, overlay_experts)
    if save_linear:
        _plot_array(array, layout, path.with_name(path.stem + "_linear.png"), title, dpi, max_plot_dim, "linear intensity", overlay_prompt, overlay_experts)


def _plot_array(
    array: np.ndarray,
    layout: NineExpertLayout,
    path: Path,
    title: str,
    dpi: int,
    max_plot_dim: int,
    label: str,
    overlay_prompt: bool,
    overlay_experts: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display, _stride = downsample_for_plot(array, max_plot_dim)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(display, cmap="inferno", extent=(0, layout.canvas_width, layout.canvas_height, 0))
    if overlay_prompt:
        for cell in layout.prompt_cells:
            ax.add_patch(Rectangle((cell.x0, cell.y0), cell.width, cell.height, fill=False, edgecolor="violet", linewidth=1.0, linestyle="--"))
    if overlay_experts:
        for expert in layout.expert_apertures:
            ax.add_patch(Rectangle((expert.x0, expert.y0), expert.width, expert.height, fill=False, edgecolor="cyan", linewidth=1.1))
            cy, cx = expert.center
            ax.text(cx, cy, expert.name, color="white", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_phase(phase: torch.Tensor, path: Path, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = torch.remainder(phase, 2.0 * math.pi).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(wrapped, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_heatmap_3x3(values: Sequence[float], path: Path, title: str, label: str, dpi: int, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(array, cmap=cmap)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])
    for r in range(3):
        for c in range(3):
            ax.text(c, r, "%.3f" % array[r, c], ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_bar(rows: List[Dict], key: str, path: Path, title: str, ylabel: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["amplitude_case"] for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.65), 4.5))
    ax.bar(np.arange(len(names)), values)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_matrix(matrix: np.ndarray, path: Path, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels(EXPERT_IDS, rotation=45, ha="right")
    ax.set_yticklabels(EXPERT_IDS)
    ax.set_xlabel("measured expert")
    ax.set_ylabel("activated prompt cell")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
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


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activated\\measured"] + EXPERT_IDS)
        for index, expert_id in enumerate(EXPERT_IDS):
            writer.writerow([expert_id] + [float(value) for value in matrix[index]])


def run_single_case(
    input_type: str,
    case_name: str,
    amplitudes: Sequence[float],
    layout: NineExpertLayout,
    physical: Dict,
    propagators: Dict,
    masks: Dict,
    args,
    output_dir: Path,
    device: torch.device,
    incident_ratios: torch.Tensor,
) -> Dict:
    prompt = NineCellMicrolensPrompt(
        layout=layout,
        wavelength_m=physical["wavelength_m"],
        pixel_size_m=physical["pixel_size_m"],
        focal_length_m=physical["focal_length_m"],
        input_to_prompt_m=physical["input_to_prompt_m"],
        amplitudes=amplitudes,
        device=device,
    )
    input_field = make_input_field(input_type, layout, device)
    after_input = propagators["input_to_prompt"](input_field)
    after_prompt = prompt.forward(after_input)
    expert_raw = propagators["prompt_to_expert"](after_prompt)
    raw_intensity = torch.abs(expert_raw).square()
    expert_ratios, outside_raw, total_raw = energy_ratios(raw_intensity, masks["experts"])
    after_aperture = expert_raw
    if args.hard_aperture:
        after_aperture = expert_raw * masks["expert_union"].unsqueeze(0).to(torch.complex64)
    field = after_aperture
    for _idx in range(int(args.num_dummy_layers)):
        field = propagators["inter_layer"](field)
        if args.hard_aperture:
            field = field * masks["expert_union"].unsqueeze(0).to(torch.complex64)
    detector = propagators["inter_layer"](field)
    detector_intensity = torch.abs(detector).square()

    commanded = prompt.normalized_commanded_power().detach().cpu().numpy()
    corrected = (prompt.commanded_power() * incident_ratios.to(device)).detach()
    corrected = (corrected / (corrected.sum() + EPS)).cpu().numpy()
    measured = expert_ratios.detach().cpu().numpy()
    vec = vector_metrics(commanded, corrected, measured)
    top_index = int(np.argmax(measured))
    second = float(np.sort(measured)[-2]) if len(measured) > 1 else 0.0
    cy, cx, radius = centroid_and_radius(raw_intensity)
    expected_box_ratios = []
    for expert_index in range(9):
        box_mask = expected_box_mask(
            layout,
            expert_index,
            physical["expected_image_size_px"],
            device,
        )
        expected_box_ratios.append(float((raw_intensity[0] * box_mask).sum().item() / (total_raw + EPS)))

    row = {
        "input_type": input_type,
        "amplitude_case": case_name,
        "geometry_preset": physical["geometry_preset"],
        "prompt_cell_size": layout.prompt_cell_size,
        "magnification": physical["magnification"],
        "expected_image_size_px": physical["expected_image_size_px"],
        "expected_image_to_expert_ratio": physical["expected_image_to_expert_ratio"],
        "total_energy_expert_entrance_raw": total_raw,
        "outside_all_experts_energy_ratio": outside_raw,
        "top_expert_id": EXPERT_IDS[top_index],
        "second_largest_expert_energy_ratio": second,
        "expert_energy_entropy": entropy(measured),
        "centroid_y": cy,
        "centroid_x": cx,
        "second_moment_radius": radius,
    }
    row.update(vec)
    for index, expert_id in enumerate(EXPERT_IDS):
        row[expert_id + "_amplitude"] = float(amplitudes[index])
        row[expert_id + "_commanded_power"] = float(prompt.commanded_power()[index].item())
        row[expert_id + "_normalized_commanded_power"] = float(commanded[index])
        row[expert_id + "_normalized_corrected_command"] = float(corrected[index])
        row[expert_id + "_energy_ratio"] = float(measured[index])
        row[expert_id + "_expected_box_energy_ratio"] = float(expected_box_ratios[index])
    if case_name.startswith("onehot_"):
        active = EXPERT_IDS.index(case_name.replace("onehot_", ""))
        row["active_expert_id"] = EXPERT_IDS[active]
        row["energy_inside_target_expert"] = float(measured[active])
        row["energy_inside_expected_image_box"] = float(expected_box_ratios[active])

    case_dir = output_dir / "figures" / input_type / case_name
    should_plot = input_type == "gaussian" or not case_name.startswith("onehot_")
    if should_plot:
        plot_heatmap_3x3(amplitudes, case_dir / "prompt_amplitude_3x3.png", case_name + " amplitudes", "amplitude", args.plot_dpi)
        plot_heatmap_3x3(commanded, case_dir / "normalized_commanded_power_3x3.png", case_name + " commanded power", "normalized power", args.plot_dpi)
        plot_heatmap_3x3(corrected, case_dir / "normalized_corrected_command_3x3.png", case_name + " corrected command", "normalized corrected command", args.plot_dpi)
        plot_heatmap_3x3(measured, case_dir / "measured_expert_energy_3x3.png", case_name + " measured expert energy", "energy ratio", args.plot_dpi)
        plot_intensity(prompt.amplitude_map(), layout, case_dir / "composite_prompt_amplitude_map.png", case_name + " prompt amplitude map", args.plot_dpi, args.max_plot_dim, overlay_prompt=True, overlay_experts=False, save_linear=False)
        plot_phase(prompt.phase_map(), case_dir / "composite_prompt_phase_wrapped.png", case_name + " composite prompt phase", args.plot_dpi)
        plot_intensity(expert_raw, layout, case_dir / "expert_entrance_raw_intensity.png", case_name + " expert entrance raw", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_intensity(after_aperture, layout, case_dir / "expert_entrance_after_aperture_intensity.png", case_name + " after hard aperture", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_intensity(detector_intensity, layout, case_dir / "detector_plane_intensity.png", case_name + " detector plane", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
    return row


def status_from_threshold(value: float, pass_value: float, warn_value: float, lower_is_better: bool) -> str:
    if lower_is_better:
        if value < pass_value:
            return "PASS"
        if value <= warn_value:
            return "WARN"
        return "FAIL"
    if value > pass_value:
        return "PASS"
    if value >= warn_value:
        return "WARN"
    return "FAIL"


def summarize(rows: List[Dict], crosstalk: np.ndarray, physical: Dict) -> Dict:
    gaussian = [row for row in rows if row["input_type"] == "gaussian"]
    uniform_rows = [row for row in gaussian if row["amplitude_case"] == "uniform"]
    uniform = uniform_rows[0] if uniform_rows else None
    if uniform:
        active_count = sum(1 for expert_id in EXPERT_IDS if uniform[expert_id + "_energy_ratio"] > 0.02)
        uniform_status = "PASS" if active_count >= 7 else ("WARN" if active_count >= 5 else "FAIL")
        outside_status = status_from_threshold(uniform["outside_all_experts_energy_ratio"], 0.35, 0.60, True)
    else:
        active_count = 0
        uniform_status = "FAIL"
        outside_status = "FAIL"
    cosine_values = [row["cosine_corrected_measured"] for row in gaussian if math.isfinite(row["cosine_corrected_measured"])]
    mean_cosine = float(np.mean(cosine_values)) if cosine_values else 0.0
    cosine_status = status_from_threshold(mean_cosine, 0.85, 0.65, False)
    diagonal_top = 0
    for index in range(9):
        if int(np.argmax(crosstalk[index])) == index:
            diagonal_top += 1
    crosstalk_status = "PASS" if diagonal_top >= 7 else ("WARN" if diagonal_top >= 5 else "FAIL")
    statuses = [uniform_status, outside_status, cosine_status, crosstalk_status]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    notes = []
    if physical["geometry_preset"] == "strict_training_geometry" and physical["expected_image_to_expert_ratio"] > 1.2:
        notes.append("Strict 1x imaging maps a 200 px input onto a 134 px expert aperture, so aperture clipping is expected.")
    if physical["geometry_preset"] == "matched_magnification_134":
        notes.append("Matched magnification should compress the 200 px input close to the 134 px expert aperture.")
    return {
        "overall_status": overall,
        "uniform_active_expert_count_gt_0p02": active_count,
        "uniform_status": uniform_status,
        "mean_cosine_corrected_vs_measured": mean_cosine,
        "cosine_corrected_status": cosine_status,
        "uniform_outside_energy_status": outside_status,
        "one_hot_diagonal_top_count": diagonal_top,
        "one_hot_crosstalk_status": crosstalk_status,
        "notes": notes,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    layout = build_layout(args)
    physical = resolve_geometry(args)
    physical["wavelength_nm"] = args.wavelength_nm
    physical["pixel_size_um"] = args.pixel_size_um
    physical["wavelength_m"] = nm_to_m(args.wavelength_nm)
    physical["pixel_size_m"] = um_to_m(args.pixel_size_um)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "nine_expert_composite_microlens" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_area = 4 * 200 * 200
    nine_area = 9 * args.expert_size * args.expert_size
    print("9-expert composite microlens prompt test")
    print("device: %s" % device)
    print("output_dir: %s" % output_dir)
    print("4-expert baseline area = 4 x 200 x 200 = %d" % baseline_area)
    print("9-expert matched area  = 9 x %d x %d = %d" % (args.expert_size, args.expert_size, nine_area))
    print("relative difference = %+0.2f%%" % ((nine_area - baseline_area) / float(baseline_area) * 100.0))
    print("geometry preset: %s" % physical["geometry_preset"])
    print("input_to_prompt_cm: %.3f" % physical["input_to_prompt_cm"])
    print("prompt_to_expert_cm: %.3f" % physical["prompt_to_expert_cm"])
    print("focal_length_cm: %.3f" % physical["focal_length_cm"])
    print("magnification: %.4f" % physical["magnification"])
    print("expected_image_size_px: %.2f" % physical["expected_image_size_px"])
    print("expected_image_size_px / expert_size: %.3f" % physical["expected_image_to_expert_ratio"])

    save_json(output_dir / "layout.json", layout.to_dict())
    save_json(output_dir / "physical_params.json", physical)
    plot_layout(layout, output_dir / "figures" / "layout_overlay.png", args.plot_dpi)

    propagators = {
        "input_to_prompt": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["input_to_prompt_m"], device),
        "prompt_to_expert": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["prompt_to_expert_m"], device),
        "inter_layer": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["inter_layer_m"], device),
    }
    masks = {
        "experts": expert_masks(layout, device),
        "prompt_cells": prompt_cell_masks(layout, device),
    }
    masks["expert_union"] = torch.clamp(masks["experts"].sum(dim=0), 0.0, 1.0)

    input_types = [item.strip() for item in args.input_types.split(",") if item.strip()]
    requested_cases = [item.strip() for item in args.amplitude_cases.split(",") if item.strip()]
    all_cases = amplitude_case_dict(args.seed, args.custom_amplitudes)
    onehot_cases = ["onehot_" + expert_id for expert_id in EXPERT_IDS]
    cases_to_run = []
    for name in requested_cases + onehot_cases:
        if name not in all_cases:
            raise ValueError("Unknown amplitude case: %s" % name)
        if name not in cases_to_run:
            cases_to_run.append(name)
    save_json(output_dir / "amplitude_cases.json", {name: all_cases[name] for name in cases_to_run})

    incident_rows = []
    incident_by_input = {}
    for input_type in input_types:
        field = make_input_field(input_type, layout, device)
        after_input = propagators["input_to_prompt"](field)
        intensity = torch.abs(after_input).square()
        incident_ratios, _outside, _total = energy_ratios(intensity, masks["prompt_cells"])
        incident_by_input[input_type] = incident_ratios.detach().cpu()
        row = {"input_type": input_type}
        for index, cell_id in enumerate(CELL_IDS):
            row[cell_id + "_incident_energy_ratio"] = float(incident_ratios[index].item())
        incident_rows.append(row)
        if input_type == input_types[0]:
            plot_intensity(after_input, layout, output_dir / "figures" / "after_input_to_prompt_intensity.png", "after input-to-prompt propagation", args.plot_dpi, args.max_plot_dim, overlay_prompt=True, overlay_experts=False, save_linear=args.save_linear_intensity)
            plot_heatmap_3x3(incident_ratios.detach().cpu().numpy(), output_dir / "figures" / "prompt_cell_incident_energy_3x3.png", "Prompt cell incident energy", "energy ratio", args.plot_dpi)
    save_csv(output_dir / "prompt_cell_incident_energy_3x3.csv", incident_rows)

    rows = []
    for input_type in input_types:
        for case_name in cases_to_run:
            row = run_single_case(
                input_type=input_type,
                case_name=case_name,
                amplitudes=all_cases[case_name],
                layout=layout,
                physical=physical,
                propagators=propagators,
                masks=masks,
                args=args,
                output_dir=output_dir,
                device=device,
                incident_ratios=incident_by_input[input_type],
            )
            rows.append(row)
    save_csv(output_dir / "metrics.csv", rows)

    crosstalk = np.zeros((9, 9), dtype=np.float32)
    for row in rows:
        if row["input_type"] != "gaussian":
            continue
        case_name = row["amplitude_case"]
        if not case_name.startswith("onehot_"):
            continue
        active = EXPERT_IDS.index(case_name.replace("onehot_", ""))
        for expert_index, expert_id in enumerate(EXPERT_IDS):
            crosstalk[active, expert_index] = float(row[expert_id + "_energy_ratio"])
    save_matrix_csv(output_dir / "one_hot_crosstalk_expert_entrance_raw.csv", crosstalk)
    plot_matrix(crosstalk, output_dir / "figures" / "one_hot_crosstalk_matrix.png", "One-hot composite prompt crosstalk", args.plot_dpi)

    gaussian_main = [
        row
        for row in rows
        if row["input_type"] == "gaussian" and not row["amplitude_case"].startswith("onehot_")
    ]
    plot_bar(gaussian_main, "cosine_commanded_measured", output_dir / "figures" / "cosine_commanded_vs_measured_bar.png", "Cosine: commanded vs measured", "cosine", args.plot_dpi)
    plot_bar(gaussian_main, "cosine_corrected_measured", output_dir / "figures" / "cosine_corrected_vs_measured_bar.png", "Cosine: corrected command vs measured", "cosine", args.plot_dpi)
    plot_bar(gaussian_main, "outside_all_experts_energy_ratio", output_dir / "figures" / "outside_energy_ratio_bar.png", "Outside all experts energy ratio", "outside ratio", args.plot_dpi)

    summary = summarize(rows, crosstalk, physical)
    summary["layout"] = layout.to_dict()
    summary["physical_params"] = physical
    summary["amplitude_cases"] = {name: all_cases[name] for name in cases_to_run}
    save_json(output_dir / "summary.json", summary)
    print("overall status: %s" % summary["overall_status"])
    print("metrics: %s" % (output_dir / "metrics.csv"))
    print("figures: %s" % (output_dir / "figures"))


if __name__ == "__main__":
    main()
