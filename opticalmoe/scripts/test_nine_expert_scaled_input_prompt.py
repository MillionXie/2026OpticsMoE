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
DEFAULT_CASES = [
    "uniform",
    "center_only",
    "corner_only_E00",
    "corner_only_E22",
    "top_row",
    "left_col",
    "diagonal",
    "sparse_mix",
    "random_seeded",
    "task_like_mnist",
    "task_like_fashion",
    "task_like_emnist",
] + ["onehot_" + expert_id for expert_id in EXPERT_IDS]


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

    @property
    def area(self) -> int:
        return self.height * self.width

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
            "area": self.area,
        }


@dataclass
class ScaledInputLayout:
    canvas_height: int
    canvas_width: int
    input_size_mode: str
    input_size: int
    expert_size: int
    prompt_cell_size: int
    center_coords: List[int]
    input_aperture: Aperture
    prompt_cells: List[Aperture]
    expert_apertures: List[Aperture]
    prompt_union_bbox: Aperture

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return (self.canvas_height, self.canvas_width)

    @property
    def canvas_center(self) -> Tuple[float, float]:
        return (self.canvas_height / 2.0, self.canvas_width / 2.0)

    def input_prompt_union_overlap_ratio(self) -> float:
        overlap = intersection_area(self.input_aperture, self.prompt_union_bbox)
        return overlap / float(max(self.prompt_union_bbox.area, 1))

    def to_dict(self) -> Dict:
        baseline_area = 4 * 200 * 200
        nine_area = 9 * self.expert_size * self.expert_size
        return {
            "canvas_shape": list(self.canvas_shape),
            "canvas_center": list(self.canvas_center),
            "input_size_mode": self.input_size_mode,
            "input_size": self.input_size,
            "input_aperture": self.input_aperture.to_dict(),
            "expert_size": self.expert_size,
            "prompt_cell_size": self.prompt_cell_size,
            "expert_center_coords": list(self.center_coords),
            "prompt_union_bbox": self.prompt_union_bbox.to_dict(),
            "input_to_prompt_overlap_with_prompt_union": self.input_prompt_union_overlap_ratio(),
            "prompt_cells": [item.to_dict() for item in self.prompt_cells],
            "expert_apertures": [item.to_dict() for item in self.expert_apertures],
            "four_expert_baseline_area": baseline_area,
            "nine_expert_matched_area": nine_area,
            "relative_area_difference": (nine_area - baseline_area) / float(baseline_area),
        }


class NineCellMicrolensPrompt:
    """Partitioned 9-cell local lens + local grating prompt.

    The cells are spatially disjoint. This is not a global fan-out hologram and
    not a full-aperture phase superposition.
    """

    def __init__(
        self,
        layout: ScaledInputLayout,
        wavelength_m: float,
        pixel_size_m: float,
        focal_length_m: float,
        input_to_prompt_m: float,
        prompt_to_expert_m: float,
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
        self.prompt_to_expert_m = float(prompt_to_expert_m)
        self.amplitudes = torch.tensor(amplitudes, dtype=torch.float32, device=device)
        self.phase_biases = torch.tensor(phase_biases, dtype=torch.float32, device=device)
        self.device = device

        y_grid, x_grid = physical_grids(layout, pixel_size_m, device)
        masks = prompt_cell_masks(layout, device)
        lens_phases = []
        grating_phases = []
        reports = []
        for index, cell in enumerate(layout.prompt_cells):
            expert = layout.expert_apertures[index]
            cell_y, cell_x = cell.center
            canvas_y, canvas_x = layout.canvas_center
            offset_y_m = (cell_y - canvas_y) * pixel_size_m
            offset_x_m = (cell_x - canvas_x) * pixel_size_m
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
            reports.append(
                {
                    "cell_id": CELL_IDS[index],
                    "expert_id": EXPERT_IDS[index],
                    "cell_center_y_px": cell_y,
                    "cell_center_x_px": cell_x,
                    "expert_center_y_px": expert.center[0],
                    "expert_center_x_px": expert.center[1],
                    "offset_y_px": cell_y - canvas_y,
                    "offset_x_px": cell_x - canvas_x,
                    "theta_y_deg": math.degrees(theta_y),
                    "theta_x_deg": math.degrees(theta_x),
                    "grating_period_y_px": period_px(fy, pixel_size_m),
                    "grating_period_x_px": period_px(fx, pixel_size_m),
                    "focal_length_cm": self.focal_length_m * 100.0,
                    "input_to_prompt_cm": self.input_to_prompt_m * 100.0,
                    "prompt_to_expert_cm": self.prompt_to_expert_m * 100.0,
                    "prompt_cell_size": layout.prompt_cell_size,
                    "expert_size": layout.expert_size,
                }
            )
        self.masks = masks
        self.lens_phases = torch.stack(lens_phases, dim=0)
        self.grating_phases = torch.stack(grating_phases, dim=0)
        self.cell_reports = reports

    def commanded_power(self) -> torch.Tensor:
        return self.amplitudes.square()

    def normalized_commanded_power(self) -> torch.Tensor:
        power = self.commanded_power()
        return power / (power.sum() + EPS)

    def amplitude_map(self) -> torch.Tensor:
        return torch.sum(self.masks * self.amplitudes.view(9, 1, 1), dim=0)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scaled-input 9-cell composite microlens prompt test."
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--canvas_height", type=int, default=700)
    parser.add_argument("--canvas_width", type=int, default=700)
    parser.add_argument("--input_size_mode", default="prompt_matched", choices=["center_200", "prompt_matched", "custom"])
    parser.add_argument("--input_size", type=int, default=200)
    parser.add_argument("--expert_size", type=int, default=134)
    parser.add_argument("--expert_center_coords", default="167,350,533")
    parser.add_argument("--prompt_cell_size", type=int, default=180)

    parser.add_argument(
        "--geometry_preset",
        default="prompt_to_expert_matched_134",
        choices=["strict_training_geometry", "prompt_to_expert_matched_134"],
    )
    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--input_to_prompt_cm", type=float, default=None)
    parser.add_argument("--prompt_to_expert_cm", type=float, default=None)
    parser.add_argument("--focal_length_cm", type=float, default=None)
    parser.add_argument("--inter_layer_cm", type=float, default=5.0)
    parser.add_argument("--num_dummy_layers", type=int, default=5)

    parser.add_argument("--input_types", default="flat_top,gaussian_wide,f_pattern,digit_like")
    parser.add_argument("--amplitude_cases", default=",".join(DEFAULT_CASES))
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


def period_px(spatial_frequency_per_m: float, pixel_size_m: float) -> float:
    if abs(spatial_frequency_per_m) < 1e-20:
        return float("inf")
    return 1.0 / (abs(spatial_frequency_per_m) * pixel_size_m)


def intersection_area(a: Aperture, b: Aperture) -> int:
    y0 = max(a.y0, b.y0)
    y1 = min(a.y1, b.y1)
    x0 = max(a.x0, b.x0)
    x1 = min(a.x1, b.x1)
    return max(0, y1 - y0) * max(0, x1 - x0)


def union_bbox(apertures: Sequence[Aperture], name: str) -> Aperture:
    return Aperture(
        name,
        min(item.y0 for item in apertures),
        max(item.y1 for item in apertures),
        min(item.x0 for item in apertures),
        max(item.x1 for item in apertures),
    )


def centered_aperture(name: str, center_y: int, center_x: int, size: int) -> Aperture:
    half = int(size) // 2
    return Aperture(name, center_y - half, center_y + half, center_x - half, center_x + half)


def build_layout(args) -> ScaledInputLayout:
    centers = parse_int_list(args.expert_center_coords)
    prompt_cells = []
    expert_apertures = []
    for row, center_y in enumerate(centers):
        for col, center_x in enumerate(centers):
            suffix = "%d%d" % (row, col)
            prompt_cells.append(centered_aperture("C" + suffix, center_y, center_x, args.prompt_cell_size))
            expert_apertures.append(centered_aperture("E" + suffix, center_y, center_x, args.expert_size))
    prompt_bbox = union_bbox(prompt_cells, "prompt_union_bbox")
    canvas_center_y = args.canvas_height // 2
    canvas_center_x = args.canvas_width // 2
    if args.input_size_mode == "center_200":
        input_size = 500 #200
        input_aperture = centered_aperture("input", canvas_center_y, canvas_center_x, input_size)
    elif args.input_size_mode == "prompt_matched":
        if prompt_bbox.height != prompt_bbox.width:
            raise ValueError("prompt_matched expects square prompt union bbox.")
        input_size = prompt_bbox.height
        input_aperture = Aperture("input", prompt_bbox.y0, prompt_bbox.y1, prompt_bbox.x0, prompt_bbox.x1)
    elif args.input_size_mode == "custom":
        input_size = int(args.input_size)
        input_aperture = centered_aperture("input", canvas_center_y, canvas_center_x, input_size)
    else:
        raise ValueError("Unsupported input_size_mode.")
    layout = ScaledInputLayout(
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        input_size_mode=args.input_size_mode,
        input_size=input_size,
        expert_size=args.expert_size,
        prompt_cell_size=args.prompt_cell_size,
        center_coords=centers,
        input_aperture=input_aperture,
        prompt_cells=prompt_cells,
        expert_apertures=expert_apertures,
        prompt_union_bbox=prompt_bbox,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: ScaledInputLayout) -> None:
    for aperture in [layout.input_aperture] + layout.prompt_cells + layout.expert_apertures:
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


def resolve_geometry(args, layout: ScaledInputLayout) -> Dict:
    if args.geometry_preset == "strict_training_geometry":
        input_to_prompt_cm = 20.0
        prompt_to_expert_cm = 20.0
        focal_length_cm = 10.0
    elif args.geometry_preset == "prompt_to_expert_matched_134":
        input_to_prompt_cm = 20.0
        magnification = float(layout.expert_size) / float(layout.prompt_cell_size)
        prompt_to_expert_cm = input_to_prompt_cm * magnification
        focal_length_cm = input_to_prompt_cm * prompt_to_expert_cm / (input_to_prompt_cm + prompt_to_expert_cm)
    else:
        raise ValueError("Unsupported geometry preset.")
    if args.input_to_prompt_cm is not None:
        input_to_prompt_cm = float(args.input_to_prompt_cm)
    if args.prompt_to_expert_cm is not None:
        prompt_to_expert_cm = float(args.prompt_to_expert_cm)
    if args.focal_length_cm is not None:
        focal_length_cm = float(args.focal_length_cm)
    magnification = prompt_to_expert_cm / input_to_prompt_cm
    expected_cell_image_size_px = layout.prompt_cell_size * magnification
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
        "expected_cell_image_size_px": expected_cell_image_size_px,
        "expected_cell_image_to_expert_ratio": expected_cell_image_size_px / float(layout.expert_size),
        "prompt_cell_size": layout.prompt_cell_size,
        "expert_size": layout.expert_size,
        "num_dummy_layers": args.num_dummy_layers,
    }


def aperture_masks(layout: ScaledInputLayout, apertures: Sequence[Aperture], device: torch.device) -> torch.Tensor:
    masks = []
    for aperture in apertures:
        mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)


def prompt_cell_masks(layout: ScaledInputLayout, device: torch.device) -> torch.Tensor:
    return aperture_masks(layout, layout.prompt_cells, device)


def expert_masks(layout: ScaledInputLayout, device: torch.device) -> torch.Tensor:
    return aperture_masks(layout, layout.expert_apertures, device)


def pixel_grids(layout: ScaledInputLayout, device: torch.device):
    y = torch.arange(layout.canvas_height, dtype=torch.float32, device=device)
    x = torch.arange(layout.canvas_width, dtype=torch.float32, device=device)
    return torch.meshgrid(y, x, indexing="ij")


def physical_grids(layout: ScaledInputLayout, pixel_size_m: float, device: torch.device):
    cy, cx = layout.canvas_center
    y = (torch.arange(layout.canvas_height, dtype=torch.float32, device=device) - cy) * pixel_size_m
    x = (torch.arange(layout.canvas_width, dtype=torch.float32, device=device) - cx) * pixel_size_m
    return torch.meshgrid(y, x, indexing="ij")


def make_flat_top(layout: ScaledInputLayout, device: torch.device) -> torch.Tensor:
    amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    ap = layout.input_aperture
    amplitude[ap.y0 : ap.y1, ap.x0 : ap.x1] = 1.0
    return amplitude


def make_gaussian_wide(layout: ScaledInputLayout, device: torch.device) -> torch.Tensor:
    y_grid, x_grid = pixel_grids(layout, device)
    cy, cx = layout.input_aperture.center
    sigma = layout.input_size / 4.0
    amplitude = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    mask = make_flat_top(layout, device)
    return amplitude * mask


def make_f_pattern(size: int, device: torch.device) -> torch.Tensor:
    pattern = torch.zeros((size, size), dtype=torch.float32, device=device)
    t = max(10, size // 10)
    margin = max(10, size // 10)
    pattern[margin : size - margin, margin : margin + t] = 1.0
    pattern[margin : margin + t, margin : size - margin] = 1.0
    mid = size // 2
    pattern[mid - t // 2 : mid + t // 2, margin : int(size * 0.68)] = 1.0
    pattern[int(size * 0.72) : int(size * 0.82), margin : int(size * 0.45)] = 0.6
    return pattern


def make_digit_like(size: int, device: torch.device) -> torch.Tensor:
    pattern = torch.zeros((size, size), dtype=torch.float32, device=device)
    t = max(9, size // 12)
    m = max(10, size // 8)
    pattern[m : m + t, m : size - m] = 1.0
    pattern[m : size // 2, m : m + t] = 1.0
    pattern[size // 2 - t // 2 : size // 2 + t // 2, m : size - m] = 1.0
    pattern[size // 2 : size - m, size - m - t : size - m] = 1.0
    pattern[size - m - t : size - m, m : size - m] = 1.0
    pattern[int(size * 0.67) : int(size * 0.75), int(size * 0.35) : int(size * 0.50)] = 0.55
    return pattern


def make_input_field(input_type: str, layout: ScaledInputLayout, device: torch.device) -> torch.Tensor:
    if input_type == "flat_top":
        amplitude = make_flat_top(layout, device)
    elif input_type == "gaussian_wide":
        amplitude = make_gaussian_wide(layout, device)
    elif input_type in {"f_pattern", "digit_like"}:
        amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        local = make_f_pattern(layout.input_size, device) if input_type == "f_pattern" else make_digit_like(layout.input_size, device)
        ap = layout.input_aperture
        amplitude[ap.y0 : ap.y1, ap.x0 : ap.x1] = local
    elif input_type == "gaussian":
        amplitude = make_gaussian_wide(layout, device)
    else:
        raise ValueError("Unsupported input_type: %s" % input_type)
    return amplitude.unsqueeze(0).to(torch.complex64)


def build_propagator(wavelength_m: float, pixel_size_m: float, grid_size, distance_m: float, device: torch.device):
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


def energy_ratios(intensity: torch.Tensor, masks: torch.Tensor):
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    total = float(intensity.sum().item())
    energies = torch.einsum("bhw,khw->bk", intensity, masks)[0]
    ratios = energies / (total + EPS)
    outside = max(0.0, total - float(energies.sum().item())) / (total + EPS)
    return ratios, outside, total


def incident_stats(ratios: torch.Tensor) -> Dict:
    array = ratios.detach().cpu().numpy().astype(np.float64)
    mean = float(array.mean())
    std = float(array.std())
    return {
        "prompt_incident_min": float(array.min()),
        "prompt_incident_max": float(array.max()),
        "prompt_incident_std": std,
        "prompt_incident_cv": std / (mean + EPS),
    }


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


def plot_layout(layout: ScaledInputLayout, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("black")
    input_ap = layout.input_aperture
    ax.add_patch(Rectangle((input_ap.x0, input_ap.y0), input_ap.width, input_ap.height, fill=False, edgecolor="orange", linewidth=2.0, linestyle="-"))
    for cell in layout.prompt_cells:
        ax.add_patch(Rectangle((cell.x0, cell.y0), cell.width, cell.height, fill=False, edgecolor="violet", linewidth=1.1, linestyle="--"))
        cy, cx = cell.center
        ax.text(cx, cy - 18, cell.name, color="violet", ha="center", va="center", fontsize=8)
    for expert in layout.expert_apertures:
        ax.add_patch(Rectangle((expert.x0, expert.y0), expert.width, expert.height, fill=False, edgecolor="cyan", linewidth=1.8))
        cy, cx = expert.center
        ax.text(cx, cy + 16, expert.name, color="white", ha="center", va="center", fontsize=8)
    cy, cx = layout.canvas_center
    ax.axhline(cy, color="gray", linewidth=0.8)
    ax.axvline(cx, color="gray", linewidth=0.8)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_aspect("equal")
    ax.set_title("Scaled input / 9-cell prompt / 9-expert layout")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_array(array, layout, path, title, dpi, max_plot_dim, label, overlay_prompt=False, overlay_experts=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    stride = max(1, int(math.ceil(max(array.shape) / max(1, int(max_plot_dim)))))
    display = array[::stride, ::stride] if stride > 1 else array
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(display, cmap="inferno", extent=(0, layout.canvas_width, layout.canvas_height, 0))
    if overlay_prompt:
        for cell in layout.prompt_cells:
            ax.add_patch(Rectangle((cell.x0, cell.y0), cell.width, cell.height, fill=False, edgecolor="violet", linewidth=1.0, linestyle="--"))
    if overlay_experts:
        for expert in layout.expert_apertures:
            ax.add_patch(Rectangle((expert.x0, expert.y0), expert.width, expert.height, fill=False, edgecolor="cyan", linewidth=1.0))
            cy, cx = expert.center
            ax.text(cx, cy, expert.name, color="white", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_intensity(field_or_intensity, layout, path, title, dpi, max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=False):
    if torch.is_complex(field_or_intensity):
        intensity = torch.abs(field_or_intensity.to(torch.complex64)).square()
    else:
        intensity = field_or_intensity.float()
    if intensity.ndim == 3:
        intensity = intensity[0]
    array = intensity.detach().cpu().float().numpy()
    log_array = np.log10(array / (array.max() + EPS) + 1e-8)
    plot_array(log_array, layout, path, title, dpi, max_plot_dim, "log10(I/Imax+1e-8)", overlay_prompt, overlay_experts)
    if save_linear:
        plot_array(array, layout, path.with_name(path.stem + "_linear.png"), title, dpi, max_plot_dim, "linear intensity", overlay_prompt, overlay_experts)


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


def plot_heatmap_3x3(values, path: Path, title: str, label: str, dpi: int, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(array, cmap=cmap)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])
    for row in range(3):
        for col in range(3):
            ax.text(col, row, "%.3f" % array[row, col], ha="center", va="center", color="white")
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
    layout: ScaledInputLayout,
    physical: Dict,
    propagators: Dict,
    masks: Dict,
    args,
    output_dir: Path,
    device: torch.device,
    incident_ratios: torch.Tensor,
    prompt_reports: Optional[List[Dict]] = None,
) -> Tuple[Dict, List[Dict]]:
    prompt = NineCellMicrolensPrompt(
        layout=layout,
        wavelength_m=physical["wavelength_m"],
        pixel_size_m=physical["pixel_size_m"],
        focal_length_m=physical["focal_length_m"],
        input_to_prompt_m=physical["input_to_prompt_m"],
        prompt_to_expert_m=physical["prompt_to_expert_m"],
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

    commanded = prompt.normalized_commanded_power().detach().cpu().numpy()
    corrected = (prompt.commanded_power() * incident_ratios.to(device)).detach()
    corrected = (corrected / (corrected.sum() + EPS)).cpu().numpy()
    measured = expert_ratios.detach().cpu().numpy()
    vec = vector_metrics(commanded, corrected, measured)
    top_index = int(np.argmax(measured))
    second = float(np.sort(measured)[-2]) if len(measured) > 1 else 0.0
    incident = incident_ratios.detach().cpu().numpy()
    stats = incident_stats(incident_ratios)
    row = {
        "input_type": input_type,
        "input_size_mode": layout.input_size_mode,
        "input_size": layout.input_size,
        "amplitude_case": case_name,
        "geometry_preset": physical["geometry_preset"],
        "prompt_cell_size": layout.prompt_cell_size,
        "expert_size": layout.expert_size,
        "magnification": physical["magnification"],
        "expected_cell_image_size_px": physical["expected_cell_image_size_px"],
        "expected_cell_image_to_expert_ratio": physical["expected_cell_image_to_expert_ratio"],
        "input_prompt_union_overlap_ratio": layout.input_prompt_union_overlap_ratio(),
        "total_energy_expert_entrance_raw": total_raw,
        "outside_all_experts_energy_ratio": outside_raw,
        "top_expert_id": EXPERT_IDS[top_index],
        "second_largest_expert_energy_ratio": second,
        "expert_energy_entropy": entropy(measured),
    }
    row.update(stats)
    row.update(vec)
    for index, cell_id in enumerate(CELL_IDS):
        row[cell_id + "_incident_energy_ratio"] = float(incident[index])
    for index, expert_id in enumerate(EXPERT_IDS):
        row[expert_id + "_amplitude"] = float(amplitudes[index])
        row[expert_id + "_normalized_commanded_power"] = float(commanded[index])
        row[expert_id + "_normalized_corrected_command"] = float(corrected[index])
        row[expert_id + "_energy_ratio"] = float(measured[index])

    should_plot = input_type == "flat_top" or not case_name.startswith("onehot_")
    if should_plot:
        case_dir = output_dir / "figures" / input_type / case_name
        plot_heatmap_3x3(amplitudes, case_dir / "prompt_amplitude_3x3.png", case_name + " amplitudes", "amplitude", args.plot_dpi)
        plot_intensity(prompt.amplitude_map(), layout, case_dir / "composite_prompt_amplitude_map.png", case_name + " prompt amplitude map", args.plot_dpi, args.max_plot_dim, overlay_prompt=True, overlay_experts=False)
        plot_phase(prompt.phase_map(), case_dir / "composite_prompt_phase_wrapped.png", case_name + " composite prompt phase", args.plot_dpi)
        plot_intensity(expert_raw, layout, case_dir / "expert_entrance_raw_intensity.png", case_name + " expert entrance raw", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_intensity(after_aperture, layout, case_dir / "expert_entrance_after_aperture_intensity.png", case_name + " after aperture", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_heatmap_3x3(measured, case_dir / "measured_expert_energy_3x3.png", case_name + " measured expert energy", "energy ratio", args.plot_dpi)
        plot_intensity(detector, layout, case_dir / "detector_plane_intensity.png", case_name + " detector plane", args.plot_dpi, args.max_plot_dim, overlay_prompt=False, overlay_experts=True, save_linear=args.save_linear_intensity)
    reports = prompt.cell_reports if prompt_reports is None else prompt_reports
    return row, reports


def pass_warn_fail(value: float, pass_value: float, warn_value: float, lower_is_better: bool) -> str:
    if lower_is_better:
        if value < pass_value:
            return "PASS"
        if value < warn_value:
            return "WARN"
        return "FAIL"
    if value >= pass_value:
        return "PASS"
    if value >= warn_value:
        return "WARN"
    return "FAIL"


def summarize(rows: List[Dict], crosstalk: np.ndarray, layout: ScaledInputLayout) -> Dict:
    target_rows = [
        row
        for row in rows
        if row["input_type"] == "flat_top" and row["amplitude_case"] == "uniform"
    ]
    target = target_rows[0] if target_rows else None
    if target is None:
        incident_status = "FAIL"
        active_status = "FAIL"
        outside_status = "FAIL"
        incident_cv = float("inf")
        active_count = 0
        outside = 1.0
    else:
        incident_cv = float(target["prompt_incident_cv"])
        incident_status = pass_warn_fail(incident_cv, 0.25, 0.50, True)
        active_count = sum(1 for expert_id in EXPERT_IDS if target[expert_id + "_energy_ratio"] > 0.02)
        active_status = "PASS" if active_count >= 7 else ("WARN" if active_count >= 5 else "FAIL")
        outside = float(target["outside_all_experts_energy_ratio"])
        outside_status = pass_warn_fail(outside, 0.35, 0.60, True)
    diagonal_top = 0
    for index in range(9):
        if int(np.argmax(crosstalk[index])) == index:
            diagonal_top += 1
    crosstalk_status = "PASS" if diagonal_top >= 7 else ("WARN" if diagonal_top >= 5 else "FAIL")
    statuses = [incident_status, active_status, outside_status, crosstalk_status]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    if layout.input_size_mode == "center_200" and target is not None and target["C11_incident_energy_ratio"] > 0.5:
        comparison_note = (
            "center_200 reproduces the expected issue: illumination is dominated "
            "by the center prompt cell."
        )
    elif layout.input_size_mode == "prompt_matched" and incident_status in {"PASS", "WARN"} and active_status == "FAIL":
        comparison_note = (
            "prompt_matched improves prompt-cell illumination, but experts are "
            "still not all active; inspect prompt phase, magnification, and aperture clipping."
        )
    elif layout.input_size_mode == "prompt_matched" and incident_status == "PASS":
        comparison_note = (
            "prompt_matched makes prompt-cell incident energy substantially more uniform."
        )
    else:
        comparison_note = (
            "Run both center_200 and prompt_matched with the same geometry to compare input coverage."
        )
    return {
        "overall_status": overall,
        "flat_top_uniform_prompt_incident_cv": incident_cv,
        "prompt_incident_uniformity_status": incident_status,
        "flat_top_uniform_active_expert_count_gt_0p02": active_count,
        "active_experts_status": active_status,
        "flat_top_uniform_outside_energy_ratio": outside,
        "outside_energy_status": outside_status,
        "one_hot_diagonal_top_count": diagonal_top,
        "one_hot_crosstalk_status": crosstalk_status,
        "comparison_note": comparison_note,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    layout = build_layout(args)
    physical = resolve_geometry(args, layout)
    physical["wavelength_nm"] = args.wavelength_nm
    physical["pixel_size_um"] = args.pixel_size_um
    physical["wavelength_m"] = nm_to_m(args.wavelength_nm)
    physical["pixel_size_m"] = um_to_m(args.pixel_size_um)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "nine_expert_scaled_input_prompt" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_area = 4 * 200 * 200
    nine_area = 9 * args.expert_size * args.expert_size
    print("9-expert scaled-input composite prompt test")
    print("device: %s" % device)
    print("output_dir: %s" % output_dir)
    print("4-expert baseline area = 4 x 200 x 200 = %d" % baseline_area)
    print("9-expert matched area  = 9 x %d x %d = %d" % (args.expert_size, args.expert_size, nine_area))
    print("relative difference = %+0.2f%%" % ((nine_area - baseline_area) / float(baseline_area) * 100.0))
    print("input_size_mode: %s" % layout.input_size_mode)
    print("input_size: %d" % layout.input_size)
    print("prompt_union_bbox: %s" % layout.prompt_union_bbox.to_dict())
    print("input/prompt-union overlap ratio: %.4f" % layout.input_prompt_union_overlap_ratio())
    print("geometry preset: %s" % physical["geometry_preset"])
    print("input_to_prompt_cm: %.3f" % physical["input_to_prompt_cm"])
    print("prompt_to_expert_cm: %.3f" % physical["prompt_to_expert_cm"])
    print("focal_length_cm: %.3f" % physical["focal_length_cm"])
    print("magnification: %.4f" % physical["magnification"])
    print("expected_cell_image_size_px: %.2f" % physical["expected_cell_image_size_px"])
    print("expected_cell_image_size_px / expert_size: %.3f" % physical["expected_cell_image_to_expert_ratio"])

    save_json(output_dir / "layout.json", layout.to_dict())
    save_json(output_dir / "physical_params.json", physical)
    plot_layout(layout, output_dir / "figures" / "layout_overlay.png", args.plot_dpi)

    propagators = {
        "input_to_prompt": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["input_to_prompt_m"], device),
        "prompt_to_expert": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["prompt_to_expert_m"], device),
        "inter_layer": build_propagator(physical["wavelength_m"], physical["pixel_size_m"], layout.canvas_shape, physical["inter_layer_m"], device),
    }
    masks = {
        "prompt_cells": prompt_cell_masks(layout, device),
        "experts": expert_masks(layout, device),
    }
    masks["expert_union"] = torch.clamp(masks["experts"].sum(dim=0), 0.0, 1.0)

    input_types = [item.strip() for item in args.input_types.split(",") if item.strip()]
    requested_cases = [item.strip() for item in args.amplitude_cases.split(",") if item.strip()]
    all_cases = amplitude_case_dict(args.seed, args.custom_amplitudes)
    cases_to_run = []
    for name in requested_cases + ["onehot_" + expert_id for expert_id in EXPERT_IDS]:
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
        row.update(incident_stats(incident_ratios))
        for index, cell_id in enumerate(CELL_IDS):
            row[cell_id + "_incident_energy_ratio"] = float(incident_ratios[index].item())
        incident_rows.append(row)
        if input_type == "flat_top":
            plot_intensity(field, layout, output_dir / "figures" / "input_plane_intensity.png", "input plane: flat_top", args.plot_dpi, args.max_plot_dim, overlay_prompt=True, overlay_experts=False, save_linear=args.save_linear_intensity)
            plot_intensity(after_input, layout, output_dir / "figures" / "after_input_to_prompt_intensity.png", "after input-to-prompt: flat_top", args.plot_dpi, args.max_plot_dim, overlay_prompt=True, overlay_experts=False, save_linear=args.save_linear_intensity)
            plot_heatmap_3x3(incident_ratios.detach().cpu().numpy(), output_dir / "figures" / "prompt_cell_incident_energy_3x3.png", "Prompt cell incident energy: flat_top", "energy ratio", args.plot_dpi)
    save_csv(output_dir / "prompt_cell_incident_energy_3x3.csv", incident_rows)

    rows = []
    prompt_reports = None
    for input_type in input_types:
        for case_name in cases_to_run:
            row, reports = run_single_case(
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
                prompt_reports=prompt_reports,
            )
            rows.append(row)
            if prompt_reports is None:
                prompt_reports = reports
    save_json(output_dir / "prompt_cell_reports.json", prompt_reports or [])
    save_csv(output_dir / "metrics.csv", rows)

    crosstalk = np.zeros((9, 9), dtype=np.float32)
    for row in rows:
        if row["input_type"] != "flat_top":
            continue
        case_name = row["amplitude_case"]
        if not case_name.startswith("onehot_"):
            continue
        active = EXPERT_IDS.index(case_name.replace("onehot_", ""))
        for expert_index, expert_id in enumerate(EXPERT_IDS):
            crosstalk[active, expert_index] = float(row[expert_id + "_energy_ratio"])
    save_matrix_csv(output_dir / "one_hot_crosstalk_expert_entrance_raw.csv", crosstalk)
    plot_matrix(crosstalk, output_dir / "figures" / "one_hot_crosstalk_matrix.png", "One-hot crosstalk: flat_top", args.plot_dpi)

    flat_main = [
        row
        for row in rows
        if row["input_type"] == "flat_top" and not row["amplitude_case"].startswith("onehot_")
    ]
    plot_bar(flat_main, "prompt_incident_cv", output_dir / "figures" / "incident_cv_bar.png", "Prompt incident CV", "CV", args.plot_dpi)
    plot_bar(flat_main, "outside_all_experts_energy_ratio", output_dir / "figures" / "outside_energy_ratio_bar.png", "Outside all experts energy ratio", "outside ratio", args.plot_dpi)
    plot_bar(flat_main, "cosine_corrected_measured", output_dir / "figures" / "cosine_corrected_vs_measured_bar.png", "Cosine: corrected command vs measured", "cosine", args.plot_dpi)

    summary = summarize(rows, crosstalk, layout)
    summary["layout"] = layout.to_dict()
    summary["physical_params"] = physical
    summary["amplitude_cases"] = {name: all_cases[name] for name in cases_to_run}
    save_json(output_dir / "summary.json", summary)
    print("overall status: %s" % summary["overall_status"])
    print("comparison_note: %s" % summary["comparison_note"])
    print("metrics: %s" % (output_dir / "metrics.csv"))
    print("figures: %s" % (output_dir / "figures"))


if __name__ == "__main__":
    main()
