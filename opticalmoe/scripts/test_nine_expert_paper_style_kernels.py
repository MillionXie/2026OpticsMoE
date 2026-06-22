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
KERNEL_IDS = ["K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"]
DEFAULT_AMPLITUDE_CASES = [
    "uniform",
    "center_only",
    "onehot_E00",
    "onehot_E11",
    "onehot_E22",
    "diagonal",
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
class PaperKernelLayout:
    canvas_height: int
    canvas_width: int
    input_size: int
    expert_size: int
    kernel_region_size: int
    center_coords: List[int]
    input_aperture: Aperture
    expert_apertures: List[Aperture]
    kernel_regions: List[Aperture]

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
            "expert_center_coords": list(self.center_coords),
            "expert_apertures": [aperture.to_dict() for aperture in self.expert_apertures],
            "kernel_region_size": self.kernel_region_size,
            "kernel_regions": [aperture.to_dict() for aperture in self.kernel_regions],
            "four_expert_baseline_area": baseline_area,
            "nine_expert_matched_area": nine_area,
            "relative_area_difference": (nine_area - baseline_area) / float(baseline_area),
        }


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, got %s" % value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style 9-kernel optical prompt diagnostic."
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--canvas_height", type=int, default=700)
    parser.add_argument("--canvas_width", type=int, default=700)
    parser.add_argument("--input_size", type=int, default=200)
    parser.add_argument("--expert_size", type=int, default=134)
    parser.add_argument("--kernel_region_size", type=int, default=180)
    parser.add_argument("--center_coords", default="167,350,533")

    parser.add_argument("--propagation_mode", default="fft_4f", choices=["fft_4f", "angular_spectrum_lens"])
    parser.add_argument("--kernel_phase_mode", default="same_identity", choices=[
        "same_identity",
        "same_random",
        "independent_random",
        "vortex",
        "random_plus_vortex",
    ])
    parser.add_argument("--use_global_lens_phase", type=str2bool, default=True)
    parser.add_argument("--calibrate_grating_signs", type=str2bool, default=True)
    parser.add_argument("--grating_sign_x", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--grating_sign_y", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--grating_shift_scale", type=float, default=1.0)

    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--focal_length_cm", type=float, default=10.0)
    parser.add_argument("--input_to_mask_cm", type=float, default=20.0)
    parser.add_argument("--mask_to_expert_cm", type=float, default=20.0)
    parser.add_argument("--inter_layer_cm", type=float, default=5.0)
    parser.add_argument("--num_dummy_layers", type=int, default=5)

    parser.add_argument("--input_types", default="flat_top_200,gaussian_200,f_pattern_200,digit_like_200")
    parser.add_argument("--amplitude_cases", default=",".join(DEFAULT_AMPLITUDE_CASES))
    parser.add_argument("--custom_amplitudes", default=None)

    parser.add_argument("--hard_aperture", type=str2bool, default=True)
    parser.add_argument("--recenter_similarity", type=str2bool, default=False)
    parser.add_argument("--save_linear_intensity", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=130)
    parser.add_argument("--max_plot_dim", type=int, default=1400)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def parse_int_list(text: str) -> List[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    return [int(item) for item in values]


def parse_float_list(text: str, expected: int) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError("Expected %d values, got %d." % (expected, len(values)))
    return values


def centered_aperture(name: str, center_y: int, center_x: int, size: int) -> Aperture:
    half = int(size) // 2
    return Aperture(name, center_y - half, center_y + half, center_x - half, center_x + half)


def build_layout(args) -> PaperKernelLayout:
    centers = parse_int_list(args.center_coords)
    if len(centers) != 3:
        raise ValueError("--center_coords must contain exactly three integers.")
    canvas_center_y = args.canvas_height // 2
    canvas_center_x = args.canvas_width // 2
    input_aperture = centered_aperture("input", canvas_center_y, canvas_center_x, args.input_size)
    expert_apertures = []
    kernel_regions = []
    for row, center_y in enumerate(centers):
        for col, center_x in enumerate(centers):
            suffix = "%d%d" % (row, col)
            expert_apertures.append(centered_aperture("E" + suffix, center_y, center_x, args.expert_size))
            kernel_regions.append(centered_aperture("K" + suffix, center_y, center_x, args.kernel_region_size))
    layout = PaperKernelLayout(
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        input_size=args.input_size,
        expert_size=args.expert_size,
        kernel_region_size=args.kernel_region_size,
        center_coords=centers,
        input_aperture=input_aperture,
        expert_apertures=expert_apertures,
        kernel_regions=kernel_regions,
    )
    validate_layout(layout)
    return layout


def validate_layout(layout: PaperKernelLayout) -> None:
    for aperture in [layout.input_aperture] + layout.expert_apertures + layout.kernel_regions:
        if aperture.height <= 0 or aperture.width <= 0:
            raise ValueError("%s has invalid size." % aperture.name)
        if aperture.y0 < 0 or aperture.x0 < 0:
            raise ValueError("%s starts outside canvas." % aperture.name)
        if aperture.y1 > layout.canvas_height or aperture.x1 > layout.canvas_width:
            raise ValueError("%s ends outside canvas." % aperture.name)
    kernel_overlap = aperture_masks(layout, layout.kernel_regions, torch.device("cpu")).sum(dim=0)
    if torch.any(kernel_overlap > 1.0):
        raise ValueError("Kernel regions overlap. Reduce --kernel_region_size.")


def aperture_masks(layout: PaperKernelLayout, apertures: Sequence[Aperture], device: torch.device) -> torch.Tensor:
    masks = []
    for aperture in apertures:
        mask = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        mask[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)


def pixel_grids(layout: PaperKernelLayout, device: torch.device):
    y = torch.arange(layout.canvas_height, dtype=torch.float32, device=device)
    x = torch.arange(layout.canvas_width, dtype=torch.float32, device=device)
    return torch.meshgrid(y, x, indexing="ij")


def centered_pixel_grids(layout: PaperKernelLayout, device: torch.device):
    y = torch.arange(layout.canvas_height, dtype=torch.float32, device=device) - layout.canvas_center[0]
    x = torch.arange(layout.canvas_width, dtype=torch.float32, device=device) - layout.canvas_center[1]
    return torch.meshgrid(y, x, indexing="ij")


def physical_grids(layout: PaperKernelLayout, pixel_size_m: float, device: torch.device):
    y_px, x_px = centered_pixel_grids(layout, device)
    return y_px * pixel_size_m, x_px * pixel_size_m


def make_flat_top(layout: PaperKernelLayout, device: torch.device) -> torch.Tensor:
    amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    aperture = layout.input_aperture
    amplitude[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = 1.0
    return amplitude


def make_gaussian(layout: PaperKernelLayout, device: torch.device) -> torch.Tensor:
    y_grid, x_grid = pixel_grids(layout, device)
    cy, cx = layout.input_aperture.center
    sigma = layout.input_size / 4.0
    amplitude = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    return amplitude * make_flat_top(layout, device)


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
    margin = max(10, size // 8)
    pattern[margin : margin + t, margin : size - margin] = 1.0
    pattern[margin : size // 2, margin : margin + t] = 1.0
    pattern[size // 2 - t // 2 : size // 2 + t // 2, margin : size - margin] = 1.0
    pattern[size // 2 : size - margin, size - margin - t : size - margin] = 1.0
    pattern[size - margin - t : size - margin, margin : size - margin] = 1.0
    pattern[int(size * 0.67) : int(size * 0.75), int(size * 0.35) : int(size * 0.50)] = 0.55
    return pattern


def make_input_field(input_type: str, layout: PaperKernelLayout, device: torch.device) -> torch.Tensor:
    if input_type == "flat_top_200":
        amplitude = make_flat_top(layout, device)
    elif input_type == "gaussian_200":
        amplitude = make_gaussian(layout, device)
    elif input_type in {"f_pattern_200", "digit_like_200"}:
        amplitude = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
        local = make_f_pattern(layout.input_size, device) if input_type == "f_pattern_200" else make_digit_like(layout.input_size, device)
        ap = layout.input_aperture
        amplitude[ap.y0 : ap.y1, ap.x0 : ap.x1] = local
    else:
        raise ValueError("Unsupported input_type: %s" % input_type)
    return amplitude.unsqueeze(0).to(torch.complex64)


def centered_fft2(field: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))


def centered_ifft2(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))), dim=(-2, -1))


def make_global_lens_phase(layout: PaperKernelLayout, wavelength_m: float, pixel_size_m: float, focal_length_m: float, device: torch.device) -> torch.Tensor:
    y_m, x_m = physical_grids(layout, pixel_size_m, device)
    return -math.pi / (wavelength_m * focal_length_m) * (x_m ** 2 + y_m ** 2)


def make_grating_phase_for_shift(dx_px: float, dy_px: float, layout: PaperKernelLayout, sign_x: int, sign_y: int, scale: float, device: torch.device) -> torch.Tensor:
    y_freq, x_freq = centered_pixel_grids(layout, device)
    # In centered FFT coordinates, multiplying the pupil spectrum by a linear
    # phase shifts the output feature map. The sign is calibrated because FFT
    # conventions differ across optical derivations.
    return (
        sign_x * 2.0 * math.pi * float(dx_px) * float(scale) * x_freq / float(layout.canvas_width)
        + sign_y * 2.0 * math.pi * float(dy_px) * float(scale) * y_freq / float(layout.canvas_height)
    )


def local_patch_grid(aperture: Aperture, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(aperture.height, dtype=torch.float32, device=device) - aperture.height / 2.0
    x = torch.arange(aperture.width, dtype=torch.float32, device=device) - aperture.width / 2.0
    return torch.meshgrid(y, x, indexing="ij")


def make_vortex_patch(aperture: Aperture, device: torch.device, charge: int = 1) -> torch.Tensor:
    y, x = local_patch_grid(aperture, device)
    return charge * torch.atan2(y, x + EPS)


def build_kernel_phase_map(layout: PaperKernelLayout, mode: str, device: torch.device) -> torch.Tensor:
    phase = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    if mode == "same_identity":
        return phase

    first = layout.kernel_regions[0]
    if mode == "same_random":
        base_patch = torch.rand((first.height, first.width), dtype=torch.float32, device=device) * (2.0 * math.pi)
    elif mode == "vortex":
        base_patch = make_vortex_patch(first, device)
    elif mode == "random_plus_vortex":
        base_patch = torch.rand((first.height, first.width), dtype=torch.float32, device=device) * (2.0 * math.pi)
        base_patch = base_patch + make_vortex_patch(first, device)
    else:
        base_patch = None

    for aperture in layout.kernel_regions:
        if mode == "independent_random":
            patch = torch.rand((aperture.height, aperture.width), dtype=torch.float32, device=device) * (2.0 * math.pi)
        elif mode in {"same_random", "vortex", "random_plus_vortex"}:
            patch = base_patch
            if patch.shape != (aperture.height, aperture.width):
                raise ValueError("Kernel regions must have equal sizes for shared kernel modes.")
        else:
            raise ValueError("Unsupported kernel_phase_mode: %s" % mode)
        phase[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = patch
    return phase


def amplitude_case_dict(seed: int, custom: Optional[str]) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    cases = {
        "uniform": [1.0] * 9,
        "center_only": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "diagonal": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "top_row": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "left_col": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "sparse_mix": [1.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.2, 0.0, 0.0],
        "random_seeded": [float(value) for value in rng.uniform(0.0, 1.0, size=9)],
        "task_like_mnist": [1.0, 0.1, 0.2, 0.1, 0.9, 0.1, 0.2, 0.1, 0.7],
        "task_like_fashion": [0.8, 0.9, 0.8, 0.2, 0.7, 0.2, 0.1, 0.1, 0.1],
        "task_like_emnist": [0.9, 0.1, 0.1, 0.8, 0.3, 0.1, 0.7, 0.1, 0.6],
    }
    for index, expert_id in enumerate(EXPERT_IDS):
        values = [0.0] * 9
        values[index] = 1.0
        cases["onehot_" + expert_id] = values
    if custom:
        cases["custom"] = parse_float_list(custom, 9)
    return cases


def build_grating_phase_map(layout: PaperKernelLayout, sign_x: int, sign_y: int, scale: float, device: torch.device) -> torch.Tensor:
    phase = torch.zeros(layout.canvas_shape, dtype=torch.float32, device=device)
    canvas_y, canvas_x = layout.canvas_center
    for index, region in enumerate(layout.kernel_regions):
        expert = layout.expert_apertures[index]
        expert_y, expert_x = expert.center
        dx_px = expert_x - canvas_x
        dy_px = expert_y - canvas_y
        region_phase = make_grating_phase_for_shift(dx_px, dy_px, layout, sign_x, sign_y, scale, device)
        phase[region.y0 : region.y1, region.x0 : region.x1] = region_phase[region.y0 : region.y1, region.x0 : region.x1]
    return phase


def build_total_mask(
    layout: PaperKernelLayout,
    amplitudes: Sequence[float],
    kernel_phase: torch.Tensor,
    grating_phase: torch.Tensor,
    global_lens_phase: torch.Tensor,
    use_global_lens: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(amplitudes) != 9:
        raise ValueError("amplitudes must contain exactly 9 values.")
    kernel_masks = aperture_masks(layout, layout.kernel_regions, device)
    amplitude_tensor = torch.tensor(amplitudes, dtype=torch.float32, device=device)
    total_phase = kernel_phase + grating_phase
    if use_global_lens:
        total_phase = total_phase + global_lens_phase
    amplitude_map = torch.sum(kernel_masks * amplitude_tensor.view(9, 1, 1), dim=0)
    transmission = torch.sum(
        kernel_masks
        * amplitude_tensor.view(9, 1, 1)
        * torch.exp(1j * total_phase).to(torch.complex64).unsqueeze(0),
        dim=0,
    )
    return transmission.to(torch.complex64), total_phase, amplitude_map


def energy_ratios(intensity: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    if intensity.ndim == 3:
        intensity = intensity[0]
    masked = (intensity.unsqueeze(0) * masks).sum(dim=(-2, -1))
    total = float(intensity.sum().item() + EPS)
    inside = float(masked.sum().item())
    outside = max(total - inside, 0.0)
    return masked / total, outside / total, total


def normalized_inside_ratios(intensity: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    ratios, _outside, _total = energy_ratios(intensity, masks)
    values = ratios.detach().cpu().numpy().astype(np.float64)
    return values / (values.sum() + EPS)


def centroid(intensity: torch.Tensor) -> Tuple[float, float]:
    if intensity.ndim == 3:
        intensity = intensity[0]
    total_raw = float(intensity.sum().item())
    if total_raw < EPS:
        return (float(intensity.shape[-2]) / 2.0, float(intensity.shape[-1]) / 2.0)
    total = intensity.sum() + EPS
    y, x = pixel_grids_from_shape(intensity.shape, intensity.device)
    cy = float((intensity * y).sum().item() / total.item())
    cx = float((intensity * x).sum().item() / total.item())
    return cy, cx


def pixel_grids_from_shape(shape, device):
    height, width = int(shape[-2]), int(shape[-1])
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    return torch.meshgrid(y, x, indexing="ij")


def crop_aperture(tensor: torch.Tensor, aperture: Aperture) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor[0]
    return tensor[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]


def normalize_patch(patch: torch.Tensor) -> np.ndarray:
    values = patch.detach().cpu().float().numpy().astype(np.float64)
    norm = np.linalg.norm(values.reshape(-1)) + EPS
    return values / norm


def patch_similarity(intensity: torch.Tensor, layout: PaperKernelLayout) -> Tuple[List[Dict], np.ndarray, Dict]:
    patches = [normalize_patch(crop_aperture(intensity, aperture)) for aperture in layout.expert_apertures]
    matrix = np.zeros((9, 9), dtype=np.float64)
    rows = []
    nrmse_values = []
    cosine_values = []
    corr_values = []
    for i in range(9):
        for j in range(9):
            a = patches[i].reshape(-1)
            b = patches[j].reshape(-1)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
            corr = cosine
            nrmse = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + EPS))
            matrix[i, j] = corr
            if i < j:
                rows.append({
                    "expert_a": EXPERT_IDS[i],
                    "expert_b": EXPERT_IDS[j],
                    "pairwise_correlation": corr,
                    "pairwise_cosine": cosine,
                    "pairwise_nrmse": nrmse,
                })
                corr_values.append(corr)
                cosine_values.append(cosine)
                nrmse_values.append(nrmse)
    stats = {
        "mean_pairwise_correlation": float(np.mean(corr_values)) if corr_values else 0.0,
        "min_pairwise_correlation": float(np.min(corr_values)) if corr_values else 0.0,
        "mean_pairwise_cosine": float(np.mean(cosine_values)) if cosine_values else 0.0,
        "max_pairwise_nrmse": float(np.max(nrmse_values)) if nrmse_values else 0.0,
    }
    return rows, matrix, stats


def vector_metrics(commanded: np.ndarray, measured: np.ndarray) -> Dict:
    commanded = np.asarray(commanded, dtype=np.float64)
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
        "rmse_commanded_measured": rmse(commanded, measured),
        "pearson_commanded_measured": pearson(commanded, measured),
    }


def entropy(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values / (values.sum() + EPS)
    return float(-(values * np.log(values + EPS)).sum())


def local_phase_gradient(field: torch.Tensor, layout: PaperKernelLayout) -> List[Dict]:
    if field.ndim == 3:
        field = field[0]
    rows = []
    for index, aperture in enumerate(layout.expert_apertures):
        patch = crop_aperture(field, aperture)
        if patch.numel() == 0:
            grad_x = 0.0
            grad_y = 0.0
        else:
            diff_x = patch[:, 1:] * torch.conj(patch[:, :-1])
            diff_y = patch[1:, :] * torch.conj(patch[:-1, :])
            grad_x = float(torch.angle(diff_x).mean().item()) if diff_x.numel() else 0.0
            grad_y = float(torch.angle(diff_y).mean().item()) if diff_y.numel() else 0.0
        rows.append({
            "expert_id": EXPERT_IDS[index],
            "phase_gradient_x_rad_per_px": grad_x,
            "phase_gradient_y_rad_per_px": grad_y,
        })
    return rows


def build_propagator(wavelength_m: float, pixel_size_m: float, grid_size, distance_m: float, device: torch.device):
    return AngularSpectrumPropagator(
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
        grid_size=grid_size,
        distance_m=distance_m,
    ).to(device)


def run_optical_path(
    input_field: torch.Tensor,
    transmission: torch.Tensor,
    layout: PaperKernelLayout,
    args,
    physical: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.propagation_mode == "fft_4f":
        spectrum = centered_fft2(input_field)
        output = centered_ifft2(spectrum * transmission.unsqueeze(0))
        return spectrum, output
    input_to_mask = build_propagator(
        physical["wavelength_m"],
        physical["pixel_size_m"],
        layout.canvas_shape,
        physical["input_to_mask_m"],
        device,
    )
    mask_to_expert = build_propagator(
        physical["wavelength_m"],
        physical["pixel_size_m"],
        layout.canvas_shape,
        physical["mask_to_expert_m"],
        device,
    )
    incident = input_to_mask(input_field)
    output = mask_to_expert(incident * transmission.unsqueeze(0))
    return incident, output


def propagate_identity_layers(
    expert_entrance: torch.Tensor,
    layout: PaperKernelLayout,
    masks: Dict[str, torch.Tensor],
    physical: Dict,
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict]]:
    field = expert_entrance
    if args.hard_aperture:
        field = field * masks["expert_union"].unsqueeze(0).to(torch.complex64)
    propagator = build_propagator(
        physical["wavelength_m"],
        physical["pixel_size_m"],
        layout.canvas_shape,
        physical["inter_layer_m"],
        device,
    )
    rows = []
    reference = None
    for layer_idx in range(int(args.num_dummy_layers) + 1):
        intensity = torch.abs(field).square()
        for expert_index, aperture in enumerate(layout.expert_apertures):
            patch = crop_aperture(intensity, aperture)
            cy_local, cx_local = centroid(patch)
            cy = cy_local + aperture.y0
            cx = cx_local + aperture.x0
            if layer_idx == 0:
                if reference is None:
                    reference = {}
                reference[expert_index] = (cy, cx)
            ref_y, ref_x = reference.get(expert_index, (cy, cx)) if reference else (cy, cx)
            rows.append({
                "plane": "layer%d" % layer_idx if layer_idx > 0 else "layer0_expert_entrance",
                "expert_id": EXPERT_IDS[expert_index],
                "centroid_y": cy,
                "centroid_x": cx,
                "drift_from_entrance_px": math.sqrt((cy - ref_y) ** 2 + (cx - ref_x) ** 2),
            })
        if layer_idx < int(args.num_dummy_layers):
            field = propagator(field)
            if args.hard_aperture:
                field = field * masks["expert_union"].unsqueeze(0).to(torch.complex64)
    detector = propagator(field)
    return detector, rows


def calibrate_grating_signs(
    layout: PaperKernelLayout,
    args,
    physical: Dict,
    masks: Dict[str, torch.Tensor],
    kernel_phase: torch.Tensor,
    global_lens_phase: torch.Tensor,
    device: torch.device,
) -> Tuple[int, int, List[Dict]]:
    input_field = make_input_field("flat_top_200", layout, device)
    rows = []
    best = None
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            grating_phase = build_grating_phase_map(layout, sign_x, sign_y, args.grating_shift_scale, device)
            total_score = 0.0
            diagonal_top = 0
            total_centroid_error = 0.0
            for target_index, expert_id in enumerate(EXPERT_IDS):
                amps = [0.0] * 9
                amps[target_index] = 1.0
                transmission, _phase, _amp = build_total_mask(
                    layout,
                    amps,
                    kernel_phase,
                    grating_phase,
                    global_lens_phase,
                    args.use_global_lens_phase,
                    device,
                )
                _incident, output = run_optical_path(input_field, transmission, layout, args, physical, device)
                intensity = torch.abs(output).square()
                inside = normalized_inside_ratios(intensity, masks["experts"])
                top = int(np.argmax(inside))
                if top == target_index:
                    diagonal_top += 1
                patch = crop_aperture(intensity, layout.expert_apertures[target_index])
                cy_local, cx_local = centroid(patch)
                aperture = layout.expert_apertures[target_index]
                cy = cy_local + aperture.y0
                cx = cx_local + aperture.x0
                target_y, target_x = aperture.center
                error = math.sqrt((cy - target_y) ** 2 + (cx - target_x) ** 2)
                total_centroid_error += error
                total_score += float(inside[target_index]) - 0.002 * error
            row = {
                "sign_x": sign_x,
                "sign_y": sign_y,
                "score": total_score,
                "diagonal_top_count": diagonal_top,
                "mean_centroid_error_px": total_centroid_error / 9.0,
            }
            rows.append(row)
            if best is None or (row["diagonal_top_count"], row["score"]) > (best["diagonal_top_count"], best["score"]):
                best = row
    return int(best["sign_x"]), int(best["sign_y"]), rows


def plot_layout(layout: PaperKernelLayout, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("black")
    input_ap = layout.input_aperture
    ax.add_patch(Rectangle((input_ap.x0, input_ap.y0), input_ap.width, input_ap.height, fill=False, edgecolor="orange", linewidth=2.0))
    ax.text(input_ap.x0, input_ap.y0 - 5, "input 200x200", color="orange", fontsize=9)
    for index, aperture in enumerate(layout.kernel_regions):
        ax.add_patch(Rectangle((aperture.x0, aperture.y0), aperture.width, aperture.height, fill=False, edgecolor="cyan", linewidth=1.3, linestyle="--"))
        cy, cx = aperture.center
        ax.text(cx - 18, cy - 4, KERNEL_IDS[index], color="cyan", fontsize=8)
    for index, aperture in enumerate(layout.expert_apertures):
        ax.add_patch(Rectangle((aperture.x0, aperture.y0), aperture.width, aperture.height, fill=False, edgecolor="lime", linewidth=1.4))
        cy, cx = aperture.center
        ax.plot(cx, cy, "o", color="lime", markersize=3)
        ax.text(cx - 18, cy + 13, EXPERT_IDS[index], color="lime", fontsize=8)
    cy, cx = layout.canvas_center
    ax.axhline(cy, color="white", alpha=0.35, linewidth=0.8)
    ax.axvline(cx, color="white", alpha=0.35, linewidth=0.8)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_title("Paper-style 9-kernel layout")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_array(array, layout: PaperKernelLayout, path: Path, title: str, dpi: int, max_plot_dim: int, label: str, overlay_experts: bool = True, overlay_kernels: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    scale = min(1.0, float(max_plot_dim) / float(max(height, width)))
    fig, ax = plt.subplots(figsize=(max(5, width * scale / 120), max(5, height * scale / 120)))
    im = ax.imshow(array, cmap="inferno")
    if overlay_kernels:
        for aperture in layout.kernel_regions:
            ax.add_patch(Rectangle((aperture.x0, aperture.y0), aperture.width, aperture.height, fill=False, edgecolor="cyan", linewidth=0.9, linestyle="--"))
    if overlay_experts:
        for aperture in layout.expert_apertures:
            ax.add_patch(Rectangle((aperture.x0, aperture.y0), aperture.width, aperture.height, fill=False, edgecolor="lime", linewidth=0.9))
    input_ap = layout.input_aperture
    ax.add_patch(Rectangle((input_ap.x0, input_ap.y0), input_ap.width, input_ap.height, fill=False, edgecolor="orange", linewidth=0.8))
    ax.set_title(title)
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_intensity(field_or_intensity, layout: PaperKernelLayout, path: Path, title: str, dpi: int, max_plot_dim: int, overlay_experts: bool = True, overlay_kernels: bool = False, save_linear: bool = False):
    if torch.is_complex(field_or_intensity):
        intensity = torch.abs(field_or_intensity.to(torch.complex64)).square()
    else:
        intensity = field_or_intensity.float()
    if intensity.ndim == 3:
        intensity = intensity[0]
    array = intensity.detach().cpu().float().numpy()
    log_array = np.log10(array / (array.max() + EPS) + 1e-8)
    plot_array(log_array, layout, path, title, dpi, max_plot_dim, "log10(I/Imax+1e-8)", overlay_experts, overlay_kernels)
    if save_linear:
        plot_array(array, layout, path.with_name(path.stem + "_linear.png"), title, dpi, max_plot_dim, "linear intensity", overlay_experts, overlay_kernels)


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
    array = np.asarray(values, dtype=np.float64).reshape(3, 3)
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


def plot_matrix(matrix: np.ndarray, path: Path, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels(EXPERT_IDS, rotation=45, ha="right")
    ax.set_yticklabels(EXPERT_IDS)
    ax.set_xlabel("measured expert")
    ax.set_ylabel("activated kernel channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
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


def plot_drift_summary(rows: List[Dict], path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_expert: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        expert = row["expert_id"]
        plane = row["plane"]
        if plane == "layer0_expert_entrance":
            layer = 0
        else:
            layer = int(plane.replace("layer", ""))
        by_expert.setdefault(expert, []).append((layer, float(row["drift_from_entrance_px"])))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for expert, values in by_expert.items():
        values = sorted(values)
        ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=expert)
    ax.set_xlabel("identity propagation layer")
    ax.set_ylabel("centroid drift from entrance (px)")
    ax.set_title("Centroid drift by expert")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
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


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activated\\measured"] + EXPERT_IDS)
        for index, expert_id in enumerate(EXPERT_IDS):
            writer.writerow([expert_id] + [float(value) for value in matrix[index]])


def json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def run_case(
    input_type: str,
    amplitude_case: str,
    amplitudes: Sequence[float],
    layout: PaperKernelLayout,
    physical: Dict,
    args,
    masks: Dict[str, torch.Tensor],
    kernel_phase: torch.Tensor,
    grating_phase: torch.Tensor,
    global_lens_phase: torch.Tensor,
    output_dir: Path,
    device: torch.device,
) -> Tuple[Dict, List[Dict], List[Dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_field = make_input_field(input_type, layout, device)
    transmission, total_phase, amplitude_map = build_total_mask(
        layout,
        amplitudes,
        kernel_phase,
        grating_phase,
        global_lens_phase,
        args.use_global_lens_phase,
        device,
    )
    incident, expert_entrance = run_optical_path(input_field, transmission, layout, args, physical, device)
    intensity = torch.abs(expert_entrance).square()
    expert_energy, outside_ratio, total_energy = energy_ratios(intensity, masks["experts"])
    measured_inside = normalized_inside_ratios(intensity, masks["experts"])
    commanded = np.asarray(amplitudes, dtype=np.float64) ** 2
    commanded = commanded / (commanded.sum() + EPS)
    vec = vector_metrics(commanded, measured_inside)
    top = int(np.argmax(measured_inside))
    second = float(np.sort(measured_inside)[-2]) if measured_inside.size > 1 else 0.0

    detector, drift_rows = propagate_identity_layers(expert_entrance, layout, masks, physical, args, device)
    gradient_rows = local_phase_gradient(expert_entrance, layout)
    for row in drift_rows:
        row.update({
            "input_type": input_type,
            "kernel_phase_mode": args.kernel_phase_mode,
            "amplitude_case": amplitude_case,
        })
    for row in gradient_rows:
        row.update({
            "input_type": input_type,
            "kernel_phase_mode": args.kernel_phase_mode,
            "amplitude_case": amplitude_case,
        })

    row = {
        "input_type": input_type,
        "propagation_mode": args.propagation_mode,
        "kernel_phase_mode": args.kernel_phase_mode,
        "amplitude_case": amplitude_case,
        "use_global_lens_phase": bool(args.use_global_lens_phase),
        "total_energy_expert_entrance": total_energy,
        "outside_all_experts_energy_ratio": outside_ratio,
        "expert_energy_entropy": entropy(measured_inside),
        "top_expert_id": EXPERT_IDS[top],
        "second_largest_expert_energy_ratio": second,
    }
    row.update(vec)
    for index, expert_id in enumerate(EXPERT_IDS):
        row[expert_id + "_amplitude"] = float(amplitudes[index])
        row[expert_id + "_commanded_power"] = float(commanded[index])
        row[expert_id + "_energy_ratio"] = float(expert_energy[index].detach().cpu().item())
        row[expert_id + "_measured_inside_ratio"] = float(measured_inside[index])

    should_plot = input_type == "flat_top_200" or amplitude_case in {"uniform", "center_only", "diagonal"}
    if should_plot:
        case_dir = output_dir / "figures" / input_type / amplitude_case
        plot_intensity(expert_entrance, layout, case_dir / "output_expert_entrance_intensity.png", "expert entrance: %s" % amplitude_case, args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)
        after_aperture = expert_entrance * masks["expert_union"].unsqueeze(0).to(torch.complex64) if args.hard_aperture else expert_entrance
        plot_intensity(after_aperture, layout, case_dir / "output_after_expert_aperture.png", "after expert aperture: %s" % amplitude_case, args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_heatmap_3x3(measured_inside, case_dir / "measured_expert_energy_3x3.png", "measured expert energy: %s" % amplitude_case, "inside-normalized energy", args.plot_dpi)
        plot_intensity(detector, layout, case_dir / "detector_plane_intensity.png", "detector plane: %s" % amplitude_case, args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_heatmap_3x3(commanded, case_dir / "commanded_power_3x3.png", "commanded power: %s" % amplitude_case, "power", args.plot_dpi)

    return row, drift_rows, gradient_rows, input_field, incident, expert_entrance, detector, amplitude_map


def status_lower(value: float, pass_value: float, warn_value: float) -> str:
    if value < pass_value:
        return "PASS"
    if value <= warn_value:
        return "WARN"
    return "FAIL"


def status_higher(value: float, pass_value: float, warn_value: float) -> str:
    if value > pass_value:
        return "PASS"
    if value >= warn_value:
        return "WARN"
    return "FAIL"


def build_summary(rows: List[Dict], similarity_stats: Dict, crosstalk: np.ndarray, drift_rows: List[Dict]) -> Dict:
    baseline = None
    for row in rows:
        if (
            row["input_type"] == "flat_top_200"
            and row["kernel_phase_mode"] == "same_identity"
            and row["amplitude_case"] == "uniform"
        ):
            baseline = row
            break
    if baseline is None:
        baseline = next((row for row in rows if row["input_type"] == "flat_top_200" and row["amplitude_case"] == "uniform"), None)
    if baseline:
        active_count = sum(1 for expert_id in EXPERT_IDS if float(baseline[expert_id + "_measured_inside_ratio"]) > 0.02)
    else:
        active_count = 0
    active_status = "PASS" if active_count >= 7 else ("WARN" if active_count >= 5 else "FAIL")
    mean_corr = float(similarity_stats.get("mean_pairwise_correlation", 0.0))
    similarity_status = status_higher(mean_corr, 0.85, 0.65)
    diagonal_top = 0
    for index in range(9):
        if int(np.argmax(crosstalk[index])) == index:
            diagonal_top += 1
    crosstalk_status = "PASS" if diagonal_top >= 7 else ("WARN" if diagonal_top >= 5 else "FAIL")
    max_drift = 0.0
    for row in drift_rows:
        max_drift = max(max_drift, float(row.get("drift_from_entrance_px", 0.0)))
    drift_status = status_lower(max_drift, 25.0, 45.0)
    statuses = [active_status, similarity_status, crosstalk_status, drift_status]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    conclusions = []
    if similarity_status == "PASS" and active_status == "PASS":
        conclusions.append("same-kernel outputs are spatially separated and mutually similar")
    elif active_status == "PASS":
        conclusions.append("grating shift reaches target experts but same-kernel similarity is limited")
    else:
        conclusions.append("current phase organization does not yet produce equivalent feature maps")
    if crosstalk_status in {"PASS", "WARN"}:
        conclusions.append("amplitude weights correlate with measured expert entrance energy")
    if drift_status != "PASS":
        conclusions.append("grating shift reaches target experts but patches drift during expert propagation")

    return {
        "overall_status": overall,
        "active_expert_count_gt_0p02": active_count,
        "active_experts_status": active_status,
        "mean_pairwise_correlation": mean_corr,
        "same_kernel_similarity_status": similarity_status,
        "one_hot_diagonal_top_count": diagonal_top,
        "one_hot_crosstalk_status": crosstalk_status,
        "max_centroid_drift_px": max_drift,
        "drift_status": drift_status,
        "conclusion": conclusions,
    }


def kernel_incident_summary(kernel_incident_ratios: torch.Tensor) -> Dict:
    values = kernel_incident_ratios.detach().cpu().numpy().astype(np.float64)
    top_index = int(np.argmax(values))
    center_index = EXPERT_IDS.index("E11")
    mean = float(values.mean())
    std = float(values.std())
    center_ratio = float(values[center_index])
    max_ratio = float(values[top_index])
    min_ratio = float(values.min())
    cv = std / (mean + EPS)
    if center_ratio > 0.5:
        status = "CENTER_DOMINATED"
        diagnosis = (
            "Most incident optical power reaches the center kernel region. "
            "A spatially partitioned kernel mask cannot route energy from dark "
            "kernel regions; amplitudes only gate light that already reaches "
            "each region."
        )
    elif cv < 0.25:
        status = "UNIFORM"
        diagnosis = "Kernel regions receive relatively balanced incident power."
    else:
        status = "UNBALANCED"
        diagnosis = "Kernel regions receive uneven incident power; channel efficiencies will differ."
    payload = {
        "status": status,
        "diagnosis": diagnosis,
        "top_kernel_id": KERNEL_IDS[top_index],
        "top_kernel_incident_energy_ratio": max_ratio,
        "center_kernel_incident_energy_ratio": center_ratio,
        "min_kernel_incident_energy_ratio": min_ratio,
        "mean_kernel_incident_energy_ratio": mean,
        "std_kernel_incident_energy_ratio": std,
        "cv_kernel_incident_energy_ratio": cv,
    }
    for index, kernel_id in enumerate(KERNEL_IDS):
        payload[kernel_id + "_incident_energy_ratio"] = float(values[index])
    return payload


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    layout = build_layout(args)

    wavelength_m = nm_to_m(args.wavelength_nm)
    pixel_size_m = um_to_m(args.pixel_size_um)
    physical = {
        "wavelength_nm": args.wavelength_nm,
        "pixel_size_um": args.pixel_size_um,
        "wavelength_m": wavelength_m,
        "pixel_size_m": pixel_size_m,
        "focal_length_cm": args.focal_length_cm,
        "focal_length_m": cm_to_m(args.focal_length_cm),
        "input_to_mask_cm": args.input_to_mask_cm,
        "input_to_mask_m": cm_to_m(args.input_to_mask_cm),
        "mask_to_expert_cm": args.mask_to_expert_cm,
        "mask_to_expert_m": cm_to_m(args.mask_to_expert_cm),
        "inter_layer_cm": args.inter_layer_cm,
        "inter_layer_m": cm_to_m(args.inter_layer_cm),
        "num_dummy_layers": args.num_dummy_layers,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "nine_expert_paper_style_kernels" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("paper-style 9-kernel prompt diagnostic")
    print("device: %s" % device)
    print("output_dir: %s" % output_dir)
    print("propagation_mode: %s" % args.propagation_mode)
    print("kernel_phase_mode: %s" % args.kernel_phase_mode)
    print("use_global_lens_phase: %s" % args.use_global_lens_phase)
    baseline_area = 4 * 200 * 200
    nine_area = 9 * args.expert_size * args.expert_size
    print("4-expert baseline area = %d" % baseline_area)
    print("9-expert matched area = %d, relative difference = %+0.2f%%" % (nine_area, (nine_area - baseline_area) / float(baseline_area) * 100.0))

    save_json(output_dir / "layout.json", layout.to_dict())
    save_json(output_dir / "kernel_regions.json", [aperture.to_dict() for aperture in layout.kernel_regions])
    save_json(output_dir / "physical_params.json", physical)
    plot_layout(layout, output_dir / "figures" / "layout_overlay.png", args.plot_dpi)

    masks = {
        "experts": aperture_masks(layout, layout.expert_apertures, device),
        "kernels": aperture_masks(layout, layout.kernel_regions, device),
    }
    masks["expert_union"] = torch.clamp(masks["experts"].sum(dim=0), 0.0, 1.0)
    masks["kernel_union"] = torch.clamp(masks["kernels"].sum(dim=0), 0.0, 1.0)

    kernel_phase = build_kernel_phase_map(layout, args.kernel_phase_mode, device)
    global_lens_phase = make_global_lens_phase(layout, wavelength_m, pixel_size_m, physical["focal_length_m"], device)

    if args.calibrate_grating_signs:
        sign_x, sign_y, calibration_rows = calibrate_grating_signs(
            layout,
            args,
            physical,
            masks,
            kernel_phase,
            global_lens_phase,
            device,
        )
    else:
        sign_x, sign_y = args.grating_sign_x, args.grating_sign_y
        calibration_rows = [{"sign_x": sign_x, "sign_y": sign_y, "score": "", "diagonal_top_count": "", "mean_centroid_error_px": ""}]
    save_csv(output_dir / "calibrated_grating_signs.csv", calibration_rows)
    save_json(output_dir / "calibrated_grating_signs.json", {"sign_x": sign_x, "sign_y": sign_y, "calibrate_grating_signs": bool(args.calibrate_grating_signs)})
    print("grating signs: sign_x=%d, sign_y=%d" % (sign_x, sign_y))

    grating_phase = build_grating_phase_map(layout, sign_x, sign_y, args.grating_shift_scale, device)
    all_cases = amplitude_case_dict(args.seed, args.custom_amplitudes)
    requested_cases = [item.strip() for item in args.amplitude_cases.split(",") if item.strip()]
    cases_to_run = []
    for name in requested_cases + ["onehot_" + expert_id for expert_id in EXPERT_IDS]:
        if name not in all_cases:
            raise ValueError("Unknown amplitude case: %s" % name)
        if name not in cases_to_run:
            cases_to_run.append(name)
    save_json(output_dir / "amplitude_cases.json", {name: all_cases[name] for name in cases_to_run})

    input_types = [item.strip() for item in args.input_types.split(",") if item.strip()]
    if "flat_top_200" not in input_types:
        input_types = ["flat_top_200"] + input_types

    input_for_plots = make_input_field("flat_top_200", layout, device)
    reference_transmission, reference_total_phase, reference_amplitude_map = build_total_mask(
        layout,
        all_cases["uniform"],
        kernel_phase,
        grating_phase,
        global_lens_phase,
        args.use_global_lens_phase,
        device,
    )
    incident_plane, reference_output = run_optical_path(input_for_plots, reference_transmission, layout, args, physical, device)
    kernel_incident_intensity = torch.abs(incident_plane).square() if args.propagation_mode == "angular_spectrum_lens" else torch.abs(incident_plane).square()
    kernel_incident_ratios, _outside_kernel, _total_kernel = energy_ratios(kernel_incident_intensity, masks["kernels"])
    kernel_incident_payload = kernel_incident_summary(kernel_incident_ratios)
    save_csv(
        output_dir / "kernel_region_incident_energy.csv",
        [{"kernel_id": kernel_id, "incident_energy_ratio": float(kernel_incident_ratios[index].detach().cpu().item())} for index, kernel_id in enumerate(KERNEL_IDS)],
    )
    plot_intensity(input_for_plots, layout, output_dir / "figures" / "input_intensity.png", "input intensity", args.plot_dpi, args.max_plot_dim, overlay_experts=False, overlay_kernels=True, save_linear=args.save_linear_intensity)
    plot_intensity(kernel_incident_intensity, layout, output_dir / "figures" / "kernel_plane_incident_intensity.png", "kernel plane incident intensity", args.plot_dpi, args.max_plot_dim, overlay_experts=False, overlay_kernels=True, save_linear=args.save_linear_intensity)
    plot_heatmap_3x3(kernel_incident_ratios.detach().cpu().numpy(), output_dir / "figures" / "kernel_region_incident_energy_3x3.png", "kernel region incident energy", "energy ratio", args.plot_dpi)
    plot_phase(kernel_phase, output_dir / "figures" / "kernel_phase_wrapped.png", "kernel phase", args.plot_dpi)
    plot_phase(grating_phase, output_dir / "figures" / "grating_phase_wrapped.png", "grating phase", args.plot_dpi)
    plot_phase(global_lens_phase, output_dir / "figures" / "global_lens_phase_wrapped.png", "global lens phase", args.plot_dpi)
    plot_phase(reference_total_phase, output_dir / "figures" / "total_phase_profile_wrapped.png", "total phase profile", args.plot_dpi)
    plot_intensity(reference_amplitude_map, layout, output_dir / "figures" / "total_amplitude_map.png", "total amplitude map", args.plot_dpi, args.max_plot_dim, overlay_experts=False, overlay_kernels=True)

    rows: List[Dict] = []
    drift_rows: List[Dict] = []
    gradient_rows: List[Dict] = []
    baseline_outputs = None
    for input_type in input_types:
        for case_name in cases_to_run:
            row, drift, gradient, input_field, incident, entrance, detector, amplitude_map = run_case(
                input_type,
                case_name,
                all_cases[case_name],
                layout,
                physical,
                args,
                masks,
                kernel_phase,
                grating_phase,
                global_lens_phase,
                output_dir,
                device,
            )
            rows.append(row)
            drift_rows.extend(drift)
            gradient_rows.extend(gradient)
            if input_type == "flat_top_200" and case_name == "uniform":
                baseline_outputs = (input_field, incident, entrance, detector, amplitude_map)

    if baseline_outputs is not None:
        _input_field, _incident, entrance, detector, _amplitude_map = baseline_outputs
        after_aperture = entrance * masks["expert_union"].unsqueeze(0).to(torch.complex64) if args.hard_aperture else entrance
        measured = normalized_inside_ratios(torch.abs(entrance).square(), masks["experts"])
        plot_intensity(entrance, layout, output_dir / "figures" / "output_expert_entrance_intensity.png", "output expert entrance: flat_top uniform", args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_intensity(after_aperture, layout, output_dir / "figures" / "output_after_expert_aperture.png", "output after expert aperture: flat_top uniform", args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)
        plot_heatmap_3x3(measured, output_dir / "figures" / "measured_expert_energy_3x3.png", "measured expert energy: flat_top uniform", "inside-normalized energy", args.plot_dpi)
        plot_intensity(detector, layout, output_dir / "figures" / "detector_plane_intensity.png", "detector plane: flat_top uniform", args.plot_dpi, args.max_plot_dim, overlay_experts=True, save_linear=args.save_linear_intensity)

    save_csv(output_dir / "metrics.csv", rows)
    amp_rows = [
        {
            "input_type": row["input_type"],
            "kernel_phase_mode": row["kernel_phase_mode"],
            "amplitude_case": row["amplitude_case"],
            "cosine_commanded_measured": row["cosine_commanded_measured"],
            "rmse_commanded_measured": row["rmse_commanded_measured"],
            "pearson_commanded_measured": row["pearson_commanded_measured"],
        }
        for row in rows
    ]
    save_csv(output_dir / "amplitude_command_vs_measured.csv", amp_rows)
    plot_bar([row for row in rows if row["input_type"] == "flat_top_200" and not row["amplitude_case"].startswith("onehot_")], "cosine_commanded_measured", output_dir / "figures" / "amplitude_command_vs_measured_bar.png", "Amplitude command vs measured", "cosine", args.plot_dpi)

    crosstalk = np.zeros((9, 9), dtype=np.float64)
    for row in rows:
        if row["input_type"] != "flat_top_200":
            continue
        case_name = row["amplitude_case"]
        if not case_name.startswith("onehot_"):
            continue
        active = EXPERT_IDS.index(case_name.replace("onehot_", ""))
        for expert_index, expert_id in enumerate(EXPERT_IDS):
            crosstalk[active, expert_index] = float(row[expert_id + "_measured_inside_ratio"])
    save_matrix_csv(output_dir / "one_hot_crosstalk_matrix.csv", crosstalk)
    plot_matrix(crosstalk, output_dir / "figures" / "one_hot_crosstalk_matrix.png", "One-hot crosstalk matrix", args.plot_dpi)

    if baseline_outputs is not None:
        _input_field, _incident, entrance, _detector, _amplitude_map = baseline_outputs
        similarity_rows, similarity_matrix, similarity_stats = patch_similarity(torch.abs(entrance).square(), layout)
    else:
        similarity_rows, similarity_matrix, similarity_stats = [], np.zeros((9, 9)), {}
    save_csv(output_dir / "expert_patch_similarity.csv", similarity_rows)
    plot_matrix(similarity_matrix, output_dir / "figures" / "expert_patch_similarity_heatmap.png", "Expert patch similarity", args.plot_dpi)

    save_csv(output_dir / "centroid_drift_by_layer.csv", drift_rows)
    plot_drift_summary(
        [row for row in drift_rows if row["input_type"] == "flat_top_200" and row["amplitude_case"] == "uniform"],
        output_dir / "figures" / "centroid_drift_summary.png",
        args.plot_dpi,
    )
    save_csv(output_dir / "local_phase_gradient_by_expert.csv", gradient_rows)

    summary = build_summary(rows, similarity_stats, crosstalk, [row for row in drift_rows if row["input_type"] == "flat_top_200" and row["amplitude_case"] == "uniform"])
    summary["layout"] = layout.to_dict()
    summary["physical_params"] = physical
    summary["propagation_mode"] = args.propagation_mode
    summary["kernel_phase_mode"] = args.kernel_phase_mode
    summary["use_global_lens_phase"] = bool(args.use_global_lens_phase)
    summary["grating_signs"] = {"sign_x": sign_x, "sign_y": sign_y}
    summary["similarity_stats"] = similarity_stats
    summary["kernel_region_incident_energy"] = kernel_incident_payload
    if kernel_incident_payload["status"] == "CENTER_DOMINATED":
        summary["conclusion"].append("center kernel illumination dominates; the current partitioned prompt is not copying the 200x200 input to all kernels")
    save_json(output_dir / "summary.json", summary)

    print("overall status: %s" % summary["overall_status"])
    print("conclusion: %s" % "; ".join(summary["conclusion"]))
    print("metrics: %s" % (output_dir / "metrics.csv"))
    print("summary: %s" % (output_dir / "summary.json"))


if __name__ == "__main__":
    main()
