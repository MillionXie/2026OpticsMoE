import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch

from .moe_layout import Aperture


@dataclass(frozen=True)
class SteeringParams:
    shift_m: float
    theta_rad: float
    theta_deg: float
    grating_period_px: float
    phase_increment_rad_per_px: float
    drift_per_inter_layer_px: float

    def to_dict(self):
        return asdict(self)


def compute_steering_params(
    wavelength_m: float,
    pixel_size_m: float,
    shift_pixels: float,
    distance_m: float,
    inter_layer_m: Optional[float] = None,
) -> SteeringParams:
    """Compute grating geometry for steering from canvas center to an expert.

    The grating period is measured in simulation pixels. `distance_m` is the
    prompt-to-first-layer distance that creates the requested lateral shift.
    `inter_layer_m` is optional because some diagnostics want the predicted
    residual drift per later layer after a tilt is left uncompensated.
    """

    shift_m = float(shift_pixels) * float(pixel_size_m)
    theta_rad = math.atan(shift_m / float(distance_m))
    sin_theta = math.sin(abs(theta_rad))
    if sin_theta <= 0.0:
        grating_period_px = float("inf")
        phase_increment = 0.0
    else:
        grating_period_px = float(wavelength_m) / (float(pixel_size_m) * sin_theta)
        phase_increment = 2.0 * math.pi / grating_period_px

    drift_distance = float(distance_m if inter_layer_m is None else inter_layer_m)
    drift = drift_distance * math.tan(theta_rad) / float(pixel_size_m)
    return SteeringParams(
        shift_m=shift_m,
        theta_rad=theta_rad,
        theta_deg=math.degrees(theta_rad),
        grating_period_px=grating_period_px,
        phase_increment_rad_per_px=phase_increment,
        drift_per_inter_layer_px=drift,
    )


def build_linear_grating_phase(
    canvas_shape: Tuple[int, int],
    period_px: float,
    direction: str,
    slope_sign: int = 1,
    multiplier: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a full-canvas x-direction linear phase ramp.

    `slope_sign` is deliberately explicit because FFT/ASM sign conventions can
    differ between implementations. Calibrate it once, then store it in config
    or checkpoint metadata.
    """

    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    if period_px == float("inf"):
        return torch.zeros(canvas_shape, dtype=torch.float32, device=device)

    height, width = int(canvas_shape[0]), int(canvas_shape[1])
    x = torch.arange(width, dtype=torch.float32, device=device).view(1, width)
    x = x - (width / 2.0)
    direction_sign = 1.0 if direction == "right" else -1.0
    phase_increment = 2.0 * math.pi / float(period_px)
    phase = direction_sign * float(slope_sign) * float(multiplier) * phase_increment * x
    return phase.repeat(height, 1)


def build_aperture_mask(
    canvas_shape: Tuple[int, int],
    aperture: Aperture,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    height, width = int(canvas_shape[0]), int(canvas_shape[1])
    mask = torch.zeros(height, width, dtype=torch.float32, device=device)
    y0 = max(0, int(aperture.y0))
    y1 = min(height, int(aperture.y1))
    x0 = max(0, int(aperture.x0))
    x1 = min(width, int(aperture.x1))
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def build_expert_aperture_union(
    canvas_shape: Tuple[int, int],
    left_aperture: Aperture,
    right_aperture: Aperture,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    left = build_aperture_mask(canvas_shape, left_aperture, device=device)
    right = build_aperture_mask(canvas_shape, right_aperture, device=device)
    return torch.clamp(left + right, 0.0, 1.0)


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
    """Build the entrance de-tilt phase for one expert aperture.

    The default de-tilt is the negative of the prompt steering grating and is
    zero outside the chosen expert aperture.
    """

    prompt_phase = build_linear_grating_phase(
        canvas_shape=canvas_shape,
        period_px=period_px,
        direction=direction,
        slope_sign=prompt_slope_sign,
        multiplier=multiplier,
        device=device,
    )
    detilt_phase = prompt_phase if wrong_sign else -prompt_phase
    mask = build_aperture_mask(canvas_shape, aperture, device=device)
    return detilt_phase * mask
