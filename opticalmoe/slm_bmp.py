"""Utilities for exporting simulated optical planes to physical SLM BMP files."""

import math
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F


def _to_2d_float(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()
    while value.ndim > 2:
        value = value[0]
    return value.float()


def encode_amplitude_uint8(amplitude: torch.Tensor) -> torch.Tensor:
    value = _to_2d_float(amplitude).clamp(0.0, 1.0)
    return torch.round(value * 255.0).to(torch.uint8)


def encode_phase_uint8(phase: torch.Tensor) -> torch.Tensor:
    value = torch.remainder(_to_2d_float(phase), 2.0 * math.pi)
    return torch.round(value * (255.0 / (2.0 * math.pi))).to(torch.uint8)


def nearest_scale_uint8(image: torch.Tensor, scale_factor: int) -> torch.Tensor:
    factor = int(scale_factor)
    if factor <= 0:
        raise ValueError("scale_factor must be positive.")
    value = image.to(torch.float32).unsqueeze(0).unsqueeze(0)
    scaled = F.interpolate(value, scale_factor=factor, mode="nearest")
    return scaled[0, 0].round().to(torch.uint8)


def center_pad_to_slm(image: torch.Tensor, slm_width: int = 1920, slm_height: int = 1200) -> torch.Tensor:
    value = image.to(torch.uint8)
    height, width = value.shape
    if height > int(slm_height) or width > int(slm_width):
        raise ValueError(
            f"Scaled optical plane {width}x{height} exceeds SLM {slm_width}x{slm_height}."
        )
    canvas = torch.zeros(int(slm_height), int(slm_width), dtype=torch.uint8)
    y0 = (int(slm_height) - height) // 2
    x0 = (int(slm_width) - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = value
    return canvas


def export_plane_bmp(
    tensor: torch.Tensor,
    path,
    value_type: str,
    scale_factor: int = 2,
    slm_width: int = 1920,
    slm_height: int = 1200,
):
    if value_type == "amplitude":
        encoded = encode_amplitude_uint8(tensor)
    elif value_type == "phase":
        encoded = encode_phase_uint8(tensor)
    else:
        raise ValueError("value_type must be amplitude or phase.")
    scaled = nearest_scale_uint8(encoded, scale_factor)
    slm = center_pad_to_slm(scaled, slm_width=slm_width, slm_height=slm_height)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(slm.numpy(), mode="L").save(path, format="BMP")
    return {
        "path": str(path),
        "value_type": value_type,
        "source_shape": list(encoded.shape),
        "scaled_shape": list(scaled.shape),
        "slm_shape_hw": list(slm.shape),
        "slm_size_wh": [int(slm_width), int(slm_height)],
        "scale_factor": int(scale_factor),
        "padding_value": 0,
    }
