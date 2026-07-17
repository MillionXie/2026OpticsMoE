from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F


def encode_amplitude_uint8(amplitude: torch.Tensor) -> torch.Tensor:
    return torch.round(amplitude.detach().cpu().float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def encode_phase_uint8(phase: torch.Tensor) -> torch.Tensor:
    wrapped = torch.remainder(phase.detach().cpu().float(), 2.0 * math.pi)
    return torch.round(wrapped * (255.0 / (2.0 * math.pi))).to(torch.uint8)


def export_plane_bmp(tensor: torch.Tensor, path: Path, value_type: str, scale_factor: int,
                     slm_width: int, slm_height: int) -> dict[str, object]:
    encoded = encode_amplitude_uint8(tensor) if value_type == "amplitude" else encode_phase_uint8(tensor)
    scaled = F.interpolate(encoded.float()[None, None], scale_factor=scale_factor, mode="nearest")[0, 0].to(torch.uint8)
    height, width = scaled.shape
    if height > slm_height or width > slm_width:
        raise ValueError(f"Scaled plane {width}x{height} exceeds SLM {slm_width}x{slm_height}")
    canvas = torch.zeros(slm_height, slm_width, dtype=torch.uint8)
    y0, x0 = (slm_height - height) // 2, (slm_width - width) // 2
    canvas[y0:y0 + height, x0:x0 + width] = scaled
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy(), mode="L").save(path, format="BMP")
    return {
        "path": str(path), "value_type": value_type, "source_shape_hw": list(encoded.shape),
        "scaled_shape_hw": list(scaled.shape), "slm_size_wh": [slm_width, slm_height],
        "active_bounds_xyxy": [x0, y0, x0 + width, y0 + height], "scale_factor": scale_factor,
    }

