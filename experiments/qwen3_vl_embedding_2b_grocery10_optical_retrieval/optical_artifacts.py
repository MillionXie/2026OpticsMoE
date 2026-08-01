from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F


def tensor_stats(value: torch.Tensor) -> dict[str, float | int | list[int]]:
    tensor = value.detach().cpu()
    measured = tensor.abs().float() if tensor.is_complex() else tensor.float()
    return {
        "shape": list(tensor.shape),
        "min": float(measured.min()) if measured.numel() else 0.0,
        "max": float(measured.max()) if measured.numel() else 0.0,
        "mean": float(measured.mean()) if measured.numel() else 0.0,
        "std": float(measured.std(unbiased=False)) if measured.numel() else 0.0,
        "finite": int(torch.isfinite(measured).sum()),
        "numel": int(measured.numel()),
    }


def physical_phase(raw_phase: torch.Tensor, parameterization: str) -> torch.Tensor:
    if parameterization == "sigmoid":
        return 2.0 * math.pi * torch.sigmoid(raw_phase.float())
    if parameterization == "unconstrained":
        return torch.remainder(raw_phase.float(), 2.0 * math.pi)
    raise ValueError(f"Unsupported phase parameterization {parameterization!r}")


def phase_tensors(core: Any) -> dict[str, torch.Tensor]:
    if len(core.expert_layers) != 1:
        raise RuntimeError(
            "The Grocery retrieval hardware exporter expects exactly one expert phase "
            f"plane, got {len(core.expert_layers)}"
        )
    expert_modules = core.expert_layers[0].experts
    raw_experts = torch.stack(
        [module.raw_phase.detach().cpu().float() for module in expert_modules]
    )
    parameterization = str(expert_modules[0].parameterization)
    if any(str(module.parameterization) != parameterization for module in expert_modules):
        raise RuntimeError("All expert masks must use one phase parameterization")
    raw_global = core.global_phase.phase.raw_phase.detach().cpu().float()
    global_parameterization = str(core.global_phase.phase.parameterization)
    physical_experts = physical_phase(raw_experts, parameterization)
    physical_global = physical_phase(raw_global, global_parameterization)
    mosaic = torch.zeros(core.geometry.active_size, core.geometry.active_size)
    active = core.geometry.active_aperture
    for phase, aperture in zip(physical_experts, core.geometry.expert_apertures):
        y0, x0 = aperture.y0 - active.y0, aperture.x0 - active.x0
        mosaic[y0 : y0 + core.geometry.expert_size, x0 : x0 + core.geometry.expert_size] = phase
    return {
        "raw_expert_phase": raw_experts,
        "physical_expert_phase_rad": physical_experts,
        "physical_expert_mosaic_rad": mosaic,
        "raw_global_phase": raw_global,
        "physical_global_phase_rad": physical_global,
    }


def save_phase_snapshot(
    replacement: Any,
    output_dir: Path,
    *,
    epoch: int,
    train_loss: float,
    weight_variant: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stacks = {
        "vision": phase_tensors(replacement.vision_surrogate.core),
        "language": phase_tensors(replacement.language_surrogate.core),
    }
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "vision": stacks["vision"],
        "language": stacks["language"],
    }
    destination = output_dir / "phase_parameters.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    report: dict[str, Any] = {
        "schema_version": 1,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "weight_variant": str(weight_variant),
        "phase_parameters": str(destination),
        "stacks": {},
    }
    for name, values in stacks.items():
        report["stacks"][name] = {
            key: tensor_stats(value) for key, value in values.items()
        }
    (output_dir / "phase_parameters.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def encode_phase_uint8(phase: torch.Tensor) -> torch.Tensor:
    wrapped = torch.remainder(phase.detach().cpu().float(), 2.0 * math.pi)
    return torch.round(wrapped * (255.0 / (2.0 * math.pi))).to(torch.uint8)


def encode_amplitude_uint8(
    amplitude: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    value = amplitude.detach().cpu().float()
    if not torch.isfinite(value).all():
        raise RuntimeError("Amplitude contains NaN or Inf and cannot be exported")
    minimum = float(value.min()) if value.numel() else 0.0
    if minimum < -1.0e-7:
        raise RuntimeError(f"Amplitude must be nonnegative, got min={minimum}")
    value = value.clamp_min(0.0)
    maximum = float(value.max()) if value.numel() else 0.0
    scale = maximum if maximum > 0.0 else 1.0
    # OEO LayerNorm/ReLU amplitudes are not naturally bounded by one.  A hard
    # clamp would destroy their relative structure, so the physical 8-bit SLM
    # encoding uses one documented scalar normalization per plane.
    encoded = torch.round((value / scale).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return encoded, {
        "encoding": "per_plane_max_then_uint8",
        "raw_min": minimum,
        "raw_max": maximum,
        "normalization_divisor": scale,
    }


def export_centered_bmp(
    value: torch.Tensor,
    path: Path,
    *,
    value_type: str,
    scale_factor: int,
    slm_width: int,
    slm_height: int,
) -> dict[str, Any]:
    if value.ndim != 2:
        raise ValueError(f"SLM source plane must be 2-D, got {tuple(value.shape)}")
    if value_type == "phase":
        encoded = encode_phase_uint8(value)
        encoding: dict[str, Any] = {
            "encoding": "phase_mod_2pi_to_uint8",
            "phase_zero_uint8": 0,
            "phase_2pi_exclusive_uint8": 255,
        }
    elif value_type == "amplitude":
        encoded, encoding = encode_amplitude_uint8(value)
    else:
        raise ValueError("value_type must be 'phase' or 'amplitude'")
    factor = int(scale_factor)
    if factor <= 0:
        raise ValueError("scale_factor must be positive")
    scaled = F.interpolate(
        encoded.float()[None, None], scale_factor=factor, mode="nearest"
    )[0, 0].round().to(torch.uint8)
    height, width = map(int, scaled.shape)
    if width > int(slm_width) or height > int(slm_height):
        raise ValueError(
            f"Scaled plane {width}x{height} exceeds SLM {slm_width}x{slm_height}; "
            "do not silently crop or resize the physical pattern"
        )
    x0 = (int(slm_width) - width) // 2
    y0 = (int(slm_height) - height) // 2
    canvas = torch.zeros(int(slm_height), int(slm_width), dtype=torch.uint8)
    canvas[y0 : y0 + height, x0 : x0 + width] = scaled
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy(), mode="L").save(path, format="BMP")
    with Image.open(path) as check:
        if check.mode != "L" or check.size != (int(slm_width), int(slm_height)):
            raise RuntimeError(
                f"Invalid BMP after save: mode={check.mode}, size={check.size}"
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "value_type": value_type,
        "source_shape_hw": list(encoded.shape),
        "scaled_shape_hw": list(scaled.shape),
        "slm_size_wh": [int(slm_width), int(slm_height)],
        "active_bounds_xyxy": [x0, y0, x0 + width, y0 + height],
        "center_padding_lrtb": [
            x0,
            int(slm_width) - x0 - width,
            y0,
            int(slm_height) - y0 - height,
        ],
        "scale_factor": factor,
        "padding_uint8": 0,
        **encoding,
    }
