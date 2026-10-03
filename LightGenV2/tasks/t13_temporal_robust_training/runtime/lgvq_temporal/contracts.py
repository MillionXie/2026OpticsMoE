"""Unit-explicit physical contracts for the standalone Temporal model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DeviceRaster:
    pixel_pitch_um: float
    pixels: int
    width_mm: float
    relative_width_error: float


@dataclass(frozen=True)
class OpticalContract:
    wavelength_nm: float
    logical_pixel_pitch_um: float
    propagation_distance_m: float
    active_pixels: int
    canvas_pixels: int

    def __post_init__(self) -> None:
        values = (
            self.wavelength_nm,
            self.logical_pixel_pitch_um,
            self.propagation_distance_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Optical physical values must be finite and positive")
        if self.active_pixels <= 0 or self.canvas_pixels < self.active_pixels:
            raise ValueError("Invalid active/canvas pixel geometry")

    @property
    def active_width_mm(self) -> float:
        return self.active_pixels * self.logical_pixel_pitch_um / 1000.0

    def device_raster(self, device_pixel_pitch_um: float) -> DeviceRaster:
        if not math.isfinite(device_pixel_pitch_um) or device_pixel_pitch_um <= 0:
            raise ValueError("device_pixel_pitch_um must be finite and positive")
        model_width_um = self.active_width_mm * 1000.0
        pixels = round(model_width_um / device_pixel_pitch_um)
        device_width_um = pixels * device_pixel_pitch_um
        return DeviceRaster(
            pixel_pitch_um=float(device_pixel_pitch_um),
            pixels=int(pixels),
            width_mm=device_width_um / 1000.0,
            relative_width_error=(device_width_um - model_width_um) / model_width_um,
        )

    def require_settings(self, settings: Any) -> None:
        checks = {
            "wavelength_nm": self.wavelength_nm,
            "pixel_pitch_um": self.logical_pixel_pitch_um,
            "distance_m": self.propagation_distance_m,
        }
        for name, expected in checks.items():
            actual = float(getattr(settings, name))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"Checkpoint/config physical mismatch: {name}={actual}")
        geometry = settings.geometry
        if (
            int(geometry.active_size) != self.active_pixels
            or int(geometry.canvas_size) != self.canvas_pixels
        ):
            raise ValueError("Checkpoint/config active or canvas geometry mismatch")


def load_reference_contract(path: str | Path) -> OpticalContract:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    optics, geometry = raw["optics"], raw["geometry"]
    return OpticalContract(
        wavelength_nm=float(optics["wavelength_nm"]),
        logical_pixel_pitch_um=float(optics["pixel_pitch_um"]),
        propagation_distance_m=float(optics["distance_m"]),
        active_pixels=int(geometry["active_size"]),
        canvas_pixels=int(geometry["canvas_size"]),
    )


__all__ = ["DeviceRaster", "OpticalContract", "load_reference_contract"]
