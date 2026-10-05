"""Unit-explicit optical geometry shared by LightGenV2 tasks.

This module deliberately contains no Torch dependency.  It defines physical
contracts and device-raster calculations; task-owned propagation and model
graphs remain inside their task directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class OpticalContract:
    """Physical parameters for one free-space propagation profile."""

    wavelength_nm: float
    logical_pixel_pitch_um: float
    propagation_distance_m: float
    active_pixels: int | None = None
    canvas_pixels: int | None = None

    def __post_init__(self) -> None:
        positive = (
            self.wavelength_nm,
            self.logical_pixel_pitch_um,
            self.propagation_distance_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Wavelength, logical pitch and propagation distance must be positive")
        if self.active_pixels is not None and self.active_pixels <= 0:
            raise ValueError("active_pixels must be positive when specified")
        if self.canvas_pixels is not None and self.canvas_pixels <= 0:
            raise ValueError("canvas_pixels must be positive when specified")
        if (
            self.active_pixels is not None
            and self.canvas_pixels is not None
            and self.canvas_pixels < self.active_pixels
        ):
            raise ValueError("canvas_pixels must contain the active aperture")

    @property
    def propagation_distance_cm(self) -> float:
        return 100.0 * self.propagation_distance_m

    @property
    def active_width_mm(self) -> float | None:
        if self.active_pixels is None:
            return None
        return self.active_pixels * self.logical_pixel_pitch_um / 1000.0

    def device_raster(self, device_pixel_pitch_um: float) -> "DeviceRaster":
        if self.active_pixels is None:
            raise ValueError("active_pixels is required for device rasterization")
        if not math.isfinite(device_pixel_pitch_um) or device_pixel_pitch_um <= 0:
            raise ValueError("device_pixel_pitch_um must be positive")
        model_width_um = self.active_pixels * self.logical_pixel_pitch_um
        pixels = round(model_width_um / device_pixel_pitch_um)
        device_width_um = pixels * device_pixel_pitch_um
        return DeviceRaster(
            pixel_pitch_um=float(device_pixel_pitch_um),
            pixels=int(pixels),
            width_mm=device_width_um / 1000.0,
            relative_width_error=(device_width_um - model_width_um) / model_width_um,
        )

    def with_aperture(
        self, active_pixels: int, *, canvas_pixels: int | None = None
    ) -> "OpticalContract":
        """Bind a task-specific aperture without changing propagation physics."""

        return OpticalContract(
            wavelength_nm=self.wavelength_nm,
            logical_pixel_pitch_um=self.logical_pixel_pitch_um,
            propagation_distance_m=self.propagation_distance_m,
            active_pixels=active_pixels,
            canvas_pixels=canvas_pixels,
        )

    def require_same_physics(
        self, reference: "OpticalContract", *, absolute_tolerance: float = 1.0e-12
    ) -> None:
        for name in (
            "wavelength_nm",
            "logical_pixel_pitch_um",
            "propagation_distance_m",
        ):
            actual, expected = float(getattr(self, name)), float(getattr(reference, name))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute_tolerance):
                raise ValueError(f"Optical contract mismatch for {name}: {actual} != {expected}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OpticalContract":
        return cls(
            wavelength_nm=float(values["wavelength_nm"]),
            logical_pixel_pitch_um=float(
                values.get("logical_pixel_pitch_um", values.get("pixel_pitch_um"))
            ),
            propagation_distance_m=float(
                values.get("propagation_distance_m", values.get("distance_m"))
            ),
            active_pixels=(
                None if values.get("active_pixels") is None else int(values["active_pixels"])
            ),
            canvas_pixels=(
                None if values.get("canvas_pixels") is None else int(values["canvas_pixels"])
            ),
        )


@dataclass(frozen=True)
class DeviceRaster:
    """Physical aperture after sampling on a device pixel grid."""

    pixel_pitch_um: float
    pixels: int
    width_mm: float
    relative_width_error: float


REFERENCE_532NM_17UM_10CM = OpticalContract(
    wavelength_nm=532.0,
    logical_pixel_pitch_um=17.0,
    propagation_distance_m=0.10,
)


__all__ = ["DeviceRaster", "OpticalContract", "REFERENCE_532NM_17UM_10CM"]
