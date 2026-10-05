"""Stable components shared by at least two LightGenV2 tasks."""

from .optical_contract import (
    DeviceRaster,
    OpticalContract,
    REFERENCE_532NM_17UM_10CM,
)

__all__ = ["DeviceRaster", "OpticalContract", "REFERENCE_532NM_17UM_10CM"]
