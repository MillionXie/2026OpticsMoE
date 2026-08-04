"""Shared, replaceable hardware adapters and bench-test utilities.

Vendor SDK binaries deliberately live beside this package but are ignored by
Git.  Experiment code imports only the stable interfaces from ``devices``.
"""

from .devices import (
    CameraDriver,
    DeviceError,
    DvpCamera,
    DvpSubprocessCamera,
    HoloeyeSLM,
    ManualSLM,
    SLMDriver,
    build_camera,
    build_slm,
)

__all__ = [
    "CameraDriver",
    "DeviceError",
    "DvpCamera",
    "DvpSubprocessCamera",
    "HoloeyeSLM",
    "ManualSLM",
    "SLMDriver",
    "build_camera",
    "build_slm",
]
