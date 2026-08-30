"""Meadowlark Blink PCIe SLM adapter for the 1024x1024 amplitude device.

The uploaded vendor SDK uses the board-indexed ``Blink_C_wrapper`` API.  It is
not API-compatible with the separate HDMI phase-SLM example.  This adapter
performs no resize, flip, normalization, wavefront correction, or LUT fitting.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..devices import DeviceError, SLMDriver
except ImportError:  # direct execution from inside hardware_sdk/
    from devices import DeviceError, SLMDriver


def load_meadowlark_frame(
    path: Path,
    expected_resolution: tuple[int, int],
) -> np.ndarray:
    """Load one exact-size 8-bit grayscale BMP without implicit conversion."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".bmp":
        raise DeviceError(f"Meadowlark playback requires BMP files: {path.name}")
    with Image.open(path) as image:
        if image.format != "BMP":
            raise DeviceError(f"{path.name} does not contain BMP data")
        if image.mode != "L":
            raise DeviceError(
                f"{path.name} mode={image.mode}; require native 8-bit grayscale L"
            )
        if tuple(image.size) != tuple(expected_resolution):
            raise DeviceError(
                f"{path.name} size={image.size}; Meadowlark requires "
                f"{expected_resolution} with no implicit scaling"
            )
        frame = np.asarray(image, dtype=np.uint8).copy()
    return np.ascontiguousarray(frame)


class MeadowlarkPCIeSLM(SLMDriver):
    """Thin owner of one board in the Meadowlark Blink PCIe controller."""

    REQUIRED_DLLS = ("Blink_C_wrapper.dll", "Blink_SDK.dll")

    def __init__(
        self,
        sdk_path: Path,
        lut_file: Path,
        *,
        board_number: int = 1,
        expected_resolution: tuple[int, int] | None = (1024, 1024),
        expected_bit_depth: int | None = 8,
        expected_pixel_pitch_um: float | None = 17.0,
        timeout_ms: int = 5000,
        wait_for_trigger: bool = False,
        flip_immediate: bool = False,
        output_pulse: bool = False,
        preload: bool = False,
        blank_on_close: bool = True,
        expected_lut_sha256: str | None = None,
    ) -> None:
        self.sdk_path = Path(sdk_path).expanduser().resolve()
        self.sdk_dir = (
            self.sdk_path / "SDK"
            if (self.sdk_path / "SDK").is_dir()
            else self.sdk_path
        )
        self.lut_file = Path(lut_file).expanduser().resolve()
        self.board_number = int(board_number)
        self.expected_resolution = expected_resolution
        self.expected_bit_depth = expected_bit_depth
        self.expected_pixel_pitch_um = expected_pixel_pitch_um
        self.timeout_ms = int(timeout_ms)
        self.wait_for_trigger = bool(wait_for_trigger)
        self.flip_immediate = bool(flip_immediate)
        self.output_pulse = bool(output_pulse)
        self.preload = bool(preload)
        self.blank_on_close = bool(blank_on_close)
        self.expected_lut_sha256 = (
            None
            if expected_lut_sha256 is None
            else str(expected_lut_sha256).strip().lower()
        )
        self._library: Any = None
        self._sdk_created = False
        self._dll_directories: list[Any] = []
        self._frames: dict[Path, np.ndarray] = {}
        self.width = 0
        self.height = 0
        self.depth = 0
        self.board_count = 0
        self.serial_number: int | None = None
        self.pixel_pitch_um: float | None = None
        self.temperature_c: float | None = None

    def validate_runtime(self) -> None:
        if not sys.platform.startswith("win"):
            raise DeviceError("Meadowlark Blink PCIe SDK is Windows-only")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise DeviceError("The supplied Meadowlark PCIe SDK requires 64-bit Python")
        missing = [
            str(self.sdk_dir / name)
            for name in self.REQUIRED_DLLS
            if not (self.sdk_dir / name).is_file()
        ]
        if missing:
            raise DeviceError(f"Meadowlark PCIe SDK is incomplete; missing: {missing}")
        if not self.lut_file.is_file():
            raise DeviceError(
                f"Meadowlark LUT is missing: {self.lut_file}. Select the calibrated "
                "LUT for this SLM/temperature; do not substitute image scaling."
            )
        self.lut_sha256 = hashlib.sha256(self.lut_file.read_bytes()).hexdigest()
        if (
            self.expected_lut_sha256 is not None
            and self.lut_sha256 != self.expected_lut_sha256
        ):
            raise DeviceError(
                "Meadowlark LUT SHA256 mismatch; refusing to open hardware. "
                f"file={self.lut_file}, actual={self.lut_sha256}, "
                f"expected={self.expected_lut_sha256}"
            )
        if self.board_number <= 0:
            raise DeviceError("Meadowlark board_number is one-based and must be positive")
        if self.timeout_ms <= 0:
            raise DeviceError("Meadowlark timeout_ms must be positive")
        dll_handles: list[Any] = []
        try:
            if hasattr(os, "add_dll_directory"):
                dll_handles.append(os.add_dll_directory(str(self.sdk_dir)))
            library = ctypes.CDLL(str(self.sdk_dir / "Blink_C_wrapper.dll"))
            self._configure_api(library)
            del library
        except (OSError, AttributeError) as exc:
            raise DeviceError(
                "Could not load the Meadowlark Blink PCIe wrapper and its "
                "dependencies. Install the vendor Blink Plus PCIe runtime/driver "
                f"and keep its dependent DLLs available: {exc}"
            ) from exc
        finally:
            for handle in dll_handles:
                try:
                    handle.close()
                except Exception:
                    pass

    @staticmethod
    def _configure_api(library: Any) -> None:
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        library.Create_SDK.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.Create_SDK.restype = None
        library.Delete_SDK.argtypes = []
        library.Delete_SDK.restype = None
        library.Get_last_error_message.argtypes = []
        library.Get_last_error_message.restype = ctypes.c_char_p
        for name in ("Get_image_width", "Get_image_height", "Get_image_depth"):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_int
        library.Get_pixel_pitch.argtypes = [ctypes.c_int]
        library.Get_pixel_pitch.restype = ctypes.c_double
        library.Read_SLM_temperature.argtypes = [ctypes.c_int]
        library.Read_SLM_temperature.restype = ctypes.c_double
        library.Read_Serial_Number.argtypes = [ctypes.c_int]
        library.Read_Serial_Number.restype = ctypes.c_int
        for name in ("SetWaitForTrigger", "SetFlipImmediate", "SetOutputPulse"):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_int, ctypes.c_bool]
            function.restype = ctypes.c_int
        library.Load_LUT_file.argtypes = [ctypes.c_int, ctypes.c_char_p]
        library.Load_LUT_file.restype = ctypes.c_int
        library.Write_image.argtypes = [
            ctypes.c_int,
            byte_pointer,
            ctypes.c_uint,
        ]
        library.Write_image.restype = ctypes.c_int
        library.ImageWriteComplete.argtypes = [ctypes.c_int, ctypes.c_uint]
        library.ImageWriteComplete.restype = ctypes.c_int

    def _last_error(self) -> str:
        if self._library is None:
            return ""
        try:
            value = self._library.Get_last_error_message()
            return "" if not value else value.decode("utf-8", "replace")
        except Exception:
            return ""

    def _check(self, result: Any, operation: str) -> None:
        if int(result) != 1:
            detail = self._last_error()
            suffix = f": {detail}" if detail else ""
            raise DeviceError(
                f"Meadowlark board {self.board_number} {operation} failed "
                f"with code {result}{suffix}"
            )

    def open(self) -> None:
        self.validate_runtime()
        if self._library is not None:
            raise DeviceError("Meadowlark PCIe SLM is already open")
        if hasattr(os, "add_dll_directory"):
            self._dll_directories.append(os.add_dll_directory(str(self.sdk_dir)))
        wrapper = self.sdk_dir / "Blink_C_wrapper.dll"
        try:
            self._library = ctypes.CDLL(str(wrapper))
            self._configure_api(self._library)
            count = ctypes.c_uint(0)
            constructed = ctypes.c_int(0)
            self._library.Create_SDK(ctypes.byref(count), ctypes.byref(constructed))
            self._sdk_created = True
            self.board_count = int(count.value)
            if int(constructed.value) != 1:
                raise DeviceError(
                    "Meadowlark Create_SDK failed"
                    + (f": {self._last_error()}" if self._last_error() else "")
                )
            if not 1 <= self.board_number <= self.board_count:
                raise DeviceError(
                    f"Meadowlark found {self.board_count} board(s), but "
                    f"board_number={self.board_number}"
                )
            self.width = int(self._library.Get_image_width(self.board_number))
            self.height = int(self._library.Get_image_height(self.board_number))
            self.depth = int(self._library.Get_image_depth(self.board_number))
            actual = (self.width, self.height)
            if self.expected_resolution is not None and actual != tuple(
                self.expected_resolution
            ):
                raise DeviceError(
                    f"Meadowlark resolution={actual}, expected={self.expected_resolution}; "
                    "implicit fit/tile scaling is forbidden"
                )
            if self.expected_bit_depth is not None and self.depth != int(
                self.expected_bit_depth
            ):
                raise DeviceError(
                    f"Meadowlark input depth={self.depth}, "
                    f"expected={self.expected_bit_depth}"
                )
            self._check(
                self._library.SetWaitForTrigger(
                    self.board_number, self.wait_for_trigger
                ),
                "SetWaitForTrigger",
            )
            self._check(
                self._library.SetFlipImmediate(
                    self.board_number, self.flip_immediate
                ),
                "SetFlipImmediate",
            )
            self._check(
                self._library.SetOutputPulse(self.board_number, self.output_pulse),
                "SetOutputPulse",
            )
            self._check(
                self._library.Load_LUT_file(
                    self.board_number, os.fsencode(self.lut_file)
                ),
                f"load LUT {self.lut_file.name}",
            )
            self.serial_number = int(
                self._library.Read_Serial_Number(self.board_number)
            )
            self.pixel_pitch_um = float(
                self._library.Get_pixel_pitch(self.board_number)
            )
            if (
                self.expected_pixel_pitch_um is not None
                and abs(self.pixel_pitch_um - float(self.expected_pixel_pitch_um))
                > 0.25
            ):
                raise DeviceError(
                    f"Meadowlark pixel pitch={self.pixel_pitch_um:.4g} um, "
                    f"expected={self.expected_pixel_pitch_um:.4g} um; check that "
                    "the configured board is the intended amplitude SLM"
                )
            self.temperature_c = float(
                self._library.Read_SLM_temperature(self.board_number)
            )
        except Exception:
            # Cleanup must not hide the actionable Create/LUT/device error.
            try:
                self.close()
            except Exception:
                pass
            raise

    def _expected_size(self) -> tuple[int, int]:
        if self.width > 0 and self.height > 0:
            return self.width, self.height
        if self.expected_resolution is None:
            raise DeviceError(
                "expected_resolution_wh is required for pre-open file validation"
            )
        return tuple(self.expected_resolution)

    def validate_files(self, paths: list[Path]) -> None:
        expected = self._expected_size()
        for path in paths:
            load_meadowlark_frame(path, expected)

    def preload_files(self, paths: list[Path]) -> None:
        self.validate_files(paths)
        self._frames.clear()
        if self.preload:
            expected = self._expected_size()
            self._frames = {
                Path(path).expanduser().resolve(): load_meadowlark_frame(
                    path, expected
                )
                for path in paths
            }
            print(
                f"[Meadowlark PCIe] cached {len(self._frames)} frame(s) in host memory"
            )

    def display_file(self, path: Path) -> None:
        if self._library is None or not self._sdk_created:
            raise DeviceError("Meadowlark PCIe SLM is not open")
        resolved = Path(path).expanduser().resolve()
        frame = self._frames.get(resolved)
        if frame is None:
            frame = load_meadowlark_frame(resolved, self._expected_size())
        pointer = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self._check(
            self._library.Write_image(
                self.board_number, pointer, self.timeout_ms
            ),
            f"DMA {resolved.name}",
        )
        self._check(
            self._library.ImageWriteComplete(
                self.board_number, self.timeout_ms
            ),
            f"wait for {resolved.name}",
        )
        self.temperature_c = float(
            self._library.Read_SLM_temperature(self.board_number)
        )

    def close(self) -> None:
        cleanup_error: Exception | None = None
        if self._library is not None and self._sdk_created:
            if (
                self.blank_on_close
                and not self.wait_for_trigger
                and self.width > 0
                and self.height > 0
            ):
                try:
                    blank = np.zeros((self.height, self.width), dtype=np.uint8)
                    self._check(
                        self._library.Write_image(
                            self.board_number,
                            blank.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                            self.timeout_ms,
                        ),
                        "write close blank",
                    )
                    self._check(
                        self._library.ImageWriteComplete(
                            self.board_number, self.timeout_ms
                        ),
                        "complete close blank",
                    )
                except Exception as exc:
                    cleanup_error = exc
            try:
                self._library.Delete_SDK()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        self._library = None
        self._sdk_created = False
        self._frames.clear()
        for handle in self._dll_directories:
            try:
                handle.close()
            except Exception:
                pass
        self._dll_directories.clear()
        if cleanup_error is not None:
            raise cleanup_error

    def device_info(self) -> dict[str, Any]:
        return {
            "driver": "meadowlark_pcie",
            "sdk_path": str(self.sdk_path),
            "sdk_dir": str(self.sdk_dir),
            "lut_file": str(self.lut_file),
            "lut_sha256": getattr(self, "lut_sha256", None),
            "expected_lut_sha256": self.expected_lut_sha256,
            "board_number": self.board_number,
            "board_count": self.board_count,
            "serial_number": self.serial_number,
            "resolution": [self.width, self.height] if self.width else None,
            "input_bit_depth": self.depth or self.expected_bit_depth,
            "pixel_pitch_um": self.pixel_pitch_um,
            "expected_pixel_pitch_um": self.expected_pixel_pitch_um,
            "temperature_c": self.temperature_c,
            "wait_for_trigger": self.wait_for_trigger,
            "flip_immediate": self.flip_immediate,
            "output_pulse": self.output_pulse,
            "host_preload": self.preload,
            "blank_on_close": self.blank_on_close,
        }
