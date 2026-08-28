"""Thin adapter for the vendor TUCam Python/ctypes SDK.

The implementation follows the uploaded ``00_init_open.py``,
``05_wait_frame.py``, ``06_roi_mode.py`` and ``25_set_exposure.py`` demos.
No vendor source or DLL is copied into this tracked module.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..devices import (
        CameraDriver,
        DeviceError,
        convert_detector_bit_depth,
        resolve_detector_resize_mode,
        resize_detector_intensity,
    )
except ImportError:  # imported by a direct hardware_sdk script
    from devices import (
        CameraDriver,
        DeviceError,
        convert_detector_bit_depth,
        resolve_detector_resize_mode,
        resize_detector_intensity,
    )


class TucamCamera(CameraDriver):
    """TUCam/Dhyana camera backend preserving raw monochrome sample values."""

    def __init__(
        self,
        sdk_path: Path,
        camera_index: int = 0,
        timeout_ms: int = 4000,
        config_file: Path | None = None,
        auto_exposure: bool | None = False,
        exposure_us: float | None = None,
        analog_gain: float | None = None,
        anti_flicker_hz: int | None = None,
        device_roi_xywh: tuple[int, int, int, int] | None = None,
        resolution_mode: int | None = None,
        warmup_frames: int = 3,
        discard_frames_after_display: int = 1,
        saved_frame_size_wh: tuple[int, int] | None = None,
        saved_frame_resize_mode: str = "none",
        saved_frame_bit_depth: int | None = None,
        saved_frame_input_range: tuple[float, float] | None = None,
    ) -> None:
        self.sdk_path = Path(sdk_path)
        self.camera_index = int(camera_index)
        self.timeout_ms = int(timeout_ms)
        self.config_file = config_file
        self.auto_exposure = auto_exposure
        self.exposure_us = exposure_us
        self.analog_gain = analog_gain
        self.anti_flicker_hz = anti_flicker_hz
        self.device_roi_xywh = device_roi_xywh
        self.resolution_mode = resolution_mode
        self.warmup_frames = int(warmup_frames)
        self.discard_frames_after_display = int(discard_frames_after_display)
        self.saved_frame_size_wh = saved_frame_size_wh
        self.saved_frame_resize_mode = str(saved_frame_resize_mode).lower()
        self.saved_frame_bit_depth = saved_frame_bit_depth
        self.saved_frame_input_range = saved_frame_input_range
        self._module: ModuleType | None = None
        self._init: Any = None
        self._open: Any = None
        self._frame: Any = None
        self._dll_directory: Any = None
        self._api_initialized = False
        self._buffer_allocated = False
        self._capture_started = False
        self._last_capture_info: dict[str, Any] | None = None
        self._info: dict[str, Any] = {}

    def validate_runtime(self) -> None:
        if not sys.platform.startswith("win"):
            raise DeviceError("The uploaded TUCam/Mosaic SDK is Windows-only")
        required = [
            self.sdk_path / "TUCam.py",
            self.sdk_path / "lib" / "x64" / "TUCam.dll",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise DeviceError(f"TUCam SDK is incomplete; missing: {missing}")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise DeviceError("TUCam x64 requires a 64-bit Python interpreter")
        if self.config_file is not None:
            raise DeviceError(
                "TUCam config_file loading is not enabled in this minimal adapter; "
                "set exposure/ROI explicitly in JSON instead"
            )
        if self.anti_flicker_hz not in (None, 0):
            raise DeviceError("TUCam scientific-camera backend does not use anti_flicker_hz")

    @staticmethod
    def validate_roi(roi_xywh: tuple[int, int, int, int]) -> None:
        """Validate the ROI restrictions measured on the current TUCam.

        The SDK accepts left/top/height in four-pixel increments, while the
        active camera quantizes width to eight pixels.  Rejecting a 4-only
        aligned width here avoids a later, much less obvious ROI mismatch.
        """
        left, top, width, height = (int(value) for value in roi_xywh)
        if min(left, top) < 0 or min(width, height) <= 0:
            raise DeviceError("camera.device_roi_xywh contains invalid values")
        alignment = {"left": 4, "top": 4, "width": 8, "height": 4}
        invalid = [
            f"{name} (requires {alignment[name]})"
            for name, value in zip(alignment, (left, top, width, height))
            if value % alignment[name]
        ]
        if invalid:
            raise DeviceError(
                "TUCam ROI alignment is left/top/height=4 px and width=8 px; "
                f"invalid fields: {', '.join(invalid)}"
            )
        if left + width > 2048 or top + height > 2048:
            raise DeviceError(
                "TUCam ROI exceeds the 2048x2048 sensor: "
                f"[left, top, width, height]={list(roi_xywh)}"
            )

    @staticmethod
    def _result_value(result: Any) -> int | None:
        if result is None:
            return None
        value = getattr(result, "value", result)
        try:
            # TUCAM status values use the high bit for errors.  Some ctypes
            # paths expose them as signed int32 (for example 0x80000312 is
            # returned as -2147482862), while the vendor enum stores the same
            # value as an unsigned integer.  Normalize both representations.
            return int(value) & 0xFFFFFFFF
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _configure_vendor_return_types(module: ModuleType) -> None:
        """Return raw signed int32 status codes instead of constructing TUCAMRET.

        The uploaded vendor binding assigns ``TUCAMRET`` (a Python ``Enum``)
        directly as the ctypes ``restype``.  On Windows, error statuses whose
        high bit is set can reach that enum constructor as signed integers.
        Even valid statuses such as ``TUCAMRET_NOT_SUPPORT`` then raise
        ``ValueError`` before this adapter can handle the error.  Keeping the
        vendor file untouched and normalizing its loaded function objects here
        lets ``_check`` process every status consistently.  TUCam loads its DLL
        through ``OleDLL``; using ``c_uint32`` here makes high-bit error codes
        overflow OleDLL's signed C-long result path, so ``c_int32`` is required.
        ``_result_value`` converts that signed result back to its uint32 bit
        pattern before comparison and reporting.
        """
        enum_type = getattr(module, "TUCAMRET", None)
        if enum_type is None:
            raise DeviceError("Loaded TUCam module does not define TUCAMRET")
        for name in dir(module):
            if not name.startswith(("TUCAM_", "TUIMG_")):
                continue
            function = getattr(module, name, None)
            if getattr(function, "restype", None) is enum_type:
                function.restype = ctypes.c_int32

    def _check(self, result: Any, operation: str) -> None:
        assert self._module is not None
        success = int(self._module.TUCAMRET.TUCAMRET_SUCCESS.value)
        actual = self._result_value(result)
        if actual != success:
            detail = repr(result) if actual is None else f"0x{actual & 0xFFFFFFFF:08X}"
            raise DeviceError(f"TUCam {operation} failed: {detail}")

    def _load_vendor_module(self) -> ModuleType:
        module_path = self.sdk_path / "TUCam.py"
        library_dir = self.sdk_path / "lib" / "x64"
        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(library_dir))
        old_cwd = Path.cwd()
        old_path = list(sys.path)
        try:
            # The supplied TUCam.py loads ./lib/x64/TUCam.dll.  Import it from
            # its own SDK root so that this vendor-relative path stays valid.
            os.chdir(self.sdk_path)
            sys.path.insert(0, str(self.sdk_path))
            spec = importlib.util.spec_from_file_location(
                "hardware_sdk_vendor_tucam", module_path
            )
            if spec is None or spec.loader is None:
                raise DeviceError(f"Could not create an import spec for {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._configure_vendor_return_types(module)
            return module
        except Exception as exc:
            raise DeviceError(
                f"Could not load TUCam.py/TUCam.dll from {self.sdk_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path

    def _handle(self) -> Any:
        if self._open is None or not self._open.hIdxTUCam:
            raise DeviceError("TUCam camera is not open")
        return self._open.hIdxTUCam

    def _set_capability(self, identifier: Any, value: int, operation: str) -> None:
        assert self._module is not None
        self._check(
            self._module.TUCAM_Capa_SetValue(self._handle(), identifier.value, int(value)),
            operation,
        )

    def _set_property(self, identifier: Any, value: float, operation: str) -> None:
        assert self._module is not None
        self._check(
            self._module.TUCAM_Prop_SetValue(
                self._handle(), identifier.value, float(value), 0
            ),
            operation,
        )

    def _get_property(self, identifier: Any) -> float | None:
        assert self._module is not None
        value = ctypes.c_double()
        try:
            self._check(
                self._module.TUCAM_Prop_GetValue(
                    self._handle(), identifier.value, ctypes.pointer(value), 0
                ),
                f"read property {identifier.name}",
            )
            return float(value.value)
        except (DeviceError, ValueError, OverflowError, OSError):
            return None

    def _get_capability(self, identifier: Any) -> int | None:
        assert self._module is not None
        value = ctypes.c_int32()
        try:
            self._check(
                self._module.TUCAM_Capa_GetValue(
                    self._handle(), identifier.value, ctypes.pointer(value)
                ),
                f"read capability {identifier.name}",
            )
            return int(value.value)
        except (DeviceError, ValueError, OverflowError, OSError):
            return None

    def _get_model(self) -> str | None:
        assert self._module is not None
        try:
            identifier = self._module.TUCAM_IDINFO.TUIDI_CAMERA_MODEL
            value = self._module.TUCAM_VALUE_INFO(identifier.value, 0, None, 0)
            self._check(
                self._module.TUCAM_Dev_GetInfo(self._handle(), ctypes.pointer(value)),
                "read camera model",
            )
            return None if not value.pText else value.pText.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _get_roi(self) -> list[int] | None:
        assert self._module is not None
        roi = self._module.TUCAM_ROI_ATTR()
        try:
            self._check(
                self._module.TUCAM_Cap_GetROI(self._handle(), ctypes.pointer(roi)),
                "read ROI",
            )
            if not int(roi.bEnable):
                return None
            return [int(roi.nHOffset), int(roi.nVOffset), int(roi.nWidth), int(roi.nHeight)]
        except DeviceError:
            return None

    def open(self) -> None:
        self.validate_runtime()
        try:
            self._module = self._load_vendor_module()
            self._init = self._module.TUCAM_INIT(
                0, str(self.sdk_path).encode("utf-8")
            )
            self._check(
                self._module.TUCAM_Api_Init(
                    ctypes.pointer(self._init), self.timeout_ms
                ),
                "API init",
            )
            self._api_initialized = True
            count = int(self._init.uiCamCount)
            if not 0 <= self.camera_index < count:
                raise DeviceError(
                    f"camera_index={self.camera_index} is outside 0..{count - 1}"
                )
            self._open = self._module.TUCAM_OPEN(self.camera_index, 0)
            self._check(
                self._module.TUCAM_Dev_Open(ctypes.pointer(self._open)), "open camera"
            )
            if not self._open.hIdxTUCam:
                raise DeviceError("TUCam returned a null camera handle")

            if self.resolution_mode is not None:
                self._set_capability(
                    self._module.TUCAM_IDCAPA.TUIDC_RESOLUTION,
                    int(self.resolution_mode), "set resolution mode",
                )
            if self.auto_exposure is not None:
                self._set_capability(
                    self._module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE,
                    int(bool(self.auto_exposure)), "set auto exposure",
                )
            if self.exposure_us is not None:
                if self.exposure_us <= 0:
                    raise DeviceError("camera.exposure_us must be positive")
                # The uploaded 25_set_exposure.py demo defines this property in ms.
                self._set_property(
                    self._module.TUCAM_IDPROP.TUIDP_EXPOSURETM,
                    float(self.exposure_us) / 1000.0,
                    "set exposure time (ms)",
                )
            if self.analog_gain is not None:
                self._set_property(
                    self._module.TUCAM_IDPROP.TUIDP_GLOBALGAIN,
                    float(self.analog_gain), "set global gain",
                )
            if self.device_roi_xywh is not None:
                left, top, width, height = self.device_roi_xywh
                self.validate_roi((left, top, width, height))
                roi = self._module.TUCAM_ROI_ATTR(1, left, top, width, height)
                self._check(
                    self._module.TUCAM_Cap_SetROI(self._handle(), roi), "set ROI"
                )

            self._frame = self._module.TUCAM_FRAME()
            self._frame.pBuffer = 0
            self._frame.ucFormatGet = self._module.TUFRM_FORMATS.TUFRM_FMT_USUAl.value
            self._frame.uiRsdSize = 1
            self._check(
                self._module.TUCAM_Buf_Alloc(
                    self._handle(), ctypes.pointer(self._frame)
                ),
                "allocate frame buffer",
            )
            self._buffer_allocated = True
            self._check(
                self._module.TUCAM_Cap_Start(
                    self._handle(),
                    self._module.TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value,
                ),
                "start sequence capture",
            )
            self._capture_started = True
            for _ in range(self.warmup_frames):
                self._grab_array()
            exposure_ms = self._get_property(
                self._module.TUCAM_IDPROP.TUIDP_EXPOSURETM
            )
            self._info = {
                "camera_model": self._get_model(),
                "camera_count": count,
                "device_roi_xywh": self._get_roi(),
                "Exposure": None if exposure_ms is None else exposure_ms * 1000.0,
                "exposure_unit": "us",
                "auto_exposure_enabled": self._get_capability(
                    self._module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE
                ),
                "analog_gain": self._get_property(
                    self._module.TUCAM_IDPROP.TUIDP_GLOBALGAIN
                ),
                "black_level": self._get_property(
                    self._module.TUCAM_IDPROP.TUIDP_BLACKLEVEL
                ),
                "black_level_high_gain": self._get_property(
                    self._module.TUCAM_IDPROP.TUIDP_BLACKLEVELHG
                ),
                "black_level_low_gain": self._get_property(
                    self._module.TUCAM_IDPROP.TUIDP_BLACKLEVELLG
                ),
                "black_level_correction_enabled": self._get_capability(
                    self._module.TUCAM_IDCAPA.TUIDC_ENABLEBLACKLEVEL
                ),
                "sensor_native_resolution_wh": [2048, 2048],
                "sensor_pixel_pitch_um": 6.5,
            }
        except Exception:
            self.close()
            raise

    @staticmethod
    def frame_to_array(frame: Any) -> np.ndarray:
        """Copy a TUCAM_FRAME into a tightly packed monochrome NumPy array."""
        width, height = int(frame.usWidth), int(frame.usHeight)
        channels, elem_bytes = int(frame.ucChannels), int(frame.ucElemBytes)
        if width <= 0 or height <= 0 or not frame.pBuffer:
            raise DeviceError("TUCam returned an empty frame")
        if channels != 1:
            raise DeviceError(
                f"TUCam returned {channels} channels; configure raw monochrome output"
            )
        if elem_bytes not in (1, 2):
            raise DeviceError(f"Unsupported TUCam element size: {elem_bytes} bytes")
        row_bytes = width * channels * elem_bytes
        stride = int(frame.uiWidthStep) or row_bytes
        image_size = int(frame.uiImgSize)
        minimum_size = stride * height
        if image_size < minimum_size:
            # Some SDK revisions report packed image size while WidthStep is
            # stale. Accept exactly packed data, but reject shorter buffers.
            if image_size < row_bytes * height:
                raise DeviceError(
                    f"TUCam buffer is too small: {image_size} < {row_bytes * height}"
                )
            stride = row_bytes
            minimum_size = row_bytes * height
        pointer = int(frame.pBuffer) + int(frame.usHeader)
        copied = ctypes.string_at(pointer, minimum_size)
        rows = np.frombuffer(copied, dtype=np.uint8).reshape(height, stride)
        packed = np.ascontiguousarray(rows[:, :row_bytes])
        dtype = np.uint8 if elem_bytes == 1 else np.dtype("<u2")
        return packed.view(dtype).reshape(height, width).copy()

    def _grab_array(self) -> np.ndarray:
        if self._module is None or self._frame is None:
            raise DeviceError("TUCam camera is not capturing")
        self._check(
            self._module.TUCAM_Buf_WaitForFrame(
                self._handle(), ctypes.pointer(self._frame), self.timeout_ms
            ),
            "wait for frame",
        )
        return self.frame_to_array(self._frame)

    def capture(self, path: Path) -> None:
        for _ in range(self.discard_frames_after_display):
            self._grab_array()
        array = self._grab_array()
        source_size = [int(array.shape[1]), int(array.shape[0])]
        source_dtype = str(array.dtype)
        resolved_resize_mode = resolve_detector_resize_mode(
            tuple(source_size), self.saved_frame_size_wh, self.saved_frame_resize_mode
        )
        array = resize_detector_intensity(
            array, self.saved_frame_size_wh, resolved_resize_mode
        )
        array = convert_detector_bit_depth(
            array, self.saved_frame_bit_depth, self.saved_frame_input_range
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            np.save(path, array)
        elif suffix in {".png", ".tif", ".tiff"}:
            Image.fromarray(array).save(
                path, format="PNG" if suffix == ".png" else "TIFF"
            )
        else:
            raise DeviceError("TUCam supports lossless .npy/.png/.tif/.tiff output")
        saved_size = [int(array.shape[1]), int(array.shape[0])]
        self._last_capture_info = {
            "source_size_wh": source_size,
            "saved_size_wh": saved_size,
            "resize_mode": resolved_resize_mode,
            "resized": source_size != saved_size,
            "dtype": str(array.dtype),
            "source_dtype": source_dtype,
            "saved_frame_bit_depth": self.saved_frame_bit_depth,
            "saved_frame_input_range": (
                None if self.saved_frame_input_range is None
                else list(self.saved_frame_input_range)
            ),
            "sensor_bit_depth": int(getattr(self._frame, "ucDepth", 0)),
            "camera_frame_index": int(getattr(self._frame, "uiIndex", 0)),
        }

    def close(self) -> None:
        module, opened = self._module, self._open
        handle = None if opened is None else opened.hIdxTUCam
        if module is not None and handle:
            if self._capture_started:
                try:
                    module.TUCAM_Buf_AbortWait(handle)
                except Exception:
                    pass
                try:
                    module.TUCAM_Cap_Stop(handle)
                except Exception:
                    pass
            if self._buffer_allocated:
                try:
                    module.TUCAM_Buf_Release(handle)
                except Exception:
                    pass
            try:
                module.TUCAM_Dev_Close(handle)
            except Exception:
                pass
        if module is not None and self._api_initialized:
            try:
                module.TUCAM_Api_Uninit()
            except Exception:
                pass
        if self._dll_directory is not None:
            try:
                self._dll_directory.close()
            except Exception:
                pass
        self._module = None
        self._init = None
        self._open = None
        self._frame = None
        self._dll_directory = None
        self._api_initialized = False
        self._buffer_allocated = False
        self._capture_started = False

    def device_info(self) -> dict[str, Any]:
        return {
            "driver": "tucam",
            "sdk_path": str(self.sdk_path),
            "saved_frame_size_wh": (
                None if self.saved_frame_size_wh is None
                else list(self.saved_frame_size_wh)
            ),
            "saved_frame_resize_mode": self.saved_frame_resize_mode,
            "saved_frame_bit_depth": self.saved_frame_bit_depth,
            "saved_frame_input_range": (
                None if self.saved_frame_input_range is None
                else list(self.saved_frame_input_range)
            ),
            "last_capture": self._last_capture_info,
            **self._info,
        }
