"""Replaceable SLM/camera drivers for the staged optical experiment.

The orchestration code depends only on the small interfaces in this file.  A
new vendor therefore requires a new driver here, not changes to the optical or
electronic model implementation.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
    "resize_detector_intensity",
]


class DeviceError(RuntimeError):
    pass


def resize_detector_intensity(
    array: np.ndarray,
    size_wh: tuple[int, int] | None,
    mode: str = "area",
) -> np.ndarray:
    """Resize a monochrome detector frame while preserving its integer dtype.

    ``area`` averages source detector pixels and is the recommended compression
    mode. ``nearest`` is provided for exact legacy-coordinate experiments.
    Neither mode performs display normalization, gamma, or contrast stretching.
    """
    if array.ndim != 2 or array.dtype not in (np.uint8, np.uint16):
        raise DeviceError(
            "CCD frame resizing expects a 2-D uint8/uint16 intensity array; "
            f"got shape={array.shape} dtype={array.dtype}"
        )
    mode = str(mode).lower()
    if size_wh is None or mode == "none":
        return array
    width, height = (int(size_wh[0]), int(size_wh[1]))
    if width <= 0 or height <= 0:
        raise DeviceError("camera.saved_frame_size_wh values must be positive")
    source_height, source_width = array.shape
    if (source_width, source_height) == (width, height):
        return array
    if mode == "nearest":
        x = np.floor(np.arange(width, dtype=np.float64) * source_width / width)
        y = np.floor(np.arange(height, dtype=np.float64) * source_height / height)
        x = np.minimum(x.astype(np.int64), source_width - 1)
        y = np.minimum(y.astype(np.int64), source_height - 1)
        return array[np.ix_(y, x)].copy()
    if mode != "area":
        raise DeviceError("camera.saved_frame_resize_mode must be none, area, or nearest")
    if width > source_width or height > source_height:
        raise DeviceError(
            "area mode is for CCD downsampling only; choose nearest to enlarge a frame"
        )
    floating = Image.fromarray(array.astype(np.float32), mode="F")
    resampling = getattr(Image, "Resampling", Image)
    resized = np.asarray(
        floating.resize((width, height), resample=resampling.BOX),
        dtype=np.float32,
    )
    limit = np.iinfo(array.dtype)
    return np.rint(resized).clip(limit.min, limit.max).astype(array.dtype)


class SLMDriver(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def display_file(self, path: Path) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def preload_files(self, paths: list[Path]) -> None:
        """Optionally upload a plane's frames before acquisition."""

    def device_info(self) -> dict[str, Any]:
        return {"driver": type(self).__name__}

    def validate_runtime(self) -> None:
        """Check local dependencies without opening the physical device."""

    def __enter__(self) -> "SLMDriver":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CameraDriver(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def capture(self, path: Path) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def device_info(self) -> dict[str, Any]:
        return {"driver": type(self).__name__}

    def validate_runtime(self) -> None:
        """Check local dependencies without opening the physical device."""

    def __enter__(self) -> "CameraDriver":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ManualSLM(SLMDriver):
    """Marker driver: a human loads the shared phase mask."""

    def open(self) -> None:
        return None

    def display_file(self, path: Path) -> None:
        print(f"[manual SLM] load phase mask: {path}")

    def close(self) -> None:
        return None

    def device_info(self) -> dict[str, Any]:
        return {"driver": "manual", "automatic": False}


class HoloeyeSLM(SLMDriver):
    def __init__(
        self,
        sdk_path: Path,
        binary_folder: Path | None = None,
        expected_resolution: tuple[int, int] | None = None,
        minimum_refresh_hz: float | None = None,
        preload: bool = True,
        wait_until_visible: bool = True,
    ) -> None:
        self.sdk_path = sdk_path
        self.binary_folder = binary_folder
        self.expected_resolution = expected_resolution
        self.minimum_refresh_hz = minimum_refresh_hz
        self.preload = bool(preload)
        self.wait_until_visible = bool(wait_until_visible)
        self._module: Any = None
        self._slm: Any = None
        self._handles: dict[Path, Any] = {}

    def open(self) -> None:
        if not self.sdk_path.is_dir():
            raise DeviceError(f"HOLOEYE SDK directory is missing: {self.sdk_path}")
        sys.path.insert(0, str(self.sdk_path))
        errors: list[str] = []
        for name in ("slmdisplaysdk", "holoeye.slmdisplaysdk"):
            try:
                self._module = importlib.import_module(name)
                break
            except Exception as exc:  # native-library errors need full context
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if self._module is None:
            raise DeviceError(
                "Could not import the HOLOEYE SLM Display SDK. Install the SDK "
                "native runtime on the acquisition computer. Attempts:\n  "
                + "\n  ".join(errors)
            )
        binary_folder = self.binary_folder
        if binary_folder is None:
            environment_root = os.environ.get("HEDS_3_2_PYTHON")
            if environment_root:
                platform = "win64" if sys.platform.startswith("win") and sys.maxsize > 2**32 else ("win32" if sys.platform.startswith("win") else "linux")
                binary_folder = Path(environment_root) / platform
        library_name = "holoeye_slmdisplaysdk.dll" if sys.platform.startswith("win") else "libholoeye_slmdisplaysdk.so"
        if binary_folder is None or not (binary_folder / library_name).is_file():
            raise DeviceError(
                "The HOLOEYE Python wrapper is present, but its native runtime is "
                f"missing. Expected {library_name} under slm.binary_folder (current: "
                f"{binary_folder}). Install the HOLOEYE SLM Display SDK runtime and "
                "set devices.amplitude_slm.binary_folder in the hardware YAML."
            )
        self.binary_folder = binary_folder
        try:
            self._slm = self._module.SLMInstance(binaryFolder=str(binary_folder))
        except Exception as exc:
            raise DeviceError(f"Could not initialize the HOLOEYE native runtime: {exc}") from exc
        if not self._slm.requiresVersion(5):
            raise DeviceError("HOLOEYE runtime API version 5 or newer is required")
        self._check(self._slm.open(), "open")
        actual = (int(self._slm.width_px), int(self._slm.height_px))
        if self.expected_resolution is not None and actual != self.expected_resolution:
            raise DeviceError(
                f"HOLOEYE resolution is {actual}, expected {self.expected_resolution}. "
                "Do not allow the SDK/GPU to rescale experimental BMPs."
            )
        refresh = float(self._slm.refreshrate_hz)
        if self.minimum_refresh_hz is not None and refresh < self.minimum_refresh_hz:
            raise DeviceError(
                f"HOLOEYE refresh rate is {refresh:.3f} Hz, below required "
                f"{self.minimum_refresh_hz:.3f} Hz"
            )

    def validate_runtime(self) -> None:
        if not self.sdk_path.is_dir():
            raise DeviceError(f"HOLOEYE SDK directory is missing: {self.sdk_path}")

    def _check(self, result: Any, operation: str) -> None:
        no_error = getattr(getattr(self._module, "ErrorCode", object), "NoError", 0)
        if result not in (None, 0, no_error):
            detail = ""
            if self._slm is not None and hasattr(self._slm, "errorString"):
                try:
                    detail = f": {self._slm.errorString(result)}"
                except Exception:
                    pass
            raise DeviceError(f"HOLOEYE {operation} failed with code {result}{detail}")

    def _validate_image(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            size = tuple(image.size)
        expected = self.expected_resolution
        if expected is None and self._slm is not None:
            expected = (int(self._slm.width_px), int(self._slm.height_px))
        if expected is not None and size != expected:
            raise DeviceError(
                f"{path.name} has size {size}, but the amplitude SLM requires "
                f"{expected}; implicit fit/tile scaling is forbidden"
            )
        return path

    def _release_handles(self) -> None:
        for handle in self._handles.values():
            try:
                handle.release()
            except Exception:
                pass
        self._handles.clear()

    def preload_files(self, paths: list[Path]) -> None:
        if not self.preload:
            return
        if self._slm is None:
            raise DeviceError("HOLOEYE SLM is not open")
        self._release_handles()
        for raw_path in paths:
            path = self._validate_image(raw_path)
            error, handle = self._slm.loadDataFromFile(str(path))
            self._check(error, f"load {path.name}")
            self._check(
                self._slm.datahandleWaitFor(handle, self._module.State.LoadingFile),
                f"read {path.name}",
            )
            self._check(
                self._slm.datahandleWaitFor(handle, self._module.State.ReadyToRender),
                f"preload {path.name}",
            )
            self._handles[path] = handle
        print(f"[HOLOEYE] preloaded {len(self._handles)} BMP files to GPU")

    def display_file(self, path: Path) -> None:
        if self._slm is None:
            raise DeviceError("HOLOEYE SLM is not open")
        path = self._validate_image(path)
        handle = self._handles.get(path)
        if handle is not None:
            flags = self._module.ShowFlags.PresentAutomatic
            self._check(self._slm.showDatahandle(handle, flags), f"display {path.name}")
            if self.wait_until_visible:
                self._check(
                    self._slm.datahandleWaitFor(handle, self._module.State.Visible),
                    f"wait until visible {path.name}",
                )
        else:
            flags = self._module.ShowFlags.PresentAutomatic
            self._check(
                self._slm.showDataFromFile(str(path), flags),
                f"display {path.name}",
            )

    def close(self) -> None:
        if self._slm is not None:
            try:
                self._release_handles()
                self._slm.close()
            finally:
                self._slm = None

    def device_info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "driver": "holoeye",
            "sdk_path": str(self.sdk_path),
            "binary_folder": None if self.binary_folder is None else str(self.binary_folder),
            "preload": self.preload,
            "wait_until_visible": self.wait_until_visible,
        }
        if self._slm is not None:
            result.update(
                resolution=[int(self._slm.width_px), int(self._slm.height_px)],
                refresh_hz=float(self._slm.refreshrate_hz),
                pixel_pitch_um=float(self._slm.pixelsize_um),
            )
        return result


def _set_dvp_value(camera: Any, name: str, value: Any) -> None:
    if value is None:
        return
    if not hasattr(camera, name):
        raise DeviceError(f"The uploaded DVP SDK does not expose camera.{name}")
    setattr(camera, name, value)


def _configure_dvp_camera(
    camera: Any,
    module: Any,
    *,
    config_file: Path | None,
    auto_exposure: bool | None,
    exposure_us: float | None,
    analog_gain: float | None,
    anti_flicker_hz: int | None,
    device_roi_xywh: tuple[int, int, int, int] | None,
    resolution_mode: int | None,
) -> None:
    # A saved vendor configuration is a base; explicit YAML values always win.
    if config_file is not None:
        if not config_file.is_file():
            raise DeviceError(f"DVP camera config is missing: {config_file}")
        camera.LoadConfig(str(config_file))
    camera.TriggerState = False
    if resolution_mode is not None:
        _set_dvp_value(camera, "ResolutionModeSel", int(resolution_mode))
    if device_roi_xywh is not None:
        x, y, width, height = device_roi_xywh
        roi = camera.Roi
        roi.X, roi.Y, roi.W, roi.H = int(x), int(y), int(width), int(height)
        camera.Roi = roi
    if auto_exposure is not None:
        operation = (
            module.AeOperation.AE_OP_CONTINUOUS
            if auto_exposure
            else module.AeOperation.AE_OP_OFF
        )
        camera.AeOperation = operation
    if anti_flicker_hz is not None:
        values = {
            0: module.AntiFlick.ANTIFLICK_DISABLE,
            50: module.AntiFlick.ANTIFLICK_50HZ,
            60: module.AntiFlick.ANTIFLICK_60HZ,
        }
        if anti_flicker_hz not in values:
            raise DeviceError("anti_flicker_hz must be 0, 50, or 60")
        camera.AntiFlick = values[anti_flicker_hz]
    _set_dvp_value(camera, "Exposure", exposure_us)
    _set_dvp_value(camera, "AnalogGain", analog_gain)


def _safe_dvp_info(camera: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("TriggerState", "Exposure", "AnalogGain", "AeOperation", "AntiFlick", "ResolutionModeSel"):
        try:
            value = getattr(camera, name)
            if isinstance(value, (str, int, float, bool)):
                result[name] = value
            elif hasattr(value, "value"):
                result[name] = value.value
            else:
                result[name] = str(value)
        except Exception:
            result[name] = None
    try:
        roi = camera.Roi
        result["device_roi_xywh"] = [int(roi.X), int(roi.Y), int(roi.W), int(roi.H)]
    except Exception:
        result["device_roi_xywh"] = None
    return result


class DvpCamera(CameraDriver):
    """Adapter for the uploaded DVP Python SDK.

    The uploaded binaries target Python 3.5 (Linux) / 3.6 (Windows).  Loading
    them in Python 3.11 is impossible; the raised error deliberately explains
    this instead of making the experiment appear to hang.
    """

    def __init__(
        self,
        sdk_path: Path,
        camera_index: int = 0,
        timeout_ms: int = 4000,
        config_file: Path | None = None,
        auto_exposure: bool | None = False,
        exposure_us: float | None = None,
        analog_gain: float | None = None,
        anti_flicker_hz: int | None = 0,
        device_roi_xywh: tuple[int, int, int, int] | None = None,
        resolution_mode: int | None = None,
        warmup_frames: int = 3,
        discard_frames_after_display: int = 1,
        saved_frame_size_wh: tuple[int, int] | None = None,
        saved_frame_resize_mode: str = "area",
    ) -> None:
        self.sdk_path = sdk_path
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
        self._last_capture_info: dict[str, Any] | None = None
        self._module: Any = None
        self._camera: Any = None
        self._info: dict[str, Any] = {}

    def open(self) -> None:
        if not self.sdk_path.is_dir():
            raise DeviceError(f"DVP SDK directory is missing: {self.sdk_path}")
        sys.path.insert(0, str(self.sdk_path))
        try:
            self._module = importlib.import_module("dvp")
        except Exception as exc:
            raise DeviceError(
                "Could not import the DVP camera SDK. The uploaded package contains "
                "Python 3.5/3.6 native modules, while this interpreter is "
                f"Python {sys.version_info.major}.{sys.version_info.minor}. Run the "
                "acquisition command in a vendor-compatible environment or replace "
                "the camera driver. Original error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        devices = self._module.Refresh()
        if not devices:
            raise DeviceError("DVP Refresh() found no camera")
        if not 0 <= self.camera_index < len(devices):
            raise DeviceError(
                f"camera_index={self.camera_index} is outside 0..{len(devices)-1}"
            )
        self._camera = self._module.Camera(self.camera_index)
        _configure_dvp_camera(
            self._camera,
            self._module,
            config_file=self.config_file,
            auto_exposure=self.auto_exposure,
            exposure_us=self.exposure_us,
            analog_gain=self.analog_gain,
            anti_flicker_hz=self.anti_flicker_hz,
            device_roi_xywh=self.device_roi_xywh,
            resolution_mode=self.resolution_mode,
        )
        self._camera.Start()
        for _ in range(self.warmup_frames):
            self._camera.GetFrame(self.timeout_ms)
        self._info = _safe_dvp_info(self._camera)

    @staticmethod
    def _frame_to_array(frame_buffer: Any, module: Any) -> np.ndarray:
        frame, buffer = frame_buffer
        dtype = np.uint8 if frame.bits == module.Bits.BITS_8 else np.uint16
        if module.ImageFormat.FORMAT_MONO <= frame.format <= module.ImageFormat.FORMAT_BAYER_RG:
            channels = 1
        elif frame.format in (module.ImageFormat.FORMAT_BGR24, module.ImageFormat.FORMAT_RGB24):
            channels = 3
        elif frame.format in (module.ImageFormat.FORMAT_BGR32, module.ImageFormat.FORMAT_RGB32):
            channels = 4
        else:
            raise DeviceError(f"Unsupported DVP image format: {frame.format}")
        array = np.frombuffer(buffer, dtype=dtype).reshape(frame.iHeight, frame.iWidth, channels)
        if channels == 1:
            return array[..., 0].copy()
        rgb = array[..., :3]
        if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
            raise DeviceError(
                "DVP returned a color frame. Configure the camera/SDK for raw MONO "
                "intensity; silently converting RGB would change detector physics."
            )
        return rgb[..., 0].copy()

    def capture(self, path: Path) -> None:
        if self._camera is None:
            raise DeviceError("DVP camera is not open")
        for _ in range(self.discard_frames_after_display):
            self._camera.GetFrame(self.timeout_ms)
        array = self._frame_to_array(self._camera.GetFrame(self.timeout_ms), self._module)
        source_size = [int(array.shape[1]), int(array.shape[0])]
        array = resize_detector_intensity(
            array, self.saved_frame_size_wh, self.saved_frame_resize_mode
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npy":
            np.save(path, array)
        else:
            Image.fromarray(array).save(path)
        self._last_capture_info = {
            "source_size_wh": source_size,
            "saved_size_wh": [int(array.shape[1]), int(array.shape[0])],
            "resize_mode": self.saved_frame_resize_mode,
            "resized": source_size != [int(array.shape[1]), int(array.shape[0])],
            "dtype": str(array.dtype),
        }

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.Stop()
            finally:
                self._camera.Close()
                self._camera = None

    def device_info(self) -> dict[str, Any]:
        return {
            "driver": "dvp",
            "saved_frame_size_wh": (
                None if self.saved_frame_size_wh is None else list(self.saved_frame_size_wh)
            ),
            "saved_frame_resize_mode": self.saved_frame_resize_mode,
            "last_capture": self._last_capture_info,
            **self._info,
        }


class DvpSubprocessCamera(CameraDriver):
    """Keep the legacy DVP SDK in a vendor Python subprocess.

    This is the recommended bridge for the uploaded Python-3.5 Linux module:
    the model stays in Python 3.11 while only acquisition runs in Python 3.5.
    Frames are exchanged as lossless NumPy arrays, never through JPEG.
    """

    def __init__(
        self,
        sdk_path: Path | None,
        python_executable: str,
        camera_index: int = 0,
        timeout_ms: int = 4000,
        config_file: Path | None = None,
        auto_exposure: bool | None = False,
        exposure_us: float | None = None,
        analog_gain: float | None = None,
        anti_flicker_hz: int | None = 0,
        device_roi_xywh: tuple[int, int, int, int] | None = None,
        resolution_mode: int | None = None,
        warmup_frames: int = 3,
        discard_frames_after_display: int = 1,
        saved_frame_size_wh: tuple[int, int] | None = None,
        saved_frame_resize_mode: str = "area",
    ) -> None:
        self.sdk_path = sdk_path
        self.python_executable = python_executable
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
        self._process: subprocess.Popen[str] | None = None
        self._info: dict[str, Any] = {}
        self._runtime_dir: Path | None = None
        self._last_capture_info: dict[str, Any] | None = None

    def validate_runtime(self) -> None:
        raw = str(self.python_executable).strip()
        unresolved = re.findall(r"%[^%]+%|\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", raw)
        if unresolved:
            variable = unresolved[0].strip("%${}")
            sdk_hint = "CPython 3.6 x64" if "python3.6" in str(self.sdk_path).lower() else "the Python ABI matching the vendor SDK"
            raise DeviceError(
                f"Camera runtime variable {unresolved[0]!r} is not set. The DVP "
                f"module under {self.sdk_path} requires {sdk_hint}. In the same "
                "PowerShell window, set for example:\n"
                f"  $env:{variable} = 'C:\\path\\to\\Python36\\python.exe'\n"
                f"Then verify it with: & $env:{variable} -c \"import sys; print(sys.executable)\""
            )
        candidate = Path(raw).expanduser()
        resolved: str | None
        if candidate.is_absolute() or candidate.parent != Path("."):
            resolved = str(candidate.resolve()) if candidate.is_file() else None
        else:
            resolved = shutil.which(raw)
        if resolved is None:
            raise DeviceError(
                f"DVP vendor Python executable was not found: {raw!r}. The current "
                f"camera SDK is {self.sdk_path}. Set camera.python_executable (or "
                "%DVP_PYTHON%) to the exact vendor-compatible python.exe path."
            )
        self.python_executable = resolved
        probe = subprocess.run(
            [
                self.python_executable,
                "-c",
                "import json,struct,sys; print(json.dumps({'version':[sys.version_info[0],sys.version_info[1]],'bits':struct.calcsize('P')*8,'executable':sys.executable}))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode != 0:
            raise DeviceError(
                f"Could not inspect DVP Python {self.python_executable!r}: {probe.stderr.strip()}"
            )
        python_info = json.loads(probe.stdout.strip())
        if int(python_info["bits"]) != 64:
            raise DeviceError(
                f"DVP camera requires a 64-bit Python, got {python_info['bits']}-bit "
                f"from {self.python_executable}"
            )
        if self.sdk_path is not None:
            required_files = (
                [self.sdk_path / "dvp.pyd", self.sdk_path / "DVPCamera64.dll"]
                if sys.platform.startswith("win")
                else [self.sdk_path / "dvp.so", self.sdk_path / "libdvp.so"]
            )
            missing = [path for path in required_files if not path.is_file()]
            if missing:
                raise DeviceError(
                    "DVP runtime is incomplete. The vendor requires dvp.pyd and "
                    f"DVPCamera64.dll together. Missing: {missing}"
                )

    def _prepare_inplace_runtime(self) -> tuple[Path, Path]:
        source_worker = Path(__file__).parent / "legacy" / "dvp_capture_worker.py"
        if self.sdk_path is None or not sys.platform.startswith("win"):
            return source_worker, source_worker.parent
        # The vendor explicitly requires the demo Python file, dvp.pyd and
        # DVPCamera64.dll to be colocated.  Stage exactly that layout instead
        # of relying on cross-directory DLL search behavior.
        runtime = Path(__file__).with_name("dvp_runtime")
        runtime.mkdir(parents=True, exist_ok=True)
        staged_worker = runtime / "dvp_capture_worker.py"
        shutil.copy2(source_worker, staged_worker)
        for name in ("dvp.pyd", "DVPCamera64.dll"):
            shutil.copy2(self.sdk_path / name, runtime / name)
        self._runtime_dir = runtime
        return staged_worker, runtime

    def open(self) -> None:
        self.validate_runtime()
        worker, runtime_dir = self._prepare_inplace_runtime()
        command = [
            self.python_executable,
            str(worker),
            "--camera-index",
            str(self.camera_index),
            "--timeout-ms",
            str(self.timeout_ms),
        ]
        if self.sdk_path is not None:
            # Keep this argument even for the staged Windows runtime.  Older
            # vendor-compatible workers declared --sdk-path as required; the
            # staged directory is also the correct import/DLL directory.
            command += [
                "--sdk-path",
                str(runtime_dir if sys.platform.startswith("win") else self.sdk_path),
            ]
        if self.config_file is not None:
            command += ["--config-file", str(self.config_file)]
        if self.auto_exposure is not None:
            command += ["--auto-exposure", "on" if self.auto_exposure else "off"]
        if self.exposure_us is not None:
            command += ["--exposure-us", str(self.exposure_us)]
        if self.analog_gain is not None:
            command += ["--analog-gain", str(self.analog_gain)]
        if self.anti_flicker_hz is not None:
            command += ["--anti-flicker-hz", str(self.anti_flicker_hz)]
        if self.device_roi_xywh is not None:
            command += ["--device-roi-xywh", *[str(value) for value in self.device_roi_xywh]]
        if self.resolution_mode is not None:
            command += ["--resolution-mode", str(self.resolution_mode)]
        command += [
            "--warmup-frames",
            str(self.warmup_frames),
            "--discard-frames-after-display",
            str(self.discard_frames_after_display),
        ]
        environment = os.environ.copy()
        if sys.platform.startswith("win"):
            python_root = Path(self.python_executable).resolve().parent
            conda_runtime_paths = [
                runtime_dir,
                python_root,
                python_root / "Library" / "bin",
                python_root / "DLLs",
            ]
            prefix = os.pathsep.join(
                str(path) for path in conda_runtime_paths if path.is_dir()
            )
            environment["PATH"] = prefix + os.pathsep + environment.get("PATH", "")
        if self.sdk_path is not None and sys.platform.startswith("linux"):
            environment["LD_LIBRARY_PATH"] = str(self.sdk_path) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
                cwd=str(runtime_dir),
            )
        except OSError as exc:
            raise DeviceError(
                f"Could not start vendor Python {self.python_executable!r}: {exc}"
            ) from exc
        line = self._process.stdout.readline().strip() if self._process.stdout else ""
        try:
            ready = json.loads(line)
        except json.JSONDecodeError:
            ready = {}
        if not ready.get("ready"):
            detail = self._process.stderr.read() if self._process.stderr else ""
            self.close()
            raise DeviceError(
                "DVP subprocess did not become ready. Install NumPy and the DVP "
                f"module in {self.python_executable!r}. If camera.sdk_path is null, "
                "verify `import dvp` directly in that environment. If it is set, "
                f"verify that dvp.pyd and its vendor DLL are both under {self.sdk_path}. "
                f"stdout={line!r} stderr={detail}"
            )
        self._info = dict(ready.get("device", {}))

    def capture(self, path: Path) -> None:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise DeviceError("DVP subprocess camera is not open")
        suffix = path.suffix.lower()
        if suffix not in {".npy", ".png", ".tif", ".tiff"}:
            raise DeviceError(
                "dvp_subprocess supports lossless .npy, .png, .tif, or .tiff captures"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_postprocess = suffix != ".npy" or self.saved_frame_size_wh is not None
        raw_path = (
            path.parent / f".{path.stem}.dvp_raw.npy"
            if needs_postprocess
            else path
        )
        if raw_path != path and raw_path.exists():
            raw_path.unlink()
        self._process.stdin.write(json.dumps({"command": "capture", "path": str(raw_path)}) + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline().strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise DeviceError(f"Invalid DVP worker response: {response!r}") from exc
        if not payload.get("ok"):
            raise DeviceError(f"DVP capture failed: {payload.get('error', payload)}")
        if needs_postprocess:
            try:
                array = np.load(raw_path, allow_pickle=False)
                if array.ndim != 2 or array.dtype not in (np.uint8, np.uint16):
                    raise DeviceError(
                        f"DVP lossless image export expects 2-D uint8/uint16, got "
                        f"shape={array.shape} dtype={array.dtype}"
                    )
                source_size = [int(array.shape[1]), int(array.shape[0])]
                array = resize_detector_intensity(
                    array, self.saved_frame_size_wh, self.saved_frame_resize_mode
                )
                if suffix == ".npy":
                    np.save(path, array)
                else:
                    image = Image.fromarray(array)
                    image.save(path, format="PNG" if suffix == ".png" else "TIFF")
                saved_size = [int(array.shape[1]), int(array.shape[0])]
                self._last_capture_info = {
                    "source_size_wh": source_size,
                    "saved_size_wh": saved_size,
                    "resize_mode": self.saved_frame_resize_mode,
                    "resized": source_size != saved_size,
                    "dtype": str(array.dtype),
                }
            finally:
                raw_path.unlink(missing_ok=True)
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            self._last_capture_info = {
                "source_size_wh": [int(array.shape[1]), int(array.shape[0])],
                "saved_size_wh": [int(array.shape[1]), int(array.shape[0])],
                "resize_mode": self.saved_frame_resize_mode,
                "resized": False,
                "dtype": str(array.dtype),
            }

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
        except Exception:
            process.kill()
        finally:
            if process.poll() is None:
                process.kill()

    def device_info(self) -> dict[str, Any]:
        return {
            "driver": "dvp_subprocess",
            "python_executable": self.python_executable,
            "runtime_dir": None if self._runtime_dir is None else str(self._runtime_dir),
            "saved_frame_size_wh": (
                None if self.saved_frame_size_wh is None else list(self.saved_frame_size_wh)
            ),
            "saved_frame_resize_mode": self.saved_frame_resize_mode,
            "last_capture": self._last_capture_info,
            **self._info,
        }


def _expand_environment(value: Any) -> str:
    raw = str(value)
    # os.path.expandvars handles $VAR on every platform and %VAR% on Windows.
    # Explicit percent expansion keeps a Windows JSON config testable on Linux.
    raw = re.sub(
        r"%([^%]+)%",
        lambda match: os.environ.get(match.group(1), match.group(0)),
        raw,
    )
    return os.path.expandvars(raw)


def _resolve_optional(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(_expand_environment(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def build_slm(config: dict[str, Any], base: Path) -> SLMDriver:
    driver = str(config.get("driver", "manual")).lower()
    if driver == "manual":
        return ManualSLM()
    if driver == "holoeye":
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        if sdk_path is None:
            raise ValueError("A holoeye SLM requires devices.*.sdk_path")
        return HoloeyeSLM(
            sdk_path,
            binary_folder=_resolve_optional(config.get("binary_folder"), base),
            expected_resolution=(
                tuple(int(value) for value in config["expected_resolution_wh"])
                if config.get("expected_resolution_wh") is not None
                else None
            ),
            minimum_refresh_hz=(
                float(config["minimum_refresh_hz"])
                if config.get("minimum_refresh_hz") is not None
                else None
            ),
            preload=bool(config.get("preload", True)),
            wait_until_visible=bool(config.get("wait_until_visible", True)),
        )
    raise ValueError(f"Unknown SLM driver {driver!r}; supported: manual, holoeye")


def build_camera(config: dict[str, Any], base: Path) -> CameraDriver:
    roi_raw = config.get("device_roi_xywh")
    device_roi = (
        tuple(int(value) for value in roi_raw) if roi_raw is not None else None
    )
    if device_roi is not None and len(device_roi) != 4:
        raise ValueError("camera.device_roi_xywh must contain [x,y,width,height]")
    saved_size_raw = config.get("saved_frame_size_wh")
    saved_frame_size = (
        tuple(int(value) for value in saved_size_raw)
        if saved_size_raw is not None
        else None
    )
    if saved_frame_size is not None and (
        len(saved_frame_size) != 2 or any(value <= 0 for value in saved_frame_size)
    ):
        raise ValueError("camera.saved_frame_size_wh must contain positive [width,height]")
    saved_frame_resize_mode = str(
        config.get("saved_frame_resize_mode", "area")
    ).lower()
    if saved_frame_resize_mode not in {"none", "area", "nearest"}:
        raise ValueError(
            "camera.saved_frame_resize_mode must be none, area, or nearest"
        )
    if saved_frame_resize_mode == "none" and saved_frame_size is not None:
        raise ValueError(
            "camera.saved_frame_size_wh must be null when "
            "saved_frame_resize_mode is none"
        )
    common = dict(
        camera_index=int(config.get("camera_index", 0)),
        timeout_ms=int(config.get("timeout_ms", 4000)),
        config_file=_resolve_optional(config.get("config_file"), base),
        auto_exposure=(
            bool(config["auto_exposure"])
            if config.get("auto_exposure") is not None
            else None
        ),
        exposure_us=(
            float(config["exposure_us"])
            if config.get("exposure_us") is not None
            else None
        ),
        analog_gain=(
            float(config["analog_gain"])
            if config.get("analog_gain") is not None
            else None
        ),
        anti_flicker_hz=(
            int(config["anti_flicker_hz"])
            if config.get("anti_flicker_hz") is not None
            else None
        ),
        device_roi_xywh=device_roi,
        resolution_mode=(
            int(config["resolution_mode"])
            if config.get("resolution_mode") is not None
            else None
        ),
        warmup_frames=int(config.get("warmup_frames", 3)),
        discard_frames_after_display=int(config.get("discard_frames_after_display", 1)),
        saved_frame_size_wh=saved_frame_size,
        saved_frame_resize_mode=saved_frame_resize_mode,
    )
    if common["warmup_frames"] < 0 or common["discard_frames_after_display"] < 0:
        raise ValueError("camera warmup/discard frame counts cannot be negative")
    driver = str(config.get("driver", "dvp")).lower()
    if driver == "dvp":
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        if sdk_path is None:
            raise ValueError("A DVP camera requires devices.camera.sdk_path")
        return DvpCamera(
            sdk_path=sdk_path,
            **common,
        )
    if driver == "dvp_subprocess":
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        python_executable = config.get("python_executable")
        conda_env = config.get("conda_env")
        if python_executable is not None and conda_env is not None:
            raise ValueError("Set only one of camera.python_executable and camera.conda_env")
        if conda_env is not None:
            executable_name = "python.exe" if sys.platform.startswith("win") else "python"
            # .../miniconda/envs/current/bin/python -> .../miniconda
            executable = Path(sys.executable).resolve()
            if len(executable.parents) < 4 or executable.parents[2].name != "envs":
                raise ValueError(
                    "camera.conda_env requires the main interpreter to live under "
                    "<conda-root>/envs/<current>/bin/python; set python_executable explicitly"
                )
            conda_root = executable.parents[3]
            python_executable = str(conda_root / "envs" / str(conda_env) / ("Scripts" if sys.platform.startswith("win") else "bin") / executable_name)
        elif python_executable is not None:
            python_executable = _expand_environment(python_executable)
        return DvpSubprocessCamera(
            sdk_path=sdk_path,
            python_executable=str(python_executable or "python3.5"),
            **common,
        )
    if driver in {"tucam", "mosaic"}:
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        if sdk_path is None:
            raise ValueError("A TUCam camera requires camera.sdk_path")
        try:
            from .drivers.tucam_camera import TucamCamera
        except ImportError:  # direct execution from inside hardware_sdk/
            from drivers.tucam_camera import TucamCamera

        return TucamCamera(sdk_path=sdk_path, **common)
    raise ValueError(
        f"Unknown camera driver {driver!r}; supported: dvp, dvp_subprocess, tucam"
    )
