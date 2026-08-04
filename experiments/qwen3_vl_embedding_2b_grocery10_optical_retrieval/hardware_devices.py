"""Replaceable SLM/camera drivers for the staged optical experiment.

The orchestration code depends only on the small interfaces in this file.  A
new vendor therefore requires a new driver here, not changes to the optical or
electronic model implementation.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class DeviceError(RuntimeError):
    pass


class SLMDriver(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def display_file(self, path: Path) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

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


class HoloeyeSLM(SLMDriver):
    def __init__(self, sdk_path: Path, binary_folder: Path | None = None) -> None:
        self.sdk_path = sdk_path
        self.binary_folder = binary_folder
        self._module: Any = None
        self._slm: Any = None

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
        try:
            self._slm = self._module.SLMInstance(binaryFolder=str(binary_folder))
        except Exception as exc:
            raise DeviceError(f"Could not initialize the HOLOEYE native runtime: {exc}") from exc
        self._check(self._slm.open(), "open")

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

    def display_file(self, path: Path) -> None:
        if self._slm is None:
            raise DeviceError("HOLOEYE SLM is not open")
        if not path.is_file():
            raise FileNotFoundError(path)
        self._check(self._slm.showDataFromFile(str(path)), f"display {path.name}")

    def close(self) -> None:
        if self._slm is not None:
            try:
                self._slm.close()
            finally:
                self._slm = None


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
    ) -> None:
        self.sdk_path = sdk_path
        self.camera_index = int(camera_index)
        self.timeout_ms = int(timeout_ms)
        self.config_file = config_file
        self._module: Any = None
        self._camera: Any = None

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
        self._camera.TriggerState = False
        if self.config_file is not None:
            if not self.config_file.is_file():
                raise DeviceError(f"DVP camera config is missing: {self.config_file}")
            self._camera.LoadConfig(str(self.config_file))
        self._camera.Start()

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
        array = self._frame_to_array(
            self._camera.GetFrame(self.timeout_ms), self._module
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.Stop()
            finally:
                self._camera.Close()
                self._camera = None


class DvpSubprocessCamera(CameraDriver):
    """Keep the legacy DVP SDK in a vendor Python subprocess.

    This is the recommended bridge for the uploaded Python-3.5 Linux module:
    the model stays in Python 3.11 while only acquisition runs in Python 3.5.
    Frames are exchanged as lossless NumPy arrays, never through JPEG.
    """

    def __init__(
        self,
        sdk_path: Path,
        python_executable: str,
        camera_index: int = 0,
        timeout_ms: int = 4000,
        config_file: Path | None = None,
    ) -> None:
        self.sdk_path = sdk_path
        self.python_executable = python_executable
        self.camera_index = int(camera_index)
        self.timeout_ms = int(timeout_ms)
        self.config_file = config_file
        self._process: subprocess.Popen[str] | None = None

    def open(self) -> None:
        worker = Path(__file__).with_name("dvp_capture_worker.py")
        command = [
            self.python_executable,
            str(worker),
            "--sdk-path",
            str(self.sdk_path),
            "--camera-index",
            str(self.camera_index),
            "--timeout-ms",
            str(self.timeout_ms),
        ]
        if self.config_file is not None:
            command += ["--config-file", str(self.config_file)]
        environment = os.environ.copy()
        if sys.platform.startswith("linux"):
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
            )
        except OSError as exc:
            raise DeviceError(
                f"Could not start vendor Python {self.python_executable!r}: {exc}"
            ) from exc
        line = self._process.stdout.readline().strip() if self._process.stdout else ""
        if line != "READY":
            detail = self._process.stderr.read() if self._process.stderr else ""
            self.close()
            raise DeviceError(
                "DVP subprocess did not become ready. Install NumPy and the DVP "
                f"module in {self.python_executable!r}. stdout={line!r} stderr={detail}"
            )

    def capture(self, path: Path) -> None:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise DeviceError("DVP subprocess camera is not open")
        if path.suffix.lower() != ".npy":
            raise DeviceError("dvp_subprocess stores lossless raw frames as .npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._process.stdin.write(json.dumps({"command": "capture", "path": str(path)}) + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline().strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise DeviceError(f"Invalid DVP worker response: {response!r}") from exc
        if not payload.get("ok"):
            raise DeviceError(f"DVP capture failed: {payload.get('error', payload)}")

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


def _resolve_optional(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
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
        )
    raise ValueError(f"Unknown SLM driver {driver!r}; supported: manual, holoeye")


def build_camera(config: dict[str, Any], base: Path) -> CameraDriver:
    driver = str(config.get("driver", "dvp")).lower()
    if driver == "dvp":
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        if sdk_path is None:
            raise ValueError("A DVP camera requires devices.camera.sdk_path")
        return DvpCamera(
            sdk_path=sdk_path,
            camera_index=int(config.get("camera_index", 0)),
            timeout_ms=int(config.get("timeout_ms", 4000)),
            config_file=_resolve_optional(config.get("config_file"), base),
        )
    if driver == "dvp_subprocess":
        sdk_path = _resolve_optional(config.get("sdk_path"), base)
        if sdk_path is None:
            raise ValueError("A DVP subprocess camera requires devices.camera.sdk_path")
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
        return DvpSubprocessCamera(
            sdk_path=sdk_path,
            python_executable=str(python_executable or "python3.5"),
            camera_index=int(config.get("camera_index", 0)),
            timeout_ms=int(config.get("timeout_ms", 4000)),
            config_file=_resolve_optional(config.get("config_file"), base),
        )
    raise ValueError(
        f"Unknown camera driver {driver!r}; supported: dvp, dvp_subprocess"
    )
