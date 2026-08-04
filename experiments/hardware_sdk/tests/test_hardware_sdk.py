from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.hardware_sdk.amplitude_camera_demo import generate_digit_bmps
from experiments.hardware_sdk.acquire_folder import run as run_folder_acquisition
from experiments.hardware_sdk.devices import (
    DeviceError,
    DvpSubprocessCamera,
    HoloeyeSLM,
    _configure_dvp_camera,
    build_camera,
    build_slm,
)
from experiments.hardware_sdk.phase_slm_demo import prepare_phase_frame
from experiments.hardware_sdk.roi_calibration import recommend_roi


def test_digit_bmps_have_stable_order_and_exact_slm_shape(tmp_path: Path) -> None:
    digits = []
    for digit in range(10):
        value = np.zeros((28, 28), dtype=np.uint8)
        value[4:24, 4 + digit % 4 : 20 + digit % 4] = 20 + digit * 20
        digits.append(value)
    paths = generate_digit_bmps(
        digits,
        tmp_path,
        slm_size_wh=(192, 108),
        active_size_wh=(96, 96),
        digit_size_px=70,
    )
    assert [path.name for path in paths] == [f"{i:03d}_digit_{i}.bmp" for i in range(10)]
    assert all(Image.open(path).size == (192, 108) for path in paths)
    assert all(Image.open(path).mode == "L" for path in paths)


def test_phase_frame_applies_flip_and_wfc_modulo_256(tmp_path: Path) -> None:
    source = np.array([[0, 1], [254, 255]], dtype=np.uint8)
    correction = np.array([[2, 2], [2, 2]], dtype=np.uint8)
    source_path = tmp_path / "phase.bmp"
    correction_path = tmp_path / "wfc.bmp"
    Image.fromarray(source).save(source_path)
    Image.fromarray(correction).save(correction_path)
    frame = prepare_phase_frame(
        source_path,
        (2, 2),
        wavefront_correction=correction_path,
        flip_vertical=True,
    )
    assert frame.dtype == np.uint8
    assert frame.tolist() == [[0, 1], [2, 3]]


def test_dvp_explicit_settings_override_loaded_config() -> None:
    class Camera:
        def __init__(self) -> None:
            self.loaded = None
            self.TriggerState = True
            self.AeOperation = None
            self.AntiFlick = None
            self.Exposure = 1.0
            self.AnalogGain = 9.0
            self.ResolutionModeSel = 2
            self.Roi = SimpleNamespace(X=0, Y=0, W=100, H=100)

        def LoadConfig(self, path: str) -> None:
            self.loaded = path
            self.Exposure = 123.0

    module = SimpleNamespace(
        AeOperation=SimpleNamespace(AE_OP_CONTINUOUS=1, AE_OP_OFF=0),
        AntiFlick=SimpleNamespace(
            ANTIFLICK_DISABLE=0,
            ANTIFLICK_50HZ=50,
            ANTIFLICK_60HZ=60,
        ),
    )
    config = Path(__file__)
    camera = Camera()
    _configure_dvp_camera(
        camera,
        module,
        config_file=config,
        auto_exposure=False,
        exposure_us=10000.0,
        analog_gain=1.0,
        anti_flicker_hz=0,
        device_roi_xywh=(10, 20, 80, 70),
        resolution_mode=0,
    )
    assert camera.loaded == str(config)
    assert camera.TriggerState is False
    assert camera.AeOperation == 0
    assert camera.Exposure == pytest.approx(10000.0)
    assert camera.AnalogGain == pytest.approx(1.0)
    assert (camera.Roi.X, camera.Roi.Y, camera.Roi.W, camera.Roi.H) == (10, 20, 80, 70)


def test_shared_factories_preserve_replaceable_vendor_interfaces(tmp_path: Path) -> None:
    slm = build_slm(
        {
            "driver": "holoeye",
            "sdk_path": ".",
            "expected_resolution_wh": [1920, 1080],
            "preload": True,
        },
        tmp_path,
    )
    assert isinstance(slm, HoloeyeSLM)
    camera = build_camera(
        {
            "driver": "dvp_subprocess",
            "sdk_path": ".",
            "python_executable": "python3.5",
            "auto_exposure": False,
            "exposure_us": 10000,
            "analog_gain": 1.0,
            "discard_frames_after_display": 1,
        },
        tmp_path,
    )
    assert isinstance(camera, DvpSubprocessCamera)
    assert camera.exposure_us == pytest.approx(10000.0)
    assert camera.discard_frames_after_display == 1


def test_roi_recommendation_uses_black_reference_and_exact_requested_shape() -> None:
    black = np.full((120, 180), 4.0)
    checker = black.copy()
    checker[30:90, 70:130] += 100.0
    roi, stats = recommend_roi(checker, black, (80, 70), 80.0)
    assert roi == (60, 25, 80, 70)
    assert stats["signal_center_x"] == pytest.approx(99.5)
    assert stats["signal_center_y"] == pytest.approx(59.5)


def test_layer_agnostic_folder_acquisition_preserves_sorted_basenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    input_dir.mkdir()
    for name in ("002_c.bmp", "000_a.bmp", "001_b.bmp"):
        Image.new("L", (8, 6), 255).save(input_dir / name)

    class FakeSlm:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def validate_runtime(self): return None
        def preload_files(self, paths): self.paths = list(paths)
        def display_file(self, path): self.current = path
        def device_info(self): return {"driver": "fake_slm"}

    class FakeCamera:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def validate_runtime(self): return None
        def capture(self, path): np.save(path, np.ones((4, 4), dtype=np.uint8))
        def device_info(self): return {"driver": "fake_camera"}

    monkeypatch.setattr(
        "experiments.hardware_sdk.acquire_folder.build_slm", lambda *_: FakeSlm()
    )
    monkeypatch.setattr(
        "experiments.hardware_sdk.acquire_folder.build_camera", lambda *_: FakeCamera()
    )
    config = tmp_path / "acquisition.json"
    config.write_text(
        __import__("json").dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
                "output_extension": ".npy",
                "settle_delay_ms": 0,
                "confirm_before_start": False,
                "amplitude_slm": {"driver": "manual"},
                "camera": {"driver": "unused"},
            }
        ),
        encoding="utf-8",
    )
    report = run_folder_acquisition(config, assume_yes=True)
    assert report["count"] == 3
    assert [path.name for path in sorted(output_dir.glob("*.npy"))] == [
        "000_a.npy",
        "001_b.npy",
        "002_c.npy",
    ]
    assert (log_dir / "capture_manifest.csv").is_file()


def test_unset_dvp_python_is_rejected_before_device_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DVP_PYTHON_TEST_ONLY", raising=False)
    camera = build_camera(
        {
            "driver": "dvp_subprocess",
            "sdk_path": ".",
            "python_executable": "%DVP_PYTHON_TEST_ONLY%",
        },
        tmp_path,
    )
    with pytest.raises(DeviceError, match="DVP_PYTHON_TEST_ONLY.*not set"):
        camera.validate_runtime()
