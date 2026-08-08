from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.hardware_sdk.demos.amplitude_camera_demo import generate_digit_bmps
from experiments.hardware_sdk.workflows.acquire_folder import run as run_folder_acquisition
from experiments.hardware_sdk.devices import (
    DeviceError,
    DvpSubprocessCamera,
    HoloeyeSLM,
    _configure_dvp_camera,
    build_camera,
    build_slm,
    resize_detector_intensity,
    verify_camera_roi,
)
from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
from experiments.hardware_sdk.workflows.batch_postprocess import run_batch_postprocess
from experiments.hardware_sdk.demos.phase_slm_demo import prepare_phase_frame
from experiments.hardware_sdk.workflows.roi_calibration import (
    exposure_patch,
    gaussian_marker,
    generate_calibration_files,
    rectangle_marker,
    roi_boundary_source_points,
)


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
            "saved_frame_size_wh": [956, 956],
            "saved_frame_resize_mode": "area",
        },
        tmp_path,
    )
    assert isinstance(camera, DvpSubprocessCamera)
    assert camera.exposure_us == pytest.approx(10000.0)
    assert camera.discard_frames_after_display == 1
    assert camera.saved_frame_size_wh == (956, 956)
    assert camera.saved_frame_resize_mode == "area"


def test_factory_builds_tucam_without_changing_legacy_dvp(tmp_path: Path) -> None:
    camera = build_camera(
        {
            "driver": "tucam",
            "sdk_path": ".",
            "exposure_us": 5000,
            "analog_gain": None,
            "anti_flicker_hz": None,
            "saved_frame_size_wh": None,
            "saved_frame_resize_mode": "none",
        },
        tmp_path,
    )
    assert isinstance(camera, TucamCamera)
    assert camera.exposure_us == pytest.approx(5000)
    assert camera.saved_frame_resize_mode == "none"


def test_tucam_roi_requires_four_pixel_alignment() -> None:
    TucamCamera.validate_roi((12, 20, 956, 956))
    with pytest.raises(DeviceError, match="multiples of 4"):
        TucamCamera.validate_roi((10, 20, 956, 956))
    with pytest.raises(DeviceError, match="exceeds the 2048x2048 sensor"):
        TucamCamera.validate_roi((1200, 1200, 956, 956))


def test_manual_roi_is_required_and_must_match_camera_report() -> None:
    with pytest.raises(DeviceError, match="device_roi_xywh is null"):
        verify_camera_roi({"require_device_roi": True, "device_roi_xywh": None})
    config = {
        "require_device_roi": True,
        "device_roi_xywh": [12, 20, 956, 956],
    }
    assert verify_camera_roi(config) == (12, 20, 956, 956)
    assert verify_camera_roi(
        config, {"device_roi_xywh": [12, 20, 956, 956]}
    ) == (12, 20, 956, 956)
    with pytest.raises(DeviceError, match="Camera ROI mismatch"):
        verify_camera_roi(config, {"device_roi_xywh": [16, 20, 956, 956]})


def test_tucam_frame_copy_preserves_uint16_stride_and_header() -> None:
    import ctypes

    height, width, header, stride = 3, 4, 8, 12
    expected = np.array(
        [[1, 2, 3, 4], [100, 200, 300, 400], [4095, 8191, 16383, 65535]],
        dtype=np.uint16,
    )
    payload = bytearray(header + stride * height)
    for row in range(height):
        encoded = expected[row].astype("<u2").tobytes()
        start = header + row * stride
        payload[start : start + len(encoded)] = encoded
    buffer = ctypes.create_string_buffer(bytes(payload))
    frame = SimpleNamespace(
        usWidth=width,
        usHeight=height,
        ucChannels=1,
        ucElemBytes=2,
        uiWidthStep=stride,
        uiImgSize=stride * height,
        usHeader=header,
        pBuffer=ctypes.addressof(buffer),
    )
    actual = TucamCamera.frame_to_array(frame)
    assert actual.dtype == np.uint16
    assert np.array_equal(actual, expected)


def test_saved_frame_nearest_resize_has_exact_shape_and_mapping() -> None:
    source = np.arange(24, dtype=np.uint16).reshape(4, 6)
    resized = resize_detector_intensity(source, (3, 2), "nearest")
    assert resized.shape == (2, 3)
    assert resized.dtype == np.uint16
    assert resized.tolist() == [[0, 2, 4], [12, 14, 16]]


def test_saved_frame_area_resize_preserves_constant_uint16_intensity() -> None:
    source = np.full((2000, 3000), 4095, dtype=np.uint16)
    resized = resize_detector_intensity(source, (956, 956), "area")
    assert resized.shape == (956, 956)
    assert resized.dtype == np.uint16
    assert np.all(resized == 4095)


def test_saved_frame_area_mode_rejects_enlargement() -> None:
    with pytest.raises(DeviceError, match="downsampling only"):
        resize_detector_intensity(np.zeros((10, 10), dtype=np.uint8), (20, 20), "area")


def test_saved_frame_none_mode_preserves_raw_array_without_copy() -> None:
    source = np.arange(35, dtype=np.uint16).reshape(5, 7)
    actual = resize_detector_intensity(source, None, "none")
    assert actual is source
    assert actual.shape == (5, 7)
    assert np.array_equal(actual, source)


def test_build_camera_accepts_explicit_unprocessed_raw_mode(tmp_path: Path) -> None:
    camera = build_camera(
        {
            "driver": "dvp_subprocess",
            "sdk_path": None,
            "python_executable": __import__("sys").executable,
            "saved_frame_size_wh": None,
            "saved_frame_resize_mode": "none",
        },
        tmp_path,
    )
    assert camera.saved_frame_size_wh is None
    assert camera.saved_frame_resize_mode == "none"


def test_gaussian_marker_is_uint8_and_centered() -> None:
    marker = gaussian_marker((200, 160), (91.5, 73.5), 80, 18)
    assert marker.dtype == np.uint8
    assert marker.shape == (160, 200)
    yy, xx = np.indices(marker.shape)
    assert float((marker * xx).sum() / marker.sum()) == pytest.approx(91.5, abs=0.1)
    assert float((marker * yy).sum() / marker.sum()) == pytest.approx(73.5, abs=0.1)


def test_exposure_patch_has_uniform_center_and_cosine_taper() -> None:
    patch = exposure_patch((300, 240), (149.5, 119.5), 200, 160, 128, 16)
    assert patch.shape == (240, 300)
    assert patch.dtype == np.uint8
    assert patch[120, 150] == 200
    assert patch[40, 70] == 0
    assert 0 < patch[48, 75] < 200


def test_generate_calibration_masks_have_exact_8bit_slm_sizes(tmp_path: Path) -> None:
    config = {
        "paths": {"masks_dir": "masks", "results_dir": "results"},
        "amplitude_slm": {"width": 192, "height": 108},
        "phase_slm": {"width": 192, "height": 120},
        "amplitude_roi": {"width": 96, "height": 96, "center_x": 95.5, "center_y": 53.5},
        "calibration": {"marker_size_px": 20, "marker_sigma_px": 5},
        "exposure_calibration": {
            "gray_start": 0, "gray_stop": 2, "gray_step": 1,
            "patch_size_px": 20, "patch_inner_size_px": 16, "edge_taper_px": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    report = generate_calibration_files(config, config_path)
    amp = Image.open(tmp_path / "masks" / "amplitude" / "amplitude_zero.bmp")
    phase = Image.open(tmp_path / "masks" / "phase" / "phase_zero.bmp")
    assert amp.mode == phase.mode == "L"
    assert amp.size == (192, 108)
    assert phase.size == (192, 120)
    assert report["verification_patterns"] == 3
    assert report["automatic_geometry_calibration"] is False
    boundary = roi_boundary_source_points(config)
    assert boundary == [
        (95.5, 53.5),
        (47.5, 5.5),
        (143.5, 5.5),
        (47.5, 101.5),
        (143.5, 101.5),
    ]
    points = Image.open(tmp_path / "masks" / "amplitude" / "verify_roi_5points.bmp")
    rectangles = Image.open(
        tmp_path / "masks" / "amplitude" / "verify_roi_5rectangles.bmp"
    )
    outline = Image.open(tmp_path / "masks" / "amplitude" / "verify_roi_outline.bmp")
    assert points.mode == rectangles.mode == outline.mode == "L"
    assert points.size == rectangles.size == outline.size == (192, 108)
    assert (tmp_path / "masks" / "manifest.csv").is_file()


def test_rectangle_marker_is_filled_and_centered() -> None:
    marker = rectangle_marker((100, 80), (50, 40), 20)
    assert marker.shape == (80, 100)
    assert int((marker > 0).sum()) == 400
    assert marker[30:50, 40:60].min() == 255


def test_batch_postprocess_saves_quantitative_outputs_without_normalization(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    raw = np.arange(100, dtype=np.uint16).reshape(10, 10) + 10
    background = np.full((10, 10), 10, dtype=np.float32)
    np.save(input_dir / "sample.npy", raw)
    np.save(tmp_path / "background.npy", background)
    (tmp_path / "config.yaml").write_text(
        "postprocess:\n"
        "  resize_enabled: false\n"
        "  target_width: 10\n  target_height: 10\n  resize_mode: area\n"
        "  save_npy: true\n  save_tiff: true\n  save_png_preview: false\n"
        "  saturation_fraction_warning: 1.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed"
    summary = run_batch_postprocess(
        tmp_path / "config.yaml", input_dir, tmp_path / "background.npy", output_dir,
    )
    processed = np.load(output_dir / "sample.npy")
    assert processed.dtype == np.float32
    assert processed.shape == (10, 10)
    assert processed.min() == pytest.approx(0.0)
    assert processed.max() == pytest.approx(99.0)
    assert summary["per_image_normalization"] is False
    assert summary["geometry_transform"] == "none"
    assert (output_dir / "sample.tif").is_file()
    assert (output_dir / "processing_manifest.csv").is_file()
    assert (output_dir / "before_after_preview.png").is_file()


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
        "experiments.hardware_sdk.workflows.acquire_folder.build_slm",
        lambda *_: FakeSlm(),
    )
    monkeypatch.setattr(
        "experiments.hardware_sdk.workflows.acquire_folder.build_camera",
        lambda *_: FakeCamera(),
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


def test_dvp_subprocess_converts_raw_uint16_to_lossless_png(tmp_path: Path) -> None:
    expected = np.array([[0, 1, 255], [256, 4095, 65535]], dtype=np.uint16)

    class FakeInput:
        def write(self, line: str) -> None:
            request = __import__("json").loads(line)
            np.save(request["path"], expected)

        def flush(self) -> None:
            return None

    class FakeOutput:
        def readline(self) -> str:
            return '{"ok": true}\n'

    camera = build_camera(
        {
            "driver": "dvp_subprocess",
            "sdk_path": None,
            "python_executable": __import__("sys").executable,
        },
        tmp_path,
    )
    camera._process = SimpleNamespace(stdin=FakeInput(), stdout=FakeOutput())
    output = tmp_path / "frame.png"
    camera.capture(output)
    actual = np.asarray(Image.open(output))
    # Pillow exposes 16-bit PNG as mode I / int32 on some versions, while all
    # original uint16 sample values remain bit-exact.
    assert np.issubdtype(actual.dtype, np.integer)
    assert np.array_equal(actual.astype(np.uint16), expected)
    assert not (tmp_path / ".frame.dvp_raw.npy").exists()


def test_dvp_subprocess_resizes_raw_frame_before_png_save(tmp_path: Path) -> None:
    expected = np.full((1200, 1600), 1234, dtype=np.uint16)

    class FakeInput:
        def write(self, line: str) -> None:
            request = __import__("json").loads(line)
            np.save(request["path"], expected)

        def flush(self) -> None:
            return None

    class FakeOutput:
        def readline(self) -> str:
            return '{"ok": true}\n'

    camera = build_camera(
        {
            "driver": "dvp_subprocess",
            "sdk_path": None,
            "python_executable": __import__("sys").executable,
            "saved_frame_size_wh": [956, 956],
            "saved_frame_resize_mode": "area",
        },
        tmp_path,
    )
    camera._process = SimpleNamespace(stdin=FakeInput(), stdout=FakeOutput())
    output = tmp_path / "frame_956.png"
    camera.capture(output)
    actual = np.asarray(Image.open(output))
    assert actual.shape == (956, 956)
    assert np.all(actual.astype(np.uint16) == 1234)
    assert camera.device_info()["last_capture"] == {
        "source_size_wh": [1600, 1200],
        "saved_size_wh": [956, 956],
        "resize_mode": "area",
        "resized": True,
        "dtype": "uint16",
    }
    assert not (tmp_path / ".frame_956.dvp_raw.npy").exists()
