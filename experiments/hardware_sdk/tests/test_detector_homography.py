from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.hardware_sdk.workflows.acquire_folder import (
    _capture_with_optional_geometry,
    _resolve_detector_geometry,
)
from experiments.hardware_sdk.workflows.detector_homography import (
    build_geometry_contract,
    load_geometry_contract,
    transform_points,
    warp_detector_intensity,
    write_geometry_contract,
)


def _config() -> dict:
    return {
        "device_roi_xywh_full_sensor": [100, 200, 480, 480],
        "source_points_full_sensor_xy": {
            "top_left": [109.5, 207.5],
            "top_right": [568.5, 203.5],
            "bottom_right": [574.5, 670.5],
            "bottom_left": [103.5, 674.5],
        },
        "target_size_wh": [478, 478],
        "orientation": {
            "flip_vertical_after_warp": False,
            "flip_horizontal_after_warp": False,
            "downstream_loader_flip_vertical": False,
            "downstream_loader_flip_horizontal": False,
        },
    }


def test_logical_four_corners_map_to_continuous_478_boundaries() -> None:
    contract = build_geometry_contract(_config())
    source = contract["source"]["logical_points_device_roi_local_xy"]
    points = np.asarray(
        [source[label] for label in ("top_left", "top_right", "bottom_right", "bottom_left")]
    )
    mapped = transform_points(
        points, np.asarray(contract["homography_source_to_destination"])
    )
    assert np.allclose(
        mapped,
        [[-0.5, -0.5], [477.5, -0.5], [477.5, 477.5], [-0.5, 477.5]],
        atol=1.0e-7,
    )
    orientation = contract["orientation_contract"]
    assert contract["orientation_canonicalized"] is True
    assert orientation["rotation_and_mirroring_resolved_by_homography"] is True
    assert orientation["saved_frame_orientation"] == "canonical_model_xy"
    assert orientation["downstream_loader_flip_vertical_required"] is False
    assert orientation["downstream_loader_flip_horizontal_required"] is False


def test_warp_is_single_pass_478_square_and_preserves_constant_uint16() -> None:
    contract = build_geometry_contract(_config())
    source = np.full((480, 480), 12345, dtype=np.uint16)
    actual = warp_detector_intensity(source, contract)
    assert actual.shape == (478, 478)
    assert actual.dtype == np.uint16
    assert np.all(actual == 12345)


def test_logical_corner_labels_canonicalize_a_180_degree_camera_image() -> None:
    config = {
        "device_roi_xywh_full_sensor": [100, 200, 480, 480],
        "source_points_full_sensor_xy": {
            # Logical TL is physically at the camera's bottom-right, etc.
            "top_left": [559.5, 659.5],
            "top_right": [119.5, 659.5],
            "bottom_right": [119.5, 219.5],
            "bottom_left": [559.5, 219.5],
        },
        "target_size_wh": [478, 478],
    }
    contract = build_geometry_contract(config)
    source = np.zeros((480, 480), dtype=np.uint16)
    for (x, y), level in zip(
        ((459, 459), (19, 459), (19, 19), (459, 19)),
        (1000, 2000, 3000, 4000),
    ):
        source[y - 8 : y + 9, x - 8 : x + 9] = level
    actual = warp_detector_intensity(source, contract)
    assert int(actual[0:12, 0:12].max()) == 1000
    assert int(actual[0:12, -12:].max()) == 2000
    assert int(actual[-12:, -12:].max()) == 3000
    assert int(actual[-12:, 0:12].max()) == 4000


def test_logically_labelled_homography_rejects_any_later_flip() -> None:
    config = _config()
    config["orientation"]["downstream_loader_flip_vertical"] = True
    with pytest.raises(ValueError, match="already resolve detector orientation"):
        build_geometry_contract(config)


def test_contract_file_and_payload_sha_are_both_verified(tmp_path: Path) -> None:
    path = tmp_path / "geometry.json"
    report = write_geometry_contract(build_geometry_contract(_config()), path)
    loaded, metadata = load_geometry_contract(
        path, expected_file_sha256=report["file_sha256"]
    )
    assert metadata["payload_sha256"] == loaded["payload_sha256"]
    assert path.with_suffix(".json.sha256").is_file()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["destination"]["coordinate_system"] = "tampered"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        load_geometry_contract(path)


def test_independent_n9_style_points_can_reject_bad_geometry() -> None:
    config = _config()
    base = build_geometry_contract(config)
    inverse = np.asarray(base["homography_destination_to_source"])
    expected = {
        "top_mid": [238.5, -0.5],
        "right_mid": [477.5, 238.5],
        "bottom_mid": [238.5, 477.5],
        "left_mid": [-0.5, 238.5],
        "center": [238.5, 238.5],
    }
    destination = np.asarray(list(expected.values()), dtype=np.float64)
    local = transform_points(destination, inverse)
    full = local + np.asarray([100.0, 200.0])
    labels = list(expected)
    config["validation_points_full_sensor_xy"] = {
        label: full[index].tolist() for index, label in enumerate(labels)
    }
    config["validation_expected_target_xy"] = expected
    contract = build_geometry_contract(config)
    assert contract["independent_validation"]["passed"] is True
    assert contract["independent_validation"]["rms_error_px"] < 1.0e-7

    config["validation_points_full_sensor_xy"]["center"][0] += 10.0
    with pytest.raises(ValueError, match="independent geometry validation failed"):
        build_geometry_contract(config)


def test_acquisition_warps_raw_roi_before_fixed_uint8_conversion(tmp_path: Path) -> None:
    contract = build_geometry_contract(_config())

    class Camera:
        def __init__(self) -> None:
            self.info = None

        def capture(self, path: Path) -> None:
            raw = np.full((480, 480), 32768, dtype=np.uint16)
            np.save(path, raw)
            self.info = {
                "source_size_wh": [480, 480],
                "saved_size_wh": [480, 480],
                "resize_mode": "none",
                "dtype": "uint16",
                "source_dtype": "uint16",
            }

        def device_info(self) -> dict:
            return {"last_capture": self.info}

    output = tmp_path / "sample.png"
    info = _capture_with_optional_geometry(
        Camera(),
        output,
        {"saved_frame_bit_depth": 8, "saved_frame_input_range": [0, 65535]},
        contract,
    )
    with Image.open(output) as image:
        actual = np.asarray(image)
        assert image.mode == "L"
        assert image.size == (478, 478)
    assert set(np.unique(actual)) == {128}
    assert info["detector_geometry_applied"] is True
    assert info["saved_frame_orientation"] == "canonical_model_xy"
    assert info["downstream_loader_flip_required"] is False
    assert not (tmp_path / ".sample.raw_device_roi.npy").exists()


def test_acquisition_geometry_is_sha_pinned_and_disables_legacy_preprocessing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "geometry.json"
    report = write_geometry_contract(build_geometry_contract(_config()), path)
    camera_config = {
        "require_device_roi": True,
        "device_roi_xywh": [100, 200, 480, 480],
        "saved_frame_size_wh": [478, 478],
        "saved_frame_resize_mode": "auto",
        "saved_frame_bit_depth": 8,
        "saved_frame_input_range": [0, 65535],
        "detector_geometry": {
            "enabled": True,
            "contract_file": path.name,
            "expected_file_sha256": report["file_sha256"],
        },
    }
    contract, metadata, raw = _resolve_detector_geometry(camera_config, tmp_path)
    assert contract is not None
    assert metadata["file_sha256"] == report["file_sha256"]
    assert raw["saved_frame_size_wh"] is None
    assert raw["saved_frame_resize_mode"] == "none"
    assert raw["saved_frame_bit_depth"] is None
    assert "detector_geometry" not in raw

    mixed = json.loads(json.dumps(camera_config))
    mixed["detector_geometry"]["downstream_loader_flip_horizontal"] = True
    with pytest.raises(ValueError, match="cannot be mixed"):
        _resolve_detector_geometry(mixed, tmp_path)

    legacy = dict(camera_config)
    legacy["detector_geometry"] = {"enabled": False}
    contract, metadata, raw = _resolve_detector_geometry(legacy, tmp_path)
    assert contract is metadata is None
    assert raw["saved_frame_size_wh"] == [478, 478]
