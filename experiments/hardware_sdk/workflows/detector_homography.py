"""Audited four-corner detector rectification for optical experiments.

The four source points are *logical* top-left, top-right, bottom-right and
bottom-left optical-field vertices expressed in full-sensor coordinates.  A
projective homography maps them directly to the continuous boundaries of the
model detector canvas.  Therefore rotation and mirroring are already resolved
by the point labels; neither this workflow nor a downstream model loader may
flip the saved frame again.

This module deliberately uses NumPy rather than OpenCV.  The laboratory
environment stays lightweight, and the exact interpolation and boundary
conventions remain under repository control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image


POINT_LABELS = ("top_left", "top_right", "bottom_right", "bottom_left")
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_TYPE = "logical_four_corner_detector_homography"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _points_from_mapping(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must map {', '.join(POINT_LABELS)} to [x,y]")
    missing = [label for label in POINT_LABELS if label not in value]
    unexpected = sorted(set(value) - set(POINT_LABELS))
    if missing or unexpected:
        raise ValueError(
            f"{field} labels mismatch: missing={missing}, unexpected={unexpected}"
        )
    points = np.asarray([value[label] for label in POINT_LABELS], dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{field} must contain four finite [x,y] points")
    return points


def _point_mapping(points: np.ndarray) -> dict[str, list[float]]:
    return {
        label: [float(points[index, 0]), float(points[index, 1])]
        for index, label in enumerate(POINT_LABELS)
    }


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _validate_convex_quad(points: np.ndarray, source_size_wh: tuple[int, int]) -> None:
    width, height = source_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("source frame size must be positive")
    if (
        np.any(points[:, 0] < -0.5)
        or np.any(points[:, 0] > width - 0.5)
        or np.any(points[:, 1] < -0.5)
        or np.any(points[:, 1] > height - 0.5)
    ):
        raise ValueError(
            "source points must lie inside the device ROI's continuous pixel bounds"
        )
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if float(lengths.min()) < 2.0:
        raise ValueError("source quadrilateral has a degenerate edge")
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(np.abs(cross) < 1.0e-8) or not (
        np.all(cross > 0.0) or np.all(cross < 0.0)
    ):
        raise ValueError(
            "logical corner order is self-intersecting or the quadrilateral is not convex"
        )
    area = abs(_signed_area(points))
    if area < max(16.0, 1.0e-4 * width * height):
        raise ValueError("source quadrilateral area is too small for stable rectification")


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_distance <= 0.0:
        raise ValueError("cannot normalize coincident points")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def compute_homography(source_xy: np.ndarray, destination_xy: np.ndarray) -> np.ndarray:
    """Return a normalized-DLT homography mapping source to destination."""

    source = np.asarray(source_xy, dtype=np.float64)
    destination = np.asarray(destination_xy, dtype=np.float64)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("homography requires exactly four 2-D correspondences")
    source_n, source_transform = _normalize_points(source)
    destination_n, destination_transform = _normalize_points(destination)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source_n, destination_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, singular_values, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    if singular_values[-2] <= 0.0 or singular_values[0] / singular_values[-2] > 1.0e12:
        raise ValueError("four-point system is numerically degenerate")
    normalized_h = vh[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(destination_transform)
        @ normalized_h
        @ source_transform
    )
    scale = float(homography[2, 2])
    if abs(scale) < 1.0e-12:
        scale = float(np.linalg.norm(homography))
    if abs(scale) < 1.0e-12:
        raise ValueError("homography has zero scale")
    homography /= scale
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1.0e-15:
        raise ValueError("homography is singular or non-finite")
    return homography


def transform_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    denominator = mapped[:, 2]
    if np.any(np.abs(denominator) < 1.0e-12):
        raise ValueError("homography maps a point to infinity")
    return mapped[:, :2] / denominator[:, None]


def _destination_boundary_points(size_wh: tuple[int, int]) -> np.ndarray:
    width, height = size_wh
    if width <= 0 or height <= 0:
        raise ValueError("target_size_wh must be positive")
    # Integer coordinates are pixel centers.  These are the continuous outer
    # boundaries of an image whose centers are 0..width-1 / 0..height-1.
    return np.asarray(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ],
        dtype=np.float64,
    )


def build_geometry_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build and self-hash an immutable detector rectification contract."""

    roi_raw = config.get("device_roi_xywh_full_sensor")
    if not isinstance(roi_raw, Sequence) or isinstance(roi_raw, (str, bytes)) or len(roi_raw) != 4:
        raise ValueError("device_roi_xywh_full_sensor must be [left,top,width,height]")
    left, top, width, height = (int(value) for value in roi_raw)
    if min(left, top) < 0 or min(width, height) <= 0:
        raise ValueError("device_roi_xywh_full_sensor is invalid")
    if any(value % 4 for value in (left, top, width, height)):
        raise ValueError("TUCam device ROI left/top/width/height must be multiples of four")

    full_points = _points_from_mapping(
        config.get("source_points_full_sensor_xy"), "source_points_full_sensor_xy"
    )
    local_points = full_points - np.asarray([left, top], dtype=np.float64)
    source_size = (width, height)
    _validate_convex_quad(local_points, source_size)

    target_raw = config.get("target_size_wh", [478, 478])
    if not isinstance(target_raw, Sequence) or isinstance(target_raw, (str, bytes)) or len(target_raw) != 2:
        raise ValueError("target_size_wh must be [width,height]")
    target_size = (int(target_raw[0]), int(target_raw[1]))
    if target_size != (478, 478):
        raise ValueError("the audited optical detector contract requires 478x478")
    destination_points = _destination_boundary_points(target_size)
    homography = compute_homography(local_points, destination_points)
    inverse = np.linalg.inv(homography)
    reprojection = transform_points(local_points, homography)
    max_corner_error = float(np.max(np.abs(reprojection - destination_points)))
    if max_corner_error > 1.0e-7:
        raise RuntimeError(f"homography corner reprojection error is {max_corner_error}")

    orientation = dict(config.get("orientation", {}))
    forbidden_true = [
        key
        for key in (
            "flip_vertical_after_warp",
            "flip_horizontal_after_warp",
            "downstream_loader_flip_vertical",
            "downstream_loader_flip_horizontal",
        )
        if bool(orientation.get(key, False))
    ]
    if forbidden_true:
        raise ValueError(
            "logical corner labels already resolve detector orientation; flips must "
            f"remain false, but these were true: {forbidden_true}"
        )

    validation_report: dict[str, Any] | None = None
    validation = config.get("validation_points_full_sensor_xy")
    expected = config.get("validation_expected_target_xy")
    if validation is not None or expected is not None:
        if not isinstance(validation, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(
                "validation_points_full_sensor_xy and validation_expected_target_xy "
                "must both be mappings with identical labels"
            )
        if set(validation) != set(expected) or not validation:
            raise ValueError("validation point labels must be identical and non-empty")
        labels = sorted(validation)
        observed_full = np.asarray([validation[label] for label in labels], dtype=np.float64)
        expected_target = np.asarray([expected[label] for label in labels], dtype=np.float64)
        if (
            observed_full.shape[1:] != (2,)
            or expected_target.shape != observed_full.shape
            or not np.isfinite(observed_full).all()
            or not np.isfinite(expected_target).all()
        ):
            raise ValueError("validation points must be finite [x,y] coordinates")
        observed_target = transform_points(
            observed_full - np.asarray([left, top]), homography
        )
        errors = np.linalg.norm(observed_target - expected_target, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        maximum = float(errors.max())
        max_rms = float(config.get("validation_max_rms_error_px", 1.5))
        max_error = float(config.get("validation_max_error_px", 3.0))
        if rms > max_rms or maximum > max_error:
            raise ValueError(
                "independent geometry validation failed: "
                f"rms={rms:.4f}px max={maximum:.4f}px, "
                f"limits={max_rms:.4f}/{max_error:.4f}px"
            )
        validation_report = {
            "labels": labels,
            "observed_target_xy": observed_target.tolist(),
            "expected_target_xy": expected_target.tolist(),
            "euclidean_errors_px": errors.tolist(),
            "rms_error_px": rms,
            "max_error_px": maximum,
            "acceptance_rms_error_px": max_rms,
            "acceptance_max_error_px": max_error,
            "passed": True,
        }

    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "type": CONTRACT_TYPE,
        "source": {
            "coordinate_system": "full sensor: x right, y down, units are CCD pixels",
            "device_roi_xywh_full_sensor": [left, top, width, height],
            "device_roi_frame_size_wh": [width, height],
            "logical_corner_order": list(POINT_LABELS),
            "logical_points_full_sensor_xy": _point_mapping(full_points),
            "logical_points_device_roi_local_xy": _point_mapping(local_points),
        },
        "destination": {
            "size_wh": list(target_size),
            "coordinate_system": "canonical model image: x right, y down",
            "logical_boundary_points_xy": _point_mapping(destination_points),
            "boundary_convention": (
                "integer coordinates are pixel centers; optical ROI vertices map "
                "to continuous boundaries -0.5 and size-0.5"
            ),
        },
        "homography_source_to_destination": homography.tolist(),
        "homography_destination_to_source": inverse.tolist(),
        "interpolation": "bilinear_intensity",
        "orientation_canonicalized": True,
        "orientation_contract": {
            "source_points_are_logically_labeled": True,
            "rotation_and_mirroring_resolved_by_homography": True,
            "saved_frame_orientation": "canonical_model_xy",
            "post_warp_flip_vertical": False,
            "post_warp_flip_horizontal": False,
            "downstream_loader_flip_vertical_required": False,
            "downstream_loader_flip_horizontal_required": False,
            "phase_slm_export_flip_is_independent": True,
        },
        "intensity_processing_contract": {
            "homography_precedes_bit_depth_conversion": True,
            "background_subtraction": False,
            "per_frame_minmax_normalization": False,
            "gamma_or_log_transform": False,
            "additional_resize_after_warp": False,
        },
        "corner_reprojection_max_abs_error_px": max_corner_error,
        "independent_validation": validation_report,
    }
    contract["payload_sha256"] = _payload_sha256(contract)
    return contract


def validate_geometry_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(contract))
    if (
        int(value.get("schema_version", -1)) != CONTRACT_SCHEMA_VERSION
        or value.get("type") != CONTRACT_TYPE
    ):
        raise ValueError("unsupported detector homography contract")
    declared = str(value.get("payload_sha256", "")).lower()
    observed = _payload_sha256(value)
    if len(declared) != 64 or declared != observed:
        raise ValueError("detector homography payload SHA-256 mismatch")
    orientation = value.get("orientation_contract")
    if not bool(value.get("orientation_canonicalized")):
        raise ValueError("detector homography must canonicalize output orientation")
    if not isinstance(orientation, Mapping) or any(
        (
            not bool(orientation.get("source_points_are_logically_labeled")),
            not bool(orientation.get("rotation_and_mirroring_resolved_by_homography")),
            orientation.get("saved_frame_orientation") != "canonical_model_xy",
            bool(orientation.get("post_warp_flip_vertical")),
            bool(orientation.get("post_warp_flip_horizontal")),
            bool(orientation.get("downstream_loader_flip_vertical_required")),
            bool(orientation.get("downstream_loader_flip_horizontal_required")),
        )
    ):
        raise ValueError("detector orientation contract permits an ambiguous/double flip")
    destination = value.get("destination")
    if not isinstance(destination, Mapping) or destination.get("size_wh") != [478, 478]:
        raise ValueError("detector homography destination must be 478x478")
    if value.get("interpolation") != "bilinear_intensity":
        raise ValueError("detector homography interpolation must be bilinear_intensity")
    intensity = value.get("intensity_processing_contract")
    if not isinstance(intensity, Mapping) or any(
        (
            not bool(intensity.get("homography_precedes_bit_depth_conversion")),
            bool(intensity.get("background_subtraction")),
            bool(intensity.get("per_frame_minmax_normalization")),
            bool(intensity.get("gamma_or_log_transform")),
            bool(intensity.get("additional_resize_after_warp")),
        )
    ):
        raise ValueError("detector intensity processing contract is not audited")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("detector homography source metadata is missing")
    source_size = source.get("device_roi_frame_size_wh")
    if not isinstance(source_size, list) or len(source_size) != 2:
        raise ValueError("detector homography source frame size is missing")
    roi = source.get("device_roi_xywh_full_sensor")
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or [int(roi[2]), int(roi[3])] != [int(item) for item in source_size]
    ):
        raise ValueError("detector homography source ROI/size contract is inconsistent")
    if source.get("logical_corner_order") != list(POINT_LABELS):
        raise ValueError("detector homography logical corner order is not canonical")
    forward = np.asarray(value.get("homography_source_to_destination"), dtype=np.float64)
    inverse = np.asarray(value.get("homography_destination_to_source"), dtype=np.float64)
    if forward.shape != (3, 3) or inverse.shape != (3, 3):
        raise ValueError("detector homography matrices must be 3x3")
    if not np.allclose(forward @ inverse, np.eye(3), atol=1.0e-7):
        raise ValueError("detector homography inverse matrix is inconsistent")
    return value


def write_geometry_contract(contract: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    value = validate_geometry_contract(contract)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    file_sha = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{file_sha}  {path.name}\n", encoding="ascii")
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "payload_sha256": value["payload_sha256"],
        "sha256_sidecar": str(sidecar),
    }


def load_geometry_contract(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"detector homography contract is missing: {path}")
    file_sha = sha256_file(path)
    if expected_file_sha256 is not None:
        expected = str(expected_file_sha256).strip().lower()
        if len(expected) != 64 or file_sha != expected:
            raise ValueError(
                "detector homography file SHA-256 mismatch: "
                f"expected={expected}, observed={file_sha}"
            )
    value = validate_geometry_contract(json.loads(path.read_text(encoding="utf-8")))
    return value, {
        "path": str(path),
        "file_sha256": file_sha,
        "payload_sha256": str(value["payload_sha256"]),
    }


def warp_detector_intensity(frame: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    """Warp one raw device-ROI frame once, preserving its numeric dtype."""

    value = validate_geometry_contract(contract)
    source = np.asarray(frame)
    if source.ndim != 2 or source.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(
            "detector homography expects a 2-D uint8/uint16/float intensity frame"
        )
    expected_wh = tuple(int(item) for item in value["source"]["device_roi_frame_size_wh"])
    if (source.shape[1], source.shape[0]) != expected_wh:
        raise ValueError(
            f"raw detector ROI shape={source.shape[::-1]}, expected={expected_wh}"
        )
    target_width, target_height = (
        int(item) for item in value["destination"]["size_wh"]
    )
    inverse = np.asarray(value["homography_destination_to_source"], dtype=np.float64)
    yy, xx = np.indices((target_height, target_width), dtype=np.float64)
    destinations = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=0)
    mapped = inverse @ destinations
    denominator = mapped[2]
    if np.any(np.abs(denominator) < 1.0e-12):
        raise ValueError("homography maps output samples to infinity")
    source_x = mapped[0] / denominator
    source_y = mapped[1] / denominator
    # Pixel centers may legitimately land within half a pixel of the source
    # image's outer boundary.  Edge replication there preserves the measured
    # boundary intensity and avoids inventing a black border.
    if (
        np.any(source_x < -0.500001)
        or np.any(source_x > expected_wh[0] - 0.499999)
        or np.any(source_y < -0.500001)
        or np.any(source_y > expected_wh[1] - 0.499999)
    ):
        raise ValueError("homography samples outside the calibrated device ROI")
    source_x = np.clip(source_x, 0.0, expected_wh[0] - 1.0)
    source_y = np.clip(source_y, 0.0, expected_wh[1] - 1.0)
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, expected_wh[0] - 1)
    y1 = np.minimum(y0 + 1, expected_wh[1] - 1)
    wx = source_x - x0
    wy = source_y - y0
    floating = source.astype(np.float64, copy=False)
    result = (
        floating[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + floating[y0, x1] * wx * (1.0 - wy)
        + floating[y1, x0] * (1.0 - wx) * wy
        + floating[y1, x1] * wx * wy
    ).reshape(target_height, target_width)
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        return np.rint(result).clip(limits.min, limits.max).astype(source.dtype)
    return result.astype(source.dtype)


def _load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = np.asarray(Image.open(path))
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError(f"expected one monochrome frame, got {value.shape}: {path}")
    return value


def _save_frame(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, value)
    elif path.suffix.lower() in {".png", ".tif", ".tiff"}:
        Image.fromarray(value).save(path)
    else:
        raise ValueError("rectified output must be .npy, .png, .tif, or .tiff")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="build an immutable contract from YAML")
    fit.add_argument("--config", required=True)
    fit.add_argument("--output", required=True)
    apply = subparsers.add_parser("apply", help="rectify one raw device-ROI frame")
    apply.add_argument("--contract", required=True)
    apply.add_argument("--expected-file-sha256")
    apply.add_argument("--input", required=True)
    apply.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "fit":
        config_path = Path(args.config).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("detector homography YAML must be a mapping")
        report = write_geometry_contract(build_geometry_contract(raw), args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    contract, metadata = load_geometry_contract(
        args.contract, expected_file_sha256=args.expected_file_sha256
    )
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rectified = warp_detector_intensity(_load_frame(input_path), contract)
    _save_frame(output_path, rectified)
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_shape_hw": list(rectified.shape),
        "output_dtype": str(rectified.dtype),
        "geometry_contract": metadata,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".geometry.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
