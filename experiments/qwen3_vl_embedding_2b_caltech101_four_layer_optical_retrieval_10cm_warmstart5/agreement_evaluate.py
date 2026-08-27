"""Strict, model-free evaluation of simulated and measured CCD intensity."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .agreement_common import (
    SCHEMA_VERSION,
    STAGES,
    read_csv,
    read_json,
    require_sha256,
    safe_relative_file,
    sha256_file,
    stage_directory,
    verify_file,
    write_csv,
    write_json,
)


REFERENCE_COLUMNS = {
    "ideal_model_fp32": ("ideal_reference_file", "ideal_reference_sha256"),
    "transport_quantized": (
        "transport_reference_file",
        "transport_reference_sha256",
    ),
}
DOMAINS = ("linear", "network_input")
ORIENTATIONS = ("identity", "flip_vertical", "flip_horizontal", "rotate_180")
METRIC_NAMES = (
    "pcc_full",
    "pcc_signal",
    "ssim",
    "shape_nrmse",
    "energy_ratio_raw",
    "energy_ratio_calibrated",
    "centroid_dx_px",
    "centroid_dy_px",
    "centroid_distance_px",
    "outside_energy_fraction",
    "saturation_fraction",
)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _load_reference(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"intensity"}:
            raise RuntimeError(f"Reference NPZ must contain only 'intensity': {path}")
        value = np.asarray(payload["intensity"], dtype=np.float32)
    if value.shape != expected_shape:
        raise RuntimeError(
            f"Reference {path.name} shape={value.shape}, expected={expected_shape}"
        )
    if not np.isfinite(value).all() or np.any(value < 0):
        raise RuntimeError(f"Reference {path.name} is not finite nonnegative intensity")
    return value


def _load_ccd(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            raw = np.asarray(image)
    if raw.ndim == 3:
        if raw.shape[-1] not in (3, 4):
            raise RuntimeError(f"CCD image has unsupported shape {raw.shape}: {path}")
        raw = raw[..., :3].astype(np.float64).mean(axis=-1)
    if raw.ndim != 2 or raw.shape != expected_shape:
        raise RuntimeError(
            f"Registered CCD {path.name} must be {expected_shape}, got {raw.shape}. "
            "Apply the fixed four-corner transform before agreement evaluation."
        )
    if np.issubdtype(raw.dtype, np.integer):
        saturation_value = float(np.iinfo(raw.dtype).max)
    else:
        saturation_value = float(np.nanmax(raw)) if np.size(raw) else 1.0
    value = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise RuntimeError(f"CCD {path.name} is not finite nonnegative intensity")
    return value, saturation_value


def pcc(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("PCC inputs must have identical shapes")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != x.shape:
            raise ValueError("PCC mask shape mismatch")
        x, y = x[selected], y[selected]
    else:
        x, y = x.reshape(-1), y.reshape(-1)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    if denominator <= np.finfo(np.float64).eps * scale * scale:
        return None
    return float(np.dot(x, y) / denominator)


def signal_mask(theory: np.ndarray, energy_fraction: float = 0.99) -> np.ndarray:
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("signal energy fraction must lie in (0,1]")
    value = np.clip(np.asarray(theory, dtype=np.float64), 0.0, None)
    flat = value.reshape(-1)
    total = float(flat.sum())
    if total <= 1.0e-12:
        return np.zeros_like(value, dtype=bool)
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    count = int(np.searchsorted(cumulative, energy_fraction * total, side="left")) + 1
    selected = np.zeros(flat.size, dtype=bool)
    selected[order[:count]] = True
    return selected.reshape(value.shape)


def shape_nrmse(measured: np.ndarray, theory: np.ndarray) -> float | None:
    left = np.clip(np.asarray(measured, dtype=np.float64), 0.0, None)
    right = np.clip(np.asarray(theory, dtype=np.float64), 0.0, None)
    left_sum, right_sum = float(left.sum()), float(right.sum())
    if left_sum <= 1.0e-12 or right_sum <= 1.0e-12:
        return None
    left /= left_sum
    right /= right_sum
    denominator = float(np.linalg.norm(right))
    return None if denominator <= 1.0e-12 else float(np.linalg.norm(left - right) / denominator)


def _gaussian_filter(value: np.ndarray, sigma: float = 1.5, size: int = 11) -> np.ndarray:
    radius = size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    source = np.asarray(value, dtype=np.float64)
    horizontal_pad = np.pad(source, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = sum(
        float(weight) * horizontal_pad[:, offset : offset + source.shape[1]]
        for offset, weight in enumerate(kernel)
    )
    vertical_pad = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    return sum(
        float(weight) * vertical_pad[offset : offset + source.shape[0], :]
        for offset, weight in enumerate(kernel)
    )


def structural_similarity(left: np.ndarray, right: np.ndarray, data_range: float = 1.0) -> float:
    """Wang-style local SSIM using an 11x11, sigma=1.5 Gaussian window."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("SSIM inputs must be same-shape 2-D arrays")
    if data_range <= 0:
        raise ValueError("SSIM data_range must be positive")
    mu_x = _gaussian_filter(x)
    mu_y = _gaussian_filter(y)
    sigma_x = np.maximum(_gaussian_filter(x * x) - mu_x * mu_x, 0.0)
    sigma_y = np.maximum(_gaussian_filter(y * y) - mu_y * mu_y, 0.0)
    sigma_xy = _gaussian_filter(x * y) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1.0e-18)))


def _adaptive_avg_pool(value: np.ndarray, output_size: int = 224) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    height, width = source.shape
    integral = np.pad(source, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = np.empty((output_size, output_size), dtype=np.float64)
    for out_y in range(output_size):
        y0 = int(math.floor(out_y * height / output_size))
        y1 = int(math.ceil((out_y + 1) * height / output_size))
        x0 = np.floor(np.arange(output_size) * width / output_size).astype(int)
        x1 = np.ceil((np.arange(output_size) + 1) * width / output_size).astype(int)
        sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        result[out_y] = sums / ((y1 - y0) * (x1 - x0))
    return result.astype(np.float32)


def network_input_map(
    value: np.ndarray,
    *,
    relative_clip: float = 12.0,
    log_compression: float = 1.0,
    pool_size: int = 224,
) -> np.ndarray:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    mean = max(float(source.mean()), 1.0e-6)
    mapped = np.log1p(log_compression * np.minimum(source / mean, relative_clip))
    return _adaptive_avg_pool(mapped, pool_size)


def apply_orientation(value: np.ndarray, orientation: str) -> np.ndarray:
    source = np.asarray(value)
    if orientation == "identity":
        return source
    if orientation == "flip_vertical":
        return np.flip(source, axis=0)
    if orientation == "flip_horizontal":
        return np.flip(source, axis=1)
    if orientation == "rotate_180":
        return np.flip(source, axis=(0, 1))
    raise ValueError(f"Unknown orientation candidate {orientation!r}")


def orientation_diagnostics(
    measured: np.ndarray,
    theory: np.ndarray,
    *,
    relative_clip: float,
) -> list[dict[str, Any]]:
    """Score fixed flip candidates without applying any to primary metrics."""

    right = _shape_for_ssim(theory, relative_clip)
    result: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        candidate = apply_orientation(measured, orientation)
        result.append(
            {
                "orientation": orientation,
                "pcc_full": pcc(candidate, theory),
                "ssim": structural_similarity(
                    _shape_for_ssim(candidate, relative_clip), right, data_range=1.0
                ),
            }
        )
    return result


def _shape_for_ssim(value: np.ndarray, relative_clip: float) -> np.ndarray:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    mean = max(float(source.mean()), 1.0e-12)
    return np.minimum(source / mean, relative_clip) / relative_clip


def _centroid(value: np.ndarray) -> tuple[float | None, float | None]:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    total = float(source.sum())
    if total <= 1.0e-12:
        return None, None
    y, x = np.mgrid[: source.shape[0], : source.shape[1]]
    return float((source * x).sum() / total), float((source * y).sum() / total)


def pair_metrics(
    measured: np.ndarray,
    theory: np.ndarray,
    *,
    domain: str,
    energy_gain: float,
    saturation_value: float,
    signal_energy_fraction: float,
    relative_clip: float,
    log_compression: float,
) -> dict[str, float | None]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown metric domain {domain!r}")
    measured_linear = np.asarray(measured, dtype=np.float64)
    theory_linear = np.asarray(theory, dtype=np.float64)
    saturation = (
        float(np.mean(measured_linear >= saturation_value))
        if saturation_value > 0
        else None
    )
    if domain == "network_input":
        left = network_input_map(
            measured_linear,
            relative_clip=relative_clip,
            log_compression=log_compression,
        )
        right = network_input_map(
            theory_linear,
            relative_clip=relative_clip,
            log_compression=log_compression,
        )
        maximum = math.log1p(log_compression * relative_clip)
        left_ssim, right_ssim = left / maximum, right / maximum
        calibrated_ratio = (
            float(left.mean() / right.mean()) if float(right.mean()) > 1.0e-12 else None
        )
    else:
        left, right = measured_linear, theory_linear
        left_ssim = _shape_for_ssim(left, relative_clip)
        right_ssim = _shape_for_ssim(right, relative_clip)
        calibrated_ratio = (
            float(left.mean() / (energy_gain * right.mean()))
            if energy_gain > 0 and float(right.mean()) > 1.0e-12
            else None
        )
    mask = signal_mask(right, signal_energy_fraction)
    outside = (
        float(np.clip(left, 0, None)[~mask].sum() / np.clip(left, 0, None).sum())
        if mask.any() and float(np.clip(left, 0, None).sum()) > 1.0e-12
        else None
    )
    measured_x, measured_y = _centroid(left)
    theory_x, theory_y = _centroid(right)
    dx = (
        measured_x - theory_x
        if measured_x is not None and theory_x is not None
        else None
    )
    dy = (
        measured_y - theory_y
        if measured_y is not None and theory_y is not None
        else None
    )
    return {
        "pcc_full": pcc(left, right),
        "pcc_signal": pcc(left, right, mask=mask) if mask.any() else None,
        "ssim": structural_similarity(left_ssim, right_ssim, data_range=1.0),
        "shape_nrmse": shape_nrmse(left, right),
        "energy_ratio_raw": (
            float(left.mean() / right.mean()) if float(right.mean()) > 1.0e-12 else None
        ),
        "energy_ratio_calibrated": calibrated_ratio,
        "centroid_dx_px": dx,
        "centroid_dy_px": dy,
        "centroid_distance_px": (
            math.hypot(dx, dy) if dx is not None and dy is not None else None
        ),
        "outside_energy_fraction": outside,
        "measured_mean": float(left.mean()),
        "theory_mean": float(right.mean()),
        "saturation_fraction": saturation,
    }


def _amplitude_reconstruction(stage_dir: Path) -> dict[str, dict[str, str]]:
    path = stage_dir / "amplitude_to_play" / "reconstruction_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing amplitude reconstruction manifest: {path}. Run reconstruct_slm first."
        )
    rows = read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("output_bmp", "")
        if not name or name in result:
            raise RuntimeError(f"Invalid duplicate amplitude reconstruction row {name!r}")
        result[name] = row
    return result


def _capture_rows(stage_dir: Path) -> dict[str, dict[str, str]]:
    path = stage_dir / "acquisition_logs" / "capture_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing acquisition manifest: {path}")
    rows = read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("amplitude_bmp", "")
        if not name or name in result:
            raise RuntimeError(f"Invalid duplicate acquisition row {name!r}")
        result[name] = row
    return result


def _manifest_bool(row: dict[str, str], field: str) -> bool:
    raw = str(row.get(field, "")).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"Acquisition manifest field {field!r} is not boolean: {raw!r}")


def _validate_canonical_capture_row(row: dict[str, str], key: str) -> tuple[str, str]:
    if not _manifest_bool(row, "orientation_canonicalized"):
        raise RuntimeError(f"Capture {key} was not canonicalized by detector homography")
    if row.get("saved_frame_orientation") != "canonical_model_xy":
        raise RuntimeError(f"Capture {key} is not in canonical_model_xy orientation")
    if _manifest_bool(row, "downstream_loader_flip_required"):
        raise RuntimeError(f"Capture {key} requests a forbidden downstream/double flip")
    if _manifest_bool(row, "background_subtraction"):
        raise RuntimeError(f"Capture {key} used forbidden background subtraction")
    if _manifest_bool(row, "per_frame_minmax_normalization"):
        raise RuntimeError(f"Capture {key} used forbidden per-frame min-max normalization")
    geometry_file_sha = str(row.get("detector_geometry_file_sha256", "")).strip().lower()
    geometry_payload_sha = str(row.get("detector_geometry_payload_sha256", "")).strip().lower()
    for label, digest in (
        ("detector geometry file", geometry_file_sha),
        ("detector geometry payload", geometry_payload_sha),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"Capture {key} has no valid {label} SHA-256")
    return geometry_file_sha, geometry_payload_sha


def _phase_output(stage_dir: Path, contract: dict[str, Any]) -> tuple[Path, str]:
    manifest_path = safe_relative_file(
        stage_dir,
        contract["phase_reconstruction_manifest"],
        label="phase reconstruction manifest",
    )
    verify_file(
        manifest_path,
        contract["phase_reconstruction_manifest_sha256"],
        label="phase reconstruction manifest",
    )
    rows = read_csv(manifest_path)
    if len(rows) != 1:
        raise RuntimeError("Agreement phase reconstruction manifest must contain one row")
    output = safe_relative_file(
        stage_dir,
        f"phase_to_play/{rows[0]['output_bmp']}",
        label="phase BMP",
    )
    digest = verify_file(output, rows[0]["output_sha256"], label="phase BMP")
    return output, digest


def _bootstrap_interval(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    medians = np.median(
        array[generator.integers(0, len(array), size=(samples, len(array)))], axis=1
    )
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _summary_rows(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["reference_kind"], row["domain"], row["role"])].append(row)
    result: list[dict[str, Any]] = []
    for group_key, group in sorted(grouped.items()):
        stage, reference_kind, domain, role = group_key
        summary: dict[str, Any] = {
            "stage": stage,
            "reference_kind": reference_kind,
            "domain": domain,
            "role": role,
            "independent_probes": len(group),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "") and math.isfinite(float(row[metric]))]
            low, high = _bootstrap_interval(values, bootstrap_samples, seed + len(result) * 31 + len(metric))
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_median"] = float(np.median(values)) if values else None
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if values else None
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if values else None
            summary[f"{metric}_median_ci_low"] = low
            summary[f"{metric}_median_ci_high"] = high
        result.append(summary)
    return result


def _pairing_contract(
    *,
    stage_dir: Path,
    contract: dict[str, Any],
    probe_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, float]], dict[tuple[str, str], np.ndarray]]:
    expected_shape = tuple(int(value) for value in contract["active_shape_hw"])
    reconstruction = _amplitude_reconstruction(stage_dir)
    captures = _capture_rows(stage_dir)
    phase_path, phase_sha = _phase_output(stage_dir, contract)
    pairing: list[dict[str, Any]] = []
    measured_values: dict[str, tuple[np.ndarray, float]] = {}
    theories: dict[tuple[str, str], np.ndarray] = {}
    expected_bmps = {row["amplitude_file"] for row in probe_rows}
    geometry_file_shas: set[str] = set()
    geometry_payload_shas: set[str] = set()
    if set(reconstruction) != expected_bmps:
        raise RuntimeError("Amplitude reconstruction manifest does not exactly match probe manifest")
    if set(captures) != expected_bmps:
        raise RuntimeError("Acquisition manifest does not exactly match probe manifest")
    for row in probe_rows:
        key = row["capture_key"]
        amplitude_name = row["amplitude_file"]
        compact = safe_relative_file(
            stage_dir, row["compact_amplitude_file"], label="compact amplitude"
        )
        compact_sha = verify_file(
            compact, row["compact_amplitude_sha256"], label="compact amplitude"
        )
        reconstruction_row = reconstruction[amplitude_name]
        if reconstruction_row.get("source_sha256", "").lower() != compact_sha:
            raise RuntimeError(f"Amplitude source SHA mismatch for {amplitude_name}")
        amplitude_bmp = stage_dir / "amplitude_to_play" / amplitude_name
        amplitude_sha = verify_file(
            amplitude_bmp,
            reconstruction_row["output_sha256"],
            label="amplitude BMP",
        )
        capture_row = captures[amplitude_name]
        played_amplitude_sha = str(
            capture_row.get("amplitude_bmp_sha256", "")
        ).strip().lower()
        if played_amplitude_sha != amplitude_sha:
            raise RuntimeError(
                f"Acquisition amplitude SHA mismatch for {key}: "
                "the capture manifest is not bound to the reconstructed BMP"
            )
        geometry_file_sha, geometry_payload_sha = _validate_canonical_capture_row(
            capture_row, key
        )
        geometry_file_shas.add(geometry_file_sha)
        geometry_payload_shas.add(geometry_payload_sha)
        if capture_row.get("phase_mask_sha256", "").lower() != phase_sha:
            raise RuntimeError(f"Acquisition phase SHA mismatch for {key}")
        capture_name = capture_row.get("ccd_capture", "")
        if Path(capture_name).name != capture_name or Path(capture_name).stem != key:
            raise RuntimeError(f"Acquisition CCD basename does not match {key}")
        ccd_path = stage_dir / "ccd_captured" / capture_name
        declared_ccd_sha = str(capture_row.get("output_sha256", "")).strip().lower()
        observed_ccd_sha = verify_file(
            ccd_path, declared_ccd_sha, label="canonical CCD capture"
        )
        value, saturation_value = _load_ccd(ccd_path, expected_shape)
        measured_values[key] = (value, saturation_value)
        for reference_kind, (file_column, sha_column) in REFERENCE_COLUMNS.items():
            reference = safe_relative_file(stage_dir, row[file_column], label=reference_kind)
            reference_sha = verify_file(reference, row[sha_column], label=reference_kind)
            theories[(key, reference_kind)] = _load_reference(reference, expected_shape)
            pairing.append(
                {
                    "stage": contract["stage"],
                    "capture_key": key,
                    "canonical_key": row["canonical_key"],
                    "reference_kind": reference_kind,
                    "compact_amplitude_sha256": compact_sha,
                    "amplitude_bmp_sha256": amplitude_sha,
                    "phase_bmp": phase_path.name,
                    "phase_bmp_sha256": phase_sha,
                    "detector_geometry_file_sha256": geometry_file_sha,
                    "detector_geometry_payload_sha256": geometry_payload_sha,
                    "saved_frame_orientation": "canonical_model_xy",
                    "downstream_loader_flip_required": False,
                    "ccd_file": ccd_path.relative_to(stage_dir).as_posix(),
                    "ccd_sha256": observed_ccd_sha,
                    "reference_file": reference.relative_to(stage_dir).as_posix(),
                    "reference_sha256": reference_sha,
                    "status": "verified",
                }
            )
    if len(geometry_file_shas) != 1 or len(geometry_payload_shas) != 1:
        raise RuntimeError(
            "All captures in one agreement stage must use exactly one detector "
            "geometry file/payload SHA-256"
        )
    return pairing, measured_values, theories


def _energy_gains(
    rows: list[dict[str, str]],
    measured: dict[str, tuple[np.ndarray, float]],
    theories: dict[tuple[str, str], np.ndarray],
) -> dict[str, float]:
    gains: dict[str, float] = {}
    for reference_kind in REFERENCE_COLUMNS:
        ratios: list[float] = []
        for row in rows:
            if row["role"] != "calibration" or row["canonical_key"].endswith("dark"):
                continue
            left = measured[row["capture_key"]][0]
            right = theories[(row["capture_key"], reference_kind)]
            if float(right.mean()) > 1.0e-12 and float(left.mean()) > 0:
                ratios.append(float(left.mean() / right.mean()))
        gains[reference_kind] = float(np.median(ratios)) if ratios else 1.0
    return gains


def _metric_row(
    *,
    metadata: dict[str, Any],
    measured: np.ndarray,
    theory: np.ndarray,
    saturation_value: float,
    reference_kind: str,
    domain: str,
    gain: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        **metadata,
        "reference_kind": reference_kind,
        "domain": domain,
        **pair_metrics(
            measured,
            theory,
            domain=domain,
            energy_gain=gain,
            saturation_value=saturation_value,
            signal_energy_fraction=float(settings["signal_energy_fraction"]),
            relative_clip=float(settings["relative_clip"]),
            log_compression=float(settings["log_compression"]),
        ),
    }


def evaluate_session(
    session_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    stages: Iterable[str] | None = None,
) -> dict[str, Any]:
    session_dir = Path(session_dir).expanduser().resolve()
    root_manifest_path = session_dir / "agreement_manifest.json"
    root_manifest = read_json(root_manifest_path)
    if not isinstance(root_manifest, dict):
        raise RuntimeError("Agreement session manifest must be a JSON object")
    if root_manifest.get("schema_version") != SCHEMA_VERSION or root_manifest.get(
        "type"
    ) != "qwen_warmstart5_sim_to_real_agreement_session":
        raise RuntimeError("Unsupported agreement session schema")
    root_checkpoint_sha = require_sha256(
        root_manifest.get("checkpoint_sha256", ""),
        label="session checkpoint digest",
    )
    root_config_sha = require_sha256(
        root_manifest.get("resolved_config_sha256", ""),
        label="session resolved-config digest",
    )
    manifest_stages = root_manifest.get("stages")
    if not isinstance(manifest_stages, list) or not manifest_stages:
        raise RuntimeError("Agreement session must contain at least one stage")
    manifest_stage_names = [
        item.get("stage") if isinstance(item, dict) else None for item in manifest_stages
    ]
    if (
        any(name not in STAGES for name in manifest_stage_names)
        or len(manifest_stage_names) != len(set(manifest_stage_names))
    ):
        raise RuntimeError("Agreement session has invalid or duplicate stages")
    selected_stages = list(stages) if stages is not None else list(manifest_stage_names)
    if not selected_stages or any(stage not in STAGES for stage in selected_stages):
        raise ValueError(f"Invalid agreement stages: {selected_stages}")
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else session_dir / "agreement_evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)

    all_pairing: list[dict[str, Any]] = []
    frame_metrics: list[dict[str, Any]] = []
    probe_metrics: list[dict[str, Any]] = []
    repeatability: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    input_files = [root_manifest_path]
    stage_lookup = {item["stage"]: item for item in manifest_stages}
    bootstrap_samples = 2000
    bootstrap_seed = 42

    for stage in selected_stages:
        if stage not in stage_lookup:
            raise RuntimeError(f"Stage {stage} is absent from agreement_manifest.json")
        stage_item = stage_lookup[stage]
        stage_dir = stage_directory(session_dir, stage)
        expected_directory = stage_dir.relative_to(session_dir).as_posix()
        if stage_item.get("directory") != expected_directory:
            raise RuntimeError(f"Agreement stage directory mismatch for {stage}")
        contract_path = safe_relative_file(session_dir, stage_item["contract"], label="stage contract")
        verify_file(contract_path, stage_item["contract_sha256"], label="stage contract")
        contract = read_json(contract_path)
        if (
            not isinstance(contract, dict)
            or contract.get("schema_version") != SCHEMA_VERSION
            or contract.get("type") != "qwen_warmstart5_sim_to_real_agreement"
            or contract.get("stage") != stage
        ):
            raise RuntimeError(f"Agreement stage contract mismatch for {stage}")
        if contract.get("checkpoint_sha256") != root_checkpoint_sha:
            raise RuntimeError(f"Agreement checkpoint provenance mismatch for {stage}")
        if contract.get("resolved_config_sha256") != root_config_sha:
            raise RuntimeError(f"Agreement resolved-config provenance mismatch for {stage}")
        probe_manifest = stage_dir / contract["probe_manifest"]
        verify_file(probe_manifest, contract["probe_manifest_sha256"], label="probe manifest")
        rows = read_csv(probe_manifest)
        if len(rows) != int(contract["probe_count"]):
            raise RuntimeError(f"Probe manifest count mismatch for {stage}")
        keys = [row["capture_key"] for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate capture keys in {probe_manifest}")
        pairing, measured, theories = _pairing_contract(
            stage_dir=stage_dir, contract=contract, probe_rows=rows
        )
        all_pairing.extend(pairing)
        agreement = contract["agreement"]
        bootstrap_samples = int(agreement.get("bootstrap_samples", bootstrap_samples))
        bootstrap_seed = int(agreement.get("probe_seed", bootstrap_seed))
        gains = _energy_gains(rows, measured, theories)

        # Orientation is diagnosed only from predeclared calibration probes and
        # the transport-matched reference.  The primary metrics below always use
        # the identity/canonical registered CCD exactly as supplied.
        for row in rows:
            if row["role"] != "calibration":
                continue
            value = measured[row["capture_key"]][0]
            theory = theories[(row["capture_key"], "transport_quantized")]
            for candidate in orientation_diagnostics(
                value,
                theory,
                relative_clip=float(agreement["relative_clip"]),
            ):
                orientation_rows.append(
                    {
                        "stage": stage,
                        "capture_key": row["capture_key"],
                        "canonical_key": row["canonical_key"],
                        **candidate,
                        "used_for_primary_metrics": False,
                    }
                )

        rows_by_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_canonical[row["canonical_key"]].append(row)
            value, saturation_value = measured[row["capture_key"]]
            metadata = {
                "stage": stage,
                "capture_key": row["capture_key"],
                "canonical_key": row["canonical_key"],
                "source_kind": row["source_kind"],
                "role": row["role"],
                "split": row["split"],
                "sku_index": row["sku_index"],
                "sku_name": row["sku_name"],
                "repeat_index": int(row["repeat_index"]),
                "replicate_count": 1,
            }
            for reference_kind in REFERENCE_COLUMNS:
                theory = theories[(row["capture_key"], reference_kind)]
                for domain in DOMAINS:
                    frame_metrics.append(
                        _metric_row(
                            metadata=metadata,
                            measured=value,
                            theory=theory,
                            saturation_value=saturation_value,
                            reference_kind=reference_kind,
                            domain=domain,
                            gain=gains[reference_kind],
                            settings=agreement,
                        )
                    )

        for canonical_key, group in rows_by_canonical.items():
            arrays = [measured[row["capture_key"]][0] for row in group]
            saturation_values = [measured[row["capture_key"]][1] for row in group]
            average = np.mean(np.stack(arrays), axis=0)
            first = group[0]
            metadata = {
                "stage": stage,
                "capture_key": first["capture_key"],
                "canonical_key": canonical_key,
                "source_kind": first["source_kind"],
                "role": first["role"],
                "split": first["split"],
                "sku_index": first["sku_index"],
                "sku_name": first["sku_name"],
                "repeat_index": "mean",
                "replicate_count": len(group),
            }
            for reference_kind in REFERENCE_COLUMNS:
                theory = theories[(first["capture_key"], reference_kind)]
                for domain in DOMAINS:
                    probe_metrics.append(
                        _metric_row(
                            metadata=metadata,
                            measured=average,
                            theory=theory,
                            saturation_value=max(saturation_values),
                            reference_kind=reference_kind,
                            domain=domain,
                            gain=gains[reference_kind],
                            settings=agreement,
                        )
                    )
            if len(group) >= 2:
                for left_row, right_row in itertools.combinations(group, 2):
                    left = measured[left_row["capture_key"]][0]
                    right = measured[right_row["capture_key"]][0]
                    for domain in DOMAINS:
                        if domain == "network_input":
                            left_domain = network_input_map(
                                left,
                                relative_clip=float(agreement["relative_clip"]),
                                log_compression=float(agreement["log_compression"]),
                            )
                            right_domain = network_input_map(
                                right,
                                relative_clip=float(agreement["relative_clip"]),
                                log_compression=float(agreement["log_compression"]),
                            )
                        else:
                            left_domain, right_domain = left, right
                        repeatability.append(
                            {
                                "stage": stage,
                                "canonical_key": canonical_key,
                                "left_capture_key": left_row["capture_key"],
                                "right_capture_key": right_row["capture_key"],
                                "domain": domain,
                                "pcc_full": pcc(left_domain, right_domain),
                                "shape_nrmse": shape_nrmse(left_domain, right_domain),
                                "energy_ratio": (
                                    float(left_domain.mean() / right_domain.mean())
                                    if float(right_domain.mean()) > 1.0e-12
                                    else None
                                ),
                            }
                        )
        input_files.extend((contract_path, probe_manifest))

    summary_rows = _summary_rows(
        probe_metrics,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    pairing_fields = list(all_pairing[0])
    metric_fields = list(frame_metrics[0])
    probe_fields = list(probe_metrics[0])
    write_csv(output / "pairing_audit.csv", all_pairing, pairing_fields)
    write_csv(output / "metrics_per_frame.csv", frame_metrics, metric_fields)
    write_csv(output / "metrics_per_probe.csv", probe_metrics, probe_fields)
    write_csv(
        output / "summary_by_stage.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    repeat_fields = (
        list(repeatability[0])
        if repeatability
        else (
            "stage",
            "canonical_key",
            "left_capture_key",
            "right_capture_key",
            "domain",
            "pcc_full",
            "shape_nrmse",
            "energy_ratio",
        )
    )
    write_csv(output / "repeatability.csv", repeatability, repeat_fields)
    orientation_fields = (
        list(orientation_rows[0])
        if orientation_rows
        else (
            "stage",
            "capture_key",
            "canonical_key",
            "orientation",
            "pcc_full",
            "ssim",
            "used_for_primary_metrics",
        )
    )
    write_csv(output / "orientation_diagnostic.csv", orientation_rows, orientation_fields)
    orientation_summary: list[dict[str, Any]] = []
    grouped_orientation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in orientation_rows:
        grouped_orientation[(row["stage"], row["orientation"])].append(row)
    for (stage, orientation), values in sorted(grouped_orientation.items()):
        pcc_values = [float(row["pcc_full"]) for row in values if row["pcc_full"] is not None]
        ssim_values = [float(row["ssim"]) for row in values if row["ssim"] is not None]
        orientation_summary.append(
            {
                "stage": stage,
                "orientation": orientation,
                "calibration_probes": len(values),
                "median_pcc": float(np.median(pcc_values)) if pcc_values else None,
                "median_ssim": float(np.median(ssim_values)) if ssim_values else None,
                "used_for_primary_metrics": False,
            }
        )
    for stage in selected_stages:
        stage_values = [row for row in orientation_summary if row["stage"] == stage and row["median_pcc"] is not None]
        if stage_values:
            best = max(stage_values, key=lambda row: float(row["median_pcc"]))
            for row in stage_values:
                row["diagnostic_best_candidate"] = row is best
    write_csv(
        output / "orientation_summary.csv",
        orientation_summary,
        list(orientation_summary[0]) if orientation_summary else (
            "stage",
            "orientation",
            "calibration_probes",
            "median_pcc",
            "median_ssim",
            "used_for_primary_metrics",
            "diagnostic_best_candidate",
        ),
    )

    evidence_files = [
        output / "pairing_audit.csv",
        output / "metrics_per_frame.csv",
        output / "metrics_per_probe.csv",
        output / "summary_by_stage.csv",
        output / "repeatability.csv",
        output / "orientation_diagnostic.csv",
        output / "orientation_summary.csv",
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "type": "qwen_warmstart5_sim_to_real_agreement_evaluation",
        "session_dir": str(session_dir),
        "stages": selected_stages,
        "reference_kinds": list(REFERENCE_COLUMNS),
        "domains": list(DOMAINS),
        "pairings_verified": len(all_pairing),
        "frame_metric_rows": len(frame_metrics),
        "independent_probe_metric_rows": len(probe_metrics),
        "repeatability_rows": len(repeatability),
        "orientation_diagnostic": {
            "candidate_set": list(ORIENTATIONS),
            "summaries": orientation_summary,
            "policy": (
                "Candidates are scored on calibration probes only. The diagnostic "
                "winner is never applied to primary metrics; fix the session-level "
                "canonical transform and reacquire/reregister instead."
            ),
        },
        "primary_aggregation": (
            "Repeated captures are averaged within canonical_key; summary_by_stage "
            "uses independent canonical probes, never replicate frames as independent n."
        ),
        "registration_policy": (
            "Only an externally audited, session-fixed four-corner transform is allowed. "
            "No per-frame shift optimization is performed here."
        ),
        "normalization_policy": (
            "Linear metrics use nonnegative registered intensity without background "
            "subtraction or per-frame contrast fit. Network-input metrics reproduce "
            "mean normalization, relative clipping, log1p and 478-to-224 pooling."
        ),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in input_files
        ],
        "evidence_files": [
            {"path": path.name, "sha256": sha256_file(path)} for path in evidence_files
        ],
        "summaries": summary_rows,
    }
    write_json(output / "evaluation_manifest.json", result)
    print(
        f"[agreement_evaluate] stages={len(selected_stages)} "
        f"verified_pairs={len(all_pairing)} independent_rows={len(probe_metrics)} "
        f"output={output}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly compare Qwen warmstart5 simulated and measured CCDs"
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    args = parser.parse_args()
    evaluate_session(args.session_dir, output_dir=args.output_dir, stages=args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
