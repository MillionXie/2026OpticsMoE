"""Measure SLM-to-CCD timing and visualize the MNIST-4 detector orientation.

The diagnostic deliberately uses four fixed quick40 samples (one per class) at
five configurable SLM settling delays.  The longest delay is the per-sample
reference, so the shorter-delay captures can be compared without confusing one
handwritten digit with another.  Captured intensities are never normalized or
modified for classification; gain alignment is used only for the timing
similarity diagnostic and is explicitly reported as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from experiments.hardware_sdk.devices import build_camera, build_slm, verify_camera_roi
from experiments.hardware_sdk.workflows.acquire_folder import (
    _capture_with_optional_geometry,
    _phase_mask_metadata,
    _resolve_detector_geometry,
)
from experiments.hardware_sdk.workflows.calibration_common import load_yaml_config


OWNER = "experiments.lab_qwen.mnist_timing_diagnostic"
CLASS_IDS = (0, 1, 2, 3)
EXPECTED_FRAME_SIZE = (478, 478)
DEFAULT_DELAYS_MS = (0.0, 50.0, 100.0, 200.0, 400.0)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float_list(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{label} values must be finite and non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} cannot contain duplicate values")
    return tuple(sorted(result))


def _timing_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("timing_diagnostic", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("timing_diagnostic must be a YAML mapping")
    delays = _as_float_list(
        value.get("settle_delay_ms_values", DEFAULT_DELAYS_MS),
        label="timing_diagnostic.settle_delay_ms_values",
    )
    repeats = int(value.get("repeats_per_digit_and_delay", 1))
    discard = int(value.get("discard_frames_after_display", 1))
    if repeats <= 0:
        raise ValueError("timing diagnostic repeats must be positive")
    if discard < 0:
        raise ValueError("timing diagnostic discard frame count cannot be negative")
    thresholds = {
        "minimum_pcc_to_reference": float(
            value.get("minimum_pcc_to_reference", 0.95)
        ),
        "maximum_gain_aligned_nmae": float(
            value.get("maximum_gain_aligned_nmae", 0.15)
        ),
        "maximum_absolute_mean_ratio_error": float(
            value.get("maximum_absolute_mean_ratio_error", 0.15)
        ),
        "maximum_saturation_fraction": float(
            value.get("maximum_saturation_fraction", 0.001)
        ),
        "minimum_p99_uint8": float(value.get("minimum_p99_uint8", 16.0)),
    }
    if not 0.0 <= thresholds["minimum_pcc_to_reference"] <= 1.0:
        raise ValueError("minimum_pcc_to_reference must be in [0,1]")
    if min(thresholds[key] for key in thresholds if key != "minimum_pcc_to_reference") < 0:
        raise ValueError("timing/exposure thresholds must be non-negative")
    return {
        "settle_delay_ms_values": delays,
        "repeats_per_digit_and_delay": repeats,
        "discard_frames_after_display": discard,
        "reference_delay_ms": max(delays),
        "thresholds": thresholds,
    }


def build_schedule(
    rows: Iterable[Mapping[str, str]],
    delays_ms: Iterable[float],
    repeats: int,
) -> list[dict[str, Any]]:
    """Choose one deterministic sample per digit and make a balanced schedule."""

    selected: dict[int, Mapping[str, str]] = {}
    for row in rows:
        label = int(row["label"])
        if label in CLASS_IDS and label not in selected:
            selected[label] = row
    if set(selected) != set(CLASS_IDS):
        raise ValueError(f"MNIST timing stage lacks one or more labels: {sorted(selected)}")
    schedule: list[dict[str, Any]] = []
    for delay in delays_ms:
        for repeat in range(repeats):
            for label in CLASS_IDS:
                source = selected[label]
                schedule.append(
                    {
                        "label": label,
                        "key": str(source["key"]),
                        "amplitude_file": str(source["amplitude_file"]),
                        "settle_delay_ms": float(delay),
                        "repeat": repeat,
                    }
                )
    return schedule


def _load_stage(stage: Path) -> tuple[list[dict[str, str]], dict[str, Any], Path, Path]:
    samples = _read_csv(stage / "samples.csv")
    contract_path = stage / "stage_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("profile") != "quick40":
        raise ValueError("Timing diagnostic requires a quick40 MNIST-4 session")
    if tuple(contract.get("ccd_shape_hw", ())) != (478, 478):
        raise ValueError("Timing diagnostic requires the canonical 478x478 CCD contract")
    phase_dir = stage / "phase_to_play"
    phase_files = sorted(phase_dir.glob("*.bmp"))
    if len(phase_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one phase BMP under {phase_dir}; found {len(phase_files)}"
        )
    phase_manifest = phase_dir / "reconstruction_manifest.csv"
    if not phase_manifest.is_file():
        raise FileNotFoundError(f"Phase reconstruction manifest is missing: {phase_manifest}")
    return samples, contract, phase_files[0], phase_manifest


def _prepare_output(output: Path, clear_output: bool) -> None:
    marker = output / ".mnist_timing_diagnostic.json"
    if output.exists() and any(output.iterdir()):
        if not clear_output:
            raise FileExistsError(
                f"Timing output already exists: {output}. Pass --clear-output to replace it."
            )
        if not marker.is_file():
            raise RuntimeError(
                f"Refusing to clear an unowned directory (marker missing): {output}"
            )
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("owner") != OWNER:
            raise RuntimeError(f"Refusing to clear a foreign directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        marker,
        {"owner": OWNER, "created_utc": datetime.now(timezone.utc).isoformat()},
    )


def acquire_timing_sweep(
    *,
    config_path: Path,
    stage: Path,
    output: Path,
    clear_output: bool,
    assume_yes: bool,
) -> dict[str, Any]:
    raw, resolved_config = load_yaml_config(config_path)
    base = resolved_config.parent
    timing = _timing_settings(raw)
    samples, contract, phase_file, phase_manifest = _load_stage(stage)
    schedule = build_schedule(
        samples,
        timing["settle_delay_ms_values"],
        timing["repeats_per_digit_and_delay"],
    )
    amplitude_dir = stage / "amplitude_to_play"
    amplitude_files = sorted(
        {amplitude_dir / str(item["amplitude_file"]) for item in schedule}
    )
    missing = [str(path) for path in amplitude_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Timing amplitude BMPs are missing: {missing}")

    phase_config = dict(raw.get("phase_slm", {}))
    expected_phase_size = tuple(
        int(value)
        for value in phase_config.get(
            "expected_resolution_wh",
            (phase_config.get("width", 1920), phase_config.get("height", 1200)),
        )
    )
    phase_metadata = _phase_mask_metadata(
        phase_file, expected_phase_size, phase_manifest
    )
    camera_config = dict(raw["camera"])
    camera_config["discard_frames_after_display"] = timing[
        "discard_frames_after_display"
    ]
    verify_camera_roi(camera_config)
    geometry, geometry_metadata, raw_camera_config = _resolve_detector_geometry(
        camera_config, base
    )
    if geometry is None:
        raise ValueError(
            "Timing diagnostic requires generated/formal_hardware.yaml with the "
            "four-point detector homography enabled"
        )
    slm_driver = build_slm(dict(raw["amplitude_slm"]), base)
    camera_driver = build_camera(raw_camera_config, base)
    slm_driver.validate_runtime()
    camera_driver.validate_runtime()
    slm_driver.validate_files(amplitude_files)

    if bool(raw.get("confirm_before_start", True)) and not assume_yes:
        answer = input(
            "MNIST timing sweep will play "
            f"{len(schedule)} frames. Confirm that this exact phase mask is visible:\n"
            f"  {phase_file}\n  sha256={phase_metadata['sha256']}\n"
            "Enter y to start: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise KeyboardInterrupt("operator cancelled timing diagnostic")

    _prepare_output(output, clear_output)
    captures = output / "ccd"
    captures.mkdir()
    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        slm = stack.enter_context(slm_driver)
        camera = stack.enter_context(camera_driver)
        verify_camera_roi(camera_config, camera.device_info())
        slm.preload_files(amplitude_files)
        for index, item in enumerate(schedule):
            source = amplitude_dir / item["amplitude_file"]
            delay_ms = float(item["settle_delay_ms"])
            output_name = (
                f"wait_{int(round(delay_ms)):04d}ms_r{int(item['repeat']):02d}_"
                f"digit{int(item['label'])}_{item['key']}.png"
            )
            destination = captures / output_name
            frame_started = time.perf_counter_ns()
            slm_started = time.perf_counter_ns()
            slm.display_file(source)
            slm_finished = time.perf_counter_ns()
            settle_started = slm_finished
            time.sleep(delay_ms / 1000.0)
            capture_started = time.perf_counter_ns()
            capture_info = _capture_with_optional_geometry(
                camera, destination, camera_config, geometry
            )
            capture_finished = time.perf_counter_ns()
            if capture_info.get("saved_size_wh") != [478, 478]:
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    "Timing capture was not canonical 478x478: "
                    f"{capture_info.get('saved_size_wh')}"
                )
            with Image.open(destination) as opened:
                frame = np.asarray(opened, dtype=np.float64)
            last = dict(camera.device_info().get("last_capture") or {})
            row = {
                "play_index": index,
                "key": item["key"],
                "label": int(item["label"]),
                "repeat": int(item["repeat"]),
                "requested_settle_ms": delay_ms,
                "slm_write_and_complete_ms": (slm_finished - slm_started) / 1e6,
                "actual_post_slm_wait_ms": (capture_started - settle_started) / 1e6,
                "ccd_discard_frames": timing["discard_frames_after_display"],
                "ccd_capture_rectify_save_ms": (capture_finished - capture_started) / 1e6,
                "total_pattern_cycle_ms": (capture_finished - frame_started) / 1e6,
                "camera_frame_index": last.get("camera_frame_index"),
                "camera_exposure_us": raw_camera_config.get("exposure_us"),
                "frame_mean_uint8": float(frame.mean()),
                "frame_p99_uint8": float(np.percentile(frame, 99.0)),
                "frame_p999_uint8": float(np.percentile(frame, 99.9)),
                "saturation_fraction_ge_254": float(np.mean(frame >= 254.0)),
                "amplitude_bmp": source.name,
                "amplitude_bmp_sha256": _sha256(source),
                "ccd_capture": output_name,
                "ccd_capture_sha256": _sha256(destination),
                "captured_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            print(
                f"[timing] {index + 1}/{len(schedule)} digit={row['label']} "
                f"wait={delay_ms:g} ms -> {output_name}"
            )
    _write_csv(output / "timing_capture_manifest.csv", rows)
    _write_json(
        output / "acquisition_contract.json",
        {
            "schema_version": 1,
            "owner": OWNER,
            "stage": str(stage),
            "stage_contract_sha256": _sha256(stage / "stage_contract.json"),
            "hardware_config": str(resolved_config),
            "phase_mask": phase_metadata,
            "detector_geometry": geometry_metadata,
            "camera_warmup_frames_once_at_open": int(
                raw_camera_config.get("warmup_frames", 3)
            ),
            "discard_frames_after_every_slm_display": timing[
                "discard_frames_after_display"
            ],
            "settle_delay_ms_values": list(timing["settle_delay_ms_values"]),
            "schedule_count": len(schedule),
            "timing_boundaries": {
                "slm_write_and_complete_ms": (
                    "Meadowlark Write_image through ImageWriteComplete"
                ),
                "actual_post_slm_wait_ms": (
                    "ImageWriteComplete return to CCD capture() call"
                ),
                "ccd_capture_rectify_save_ms": (
                    "discard configured stream frames, acquire one frame, "
                    "homography, fixed bit-depth conversion, and PNG save"
                ),
            },
            "classification_intensity_processing": "none",
        },
    )
    return {
        "captured": len(rows),
        "output": str(output),
        "reference_delay_ms": timing["reference_delay_ms"],
    }


def _read_frame(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.mode != "L" or opened.size != EXPECTED_FRAME_SIZE:
            raise ValueError(
                f"Timing CCD frame must be 478x478 8-bit L; got "
                f"{opened.mode}/{opened.size}: {path}"
            )
        return np.asarray(opened, dtype=np.float64)


def _pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.reshape(-1) - float(left.mean())
    y = right.reshape(-1) - float(right.mean())
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return None
    return float(np.dot(x, y) / denominator)


def _gain_aligned_nmae(observed: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.dot(observed.reshape(-1), observed.reshape(-1)))
    gain = 1.0 if denominator <= 1e-12 else float(
        np.dot(observed.reshape(-1), reference.reshape(-1)) / denominator
    )
    scale = max(float(np.mean(np.abs(reference))), 1e-12)
    return float(np.mean(np.abs(gain * observed - reference)) / scale)


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return None if not finite else float(np.median(finite))


def _make_roi_overlay(
    frame: np.ndarray,
    bounds: list[list[int]],
    energies: list[float],
    *,
    label: int,
    delay_ms: float,
    source_name: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.patches import Rectangle

    candidates = ["Arial", "Liberation Sans", "DejaVu Sans"]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(6.0 / 2.54, 6.0 / 2.54), dpi=300)
    axis.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    winner = int(np.argmax(energies))
    for region, (left, top, right, bottom) in enumerate(bounds):
        axis.add_patch(
            Rectangle(
                (left - 0.5, top - 0.5),
                right - left,
                bottom - top,
                fill=False,
                edgecolor=COLORS[region],
                linewidth=1.0,
            )
        )
        axis.text(
            left,
            max(2, top - 4),
            f"digit {region}{'  WIN' if region == winner else ''}",
            color=COLORS[region],
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1, "edgecolor": "none"},
        )
    axis.text(238.5, 5, "CANONICAL TOP", color="white", ha="center", va="top")
    axis.text(5, 238.5, "LEFT", color="white", ha="left", va="center", rotation=90)
    axis.text(472, 238.5, "RIGHT", color="white", ha="right", va="center", rotation=270)
    axis.text(238.5, 472, "BOTTOM", color="white", ha="center", va="bottom")
    axis.set_title(
        f"MNIST-4 CCD ROI orientation | true={label}, predicted={winner}, "
        f"wait={delay_ms:g} ms\n{source_name}"
    )
    axis.set_xlim(-0.5, 477.5)
    axis.set_ylim(477.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout(pad=0.25)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _make_summary_figure(
    rows: list[dict[str, Any]],
    delay_summary: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager, pyplot as plt

    candidates = ["Arial", "Liberation Sans", "DejaVu Sans"]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.5 / 2.54, 5.2 / 2.54), dpi=300)
    delays = sorted({float(row["requested_settle_ms"]) for row in rows})
    for label, color in zip(CLASS_IDS, COLORS):
        subset = [row for row in rows if int(row["label"]) == label]
        axes[0].plot(
            [float(row["requested_settle_ms"]) for row in subset],
            [np.nan if row["pcc_to_longest_wait"] is None else row["pcc_to_longest_wait"] for row in subset],
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            color=color,
            label=str(label),
        )
    axes[0].set_xlabel("SLM settle delay (ms)")
    axes[0].set_ylabel("PCC vs longest wait")
    axes[0].set_ylim(-0.05, 1.03)
    axes[0].legend(title="Digit", frameon=False, ncol=2)

    axes[1].plot(
        delays,
        [next(item["median_gain_aligned_nmae"] for item in delay_summary if item["settle_delay_ms"] == delay) for delay in delays],
        color="#0072B2",
        marker="o",
        markersize=2.5,
        linewidth=0.8,
    )
    axes[1].set_xlabel("SLM settle delay (ms)")
    axes[1].set_ylabel("Median gain-aligned NMAE")

    slm_means = []
    capture_means = []
    for delay in delays:
        subset = [row for row in rows if float(row["requested_settle_ms"]) == delay]
        slm_means.append(float(np.mean([row["slm_write_and_complete_ms"] for row in subset])))
        capture_means.append(float(np.mean([row["ccd_capture_rectify_save_ms"] for row in subset])))
    axes[2].plot(delays, slm_means, marker="o", markersize=2.5, linewidth=0.8, label="SLM write+complete")
    axes[2].plot(delays, capture_means, marker="s", markersize=2.5, linewidth=0.8, label="CCD+warp+save")
    axes[2].plot(delays, delays, linestyle="--", linewidth=0.8, color="#777777", label="requested wait")
    axes[2].set_xlabel("SLM settle delay (ms)")
    axes[2].set_ylabel("Measured time (ms)")
    axes[2].legend(frameon=False)
    for axis, letter in zip(axes, "abc"):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.18, 1.03, letter, transform=axis.transAxes, fontweight="bold")
    figure.tight_layout(w_pad=1.0)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze_timing_sweep(*, stage: Path, output: Path) -> dict[str, Any]:
    _, contract, _, _ = _load_stage(stage)
    rows = _read_csv(output / "timing_capture_manifest.csv")
    if len(rows) < len(CLASS_IDS):
        raise RuntimeError("Timing manifest contains fewer than four captures")
    parsed: list[dict[str, Any]] = []
    frames: dict[tuple[int, float, int], np.ndarray] = {}
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for key in (
            "play_index",
            "label",
            "repeat",
            "ccd_discard_frames",
        ):
            row[key] = int(row[key])
        for key in (
            "requested_settle_ms",
            "slm_write_and_complete_ms",
            "actual_post_slm_wait_ms",
            "ccd_capture_rectify_save_ms",
            "total_pattern_cycle_ms",
            "camera_exposure_us",
            "frame_mean_uint8",
            "frame_p99_uint8",
            "frame_p999_uint8",
            "saturation_fraction_ge_254",
        ):
            row[key] = float(row[key])
        frame = _read_frame(output / "ccd" / str(row["ccd_capture"]))
        frames[(row["label"], row["requested_settle_ms"], row["repeat"])] = frame
        parsed.append(row)
    delays = sorted({float(row["requested_settle_ms"]) for row in parsed})
    reference_delay = max(delays)
    for row in parsed:
        frame = frames[(row["label"], row["requested_settle_ms"], row["repeat"])]
        reference = frames[(row["label"], reference_delay, row["repeat"])]
        pcc = _pcc(frame, reference)
        reference_mean = max(float(reference.mean()), 1e-12)
        row["pcc_to_longest_wait"] = pcc
        row["gain_aligned_nmae_to_longest_wait"] = _gain_aligned_nmae(
            frame, reference
        )
        row["mean_intensity_ratio_to_longest_wait"] = float(
            frame.mean() / reference_mean
        )

    contract_acquisition = json.loads(
        (output / "acquisition_contract.json").read_text(encoding="utf-8")
    )
    config_raw, _ = load_yaml_config(contract_acquisition["hardware_config"])
    timing = _timing_settings(config_raw)
    thresholds = timing["thresholds"]
    delay_summary: list[dict[str, Any]] = []
    for delay in delays:
        subset = [row for row in parsed if row["requested_settle_ms"] == delay]
        pcc_values = [row["pcc_to_longest_wait"] for row in subset]
        nmae_values = [row["gain_aligned_nmae_to_longest_wait"] for row in subset]
        ratio_values = [row["mean_intensity_ratio_to_longest_wait"] for row in subset]
        median_pcc = _median(pcc_values)
        median_nmae = _median(nmae_values)
        median_ratio_error = _median(abs(value - 1.0) for value in ratio_values)
        stable = bool(
            median_pcc is not None
            and median_nmae is not None
            and median_ratio_error is not None
            and median_pcc >= thresholds["minimum_pcc_to_reference"]
            and median_nmae <= thresholds["maximum_gain_aligned_nmae"]
            and median_ratio_error
            <= thresholds["maximum_absolute_mean_ratio_error"]
        )
        delay_summary.append(
            {
                "settle_delay_ms": delay,
                "captures": len(subset),
                "median_pcc_to_longest_wait": median_pcc,
                "minimum_pcc_to_longest_wait": (
                    None if any(value is None for value in pcc_values) else min(pcc_values)
                ),
                "median_gain_aligned_nmae": median_nmae,
                "maximum_gain_aligned_nmae": max(nmae_values),
                "median_absolute_mean_ratio_error": median_ratio_error,
                "stable_by_configured_thresholds": stable,
            }
        )
    stable_delays = [
        item["settle_delay_ms"]
        for item in delay_summary
        if item["stable_by_configured_thresholds"]
    ]
    recommended = min(stable_delays) if stable_delays else reference_delay

    saturation = [row["saturation_fraction_ge_254"] for row in parsed]
    p99 = [row["frame_p99_uint8"] for row in parsed]
    exposure_warnings: list[str] = []
    if max(saturation) > thresholds["maximum_saturation_fraction"]:
        exposure_warnings.append("saturation_above_configured_limit_reduce_exposure")
    if float(np.median(p99)) < thresholds["minimum_p99_uint8"]:
        exposure_warnings.append("p99_too_dark_increase_exposure_or_input_power")

    bounds = [list(map(int, value)) for value in contract["detector_bounds_xyxy"]]
    overlay_candidates = [
        row
        for row in parsed
        if row["requested_settle_ms"] == reference_delay and row["repeat"] == 0
    ]
    overlay_row = min(overlay_candidates, key=lambda item: int(item["label"]))
    overlay_frame = frames[(overlay_row["label"], reference_delay, 0)]
    energies = [
        float(overlay_frame[top:bottom, left:right].sum())
        for left, top, right, bottom in bounds
    ]
    _make_roi_overlay(
        overlay_frame,
        bounds,
        energies,
        label=int(overlay_row["label"]),
        delay_ms=reference_delay,
        source_name=str(overlay_row["ccd_capture"]),
        output=output / "mnist4_detector_regions_overlay.png",
    )
    _make_summary_figure(parsed, delay_summary, output / "timing_summary.png")

    analyzed_rows: list[dict[str, Any]] = []
    for row in parsed:
        analyzed_rows.append(
            {
                **row,
                "pcc_to_longest_wait": (
                    "" if row["pcc_to_longest_wait"] is None else row["pcc_to_longest_wait"]
                ),
            }
        )
    _write_csv(output / "timing_metrics_per_capture.csv", analyzed_rows)
    summary = {
        "schema_version": 1,
        "captures": len(parsed),
        "digits": list(CLASS_IDS),
        "settle_delay_ms_values": delays,
        "reference_delay_ms": reference_delay,
        "recommended_formal_slm_settle_delay_ms": recommended,
        "recommendation_is_diagnostic_not_accuracy": True,
        "discard_frames_after_display": parsed[0]["ccd_discard_frames"],
        "camera_exposure_us": parsed[0]["camera_exposure_us"],
        "thresholds": thresholds,
        "per_delay": delay_summary,
        "exposure_qc": {
            "maximum_saturation_fraction_ge_254": max(saturation),
            "median_p99_uint8": float(np.median(p99)),
            "warnings": exposure_warnings,
            "passed": not exposure_warnings,
        },
        "orientation_overlay": {
            "file": "mnist4_detector_regions_overlay.png",
            "source_capture": overlay_row["ccd_capture"],
            "true_digit": overlay_row["label"],
            "raw_detector_region_sums": energies,
            "predicted_digit": int(np.argmax(energies)),
            "detector_bounds_xyxy": bounds,
            "note": (
                "Overlay is after the four-point homography in canonical model "
                "coordinates. It changes visualization only; raw ROI sums are untouched."
            ),
        },
        "timing_columns": {
            "slm_write_and_complete_ms": "blocking SDK write through ImageWriteComplete",
            "actual_post_slm_wait_ms": "main configurable SLM-to-CCD delay",
            "ccd_capture_rectify_save_ms": (
                "discard frames + one CCD frame + homography + fixed conversion + save"
            ),
            "camera_warmup_frames": "once when camera opens; not repeated per image",
        },
        "classification_intensity_processing": "none",
        "timing_similarity_only_processing": (
            "PCC and least-squares scalar-gain-aligned NMAE against the same "
            "digit at the longest delay"
        ),
        "outputs": {
            "raw_timing_log": "timing_capture_manifest.csv",
            "per_capture_metrics": "timing_metrics_per_capture.csv",
            "summary_figure": "timing_summary.png",
            "roi_orientation_figure": "mnist4_detector_regions_overlay.png",
        },
    }
    _write_json(output / "timing_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("acquire", "report", "all"), default="all")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument(
        "--config", default="experiments/lab_qwen/generated/formal_hardware.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    stage = Path(args.stage_dir).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else stage / "timing_diagnostic"
    )
    report: dict[str, Any] = {}
    if args.phase in {"acquire", "all"}:
        report["acquisition"] = acquire_timing_sweep(
            config_path=Path(args.config),
            stage=stage,
            output=output,
            clear_output=args.clear_output,
            assume_yes=args.yes,
        )
    if args.phase in {"report", "all"}:
        report["analysis"] = analyze_timing_sweep(stage=stage, output=output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
