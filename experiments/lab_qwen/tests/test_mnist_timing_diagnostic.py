from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from experiments.lab_qwen.mnist_timing_diagnostic import (
    analyze_timing_sweep,
    build_schedule,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_schedule_is_five_delays_times_four_digits() -> None:
    rows = [
        {"key": f"sample_{label}", "label": str(label), "amplitude_file": f"{label}.bmp"}
        for label in (2, 0, 3, 1)
    ]
    schedule = build_schedule(rows, [0, 50, 100, 200, 400], 1)
    assert len(schedule) == 20
    assert [row["label"] for row in schedule[:4]] == [0, 1, 2, 3]
    assert schedule[-1]["settle_delay_ms"] == 400.0


def test_report_writes_timing_plot_and_canonical_roi_overlay(tmp_path: Path) -> None:
    stage = tmp_path / "quick40"
    phase = stage / "phase_to_play"
    phase.mkdir(parents=True)
    Image.new("L", (1920, 1200), color=0).save(phase / "phase.bmp")
    _write_csv(
        phase / "reconstruction_manifest.csv",
        [{"output_bmp": "phase.bmp", "output_sha256": "unused"}],
    )
    samples = [
        {
            "key": f"sample_{label}",
            "label": label,
            "amplitude_file": f"sample_{label}.bmp",
        }
        for label in range(4)
    ]
    _write_csv(stage / "samples.csv", samples)
    bounds = [
        [20, 20, 79, 79],
        [399, 20, 458, 79],
        [20, 399, 79, 458],
        [399, 399, 458, 458],
    ]
    (stage / "stage_contract.json").write_text(
        json.dumps(
            {
                "profile": "quick40",
                "ccd_shape_hw": [478, 478],
                "detector_bounds_xyxy": bounds,
            }
        ),
        encoding="utf-8",
    )

    config = tmp_path / "formal_hardware.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "timing_diagnostic": {
                    "settle_delay_ms_values": [0, 50, 100, 200, 400],
                    "discard_frames_after_display": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    output = stage / "timing_diagnostic"
    ccd = output / "ccd"
    ccd.mkdir(parents=True)
    rows = []
    index = 0
    for delay in (0, 50, 100, 200, 400):
        for label in range(4):
            rng = np.random.default_rng(label)
            frame = rng.integers(0, 20, size=(478, 478), dtype=np.uint8)
            left, top, right, bottom = bounds[label]
            frame[top:bottom, left:right] = np.clip(
                frame[top:bottom, left:right].astype(np.int16) + 100,
                0,
                255,
            ).astype(np.uint8)
            name = f"wait_{delay:04d}_digit{label}.png"
            Image.fromarray(frame, mode="L").save(ccd / name)
            rows.append(
                {
                    "play_index": index,
                    "key": f"sample_{label}",
                    "label": label,
                    "repeat": 0,
                    "requested_settle_ms": delay,
                    "slm_write_and_complete_ms": 2.0,
                    "actual_post_slm_wait_ms": float(delay),
                    "ccd_discard_frames": 1,
                    "ccd_capture_rectify_save_ms": 8.0,
                    "total_pattern_cycle_ms": 10.0 + delay,
                    "camera_exposure_us": 5000.0,
                    "frame_mean_uint8": float(frame.mean()),
                    "frame_p99_uint8": float(np.percentile(frame, 99)),
                    "frame_p999_uint8": float(np.percentile(frame, 99.9)),
                    "saturation_fraction_ge_254": 0.0,
                    "amplitude_bmp": f"sample_{label}.bmp",
                    "ccd_capture": name,
                }
            )
            index += 1
    _write_csv(output / "timing_capture_manifest.csv", rows)
    (output / "acquisition_contract.json").write_text(
        json.dumps({"hardware_config": str(config)}), encoding="utf-8"
    )

    report = analyze_timing_sweep(stage=stage, output=output)

    assert report["captures"] == 20
    assert report["recommended_formal_slm_settle_delay_ms"] == 0.0
    assert report["orientation_overlay"]["predicted_digit"] == 0
    assert (output / "timing_summary.png").is_file()
    assert (output / "mnist4_detector_regions_overlay.png").is_file()
    assert (output / "timing_metrics_per_capture.csv").is_file()
