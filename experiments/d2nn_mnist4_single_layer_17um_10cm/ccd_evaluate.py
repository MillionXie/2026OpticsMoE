"""Evaluate captured MNIST-4 CCD frames under an explicit hardware contract."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .settings import load_settings


CAPTURE_SUFFIXES = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
DEFAULT_QC_MIN_MEAN_UINT8 = 1.0
DEFAULT_QC_MAX_SATURATION_FRACTION = 0.05
DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD = 0.02


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--roi must be left,top,right,bottom")
    left, top, right, bottom = parts
    if min(left, top) < 0 or right <= left or bottom <= top:
        raise ValueError(f"Invalid ROI {parts}")
    return parts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty hardware prediction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_capture(directory: Path, key: str) -> Path:
    matches = [
        directory / f"{key}{suffix}"
        for suffix in CAPTURE_SUFFIXES
        if (directory / f"{key}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CCD image for key={key!r} in {directory}, "
            f"found {matches}"
        )
    return matches[0]


def _read_stage_contract(
    manifest: Path, explicit_path: Path | None
) -> tuple[Path, dict[str, Any]]:
    contract_path = (
        explicit_path.expanduser().resolve()
        if explicit_path is not None
        else (manifest.parent / "stage_contract.json").resolve()
    )
    if not contract_path.is_file():
        raise FileNotFoundError(
            "Hardware evaluation requires stage_contract.json beside samples.csv; "
            f"missing: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError(f"Stage contract must be a JSON object: {contract_path}")
    if not isinstance(contract.get("suitable_for_accuracy_reporting"), bool):
        raise ValueError(
            "stage_contract.json must contain boolean "
            "suitable_for_accuracy_reporting"
        )
    return contract_path, contract


def _validate_capture_manifest(
    *,
    stage_rows: list[dict[str, str]],
    capture_manifest: Path,
    ccd_dir: Path,
    expected_phase_sha256: str,
    expected_phase_file: str,
) -> dict[str, Any]:
    capture_rows = _read_csv(capture_manifest)
    expected_count = len(stage_rows)
    if len(capture_rows) != expected_count:
        raise RuntimeError(
            "Capture play count does not match samples.csv: "
            f"expected={expected_count}, captured={len(capture_rows)}"
        )

    try:
        play_indices = [int(row["play_index"]) for row in capture_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "capture_manifest.csv has invalid or missing play_index values"
        ) from error
    expected_indices = list(range(expected_count))
    if play_indices != expected_indices:
        raise RuntimeError(
            "capture_manifest.csv play_index must be the uninterrupted sequence "
            f"0..{expected_count - 1}; got={play_indices[:10]}"
        )

    expected_amplitudes = sorted(
        row.get("amplitude_file") or f"{row['key']}.bmp" for row in stage_rows
    )
    captured_amplitudes = [row.get("amplitude_bmp", "") for row in capture_rows]
    if captured_amplitudes != expected_amplitudes:
        raise RuntimeError(
            "Capture amplitude sequence does not match the sorted formal stage "
            "file set"
        )

    capture_names = [row.get("ccd_capture", "") for row in capture_rows]
    if any(not name for name in capture_names) or len(capture_names) != len(
        set(capture_names)
    ):
        raise RuntimeError(
            "capture_manifest.csv must contain unique, non-empty ccd_capture names"
        )
    expected_keys = {row["key"] for row in stage_rows}
    captured_keys = {Path(name).stem for name in capture_names}
    if captured_keys != expected_keys:
        raise RuntimeError(
            "Capture filenames do not match samples.csv keys: "
            f"missing={sorted(expected_keys - captured_keys)[:5]}, "
            f"extra={sorted(captured_keys - expected_keys)[:5]}"
        )

    actual_capture_names = {
        path.name
        for path in ccd_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CAPTURE_SUFFIXES
    }
    manifested_capture_names = set(capture_names)
    if actual_capture_names != manifested_capture_names:
        raise RuntimeError(
            "CCD directory does not exactly match capture_manifest.csv: "
            f"missing={sorted(manifested_capture_names - actual_capture_names)[:5]}, "
            f"extra={sorted(actual_capture_names - manifested_capture_names)[:5]}"
        )

    phase_hashes = {row.get("phase_mask_sha256", "") for row in capture_rows}
    if phase_hashes != {expected_phase_sha256}:
        raise RuntimeError(
            "Captured phase SHA-256 does not match stage_contract.json: "
            f"expected={expected_phase_sha256}, observed={sorted(phase_hashes)}"
        )
    phase_names = {row.get("phase_mask", "") for row in capture_rows}
    if phase_names != {Path(expected_phase_file).name}:
        raise RuntimeError(
            "Captured phase filename does not match stage_contract.json: "
            f"expected={Path(expected_phase_file).name}, observed={sorted(phase_names)}"
        )
    return {
        "verified": True,
        "path": str(capture_manifest),
        "play_count": len(capture_rows),
        "phase_sha256": expected_phase_sha256,
        "exact_file_set": True,
    }


def evaluate_directory(
    *,
    config: Path,
    manifest: Path,
    ccd_dir: Path,
    output_dir: Path,
    roi: tuple[int, int, int, int] | None,
    flip_vertical: bool,
    flip_horizontal: bool,
    stage_contract: Path | None = None,
    capture_manifest: Path | None = None,
    allow_biased_demo_metric: bool = False,
    allow_invalid_formal: bool = False,
    qc_min_mean_uint8: float = DEFAULT_QC_MIN_MEAN_UINT8,
    qc_max_saturation_fraction: float = DEFAULT_QC_MAX_SATURATION_FRACTION,
    qc_min_region_relative_spread: float = DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD,
) -> dict[str, object]:
    manifest = Path(manifest).expanduser().resolve()
    ccd_dir = Path(ccd_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if qc_min_mean_uint8 < 0:
        raise ValueError("qc_min_mean_uint8 must be nonnegative")
    if not 0.0 <= qc_max_saturation_fraction <= 1.0:
        raise ValueError("qc_max_saturation_fraction must be in [0,1]")
    if qc_min_region_relative_spread < 0:
        raise ValueError("qc_min_region_relative_spread must be nonnegative")

    settings = load_settings(config)
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError(f"Hardware manifest is empty: {manifest}")
    keys = [row.get("key", "") for row in rows]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Hardware manifest keys must be non-empty and unique")
    profiles = sorted({row.get("profile", "unspecified") for row in rows})
    if len(profiles) != 1:
        raise RuntimeError(f"One evaluation must contain one profile: {profiles}")

    contract_path, contract = _read_stage_contract(manifest, stage_contract)
    if contract.get("profile") != profiles[0] or int(
        contract.get("samples", -1)
    ) != len(rows):
        raise RuntimeError("stage_contract.json does not match samples.csv")
    suitable = bool(contract["suitable_for_accuracy_reporting"])
    if not suitable and not allow_biased_demo_metric:
        raise PermissionError(
            "This demo profile was selected using simulation correctness/margin and "
            "cannot be evaluated by default. For alignment diagnostics only, rerun "
            "with --allow-biased-demo-metric; the result will be named "
            "demo_success_rate, never accuracy."
        )

    resolved_capture_manifest = (
        capture_manifest.expanduser().resolve()
        if capture_manifest is not None
        else (manifest.parent / "acquisition_logs" / "capture_manifest.csv").resolve()
    )
    if suitable and not resolved_capture_manifest.is_file():
        raise FileNotFoundError(
            "Formal evaluation requires acquisition_logs/capture_manifest.csv to "
            "prove the complete play sequence and phase SHA. Run --phase acquire "
            f"for this stage first; missing: {resolved_capture_manifest}"
        )
    if resolved_capture_manifest.is_file():
        capture_verification = _validate_capture_manifest(
            stage_rows=rows,
            capture_manifest=resolved_capture_manifest,
            ccd_dir=ccd_dir,
            expected_phase_sha256=str(contract.get("phase_sha256", "")),
            expected_phase_file=str(contract.get("phase_file", "")),
        )
    else:
        capture_verification = {
            "verified": False,
            "path": str(resolved_capture_manifest),
            "reason": "demo_diagnostic_without_capture_manifest",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    correct = 0
    valid_count = 0
    invalid_reasons: Counter[str] = Counter()
    confusion = np.zeros((4, 4), dtype=np.int64)
    class_total: Counter[int] = Counter()
    class_valid: Counter[int] = Counter()
    class_correct: Counter[int] = Counter()
    class_invalid: Counter[int] = Counter()

    for row in rows:
        key = row["key"]
        source = _find_capture(ccd_dir, key)
        with Image.open(source) as opened:
            image = opened.convert("L")
        if roi is not None:
            if roi[2] > image.width or roi[3] > image.height:
                raise ValueError(f"ROI {roi} exceeds {source.name} size {image.size}")
            image = image.crop(roi)
        if flip_vertical:
            image = ImageOps.flip(image)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        image = image.resize(
            (settings.ccd_target_size, settings.ccd_target_size),
            resample=Image.Resampling.BOX,
        )
        array = np.asarray(image, dtype=np.float32)
        frame_mean = float(array.mean())
        saturation_fraction = float(np.mean(array >= 254.5))
        raw_region_sums = [
            float(array[top:bottom, left:right].sum())
            for left, top, right, bottom in settings.detector_bounds()
        ]
        region_mean = float(np.mean(raw_region_sums))
        region_relative_spread = (
            float(max(raw_region_sums) - min(raw_region_sums))
            / max(region_mean, float(settings.loss_eps))
        )
        reasons: list[str] = []
        if frame_mean <= qc_min_mean_uint8:
            reasons.append("near_black")
        if saturation_fraction >= qc_max_saturation_fraction:
            reasons.append("saturated")
        if region_relative_spread <= qc_min_region_relative_spread:
            reasons.append("near_equal_detector_regions")

        total = max(float(array.sum()), float(settings.loss_eps))
        scores = [value / total for value in raw_region_sums]
        label = int(row["label"])
        if label not in range(4):
            raise ValueError(f"MNIST-4 label must be 0..3, got {label} for {key}")
        class_total[label] += 1
        if reasons:
            prediction = -1
            is_correct = False
            is_valid = False
            class_invalid[label] += 1
            invalid_reasons.update(reasons)
        else:
            prediction = int(np.argmax(scores))
            is_correct = prediction == label
            is_valid = True
            valid_count += 1
            correct += int(is_correct)
            class_valid[label] += 1
            class_correct[label] += int(is_correct)
            confusion[label, prediction] += 1

        processed_path = output_dir / "processed_478" / f"{key}.png"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(processed_path, optimize=True)
        result_rows.append(
            {
                "key": key,
                "label": label,
                "prediction": prediction,
                "valid": is_valid,
                "correct": is_correct,
                "qc_reasons": ";".join(reasons),
                "frame_mean_uint8": frame_mean,
                "saturation_fraction": saturation_fraction,
                "region_relative_spread": region_relative_spread,
                "score_0": scores[0],
                "score_1": scores[1],
                "score_2": scores[2],
                "score_3": scores[3],
                "source": str(source),
                "processed": str(processed_path),
            }
        )

    invalid_count = len(result_rows) - valid_count
    per_class: dict[str, Any] = {}
    for label in range(4):
        total_count = class_total[label]
        valid_for_class = class_valid[label]
        class_summary = {
            "samples": total_count,
            "valid_count": valid_for_class,
            "invalid_count": class_invalid[label],
            "correct": class_correct[label],
        }
        class_rate = (
            None if total_count == 0 else class_correct[label] / total_count
        )
        class_valid_rate = (
            None
            if valid_for_class == 0
            else class_correct[label] / valid_for_class
        )
        if suitable:
            class_summary["accuracy"] = class_rate
            class_summary["valid_only_accuracy"] = class_valid_rate
        else:
            class_summary["success_rate"] = class_rate
            class_summary["valid_only_success_rate"] = class_valid_rate
        per_class[str(label)] = class_summary

    summary: dict[str, Any] = {
        "samples": len(result_rows),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_by_reason": dict(sorted(invalid_reasons.items())),
        "correct": correct,
        "profile": profiles[0],
        "profiles": profiles,
        "suitable_for_accuracy_reporting": suitable,
        "stage_contract": str(contract_path),
        "capture_manifest": capture_verification,
        "roi_xyxy": None if roi is None else list(roi),
        "flip_vertical": flip_vertical,
        "flip_horizontal": flip_horizontal,
        "target_size": settings.ccd_target_size,
        "resize": "PIL BOX area resampling",
        "background_subtraction": False,
        "qc_thresholds": {
            "min_mean_uint8": qc_min_mean_uint8,
            "max_saturation_fraction": qc_max_saturation_fraction,
            "min_detector_region_relative_spread": qc_min_region_relative_spread,
        },
        "selection_policy": sorted(
            {row.get("selection_policy", "unspecified") for row in rows}
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
    conservative_rate = correct / max(1, len(result_rows))
    valid_only_rate = None if valid_count == 0 else correct / valid_count
    if suitable:
        summary["accuracy"] = conservative_rate
        summary["valid_only_accuracy"] = valid_only_rate
        summary["per_class_accuracy"] = {
            label: values["accuracy"] for label, values in per_class.items()
        }
    else:
        summary["demo_success_rate"] = conservative_rate
        summary["demo_valid_only_success_rate"] = valid_only_rate
        summary["reporting_warning"] = (
            "Biased, preselected demo diagnostic; do not report as test accuracy."
        )

    predictions_path = output_dir / "hardware_predictions.csv"
    metrics_path = output_dir / "hardware_metrics.json"
    _write_csv(predictions_path, result_rows)
    _write_json(metrics_path, summary)
    if suitable and invalid_count and not allow_invalid_formal:
        raise RuntimeError(
            f"Formal hardware evaluation rejected {invalid_count} invalid CCD "
            "frame(s). Inspect QC reasons and reacquire them. Diagnostic outputs "
            f"were written to {metrics_path}. Use --allow-invalid-formal only for "
            "explicit troubleshooting, never to certify a formal run."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate captured MNIST-4 CCD frames")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ccd-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-contract", default=None)
    parser.add_argument("--capture-manifest", default=None)
    parser.add_argument("--roi", default=None, help="left,top,right,bottom in raw CCD pixels")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--allow-biased-demo-metric", action="store_true")
    parser.add_argument("--allow-invalid-formal", action="store_true")
    parser.add_argument(
        "--qc-min-mean-uint8", type=float, default=DEFAULT_QC_MIN_MEAN_UINT8
    )
    parser.add_argument(
        "--qc-max-saturation-fraction",
        type=float,
        default=DEFAULT_QC_MAX_SATURATION_FRACTION,
    )
    parser.add_argument(
        "--qc-min-region-relative-spread",
        type=float,
        default=DEFAULT_QC_MIN_REGION_RELATIVE_SPREAD,
    )
    args = parser.parse_args(argv)
    summary = evaluate_directory(
        config=Path(args.config).expanduser().resolve(),
        manifest=Path(args.manifest).expanduser().resolve(),
        ccd_dir=Path(args.ccd_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        stage_contract=(
            None
            if args.stage_contract is None
            else Path(args.stage_contract).expanduser().resolve()
        ),
        capture_manifest=(
            None
            if args.capture_manifest is None
            else Path(args.capture_manifest).expanduser().resolve()
        ),
        roi=parse_roi(args.roi),
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        allow_biased_demo_metric=args.allow_biased_demo_metric,
        allow_invalid_formal=args.allow_invalid_formal,
        qc_min_mean_uint8=args.qc_min_mean_uint8,
        qc_max_saturation_fraction=args.qc_max_saturation_fraction,
        qc_min_region_relative_spread=args.qc_min_region_relative_spread,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
