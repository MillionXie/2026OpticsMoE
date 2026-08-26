"""Evaluate an exact 478x478 raw CCD capture with no intensity processing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from experiments.d2nn_mnist4_single_layer_17um_10cm.ccd_evaluate import (
    _read_stage_contract,
    _validate_capture_manifest,
)

from .io_utils import write_csv, write_json
from .settings import load_settings


CAPTURE_SUFFIXES = (".bmp", ".png", ".tif", ".tiff")
QC_MIN_MEAN_UINT8 = 1.0
QC_MAX_SATURATION_FRACTION = 0.05
QC_MIN_ROI_RELATIVE_SPREAD = 0.02


def _read_csv(path: Path) -> list[dict[str, str]]:
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
            f"Expected exactly one raw CCD frame for {key!r}; found {matches}"
        )
    return matches[0]


def _is_quick40_diagnostic(
    rows: list[dict[str, str]], profile: str, suitable: bool
) -> bool:
    if suitable or profile.lower() != "quick40" or len(rows) != 40:
        return False
    try:
        counts = Counter(int(row["label"]) for row in rows)
    except (KeyError, TypeError, ValueError):
        return False
    return counts == Counter({label: 10 for label in range(4)})


def _frame_qc(raw: np.ndarray, energies: list[float]) -> dict[str, Any]:
    """Diagnose unusable frames without changing their values or prediction."""

    mean_uint8 = float(raw.mean())
    saturation_fraction = float(np.mean(raw >= 254.0))
    maximum = max(energies)
    roi_relative_spread = (
        0.0 if maximum <= 0.0 else float((maximum - min(energies)) / maximum)
    )
    reasons: list[str] = []
    if mean_uint8 <= QC_MIN_MEAN_UINT8:
        reasons.append("near_black_mean_le_1")
    if saturation_fraction >= QC_MAX_SATURATION_FRACTION:
        reasons.append("saturated_pixels_ge_5pct")
    if roi_relative_spread <= QC_MIN_ROI_RELATIVE_SPREAD:
        reasons.append("four_roi_relative_spread_le_2pct")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "frame_mean_uint8": mean_uint8,
        "saturation_fraction": saturation_fraction,
        "roi_relative_spread": roi_relative_spread,
    }


def evaluate_directory(
    *,
    config: Path,
    manifest: Path,
    ccd_dir: Path,
    output_dir: Path,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    allow_biased_demo_metric: bool = False,
    allow_invalid_formal: bool = False,
    generate_paper_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings(config)
    manifest = manifest.expanduser().resolve()
    ccd_dir = ccd_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError(f"Empty stage manifest: {manifest}")
    profiles = sorted({row.get("profile", "") for row in rows})
    if len(profiles) != 1:
        raise ValueError(f"One evaluation must contain one profile: {profiles}")
    contract_path, contract = _read_stage_contract(manifest, None)
    suitable = bool(contract["suitable_for_accuracy_reporting"])
    profile = profiles[0]
    quick40 = _is_quick40_diagnostic(rows, profile, suitable)
    if not suitable and not quick40 and not allow_biased_demo_metric:
        raise PermissionError(
            "demo_topk is simulation-selected. Add --allow-biased-demo-metric "
            "only for a diagnostic demo_success_rate."
        )
    capture_manifest = manifest.parent / "acquisition_logs" / "capture_manifest.csv"
    if suitable and not capture_manifest.is_file():
        raise FileNotFoundError(
            "Formal raw-CCD evaluation requires acquisition_logs/"
            f"capture_manifest.csv; missing {capture_manifest}"
        )
    capture_report: dict[str, Any]
    if capture_manifest.is_file():
        capture_report = _validate_capture_manifest(
            stage_rows=rows,
            capture_manifest=capture_manifest,
            ccd_dir=ccd_dir,
            expected_phase_sha256=str(contract["phase_sha256"]),
            expected_phase_file=str(contract["phase_file"]),
        )
    else:
        capture_report = {"verified": False, "reason": "demo_without_manifest"}

    result_rows: list[dict[str, Any]] = []
    confusion = np.zeros((4, 4), dtype=np.int64)
    correct = 0
    qc_reason_counts: Counter[str] = Counter()
    invalid_count = 0
    for row in rows:
        source = _find_capture(ccd_dir, row["key"])
        with Image.open(source) as opened:
            if opened.mode != "L" or opened.size != (478, 478):
                raise ValueError(
                    f"Raw CCD contract is 478x478 8-bit L; got "
                    f"{opened.mode}/{opened.size} for {source.name}. Refusing resize."
                )
            image = opened.copy()
        if flip_vertical:
            image = ImageOps.flip(image)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        # A dtype cast is representation only. There is deliberately no gain
        # normalization, clipping, nonlinear compression, or background subtraction.
        raw = np.asarray(image, dtype=np.float32)
        energies = [
            float(raw[top:bottom, left:right].sum())
            for left, top, right, bottom in settings.detector_bounds()
        ]
        prediction = int(np.argmax(energies))
        label = int(row["label"])
        is_correct = prediction == label
        qc = _frame_qc(raw, energies)
        invalid_count += int(not qc["valid"])
        qc_reason_counts.update(qc["reasons"])
        correct += int(is_correct)
        confusion[label, prediction] += 1
        result_rows.append(
            {
                "key": row["key"],
                "label": label,
                "prediction": prediction,
                "correct": is_correct,
                "valid": qc["valid"],
                "qc_reasons": ";".join(qc["reasons"]),
                "frame_mean_uint8": qc["frame_mean_uint8"],
                "saturation_fraction": qc["saturation_fraction"],
                "roi_relative_spread": qc["roi_relative_spread"],
                "raw_energy_0": energies[0],
                "raw_energy_1": energies[1],
                "raw_energy_2": energies[2],
                "raw_energy_3": energies[3],
                "source": str(source),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hardware_predictions_raw.csv", result_rows)
    rate = correct / len(rows)
    summary: dict[str, Any] = {
        "samples": len(rows),
        "correct": correct,
        "profile": profile,
        "suitable_for_accuracy_reporting": suitable,
        "confusion_matrix": confusion.tolist(),
        "detector_bounds_xyxy": [list(value) for value in settings.detector_bounds()],
        "ccd_input_contract": "478x478 uint8 grayscale",
        "score": "raw detector-region sum",
        "normalization": False,
        "nonlinearity": False,
        "background_subtraction": False,
        "resize": False,
        "flip_vertical": flip_vertical,
        "flip_horizontal": flip_horizontal,
        "stage_contract": str(contract_path),
        "capture_manifest": capture_report,
        "quality_control": {
            "passed": invalid_count == 0,
            "invalid_frames": invalid_count,
            "reason_counts": dict(sorted(qc_reason_counts.items())),
            "thresholds": {
                "minimum_frame_mean_uint8_exclusive": QC_MIN_MEAN_UINT8,
                "maximum_saturation_fraction_exclusive": (
                    QC_MAX_SATURATION_FRACTION
                ),
                "minimum_four_roi_relative_spread_exclusive": (
                    QC_MIN_ROI_RELATIVE_SPREAD
                ),
            },
            "note": "QC only flags frames; it never modifies CCD values or ROI sums.",
        },
    }
    if suitable:
        summary["reportable_accuracy"] = invalid_count == 0
        if invalid_count == 0:
            summary["accuracy"] = rate
        else:
            summary["diagnostic_all_frame_argmax_accuracy"] = rate
            summary["warning"] = (
                f"Formal run contains {invalid_count} invalid CCD frame(s); "
                "accuracy is not reportable until acquisition is corrected."
            )
    elif quick40:
        summary["diagnostic_success_rate"] = rate
        summary["reportable_accuracy"] = False
        summary["warning"] = (
            "quick40 is a fixed subset for alignment/exposure diagnosis only; "
            "never report it as hardware accuracy."
        )
    else:
        summary["demo_success_rate"] = rate
        summary["reportable_accuracy"] = False
        summary["warning"] = "Biased demo; never report as hardware accuracy."
    if generate_paper_report and invalid_count == 0:
        from .paper_evaluation import PredictionRunSpec, evaluate_prediction_runs

        mask_name = Path(str(contract.get("phase_file", "phase_mask"))).stem
        paper_dir = output_dir / "paper_evaluation"
        paper_report = evaluate_prediction_runs(
            runs=[
                PredictionRunSpec(
                    mask_name=mask_name,
                    predictions_path=output_dir / "hardware_predictions_raw.csv",
                    profile_override=profile,
                    suitable_override=suitable,
                    phase_sha256_override=str(contract.get("phase_sha256") or ""),
                )
            ],
            output_dir=paper_dir,
            allow_biased_diagnostic=allow_biased_demo_metric,
            make_plots=True,
        )
        summary["paper_evaluation"] = {
            "directory": str(paper_dir),
            "metrics": str(paper_dir / "paper_metrics.json"),
            "reporting_status": next(iter(paper_report["runs"].values()))[
                "reporting_status"
            ],
            "eligible_for_formal_comparison": bool(
                paper_report["formal400_run_count"]
            ),
        }
    elif generate_paper_report:
        summary["paper_evaluation"] = {
            "status": "skipped_due_to_failed_frame_qc",
            "invalid_frames": invalid_count,
        }
    write_json(output_dir / "hardware_metrics_raw.json", summary)
    if suitable and invalid_count and not allow_invalid_formal:
        raise RuntimeError(
            f"Formal raw-CCD evaluation found {invalid_count} invalid frame(s). "
            f"Diagnostics were written under {output_dir}; reacquire before "
            "reporting accuracy, or pass --allow-invalid-formal only to inspect "
            "the non-reportable diagnostic result."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ccd-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--allow-biased-demo-metric", action="store_true")
    parser.add_argument(
        "--allow-invalid-formal",
        action="store_true",
        help="Keep a non-reportable formal diagnostic when frame QC fails",
    )
    parser.add_argument(
        "--skip-paper-report",
        action="store_true",
        help="Developer-only: skip CSV/JSON and publication figure generation",
    )
    args = parser.parse_args(argv)
    report = evaluate_directory(
        config=Path(args.config),
        manifest=Path(args.manifest),
        ccd_dir=Path(args.ccd_dir),
        output_dir=Path(args.output_dir),
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        allow_biased_demo_metric=args.allow_biased_demo_metric,
        allow_invalid_formal=args.allow_invalid_formal,
        generate_paper_report=not args.skip_paper_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
