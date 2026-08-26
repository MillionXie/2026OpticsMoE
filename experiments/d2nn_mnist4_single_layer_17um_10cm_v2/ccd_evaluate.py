"""Evaluate an exact 478x478 raw CCD capture with no intensity processing."""

from __future__ import annotations

import argparse
import csv
import json
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


def evaluate_directory(
    *,
    config: Path,
    manifest: Path,
    ccd_dir: Path,
    output_dir: Path,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    allow_biased_demo_metric: bool = False,
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
    if not suitable and not allow_biased_demo_metric:
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
        correct += int(is_correct)
        confusion[label, prediction] += 1
        result_rows.append(
            {
                "key": row["key"],
                "label": label,
                "prediction": prediction,
                "correct": is_correct,
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
        "profile": profiles[0],
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
    }
    if suitable:
        summary["accuracy"] = rate
    else:
        summary["demo_success_rate"] = rate
        summary["warning"] = "Biased demo; never report as hardware accuracy."
    write_json(output_dir / "hardware_metrics_raw.json", summary)
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
    args = parser.parse_args(argv)
    report = evaluate_directory(
        config=Path(args.config),
        manifest=Path(args.manifest),
        ccd_dir=Path(args.ccd_dir),
        output_dir=Path(args.output_dir),
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        allow_biased_demo_metric=args.allow_biased_demo_metric,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
