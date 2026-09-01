from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .hardware_contract import (
    require_empty_directory,
    sha256_file,
    unique_image_files,
)
from .router import sparsify_probabilities
from .settings import load_settings


IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def _quality_thresholds(settings: object) -> dict[str, float]:
    return {
        "maximum_saturated_pixel_fraction": float(
            settings.optical_router_maximum_saturated_pixel_fraction
        ),
        "minimum_p99_uint8": float(settings.optical_router_minimum_p99_uint8),
        "minimum_dynamic_range_uint8": float(
            settings.optical_router_minimum_dynamic_range_uint8
        ),
        "minimum_topk_probability_margin": float(
            settings.optical_router_minimum_topk_probability_margin
        ),
    }


def _read_intensity(path: Path, expected_size: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise RuntimeError(
                f"Router CCD must be a single-channel uint8 image (mode L): "
                f"{path} has mode {image.mode!r}"
            )
        array = np.asarray(image)
    if array.ndim != 2 or array.shape != (expected_size, expected_size):
        raise RuntimeError(
            f"Router CCD must already be canonical {expected_size}x{expected_size}: "
            f"{path} has {array.shape}"
        )
    if array.dtype != np.uint8:
        raise RuntimeError(f"Router CCD must use uint8 pixels: {path} has {array.dtype}")
    value = array.astype(np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise RuntimeError(f"Router CCD must be finite and nonnegative: {path}")
    return value


def score_directory(
    config: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    expected_stems: list[str] | None = None,
) -> dict[str, object]:
    settings = load_settings(config)
    if settings.router_backend != "optical":
        raise ValueError("Router CCD scoring requires an optical-router config")
    paths = unique_image_files(input_dir)
    if expected_stems is not None:
        if len(expected_stems) != len(set(expected_stems)):
            raise RuntimeError("Expected Router CCD stems are not unique")
        by_stem = {path.stem: path for path in paths}
        missing = sorted(set(expected_stems).difference(by_stem))
        unexpected = sorted(set(by_stem).difference(expected_stems))
        if missing or unexpected:
            raise RuntimeError(
                "Router CCD/manifest identity mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        paths = [by_stem[stem] for stem in expected_stems]
    output_dir = require_empty_directory(output_dir, label="Router score output")
    intervals = settings.optical_router_detector_intervals
    thresholds = _quality_thresholds(settings)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for path in paths:
        value = _read_intensity(path, settings.active_size)
        energies = []
        for top, bottom in intervals:
            for left, right in intervals:
                energies.append(float(value[top:bottom, left:right].sum()))
        energy = torch.tensor(energies, dtype=torch.float64).unsqueeze(0)
        total_frame_energy = float(value.sum())
        total_detector_energy = float(sum(energies))
        fractions = energy / energy.sum(dim=-1, keepdim=True).clamp_min(
            settings.optical_router_energy_eps
        )
        if settings.optical_router_score_normalization == "standardized_region_energy":
            centered = energy - energy.mean(dim=-1, keepdim=True)
            logits = centered / centered.square().mean(
                dim=-1, keepdim=True
            ).add(settings.optical_router_energy_eps).sqrt()
        else:
            logits = fractions.clamp_min(settings.optical_router_energy_eps).log()
        probabilities = torch.softmax(logits / settings.router_temperature, dim=-1)
        weights, selected, indices = sparsify_probabilities(
            probabilities,
            settings.top_k,
            normalization=settings.router_weight_normalization,
            straight_through=False,
            eps=settings.optical_router_energy_eps,
        )
        raw_captured = (
            total_detector_energy / total_frame_energy
            if total_frame_energy > settings.optical_router_energy_eps
            else 0.0
        )
        sorted_probability = torch.sort(probabilities[0], descending=True).values
        margin = (
            float(sorted_probability[settings.top_k - 1] - sorted_probability[settings.top_k])
            if settings.top_k < 4
            else float("nan")
        )
        saturated = float(np.mean(value >= 255.0))
        p01 = float(np.percentile(value, 1.0))
        p99 = float(np.percentile(value, 99.0))
        dynamic_range = p99 - p01
        reasons: list[str] = []
        if total_frame_energy <= settings.optical_router_energy_eps:
            reasons.append("zero_total_frame_energy")
        if total_detector_energy <= settings.optical_router_energy_eps:
            reasons.append("zero_detector_energy")
        if bool(np.all(value >= 255.0)):
            reasons.append("all_pixels_saturated")
        if max(energies) - min(energies) <= settings.optical_router_energy_eps:
            reasons.append("uniform_detector_region_energy")
        if saturated > thresholds["maximum_saturated_pixel_fraction"]:
            reasons.append("saturated_pixel_fraction_above_maximum")
        if p99 < thresholds["minimum_p99_uint8"]:
            reasons.append("p99_below_minimum")
        if dynamic_range < thresholds["minimum_dynamic_range_uint8"]:
            reasons.append("dynamic_range_below_minimum")
        if (
            settings.top_k < 4
            and margin < thresholds["minimum_topk_probability_margin"]
        ):
            reasons.append("topk_probability_margin_below_minimum")
        row: dict[str, object] = {
            "filename": path.name,
            "ccd_sha256": sha256_file(path),
            "selected_experts": ",".join(map(str, indices[0].tolist())),
            # This is deliberately named raw: no measured dark frame or camera
            # offset was subtracted, so it is not a physical capture efficiency.
            "raw_capture_fraction": raw_captured,
            "topk_probability_margin": margin,
            "saturated_pixel_fraction": saturated,
            "p01_uint8": p01,
            "p99_uint8": p99,
            "dynamic_range_uint8": dynamic_range,
        }
        for index in range(4):
            row[f"energy_{index}"] = energies[index]
            row[f"energy_fraction_{index}"] = float(fractions[0, index])
            row[f"probability_{index}"] = float(probabilities[0, index])
            row[f"weight_{index}"] = float(weights[0, index])
            row[f"selected_{index}"] = bool(selected[0, index])
        if reasons:
            failures.append({**row, "failure_reasons": ";".join(reasons)})
        else:
            rows.append(row)

    if failures:
        failure_csv = output_dir / "routing_quality_failures.csv"
        with failure_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0]))
            writer.writeheader()
            writer.writerows(failures)
        failure_report = {
            "schema_version": 1,
            "quality_gate_passed": False,
            "config": str(Path(config).expanduser().resolve()),
            "config_sha256": sha256_file(Path(config)),
            "input_dir": str(input_dir.expanduser().resolve()),
            "images_checked": len(paths),
            "failed_images": len(failures),
            "failed_filenames": [str(row["filename"]) for row in failures],
            "thresholds": thresholds,
            "background_subtraction": False,
            "raw_capture_fraction_definition": (
                "sum of raw canonical uint8 codes in four detector windows divided "
                "by sum of raw codes in the full 478x478 frame; diagnostic only, "
                "not calibrated optical efficiency"
            ),
            "failure_csv": str(failure_csv),
            "failure_csv_sha256": sha256_file(failure_csv),
        }
        failure_report_path = output_dir / "routing_quality_report.json"
        failure_report_path.write_text(
            json.dumps(failure_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        preview_names = ", ".join(str(row["filename"]) for row in failures[:8])
        suffix = " ..." if len(failures) > 8 else ""
        raise RuntimeError(
            f"Router CCD quality gate rejected {len(failures)}/{len(paths)} frames: "
            f"{preview_names}{suffix}. See {failure_report_path}"
        )

    csv_path = output_dir / "routing.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with Image.open(paths[0]) as source:
        preview = source.convert("RGB")
    draw = ImageDraw.Draw(preview)
    index = 0
    for top, bottom in intervals:
        for left, right in intervals:
            draw.rectangle((left, top, right - 1, bottom - 1), outline="red", width=3)
            draw.text((left + 3, top + 3), str(index), fill="red")
            index += 1
    preview_path = output_dir / "first_ccd_detector_overlay.png"
    preview.save(preview_path)
    report = {
        "schema_version": 1,
        "config": str(Path(config).expanduser().resolve()),
        "config_sha256": sha256_file(Path(config)),
        "input_dir": str(input_dir.expanduser().resolve()),
        "images": len(rows),
        "row_order": (
            "explicit_manifest_order" if expected_stems is not None else "filename_sorted"
        ),
        "canonical_ccd_size": settings.active_size,
        "top_k": settings.top_k,
        "weight_normalization": settings.router_weight_normalization,
        "score_normalization": settings.optical_router_score_normalization,
        "background_subtraction": False,
        "quality_gate_passed": True,
        "quality_thresholds": thresholds,
        "routing_csv": str(csv_path),
        "routing_csv_sha256": sha256_file(csv_path),
        "preview": str(preview_path),
        "mean_raw_capture_fraction": float(
            np.mean([float(row["raw_capture_fraction"]) for row in rows])
        ),
        "raw_capture_fraction_definition": (
            "sum of raw canonical uint8 codes in four detector windows divided by "
            "sum of raw codes in the full 478x478 frame; diagnostic only, not "
            "calibrated optical efficiency"
        ),
    }
    (output_dir / "routing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Score canonical router CCD frames")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = score_directory(args.config, args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
