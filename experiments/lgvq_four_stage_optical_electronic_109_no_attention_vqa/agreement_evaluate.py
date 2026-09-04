"""Compare one simulated optical pass with its measured CCD captures.

The comparison deliberately does *not* estimate a background, search for a
flip/translation, or apply per-frame min/max normalization.  Geometry must
already have been canonicalized by the detector homography during capture.
Both arrays receive the same two numerical operations before metrics:

    clamp intensity to non-negative values -> divide by that frame's mean

The output therefore measures optical agreement rather than how well an
after-the-fact image registration can make two frames look alike.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .hardware_contract import OPTICAL_PASSES


EPSILON = 1.0e-8
EXPECTED_SIZE = (478, 478)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_by_key(
    rows: list[dict[str, str]], *, key_column: str, label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(key_column, "")).strip()
        if not key:
            raise RuntimeError(f"{label} contains an empty {key_column}")
        if key in result:
            raise RuntimeError(f"{label} contains duplicate key {key!r}")
        result[key] = row
    if not result:
        raise RuntimeError(f"{label} is empty")
    return result


def _csv_bool(value: Any, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"{label} must be a CSV boolean, got {value!r}")


def _manifest_contract(stage_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    amplitude_path = stage_dir / "amplitude_manifest.csv"
    amplitude = _unique_by_key(
        _read_csv(amplitude_path), key_column="key", label="amplitude_manifest.csv"
    )
    for key, row in amplitude.items():
        name = Path(str(row.get("amplitude_file", ""))).name
        if name != f"{key}.png":
            raise RuntimeError(
                f"Amplitude manifest filename/key mismatch for {key!r}: {name!r}"
            )

    capture_path = stage_dir / "acquisition_logs" / "capture_manifest.csv"
    capture_rows = _read_csv(capture_path)
    capture: dict[str, dict[str, str]] = {}
    for row in capture_rows:
        amplitude_name = Path(str(row.get("amplitude_bmp", ""))).name
        if amplitude_name.lower().endswith(".bmp"):
            key = Path(amplitude_name).stem
        else:
            raise RuntimeError(
                f"capture_manifest.csv has invalid amplitude_bmp {amplitude_name!r}"
            )
        if key in capture:
            raise RuntimeError(f"capture_manifest.csv contains duplicate key {key!r}")
        capture[key] = row
    if not capture:
        raise RuntimeError("capture_manifest.csv is empty")
    return amplitude, capture


def _npz_scalar(value: np.ndarray, *, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{label} must contain exactly one scalar")
    return str(array.reshape(-1)[0])


def _load_theoretical(path: Path, *, optical_pass: str, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {"intensity", "optical_pass", "key"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path.name} is missing NPZ fields {missing}")
        observed_pass = _npz_scalar(payload["optical_pass"], label=f"{path.name}:optical_pass")
        observed_key = _npz_scalar(payload["key"], label=f"{path.name}:key")
        if observed_pass != optical_pass:
            raise RuntimeError(
                f"{path.name} optical_pass={observed_pass!r}, expected {optical_pass!r}"
            )
        if observed_key != key:
            raise RuntimeError(f"{path.name} key={observed_key!r}, expected {key!r}")
        value = np.asarray(payload["intensity"], dtype=np.float64)
    if value.shape != EXPECTED_SIZE:
        raise RuntimeError(f"{path.name} shape={value.shape}, expected {EXPECTED_SIZE}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{path.name} contains NaN/Inf")
    return value


def _load_measured(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2 or value.shape != EXPECTED_SIZE:
        raise RuntimeError(f"{path.name} shape={value.shape}, expected {EXPECTED_SIZE}")
    if value.dtype != np.uint8:
        raise RuntimeError(f"{path.name} dtype={value.dtype}, expected uint8")
    return value.astype(np.float64)


def normalize_intensity(value: np.ndarray) -> np.ndarray:
    """Apply the only metric preprocessing: non-negative clamp and mean scale."""

    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    mean = float(array.mean())
    if not math.isfinite(mean) or mean <= EPSILON:
        raise RuntimeError("Cannot mean-normalize an empty/dark intensity frame")
    return array / mean


def pearson_correlation(reference: np.ndarray, measured: np.ndarray) -> float:
    left = reference.reshape(-1) - float(reference.mean())
    right = measured.reshape(-1) - float(measured.mean())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return 1.0 if np.allclose(reference, measured, atol=EPSILON, rtol=0.0) else 0.0
    return float(np.dot(left, right) / denominator)


def structural_similarity(reference: np.ndarray, measured: np.ndarray) -> float:
    """Local Gaussian-window SSIM on the already mean-normalized intensities."""

    def blur(value: np.ndarray) -> np.ndarray:
        sigma = 1.5
        radius = int(3.5 * sigma + 0.5)
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float64)
        kernel = torch.exp(-(coordinate * coordinate) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        tensor = torch.from_numpy(np.ascontiguousarray(value)).to(torch.float64)[None, None]
        tensor = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
        tensor = F.conv2d(tensor, kernel.reshape(1, 1, 1, -1))
        tensor = F.conv2d(tensor, kernel.reshape(1, 1, -1, 1))
        return tensor[0, 0].numpy()

    data_min = min(float(reference.min()), float(measured.min()))
    data_max = max(float(reference.max()), float(measured.max()))
    data_range = max(data_max - data_min, EPSILON)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = blur(reference)
    mu_y = blur(measured)
    sigma_x = blur(reference * reference) - mu_x * mu_x
    sigma_y = blur(measured * measured) - mu_y * mu_y
    sigma_xy = blur(reference * measured) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    score = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=np.abs(denominator) > EPSILON,
    )
    return float(np.clip(score.mean(), -1.0, 1.0))


def gain_aligned_nmae(reference: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(measured.reshape(-1), measured.reshape(-1)))
    gain = 0.0 if denominator <= EPSILON else float(
        np.dot(reference.reshape(-1), measured.reshape(-1)) / denominator
    )
    gain = max(0.0, gain)
    normalizer = max(float(np.mean(np.abs(reference))), EPSILON)
    error = float(np.mean(np.abs(reference - gain * measured)) / normalizer)
    return error, gain


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_figure(
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    ranked = sorted(pairs, key=lambda item: float(metrics[item[0]]["pcc"]))
    indices = sorted(set((0, len(ranked) // 2, len(ranked) - 1)))
    selected = [ranked[index] for index in indices]
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(18.0 / 2.54, max(5.4, 4.0 * len(selected)) / 2.54),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (key, reference, measured) in enumerate(selected):
        joint_limit = max(float(np.quantile(np.concatenate((reference.ravel(), measured.ravel())), 0.995)), EPSILON)
        difference = np.abs(reference - measured)
        difference_limit = max(float(np.quantile(difference, 0.995)), EPSILON)
        axes[row, 0].imshow(reference, cmap="gray", vmin=0.0, vmax=joint_limit)
        axes[row, 1].imshow(measured, cmap="gray", vmin=0.0, vmax=joint_limit)
        axes[row, 2].imshow(difference, cmap="magma", vmin=0.0, vmax=difference_limit)
        values = metrics[key]
        axes[row, 0].set_title(f"theory | {key}")
        axes[row, 1].set_title(
            f"measured | PCC {values['pcc']:.3f}, SSIM {values['ssim']:.3f}"
        )
        axes[row, 2].set_title(f"|difference| | NMAE {values['gain_aligned_nmae']:.3f}")
        for axis in axes[row]:
            axis.set_xticks(())
            axis.set_yticks(())
    figure.suptitle(
        "Identical preprocessing: non-negative clamp + per-frame mean normalization\n"
        "Shared 99.5th-percentile display limits only; no geometric search or background subtraction",
        fontsize=7,
    )
    png = output_dir / "agreement_examples.png"
    pdf = output_dir / "agreement_examples.pdf"
    figure.savefig(png, dpi=600)
    figure.savefig(pdf)
    plt.close(figure)
    return [str(png), str(pdf)]


def evaluate_agreement(
    *, stage_dir: str | Path, optical_pass: str, output_dir: str | Path | None = None
) -> dict[str, Any]:
    if optical_pass not in OPTICAL_PASSES:
        raise ValueError(f"Unknown optical pass {optical_pass!r}")
    stage = Path(stage_dir).expanduser().resolve()
    if not stage.is_dir():
        raise FileNotFoundError(stage)
    output = (
        stage / "agreement"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    amplitude, capture = _manifest_contract(stage)

    theoretical_dir = stage / "theoretical_ccd"
    theoretical_files = sorted(theoretical_dir.glob("*.npz"))
    if not theoretical_files:
        raise FileNotFoundError(f"No theoretical CCD NPZ files found in {theoretical_dir}")
    keys = [path.stem for path in theoretical_files]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate theoretical CCD keys")
    unknown = sorted(set(keys) - set(amplitude))
    if unknown:
        raise RuntimeError(f"Theoretical CCD keys absent from amplitude manifest: {unknown[:8]}")

    measured_dir = stage / "ccd_captured"
    rows: list[dict[str, Any]] = []
    normalized_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    metric_lookup: dict[str, dict[str, Any]] = {}
    for theoretical_path in theoretical_files:
        key = theoretical_path.stem
        if key not in capture:
            raise RuntimeError(f"Theoretical key {key!r} is absent from capture manifest")
        captured_path = measured_dir / f"{key}.png"
        if not captured_path.is_file():
            raise FileNotFoundError(captured_path)
        capture_row = capture[key]
        logged_name = Path(str(capture_row.get("ccd_capture", ""))).name
        if logged_name != captured_path.name:
            raise RuntimeError(
                f"Capture manifest filename/key mismatch for {key!r}: {logged_name!r}"
            )
        expected_hash = str(capture_row.get("output_sha256", "")).strip().lower()
        actual_hash = _sha256(captured_path)
        if expected_hash != actual_hash:
            raise RuntimeError(f"Measured CCD SHA256 mismatch for {key!r}")
        if not _csv_bool(
            capture_row.get("orientation_canonicalized"),
            label=f"orientation_canonicalized for {key}",
        ):
            raise RuntimeError(f"Measured CCD for {key!r} is not homography-canonicalized")
        if str(capture_row.get("saved_frame_orientation", "")).strip() != "canonical_model_xy":
            raise RuntimeError(f"Measured CCD for {key!r} is not in canonical model orientation")
        if _csv_bool(
            capture_row.get("downstream_loader_flip_required"),
            label=f"downstream_loader_flip_required for {key}",
        ):
            raise RuntimeError(f"Measured CCD for {key!r} still requests a downstream flip")
        if _csv_bool(
            capture_row.get("background_subtraction"),
            label=f"background_subtraction for {key}",
        ):
            raise RuntimeError(f"Measured CCD for {key!r} used background subtraction")
        if _csv_bool(
            capture_row.get("per_frame_minmax_normalization"),
            label=f"per_frame_minmax_normalization for {key}",
        ):
            raise RuntimeError(f"Measured CCD for {key!r} used per-frame min/max")

        reference_raw = _load_theoretical(
            theoretical_path, optical_pass=optical_pass, key=key
        )
        measured_raw = _load_measured(captured_path)
        reference = normalize_intensity(reference_raw)
        measured = normalize_intensity(measured_raw)
        pcc = pearson_correlation(reference, measured)
        ssim = structural_similarity(reference, measured)
        nmae, gain = gain_aligned_nmae(reference, measured)
        row = {
            "key": key,
            "optical_pass": optical_pass,
            "pcc": pcc,
            "ssim": ssim,
            "gain_aligned_nmae": nmae,
            "least_squares_gain": gain,
            "theoretical_mean_before_normalization": float(np.maximum(reference_raw, 0.0).mean()),
            "measured_mean_before_normalization": float(np.maximum(measured_raw, 0.0).mean()),
            "theoretical_npz": theoretical_path.name,
            "theoretical_sha256": _sha256(theoretical_path),
            "measured_png": captured_path.name,
            "measured_sha256": actual_hash,
        }
        rows.append(row)
        metric_lookup[key] = row
        normalized_pairs.append((key, reference, measured))

    csv_path = output / "agreement_per_sample.csv"
    _write_csv(csv_path, rows)
    figures = _write_comparison_figure(normalized_pairs, metric_lookup, output)
    report: dict[str, Any] = {
        "schema_version": 1,
        "optical_pass": optical_pass,
        "sample_count": len(rows),
        "pairing": "theoretical_ccd/<key>.npz + ccd_captured/<key>.png; key must exist in amplitude and capture manifests",
        "preprocessing_applied_identically_to_both": [
            "non-negative intensity clamp",
            "divide by per-frame mean intensity",
        ],
        "forbidden_postprocessing": {
            "background_subtraction": False,
            "per_frame_minmax": False,
            "flip_or_rotation_search": False,
            "translation_or_scale_search": False,
            "nonlinear_log_for_metrics": False,
        },
        "ssim_definition": "Gaussian-window SSIM, sigma=1.5, K1=0.01, K2=0.03, joint normalized data range",
        "gain_aligned_nmae_definition": "non-negative least-squares scalar gain, MAE divided by mean absolute theoretical intensity",
        "metrics": {
            "pcc": _summary([float(row["pcc"]) for row in rows]),
            "ssim": _summary([float(row["ssim"]) for row in rows]),
            "gain_aligned_nmae": _summary(
                [float(row["gain_aligned_nmae"]) for row in rows]
            ),
        },
        "inputs": {
            "stage_dir": str(stage),
            "amplitude_manifest_sha256": _sha256(stage / "amplitude_manifest.csv"),
            "capture_manifest_sha256": _sha256(
                stage / "acquisition_logs" / "capture_manifest.csv"
            ),
        },
        "outputs": {
            "per_sample_csv": str(csv_path),
            "per_sample_csv_sha256": _sha256(csv_path),
            "figures": figures,
            "figure_sha256": {Path(path).name: _sha256(Path(path)) for path in figures},
        },
    }
    report_path = output / "agreement_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--optical-pass", required=True, choices=OPTICAL_PASSES)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report = evaluate_agreement(
        stage_dir=args.stage_dir,
        optical_pass=args.optical_pass,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
