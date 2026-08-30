"""Diagnose and visualize fixed-pattern leakage in one measured CCD frame.

This is an analysis tool, not a silent change to the network input pipeline.
It estimates an input-independent pattern only from other measured frames taken
with the same optical phase mask.  The theoretical CCD is used for reporting
metrics and an explicitly labelled shift diagnostic, never to synthesize or
subtract image content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate import (
    network_input_map,
    pair_metrics,
)


VIRIDIS_ANCHORS = np.asarray(
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        value = np.asarray(opened)
    if value.ndim == 3:
        value = value[..., :3].astype(np.float64).mean(axis=2)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError(f"CCD image must be a finite 2-D array: {path}")
    return np.clip(value, 0.0, None)


def _load_theory(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "intensity" not in payload:
            raise KeyError(f"Theory NPZ has no 'intensity' array: {path}")
        value = np.asarray(payload["intensity"], dtype=np.float32)
    if value.ndim != 2 or np.any(value < 0) or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid theoretical CCD intensity: {path}")
    return value


def _mean_normalize(value: np.ndarray) -> np.ndarray:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    return source / max(float(source.mean()), 1.0e-12)


def _linear_display(value: np.ndarray, relative_clip: float = 12.0) -> np.ndarray:
    shape = np.minimum(_mean_normalize(value), relative_clip) / relative_clip
    return np.rint(255.0 * shape).clip(0, 255).astype(np.uint8)


def _robust_display(value: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    lower, upper = np.percentile(source, [low, high])
    mapped = (source - lower) / max(float(upper - lower), 1.0e-12)
    return np.rint(255.0 * mapped).clip(0, 255).astype(np.uint8)


def _viridis(value: np.ndarray) -> np.ndarray:
    gray = np.asarray(value, dtype=np.float64).clip(0.0, 255.0) / 255.0
    position = gray * (len(VIRIDIS_ANCHORS) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(VIRIDIS_ANCHORS) - 1)
    fraction = (position - lower)[..., None]
    rgb = VIRIDIS_ANCHORS[lower] * (1.0 - fraction) + VIRIDIS_ANCHORS[upper] * fraction
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def _save_gray(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path)
    return _sha256(path)


def _save_rgb(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)
    return _sha256(path)


def _sample_files(directory: Path, target: Path, count: int) -> list[Path]:
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}
        and path.resolve() != target.resolve()
    )
    if len(candidates) < 8:
        raise RuntimeError(
            f"Need at least 8 other same-mask CCD frames for fixed-pattern estimation; "
            f"found {len(candidates)} in {directory}"
        )
    indices = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)
    return [candidates[int(index)] for index in np.unique(indices)]


def estimate_fixed_pattern(
    files: list[Path], expected_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized: list[np.ndarray] = []
    rejected: list[str] = []
    for path in files:
        value = _load_image(path)
        if value.shape != expected_shape or float(value.mean()) <= 1.0e-12:
            rejected.append(path.name)
            continue
        normalized.append(_mean_normalize(value).astype(np.float32))
    if len(normalized) < 8:
        raise RuntimeError("Too few valid same-shape background-estimation frames")
    stack = np.stack(normalized, axis=0)
    fixed = np.median(stack, axis=0).astype(np.float32)
    temporal_mad = np.median(np.abs(stack - fixed[None, ...]), axis=0).astype(np.float32)
    return fixed, temporal_mad, {
        "requested_files": len(files),
        "accepted_files": len(normalized),
        "rejected_files": rejected,
        "estimator": "per-pixel median after per-frame mean normalization",
    }


def _fit_fixed_component(
    actual_normalized: np.ndarray,
    fixed: np.ndarray,
    temporal_mad: np.ndarray,
    stable_fraction: float,
) -> tuple[float, float, np.ndarray]:
    threshold = float(np.quantile(temporal_mad, stable_fraction))
    stable = temporal_mad <= threshold
    stable &= fixed > float(np.quantile(fixed, 0.01))
    x = np.asarray(fixed[stable], dtype=np.float64)
    y = np.asarray(actual_normalized[stable], dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(max(alpha, 0.0)), float(beta), stable


def _subtract_fixed(
    actual_normalized: np.ndarray,
    fixed: np.ndarray,
    alpha: float,
    beta: float,
    strength: float,
) -> np.ndarray:
    residual = actual_normalized - strength * (alpha * fixed + beta)
    # A fixed, theory-free floor removal keeps the result nonnegative without
    # inventing negative optical intensity.
    residual -= float(np.quantile(residual, 0.005))
    return np.clip(residual, 0.0, None).astype(np.float32)


def _moe4_common_mode(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove a local-coordinate pattern shared by at least two MoE4 quadrants.

    This operation uses only the measured frame.  At every corresponding local
    pixel it averages the two dimmest of the four expert quadrants as a common
    additive component.  It is valid only for a 2x2 expert layout, not global
    optical stages.
    """

    source = np.asarray(value, dtype=np.float32)
    height, width = source.shape
    if height % 2 or width % 2:
        raise ValueError("MoE4 common-mode rejection requires even image dimensions")
    cy, cx = height // 2, width // 2
    quadrants = np.stack(
        [
            source[:cy, :cx],
            source[:cy, cx:],
            source[cy:, :cx],
            source[cy:, cx:],
        ],
        axis=0,
    )
    common = np.sort(quadrants, axis=0)[:2].mean(axis=0)
    residual = np.clip(quadrants - common[None, ...], 0.0, None)
    result = np.empty_like(source)
    result[:cy, :cx] = residual[0]
    result[:cy, cx:] = residual[1]
    result[cy:, :cx] = residual[2]
    result[cy:, cx:] = residual[3]
    return result, common


def _pcc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denominator <= 1.0e-18 else float(np.dot(x, y) / denominator)


def _overlap_for_shift(
    measured: np.ndarray, theory: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = measured.shape
    mx0, mx1 = max(dx, 0), min(width + dx, width)
    my0, my1 = max(dy, 0), min(height + dy, height)
    tx0, tx1 = max(-dx, 0), min(width - dx, width)
    ty0, ty1 = max(-dy, 0), min(height - dy, height)
    return measured[my0:my1, mx0:mx1], theory[ty0:ty1, tx0:tx1]


def best_integer_shift(
    measured: np.ndarray, theory: np.ndarray, maximum: int
) -> tuple[int, int, float]:
    # Search at half resolution for speed, then refine around the best result.
    left = measured[::2, ::2]
    right = theory[::2, ::2]
    coarse_limit = int(math.ceil(maximum / 2))
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-coarse_limit, coarse_limit + 1):
        for dx in range(-coarse_limit, coarse_limit + 1):
            x, y = _overlap_for_shift(left, right, dx, dy)
            candidates.append((_pcc(x, y), dx * 2, dy * 2))
    _, coarse_dx, coarse_dy = max(candidates)
    refined: list[tuple[float, int, int]] = []
    for dy in range(max(-maximum, coarse_dy - 2), min(maximum, coarse_dy + 2) + 1):
        for dx in range(max(-maximum, coarse_dx - 2), min(maximum, coarse_dx + 2) + 1):
            x, y = _overlap_for_shift(measured, theory, dx, dy)
            refined.append((_pcc(x, y), dx, dy))
    score, dx, dy = max(refined)
    return dx, dy, score


def _shift_without_wrap(value: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.zeros_like(value)
    height, width = value.shape
    rx0, rx1 = max(dx, 0), min(width + dx, width)
    ry0, ry1 = max(dy, 0), min(height + dy, height)
    sx0, sx1 = max(-dx, 0), min(width - dx, width)
    sy0, sy1 = max(-dy, 0), min(height - dy, height)
    result[ry0:ry1, rx0:rx1] = value[sy0:sy1, sx0:sx1]
    return result


def _metrics(measured: np.ndarray, theory: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in ("linear", "network_input"):
        values = pair_metrics(
            measured,
            theory,
            domain=domain,
            energy_gain=1.0,
            saturation_value=0.0,
            signal_energy_fraction=0.95,
            relative_clip=12.0,
            log_compression=1.0,
        )
        result[domain] = values
    return result


def _quadrant_means(value: np.ndarray) -> dict[str, float]:
    source = _mean_normalize(value)
    height, width = source.shape
    cy, cx = height // 2, width // 2
    return {
        "top_left": float(source[:cy, :cx].mean()),
        "top_right": float(source[:cy, cx:].mean()),
        "bottom_left": float(source[cy:, :cx].mean()),
        "bottom_right": float(source[cy:, cx:].mean()),
    }


def _contact_sheet(items: list[tuple[str, Path]], destination: Path) -> None:
    columns, tile, label_height = 3, 478, 34
    rows = int(math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + tile + 8), label, fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def run(
    actual_file: Path,
    theory_file: Path,
    background_dir: Path,
    output_dir: Path,
    *,
    background_samples: int,
    stable_fraction: float,
    maximum_shift: int,
) -> dict[str, Any]:
    actual = _load_image(actual_file)
    theory = _load_theory(theory_file)
    if actual.shape != theory.shape:
        raise RuntimeError(f"Shape mismatch: actual={actual.shape}, theory={theory.shape}")
    files = _sample_files(background_dir, actual_file, background_samples)
    fixed, temporal_mad, estimation = estimate_fixed_pattern(files, actual.shape)
    actual_normalized = _mean_normalize(actual)
    alpha, beta, stable = _fit_fixed_component(
        actual_normalized, fixed, temporal_mad, stable_fraction
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actual_file, output_dir / "01_actual_raw_unchanged.png")
    theory_linear = output_dir / "00_theory_linear_same_display.png"
    theory_color = output_dir / "00_theory_linear_color.png"
    fixed_color = output_dir / "02_estimated_fixed_pattern_color.png"
    _save_gray(theory_linear, _linear_display(theory))
    _save_rgb(theory_color, _viridis(_linear_display(theory)))
    _save_rgb(fixed_color, _viridis(_robust_display(fixed)))
    _save_gray(output_dir / "02_estimated_fixed_pattern_gray.png", _robust_display(fixed))
    _save_gray(
        output_dir / "02_stable_fit_pixels.png", np.where(stable, 255, 0).astype(np.uint8)
    )

    variants: dict[str, Any] = {}
    sheet_items: list[tuple[str, Path]] = [
        ("Theory: same linear display", theory_color),
        ("Actual: raw unchanged", output_dir / "01_actual_raw_unchanged.png"),
        ("Estimated fixed pattern", fixed_color),
    ]
    best_name = ""
    best_score = -float("inf")
    best_corrected: np.ndarray | None = None
    for strength in (0.50, 0.75, 1.00):
        name = f"strength_{strength:.2f}".replace(".", "p")
        corrected = _subtract_fixed(actual_normalized, fixed, alpha, beta, strength)
        linear_path = output_dir / f"03_{name}_linear_color.png"
        enhanced_path = output_dir / f"04_{name}_display_enhanced.png"
        network_path = output_dir / f"05_{name}_network_input_color.png"
        _save_rgb(linear_path, _viridis(_linear_display(corrected)))
        _save_gray(enhanced_path, _robust_display(corrected))
        network = network_input_map(corrected, relative_clip=12.0, log_compression=1.0)
        _save_rgb(network_path, _viridis(_robust_display(network)))
        metrics = _metrics(corrected, theory)
        score = float(metrics["network_input"]["pcc_full"] or -1.0)
        variants[name] = {
            "subtraction_strength": strength,
            "metrics_without_shift": metrics,
            "quadrant_means": _quadrant_means(corrected),
            "linear_color_png": linear_path.name,
            "display_enhanced_png": enhanced_path.name,
            "network_input_color_png": network_path.name,
        }
        sheet_items.append((f"Corrected {strength:.0%}: linear", linear_path))
        if score > best_score:
            best_score, best_name, best_corrected = score, name, corrected

    common_inputs = {
        "moe4_common_raw": actual_normalized,
        "moe4_common_after_fixed_0p50": _subtract_fixed(
            actual_normalized, fixed, alpha, beta, 0.50
        ),
    }
    for name, common_input in common_inputs.items():
        corrected, common = _moe4_common_mode(common_input)
        linear_path = output_dir / f"08_{name}_linear_color.png"
        enhanced_path = output_dir / f"09_{name}_display_enhanced.png"
        common_path = output_dir / f"10_{name}_estimated_common_pattern.png"
        _save_rgb(linear_path, _viridis(_linear_display(corrected)))
        _save_gray(enhanced_path, _robust_display(corrected))
        _save_rgb(common_path, _viridis(_robust_display(common)))
        metrics = _metrics(corrected, theory)
        score = float(metrics["network_input"]["pcc_full"] or -1.0)
        variants[name] = {
            "method": "MoE4 local-coordinate common-mode rejection",
            "metrics_without_shift": metrics,
            "quadrant_means": _quadrant_means(corrected),
            "linear_color_png": linear_path.name,
            "display_enhanced_png": enhanced_path.name,
            "estimated_common_pattern_png": common_path.name,
        }
        sheet_items.append((name, linear_path))
        if score > best_score:
            best_score, best_name, best_corrected = score, name, corrected

    assert best_corrected is not None
    best_source = output_dir / variants[best_name]["linear_color_png"]
    shutil.copy2(best_source, output_dir / "OPEN_THIS_BEST_DIAGNOSTIC.png")
    dx, dy, shifted_pcc = best_integer_shift(best_corrected, theory, maximum_shift)
    aligned = _shift_without_wrap(best_corrected, dx, dy)
    aligned_path = output_dir / "06_best_variant_theory_assisted_shift_diagnostic.png"
    _save_rgb(aligned_path, _viridis(_linear_display(aligned)))
    theory_shape = _robust_display(theory).astype(np.float64) / 255.0
    actual_shape = _robust_display(aligned).astype(np.float64) / 255.0
    overlay = np.stack(
        [theory_shape, actual_shape, actual_shape], axis=-1
    )
    overlay_path = output_dir / "07_overlay_theory_red_actual_cyan.png"
    _save_rgb(overlay_path, np.rint(255.0 * overlay).clip(0, 255).astype(np.uint8))
    sheet_items.extend(
        [
            ("Best corrected + diagnostic shift", aligned_path),
            ("Overlay: theory red, actual cyan", overlay_path),
        ]
    )
    _contact_sheet(sheet_items, output_dir / "OPEN_ME_COMPARISON.png")

    report = {
        "schema_version": 1,
        "purpose": "diagnostic visualization; not automatically used by the network",
        "actual_file": str(actual_file.resolve()),
        "actual_sha256": _sha256(actual_file),
        "theory_file": str(theory_file.resolve()),
        "theory_sha256": _sha256(theory_file),
        "shape_hw": list(actual.shape),
        "background_estimation": estimation,
        "background_directory": str(background_dir.resolve()),
        "background_fit": {
            "stable_fraction_requested": stable_fraction,
            "stable_pixels": int(stable.sum()),
            "alpha": alpha,
            "beta": beta,
        },
        "raw_metrics": _metrics(actual, theory),
        "raw_quadrant_means": _quadrant_means(actual),
        "theory_quadrant_means": _quadrant_means(theory),
        "variants": variants,
        "best_variant_on_this_pair_only": best_name,
        "best_diagnostic_visualization": "OPEN_THIS_BEST_DIAGNOSTIC.png",
        "theory_assisted_shift_diagnostic": {
            "warning": "Do not optimize this independently per inference frame",
            "dx_px": dx,
            "dy_px": dy,
            "overlap_pcc": shifted_pcc,
            "metrics": _metrics(aligned, theory),
        },
        "recommended_next_capture": [
            "laser-off CCD dark frames",
            "amplitude-SLM gray=0 with the same phase mask",
            "uniform amplitude flat frames with the same phase mask",
        ],
        "scientific_limit": (
            "A single CCD frame cannot uniquely separate true signal from additive leakage. "
            "This result estimates only the component invariant across other same-mask frames."
        ),
    }
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-file", required=True, type=Path)
    parser.add_argument("--theory-file", required=True, type=Path)
    parser.add_argument("--background-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background-samples", type=int, default=128)
    parser.add_argument("--stable-fraction", type=float, default=0.35)
    parser.add_argument("--maximum-shift", type=int, default=16)
    args = parser.parse_args()
    if args.background_samples < 8:
        parser.error("--background-samples must be at least 8")
    if not 0.05 <= args.stable_fraction <= 0.80:
        parser.error("--stable-fraction must be between 0.05 and 0.80")
    if not 0 <= args.maximum_shift <= 64:
        parser.error("--maximum-shift must be between 0 and 64")
    report = run(
        args.actual_file,
        args.theory_file,
        args.background_dir,
        args.output_dir,
        background_samples=args.background_samples,
        stable_fraction=args.stable_fraction,
        maximum_shift=args.maximum_shift,
    )
    summary = {
        "status": "complete",
        "output_dir": str(args.output_dir.resolve()),
        "raw_network_pcc": report["raw_metrics"]["network_input"]["pcc_full"],
        "best_variant": report["best_variant_on_this_pair_only"],
        "best_network_pcc": report["variants"][report["best_variant_on_this_pair_only"]][
            "metrics_without_shift"
        ]["network_input"]["pcc_full"],
        "diagnostic_shift": report["theory_assisted_shift_diagnostic"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
