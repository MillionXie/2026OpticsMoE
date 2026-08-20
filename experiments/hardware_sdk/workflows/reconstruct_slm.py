"""Rebuild full SLM BMP frames from compact logical active-region PNGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_active_amplitude(
    value: np.ndarray, percentile: float = 99.5
) -> np.ndarray:
    encoded, _ = encode_active_amplitude_with_metadata(value, percentile)
    return encoded


def encode_active_amplitude_with_metadata(
    value: np.ndarray, percentile: float = 99.5
) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Active amplitude must be a finite 2-D array")
    array = np.clip(array, 0.0, None)
    positive = array[array > 0]
    scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
    encoded = np.rint(
        np.clip(array / max(scale, 1.0e-8), 0, 1) * 255
    ).astype(np.uint8)
    return encoded, {
        "percentile": float(percentile),
        "scale": scale,
        "source_min": float(array.min()),
        "source_max": float(array.max()),
    }


def encode_active_phase(value: np.ndarray) -> np.ndarray:
    phase = np.asarray(value, dtype=np.float32)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("Active phase must be a finite 2-D array")
    wrapped = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
    return np.floor(wrapped * 256.0).clip(0, 255).astype(np.uint8)


def save_active_png(value: np.ndarray, path: Path) -> None:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError("Compact SLM payload must be a 2-D uint8 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path, optimize=True)


def physical_pitch_nearest(
    value: np.ndarray,
    *,
    logical_pixel_pitch_um: float,
    slm_pixel_pitch_um: float,
) -> np.ndarray:
    """Rasterize logical pixels onto a native SLM grid by physical position.

    Pixel centers are aligned about the same continuous optical axis.  This is
    deliberately not an image-content resize: every native SLM pixel receives
    exactly one logical value, phase wrap boundaries are never interpolated,
    and non-integer pitch ratios cannot accumulate a one-sided placement error.
    """

    source = np.asarray(value)
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ValueError("Physical-pitch raster input must be a 2-D uint8 array")
    logical_pitch = float(logical_pixel_pitch_um)
    native_pitch = float(slm_pixel_pitch_um)
    if logical_pitch <= 0.0 or native_pitch <= 0.0:
        raise ValueError("Logical and SLM pixel pitches must be positive")
    source_height, source_width = source.shape
    output_width = int(round(source_width * logical_pitch / native_pitch))
    output_height = int(round(source_height * logical_pitch / native_pitch))
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Physical-pitch raster produced an empty active area")

    def source_indexes(source_size: int, output_size: int) -> np.ndarray:
        # Native pixel-center coordinate relative to the shared optical axis.
        physical = (
            np.arange(output_size, dtype=np.float64) + 0.5 - output_size / 2.0
        ) * native_pitch
        indexes = np.floor(physical / logical_pitch + source_size / 2.0)
        return np.clip(indexes.astype(np.int64), 0, source_size - 1)

    y_indexes = source_indexes(source_height, output_height)
    x_indexes = source_indexes(source_width, output_width)
    return source[np.ix_(y_indexes, x_indexes)]


def place_at_center(
    active: Image.Image,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float] | None,
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[float, float]]:
    width, height = map(int, slm_size_wh)
    if active.width > width or active.height > height:
        raise RuntimeError(f"Active payload {active.size} exceeds SLM {(width, height)}")
    requested_center = (
        (width / 2.0, height / 2.0)
        if center_xy is None
        else (float(center_xy[0]), float(center_xy[1]))
    )
    if center_xy is None:
        # Preserve the legacy placement exactly when odd dimensions make the
        # geometric center fall between two realizable integer placements.
        left = (width - active.width) // 2
        top = (height - active.height) // 2
    else:
        left = int(math.floor(requested_center[0] - active.width / 2.0 + 0.5))
        top = int(math.floor(requested_center[1] - active.height / 2.0 + 0.5))
    right = left + active.width
    bottom = top + active.height
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"Active payload {active.size} centered at {requested_center} has "
            f"bounds {(left, top, right, bottom)}, outside SLM {(width, height)}"
        )
    actual_center = (left + active.width / 2.0, top + active.height / 2.0)
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(active, (left, top))
    return canvas, (left, top, right, bottom), actual_center


def reconstruct_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    slm_size_wh: tuple[int, int],
    scale_factor: int | None = 2,
    center_xy: tuple[float, float] | None = None,
    logical_pixel_pitch_um: float | None = None,
    slm_pixel_pitch_um: float | None = None,
) -> dict[str, object]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Compact input and reconstructed output directories must differ")
    paths = sorted(input_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No compact SLM PNGs found in {input_dir}")
    width, height = map(int, slm_size_wh)
    if min(width, height) <= 0:
        raise ValueError("SLM dimensions must be positive")
    pitch_mapping = (
        logical_pixel_pitch_um is not None or slm_pixel_pitch_um is not None
    )
    if pitch_mapping and (
        logical_pixel_pitch_um is None or slm_pixel_pitch_um is None
    ):
        raise ValueError(
            "logical_pixel_pitch_um and slm_pixel_pitch_um must be provided together"
        )
    if pitch_mapping and scale_factor is not None:
        raise ValueError("Physical-pitch mapping and integer scale_factor are exclusive")
    if not pitch_mapping and (scale_factor is None or scale_factor <= 0):
        raise ValueError("scale_factor must be positive for integer-repeat mapping")
    mapping_mode = "physical_pitch_nearest" if pitch_mapping else "integer_repeat"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(paths, 1):
        with Image.open(source) as image:
            if image.mode != "L":
                raise RuntimeError(f"Compact SLM payload must be mode L: {source}")
            logical = image.copy()
        if pitch_mapping:
            active_array = physical_pitch_nearest(
                np.asarray(logical),
                logical_pixel_pitch_um=float(logical_pixel_pitch_um),
                slm_pixel_pitch_um=float(slm_pixel_pitch_um),
            )
            active = Image.fromarray(active_array, mode="L")
        else:
            active = logical.resize(
                (logical.width * int(scale_factor), logical.height * int(scale_factor)),
                resample=Image.Resampling.NEAREST,
            )
        canvas, bounds, actual_center = place_at_center(
            active, slm_size_wh=(width, height), center_xy=center_xy
        )
        left, top, right, bottom = bounds
        actual_center_x, actual_center_y = actual_center
        destination = output_dir / f"{source.stem}.bmp"
        canvas.save(destination, format="BMP")
        rows.append(
            {
                "order": index - 1,
                "basename": source.stem,
                "source_png": source.name,
                "output_bmp": destination.name,
                "source_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
                "logical_size_wh": f"{logical.width},{logical.height}",
                "active_size_wh": f"{active.width},{active.height}",
                "slm_size_wh": f"{width},{height}",
                "active_bounds_xyxy": (
                    f"{left},{top},{right},{bottom}"
                ),
                "active_center_xy": f"{actual_center_x:g},{actual_center_y:g}",
                "canvas_center_offset_xy": (
                    f"{actual_center_x - width / 2.0:g},"
                    f"{actual_center_y - height / 2.0:g}"
                ),
                "mapping_mode": mapping_mode,
                "scale_factor": "" if scale_factor is None else scale_factor,
                "logical_pixel_pitch_um": (
                    "" if logical_pixel_pitch_um is None else logical_pixel_pitch_um
                ),
                "slm_pixel_pitch_um": (
                    "" if slm_pixel_pitch_um is None else slm_pixel_pitch_um
                ),
                "physical_ratio": (
                    ""
                    if not pitch_mapping
                    else float(logical_pixel_pitch_um) / float(slm_pixel_pitch_um)
                ),
            }
        )
    with (output_dir / "reconstruction_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 2,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(rows),
        "slm_size_wh": [width, height],
        "mapping_mode": mapping_mode,
        "scale_factor": scale_factor,
        "logical_pixel_pitch_um": logical_pixel_pitch_um,
        "slm_pixel_pitch_um": slm_pixel_pitch_um,
        "physical_ratio": (
            None
            if not pitch_mapping
            else float(logical_pixel_pitch_um) / float(slm_pixel_pitch_um)
        ),
        "requested_center_xy": (
            None if center_xy is None else [float(center_xy[0]), float(center_xy[1])]
        ),
        "canvas_geometric_center_xy": [width / 2.0, height / 2.0],
        "active_center_xy": [actual_center_x, actual_center_y],
        "center_offset_xy": [
            actual_center_x - width / 2.0,
            actual_center_y - height / 2.0,
        ],
        "coordinate_convention": (
            "x increases right, y increases down, origin is the top-left "
            "canvas boundary"
        ),
        "rule": (
            "physical-coordinate nearest raster then zero-pad at configured center"
            if pitch_mapping
            else "logical pixel nearest-repeat then zero-pad at configured center"
        ),
    }
    (output_dir / "reconstruction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def resolve_reconstruction_layout(
    *,
    stage_dir: Path | None,
    payload: str | None,
    input_dir: Path | None,
    output_dir: Path | None,
    slm_width: int | None,
    slm_height: int | None,
) -> tuple[Path, Path, tuple[int, int]]:
    """Resolve either the safe stage shortcut or the legacy explicit layout."""

    if stage_dir is not None:
        if payload not in {"amplitude", "phase"}:
            raise ValueError(
                "--stage-dir requires --payload amplitude or --payload phase"
            )
        if input_dir is not None or output_dir is not None:
            raise ValueError(
                "Do not combine --stage-dir with --input-dir/--output-dir"
            )
        defaults = {
            "amplitude": (1920, 1080),
            "phase": (1920, 1200),
        }
        default_width, default_height = defaults[payload]
        root = stage_dir.expanduser().resolve()
        return (
            root / f"compact_{payload}",
            root / f"{payload}_to_play",
            (
                default_width if slm_width is None else int(slm_width),
                default_height if slm_height is None else int(slm_height),
            ),
        )
    if payload is not None:
        raise ValueError("--payload is only valid together with --stage-dir")
    missing = [
        name
        for name, value in (
            ("--input-dir", input_dir),
            ("--output-dir", output_dir),
            ("--slm-width", slm_width),
            ("--slm-height", slm_height),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Explicit mode is missing "
            + ", ".join(missing)
            + "; alternatively use --stage-dir with --payload"
        )
    return input_dir, output_dir, (int(slm_width), int(slm_height))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-dir",
        help=(
            "Hardware stage directory containing compact_amplitude/ or "
            "compact_phase/; avoids ambiguous repository-root relative paths"
        ),
    )
    parser.add_argument("--payload", choices=("amplitude", "phase"))
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--slm-width", type=int)
    parser.add_argument("--slm-height", type=int)
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=None,
        help="Legacy integer nearest-repeat factor (default: 2 when pitches are omitted)",
    )
    parser.add_argument(
        "--logical-pixel-pitch-um",
        type=float,
        help="Logical payload pitch for physical-coordinate nearest rasterization",
    )
    parser.add_argument(
        "--slm-pixel-pitch-um",
        type=float,
        help="Native SLM pitch; must accompany --logical-pixel-pitch-um",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        help="Active-region center x in full-SLM pixels (default: geometric center)",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        help="Active-region center y in full-SLM pixels (default: geometric center)",
    )
    args = parser.parse_args()
    if (args.center_x is None) != (args.center_y is None):
        parser.error("--center-x and --center-y must be provided together")
    if (args.logical_pixel_pitch_um is None) != (args.slm_pixel_pitch_um is None):
        parser.error(
            "--logical-pixel-pitch-um and --slm-pixel-pitch-um must be provided together"
        )
    if args.logical_pixel_pitch_um is not None and args.scale_factor is not None:
        parser.error("Do not combine physical pixel pitches with --scale-factor")
    try:
        input_dir, output_dir, slm_size = resolve_reconstruction_layout(
            stage_dir=(None if args.stage_dir is None else Path(args.stage_dir)),
            payload=args.payload,
            input_dir=(None if args.input_dir is None else Path(args.input_dir)),
            output_dir=(None if args.output_dir is None else Path(args.output_dir)),
            slm_width=args.slm_width,
            slm_height=args.slm_height,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = reconstruct_directory(
        input_dir,
        output_dir,
        slm_size_wh=slm_size,
        scale_factor=(
            None
            if args.logical_pixel_pitch_um is not None
            else (2 if args.scale_factor is None else args.scale_factor)
        ),
        center_xy=(
            None
            if args.center_x is None
            else (args.center_x, args.center_y)
        ),
        logical_pixel_pitch_um=args.logical_pixel_pitch_um,
        slm_pixel_pitch_um=args.slm_pixel_pitch_um,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
