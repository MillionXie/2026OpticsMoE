"""Rebuild full SLM BMP frames from compact logical active-region PNGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def reconstruct_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    slm_size_wh: tuple[int, int],
    scale_factor: int = 2,
    center_xy: tuple[int, int] | None = None,
) -> dict[str, object]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Compact input and reconstructed output directories must differ")
    paths = sorted(input_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No compact SLM PNGs found in {input_dir}")
    width, height = map(int, slm_size_wh)
    if min(width, height, scale_factor) <= 0:
        raise ValueError("SLM dimensions and scale_factor must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(paths, 1):
        with Image.open(source) as image:
            if image.mode != "L":
                raise RuntimeError(f"Compact SLM payload must be mode L: {source}")
            logical = image.copy()
        active = logical.resize(
            (logical.width * scale_factor, logical.height * scale_factor),
            resample=Image.Resampling.NEAREST,
        )
        if active.width > width or active.height > height:
            raise RuntimeError(f"Active payload {active.size} exceeds SLM {(width, height)}")
        if center_xy is None:
            # Preserve the original placement exactly for backward
            # compatibility, including odd canvas/payload combinations.
            left = (width - active.width) // 2
            top = (height - active.height) // 2
        else:
            center_x, center_y = map(int, center_xy)
            left = center_x - active.width // 2
            top = center_y - active.height // 2
        right = left + active.width
        bottom = top + active.height
        if left < 0 or top < 0 or right > width or bottom > height:
            requested = "geometric center" if center_xy is None else center_xy
            raise ValueError(
                f"Active payload {active.size} centered at {requested} has "
                f"bounds {(left, top, right, bottom)}, outside SLM {(width, height)}"
            )
        actual_center_x = left + active.width / 2.0
        actual_center_y = top + active.height / 2.0
        canvas = Image.new("L", (width, height), 0)
        canvas.paste(active, (left, top))
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
                "scale_factor": scale_factor,
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
        "scale_factor": scale_factor,
        "requested_center_xy": (
            None if center_xy is None else [int(center_xy[0]), int(center_xy[1])]
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
            "logical pixel nearest-repeat then zero-pad at configured center"
            if center_xy is not None
            else "logical pixel nearest-repeat then exact geometric-center zero padding"
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
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument(
        "--center-x",
        type=int,
        help="Active-region center x in full-SLM pixels (default: geometric center)",
    )
    parser.add_argument(
        "--center-y",
        type=int,
        help="Active-region center y in full-SLM pixels (default: geometric center)",
    )
    args = parser.parse_args()
    if (args.center_x is None) != (args.center_y is None):
        parser.error("--center-x and --center-y must be provided together")
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
        scale_factor=args.scale_factor,
        center_xy=(
            None
            if args.center_x is None
            else (args.center_x, args.center_y)
        ),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
