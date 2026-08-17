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
        left = (width - active.width) // 2
        top = (height - active.height) // 2
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
                    f"{left},{top},{left + active.width},{top + active.height}"
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
        "schema_version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(rows),
        "slm_size_wh": [width, height],
        "scale_factor": scale_factor,
        "rule": "logical pixel nearest-repeat then exact centered zero padding",
    }
    (output_dir / "reconstruction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slm-width", type=int, required=True)
    parser.add_argument("--slm-height", type=int, required=True)
    parser.add_argument("--scale-factor", type=int, default=2)
    args = parser.parse_args()
    report = reconstruct_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        slm_size_wh=(args.slm_width, args.slm_height),
        scale_factor=args.scale_factor,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
