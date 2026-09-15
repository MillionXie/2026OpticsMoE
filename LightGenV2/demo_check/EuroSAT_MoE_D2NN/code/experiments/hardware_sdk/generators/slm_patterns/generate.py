from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .settings import Settings, load_settings


def centered_canvas(active: np.ndarray, size_wh: tuple[int, int]) -> tuple[np.ndarray, list[int]]:
    if active.ndim != 2 or active.dtype != np.uint8:
        raise ValueError("active pattern must be a 2-D uint8 array")
    width, height = size_wh
    ah, aw = active.shape
    if aw > width or ah > height:
        raise ValueError(f"Active pattern {aw}x{ah} exceeds canvas {width}x{height}")
    x0 = (width - aw) // 2
    y0 = (height - ah) // 2
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[y0 : y0 + ah, x0 : x0 + aw] = active
    return canvas, [x0, y0, x0 + aw, y0 + ah]


def checkerboard(size: int, block: int) -> np.ndarray:
    y, x = np.indices((size, size))
    return (((x // block + y // block) % 2) * 255).astype(np.uint8)


def letter_mask(size: int, letter: str) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    font_size = int(size * 0.78)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), letter, font=font)
    x = (size - (box[2] - box[0])) // 2 - box[0]
    y = (size - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), letter, fill=255, font=font)
    return np.asarray(image, dtype=np.uint8)


def crosshair(size: int, width: int | None = None) -> np.ndarray:
    width = max(1, int(width or round(size * 0.012)))
    result = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    result[center - width : center + width + 1, :] = 255
    result[:, center - width : center + width + 1] = 255
    return result


def circular_aperture(size: int, radius_fraction: float = 0.42) -> np.ndarray:
    y, x = np.indices((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    radius = radius_fraction * size
    return (((x - center) ** 2 + (y - center) ** 2 <= radius**2) * 255).astype(np.uint8)


def lens_phase(size: int, pixel_pitch_um: float, wavelength_nm: float, focal_cm: float) -> np.ndarray:
    coordinate = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * pixel_pitch_um * 1e-6
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    wavelength = wavelength_nm * 1e-9
    focal_length = focal_cm * 1e-2
    phase = np.mod(-math.pi * (x * x + y * y) / (wavelength * focal_length), 2.0 * math.pi)
    return np.rint(phase * (255.0 / (2.0 * math.pi))).astype(np.uint8)


def _save(active: np.ndarray, path: Path, canvas_wh: tuple[int, int], *, flip_vertical: bool) -> dict[str, Any]:
    transformed = np.flipud(active).copy() if flip_vertical else active
    canvas, bounds = centered_canvas(transformed, canvas_wh)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(path, format="BMP")
    with Image.open(path) as check:
        if check.mode != "L" or check.size != canvas_wh:
            raise RuntimeError(f"Invalid calibration BMP {path}: mode={check.mode}, size={check.size}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canvas_size_wh": list(canvas_wh),
        "active_size_hw": list(active.shape),
        "active_bounds_xyxy": bounds,
        "flip_vertical_before_export": flip_vertical,
        "min_uint8": int(active.min()),
        "max_uint8": int(active.max()),
    }


def generate(settings: Settings) -> dict[str, Any]:
    size = settings.active_size
    amplitude_patterns = {
        "uniform_black": np.zeros((size, size), dtype=np.uint8),
        "uniform_white": np.full((size, size), 255, dtype=np.uint8),
        "uniform_gray_128": np.full((size, size), 128, dtype=np.uint8),
        "checkerboard": checkerboard(size, settings.checkerboard_block_px),
        "crosshair": crosshair(size),
        f"letter_{settings.letter.upper()}": letter_mask(size, settings.letter.upper()),
        "circular_aperture": circular_aperture(size),
    }
    phase_patterns = {
        "flat_phase_0": np.zeros((size, size), dtype=np.uint8),
        "flat_phase_pi": np.full((size, size), 128, dtype=np.uint8),
        "checkerboard_0_pi": checkerboard(size, settings.checkerboard_block_px) // 2,
        "crosshair_pi": crosshair(size) // 2,
        f"letter_{settings.letter.upper()}_pi": letter_mask(size, settings.letter.upper()) // 2,
    }
    for focal_cm in settings.lens_focal_lengths_cm:
        label = f"lens_{focal_cm:g}cm".replace(".", "p")
        phase_patterns[label] = lens_phase(
            size, settings.pixel_pitch_um, settings.wavelength_nm, focal_cm
        )

    files: dict[str, Any] = {"amplitude": {}, "phase": {}}
    for name, pattern in amplitude_patterns.items():
        files["amplitude"][name] = _save(
            pattern,
            settings.output_dir / "amplitude_bmp" / f"amplitude_{name}_1920x1080.bmp",
            settings.amplitude_size_wh,
            flip_vertical=False,
        )
    for name, pattern in phase_patterns.items():
        files["phase"][name] = _save(
            pattern,
            settings.output_dir / "phase_bmp" / f"phase_{name}_1920x1200.bmp",
            settings.phase_size_wh,
            flip_vertical=settings.phase_flip_vertical,
        )
    manifest = {
        "schema_version": 1,
        "active_size": size,
        "active_physical_size_mm": size * settings.pixel_pitch_um / 1000.0,
        "pixel_pitch_um": settings.pixel_pitch_um,
        "wavelength_nm": settings.wavelength_nm,
        "phase_flip_vertical": settings.phase_flip_vertical,
        "files": files,
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate centered amplitude/phase SLM calibration BMPs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    report = generate(settings)
    print(
        f"Generated {len(report['files']['amplitude'])} amplitude and "
        f"{len(report['files']['phase'])} phase BMPs under {settings.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
