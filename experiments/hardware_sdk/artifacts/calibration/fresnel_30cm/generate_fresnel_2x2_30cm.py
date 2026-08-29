from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_NAME = "fresnel_2x2_f300mm_w350_center980_590_1920x1200.bmp"

SLM_WIDTH = 1920
SLM_HEIGHT = 1200
PHASE_CENTER_X = 980.0
PHASE_CENTER_Y = 590.0
ARRAY_ROWS = 2
ARRAY_COLS = 2
WINDOW = 350
FOCAL_LENGTH_M = 0.30
PIXEL_PITCH_M = 8.0e-6
WAVELENGTH_M = 532.0e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if WINDOW % 2:
        raise ValueError("WINDOW must be even")
    if not PHASE_CENTER_X.is_integer() or not PHASE_CENTER_Y.is_integer():
        raise ValueError("center_xy must be integer edge coordinates for this mask")

    coordinate = (
        np.arange(WINDOW, dtype=np.float64) + 0.5 - WINDOW / 2.0
    ) * PIXEL_PITCH_M
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    wave_number = 2.0 * math.pi / WAVELENGTH_M
    tile_phase = -wave_number * (xx * xx + yy * yy) / (2.0 * FOCAL_LENGTH_M)

    phase = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.float64)
    left = int(round(PHASE_CENTER_X - ARRAY_COLS * WINDOW / 2.0))
    top = int(round(PHASE_CENTER_Y - ARRAY_ROWS * WINDOW / 2.0))
    right = left + ARRAY_COLS * WINDOW
    bottom = top + ARRAY_ROWS * WINDOW
    if left < 0 or top < 0 or right > SLM_WIDTH or bottom > SLM_HEIGHT:
        raise ValueError("Fresnel array footprint exceeds the SLM canvas")

    centers = []
    for row in range(ARRAY_ROWS):
        for col in range(ARRAY_COLS):
            y0 = top + row * WINDOW
            x0 = left + col * WINDOW
            phase[y0 : y0 + WINDOW, x0 : x0 + WINDOW] = tile_phase
            centers.append([x0 + WINDOW / 2.0, y0 + WINDOW / 2.0])

    wrapped = np.mod(phase, 2.0 * math.pi)
    phase_uint8 = np.rint(wrapped / (2.0 * math.pi) * 255.0).astype(np.uint8)
    output = OUTPUT_DIR / OUTPUT_NAME
    Image.fromarray(phase_uint8, mode="L").save(output)

    manifest = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "matlab_equivalent": "generate_fresnel_2x2_30cm.m",
        "output_bmp": output.name,
        "output_sha256": sha256_file(output),
        "output_mode": "L",
        "output_bit_depth": 8,
        "slm_size_wh": [SLM_WIDTH, SLM_HEIGHT],
        "phase_center_edge_xy": [PHASE_CENTER_X, PHASE_CENTER_Y],
        "pixel_pitch_um": PIXEL_PITCH_M * 1.0e6,
        "wavelength_nm": WAVELENGTH_M * 1.0e9,
        "focal_length_cm": FOCAL_LENGTH_M * 100.0,
        "array_shape_rc": [ARRAY_ROWS, ARRAY_COLS],
        "tile_size_wh": [WINDOW, WINDOW],
        "footprint_half_open_xyxy_zero_based": [left, top, right, bottom],
        "footprint_matlab_1_based_inclusive_xyxy": [
            left + 1,
            top + 1,
            right,
            bottom,
        ],
        "lens_centers_edge_xy": centers,
        "phase_formula": "mod(-k*(x^2+y^2)/(2*f), 2*pi)",
        "outside_value_uint8": 0,
        "outside_semantics": "zero phase, not an amplitude stop",
        "flip_applied": False,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
