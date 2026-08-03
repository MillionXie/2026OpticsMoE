from pathlib import Path

import numpy as np
from PIL import Image

from experiments.slm_calibration_bmp_generator.generate import (
    centered_canvas,
    checkerboard,
    generate,
    lens_phase,
)
from experiments.slm_calibration_bmp_generator.settings import Settings


def test_centered_canvas_and_checkerboard() -> None:
    active = checkerboard(6, 2)
    canvas, bounds = centered_canvas(active, (10, 8))
    assert canvas.shape == (8, 10)
    assert bounds == [2, 1, 8, 7]
    assert set(np.unique(active)) == {0, 255}


def test_lens_phase_is_finite_uint8() -> None:
    phase = lens_phase(64, 8.0, 532.0, 5.0)
    assert phase.shape == (64, 64)
    assert phase.dtype == np.uint8
    assert int(phase.max()) > int(phase.min())


def test_generate_expected_bmp_sizes_and_phase_flip(tmp_path: Path) -> None:
    settings = Settings(
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "out",
        active_size=20,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        amplitude_size_wh=(40, 30),
        phase_size_wh=(40, 34),
        phase_flip_vertical=True,
        checkerboard_block_px=4,
        lens_focal_lengths_cm=(5.0, 10.0),
        letter="A",
    )
    report = generate(settings)
    amplitude = Path(report["files"]["amplitude"]["checkerboard"]["path"])
    phase = Path(report["files"]["phase"]["lens_5cm"]["path"])
    assert Image.open(amplitude).size == (40, 30)
    assert Image.open(phase).size == (40, 34)
    assert report["files"]["amplitude"]["checkerboard"]["flip_vertical_before_export"] is False
    assert report["files"]["phase"]["lens_5cm"]["flip_vertical_before_export"] is True

