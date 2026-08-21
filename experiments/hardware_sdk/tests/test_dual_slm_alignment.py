import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.dual_slm_alignment import generate


def test_checker_grating_registration_pairs_share_logical_geometry(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "alignment"
    config = tmp_path / "alignment.yaml"
    config.write_text(
        "\n".join(
            (
                f"output_dir: {output_dir.as_posix()}",
                "logical:",
                "  active_size: 478",
                "  usable_size: 518",
                "  pixel_pitch_um: 17.0",
                "amplitude_slm:",
                "  size_wh: [1024, 1024]",
                "  pixel_pitch_um: 17.0",
                "  center_xy: [512.0, 512.0]",
                "phase_slm:",
                "  size_wh: [1920, 1200]",
                "  pixel_pitch_um: 8.0",
                "  center_xy: [980.0, 590.0]",
                "  flip_vertical: true",
                "  flip_horizontal: false",
            )
        ),
        encoding="utf-8",
    )

    report = generate(config)
    pairs = report["registration_protocol"]["pairs"]
    assert len(pairs) == 6
    assert {row["logical_cell_size_px"] for row in pairs} == {64, 80, 96}
    assert {row["phase_native_grating_period_px"] for row in pairs} == {17}

    primary = next(row for row in pairs if row["pair_id"] == "checker_xy_c80_p8_primary")
    with Image.open(primary["amplitude"]) as image:
        amplitude = np.asarray(image)
        assert image.size == (1024, 1024)
    with Image.open(primary["phase"]) as image:
        phase = np.asarray(image)
        assert image.size == (1920, 1200)
    with Image.open(primary["idealized_preview"]) as image:
        preview = np.asarray(image)
        assert image.size == (478, 478)

    assert set(np.unique(amplitude)) == {0, 255}
    assert set(np.unique(phase)) == {0, 128}
    assert set(np.unique(preview)) == {0, 55, 245}
    persisted = json.loads(
        (output_dir / "alignment_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["phase_slm"]["flip_vertical_before_raster"] is True
    assert persisted["registration_protocol"]["preview_is_propagation_simulation"] is False
