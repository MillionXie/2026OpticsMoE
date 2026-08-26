import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.dual_slm_alignment import (
    _polyomino_registration_amplitude,
    _registered_checker_grating,
    generate,
)


def test_polyomino_phase_is_flat_in_black_and_alternates_in_both_axes() -> None:
    amplitude = _polyomino_registration_amplitude(478, 64)
    phase = _registered_checker_grating(478, 64, 8, amplitude)
    assert np.all(phase[amplitude == 0] == 0)

    # Known O and T tetromino landmarks are entirely black.
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1), (0, 4), (1, 5)):
        patch = amplitude[row * 64 : (row + 1) * 64, column * 64 : (column + 1) * 64]
        assert np.all(patch == 0)

    # Adjacent open cells switch x/y orientation along rows and columns.
    def changes(cell_row: int, cell_column: int) -> tuple[bool, bool]:
        patch = phase[
            cell_row * 64 + 1 : (cell_row + 1) * 64,
            cell_column * 64 + 1 : (cell_column + 1) * 64,
        ]
        return bool(np.any(np.diff(patch, axis=0))), bool(
            np.any(np.diff(patch, axis=1))
        )

    assert changes(2, 2) == (False, True)
    assert changes(2, 3) == (True, False)
    assert changes(3, 2) == (True, False)


def test_regular_checker_phase_is_only_in_white_and_visible_cells_alternate() -> None:
    y, x = np.indices((478, 478))
    amplitude = (((x // 64 + y // 64) % 2) * 255).astype(np.uint8)
    phase = _registered_checker_grating(
        478,
        64,
        8,
        amplitude,
        orientation_mode="visible_checker_cells",
    )
    assert np.all(phase[amplitude == 0] == 0)

    def changes(cell_row: int, cell_column: int) -> tuple[bool, bool]:
        patch = phase[
            cell_row * 64 + 1 : (cell_row + 1) * 64,
            cell_column * 64 + 1 : (cell_column + 1) * 64,
        ]
        return bool(np.any(np.diff(patch, axis=0))), bool(
            np.any(np.diff(patch, axis=1))
        )

    # White cells (0,1)->(0,3) and (0,1)->(2,1) alternate direction.
    assert changes(0, 1) == (False, True)
    assert changes(0, 3) == (True, False)
    assert changes(2, 1) == (True, False)


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
    assert len(pairs) == 7
    assert {row["logical_cell_size_px"] for row in pairs} == {64, 80, 96}
    assert {row["phase_native_grating_period_px"] for row in pairs} == {17}

    primary = next(
        row for row in pairs if row["pair_id"] == "checker_xy_c80_p8_primary"
    )
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
    complement_c64 = next(
        row for row in pairs if row["pair_id"] == "checker_xy_c64_p8_complement"
    )
    assert complement_c64["amplitude_layout"] == "polyomino_black_landmarks"
    assert complement_c64["phase"].endswith(
        "phase_registration_checker_xy_c64_p8_1920x1200.bmp"
    )
    assert primary["phase"].endswith("_primary_1920x1200.bmp")
    regular = next(
        row for row in pairs if row["pair_id"] == "regular_checker_xy_c64_p8"
    )
    assert regular["amplitude_layout"] == "strict_binary_checkerboard"
    assert regular["amplitude"].endswith(
        "amplitude_registration_regular_checker_c64_1024x1024.bmp"
    )
    assert regular["phase"].endswith(
        "phase_registration_regular_checker_xy_c64_p8_1920x1200.bmp"
    )
    persisted = json.loads(
        (output_dir / "alignment_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["phase_slm"]["flip_vertical_before_raster"] is True
    assert (
        persisted["registration_protocol"]["preview_is_propagation_simulation"] is False
    )
    assert persisted["schema_version"] == 2


def test_recommended_checker_grating_pair_is_unambiguous_and_normal_polarity(
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
                "  invert_before_export: false",
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
    pair_dir = output_dir / "recommended_checker_grating_pair"
    amplitude_path = pair_dir / "amplitude_checker_255open_c64_1024x1024.bmp"
    phase_path = pair_dir / "phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp"
    assert sorted(path.name for path in pair_dir.glob("*.bmp")) == sorted(
        [amplitude_path.name, phase_path.name]
    )

    with Image.open(amplitude_path) as image:
        amplitude = np.asarray(image)
        assert image.mode == "L"
        assert image.size == (1024, 1024)
    with Image.open(phase_path) as image:
        phase = np.asarray(image)
        assert image.mode == "L"
        assert image.size == (1920, 1200)
    assert set(np.unique(amplitude)) == {0, 255}
    assert set(np.unique(phase)) == {0, 128}

    canonical_amplitude = Path(
        report["files"]["amplitude"]["registration_regular_checker_c64"]["path"]
    )
    canonical_phase = Path(
        report["files"]["phase"]["registration_regular_checker_xy_c64_p8"]["path"]
    )
    assert amplitude_path.read_bytes() == canonical_amplitude.read_bytes()
    assert phase_path.read_bytes() == canonical_phase.read_bytes()

    manifest = json.loads((pair_dir / "pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["amplitude_command_contract"] == {
        "white_open_value_uint8": 255,
        "black_closed_value_uint8": 0,
        "invert_in_player": False,
    }
    assert manifest["amplitude"]["active_bounds_xyxy"] == [273, 273, 751, 751]
    assert manifest["phase"]["active_bounds_xyxy"] == [472, 82, 1488, 1098]
    assert manifest["phase_transform"]["center_xy"] == [980.0, 590.0]
    assert manifest["phase_transform"]["flip_vertical_before_raster"] is True
    assert (
        manifest["amplitude"]["sha256"]
        == report["files"]["amplitude"]["registration_regular_checker_c64"]["sha256"]
    )
    assert (
        manifest["phase"]["sha256"]
        == report["files"]["phase"]["registration_regular_checker_xy_c64_p8"]["sha256"]
    )
    readme = (pair_dir / "README.md").read_text(encoding="utf-8")
    assert "255` is white/open/transmissive" in readme
    assert "Do not invert" in readme
