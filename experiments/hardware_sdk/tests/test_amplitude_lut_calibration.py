from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.hardware_sdk.workflows.amplitude_lut_calibration import (
    fit_linearized_lut,
    isotonic_non_decreasing,
    read_global_lut,
    write_global_lut,
)


def _pixel2_lut() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "vendor_sdk/amplitude_meadowlark/LUT Files"
        / "slm7930_at532-70c-pixel-2.lut"
    )


def test_packaged_pixel2_lut_is_strict_global_8bit_table() -> None:
    values = read_global_lut(_pixel2_lut())
    assert values.shape == (256,)
    assert values[0] == 1101
    assert values[80] == 1033
    assert values[255] == 928
    assert np.all(np.diff(values) <= 0)


def test_global_lut_round_trip_preserves_all_dac_values(tmp_path: Path) -> None:
    values = read_global_lut(_pixel2_lut())
    output = write_global_lut(tmp_path / "generated.lut", values, overwrite=False)
    assert np.array_equal(read_global_lut(output), values)
    assert len(output.read_text(encoding="ascii").splitlines()) == 256


def test_isotonic_fit_pools_local_response_reversal() -> None:
    fitted = isotonic_non_decreasing([0.0, 0.3, 0.2, 0.8, 1.0])
    assert np.allclose(fitted, [0.0, 0.25, 0.25, 0.8, 1.0])


def test_field_amplitude_fit_selects_dark_to_255_branch() -> None:
    base = read_global_lut(_pixel2_lut())
    gray = np.rint(np.linspace(0, 255, 64)).astype(np.float64)
    # Synthetic version of the measured U-shaped response: the true dark state
    # is near gray=80, while gray=255 is the higher-dynamic-range bright end.
    energy = np.square((gray - 80.0) / 175.0) + 0.001 * np.sin(gray / 11.0)

    generated, rows, report = fit_linearized_lut(
        base_lut=base,
        measured_gray=gray,
        measured_energy=energy,
        transfer_mode="field_amplitude",
    )

    assert report["selected_branch"] == "dark_to_gray255"
    assert 75 <= report["dark_state_measured_gray"] <= 85
    assert report["predicted_normalized_intensity_rmse"] < 0.02
    assert report["generated_voltage_reversals"] == 0
    assert len(rows) == 256
    assert np.all(np.diff(generated) <= 0)
    assert abs(rows[0]["mapped_base_gray"] - 80.0) < 6.0
    assert abs(rows[-1]["mapped_base_gray"] - 255.0) < 1.0
    # Mid-gray is half field amplitude, hence one quarter target intensity.
    assert abs(rows[128]["target_normalized_intensity"] - 0.252) < 0.005
