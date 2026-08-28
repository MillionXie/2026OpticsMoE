from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from experiments.hardware_sdk.workflows.reconstruct_slm import physical_pitch_nearest

from ..lab_session import _read_csv, _read_json
from ..simulation_agreement import (
    PlayedBMPSimulator,
    _load_amplitude,
    binary_metrics,
    binary_uint8,
    decode_played_phase,
    inverse_physical_pitch_nearest,
    monochrome_uint8,
)


def test_inverse_physical_pitch_nearest_recovers_every_logical_pixel() -> None:
    generator = np.random.default_rng(17)
    logical = generator.integers(0, 256, size=(478, 478), dtype=np.uint8)
    native = physical_pitch_nearest(
        logical,
        logical_pixel_pitch_um=17.0,
        slm_pixel_pitch_um=8.0,
    )
    assert native.shape == (1016, 1016)
    recovered = inverse_physical_pitch_nearest(native)
    np.testing.assert_array_equal(recovered, logical)


def test_inverse_preserves_export_flip_contract() -> None:
    logical = np.arange(478 * 478, dtype=np.uint32).reshape(478, 478).astype(np.uint8)
    exported = physical_pitch_nearest(
        np.flipud(logical),
        logical_pixel_pitch_um=17.0,
        slm_pixel_pitch_um=8.0,
    )
    recovered = np.flipud(inverse_physical_pitch_nearest(exported))
    np.testing.assert_array_equal(recovered, logical)


def test_monochrome_and_binary_outputs_are_declared_uint8_maps() -> None:
    value = np.arange(16, dtype=np.float32).reshape(4, 4)
    monochrome = monochrome_uint8(value)
    binary = binary_uint8(value, threshold_ratio=0.2)
    assert monochrome.dtype == np.uint8
    assert int(monochrome.min()) == 0
    assert int(monochrome.max()) == 255
    assert set(np.unique(binary)) == {0, 255}
    metrics = binary_metrics(value, value, threshold_ratio=0.2)
    assert metrics["binary_pcc"] == pytest.approx(1.0)
    assert metrics["binary_ssim"] == pytest.approx(1.0)
    assert metrics["binary_iou"] == pytest.approx(1.0)
    assert metrics["binary_dice"] == pytest.approx(1.0)


@pytest.mark.skipif(
    not os.environ.get("MNIST4_FULL_SMOKE_STAGE"),
    reason="set MNIST4_FULL_SMOKE_STAGE for the optional 1024-grid integration test",
)
def test_full_grid_played_bmp_simulation_on_configured_device() -> None:
    import torch

    stage = Path(os.environ["MNIST4_FULL_SMOKE_STAGE"]).resolve()
    bundle_root = Path(os.environ["MNIST4_FULL_SMOKE_BUNDLE_ROOT"]).resolve()
    contract = _read_json(stage / "stage_contract.json")
    phase_bmp = next((stage / "phase_to_play").glob("*.bmp"))
    phase = decode_played_phase(phase_bmp, contract)
    row = _read_csv(stage / "samples.csv")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = PlayedBMPSimulator(
        bundle_root / "payload" / "model" / "lab_model_config.yaml",
        phase,
        device,
    )
    amplitude = _load_amplitude(stage / "amplitude_to_play" / row["amplitude_file"])
    output = simulator(np.stack([amplitude]))
    assert output.shape == (1, 478, 478)
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) > 0.0
