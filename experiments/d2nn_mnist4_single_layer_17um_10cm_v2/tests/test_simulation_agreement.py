from __future__ import annotations

import numpy as np

from experiments.hardware_sdk.workflows.reconstruct_slm import physical_pitch_nearest

from ..simulation_agreement import inverse_physical_pitch_nearest


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
