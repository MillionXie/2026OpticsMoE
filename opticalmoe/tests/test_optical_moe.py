import torch

from opticalmoe.optics import MoeLayout, OpticalMoEClassifier, TranslatedDetectorArray
from opticalmoe.optics.grating import (
    build_detilt_phase_for_aperture,
    build_linear_grating_phase,
    compute_steering_params,
)


def _layout():
    return MoeLayout(
        canvas_height=64,
        canvas_width=128,
        expert_size=48,
        gap_pixels=16,
        margin_x=8,
        margin_y=8,
        input_size=16,
    )


def test_moe_layout_geometry():
    layout = _layout()
    layout.validate()
    assert layout.left.center == (32.0, 32.0)
    assert layout.right.center == (32.0, 96.0)
    assert layout.input_aperture.x0 == 56
    assert layout.input_aperture.x1 == 72


def test_grating_and_detilt_shapes():
    layout = _layout()
    params = compute_steering_params(
        wavelength_m=532e-9,
        pixel_size_m=8e-6,
        shift_pixels=layout.right_shift_pixels,
        distance_m=0.01,
        inter_layer_m=0.005,
    )
    phase = build_linear_grating_phase(layout.canvas_shape, params.grating_period_px, "right")
    detilt = build_detilt_phase_for_aperture(
        layout.canvas_shape,
        layout.right,
        params.grating_period_px,
        "right",
        prompt_slope_sign=1,
    )
    assert phase.shape == layout.canvas_shape
    assert detilt.shape == layout.canvas_shape
    assert torch.all(detilt[:, : layout.right.x0] == 0)


def test_translated_detector_outputs():
    layout = _layout()
    detector = TranslatedDetectorArray(num_classes=10, layout=layout, detector_size=4)
    field = torch.ones(2, *layout.canvas_shape, dtype=torch.complex64)
    outputs = detector(field)
    assert outputs["left_raw"].shape == (2, 10)
    assert outputs["right_raw"].shape == (2, 10)
    assert outputs["paired_sum_global"].shape == (2, 10)


def test_optical_moe_forward_and_intermediates():
    layout = _layout()
    model = OpticalMoEClassifier(
        num_classes=10,
        layout=layout,
        num_layers=2,
        detector_size=4,
        target_side="right",
        distances_m={
            "input_to_prompt": 0.001,
            "prompt_to_first_layer": 0.01,
            "inter_layer": 0.005,
            "last_layer_to_detector": 0.005,
        },
    )
    x = torch.rand(2, 1, 16, 16)
    logits, intermediates = model(x, return_intermediates=True)
    assert logits.shape == (2, 10)
    assert model.num_propagation_segments == model.num_layers + 2
    assert "after_entrance_detilt" in intermediates
    assert "detector_energies_left_raw" in intermediates
    assert "detector_energies_right_raw" in intermediates
    assert "branch_energy_ratios" in intermediates
