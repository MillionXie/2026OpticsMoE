import torch

from opticalmoe.optics.six_layer_control import (
    ParameterMatchedFullCanvasPhaseMask,
    SixLayerNoPromptControl,
)


def test_parameter_matched_mask_modulates_full_canvas():
    mask = ParameterMatchedFullCanvasPhaseMask(
        canvas_shape=(32, 40),
        parameter_grid_size=8,
        phase_init="identity",
    )
    field = torch.ones(2, 32, 40, dtype=torch.complex64)
    output = mask(field)
    assert output.shape == field.shape
    assert output.dtype == torch.complex64
    assert mask.phase.raw_phase.numel() == 64
    assert mask.get_phase_wrapped().shape == (32, 40)


def test_six_layer_control_forward_and_parameter_count():
    model = SixLayerNoPromptControl(
        num_classes=10,
        canvas_shape=(64, 64),
        input_size=16,
        parameter_grid_size=12,
        phase_init="identity",
        detector_size=8,
    )
    images = torch.rand(1, 1, 16, 16)
    logits, intermediates = model(images, return_intermediates=True)
    assert logits.shape == (1, 10)
    assert len(intermediates["after_each_mask"]) == 6
    assert intermediates["detector_energies"].shape == (1, 10)
    assert model.num_propagation_segments == 8
    assert model.optical_parameter_count() == 6 * 12 * 12
    assert not hasattr(model, "prompt")
    assert not hasattr(model, "expert_layers")

