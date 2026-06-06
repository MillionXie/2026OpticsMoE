import torch

from opticalmoe.optics.four_expert_geometry import FourExpertLayout
from opticalmoe.optics.microlens_prompt import MicrolensArrayPrompt


def test_four_expert_layout_matches_required_pixel_geometry():
    layout = FourExpertLayout()
    layout.validate()

    assert layout.canvas_shape == (700, 700)
    assert layout.input_aperture.center == (350.0, 350.0)
    assert [item.center for item in layout.experts] == [
        (200.0, 200.0),
        (200.0, 500.0),
        (500.0, 200.0),
        (500.0, 500.0),
    ]
    assert torch.all(layout.prompt_cell_masks().sum(dim=0) <= 1.0)


def test_microlens_prompt_modes_preserve_complex_field_shape():
    layout = FourExpertLayout()
    prompt = MicrolensArrayPrompt(layout)
    field = torch.ones(1, 700, 700, dtype=torch.complex64)

    for mode in prompt.MODES:
        output = prompt(field, mode=mode)
        assert output.shape == field.shape
        assert output.dtype == torch.complex64


def test_onehot_prompt_amplitude_only_opens_selected_cell():
    layout = FourExpertLayout()
    prompt = MicrolensArrayPrompt(layout, amplitudes=[0.0, 1.0, 0.0, 0.0])
    amplitude = prompt.amplitude_map()
    masks = layout.prompt_cell_masks()

    assert torch.all(amplitude[masks[1].bool()] == 1.0)
    assert torch.all(amplitude[masks[0].bool()] == 0.0)
    assert torch.all(amplitude[masks[2].bool()] == 0.0)
    assert torch.all(amplitude[masks[3].bool()] == 0.0)
