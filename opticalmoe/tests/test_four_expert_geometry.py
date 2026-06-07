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
    assert layout.prompt_cell_size == layout.expert_size
    assert layout.prompt_cells[0].to_dict() == {
        "name": "C0",
        "y0": 100,
        "y1": 300,
        "x0": 100,
        "x1": 300,
        "center": [200.0, 200.0],
    }


def test_prompt_cell_size_changes_mask_only_and_keeps_centers_fixed():
    baseline = FourExpertLayout()
    expanded = FourExpertLayout(prompt_cell_size=300)
    expanded.validate()

    assert [item.to_dict() for item in expanded.experts] == [
        item.to_dict() for item in baseline.experts
    ]
    assert [item.center for item in expanded.prompt_cells] == [
        item.center for item in expanded.experts
    ]
    assert expanded.prompt_cells[0].to_dict() == {
        "name": "C0",
        "y0": 50,
        "y1": 350,
        "x0": 50,
        "x1": 350,
        "center": [200.0, 200.0],
    }
    assert torch.all(expanded.prompt_cell_masks().sum(dim=0) <= 1.0)


def test_overlapping_prompt_cells_are_rejected():
    layout = FourExpertLayout(prompt_cell_size=302)
    try:
        layout.validate()
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Overlapping prompt cells should be rejected.")


def test_grating_period_is_invariant_to_prompt_cell_size():
    baseline = MicrolensArrayPrompt(FourExpertLayout(prompt_cell_size=200))
    expanded = MicrolensArrayPrompt(FourExpertLayout(prompt_cell_size=300))

    baseline_reports = baseline.report(prompt_to_expert_m=0.20)
    expanded_reports = expanded.report(prompt_to_expert_m=0.20)
    for baseline_row, expanded_row in zip(baseline_reports, expanded_reports):
        assert baseline_row["center_y_px"] == expanded_row["center_y_px"]
        assert baseline_row["center_x_px"] == expanded_row["center_x_px"]
        assert baseline_row["theta_y_deg"] == expanded_row["theta_y_deg"]
        assert baseline_row["theta_x_deg"] == expanded_row["theta_x_deg"]
        assert (
            baseline_row["grating_period_y_px"]
            == expanded_row["grating_period_y_px"]
        )
        assert (
            baseline_row["grating_period_x_px"]
            == expanded_row["grating_period_x_px"]
        )


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
