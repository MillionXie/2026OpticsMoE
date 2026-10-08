import pytest
import torch
from LightGenV2.tasks.t02_keypoint_detection.lab_field_units import physical_field


def test_exact_fixed_units_and_no_extra_peak_normalization():
    field = torch.complex(torch.tensor([0., .5, 2., 14.]), torch.zeros(4))
    assert torch.equal(physical_field(field) * 16, field)
    assert torch.equal(physical_field(field[:2]), physical_field(field)[:2])


def test_quantization_zero_support_and_bound():
    field = torch.polar(torch.tensor([0., .5, 8., 16.]), torch.tensor([0., .1, .5, 1.]))
    result = physical_field(field, quantize=True)
    assert result[0] == 0
    assert result.abs().max() <= 1.000001
    assert torch.max((result.abs() - field.abs()/16).abs()) <= 1/510 + 1e-6
    assert torch.allclose(torch.angle(result[1:]), torch.angle(field[1:]), atol=1e-6)


def test_reject_out_of_range_and_nonfinite_without_clipping():
    for field in (torch.tensor([17.]), torch.tensor([float('nan')]), torch.tensor([float('inf')])):
        with pytest.raises(ValueError):
            physical_field(field)
    with pytest.raises(ValueError):
        physical_field(torch.ones(1), scale=0)
