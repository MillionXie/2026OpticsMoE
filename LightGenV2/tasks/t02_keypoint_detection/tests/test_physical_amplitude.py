import pytest
import torch
from LightGenV2.tasks.t02_keypoint_detection.physical_amplitude import bounded_field


def test_bounds_zero_phase_and_gradient():
    x = torch.tensor([0j, 1+2j, -3-4j], requires_grad=True)
    y = bounded_field(x)
    assert y[0] == 0 and y.abs().max() <= 1
    assert torch.allclose(y[1:] / y[1:].abs(), x[1:] / x[1:].abs())
    y.abs().square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_uint8_exact_and_no_legacy_scale_restoration():
    x = torch.tensor([0j, .05+.1j, 4+0j])
    y = bounded_field(x, quantize=True)
    expected = torch.round(255*torch.tanh(x.abs()/.5))/255
    assert torch.allclose(y.abs(), expected)
    assert y.abs().square().max() <= 1


@pytest.mark.parametrize('scale', [0., -1., float('inf'), float('nan')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        bounded_field(torch.ones(1, dtype=torch.complex64), scale)
