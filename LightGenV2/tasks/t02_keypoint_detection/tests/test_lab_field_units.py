import pytest
import torch
from LightGenV2.tasks.t02_keypoint_detection.lab_field_units import physical_field
from LightGenV2.tasks.t02_keypoint_detection.lab_field_units import FieldUnits
from types import SimpleNamespace


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


def test_three_pass_replay_and_exception_restoration():
    class Propagator(torch.nn.Module):
        def forward(self, field):
            return torch.fft.fft2(field, norm='ortho')
    router, body = Propagator(), Propagator()
    a = SimpleNamespace(y0=0, y1=8, x0=0, x1=8)
    core = SimpleNamespace(router=SimpleNamespace(propagator=router), propagator=body,
                           geometry=SimpleNamespace(active_aperture=a))
    model = SimpleNamespace(core=SimpleNamespace(optical_branch=SimpleNamespace(core=core)))
    originals = router.forward, body.forward
    field = torch.complex(torch.rand(1, 8, 8)*8, torch.zeros(1, 8, 8))
    def run():
        return tuple(prop(field).abs().square() for prop in (router, body, body))
    expected = run()
    with FieldUnits(model) as tap:
        actual = run()
        physical = {k:v.clone() for k,v in tap.detectors.items()}
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-6) for x,y in zip(expected,actual))
    with FieldUnits(model, measured=physical):
        replayed = run()
    assert all(torch.allclose(x,y,atol=2e-5,rtol=1e-6) for x,y in zip(expected,replayed))
    with pytest.raises(RuntimeError):
        with FieldUnits(model):
            raise RuntimeError('test exit')
    assert router.forward == originals[0] and body.forward == originals[1]


def test_bounded_pre_hook_bridge_does_not_restore_legacy_units():
    from LightGenV2.tasks.t02_keypoint_detection.physical_amplitude import install_bounded_inputs
    class Propagator(torch.nn.Module):
        def forward(self, field):
            return torch.fft.fft2(field, norm='ortho')
    router, body = Propagator(), Propagator()
    core = SimpleNamespace(router=SimpleNamespace(propagator=router),propagator=body,
        geometry=SimpleNamespace(active_aperture=SimpleNamespace(y0=0,y1=8,x0=0,x1=8)))
    outer=SimpleNamespace(optical_branch=SimpleNamespace(core=core))
    model=SimpleNamespace(core=outer)
    install_bounded_inputs(outer,quantize=True)
    field=torch.complex(torch.rand(1,8,8)*10,torch.zeros(1,8,8))
    def run():return tuple(p(field).abs().square() for p in (router,body,body))
    expected=run()
    with FieldUnits(model,scale=1.,quantize=True) as tap:actual=run()
    with FieldUnits(model,scale=1.,quantize=True,measured=tap.detectors):replayed=run()
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(expected,actual))
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(expected,replayed))
    assert all(v.max()<=1.000001 for v in tap.amplitudes.values())
