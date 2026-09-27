import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amplitude import encode, bounded_graph, quantize_amplitude


def test_bounded_zero_gradient_and_bmp():
    value = torch.tensor([0., .01, .1, .5, 1., 3., 20.], requires_grad=True)
    amplitude = encode(value)
    assert amplitude[0] == 0 and bool((amplitude <= 1).all())
    assert torch.equal(amplitude, torch.tanh(value/.5))
    amplitude.sum().backward()
    assert torch.isfinite(value.grad).all() and bool((value.grad[:5] > 0).all())
    assert float((quantize_amplitude(amplitude).float()/255-amplitude).abs().max()) <= .5/255+1e-7
    with pytest.raises(ValueError):
        quantize_amplitude(torch.tensor([1.1]))


def test_all_six_methods_restore():
    sys.path.insert(0, str(ROOT / "runtime"))
    from LightGenV2.tasks.t06_video_quality_assessment.models import multivideo9x4 as optics
    from amplitude import METHODS
    before = {(name,method): getattr(getattr(optics,name),method) for name, methods in METHODS.items() for method in methods}
    with bounded_graph():
        assert all(getattr(getattr(optics,name),method) is not function for (name,method),function in before.items())
    assert all(getattr(getattr(optics,name),method) is function for (name,method),function in before.items())
