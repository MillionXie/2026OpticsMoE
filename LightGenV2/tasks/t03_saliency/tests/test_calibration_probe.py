import pytest
import torch
from LightGenV2.tasks.t03_saliency.calibration_probe import SpatialLogitCalibration, require_train_ids


def test_identity_bounds_gradients_and_no_cross_sample_mixing():
    m=SpatialLogitCalibration(8);x=torch.randn(3,1,8,8)
    assert sum(p.numel() for p in m.parameters())==2
    torch.testing.assert_close(m(x),x,rtol=0,atol=0)
    with torch.no_grad():m.raw_scale.fill_(.7);m.raw_center.fill_(.2)
    torch.testing.assert_close(m(x)[0],m(x[:1])[0])
    m(x).square().mean().backward()
    assert all(p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in m.parameters())
    a,b=m.coefficients();assert .5<=a<=2 and -2<=b<=2
    with torch.no_grad():m.raw_scale.fill_(100);m.raw_center.fill_(-100)
    a,b=m.coefficients();assert a==2 and b==-2
    with pytest.raises(ValueError):m(torch.randn(1,1,9,9))


def test_only_training_ids_are_allowed_for_fitting():
    require_train_ids(['train/1','train/2'])
    for ids in ([],['train/1','train/1'],['validation/1'],['test/1']):
        with pytest.raises(ValueError):require_train_ids(ids)
