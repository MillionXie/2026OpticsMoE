import torch
from torch import nn
from LightGenV2.tasks.t17_router_classification.model_linear_ccd import LinearD2NNCCD


def test_no_electronic_parameters_or_oeo():
    model=LinearD2NNCCD()
    assert sum(p.numel() for p in model.parameters())==3*478*478
    assert list(dict(model.named_parameters()))==['phases.0','phases.1','phases.2']
    assert not any(isinstance(m,(nn.Linear,nn.Conv2d,nn.ReLU,nn.LayerNorm)) for m in model.modules())
    assert not hasattr(model,'oeo')


def test_encoded_field_path_is_complex_linear():
    torch.set_num_threads(4);torch.manual_seed(17);m=LinearD2NNCCD()
    with torch.no_grad():
        for p in m.phases:p.normal_(0,.1)
        a=torch.randn(1,518,518,dtype=torch.complex64)
        b=torch.randn_like(a);q=.3+.2j
        expected=m.propagate_field(a)+q*m.propagate_field(b)
        actual=m.propagate_field(a+q*b)
    assert (actual-expected).abs().max()/expected.abs().max()<2e-6


def test_final_detector_and_all_three_phase_gradients():
    torch.set_num_threads(4);torch.manual_seed(17);m=LinearD2NNCCD()
    output=m(torch.rand(2,224,224))
    assert output['probabilities'].shape==(2,9)
    assert torch.allclose(output['probabilities'].sum(1),torch.ones(2))
    loss=-output['logits'][torch.arange(2),torch.tensor([0,8])].mean();loss.backward()
    for p in m.phases:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum())>0


def test_eight_class_ccd_covers_active_field_and_backpropagates():
    torch.set_num_threads(4);torch.manual_seed(17);m=LinearD2NNCCD(num_classes=8)
    x=torch.rand(2,224,224)
    out=m(x)
    expanded=torch.nn.functional.interpolate(x[:,None],(478,478),mode='bilinear',align_corners=False)[:,0]
    expanded=expanded/expanded.square().sum((-2,-1),keepdim=True).sqrt()
    field=m.propagate_field(torch.nn.functional.pad(expanded,(20,)*4).to(torch.complex64))
    total=field.abs().square()[:,20:498,20:498].sum((-2,-1))
    assert out['probabilities'].shape==(2,8)
    assert torch.allclose(out['class_energies'].sum(1),total,rtol=1e-6)
    assert torch.allclose(out['probabilities'].sum(1),torch.ones(2))
    loss=-out['logits'][torch.arange(2),torch.tensor([0,7])].mean();loss.backward()
    assert sum(p.numel() for p in m.parameters())==685452
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.phases)
