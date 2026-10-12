"""Data-screening invariants; Linux additionally checks actual MixUp gradients."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_caltech_selected import choose_classes, features


def test_screening_keeps_separable_classes_and_is_deterministic():
    train={str(i):np.array([[float(i),float(i*i)]]) for i in range(5)}
    val={c:x.copy() for c,x in train.items()}
    train['bad']=train['0'].copy();val['bad']=train['4'].copy()
    classes,steps=choose_classes(train,val,5)
    assert classes==list(map(str,range(5))) and steps[-1]['macro_recall']==1
    assert (classes,steps)==choose_classes(train,val,5)


def test_fixed_features_finite_bounded_and_no_randomness():
    x=np.zeros((150,150,3),dtype=np.uint8)
    assert np.isfinite(features(x)).all() and np.linalg.norm(features(x))==0
    x[20:80,30:60]=255
    assert np.isclose(np.linalg.norm(features(x)),1)
    assert np.array_equal(features(x),features(x))


@pytest.mark.skipif(sys.platform=='win32',reason='Use existing native Linux Torch; no shared environment changes')
def test_mixup_determinism_and_weighted_tail_gradient():
    import medicinal_pilot as p
    torch=p.torch
    a,perm,digest=p.mixup_schedule(17,12,3,7,.2)
    b,other,second=p.mixup_schedule(17,12,3,7,.2)
    assert a==b and np.array_equal(perm,other) and digest==second
    assert .5<=a<=1 and sorted(perm)==list(range(7))
    target=torch.nn.functional.one_hot(torch.arange(7)%3,3).float()
    target=target*.92+.08/3
    mixed=a*target+(1-a)*target[perm]
    assert torch.allclose(mixed.sum(1),torch.ones(7))
    logits=torch.randn(7,3,requires_grad=True)
    weights=torch.tensor([.8,1.3,1.1])
    def loss(x,y): return -(y*weights[None]*x.softmax(1).log()).sum(1).mean()
    loss(logits,mixed).backward();ref=logits.grad.clone();logits.grad=None
    for start in range(0,7,2):
        end=min(start+2,7)
        (loss(logits[start:end],mixed[start:end])*(end-start)/7).backward()
    assert torch.allclose(logits.grad,ref,atol=1e-7)
    assert torch.isfinite(ref).all() and float(ref.norm())>0


@pytest.mark.skipif(sys.platform=='win32',reason='Use existing native Linux Torch')
def test_twelve_layer_optical_graph_contract():
    import fast_optical as f
    profile=dict(depths=[2,4,12,16],detector=f.detector(20))
    graph=f.optical.graph_config(12,profile)
    assert graph['model']['num_cycles']==6 and graph['global_fc']['num_masks']==6
    assert graph['expert_bank']['d2nn']['nonlinear_schedule']==[True]*6
    cfg=dict(classes=list(range(20)),phase_init_raw_uniform=.1,residual_rho=.3)
    model=f.build(12,profile,cfg,device='cpu')
    assert sum(p.numel() for p in model.parameters())==2649088
    assert len(list(model.parameters()))==61
    assert not any(isinstance(m,(f.torch.nn.Linear,f.torch.nn.Conv2d)) for m in model.modules())
    assert not any(hasattr(m,'residual_rho') for n,m in model.named_modules() if 'router' in n)
