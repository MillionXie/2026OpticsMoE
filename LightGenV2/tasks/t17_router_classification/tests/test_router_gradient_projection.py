import pytest
import torch
from LightGenV2.tasks.t17_router_classification.router_gradient_projection import project_raw_pair,router_projection_for_batch


@pytest.mark.parametrize('weight',[.2,.5,.8])
def test_conflict_projection_uses_physical_coordinates_and_common_descent(weight):
    raw=torch.tensor([[0.,1.]],dtype=torch.float64);s=raw.sigmoid();j=2*torch.pi*s*(1-s)
    a=torch.tensor([[1.,.2]],dtype=torch.float64);b=torch.tensor([[-1.,.4]],dtype=torch.float64)
    gradient,record=project_raw_pair(a*j,b*j,raw,weight);physical=gradient/j
    assert record['projected'] and record['cosine_before']<0
    assert (physical*a).sum()>=-1e-12 and (physical*b).sum()>=-1e-12
    assert not gradient.requires_grad


@pytest.mark.parametrize('second',[[1.,2.],[0.,0.]])
def test_nonconflicting_or_zero_gradient_keeps_weighted_raw_average(second):
    a=torch.tensor([[.5,1.]]);b=torch.tensor([second]);raw=torch.tensor([[.3,-.8]])
    output,record=project_raw_pair(a,b,raw,.25)
    assert torch.equal(output,.25*a+.75*b) and not record['projected']


def test_default_projection_does_nothing_without_accessing_teacher_or_model():
    assert router_projection_for_batch(None,None,None,None,None,None,{})==(None,None)


def test_soft_projection_rejects_any_hard_test_labels():
    cfg=dict(router_gradient_projection='two_domain_physical',architecture='LG',main_layers=2,dataset='eurosat_original_f',training_scope='all',phase_optimizer='wrapped',source_test_distillation=True,alternating_head={'anchor_strength':.001},teacher={'domain_cache_runs':{'RGB':'r','SAR':'s'}})
    with pytest.raises(ValueError,match='hard labels'):router_projection_for_batch(None,None,None,torch.tensor([1]),None,None,cfg,soft_only=True)


def test_projection_rejects_invalid_weight_and_nonfinite_inputs():
    with pytest.raises(ValueError):project_raw_pair(torch.ones(2),torch.ones(2),torch.zeros(2),1.1)
    with pytest.raises(ValueError):project_raw_pair(torch.tensor([float('nan')]),torch.ones(1),torch.zeros(1),.5)


@pytest.mark.parametrize('soft_only',[False,True])
def test_unbalanced_domain_groups_match_mixed_loss_when_not_conflicting(soft_only):
    from LightGenV2.tasks.t17_router_classification.alternating_readout import anchored_classification
    from LightGenV2.tasks.t17_router_classification.f_cross_domain import soft_only_source_test_loss
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.router_phase=torch.nn.Parameter(torch.tensor([.1,.2]));self.shared_head=torch.nn.Linear(2,2,bias=False)
            with torch.no_grad():self.shared_head.weight.copy_(torch.eye(2))
    m=Toy();x=torch.tensor([[1.,1.],[2.,1.],[3.,1.],[1.,2.]]);out=dict(raw_logits=m.shared_head(x*m.router_phase));teacher=torch.tensor([[1.,-1.]]).repeat(4,1).requires_grad_();labels=torch.zeros(4,dtype=torch.long);anchor=m.shared_head.weight.detach().clone()
    cfg=dict(router_gradient_projection='two_domain_physical',architecture='LG',main_layers=2,dataset='eurosat_original_f',training_scope='all',phase_optimizer='wrapped',source_test_distillation=True,alternating_head={'anchor_strength':.001},label_smoothing=.02,teacher={'domain_cache_runs':{'RGB':'r','SAR':'s'},'weight':.5,'temperature':2.})
    projected,record=router_projection_for_batch(m,out,torch.tensor([0,0,0,1]),None if soft_only else labels,teacher,anchor,cfg,soft_only)
    loss=soft_only_source_test_loss(out['raw_logits'],teacher,m.shared_head.weight,anchor,.001,.5,2.) if soft_only else anchored_classification(out['raw_logits'],labels,m.shared_head.weight,anchor,.001,.02,teacher,.5,2.)
    expected=torch.autograd.grad(loss,m.router_phase)[0]
    assert not record['projected'] and record['domain_weights']==[.75,.25]
    assert torch.allclose(projected,expected,atol=1e-7,rtol=1e-6) and teacher.grad is None
