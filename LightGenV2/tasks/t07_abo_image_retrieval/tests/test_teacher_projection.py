import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.teacher_projection import backward_primary_teacher,projection_enabled
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config


@pytest.mark.parametrize('teacher,expected,conflict',[([-1.,1.],[1.,1.],1.),([1.,1.],[2.,1.],0.)])
def test_projection_exact_toy_gradient_and_no_parameter_step(teacher,expected,conflict):
    p=torch.nn.Parameter(torch.tensor([2.,3.]));before=p.detach().clone();calls=[]
    optimizer=torch.optim.SGD([p],lr=.1)
    def closure():
        calls.append(1);primary=p[0];kd=(p*torch.tensor(teacher)).sum()
        return dict(loss=primary+kd,primary_loss=primary,teacher_loss=kd)
    _,audit=backward_primary_teacher(closure,optimizer)
    torch.testing.assert_close(p.grad,torch.tensor(expected))
    assert torch.equal(p,before) and len(calls)==1
    assert audit['teacher_conflict_fraction']==conflict
    assert audit['teacher_shared_cosine_after']>=-1e-6


def test_private_heads_and_zero_lr_gradients_not_projected():
    shared=torch.nn.Parameter(torch.ones(2));private=torch.nn.Parameter(torch.ones(1));inactive=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.SGD([{'params':[shared,private],'lr':.1},{'params':[inactive],'lr':0.}])
    def closure():
        primary=shared[0]+3*private.sum()+7*inactive.sum();kd=-shared[0]+shared[1]+inactive.sum()
        return dict(loss=primary+kd,primary_loss=primary,teacher_loss=kd)
    backward_primary_teacher(closure,optimizer)
    torch.testing.assert_close(shared.grad,torch.ones(2));torch.testing.assert_close(private.grad,torch.tensor([3.]))
    assert inactive.grad is None


def test_zero_primary_gradient_and_teacher_only_parameters_are_finite():
    p=torch.nn.Parameter(torch.ones(2));t=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.SGD([p,t],lr=.1)
    def closure():
        primary=(p*0).sum();kd=p.sum()+t.sum()
        return dict(loss=primary+kd,primary_loss=primary,teacher_loss=kd)
    _,audit=backward_primary_teacher(closure,optimizer)
    torch.testing.assert_close(p.grad,torch.ones(2));torch.testing.assert_close(t.grad,torch.ones(1))
    assert audit['teacher_conflict_fraction']==0 and audit['teacher_shared_cosine_before']==0


def test_nonfinite_loss_fails_without_parameter_mutation():
    p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.SGD([p],lr=.1)
    with pytest.raises(RuntimeError):backward_primary_teacher(lambda:dict(loss=p.sum()*float('nan'),primary_loss=p.sum(),teacher_loss=p.sum()),optimizer)
    assert p.grad is None and float(p.detach())==1.


def test_projection_disabled_by_default_and_only_profile_change():
    assert not projection_enabled({})
    base=overlay_config({},'domain_distill_joint_restart');new=overlay_config({},'domain_distill_joint_teacherproject')
    assert projection_enabled(new) and new.pop('teacher_gradient_projection') is True
    for cfg in (base,new):cfg.pop('protocol')
    assert base==new


@pytest.mark.parametrize('extra',[{'sam_rho':.02},{'view_consistency_weight':.1},{'vision_patch_teacher_weight':.1},{'router_optimizer_coordinates':'radians'},{'teacher_gradient_projection':1}])
def test_conflicting_training_contract_rejected(extra):
    cfg={'teacher_gradient_projection':True};cfg.update(extra)
    with pytest.raises(ValueError):projection_enabled(cfg)


def test_shared_projection_matches_flat_formula_for_multiple_parameters():
    torch.manual_seed(42);ps=[torch.nn.Parameter(torch.randn(3)),torch.nn.Parameter(torch.randn(2,2))]
    pg=[torch.randn_like(p) for p in ps];tg=[-g+torch.randn_like(g)*.1 for g in pg]
    flatp=torch.cat([g.flatten() for g in pg]);flatt=torch.cat([g.flatten() for g in tg])
    expected=flatp+flatt-min(float(flatp@flatt),0.)/float(flatp@flatp)*flatp
    def closure():
        primary=sum((p*g).sum() for p,g in zip(ps,pg));kd=sum((p*g).sum() for p,g in zip(ps,tg))
        return dict(loss=primary+kd,primary_loss=primary,teacher_loss=kd)
    backward_primary_teacher(closure,torch.optim.SGD(ps,lr=.1))
    torch.testing.assert_close(torch.cat([p.grad.flatten() for p in ps]),expected,rtol=1e-6,atol=1e-6)
