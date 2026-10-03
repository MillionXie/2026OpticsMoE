from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.hard_example_cc import error_and_weights, loss
from LightGenV2.tasks.t03_saliency.training_support import task_saliency_loss


def options(mode='bounded'):
    return dict(mode=mode, weight=1.5, reference_error=.125, minimum=.5, maximum=2.)


def test_bounded_and_uniform_same_total_and_no_weight_gradient():
    torch.manual_seed(7)
    target=torch.rand(4,1,8,8)
    logits=(target+.01).log().requires_grad_()
    with torch.no_grad():
        logits[2:].normal_()
    e,w,v=error_and_weights(logits,target,options())
    _,u,_=error_and_weights(logits,target,options('uniform'))
    assert v.all() and not w.requires_grad
    assert w.min() >= .5 and w.max() <= 2
    assert w[:2].mean() < w[2:].mean()
    torch.testing.assert_close(w.sum(),u.sum())
    loss(logits,target,options())[0].backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


@pytest.mark.parametrize('constant_target',[False,True])
def test_constant_cases_finite(constant_target):
    logits=torch.zeros(2,1,8,8,requires_grad=True)
    target=torch.ones_like(logits) if constant_target else torch.rand_like(logits)
    result,_=loss(logits,target,options())
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(logits.grad).all()
    if constant_target: assert result==0


def test_permutation_invariance_and_detached_gt():
    torch.manual_seed(9)
    logits=torch.randn(3,1,8,8,requires_grad=True)
    target=torch.rand_like(logits,requires_grad=True)
    a,_=loss(logits,target,options()); b,_=loss(logits.flip(0),target.flip(0),options())
    torch.testing.assert_close(a,b)
    a.backward(); assert target.grad is None


def test_opt_in_addition_keeps_base_loss():
    torch.manual_seed(2)
    s=SimpleNamespace(distillation_loss='spatial_cc',map_kd_temperature=1,
        map_kd_weight=2.,kl_weight=1.,cc_weight=1.5,sim_weight=.25,nss_weight=.1)
    x=torch.randn(2,1,8,8,requires_grad=True); y=torch.rand_like(x); f=y>.5
    teacher=torch.randn_like(x)
    base,pieces=task_saliency_loss(x,y,f,s,teacher_logits=teacher)
    s.hard_example_cc={}
    disabled,_=task_saliency_loss(x,y,f,s,teacher_logits=teacher)
    assert torch.equal(base,disabled)
    s.hard_example_cc=options()
    new,out=task_saliency_loss(x,y,f,s,teacher_logits=teacher)
    extra,_=loss(x,y,options())
    torch.testing.assert_close(new,base+1.5*extra)
    for k in pieces: torch.testing.assert_close(pieces[k],out[k])


def test_configs_and_saved_contract(tmp_path):
    from LightGenV2.tasks.t03_saliency.settings import load_settings,save_resolved_config
    import yaml
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_hardcc_uniform30.yaml')
    b=load_settings(root/'moe_alpha40_hardcc_bounded30.yaml')
    assert a.student_epochs==b.student_epochs==30
    assert a.initialization_checkpoint==b.initialization_checkpoint
    assert a.hard_example_cc==options('uniform') and b.hard_example_cc==options()
    assert b.fusion_alpha_min==.4 and b.top_k==2 and b.staged_polish_start==21
    b.output_dir=tmp_path
    save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['hard_example_cc']==options()


def test_sam_two_pass_update_with_hard_cc():
    from LightGenV2.tasks.t03_saliency.sam_training import sam_step
    x=torch.nn.Parameter(torch.randn(2,1,8,8)); y=torch.rand_like(x)
    optimizer=torch.optim.AdamW([{'params':[x],'name':'electronic'}],lr=.001)
    calls=[]
    def closure():
        calls.append(1); return loss(x,y,options())
    before=x.detach().clone()
    sam_step(optimizer,closure,.05,torch.device('cpu'))
    assert len(calls)==2 and not torch.equal(before,x) and torch.isfinite(x).all()
