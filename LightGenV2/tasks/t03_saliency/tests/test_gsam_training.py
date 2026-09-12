from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.sam_training import sam_step
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def test_gsam_formula_only_corrects_electronics_and_steps_once():
    e=torch.nn.Parameter(torch.tensor([.7,1.2],dtype=torch.float64))
    phase=torch.nn.Parameter(torch.tensor(.4,dtype=torch.float64))
    frozen=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64),requires_grad=False)
    opt=torch.optim.SGD([{'params':[e],'name':'electronic'},
                         {'params':[phase],'name':'feature_phase'}],lr=.03)
    original=e.detach().clone(); original_phase=phase.item(); grads=[]; seen=[]; steps=[]
    e.register_hook(lambda g:grads.append(g.clone()))
    phase_grads=[];phase.register_hook(lambda g:phase_grads.append(g.clone()))
    opt.register_step_post_hook(lambda *args:steps.append(1))
    def closure():
        seen.append((e.detach().clone(),phase.item()))
        value=(e[0]+phase).square()+3*(e[1]-phase).square()+frozen
        return value,{'loss':value}
    values,_=sam_step(opt,closure,.05,torch.device('cpu'),gsam_coefficient=.1)
    a,b=grads
    perpendicular=a-(a@b)/(b@b)*b
    torch.testing.assert_close(e,original-.03*(b-.1*perpendicular))
    assert abs(float(perpendicular@b))<1e-12
    assert values['gsam_relative_correction']>0
    assert len(steps)==1 and len(seen)==2
    assert seen[0][1]==seen[1][1]==original_phase
    torch.testing.assert_close(phase,torch.tensor(original_phase,dtype=phase.dtype)-.03*phase_grads[1])
    assert frozen.item()==.2 and frozen.grad is None


def test_gsam_zero_matches_sam_bitwise_and_zero_grad_is_safe():
    def run(coefficient):
        e=torch.nn.Parameter(torch.tensor([.2,.5]))
        opt=torch.optim.AdamW([{'params':[e],'name':'electronic'}],lr=.001)
        def closure():
            v=e.square().sum();return v,{'loss':v}
        kwargs={} if coefficient is None else {'gsam_coefficient':coefficient}
        sam_step(opt,closure,.05,torch.device('cpu'),**kwargs)
        return e.detach()
    assert torch.equal(run(None),run(0.))
    e=torch.nn.Parameter(torch.zeros(2));opt=torch.optim.SGD([{'params':[e],'name':'electronic'}],lr=.1)
    values,_=sam_step(opt,lambda:(e.square().sum(),{}),.05,torch.device('cpu'),gsam_coefficient=.1)
    assert torch.equal(e,torch.zeros(2)) and values['gsam_relative_correction']==0


def test_gsam_failed_second_pass_restores_weights_rng_and_never_steps():
    e=torch.nn.Parameter(torch.tensor([.2,.4]));old=e.detach().clone()
    opt=torch.optim.AdamW([{'params':[e],'name':'electronic'}],lr=.001)
    draws=[];states=[]
    def closure():
        noise=torch.rand_like(e);draws.append(noise);states.append(torch.get_rng_state())
        if len(draws)==2:raise RuntimeError('deliberate failure')
        loss=(e*noise).square().sum();return loss,{}
    with pytest.raises(RuntimeError,match='deliberate'):
        sam_step(opt,closure,.05,torch.device('cpu'),gsam_coefficient=.1)
    assert torch.equal(e,old) and torch.equal(draws[0],draws[1])
    assert torch.equal(torch.get_rng_state(),states[0]) and not opt.state


@pytest.mark.parametrize('coefficient',[-.1,float('nan'),float('inf'),.21])
def test_gsam_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError,match='GSAM'):
        sam_step(None,None,.05,torch.device('cpu'),gsam_coefficient=coefficient)


def test_gsam_config_contract_and_serialization(tmp_path):
    import yaml
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_extra_control.yaml')
    b=load_settings(root/'moe_alpha40_gsam_20260913.yaml')
    assert a.gsam_coefficient==0 and b.gsam_coefficient==.1
    assert architecture_label(a)==architecture_label(b)
    assert b.student_epochs==20 and b.staged_polish_start==16
    for k in ['initialization_checkpoint_sha256','student_batch_size','student_learning_rate',
              'phase_learning_rate','router_learning_rate','sam_rho','ema_decay',
              'fusion_alpha_min','router_backend','top_k','router_balance_estimator',
              'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(a,k)==getattr(b,k)
    b.output_dir=tmp_path;save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['training']['gsam_coefficient']==.1
    invalid=tmp_path/'invalid.yaml'
    invalid.write_text(f'base_config: {(root/"moe_alpha40_gsam_20260913.yaml").as_posix()}\ntraining:\n  sam_rho: 0\n')
    with pytest.raises(ValueError):load_settings(invalid)
