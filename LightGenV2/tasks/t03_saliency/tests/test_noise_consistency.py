from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.sam_training import (
    symmetric_density_kl, noise_consistent_closure, sam_step,
)
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def test_symmetric_kl_matches_bidirectional_kl_and_has_two_gradients():
    a=torch.randn(3,1,4,4,dtype=torch.float64,requires_grad=True)
    b=torch.randn_like(a,requires_grad=True)
    p=a.flatten(1).softmax(1);q=b.flatten(1).softmax(1)
    expected=.5*((p*(p.log()-q.log())).sum(1)+(q*(q.log()-p.log())).sum(1)).mean()
    actual=symmetric_density_kl(a,b)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual,symmetric_density_kl(b,a))
    torch.testing.assert_close(actual,symmetric_density_kl(a+3,b-2))
    assert symmetric_density_kl(a,a)==0
    actual.backward()
    assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in (a,b))


def test_consistency_zero_keeps_single_forward_and_weighted_pair_is_correct():
    seen=[];values=[torch.randn(2,1,3,3,requires_grad=True) for _ in range(2)]
    def view():
        x=values[len(seen)];seen.append(1);loss=x.square().mean()
        return loss,{'loss':loss,'cc':loss*2},x
    loss,pieces=noise_consistent_closure(view,0)
    assert len(seen)==1 and loss is pieces['loss']
    seen.clear();actual,pieces=noise_consistent_closure(view,2)
    expected=(values[0].square().mean()+values[1].square().mean())/2+2*symmetric_density_kl(*values)
    torch.testing.assert_close(actual,expected)
    assert len(seen)==2 and pieces['loss'] is actual


@pytest.mark.parametrize('fail',[False,True])
def test_two_noisy_views_replayed_inside_sam_and_exact_restoration(fail):
    torch.manual_seed(42)
    e=torch.nn.Parameter(torch.tensor([.2,.3]));phase=torch.nn.Parameter(torch.tensor(.4))
    original=e.detach().clone();old_phase=phase.detach().clone()
    opt=torch.optim.SGD([{'params':[e],'name':'electronic'},
                         {'params':[phase],'name':'feature_phase'}],lr=.01)
    draws=[];states=[];phase_seen=[];steps=[]
    opt.register_step_post_hook(lambda *args:steps.append(1))
    def view():
        noise=torch.rand(2);draws.append(noise);states.append(torch.get_rng_state());phase_seen.append(phase.item())
        if fail and len(draws)==4:raise RuntimeError('deliberate fourth view failure')
        logits=(e*noise+phase).reshape(1,1,1,2);loss=logits.square().mean()
        return loss,{'loss':loss},logits
    def run():sam_step(opt,lambda:noise_consistent_closure(view,2),.05,torch.device('cpu'))
    if fail:
        with pytest.raises(RuntimeError,match='fourth'):run()
        assert torch.equal(e,original) and torch.equal(phase,old_phase) and not steps
    else:
        run();assert len(steps)==1 and not torch.equal(e,original) and not torch.equal(phase,old_phase)
    assert len(draws)==4 and not torch.equal(draws[0],draws[1])
    assert torch.equal(draws[0],draws[2]) and torch.equal(draws[1],draws[3])
    assert torch.equal(torch.get_rng_state(),states[1])
    assert all(x==float(old_phase) for x in phase_seen)


def test_consistency_profile_preserves_all_inference_and_noise_contracts(tmp_path):
    import yaml
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_extra_control.yaml')
    b=load_settings(root/'moe_alpha40_noise_consistency_20260913.yaml')
    assert a.noise_consistency_weight==0 and b.noise_consistency_weight==2
    assert b.student_epochs==20 and b.staged_polish_start==16
    assert architecture_label(a)==architecture_label(b)
    for k in ['initialization_checkpoint_sha256','student_batch_size','student_learning_rate',
              'phase_learning_rate','router_learning_rate','ema_decay','sam_rho','fusion_alpha_min',
              'router_backend','top_k','active_size','expert_size','pixel_pitch_um',
              'router_balance_weight','router_importance_weight','router_balance_estimator',
              'router_hard_load_balance_weight','phase_dc_weight','ccd_operating_point_weight',
              'distillation_initial_weight','distillation_final_weight']:
        assert getattr(a,k)==getattr(b,k)
    for k in vars(a):
        if k.startswith('language_optical_') or k.startswith('optical_router_'):
            assert getattr(a,k)==getattr(b,k),k
    b.output_dir=tmp_path;save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['training']['noise_consistency_weight']==2
    invalid=tmp_path/'invalid.yaml'
    invalid.write_text(f'base_config: {(root/"moe_alpha40_noise_consistency_20260913.yaml").as_posix()}\ntraining:\n  gsam_coefficient: 0.1\n')
    with pytest.raises(ValueError,match='Noise consistency'):load_settings(invalid)


@pytest.mark.parametrize('weight',[-1,6,float('nan'),float('inf')])
def test_consistency_rejects_unbounded_weights(weight):
    with pytest.raises(ValueError):noise_consistent_closure(None,weight)
