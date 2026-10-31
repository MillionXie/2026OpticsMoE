import pytest
import torch
from LightGenV2.tasks.t17_router_classification.phase_backtracking import CanonicalPhaseBacktracking
from LightGenV2.tasks.t17_router_classification.phase_training import phase_update_mode
from LightGenV2.tasks.t17_router_classification.train_figure_revision import configure_parameters
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG


def test_backtracking_reduces_same_periodic_objective_and_preserves_raw_gradient():
    p=torch.nn.Parameter(torch.tensor([-.8,.4],dtype=torch.float64))
    def loss():return (1-torch.cos(2*torch.pi*p.sigmoid()-.5)).sum()
    before=float(loss());loss().backward();grad=p.grad.clone()
    opt=CanonicalPhaseBacktracking([p],3.,max_trials=8);record=opt.step(loss,before)
    assert record['accepted'] and float(loss())<before
    assert torch.equal(p.grad,grad) and torch.isfinite(p).all()


def test_failed_trials_and_closure_exception_restore_exact_raw_values():
    p=torch.nn.Parameter(torch.tensor([-.8,.4]));p.grad=torch.ones_like(p);before=p.detach().clone()
    record=CanonicalPhaseBacktracking([p],.03,max_trials=3).step(lambda:2.,1.)
    assert not record['accepted'] and record['forward_trials']==3 and torch.equal(p,before)
    def error():raise RuntimeError('controlled failure')
    with pytest.raises(RuntimeError):CanonicalPhaseBacktracking([p],.03).step(error,1.)
    assert torch.equal(p,before)


def test_fixed_router_scope_and_policy_do_not_add_parameters():
    m=LinearReadoutLG(2,10);phases,heads=configure_parameters(m,dict(architecture='LG',training_scope='all_except_router'))
    assert not m.router_phase.requires_grad and sum(p.numel() for p in phases)==429188
    assert heads==[m.shared_head.weight] and sum(p.numel() for p in m.parameters())==487204
    cfg=dict(phase_update_mode='full_TRAIN_backtracking',architecture='LG',phase_optimizer='backtracking',alternating_head={'interval_epochs':5},training_scope='all_except_router')
    assert phase_update_mode(cfg)=='full_TRAIN_backtracking'
    with pytest.raises(ValueError):phase_update_mode(dict(cfg,training_scope='all'))


def test_actual_optical_model_accepted_update_keeps_router_head_and_top2_contract():
    torch.set_num_threads(4);torch.manual_seed(17);m=LinearReadoutLG(2,10)
    phases,heads=configure_parameters(m,dict(architecture='LG',training_scope='all_except_router'))
    x=torch.rand(2,224,224);y=torch.tensor([0,1]);before={n:p.detach().clone() for n,p in m.named_parameters()}
    def objective():return torch.nn.functional.cross_entropy(m(x)['raw_logits'],y)
    loss=objective();loss.backward();record=CanonicalPhaseBacktracking(phases,.003).step(objective,float(loss))
    assert record['accepted'] and objective()<loss
    assert torch.equal(m.router_phase,before['router_phase']) and torch.equal(m.shared_head.weight,before['shared_head.weight'])
    assert ((m(x)['route_power']>0).sum(1)==2).all()
