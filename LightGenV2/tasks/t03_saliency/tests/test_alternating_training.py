from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.alternating_training import (
    AlternatingSchedule, effective_sam_rho, validate,
)
from LightGenV2.tasks.t03_saliency.sam_training import sam_step
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA

ROOT = Path(__file__).resolve().parents[1]/'configs'


def make_problem():
    s = load_settings(ROOT/'moe_alpha40_alternating_20260914.yaml')
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.frontend = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
    model.core.e = torch.nn.Parameter(torch.tensor(.7))
    model.core.p = torch.nn.Parameter(torch.tensor(.4))
    model.core.r = torch.nn.Parameter(torch.tensor(.6))
    model.head = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.head.weight, .8)
    opt = torch.optim.AdamW([
        {'params': [model.core.e], 'name': 'electronic', 'lr': .01},
        {'params': [model.core.p], 'name': 'feature_phase', 'lr': .04},
        {'params': [model.core.r], 'name': 'optical_router', 'lr': .02},
        {'params': list(model.head.parameters()), 'name': 'saliency_head', 'lr': .01},
    ], weight_decay=.01)
    ema = ModelEMA(model, .9)
    hook = opt.register_step_post_hook(ema.update)
    schedule = AlternatingSchedule(opt, s, ema)
    def closure():
        # Electronics BEFORE fixed optical transform must still receive gradients.
        signal = torch.sin(model.core.e * model.core.p) * model.core.r
        loss = (model.head(signal.reshape(1, 1)) - .2).square().sum()
        return loss, {'loss': loss}
    return s, model, opt, ema, schedule, closure, hook


def test_real_sam_freezes_every_inactive_group_and_preserves_frontend():
    s, model, opt, ema, schedule, closure, hook = make_problem()
    for epoch, stage in [(1,'optics_only'), (10,'optics_only'),
                         (11,'electronics_only'), (25,'electronics_only'), (26,'joint_polish'), (30,'joint_polish')]:
        report = schedule.begin_epoch(epoch)
        assert report['stage'] == stage
        s.alternating_stage = stage
        if epoch in (11,26):
            assert not opt.state
            assert report['stage_boundary_reset']
            for key, module in ema.modules.items():
                for name, value in module.state_dict().items():
                    assert torch.equal(ema.shadow[key][name], value)
        rho = effective_sam_rho(s, opt)
        assert rho == (0 if stage == 'optics_only' else .05)
        sam_step(opt, closure, rho, torch.device('cpu'))
        changes = schedule.end_epoch()
        for group in opt.param_groups:
            update = changes['epoch_raw_update_rms_' + group['name']]
            assert (update > 0) == group['params'][0].requires_grad
        assert model.core.frontend.item() == 2 and not model.core.frontend.requires_grad
        if stage == 'electronics_only':
            assert model.core.e.grad is not None and model.core.e.grad.abs() > 0
            assert torch.equal(ema.shadow['core']['p'], model.core.p)
        assert model.core.p.grad is None if stage == 'electronics_only' else model.core.p.grad is not None
    hook.remove()


def test_frozen_drift_and_bad_partition_are_rejected():
    s, model, opt, ema, schedule, closure, hook = make_problem()
    schedule.begin_epoch(1)
    with torch.no_grad():
        model.core.e.add_(.01)
    with pytest.raises(RuntimeError, match='frozen'):
        schedule.end_epoch()
    s.alternating_stage = 'optics_only'
    model.core.e.requires_grad_(True)
    with pytest.raises(RuntimeError, match='partition'):
        effective_sam_rho(s, opt)
    hook.remove()


def test_stage_handoff_does_not_revert_live_optics_to_initialization_or_ema():
    s, model, opt, ema, schedule, closure, hook = make_problem()
    schedule.begin_epoch(1)
    s.alternating_stage = 'optics_only'
    initial = model.core.p.detach().clone()
    sam_step(opt, closure, effective_sam_rho(s,opt), torch.device('cpu'))
    schedule.end_epoch()
    live = model.core.p.detach().clone()
    assert not torch.equal(initial, live) and not torch.equal(ema.shadow['core']['p'], live)
    schedule.begin_epoch(11)
    assert torch.equal(model.core.p, live) and torch.equal(ema.shadow['core']['p'], live)
    hook.remove()


def test_profile_contract_and_serialized_schedule(tmp_path):
    import yaml
    s = load_settings(ROOT/'moe_alpha40_alternating_20260914.yaml')
    old = load_settings(ROOT/'moe_alpha40_sam_batch8_crosssample_20260913.yaml')
    assert architecture_label(s) == architecture_label(old)
    for key in ['top_k','router_backend','fusion_alpha_min','active_size','expert_size',
                'pixel_pitch_um','student_batch_size','sam_rho','ema_decay','phase_weight_decay',
                'distillation_initial_weight','distillation_final_weight','router_balance_estimator']:
        assert getattr(s,key) == getattr(old,key)
    for key in vars(old):
        if key.startswith(('language_optical_', 'optical_router_')):
            assert getattr(s,key) == getattr(old,key)
    assert s.student_epochs == 30 and not old.alternating
    s.output_dir = tmp_path
    save_resolved_config(s)
    saved = yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())
    assert saved['training']['alternating'] == s.alternating
    for bad in [dict(s.alternating, optics_epochs=0), dict(s.alternating, joint_epochs=6),
                dict(s.alternating, end_factor=float('nan')), dict(s.alternating, unused=True)]:
        with pytest.raises(ValueError):
            validate(bad,s)
    s.adaptive_plateau_enabled = True
    with pytest.raises(ValueError):
        validate(s.alternating,s)


def test_non_alternating_sam_unchanged():
    assert effective_sam_rho(SimpleNamespace(sam_rho=.05), None) == .05
