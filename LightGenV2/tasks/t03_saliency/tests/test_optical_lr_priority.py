from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label, optimizer
from LightGenV2.tasks.t03_saliency.training import staged_epoch
from LightGenV2.tasks.t03_saliency.lightweight_residual import configure_spatial_ffn
from .test_lightweight_residual import TinyFusion

ROOT=Path(__file__).resolve().parents[1]/'configs'


def test_optical_priority_only_changes_source_output_and_learning_rates():
    old=load_settings(ROOT/'moe_alpha40_sam_batch8_crosssample_20260913.yaml')
    new=load_settings(ROOT/'moe_alpha40_optical_lr_priority_20260913.yaml')
    allowed={'initialization_checkpoint','initialization_checkpoint_sha256','output_dir',
             'student_learning_rate','phase_learning_rate','router_learning_rate',
             'dense_readout_learning_rate','dense_head_learning_rate','ffn_spatial_learning_rate',
             'config_path','raw_config'}
    assert architecture_label(old)==architecture_label(new)
    for name in ['student_batch_size','sam_rho','ema_decay','gradient_clip_norm','phase_weight_decay',
                 'fusion_alpha_min','top_k','router_backend','active_size','expert_size','pixel_pitch_um',
                 'distillation_initial_weight','distillation_final_weight','distillation_teacher_sha256',
                 'router_balance_estimator','router_balance_weight','router_importance_weight',
                 'student_epochs','staged_warmup_epochs','staged_polish_start','mixup']:
        assert getattr(old,name)==getattr(new,name),name
    for name in vars(old):
        if name.startswith(('language_optical_','optical_router_')):
            assert getattr(old,name)==getattr(new,name),name
    for name,factor in [('phase_learning_rate',40),('router_learning_rate',20),
                        ('student_learning_rate',.1),('dense_readout_learning_rate',.1),
                        ('dense_head_learning_rate',.1),('ffn_spatial_learning_rate',.1)]:
        assert getattr(new,name)==pytest.approx(getattr(old,name)*factor)
    assert new.student_epochs==20 and new.staged_warmup_epochs==0 and new.staged_polish_start==16
    assert not new.mixup and not new.noise_consistency_weight


def test_actual_optimizer_groups_and_schedule_preserve_lr_ratios():
    s=load_settings(ROOT/'moe_alpha40_optical_lr_priority_20260913.yaml')
    model=torch.nn.Module();model.core=torch.nn.Module();model.head=torch.nn.Linear(192,1)
    model.core.hybrid=TinyFusion(s);configure_spatial_ffn(model.core.hybrid,1)
    model.core.phase=torch.nn.Parameter(torch.zeros(2));model.core.phase_parameters=lambda:[model.core.phase]
    model.core.router=torch.nn.Linear(2,2);model.core.readout=torch.nn.Linear(2,2)
    opt=optimizer(model,s)
    expected={'electronic':2.5e-7,'feature_phase':.002,'optical_router':.0001,
              'ccd_readout':5e-7,'saliency_head':5e-7,'electronic_ffn_spatial':1.25e-6}
    assert {g['name']:g['lr'] for g in opt.param_groups}==expected
    for epoch in [1,5,16,20]:
        staged_epoch(opt,s,epoch)
        groups={g['name']:g for g in opt.param_groups}
        factors=[g['lr']/expected[g['name']] for g in opt.param_groups]
        assert min(factors)==pytest.approx(max(factors))
        assert groups['feature_phase']['lr']/groups['electronic']['lr']==pytest.approx(8000)
        assert groups['optical_router']['lr']/groups['electronic']['lr']==pytest.approx(400)
        assert all(p.requires_grad for g in opt.param_groups for p in g['params'])
