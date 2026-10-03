from pathlib import Path
from types import SimpleNamespace

import pytest

from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.training import staged_epoch


def test_only_hard_balance_schedule_is_relaxed():
    root = Path(__file__).resolve().parents[1] / 'configs'
    base = load_settings(root / 'moe_alpha40_hint_control.yaml')
    trial = load_settings(root / 'moe_alpha40_soften_hard_balance.yaml')
    assert architecture_label(trial) == architecture_label(base)
    for key in ('initialization_checkpoint_sha256', 'student_learning_rate',
                'phase_learning_rate', 'router_learning_rate', 'dense_head_learning_rate',
                'dense_readout_learning_rate', 'router_balance_weight', 'router_importance_weight',
                'distillation_initial_weight', 'distillation_final_weight', 'feature_hint_initial_weight',
                'fusion_alpha_min', 'fusion_alpha_max', 'ccd_normalization', 'active_size',
                'expert_size', 'top_k', 'router_backend', 'student_epochs', 'student_batch_size',
                'language_optical_phase_zero_order_intensity_min',
                'language_optical_phase_zero_order_intensity_max'):
        assert getattr(trial, key) == getattr(base, key), key
    assert trial.fusion_alpha_min == .4 and trial.top_k == 2
    assert trial.router_backend == 'optical' and trial.feature_hint_initial_weight == 0
    assert base.router_hard_load_balance_weight == pytest.approx(.10)
    assert base.staged_final_hard_balance == pytest.approx(.10)
    assert trial.router_hard_load_balance_weight == pytest.approx(.01)
    assert trial.staged_final_hard_balance == pytest.approx(.01)
    for epoch in (1, 20, 41, 50):
        optim = lambda: SimpleNamespace(param_groups=[{'name': 'feature_phase', 'lr': .001}])
        first = staged_epoch(optim(), base, epoch)
        second = staged_epoch(optim(), trial, epoch)
        assert second['hard_balance_weight'] == pytest.approx(first['hard_balance_weight'] * .1)
        assert second['lr_feature_phase'] == first['lr_feature_phase']
