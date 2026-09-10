from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.training_support import supervision_for_epoch, task_saliency_loss

TASK = Path(__file__).resolve().parents[1]


def test_gt_gradient_removed_only_in_pretrain_then_restored():
    s = SimpleNamespace(teacher_only_epochs=15, kl_weight=1., cc_weight=1.5,
                        sim_weight=.25, nss_weight=.1, map_kd_weight=2.,
                        map_kd_temperature=1., distillation_loss='spatial_cc')
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    a, b = torch.rand_like(logits), torch.rand_like(logits)
    pre, report = supervision_for_epoch(s, 15)
    assert pre is not s and s.cc_weight == 1.5
    assert report['supervision_stage'] == 'teacher_only'
    def gradient(target, settings):
        loss, _ = task_saliency_loss(logits, target, target > .9, settings, teacher_logits=teacher)
        return torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(gradient(a, pre), gradient(b, pre), rtol=0, atol=0)
    assert gradient(a, pre).abs().sum() > 0 and teacher.grad is None
    post, report = supervision_for_epoch(s, 16)
    assert post is s and post.cc_weight == 1.5
    assert report['supervision_stage'] == 'ground_truth_plus_teacher'
    assert not torch.equal(gradient(a, post), gradient(b, post))
    s.teacher_only_epochs = 0
    assert supervision_for_epoch(s, 1)[0] is s
    with pytest.raises(ValueError): supervision_for_epoch(s, 0)
    s.teacher_only_epochs = 15
    s.map_kd_weight = 0
    with pytest.raises(ValueError): supervision_for_epoch(s, 1)


def test_pair_changes_only_supervision_duration_and_output():
    a = load_settings(TASK/'configs/moe_alpha40_reheat_joint.yaml')
    b = load_settings(TASK/'configs/moe_alpha40_teacher_pretrain15.yaml')
    assert a.teacher_only_epochs == 0 and b.teacher_only_epochs == 15
    assert architecture_label(a) == architecture_label(b)
    for name in ('student_epochs','sam_rho','student_learning_rate','phase_learning_rate',
                 'router_learning_rate','dense_head_learning_rate','ffn_spatial_learning_rate',
                 'initialization_checkpoint_sha256','distillation_initial_weight',
                 'distillation_final_weight','distillation_cache','augmentation_enabled',
                 'staged_polish_start','fusion_alpha_min','top_k','router_backend',
                 'active_size','expert_size','electronic_ffn_hidden_width','electronic_global_rank',
                 'kl_weight','cc_weight','sim_weight','nss_weight',
                 'language_optical_phase_zero_order_intensity_min',
                 'language_optical_phase_zero_order_intensity_max'):
        assert getattr(a, name) == getattr(b, name), name
    assert a.student_epochs == 60 and not a.adaptive_plateau_enabled
    assert a.fusion_alpha_min == .4 and a.top_k == 2


@pytest.mark.parametrize('override', [
    'distillation:\n  teacher_only_epochs: -1',
    'distillation:\n  teacher_only_epochs: 60',
    'distillation:\n  teacher_only_epochs: 1.5',
    'distillation:\n  teacher_only_epochs: true',
    'distillation:\n  final_weight: 0',
    'training:\n  sam_rho: 0',
    'training:\n  adaptive_plateau:\n    enabled: true',
])
def test_invalid_curriculum_rejected(tmp_path, override):
    path = tmp_path/'LightGenV2/tasks/t03_saliency/configs/invalid.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(f'base_config: {(TASK/"configs/moe_alpha40_teacher_pretrain15.yaml").as_posix()}\n{override}\n')
    with pytest.raises(ValueError): load_settings(path)


def test_resolved_config_preserves_stage_and_eval_weights(tmp_path):
    import yaml
    s = load_settings(TASK/'configs/moe_alpha40_teacher_pretrain15.yaml')
    s.output_dir = tmp_path
    s.map_kd_weight = 2.
    supervision_for_epoch(s, 1)
    save_resolved_config(s)
    raw = yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())
    assert raw['distillation']['teacher_only_epochs'] == 15
    assert s.cc_weight == 1.5
    assert raw['protocol']['primary_metric'] == 'CC'
