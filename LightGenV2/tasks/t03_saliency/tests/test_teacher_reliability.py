from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.teacher_reliability import loss
from LightGenV2.tasks.t03_saliency.training_support import spatial_correlation_distillation, task_saliency_loss
from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import objectives

TASK = Path(__file__).resolve().parents[1]
OPTIONS = {'mode': 'relative_cc', 'temperature': .025, 'minimum': .25}


def test_good_teacher_preserves_historical_kd_and_bad_teacher_is_bounded():
    torch.manual_seed(23)
    student = torch.randn(2, 1, 8, 8, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    good_gt = objectives.density_from_logits(teacher.detach())
    reference = spatial_correlation_distillation(student, teacher)
    value, info = loss(student, teacher, good_gt, OPTIONS)
    torch.testing.assert_close(value, reference)
    assert info['teacher_reliability_mean'].item() == 1
    better_gt = objectives.density_from_logits(student.detach())
    reduced, info = loss(student, teacher, better_gt, OPTIONS)
    torch.testing.assert_close(reduced, .25*reference)
    assert info['teacher_reliability_mean'].item() == .25
    reduced.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_uniform_maps_and_shape_checks():
    student = torch.randn(2, 1, 8, 8, requires_grad=True)
    teacher = torch.zeros_like(student)
    value, _ = loss(student, teacher, torch.ones_like(student), OPTIONS)
    value.backward()
    assert value.item() == 0 and torch.isfinite(student.grad).all()
    teacher = torch.randn_like(student)
    value, info = loss(student, teacher, torch.ones_like(student), OPTIONS)
    torch.testing.assert_close(value, spatial_correlation_distillation(student, teacher))
    assert info['teacher_reliability_mean'].item() == 1
    with pytest.raises(ValueError):
        loss(student, teacher[:1], torch.ones_like(student), OPTIONS)


def test_profile_changes_only_training_kd_and_persists(tmp_path):
    s = load_settings(TASK/'configs/moe_alpha40_reliable_teacher50.yaml')
    previous = load_settings(TASK/'configs/moe_alpha40_sam_spatialcc_kd2.yaml')
    assert s.teacher_reliability == OPTIONS and not previous.teacher_reliability
    assert architecture_label(s) == architecture_label(previous)
    for key in ('student_epochs', 'sam_rho', 'initialization_checkpoint_sha256',
                'phase_learning_rate', 'student_learning_rate', 'augmentation_enabled',
                'fusion_alpha_min', 'top_k', 'active_size', 'expert_size',
                'language_optical_phase_zero_order_intensity_min',
                'language_optical_phase_zero_order_intensity_max'):
        assert getattr(s, key) == getattr(previous, key)
    s.output_dir = tmp_path
    save_resolved_config(s)
    import yaml
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['teacher_reliability'] == OPTIONS
    s.map_kd_weight = 2.
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    teacher = torch.randn_like(logits)
    target = torch.rand_like(logits)
    fixation = target > .9
    total, metrics = task_saliency_loss(logits, target, fixation, s, teacher_logits=teacher)
    gt_only, gt_metrics = objectives.saliency_loss(logits, target, fixation, s)
    kd, _ = loss(logits, teacher, target, OPTIONS)
    torch.testing.assert_close(total, gt_only+2*kd)
    for key in ('kl', 'cc', 'sim', 'nss'):
        assert torch.equal(metrics[key], gt_metrics[key])
    without_teacher, _ = task_saliency_loss(logits, target, fixation, s)
    assert torch.equal(without_teacher, gt_only)
