from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.training_support import spatial_correlation_distillation,task_saliency_loss
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import objectives

TASK=Path(__file__).resolve().parents[1]


def test_positive_affine_density_invariance_and_gradients():
    teacher=torch.randn(3,1,14,14,requires_grad=True)
    prob=objectives.density_from_logits(teacher.detach())
    shifted=(.3*prob+.7/prob[0].numel()).log().requires_grad_()
    torch.testing.assert_close(spatial_correlation_distillation(shifted,teacher),torch.tensor(0.),atol=2e-7,rtol=0)
    logits=torch.randn_like(teacher,requires_grad=True)
    loss=spatial_correlation_distillation(logits,teacher)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    singles=torch.stack([spatial_correlation_distillation(logits[i:i+1],teacher[i:i+1]) for i in range(3)])
    torch.testing.assert_close(loss,singles.mean())
    assert objectives.kl_divergence(objectives.density_from_logits(shifted),prob)>0


@pytest.mark.parametrize('size',[8,224])
def test_uniform_empty_information_and_shape_checks(size):
    student=torch.zeros(2,1,size,size,requires_grad=True)
    teacher=torch.zeros_like(student)
    loss=spatial_correlation_distillation(student,teacher);loss.backward()
    assert loss.item()==0 and torch.isfinite(student.grad).all()
    student.grad=None;teacher=torch.randn_like(student)
    spatial_correlation_distillation(student,teacher).backward()
    assert torch.isfinite(student.grad).all()
    with pytest.raises(ValueError):spatial_correlation_distillation(student,teacher[:1])


def test_legacy_kl_exact_and_gt_loss_unchanged():
    s=SimpleNamespace(kl_weight=1.,cc_weight=1.5,sim_weight=.25,nss_weight=.1,
                      map_kd_weight=.6,map_kd_temperature=1.)
    logits=torch.randn(2,1,8,8);teacher=torch.randn_like(logits)
    target=torch.rand_like(logits);fixation=target>.98
    actual,p=task_saliency_loss(logits,target,fixation,s,teacher_logits=teacher)
    expected,q=objectives.saliency_loss(logits,target,fixation,s,teacher_logits=teacher)
    assert torch.equal(actual,expected) and p.keys()==q.keys()
    for k in p:assert torch.equal(p[k],q[k])
    s.distillation_loss='spatial_cc'
    actual,p=task_saliency_loss(logits,target,fixation,s,teacher_logits=teacher)
    base,q=objectives.saliency_loss(logits,target,fixation,s,teacher_logits=None)
    for k in ['kl','cc','sim','nss']:assert torch.equal(p[k],q[k])
    torch.testing.assert_close(actual,base+.6*spatial_correlation_distillation(logits,teacher))
    actual,_=task_saliency_loss(logits,target,fixation,s)
    assert torch.equal(actual,base)


def test_profile_is_loss_only_and_invalid_mode_is_rejected(tmp_path):
    s=load_settings(TASK/'configs/moe_alpha40_sam_spatialcc.yaml')
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    assert s.distillation_loss=='spatial_cc' and architecture_label(s)==architecture_label(base)
    for k in ['student_epochs','sam_rho','initialization_checkpoint_sha256','augmentation_enabled',
              'distillation_initial_weight','distillation_final_weight','fusion_alpha_min','top_k',
              'router_backend','active_size','expert_size','electronic_ffn_hidden_width',
              'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(s,k)==getattr(base,k)
    # Loader's existing cache discovery expects a task-depth config location.
    path=tmp_path/'LightGenV2/tasks/t03_saliency/configs/bad.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(f'base_config: {(TASK/"configs/moe_alpha40_sam_spatialcc.yaml").as_posix()}\ntraining:\n  sam_rho: 0\n')
    with pytest.raises(ValueError):load_settings(path)


def test_stronger_spatial_cc_changes_only_teacher_strength():
    s=load_settings(TASK/'configs/moe_alpha40_sam_spatialcc_kd2.yaml')
    base=load_settings(TASK/'configs/moe_alpha40_sam_spatialcc.yaml')
    assert s.distillation_initial_weight==s.distillation_final_weight==2.
    assert base.distillation_initial_weight==base.distillation_final_weight==.6
    assert architecture_label(s)==architecture_label(base)
    for k in ['student_epochs','sam_rho','initialization_checkpoint_sha256','augmentation_enabled',
              'distillation_end_epoch','distillation_loss','fusion_alpha_min','top_k','router_backend',
              'active_size','expert_size','electronic_ffn_hidden_width','electronic_ffn_groups',
              'kl_weight','cc_weight','sim_weight','nss_weight','student_learning_rate',
              'phase_learning_rate','router_learning_rate','dense_head_learning_rate',
              'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(s,k)==getattr(base,k)
