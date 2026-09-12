"""Train-only reliability weighting, not an inference module or a GT filter.

CA-MKD (arXiv:2201.00007) motivates using training labels to assess teacher
reliability. This single-teacher, relative spatial-CC rule is our adaptation,
NOT a reproduction of its multi-teacher classification algorithm.
"""
import math
import torch
import torch.nn.functional as F


def configure(settings, raw):
    settings.teacher_reliability = dict(raw or {})
    if not raw:
        return
    if set(raw) != {'mode', 'temperature', 'minimum'} or raw['mode'] != 'relative_cc':
        raise ValueError('Teacher reliability requires the audited relative_cc options')
    if any(type(raw[k]) not in (int, float) or not math.isfinite(raw[k])
           for k in ('temperature', 'minimum')):
        raise ValueError('Teacher reliability options must be finite real numbers')
    if not (.01 <= raw['temperature'] <= .1 and .1 <= raw['minimum'] <= 1):
        raise ValueError('Teacher reliability weights must remain bounded')
    if (settings.distillation_loss != 'spatial_cc' or settings.sam_rho <= 0
            or settings.distillation_initial_weight <= 0 or settings.teacher_only_epochs
            or settings.augmentation_enabled or settings.fixed_crop_distillation
            or settings.hard_example_cc or settings.first_stage_supervision
            or settings.masked_distillation or settings.relational_distillation
            or settings.feature_pretraining or settings.feature_hint_initial_weight
            or settings.unlabeled_weight or settings.semantic_weight
            or settings.fusion_alpha_min < .4 or settings.top_k != 2):
        raise ValueError('Reliability trial requires isolated GT+CC-KD/SAM optical training')


def loss(logits, teacher_logits, target, options):
    if logits.shape != teacher_logits.shape or logits.shape != target.shape:
        raise ValueError('Reliability requires exactly aligned student/teacher/GT maps')
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected [B,1,H,W] maps')
    x = logits.float().flatten(1).softmax(-1)
    y = teacher_logits.detach().float().flatten(1).softmax(-1)
    z = target.detach().float().flatten(1).clamp_min(0)
    z = z/z.sum(1, keepdim=True).clamp_min(1e-20)
    # Same stable centering as historical spatial-CC KD.
    x, y, z = [(v-v[:, :1])*v.shape[1] for v in (x, y, z)]
    x, y, z = [v-v.mean(1, keepdim=True) for v in (x, y, z)]
    valid_teacher = torch.linalg.vector_norm(y, dim=1) > 1e-6
    valid_gt = torch.linalg.vector_norm(z, dim=1) > 1e-6
    with torch.no_grad():
        student_cc = F.cosine_similarity(x.detach(), z, dim=1, eps=1e-6).clamp(-1, 1)
        teacher_cc = F.cosine_similarity(y, z, dim=1, eps=1e-6).clamp(-1, 1)
        disadvantage = (student_cc-teacher_cc).clamp_min(0)
        weights = torch.exp(-disadvantage/options['temperature']).clamp_min(options['minimum'])
        # Constant GT cannot rank teacher/student reliability. Keep ordinary KD.
        weights = torch.where(valid_gt, weights, torch.ones_like(weights))
    errors = 1-F.cosine_similarity(x, y, dim=1, eps=1e-6).clamp(-1, 1)
    count = valid_teacher.sum().clamp_min(1)
    value = (weights*errors*valid_teacher).sum()/count
    return value, {
        'teacher_reliability_mean': (weights*valid_teacher).sum()/count,
        'teacher_worse_fraction': ((teacher_cc < student_cc) & valid_gt & valid_teacher).float().mean(),
        'teacher_gt_cc': teacher_cc.mean(),
        'teacher_reliability_valid_gt_fraction': valid_gt.float().mean(),
    }
