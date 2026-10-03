"""Bounded train-only per-image CC emphasis; no sampler/model/eval changes.

Inspired by hard-example focusing, NOT the classification Focal Loss formula.
Both arms keep all original GT/KD/optical losses. Uniform control has the same
sum of detached weights at a given prediction, isolating their allocation.
"""
import math
import torch
import torch.nn.functional as F


def configure(settings, raw):
    settings.hard_example_cc = dict(raw or {})
    if not raw:
        return
    if set(raw) != {'mode', 'weight', 'reference_error', 'minimum', 'maximum'}:
        raise ValueError('Hard CC requires exact audited options')
    if raw['mode'] not in {'bounded', 'uniform'}:
        raise ValueError('Invalid hard CC mode')
    for name in ('weight', 'reference_error', 'minimum', 'maximum'):
        if type(raw[name]) not in (int, float) or not math.isfinite(raw[name]):
            raise ValueError('Hard CC options must be finite real numbers')
    if not (0 < raw['weight'] <= 2 and .05 <= raw['reference_error'] <= .5
            and 0 < raw['minimum'] <= 1 <= raw['maximum'] <= 2):
        raise ValueError('Hard CC weights must remain bounded')
    if (settings.sam_rho <= 0 or settings.distillation_loss != 'spatial_cc'
            or settings.teacher_only_epochs or settings.augmentation_enabled
            or settings.fixed_crop_distillation or settings.first_stage_supervision
            or settings.masked_distillation or settings.relational_distillation
            or settings.feature_pretraining or settings.feature_hint_initial_weight
            or settings.unlabeled_weight or settings.semantic_weight
            or settings.fusion_alpha_min < .4 or settings.top_k != 2):
        raise ValueError('Hard CC requires isolated GT+KD/SAM optical training')


def error_and_weights(logits, target, options):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected matching [B,1,H,W] logits and GT')
    x = logits.float().flatten(1).softmax(-1)
    y = target.detach().float().flatten(1).clamp_min(0)
    y = y / y.sum(1, keepdim=True).clamp_min(1e-20)
    x = (x-x.mean(1, keepdim=True))*x.shape[1]
    y = (y-y.mean(1, keepdim=True))*y.shape[1]
    valid = torch.linalg.vector_norm(y, dim=1) > 1e-6
    error = 1-F.cosine_similarity(x, y, dim=1, eps=1e-6).clamp(-1, 1)
    weights = (error.detach()/options['reference_error']).clamp(options['minimum'], options['maximum'])
    if options['mode'] == 'uniform':
        mean = (weights*valid).sum()/valid.sum().clamp_min(1)
        weights = mean.expand_as(weights)
    return error, weights, valid


def loss(logits, target, options):
    error, weights, valid = error_and_weights(logits, target, options)
    denominator = valid.sum().clamp_min(1)
    value = (error*weights*valid).sum()/denominator
    return value, {'hard_cc_loss': value,
                   'hard_cc_weight_mean': (weights*valid).sum()/denominator,
                   'hard_cc_valid_fraction': valid.float().mean()}
