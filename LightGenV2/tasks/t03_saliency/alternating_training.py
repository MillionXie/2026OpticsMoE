"""Training-only optics -> electronics -> joint schedule, with exact freeze audits."""
from __future__ import annotations

import math
import torch

OPTICAL = frozenset({'feature_phase', 'optical_router'})
ELECTRONIC = frozenset({'electronic', 'ccd_readout', 'saliency_head',
                        'electronic_ffn_spatial', 'electronic_global_spatial'})


def validate(raw, settings):
    if not raw:
        return {}
    allowed = {'optics_epochs', 'electronics_epochs', 'joint_epochs',
               'end_factor', 'joint_factor'}
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise ValueError('alternating requires exactly ' + ', '.join(sorted(allowed)))
    for key in ('optics_epochs', 'electronics_epochs', 'joint_epochs'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError('alternating stage lengths must be positive integers')
    for key in ('end_factor', 'joint_factor'):
        if not math.isfinite(float(raw[key])) or not 0 < float(raw[key]) <= 1:
            raise ValueError('alternating LR factors must be in (0,1]')
    if sum(raw[k] for k in ('optics_epochs', 'electronics_epochs', 'joint_epochs')) != settings.student_epochs:
        raise ValueError('alternating lengths must sum to student_epochs')
    incompatible = ('asam', 'gsam_coefficient', 'noise_consistency_weight', 'mixup',
                    'pyramid_cc', 'teacher_only_epochs', 'augmentation_enabled',
                    'fixed_crop_distillation', 'hard_example_cc', 'teacher_reliability',
                    'first_stage_supervision', 'masked_distillation', 'relational_distillation',
                    'feature_pretraining', 'feature_hint_initial_weight', 'unlabeled_weight',
                    'semantic_weight', 'adaptive_plateau_enabled')
    if (any(getattr(settings, k, False) for k in incompatible)
            or not settings.staged_training or settings.sam_rho <= 0
            or settings.distillation_loss != 'spatial_cc'
            or settings.fusion_alpha_min < .4 or settings.top_k != 2
            or settings.router_backend != 'optical'
            or settings.router_phase_coordinates != 'sigmoid' or settings.phase_parameterization != 'sigmoid'):
        raise ValueError('alternating requires isolated optical Top2/alpha40 SAM GT+CC-KD')
    return dict(raw)


class AlternatingSchedule:
    """Only original optimizer members may be unfrozen; Qwen is never included.

    No model restoration at stage boundaries: the next stage adapts to the LIVE
    result, even if its test score fell. Global best selection remains separate.
    """
    def __init__(self, optimizer, settings, ema=None):
        self.optimizer, self.settings, self.ema = optimizer, settings, ema
        self.options = settings.alternating
        groups = optimizer.param_groups
        names = [g['name'] for g in groups]
        if (len(names) != len(set(names)) or not OPTICAL.issubset(names)
                or not set(names) <= OPTICAL | ELECTRONIC):
            raise ValueError('Unexpected optimizer partition for alternating training')
        params = [p for g in groups for p in g['params']]
        if len(params) != len({id(p) for p in params}) or not all(p.requires_grad for p in params):
            raise ValueError('Alternating needs unique, originally trainable parameters')
        self.base_rates = {g['name']: g['lr'] for g in groups}
        self.previous_stage = None
        self.frozen = []
        self.before = {}

    def begin_epoch(self, epoch):
        o = self.options
        n, m, j = o['optics_epochs'], o['electronics_epochs'], o['joint_epochs']
        if not 1 <= epoch <= n + m + j:
            raise ValueError('Epoch outside alternating schedule')
        if epoch <= n:
            stage, local, length, active = 'optics_only', epoch, n, OPTICAL
        elif epoch <= n + m:
            stage, local, length, active = 'electronics_only', epoch-n, m, ELECTRONIC
        else:
            stage, local, length, active = 'joint_polish', epoch-n-m, j, OPTICAL | ELECTRONIC
        boundary = self.previous_stage is not None and self.previous_stage != stage
        if boundary:
            # Momentum from a different coordinate subproblem should not kick the
            # new stage. Reset EMA to live: frozen phases must not keep drifting
            # only in the shadow model used for testing.
            self.optimizer.state.clear()
            if self.ema is not None:
                self.ema.sync_to_live()
        self.previous_stage = stage
        factor = o['end_factor'] + (1-o['end_factor']) * (1+math.cos(math.pi*(local-1)/max(1,length-1)))/2
        if stage == 'joint_polish':
            factor *= o['joint_factor']
        self.frozen, self.before = [], {}
        for group in self.optimizer.param_groups:
            enabled = group['name'] in active
            group['lr'] = self.base_rates[group['name']] * factor if enabled else 0.
            self.before[group['name']] = [p.detach().clone() for p in group['params']]
            for p in group['params']:
                p.requires_grad_(enabled)
                p.grad = None
                if not enabled:
                    self.frozen.append((p, p.detach().clone()))
        return {'stage': stage, 'stage_epoch': local, 'stage_boundary_reset': boundary,
                'effective_sam_rho': 0. if stage == 'optics_only' else self.settings.sam_rho,
                'hard_balance_weight': self.settings.router_hard_load_balance_weight,
                **{f"lr_{g['name']}": g['lr'] for g in self.optimizer.param_groups}}

    @torch.no_grad()
    def end_epoch(self):
        if any(not torch.equal(p, old) for p, old in self.frozen):
            raise RuntimeError('A frozen alternating parameter changed; refusing checkpoint')
        result = {'frozen_parameters_verified': sum(p.numel() for p, _ in self.frozen)}
        for group in self.optimizer.param_groups:
            count, squares = 0, 0.
            for p, old in zip(group['params'], self.before[group['name']]):
                if not torch.isfinite(p).all():
                    raise RuntimeError('Nonfinite alternating weights')
                squares += float((p.double()-old.double()).square().sum())
                count += p.numel()
            result['epoch_raw_update_rms_' + group['name']] = math.sqrt(squares/max(1,count))
        self.frozen, self.before = [], {}
        return result

    def is_stage_end(self, epoch):
        o = self.options
        return epoch in (o['optics_epochs'], o['optics_epochs'] + o['electronics_epochs'],
                         self.settings.student_epochs)


def effective_sam_rho(settings, optimizer):
    """Disable SAM's *electronic* perturbation only when electronics are frozen.

    Keep train_sam_epoch's identical GT + spatial-CC KD closure; do not fall back
    to the legacy trainer (whose distillation semantics differ).
    """
    if not getattr(settings, 'alternating', {}):
        return settings.sam_rho
    stage = settings.alternating_stage
    if stage not in {'optics_only', 'electronics_only', 'joint_polish'}:
        raise ValueError('Unknown alternating stage')
    for group in optimizer.param_groups:
        enabled = (stage == 'joint_polish' or
                   (group['name'] in OPTICAL) == (stage == 'optics_only'))
        if any(p.requires_grad != enabled for p in group['params']):
            raise RuntimeError('Alternating/SAM freeze partition mismatch')
    return 0. if stage == 'optics_only' else settings.sam_rho
