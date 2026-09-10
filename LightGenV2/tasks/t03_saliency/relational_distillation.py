"""Training-only spatial-pair distillation; no projector or inference parameters.

Inspired by Liu et al., CVPR 2019 structured knowledge distillation. We use
centered 14x14 fused features, not their GAN, segmentation labels or backbone.
The Gram matrix is a LOSS target, never attention or a feature-mixing layer.
"""
import math
import torch
from torch.nn import functional as F
from .feature_pretraining import _read_pinned


def configure(settings, raw, directory):
    options = dict(raw or {})
    settings.relational_distillation = options
    if not options:
        return
    required = {'cache_file', 'cache_sha256', 'initial_weight', 'final_weight', 'end_epoch'}
    if set(options) != required:
        raise ValueError('Relational distillation requires the exact audited option set')
    for key in ('initial_weight', 'final_weight'):
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid relational weight')
    if not 0 <= options['final_weight'] <= options['initial_weight'] <= 10 or options['initial_weight'] == 0:
        raise ValueError('Relational weights require 0 <= final <= initial <= 10, initial > 0')
    if type(options['end_epoch']) is not int or not 2 <= options['end_epoch'] <= settings.student_epochs:
        raise ValueError('Invalid relational decay boundary')
    digest = options['cache_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Relational cache must be SHA256-pinned')
    options['cache_file'] = str((directory / options['cache_file']).resolve())
    if (settings.sam_rho <= 0 or settings.augmentation_enabled or settings.feature_hint_initial_weight
            or settings.unlabeled_weight or settings.semantic_weight or settings.teacher_only_epochs
            or settings.feature_pretraining.get('enabled', False) or settings.reset_fusion_on_warmstart
            or not settings.initialization_checkpoint or settings.distillation_initial_weight <= 0
            or settings.distillation_loss != 'spatial_cc' or settings.fusion_alpha_min < .4
            or settings.kl_weight <= 0 or settings.cc_weight <= 0):
        raise ValueError('Relational trial requires clean GT+spatial-CC KD/SAM, preserved optical source and alpha>=.4')


def spatial_relations(value):
    if value.ndim != 4 or value.shape[-2] * value.shape[-1] < 2:
        raise ValueError('Relations require a spatial BCHW feature map with at least two positions')
    value = value.float().flatten(2)
    value = value - value.mean(-1, keepdim=True)
    value = F.normalize(value, dim=1, eps=1e-6)
    return value.transpose(1, 2) @ value


def pairwise_loss(student, target):
    if student.shape[0] != target.shape[0] or student.shape[-2:] != target.shape[-2:]:
        raise ValueError('Pairwise features must align sample and spatial identities')
    # Channel bases need not agree; target is always detached.
    a, b = spatial_relations(student), spatial_relations(target.detach())
    off_diagonal = ~torch.eye(a.shape[-1], device=a.device, dtype=torch.bool)
    return (a[:, off_diagonal] - b[:, off_diagonal]).square().mean()


class RelationalTargets:
    def __init__(self, settings, records):
        options = settings.relational_distillation
        payload = _read_pinned(options['cache_file'], options['cache_sha256'])
        ids = [r.sample_id for r in records]
        if (not ids or ids != payload['sample_ids'] or len(set(ids)) != len(ids)
                or any(not k.startswith('train/') for k in ids)):
            raise ValueError('Relational cache must match ordered unique training identities')
        manifest = payload['manifest']
        if (manifest.get('checkpoint_sha256') != settings.distillation_teacher_sha256
                or manifest.get('feature_contract') != 'aligned_decoder_input_192x14x14_row_major_v1'
                or manifest.get('split') != 'train_only' or manifest.get('augmentation') is not False
                or manifest.get('image_size') != settings.image_size):
            raise ValueError('Relational teacher/preprocessing contract mismatch')
        self.values = payload['features'].detach().cpu()
        if (self.values.shape != (len(ids), 192, 14, 14) or not self.values.is_floating_point()
                or any(not torch.isfinite(chunk).all() for chunk in self.values.split(64))):
            raise ValueError('Invalid relational feature tensor')
        self.index = {key: i for i, key in enumerate(ids)}
        self.provenance = dict(manifest, **options, train_samples=len(ids),
            loss='MSE of off-diagonal spatial cosine similarities after per-channel spatial centering',
            training_projection_parameters=0, inference_parameters_added=0,
            replaces_student_head=False, changes_inference_features=False)

    def loss(self, spatial, ids):
        if spatial.shape != (len(ids), 192, 14, 14):
            raise ValueError('Relational student grid must be [B,192,14,14]')
        target = self.values[[self.index[k] for k in ids]].to(spatial.device, dtype=torch.float32)
        return pairwise_loss(spatial, target)
