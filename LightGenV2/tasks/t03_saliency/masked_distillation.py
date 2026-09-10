"""Training-only masked feature recovery, inspired by MGD (ECCV 2022).

SALICON adaptation: centered spatial patterns, a 96-channel generator, and
three generator warmup epochs. No generator/normalization/masking in inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .feature_pretraining import _read_pinned


def configure(settings, raw, directory):
    options = dict(raw or {})
    settings.masked_distillation = options
    if not options:
        return
    required = {'cache_file', 'cache_sha256', 'initial_weight', 'final_weight',
                'end_epoch', 'learning_rate', 'mask_probability', 'generator_warmup_epochs'}
    if set(options) != required:
        raise ValueError('Masked distillation requires the exact audited option set')
    for key in ('initial_weight', 'final_weight', 'learning_rate', 'mask_probability'):
        if type(options[key]) not in (int, float) or not math.isfinite(options[key]):
            raise ValueError('Invalid masked-distillation scalar')
    if not (0 <= options['final_weight'] <= options['initial_weight'] <= 2
            and options['initial_weight'] > 0 and 0 < options['learning_rate'] <= .001
            and 0 < options['mask_probability'] < 1):
        raise ValueError('Masked-distillation scalar out of range')
    if (type(options['end_epoch']) is not int or type(options['generator_warmup_epochs']) is not int
            or not 0 <= options['generator_warmup_epochs'] < options['end_epoch'] <= settings.student_epochs):
        raise ValueError('Invalid masked-distillation epoch boundaries')
    digest = options['cache_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Masked teacher cache must be SHA256-pinned')
    options['cache_file'] = str((directory / options['cache_file']).resolve())
    if (settings.sam_rho <= 0 or settings.augmentation_enabled or settings.feature_hint_initial_weight
            or settings.unlabeled_weight or settings.semantic_weight or settings.teacher_only_epochs
            or settings.feature_pretraining.get('enabled', False) or settings.relational_distillation
            or settings.reset_fusion_on_warmstart or not settings.initialization_checkpoint
            or settings.distillation_initial_weight <= 0 or settings.distillation_loss != 'spatial_cc'
            or settings.fusion_alpha_min < .4 or settings.kl_weight <= 0 or settings.cc_weight <= 0):
        raise ValueError('Masked trial requires isolated GT+CC KD/SAM and preserved optical source')


def spatial_patterns(value):
    """Only a training target/auxiliary-input transform, not CCD processing."""
    value = value.float()
    value = value - value.mean((-2, -1), keepdim=True)
    return value / value.square().mean((-2, -1), keepdim=True).clamp_min(.05**2).sqrt()


class MaskedTeacherRecovery(nn.Module):
    def __init__(self, settings, records):
        super().__init__()
        options = settings.masked_distillation
        payload = _read_pinned(options['cache_file'], options['cache_sha256'])
        ids = [r.sample_id for r in records]
        manifest = payload['manifest']
        if (not ids or ids != payload['sample_ids'] or len(set(ids)) != len(ids)
                or any(not k.startswith('train/') for k in ids)
                or manifest.get('checkpoint_sha256') != settings.distillation_teacher_sha256
                or manifest.get('feature_contract') != 'aligned_decoder_input_192x14x14_row_major_v1'
                or manifest.get('split') != 'train_only' or manifest.get('augmentation') is not False
                or manifest.get('image_size') != settings.image_size):
            raise ValueError('Masked teacher identity/preprocessing mismatch')
        self.values = payload['features'].detach().cpu()  # Not a model buffer.
        if (self.values.shape != (len(ids), 192, 14, 14) or not self.values.is_floating_point()
                or any(not torch.isfinite(v).all() for v in self.values.split(64))):
            raise ValueError('Invalid masked teacher feature tensor')
        self.index = {k: i for i, k in enumerate(ids)}
        self.mask_probability = options['mask_probability']
        # Do not shift the original student's dropout/data/optical-noise RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1042)
            self.generator = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1, bias=False),
                                           nn.ReLU(), nn.Conv2d(96, 192, 3, padding=1, bias=False))
        self.provenance = dict(manifest, **options, training_generator_parameters=331776,
            inference_parameters_added=0, replaces_student_head=False,
            generator='192 -> Conv3x3/96 -> ReLU -> Conv3x3/192; no bias/BN/attention',
            normalization='per-image per-channel spatial centering and RMS floor .05, training auxiliary only',
            mask='independent Bernoulli spatial mask Bx1x14x14, shared over channels',
            loss='mean squared error over ALL reconstructed spatial positions',
            checkpoint='last_checkpoint.pt:training_only_mgd; absent from best/core/head')

    def loss(self, spatial, ids, *, detach_student=False):
        if spatial.shape != (len(ids), 192, 14, 14):
            raise ValueError('Masked student grid must be [B,192,14,14]')
        target = self.values[[self.index[k] for k in ids]].to(spatial.device, dtype=torch.float32)
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            student = spatial_patterns(spatial.detach() if detach_student else spatial)
            keep = (torch.rand((len(ids), 1, 14, 14), device=spatial.device) >= self.mask_probability)
            prediction = self.generator(student * keep)
            return F.mse_loss(prediction, spatial_patterns(target.detach()))
