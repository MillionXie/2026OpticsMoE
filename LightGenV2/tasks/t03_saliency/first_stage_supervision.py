"""Training-only deep supervision of the EXISTING first fused optical stage.

No extra propagation or deployed branch. The temporary decoder and hook exist
only in the trainer; core/head inference state and forward are unchanged.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
import torch
from torch import nn
from experiments.vision2_hybrid_dense.modeling import restore_qwen_block_major_spatial
from .training_support import task_saliency_loss


def configure(settings, raw):
    options = dict(raw or {})
    settings.first_stage_supervision = options
    if not options:
        return
    if set(options) != {'initial_weight', 'final_weight', 'end_epoch', 'learning_rate', 'head_warmup_epochs'}:
        raise ValueError('First-stage supervision requires the exact audited option set')
    for key in ('initial_weight', 'final_weight', 'learning_rate'):
        if type(options[key]) not in (int, float) or not math.isfinite(options[key]):
            raise ValueError('Invalid first-stage supervision scalar')
    if not (0 <= options['final_weight'] <= options['initial_weight'] <= .5
            and options['initial_weight'] > 0 and 0 < options['learning_rate'] <= .001):
        raise ValueError('First-stage supervision scalar out of range')
    if (type(options['end_epoch']) is not int or type(options['head_warmup_epochs']) is not int
            or not 0 <= options['head_warmup_epochs'] < options['end_epoch'] <= settings.student_epochs):
        raise ValueError('Invalid first-stage supervision epoch boundaries')
    if (settings.sam_rho <= 0 or settings.augmentation_enabled or settings.feature_hint_initial_weight
            or settings.unlabeled_weight or settings.semantic_weight or settings.teacher_only_epochs
            or settings.feature_pretraining.get('enabled', False) or settings.relational_distillation
            or settings.masked_distillation or settings.reset_fusion_on_warmstart
            or not settings.initialization_checkpoint or settings.distillation_initial_weight <= 0
            or settings.distillation_loss != 'spatial_cc' or settings.fusion_alpha_min < .4
            or settings.kl_weight <= 0 or settings.cc_weight <= 0
            or settings.lightgen_model_variant != 'optical_router_scale_matched_moe'
            or settings.image_size != 224 or settings.electronic_width != 192 or settings.top_k != 2):
        raise ValueError('First-stage trial requires isolated original GT+KD/SAM optical MoE')


class FirstStageSupervisor(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Copy the initialized source head, without consuming RNG or sharing weights.
        self.decoder = deepcopy(model.head).requires_grad_(True)
        count = sum(p.numel() for p in self.decoder.parameters())
        if count != 85412:
            raise ValueError('First-stage auxiliary decoder must match the original 85412-parameter head')
        self.provenance = dict(training_auxiliary_parameters=count, inference_parameters_added=0,
            tap='input to existing hybrid.blocks[1], i.e. first scale-matched fused E/O latent',
            spatial_contract='[B,196,192] Qwen block-major -> [B,192,14,14] row-major',
            initialization='independent deepcopy of initialized original student head; no RNG draw',
            loss='same GT KL/CC/SIM/NSS weights; no teacher target in this auxiliary loss',
            extra_optical_propagations=0, extra_inference_branches=0,
            checkpoint='last_checkpoint.pt:training_only_first_stage; absent from best/core/head')

    @contextmanager
    def capture(self, model):
        captured = []

        def collect(_module, args):
            if len(captured) or not args or args[0].ndim != 3 or args[0].shape[1:] != (196, 192):
                raise RuntimeError('Expected exactly one first-fusion [B,196,192] tensor')
            captured.append(args[0])  # Keep gradients, do not mutate or detach here.

        handle = model.core.hybrid.blocks[1].register_forward_pre_hook(collect)
        try:
            yield captured
            if len(captured) != 1:
                raise RuntimeError('First-fusion capture missing; refusing stale/fallback features')
        finally:
            handle.remove()

    def loss(self, captured, grid, density, fixation, settings, *, detach_student=False):
        if len(captured) != 1:
            raise RuntimeError('First-fusion capture must contain one current forward')
        value = captured[0]
        expected = grid.new_tensor([1, 14, 14]).expand(len(value), 3)
        if grid.shape != expected.shape or not torch.equal(grid, expected):
            raise ValueError('First-stage auxiliary requires fixed 1x14x14 image grids')
        if detach_student:
            value = value.detach()
        with torch.autocast(device_type=value.device.type, enabled=False):
            spatial = restore_qwen_block_major_spatial(value.float().reshape(-1, 192), grid)
            logits = self.decoder(spatial)
            if density.shape != logits.shape or fixation.shape != logits.shape:
                raise ValueError('First-stage GT maps must match the original full 224 output')
            loss, pieces = task_saliency_loss(logits, density, fixation, settings, teacher_logits=None)
        return loss, pieces['cc']
