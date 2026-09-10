"""Fixed-basis feature pretraining; no added inference modules or parameters.

The existing decoder is initialized from the SHA-pinned aligned teacher head.
Only training targets use spatial centering/scaling. This does NOT subtract
background from the deployed CCD or change its normalization contract.
"""
from collections import defaultdict
from copy import copy
import hashlib
import io
import math
from pathlib import Path
import time

import torch
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from .sam_training import sam_step
from .training_support import task_saliency_loss


def configure(settings, raw, directory):
    options = dict(raw or {})
    settings.feature_pretraining = options
    if not options.get('enabled', False):
        return
    required = {'enabled', 'cache_file', 'cache_sha256', 'head_checkpoint',
                'head_checkpoint_sha256', 'frozen_head_epochs', 'fade_end_epoch',
                'initial_weight', 'joint_weight', 'mean_weight', 'joint_lr_multiplier'}
    if set(options) != required:
        raise ValueError('Feature pretraining requires the exact audited option set')
    for key in ('cache_file', 'head_checkpoint'):
        options[key] = str((directory / options[key]).resolve())
        digest = options[key.replace('file', 'sha256')] if key == 'cache_file' else options['head_checkpoint_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Feature pretraining inputs must be SHA256-pinned')
    if options['head_checkpoint_sha256'] != settings.distillation_teacher_sha256:
        raise ValueError('Head and feature/map teacher identities must agree')
    for key in ('frozen_head_epochs', 'fade_end_epoch'):
        if type(options[key]) is not int:
            raise ValueError('Feature stage boundaries must be integers')
    if not 1 <= options['frozen_head_epochs'] < options['fade_end_epoch'] <= settings.student_epochs:
        raise ValueError('Invalid feature pretraining stage boundaries')
    for key in ('initial_weight', 'joint_weight', 'mean_weight', 'joint_lr_multiplier'):
        if not isinstance(options[key], (int, float)) or not 0 < options[key] < float('inf'):
            raise ValueError('Invalid feature pretraining weight/rate')
    if options['joint_weight'] > options['initial_weight'] or options['joint_lr_multiplier'] > 1:
        raise ValueError('Joint stage must reduce feature weight and learning rates')
    if (settings.augmentation_enabled or settings.feature_hint_initial_weight or settings.unlabeled_weight
            or settings.semantic_weight or settings.teacher_only_epochs or settings.adaptive_plateau_enabled
            or not settings.staged_training or settings.staged_warmup_epochs
            or settings.sam_rho <= 0 or settings.distillation_initial_weight <= 0
            or not settings.initialization_checkpoint or settings.reset_fusion_on_warmstart):
        raise ValueError('Isolate feature pretraining with clean train inputs, GT+KD, joint SAM and retained fusion')


def _read_pinned(path, digest):
    # Bind the parsed checkpoint to precisely the bytes whose digest was checked.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Feature pretraining SHA256 mismatch')
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)


class FixedFeatureTargets:
    def __init__(self, settings, records):
        options = settings.feature_pretraining
        payload = _read_pinned(options['cache_file'], options['cache_sha256'])
        ids = [record.sample_id for record in records]
        if (ids != payload['sample_ids'] or len(ids) != len(set(ids)) or not ids
                or any(not key.startswith('train/') for key in ids)):
            raise ValueError('Features require exactly the ordered, unique training identities')
        manifest = payload['manifest']
        if (manifest.get('checkpoint_sha256') != options['head_checkpoint_sha256']
                or manifest.get('feature_contract') != 'aligned_decoder_input_192x14x14_row_major_v1'
                or manifest.get('augmentation') is not False or manifest.get('split') != 'train_only'
                or manifest.get('image_size') != settings.image_size):
            raise ValueError('Feature teacher/preprocessing contract mismatch')
        values = payload['features']
        if values.shape != (len(ids), 192, 14, 14) or not values.is_floating_point():
            raise ValueError('Invalid feature cache tensor')
        self.values = values.detach().cpu()
        self.index = {key: index for index, key in enumerate(ids)}
        # Statistics exclusively over training images, with bounded CPU memory.
        sums = torch.zeros(192, dtype=torch.float64)
        mean_sums = torch.zeros_like(sums)
        for batch in self.values.split(64):
            if not torch.isfinite(batch).all():
                raise ValueError('Nonfinite training features')
            batch = batch.double()
            means = batch.mean((-2, -1), keepdim=True)
            sums += (batch-means).square().mean((-2, -1)).sum(0)
            mean_sums += means.square().flatten(1).sum(0)
        self.spatial_scale = (sums/len(ids)).sqrt().clamp_min(.05).float().view(1,192,1,1)
        self.mean_scale = (mean_sums/len(ids)).sqrt().clamp_min(.05).float().view(1,192,1,1)
        self.provenance = dict(manifest, cache_sha256=options['cache_sha256'],
                               statistics_split='train_only', train_samples=len(ids),
                               spatial_scale=self.spatial_scale.flatten().tolist(),
                               mean_scale=self.mean_scale.flatten().tolist(),
                               inference_parameters_added=0, training_projection_parameters=0)

    def loss(self, spatial, ids, mean_weight):
        if spatial.shape != (len(ids),192,14,14):
            raise ValueError('Student feature grid must be [B,192,14,14]')
        target = self.values[[self.index[key] for key in ids]].to(spatial.device, dtype=torch.float32)
        student = spatial.float()
        sm, tm = student.mean((-2,-1),keepdim=True), target.mean((-2,-1),keepdim=True)
        local = ((student-sm-target+tm)/self.spatial_scale.to(spatial.device)).square().mean()
        mean = ((sm-tm)/self.mean_scale.to(spatial.device)).square().mean()
        return local + mean_weight*mean, local, mean


def initialize_teacher_decoder(model, settings):
    options = settings.feature_pretraining
    payload = _read_pinned(options['head_checkpoint'], options['head_checkpoint_sha256'])
    decoder = {key[len('decoder.'):]: value for key,value in payload['head'].items()
               if key.startswith('decoder.')}
    expected = model.head.state_dict()
    if (set(decoder) != set(expected) or any(decoder[k].shape != expected[k].shape
            or not torch.isfinite(decoder[k]).all() for k in expected)):
        raise ValueError('Teacher decoder must match the existing student head exactly')
    model.head.load_state_dict(decoder, strict=True)
    return {'head_initialization': 'same_size_aligned_teacher_decoder',
            'head_source_sha256': options['head_checkpoint_sha256'],
            'head_parameters': sum(p.numel() for p in model.head.parameters()),
            'core_weights_unchanged_by_head_initialization': True,
            'inference_parameters_added': 0}


def prepare_epoch(model, optimizer, settings, epoch):
    options = settings.feature_pretraining
    frozen = epoch <= options['frozen_head_epochs']
    model.head.requires_grad_(not frozen)
    current = copy(settings)
    current.sam_rho = 0.0 if frozen else settings.sam_rho
    progress = max(0., (epoch-options['frozen_head_epochs']-1)
                   / max(1, settings.student_epochs-options['frozen_head_epochs']-1))
    multiplier = 1.0 if frozen else options['joint_lr_multiplier']*(.05+.95*(1+math.cos(math.pi*progress))/2)
    for group in optimizer.param_groups:
        group.setdefault('schedule_base_lr', group['lr'])
        group['lr'] = group['schedule_base_lr'] * (0.0 if frozen and group['name']=='saliency_head' else multiplier)
    weight = options['initial_weight'] if frozen else options['joint_weight']*max(
        0., 1-(epoch-options['frozen_head_epochs']-1)/max(1,options['fade_end_epoch']-options['frozen_head_epochs']-1))
    report = {'feature_stage': 'fixed_teacher_decoder' if frozen else 'joint_task_finetune',
              'feature_weight': weight, 'effective_sam_rho': current.sam_rho,
              'head_frozen': frozen, **{f"lr_{g['name']}":g['lr'] for g in optimizer.param_groups}}
    return current, report


def train_epoch(model, loader, loaded, settings, optimizer, teacher, targets, feature_weight):
    model.train()
    totals = defaultdict(float)
    started = time.perf_counter()
    for index,batch in enumerate(loader,1):
        ids = batch['sample_ids']
        inputs = legacy.preprocess_vision(loaded.processor,batch['images'],loaded.device)
        density,fixation = (batch[key].to(loaded.device) for key in ('density','fixation'))
        teacher_logits = teacher.get(ids,loaded.device) if teacher is not None else None
        def closure():
            with legacy._autocast(settings,loaded.device):
                logits,spatial,_ = model(inputs['pixel_values'],inputs['image_grid_thw'])
                task,pieces = task_saliency_loss(logits,density,fixation,settings,teacher_logits=teacher_logits)
                balance,importance = model.router_losses()
                operating = model.operating_loss() if hasattr(model,'operating_loss') else logits.new_zeros(())
                dc = legacy.phase_dc_loss(model) if settings.phase_dc_weight else logits.new_zeros(())
                if feature_weight:
                    feature,local,mean = targets.loss(spatial,ids,settings.feature_pretraining['mean_weight'])
                else:
                    feature = local = mean = logits.new_zeros(())
                total = (task + feature_weight*feature + settings.router_balance_weight*balance
                         + settings.router_importance_weight*importance + settings.phase_dc_weight*dc
                         + getattr(settings,'ccd_operating_point_weight',0.)*operating)
            return total, dict(pieces,loss=total,feature_loss=feature,feature_spatial_mse=local,
                               feature_mean_mse=mean,router_balance=balance,router_importance=importance,
                               phase_dc=dc,ccd_operating_point=operating)
        values,increase = sam_step(optimizer,closure,settings.sam_rho,loaded.device,settings.gradient_clip_norm)
        totals['samples'] += len(ids)
        for key,value in dict(values,sam_loss_increase=increase).items():
            totals[key] += float(value)*len(ids)
        if index % settings.log_interval_batches == 0 or index == len(loader):
            print(f"[student feature-pretrain] batch={index}/{len(loader)} CC={totals['cc']/totals['samples']:.5f} "
                  f"feature={totals['feature_loss']/totals['samples']:.5f}",flush=True)
    count = totals.pop('samples')
    return {key:value/count for key,value in totals.items()} | {'samples':int(count),'epoch_time_sec':time.perf_counter()-started}
