"""Training-only extra-image distillation with uninterrupted SALICON GT loss.

The extra stream contains images and teacher predictions, never invented human
fixations. Both SAM passes reuse the SAME labeled and unlabeled minibatches.
"""
from collections import defaultdict
import time

import torch
from torch.utils.data import DataLoader

from .sam_training import sam_step
from .training_support import task_saliency_loss, spatial_correlation_distillation
from .unlabeled_data import AuditedUnlabeledImages, UnlabeledTeacherMaps, collate_unlabeled
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


class CyclingUnlabeledBatches:
    def __init__(self, loader, teacher):
        if not len(loader):
            raise ValueError('Empty unlabeled loader')
        self.loader, self.teacher = loader, teacher
        self.iterator = None
        self.restarts = 0
        self.closed = False

    def next_batch(self):
        if self.closed:
            raise RuntimeError('Unlabeled stream is closed')
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.restarts += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def close(self):
        # Dropping the DataLoader iterator releases its non-persistent workers.
        self.iterator = None
        self.closed = True


def build_unlabeled_stream(settings):
    dataset = AuditedUnlabeledImages(settings.unlabeled_image_manifest,
                                     settings.unlabeled_image_manifest_sha256, settings.data_root)
    teacher = UnlabeledTeacherMaps(settings.unlabeled_cache, settings.unlabeled_cache_sha256,
                                  dataset, settings.distillation_teacher_sha256)
    loader = DataLoader(dataset, batch_size=settings.student_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(settings.random_seed + 149),
                        num_workers=settings.num_workers, collate_fn=collate_unlabeled,
                        drop_last=False)
    return CyclingUnlabeledBatches(loader, teacher)


def mixed_objective(supervised_task, extra_kd, labeled_regularizer, extra_regularizer, phase_dc, weight):
    if not 0 < weight <= 2:
        raise ValueError('Extra-image distillation weight must be in (0,2]')
    # Do not dilute GT by dividing its loss by the enlarged minibatch size.
    return supervised_task + weight * extra_kd + .5 * (labeled_regularizer + extra_regularizer) + phase_dc


def train_semisupervised_epoch(model, loader, loaded, settings, optimizer, teacher_cache, extra_stream):
    if settings.sam_rho <= 0 or settings.teacher_only_epochs or settings.augmentation_enabled:
        raise ValueError('Extra-image trial requires SAM, uninterrupted GT and unaugmented inputs')
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError('SAM with BatchNorm buffers is not supported')
    model.train()
    totals = defaultdict(float)
    count_labeled = count_extra = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        extra = extra_stream.next_batch()  # Outside closure: exactly once per optimizer step.
        n, u = len(batch['sample_ids']), len(extra['sample_ids'])
        density = batch['density'].to(loaded.device)
        fixation = batch['fixation'].to(loaded.device)
        inputs = legacy.preprocess_vision(loaded.processor, batch['images'], loaded.device)
        extra_inputs = legacy.preprocess_vision(loaded.processor, extra['images'], loaded.device)
        target = teacher_cache.get(batch['sample_ids'],loaded.device) if teacher_cache is not None else None
        extra_target = extra_stream.teacher.get(extra['sample_ids'], loaded.device)
        def closure():
            with legacy._autocast(settings,loaded.device):
                logits = model(inputs['pixel_values'],inputs['image_grid_thw'])[0]
                task, pieces = task_saliency_loss(logits,density,fixation,settings,teacher_logits=target)
                balance, importance = model.router_losses()
                operating = model.operating_loss() if hasattr(model,'operating_loss') else logits.new_zeros(())
                regularizer = (settings.router_balance_weight*balance + settings.router_importance_weight*importance
                               + getattr(settings,'ccd_operating_point_weight',0.)*operating)
                # Capture labeled routing losses before the second forward replaces routing state.
                extra_logits = model(extra_inputs['pixel_values'],extra_inputs['image_grid_thw'])[0]
                extra_kd = spatial_correlation_distillation(extra_logits, extra_target)
                ubalance, uimportance = model.router_losses()
                uoperating = model.operating_loss() if hasattr(model,'operating_loss') else extra_logits.new_zeros(())
                extra_regularizer = (settings.router_balance_weight*ubalance + settings.router_importance_weight*uimportance
                                     + getattr(settings,'ccd_operating_point_weight',0.)*uoperating)
                dc = legacy.phase_dc_loss(model) if settings.phase_dc_weight>0 else logits.new_zeros(())
                loss = mixed_objective(task,extra_kd,regularizer,extra_regularizer,
                                       settings.phase_dc_weight*dc,settings.unlabeled_weight)
            return loss, dict(pieces, loss=loss, router_balance=balance, router_importance=importance,
                              phase_dc=dc, ccd_operating_point=operating, unlabeled_map_kd=extra_kd,
                              unlabeled_router_balance=ubalance, unlabeled_router_importance=uimportance,
                              unlabeled_ccd_operating_point=uoperating)
        values, increase = sam_step(optimizer,closure,settings.sam_rho,loaded.device,settings.gradient_clip_norm)
        values['sam_loss_increase'] = increase
        count_labeled += n
        count_extra += u
        for key,value in values.items():
            totals[key] += float(value) * (u if key.startswith('unlabeled_') else n)
        if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
            print(f'[student extra-image SAM] batch={batch_index}/{len(loader)} '
                  f'GT_CC={totals["cc"]/count_labeled:.5f} '
                  f'extra_KD={totals["unlabeled_map_kd"]/count_extra:.5f}',flush=True)
    if not count_labeled:
        raise ValueError('Empty supervised training loader')
    return {k:v/(count_extra if k.startswith('unlabeled_') else count_labeled) for k,v in totals.items()} | {
        'samples':count_labeled, 'unlabeled_samples':count_extra,
        'unlabeled_loader_restarts':extra_stream.restarts, 'epoch_time_sec':time.perf_counter()-started}
