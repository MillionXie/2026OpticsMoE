"""Train-only feature hints; never a student inference branch or dependency."""
from __future__ import annotations

from collections import defaultdict
import time
import torch
from torch import nn
from torch.nn import functional as F
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


class TrainFeatureHints(nn.Module):
    def __init__(self, settings, records):
        super().__init__()
        if settings.augmentation_enabled:
            raise ValueError("Feature hints currently require exact unaugmented inputs")
        payload = torch.load(settings.feature_hint_cache, map_location="cpu", weights_only=False)
        ids = [r.sample_id for r in records]
        if ids != payload["sample_ids"] or len(set(ids)) != len(ids) or any(not k.startswith('train/') for k in ids):
            raise ValueError("Feature cache must exactly match ordered training identities")
        self.manifest = payload['manifest']
        if (self.manifest['checkpoint_sha256'] != settings.distillation_teacher_sha256
            or self.manifest.get('feature_contract') != 'aligned_decoder_input_192x14x14_row_major_v1'
            or self.manifest.get('augmentation') is not False or self.manifest.get('split') != 'train_only'
            or self.manifest.get('image_size') != settings.image_size):
            raise ValueError("Feature teacher/preprocessing contract mismatch")
        self.values = payload['features']  # CPU cache, not a registered buffer.
        if self.values.shape != (len(ids),192,14,14) or not torch.isfinite(self.values).all():
            raise ValueError("Invalid teacher features")
        self.index = {k:i for i,k in enumerate(ids)}
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(192,192,1,bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(192).reshape(192,192,1,1))

    def forward(self, spatial, ids):
        if spatial.shape != (len(ids),192,14,14):
            raise ValueError("Student hint grid must be exact [B,192,14,14]")
        target = self.values[[self.index[k] for k in ids]].to(spatial.device,dtype=torch.float32)
        student = self.projection(spatial.float())
        # Channel-basis alignment is learned only for the training loss. Per-pixel
        # cosine avoids forcing arbitrary teacher activation magnitudes on optics.
        return (1 - (F.normalize(student,dim=1,eps=1e-6) *
                     F.normalize(target,dim=1,eps=1e-6)).sum(1)).mean()


def train_hint_epoch(model, loader, loaded, settings, optimizer, teacher_cache, hints):
    """Same supervised/optical losses as legacy student epoch + explicit hint.

    Task-local to keep other tasks and all old training profiles unchanged.
    Gradient clipping includes the train-only projector; EMA remains core/head.
    """
    model.train(); hints.train()
    totals=defaultdict(float); started=time.perf_counter()
    for batch_index,batch in enumerate(loader,start=1):
        density=batch['density'].to(loaded.device,non_blocking=True)
        fixation=batch['fixation'].to(loaded.device,non_blocking=True)
        inputs=legacy.preprocess_vision(loaded.processor,batch['images'],loaded.device)
        teacher_logits=teacher_cache.get(batch['sample_ids'],loaded.device) if teacher_cache is not None else None
        optimizer.zero_grad(set_to_none=True)
        with legacy._autocast(settings,loaded.device):
            logits,spatial,_=model(inputs['pixel_values'],inputs['image_grid_thw'])
            task_loss,pieces=legacy.saliency_loss(logits,density,fixation,settings,teacher_logits=teacher_logits)
            balance,importance=model.router_losses()
            operating=model.operating_loss() if hasattr(model,'operating_loss') else logits.new_zeros(())
            dc=legacy.phase_dc_loss(model) if settings.phase_dc_weight > 0 else logits.new_zeros(())
            hint=hints(spatial,batch['sample_ids']) if settings.feature_hint_current_weight > 0 else logits.new_zeros(())
            total=(task_loss + settings.router_balance_weight*balance + settings.router_importance_weight*importance
                   + settings.phase_dc_weight*dc + getattr(settings,'ccd_operating_point_weight',0.)*operating
                   + settings.feature_hint_current_weight*hint)
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite hint training loss, batch {batch_index}")
        total.backward()
        if settings.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params'] if p.requires_grad],settings.gradient_clip_norm)
        optimizer.step()
        count=len(batch['sample_ids']);totals['samples']+=count
        values=dict(pieces,loss=total,feature_hint=hint,router_balance=balance,router_importance=importance,
                    phase_dc=dc,ccd_operating_point=operating)
        for k,v in values.items():totals[k]+=float(v.detach())*count
        if batch_index % settings.log_interval_batches==0 or batch_index==len(loader):
            print(f"[student hint] batch={batch_index}/{len(loader)} loss={totals['loss']/totals['samples']:.5f} "
                  f"CC={totals['cc']/totals['samples']:.5f} hint={totals['feature_hint']/totals['samples']:.5f}",flush=True)
    count=totals.pop('samples')
    return {k:v/count for k,v in totals.items()} | {'samples':int(count),'epoch_time_sec':time.perf_counter()-started}
