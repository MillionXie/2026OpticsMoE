"""Training-only electronic-subspace SAM, based on arXiv:2010.01412.

No inference parameters, no optical geometry/noise change. The second backward
still trains ALL enabled parameters, including optical phases and router.
"""
from collections import defaultdict
import random
import time
import numpy as np
import torch
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
from .training_support import task_saliency_loss

ELECTRONIC_GROUPS = frozenset({'electronic','saliency_head','ccd_readout',
                             'electronic_ffn_spatial','electronic_global_spatial'})


def _rng(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
            random.getstate(), np.random.get_state())


def _restore_rng(state,device):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state(state[1],device)
    random.setstate(state[2]);np.random.set_state(state[3])


def sam_step(optimizer, closure, rho, device, clip_norm=0.):
    """One AdamW update; restore exact weights/RNG even if second pass raises.

    closure returns a scalar loss and detached-or-live metric dictionary.
    Same perturbation/dropout realization in both forwards; random progression
    after the step equals a single ordinary forward/backward, not two draws.
    """
    if not 0 <= rho <= .1:raise ValueError('Audited SAM radius must be in [0,.1]')
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    before=_rng(device)
    first,values=closure()
    if not torch.isfinite(first):raise RuntimeError('Nonfinite first SAM loss')
    first.backward()
    after=_rng(device)
    values={k:v.detach() for k,v in values.items()}
    increase=first.detach().new_zeros(())
    if rho:
        selected=[p for g in optimizer.param_groups if g.get('name') in ELECTRONIC_GROUPS
                  for p in g['params'] if p.requires_grad and p.grad is not None]
        if not selected:raise RuntimeError('SAM has no active electronic parameters')
        norm=torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in selected]))
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite SAM gradient norm')
        backups=[p.detach().clone() for p in selected]
        try:
            with torch.no_grad():
                for p in selected:p.add_(p.grad * (rho/norm.clamp_min(1e-12)))
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(before,device)
            second,_=closure()
            if not torch.isfinite(second):raise RuntimeError('Nonfinite second SAM loss')
            second.backward()
            increase=second.detach()-first.detach()
        finally:
            with torch.no_grad():
                for p,value in zip(selected,backups):p.copy_(value)
            _restore_rng(after,device)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise RuntimeError('Nonfinite final SAM gradients; optimizer was not stepped')
    if clip_norm:torch.nn.utils.clip_grad_norm_(params,clip_norm)
    optimizer.step()  # EMA post-step hook fires exactly once, at restored weights.
    return values, increase


def train_sam_epoch(model,loader,loaded,settings,optimizer,teacher_cache=None,relation_targets=None,masked_targets=None):
    if settings.sam_rho == 0:
        if relation_targets is not None or masked_targets is not None:
            raise ValueError('Auxiliary distillation requires the audited SAM path')
        return legacy._train_epoch('student',model,loader,loaded,settings,optimizer,
                                   teacher_cache=teacher_cache)
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError('SAM double-pass BatchNorm buffer handling is not supported')
    model.train();totals=defaultdict(float);started=time.perf_counter()
    for batch_index,batch in enumerate(loader,start=1):
        density=batch['density'].to(loaded.device,non_blocking=True)
        fixation=batch['fixation'].to(loaded.device,non_blocking=True)
        inputs=legacy.preprocess_vision(loaded.processor,batch['images'],loaded.device)
        teacher_logits=teacher_cache.get(batch['sample_ids'],loaded.device) if teacher_cache is not None else None
        def closure():
            with legacy._autocast(settings,loaded.device):
                outputs=model(inputs['pixel_values'],inputs['image_grid_thw'])
                logits=outputs[0]
                task,pieces=task_saliency_loss(logits,density,fixation,settings,teacher_logits=teacher_logits)
                balance,importance=model.router_losses()
                operating=model.operating_loss() if hasattr(model,'operating_loss') else logits.new_zeros(())
                dc=legacy.phase_dc_loss(model) if settings.phase_dc_weight>0 else logits.new_zeros(())
                total=(task+settings.router_balance_weight*balance+settings.router_importance_weight*importance
                       +settings.phase_dc_weight*dc+getattr(settings,'ccd_operating_point_weight',0.)*operating)
                relation = logits.new_zeros(())
                if relation_targets is not None and settings.relational_current_weight > 0:
                    relation = relation_targets.loss(outputs[1], batch['sample_ids'])
                    total = total + settings.relational_current_weight * relation
                masked = logits.new_zeros(())
                if masked_targets is not None and settings.masked_current_weight > 0:
                    masked = masked_targets.loss(outputs[1], batch['sample_ids'],
                        detach_student=settings.masked_generator_warmup)
                    total = total + settings.masked_current_weight * masked
            return total,dict(pieces,loss=total,router_balance=balance,router_importance=importance,
                              phase_dc=dc,ccd_operating_point=operating,
                              **({'relational_loss':relation} if relation_targets is not None else {}),
                              **({'masked_loss':masked} if masked_targets is not None else {}))
        values,increase=sam_step(optimizer,closure,settings.sam_rho,loaded.device,settings.gradient_clip_norm)
        count=len(batch['sample_ids']);totals['samples']+=count
        values['sam_loss_increase']=increase
        for key,value in values.items():totals[key]+=float(value)*count
        if batch_index%settings.log_interval_batches==0 or batch_index==len(loader):
            print(f"[student SAM] batch={batch_index}/{len(loader)} loss={totals['loss']/totals['samples']:.5f} "
                  f"CC={totals['cc']/totals['samples']:.5f} sharpness={totals['sam_loss_increase']/totals['samples']:.6f}",flush=True)
    count=totals.pop('samples')
    return {k:v/count for k,v in totals.items()}|{'samples':int(count),'epoch_time_sec':time.perf_counter()-started}
