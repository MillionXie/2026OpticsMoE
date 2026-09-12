"""Training-only electronic-subspace SAM, based on arXiv:2010.01412.

No inference parameters, no optical geometry/noise change. The second backward
still trains ALL enabled parameters, including optical phases and router.
"""
from collections import defaultdict
from contextlib import nullcontext
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
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


def symmetric_density_kl(first_logits, second_logits):
    """R-Drop-inspired consistency on spatial distributions, not mean pixels.

    Both views retain optical training corruption; no clean-view substitution.
    No stop-gradient: both stochastic predictions receive consistency gradients.
    """
    if first_logits.shape != second_logits.shape or first_logits.ndim != 4 or first_logits.shape[1] != 1:
        raise ValueError('Noise consistency requires matching [B,1,H,W] saliency logits')
    dtype = torch.float64 if first_logits.dtype == torch.float64 else torch.float32
    a = F.log_softmax(first_logits.to(dtype).flatten(1), dim=1)
    b = F.log_softmax(second_logits.to(dtype).flatten(1), dim=1)
    return .5*((a.exp()-b.exp())*(a-b)).sum(1).mean()


def noise_consistent_closure(single_view, weight):
    """Average two supervised noisy views and add symmetric KL, or legacy one view."""
    if not 0 <= weight <= 5:
        raise ValueError('Noise consistency weight must be finite and in [0,5]')
    first, pieces, logits = single_view()
    if not weight:
        return first, pieces
    second, other, second_logits = single_view()
    if pieces.keys() != other.keys():
        raise RuntimeError('Noisy views returned different metric contracts')
    consistency = symmetric_density_kl(logits, second_logits)
    total = (first+second)/2 + weight*consistency
    return total, {**{k:(pieces[k]+other[k])/2 for k in pieces},
                   'loss':total, 'noise_consistency_kl':consistency}


def sam_step(optimizer, closure, rho, device, clip_norm=0., *, asam=None, weight_parameter_ids=None,
             gsam_coefficient=0.):
    """One AdamW update; restore exact weights/RNG even if second pass raises.

    closure returns a scalar loss and detached-or-live metric dictionary.
    Same perturbation/dropout realization in both forwards; random progression
    after the step equals a single ordinary forward/backward, not two draws.
    """
    if not 0 <= rho <= .1:raise ValueError('Audited SAM radius must be in [0,.1]')
    if not 0 <= gsam_coefficient <= .2 or (gsam_coefficient and (not rho or asam)):
        raise ValueError('GSAM requires ordinary SAM and coefficient in [0,.2]')
    if asam:
        if (set(asam) != {'rho', 'eta'} or not .05 <= asam['rho'] <= .5
                or not .001 <= asam['eta'] <= .1 or weight_parameter_ids is None or rho <= 0):
            raise ValueError('Invalid ASAM radius/stability or missing weight identity')
        rho = asam['rho']  # Normalized-coordinate radius, NOT ordinary SAM L2 radius.
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
        scales = None
        if asam:
            # Element-wise ASAM, p=2, without bias normalization. Name/identity
            # comes from the model, not tensor rank (LayerNorm weights are 1D).
            scales = [p.detach().abs()+asam['eta'] if id(p) in weight_parameter_ids
                      else torch.ones_like(p) for p in selected]
            norm=torch.linalg.vector_norm(torch.stack([(p.grad*s).float().norm() for p,s in zip(selected,scales)]))
        else:
            norm=torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in selected]))
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite SAM gradient norm')
        backups=[p.detach().clone() for p in selected]
        original_gradients = [p.grad.detach().clone() for p in selected] if gsam_coefficient else None
        try:
            with torch.no_grad():
                if scales is None:
                    for p in selected:p.add_(p.grad * (rho/norm.clamp_min(1e-12)))
                else:
                    for p,s in zip(selected,scales):p.add_(p.grad*s.square()*(rho/norm.clamp_min(1e-12)))
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(before,device)
            second,_=closure()
            if not torch.isfinite(second):raise RuntimeError('Nonfinite second SAM loss')
            second.backward()
            if original_gradients is not None:
                # GSAM-inspired correction ONLY in the existing electronic SAM
                # subspace. Optical gradients keep their ordinary second pass.
                # This is an AdamW/subspace adaptation, not a full paper replica.
                with torch.no_grad():
                    if any(p.grad is None for p in selected):
                        raise RuntimeError('GSAM electronic gradient support changed between passes')
                    new_norm2 = sum(p.grad.double().square().sum() for p in selected)
                    dot = sum((g.double()*p.grad.double()).sum()
                              for g,p in zip(original_gradients,selected))
                    if not torch.isfinite(new_norm2) or not torch.isfinite(dot):
                        raise RuntimeError('Nonfinite GSAM gradient geometry')
                    correction_norm2 = new_norm2.new_zeros(())
                    if new_norm2 > 1e-24:
                        projection = dot/new_norm2
                        for g,p in zip(original_gradients,selected):
                            perpendicular = g - projection.to(p.grad)*p.grad
                            correction_norm2 += perpendicular.double().square().sum()
                            p.grad.add_(perpendicular, alpha=-gsam_coefficient)
                    values['gsam_relative_correction'] = (
                        gsam_coefficient*(correction_norm2/new_norm2.clamp_min(1e-24)).sqrt()).detach()
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


def train_sam_epoch(model,loader,loaded,settings,optimizer,teacher_cache=None,relation_targets=None,masked_targets=None,first_stage=None):
    if settings.sam_rho == 0:
        if relation_targets is not None or masked_targets is not None or first_stage is not None:
            raise ValueError('Auxiliary distillation requires the audited SAM path')
        return legacy._train_epoch('student',model,loader,loaded,settings,optimizer,
                                   teacher_cache=teacher_cache)
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError('SAM double-pass BatchNorm buffer handling is not supported')
    model.train();totals=defaultdict(float);started=time.perf_counter()
    asam = getattr(settings, 'asam', {})
    weight_ids = {id(p) for name,p in model.named_parameters() if name.endswith('.weight')} if asam else None
    for batch_index,batch in enumerate(loader,start=1):
        density=batch['density'].to(loaded.device,non_blocking=True)
        fixation=batch['fixation'].to(loaded.device,non_blocking=True)
        inputs=legacy.preprocess_vision(loaded.processor,batch['images'],loaded.device)
        teacher_logits=teacher_cache.get(batch['sample_ids'],loaded.device) if teacher_cache is not None else None
        def single_view():
            with legacy._autocast(settings,loaded.device):
                active_first = first_stage is not None and settings.first_stage_current_weight > 0
                with (first_stage.capture(model) if active_first else nullcontext([])) as captured:
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
                first_loss, first_cc = logits.new_zeros(()), logits.new_zeros(())
                if active_first:
                    first_loss, first_cc = first_stage.loss(captured, inputs['image_grid_thw'],
                        density, fixation, settings, detach_student=settings.first_stage_head_warmup)
                    total = total + settings.first_stage_current_weight * first_loss
            return total,dict(pieces,loss=total,router_balance=balance,router_importance=importance,
                              phase_dc=dc,ccd_operating_point=operating,
                              **({'relational_loss':relation} if relation_targets is not None else {}),
                              **({'masked_loss':masked} if masked_targets is not None else {}),
                              **({'first_stage_loss':first_loss, 'first_stage_cc':first_cc} if first_stage is not None else {})),logits
        def closure():
            return noise_consistent_closure(single_view,getattr(settings,'noise_consistency_weight',0.))
        values,increase=sam_step(optimizer,closure,settings.sam_rho,loaded.device,settings.gradient_clip_norm,
                                 asam=asam,weight_parameter_ids=weight_ids,
                                 gsam_coefficient=getattr(settings,'gsam_coefficient',0.))
        count=len(batch['sample_ids']);totals['samples']+=count
        values['sam_loss_increase']=increase
        for key,value in values.items():totals[key]+=float(value)*count
        if batch_index%settings.log_interval_batches==0 or batch_index==len(loader):
            print(f"[student {'ASAM' if asam else 'SAM'}] batch={batch_index}/{len(loader)} loss={totals['loss']/totals['samples']:.5f} "
                  f"CC={totals['cc']/totals['samples']:.5f} sharpness={totals['sam_loss_increase']/totals['samples']:.6f}",flush=True)
    count=totals.pop('samples')
    return {k:v/count for k,v in totals.items()}|{'samples':int(count),'epoch_time_sec':time.perf_counter()-started}
