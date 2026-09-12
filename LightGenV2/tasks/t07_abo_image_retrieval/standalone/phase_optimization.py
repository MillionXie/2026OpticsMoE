"""Training-only Router Adam coordinates; never imported by model/inference.

Forward/checkpoints remain raw -> 2*pi*sigmoid(raw), exactly as before.
Only between backward and AdamW step use theta in radians, converting gradients
with dL/dtheta = dL/draw / (2*pi*s*(1-s)). No forward/save inside the context.
Adam moments are radians-coordinate moments; fresh optimizer only, no SAM.
"""
from contextlib import contextmanager
import math
import torch


def router_coordinates(config):
    mode=config.get('router_optimizer_coordinates','raw')
    if mode not in ('raw','radians'):
        raise ValueError('Router optimizer coordinates must be raw or radians')
    if mode=='radians' and (config.get('sam_rho',0)!=0 or 'router_initial_phase_offset_turns' in config):
        raise ValueError('Radian Router control cannot combine SAM or phase-origin shift')
    return mode


def phase_to_raw(theta):
    # Finite raw parameters at the periodic branch cut; <8e-7 rad clipping.
    return torch.remainder(theta/(2*math.pi),1.).clamp(1e-7,1-1e-7).logit()


@contextmanager
def router_radian_step(optimizer, enabled=False):
    """Temporarily change only active Router parameter/gradient coordinates.

    Preflight all tensors before mutation. On any exception restore original
    raw tensors; optimizer state may have advanced, so abort that run (no retry).
    Default path is a strict no-op. Do not run forward or serialize in context.
    """
    if not enabled:
        yield
        return
    entries=[]
    with torch.no_grad():
        for group in optimizer.param_groups:
            if group.get('kind')!='router' or group['lr']==0:continue
            if group.get('weight_decay',0)!=0:
                raise ValueError('Router radian optimizer requires zero weight decay')
            for parameter in group['params']:
                if parameter.grad is None:continue
                if parameter.dtype!=torch.float32 or not torch.isfinite(parameter).all() or not torch.isfinite(parameter.grad).all():
                    raise ValueError('Router coordinates require finite FP32 parameters and gradients')
                s=parameter.detach().sigmoid()
                derivative=(2*math.pi)*s*(1-s)
                if not (derivative>0).all():raise ValueError('Exactly saturated Router cannot recover a phase gradient')
                gradient=parameter.grad.detach()/derivative
                if not torch.isfinite(gradient).all():raise ValueError('Nonfinite Router phase gradient')
                entries.append((parameter,parameter.detach().clone(),parameter.grad,(2*math.pi)*s,gradient))
        for p,raw,grad,theta,gtheta in entries:
            p.copy_(theta);p.grad=gtheta.clone()
    success=False
    try:
        yield
        with torch.no_grad():
            converted=[phase_to_raw(p) for p,*_ in entries]
            if any(not torch.isfinite(x).all() for x in converted):raise RuntimeError('Nonfinite radian Router update')
            for (p,*_),raw in zip(entries,converted):p.copy_(raw)
        success=True
    finally:
        with torch.no_grad():
            for p,raw,grad,theta,gtheta in entries:
                if not success:p.copy_(raw)
                p.grad=grad  # Retain update identity for existing EMA dispatch.


@torch.no_grad()
def circular_router_ema(target_raw, current_raw, decay):
    """Shortest angular EMA, avoiding raw-logit averaging across 0/2*pi."""
    if not 0<=decay<1:raise ValueError('EMA decay must be in [0,1)')
    old=(2*math.pi)*target_raw.sigmoid();new=(2*math.pi)*current_raw.sigmoid()
    delta=torch.atan2((new-old).sin(),(new-old).cos())
    target_raw.copy_(phase_to_raw(old+(1-decay)*delta))
