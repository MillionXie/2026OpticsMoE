"""Training-only, one-sided teacher gradient surgery; not full PCGrad.

Inspired by Yu et al., NeurIPS2020: https://arxiv.org/abs/2001.06782.
Only shared active parameters participate. Primary-only auxiliary parameters
keep their gradients. One forward/two reverse traversals, no additional noise
draw or inference layer. Euclidean raw-gradient projection does NOT guarantee
nonincreasing primary loss after adaptive AdamW/clipping or better test scores.
"""
import torch


def projection_enabled(config):
    enabled=config.get('teacher_gradient_projection',False)
    if type(enabled) is not bool:raise ValueError('Teacher projection must be boolean')
    if enabled and (config.get('sam_rho',0.) or config.get('view_consistency_weight',0.) or
                    config.get('vision_patch_teacher_weight',0.) or config.get('router_optimizer_coordinates','raw')!='raw'):
        raise ValueError('Teacher projection control excludes SAM, extra views/patch loss and radian optimizer')
    return enabled


def backward_primary_teacher(closure,optimizer):
    """Set projected grads, but never clip, step, or change parameters."""
    optimizer.zero_grad(set_to_none=True)
    active=[p for group in optimizer.param_groups if group['lr']>0 for p in group['params'] if p.requires_grad]
    if not active:raise ValueError('Teacher projection requires active parameters')
    result=closure()
    for key in ('loss','primary_loss','teacher_loss'):
        if key not in result or result[key].ndim!=0 or not torch.isfinite(result[key]):
            raise RuntimeError('Missing or nonfinite scalar projection loss')
    teacher=torch.autograd.grad(result['teacher_loss'],active,retain_graph=True,allow_unused=True)
    primary=torch.autograd.grad(result['primary_loss'],active,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in (*primary,*teacher)):
        raise RuntimeError('Nonfinite gradient before teacher projection')
    shared=[(p,t) for p,t in zip(primary,teacher) if p is not None and t is not None]
    zero=result['loss'].detach().float().new_zeros(())
    dot=sum((p.float()*t.float()).sum() for p,t in shared) if shared else zero
    p2=sum(p.float().square().sum() for p,t in shared) if shared else zero
    t2=sum(t.float().square().sum() for p,t in shared) if shared else zero
    conflict=bool(dot<0 and p2>1e-20)
    coefficient=dot/p2 if conflict else zero
    after=[]
    for param,p,t in zip(active,primary,teacher):
        projected=t-coefficient*p if conflict and p is not None and t is not None else t
        if p is not None and projected is not None:after.append((p,projected))
        param.grad=(p+projected if p is not None and projected is not None else p if projected is None else projected)
    after_dot=sum((p.float()*t.float()).sum() for p,t in after) if after else zero
    after_t2=sum(t.float().square().sum() for p,t in after) if after else zero
    audit=dict(loss_gap=0.,teacher_conflict_fraction=float(conflict),
               teacher_shared_cosine_before=float(dot/(p2*t2).sqrt().clamp_min(1e-20)),
               teacher_shared_cosine_after=float(after_dot/(p2*after_t2).sqrt().clamp_min(1e-20)))
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError('Nonfinite projected gradient')
    return result,audit
