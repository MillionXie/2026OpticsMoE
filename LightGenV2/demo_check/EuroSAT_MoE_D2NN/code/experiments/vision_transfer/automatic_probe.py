"""Read-only full-model automatic forward and target-domain phase gradients."""
import time
import torch
from torch.nn import functional as F
from . import model as M
from . import engine as E


def observe(routes, domain):
    result = {}
    for modality, route in routes.items():
        for d, label in ((0,'clean'),(1,'corrupted')):
            rows = domain == d
            count = int(rows.sum())
            if count:
                result[modality+'_'+label] = dict(samples=count,
                    selection_share=(route['selected_mask'][rows].float().mean(0)/4).detach().cpu().tolist(),
                    power_share=route['weights'][rows].detach().float().square().mean(0).cpu().tolist(),
                    probability_mean=route['probabilities'][rows].detach().float().mean(0).cpu().tolist())
    return result


def merge_observations(total, observation):
    for key, info in observation.items():
        if key not in total:
            total[key] = dict(samples=0, selection_share=[0.]*4, power_share=[0.]*4, probability_mean=[0.]*4)
        row = total[key]
        old, new = row['samples'], info['samples']
        for field in ('selection_share','power_share','probability_mean'):
            row[field] = [(a*old+b*new)/(old+new) for a,b in zip(row[field],info[field])]
        row['samples'] += new


def run_probe(loaded, r, h, inputs, y, domain, settings, task, epoch):
    saved_rng = E.rng_state()
    started = time.perf_counter()
    try:
        M.set_mode(loaded,r,h,False)
        M.force_route(r,None)
        with E.autocast(loaded,settings):
            logits = M.classification_logits(loaded.model,r,h,inputs)[0]
            rows = domain == (0 if task == 'A' else 1)
            if not bool(rows.any()):
                raise RuntimeError('Automatic diagnostic batch has no target-domain samples')
            ce = F.cross_entropy(logits[rows],y[rows])
        parameters = {name:p for name,p in M.named(r,h).items()
                      if p.requires_grad and M.group_of(name) == ('expert_a' if task == 'A' else 'expert_b')}
        if not parameters:
            raise RuntimeError('Automatic probe has no trainable target expert phases')
        gradients = torch.autograd.grad(ce,list(parameters.values()),allow_unused=True)
        norms = {name:0. if g is None else float(g.detach().float().norm()) for name,g in zip(parameters,gradients)}
        if not all(torch.isfinite(torch.tensor(value)) for value in norms.values()):
            raise RuntimeError('Nonfinite automatic target-domain CE gradient')
        return dict(routes=observe(M.routes(r),domain), ce_gradient_max=norms,
                    target_domain_ce=float(ce.detach()), seconds=time.perf_counter()-started,
                    no_forced_route=True, optimizer_updated=False)
    finally:
        M.set_mode(loaded,r,h,True,task,epoch)
        M.force_route(r,None)
        E.restore_rng(saved_rng)
