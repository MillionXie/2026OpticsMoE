"""Differentiable all-expert targets, independent of forced/empty hard masks."""
import torch
from torch.nn import functional as F

from .layout import A_EXPERTS, B_EXPERTS


def target_probabilities(domain, group_probability=.7, dtype=torch.float32):
    if domain.ndim != 1 or not bool(((domain == 0) | (domain == 1)).all()):
        raise ValueError('Training domain must be a vector of 0/1 labels')
    target = torch.full((len(domain), 4), (1-group_probability)/2, device=domain.device, dtype=dtype)
    for d, experts in ((0,A_EXPERTS),(1,B_EXPERTS)):
        rows = domain == d
        for expert in experts:
            target[rows, expert] = group_probability/2
    return target


def routing_loss(routes, domain, settings):
    if not routes:
        return None, {}
    targets = target_probabilities(domain, settings['target_group_probability'])
    kl_terms, capture_terms = [], []
    for route in routes.values():
        p = route['probabilities'].float()
        if p.shape != targets.shape or not bool(torch.isfinite(p).all()) or bool((p <= 0).any()):
            raise RuntimeError('Invalid differentiable router probabilities')
        if not torch.allclose(p.sum(1),torch.ones(len(p),device=p.device),atol=1e-4,rtol=1e-4):
            raise RuntimeError('Router probabilities are not normalized')
        # Always use the original four-way probabilities, never the teacher
        # route or post-intervention weights/masks. Missing hard groups still
        # receive probability gradients and incur a nonzero target loss.
        kl_terms.append(F.kl_div(p.clamp_min(1e-8).log(), targets, reduction='batchmean'))
        capture_terms.append(route['capture_loss'])
    terms = dict(route_target_kl=torch.stack(kl_terms).mean(), capture=torch.stack(capture_terms).mean())
    return settings['target_kl_weight']*terms['route_target_kl'] + settings['capture_weight']*terms['capture'], terms
