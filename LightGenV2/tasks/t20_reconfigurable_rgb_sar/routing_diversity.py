"""TRAIN-only modality-specialization loss for the EXISTING optical router.

Global expert use remains balanced, while two sensor domains may use different
mixtures. Domain identities are loss-only supervision, never inference inputs.
No slot is assigned to a named domain; the loss is slot-permutation invariant.
"""
import torch


def balance_and_diversity(route,domains,indices):
    active=route.index_select(1,indices)
    if len(domains.unique())!=2:raise ValueError('Paired two-domain TRAIN batches required')
    means=torch.stack([active[domains==d].mean(0) for d in domains.unique()])
    means=means/means.sum(1,keepdim=True).clamp_min(1e-12)
    mixture=means.mean(0);balance=(mixture-1/len(indices)).square().sum()
    js=(means*(means.clamp_min(1e-12).log()-mixture.clamp_min(1e-12).log())).sum(1).mean()
    return balance,js


def report_mean_JS(report):
    rows=report['routing_by_domain'];q=torch.tensor([rows[d]['mean_power'] for d in ('RGB','SAR')],dtype=torch.float64)
    return float(balance_and_diversity(q,torch.tensor([0,1]),torch.arange(q.shape[1]))[1])
