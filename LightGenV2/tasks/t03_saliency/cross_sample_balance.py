"""Training-only cross-sample estimates of population expert imbalance.

Remove within-sample products, not expert balancing itself. Finite-sample
estimates may be negative; clamping would reintroduce bias. This is a local
estimator experiment, not a change to optical routing or inference.
"""
import torch


def terms(routing):
    p = routing['probabilities']
    selected = routing['selected_mask']
    if p.ndim != 2 or p.shape != selected.shape or p.shape[1] != 4 or len(p) < 2:
        raise ValueError('Cross-sample balance requires B>=2 and four Top2 experts')
    dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
    p = p.to(dtype)
    h = selected.detach().to(dtype) / 2
    n = len(p)

    def pairs(x, y):
        return 4 * ((x.sum(0) * y.sum(0)).sum() - (x * y).sum()) / (n * (n - 1))

    # Preserve the existing hard-load straight-through gradient convention,
    # but apply it to distinct samples rather than the diagonal self-product.
    h_st = h + p - p.detach()
    return {
        'soft_delta': pairs(p, h) - 4 * (p.mean(0) * h.mean(0)).sum(),
        'importance_delta': pairs(p, p) - 4 * p.mean(0).square().sum(),
        'hard': pairs(h_st, h_st) - 1,
    }
