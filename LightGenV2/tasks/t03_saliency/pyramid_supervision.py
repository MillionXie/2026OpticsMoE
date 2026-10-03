"""Training-only density-pyramid CC; no learned weights or inference transforms."""
import math

import torch
import torch.nn.functional as F


def validate(options, image_size):
    if not options:
        return {}
    if set(options) != {'weight', 'sizes'}:
        raise ValueError('pyramid_cc requires exactly weight and sizes')
    weight, sizes = options['weight'], options['sizes']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 < weight <= 3:
        raise ValueError('pyramid_cc weight must be finite in (0,3]')
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError('pyramid_cc sizes must be a nonempty unique list')
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 2 or s > image_size or image_size % s for s in sizes):
        raise ValueError('pyramid_cc sizes must divide the image size and be >=2')
    return {'weight': float(weight), 'sizes': list(sizes)}


def loss(logits, target, sizes):
    if logits.ndim != 4 or logits.shape != target.shape or logits.shape[1] != 1 or logits.shape[-2] != logits.shape[-1]:
        raise ValueError('Expected matching square Bx1xHxW maps')
    validate({'weight': 1., 'sizes': sizes}, logits.shape[-1])
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    prediction = logits.to(dtype).flatten(1).softmax(1).reshape_as(logits)
    target = target.detach().to(device=logits.device, dtype=dtype)
    if not torch.isfinite(target).all() or (target < 0).any() or (target.sum((-2, -1)) <= 0).any():
        raise ValueError('Target must be a finite nonnegative density with positive mass')
    target = target / target.sum((-2, -1), keepdim=True)
    terms, pieces = [], {}
    for size in sizes:
        kernel = logits.shape[-1] // size
        # Pool probabilities, not logits; unit-mean scaling stabilizes norms.
        x = F.avg_pool2d(prediction, kernel).flatten(1) * logits.shape[-1]**2
        y = F.avg_pool2d(target, kernel).flatten(1) * logits.shape[-1]**2
        x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
        valid = torch.linalg.vector_norm(y, dim=1) > 1e-6
        cc = F.cosine_similarity(x, y, dim=1, eps=1e-6).clamp(-1, 1)
        term = ((1 - cc) * valid).sum() / valid.sum().clamp_min(1)
        terms.append(term)
        pieces[f'pyramid_cc_loss_{size}'] = term.detach()
    return torch.stack(terms).mean(), pieces
