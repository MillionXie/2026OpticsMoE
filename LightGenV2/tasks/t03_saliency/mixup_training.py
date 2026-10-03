"""Training-only input MixUp; paired endpoint losses preserve fixation semantics.

Qwen processor pixel normalization/patch packing is affine and all grids must
be identical, so convex mixing here is RGB mixing without uint8 quantization.
Not an assertion that real gaze or teacher response to a blended image is linear.
"""
import math
import torch


def validate(options):
    if not options:
        return {}
    if set(options) != {'alpha', 'probability', 'end_epoch'}:
        raise ValueError('MixUp requires alpha, probability and end_epoch only')
    a, p, end = options['alpha'], options['probability'], options['end_epoch']
    if (isinstance(a, bool) or isinstance(p, bool) or not math.isfinite(a) or not math.isfinite(p)
            or not 0 < a <= .4 or not 0 < p <= .5
            or isinstance(end, bool) or not isinstance(end, int) or end < 1):
        raise ValueError('MixUp requires bounded alpha/probability and integer positive end_epoch')
    return dict(options)


def prepare(inputs, sample_ids, options, *, enabled):
    """Draw once outside the SAM closure; both SAM passes use the same mixture."""
    if not options or not enabled:
        return inputs, None
    if any(not s.startswith('train/') for s in sample_ids):
        raise ValueError('MixUp is restricted to train identities')
    count = len(sample_ids)
    if count < 2 or torch.rand(()).item() >= options['probability']:
        return inputs, None
    grid = inputs['image_grid_thw']
    pixels = inputs['pixel_values']
    if (grid.shape != (count, 3) or not torch.equal(grid, grid[:1].expand_as(grid))
            or int(grid[0, 0]) != 1 or pixels.shape[0] != int(grid.prod(1).sum())):
        raise ValueError('MixUp requires identically packed static image grids')
    order = torch.randperm(count)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)  # Random cycle, no unchanged self-pair.
    lam = float(torch.distributions.Beta(options['alpha'], options['alpha']).sample())
    lam = max(lam, 1-lam)  # Equivalent symmetric loss, retain majority source identity.
    rows = pixels.reshape(count, -1, *pixels.shape[1:])
    mixed = lam*rows + (1-lam)*rows[permutation.to(pixels.device)]
    return dict(inputs, pixel_values=mixed.reshape_as(pixels)), (lam, permutation)


def paired_loss(loss_fn, logits, density, fixation, settings, teacher_logits, mix):
    first, pieces = loss_fn(logits, density, fixation, settings, teacher_logits=teacher_logits)
    if mix is None:
        return first, pieces
    lam, permutation = mix
    idx = permutation.to(density.device)
    teacher = teacher_logits[idx] if teacher_logits is not None else None
    second, other = loss_fn(logits, density[idx], fixation[idx], settings, teacher_logits=teacher)
    if pieces.keys() != other.keys():
        raise RuntimeError('MixUp endpoint metric contract mismatch')
    # Do not interpolate binary fixations then threshold them: NSS is evaluated
    # separately at each endpoint and losses weighted by the input coefficient.
    return lam*first+(1-lam)*second, {k:lam*pieces[k]+(1-lam)*other[k] for k in pieces}
