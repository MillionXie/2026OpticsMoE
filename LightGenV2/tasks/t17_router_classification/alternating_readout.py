"""Training-only block optimization of the existing head; no inference transform."""
import torch
from torch.nn import functional as F
from .train_convex_readout import training_coordinates, canonical_weight
from .original_f_teacher import teacher_KL


def domain_route_loss(dense, domains):
    """Acquisition-domain supervision before Top2; targets are entrance POWER."""
    if dense.ndim != 2 or dense.shape[1] != 4:
        raise ValueError('Four dense power ports required')
    if domains.shape != dense.shape[:1] or not bool(((domains == 0) | (domains == 1)).all()):
        raise ValueError('RGB=0/SAR=1 acquisition domains required')
    target = torch.full_like(dense, .01)
    target[domains == 0, :2] = .49
    target[domains == 1, 2:] = .49
    return F.kl_div(dense.clamp_min(1e-20).log(), target, reduction='batchmean')


def anchored_classification(raw_logits, labels, weight, anchor, strength, smoothing, teacher_logits=None, teacher_weight=0., teacher_temperature=2.):
    if strength <= 0:
        raise ValueError('Positive head anchor required')
    loss=F.cross_entropy(raw_logits, labels, label_smoothing=smoothing) + .5 * strength * (weight-anchor).square().sum()
    if teacher_weight:
        if teacher_logits is None:raise ValueError('KD requires verified TRAIN teacher logits')
        loss=loss+teacher_weight*teacher_KL(raw_logits,teacher_logits,teacher_temperature)
    return loss


def refit_existing_head(head, features, labels, anchor, strength, smoothing, inner_steps, teacher_logits=None, teacher_weight=0., teacher_temperature=2.):
    """One bounded TRAIN-only Gram-conditioned LBFGS block, exported to float32 W.

    All inputs are detached. No optical parameter can receive a gradient, and
    neither the Gram factor nor a new bias is retained in the inference model.
    """
    if inner_steps < 1 or strength <= 0 or len(features) != len(labels):
        raise ValueError('Invalid bounded head block')
    x = features.detach().double()
    target = labels.detach()
    prior = anchor.detach().double()
    transformed, start, factor, info = training_coordinates(x, head.weight.detach().double(), 'TRAIN_Gram_cholesky')
    coordinate = torch.nn.Parameter(start)
    optimizer = torch.optim.LBFGS([coordinate], lr=1., max_iter=inner_steps, max_eval=inner_steps*2,
        history_size=20, line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
    calls = 0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        weight = canonical_weight(coordinate, factor)
        loss = anchored_classification(F.linear(transformed, coordinate), target, weight, prior, strength, smoothing,
            teacher_logits.detach().double() if teacher_logits is not None else None,teacher_weight,teacher_temperature)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite alternating head loss')
        loss.backward()
        if not torch.isfinite(coordinate.grad).all():
            raise RuntimeError('Nonfinite alternating head gradient')
        calls += 1
        return loss
    before = float(closure().detach())
    optimizer.step(closure)
    after = float(closure().detach())
    if after > before + 1e-8 * max(1., abs(before)):
        raise RuntimeError('Head block increased its fixed TRAIN objective')
    with torch.no_grad():
        head.weight.copy_(canonical_weight(coordinate, factor).to(head.weight))
    return dict(**info, objective_before=before, objective_after=after,
        internal_steps=optimizer.state[coordinate].get('n_iter', 0), closure_calls=calls,
        exported_dtype=str(head.weight.dtype), TRAIN_only=True)
