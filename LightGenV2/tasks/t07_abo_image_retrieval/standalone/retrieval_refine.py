"""Training-only repair objectives; no optical forward/inference changes."""
import torch
from torch.nn import functional as F
from .io import sha256

PROFILES = {
    'standard': dict(warmup=0, category_probability=0., positive_weight=0., teacher_weight=0.),
    'route_repair': dict(warmup=3, category_probability=.5, positive_weight=.1, teacher_weight=0.),
    'route_distill': dict(warmup=3, category_probability=.5, positive_weight=.1, teacher_weight=.2),
    'shape_views': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0.),
    'sku_regularized': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=.015, weight_decay=.03, phase_dropout=.03),
}


def validate_continuation(protocol, payload, manifest_sha, fresh):
    if protocol != 'abo200_enrolled_sku_hash8train4query_v1' or fresh:
        return
    if payload.get('manifest_sha256') != manifest_sha or not payload.get('test_selected'):
        raise ValueError('ABO continuation requires a checkpoint trained/selected on EXACT new manifest; old ABO weights forbidden')


def balanced_targets(probabilities):
    """Detached Sinkhorn-like batch assignment, not used to select inference experts."""
    with torch.no_grad():
        q = probabilities.detach().float().clamp_min(1e-7).pow(2).T
        q /= q.sum()
        for _ in range(12):
            q = q / q.sum(1, keepdim=True).clamp_min(1e-12) / len(q)
            q = q / q.sum(0, keepdim=True).clamp_min(1e-12) / q.shape[1]
        return (q * q.shape[1]).T


def route_objective(model):
    losses = []
    for m in (model.vision, model.language):
        last = m.optics.router.last
        energy = last['energy'].float()
        share = energy / energy.sum(1, keepdim=True).clamp_min(1e-8)
        # Penalize physically dark detector groups, not merely softmax counts.
        light_balance = -(4 * share.mean(0).clamp_min(1e-6)).log().mean()
        p = last['probabilities'].float().clamp_min(1e-7)
        q = balanced_targets(p)
        cross_entropy = -(q * p.log()).sum(1).mean()
        losses.append(light_balance + .2 * cross_entropy)
    return torch.stack(losses).mean()


def all_view_loss(z, labels, bank, bank_labels, count, excluded):
    logits = F.normalize(z[:count].float(), dim=-1) @ F.normalize(bank.detach().float(), dim=-1).T / .1
    positive = labels[:count, None].eq(bank_labels[None])
    if excluded is not None:
        positive &= ~excluded
        logits = logits.masked_fill(excluded, -1e4)
    if not positive.any(1).all():
        raise ValueError('No nonself positive for all-view objective')
    logp = logits.log_softmax(1)
    return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def load_train_teacher(path, expected_sha, manifest_sha, groups, fit):
    if not expected_sha or sha256(path) != expected_sha:
        raise ValueError('Teacher cache SHA mismatch')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    expected_ids = [r['sample_id'] for r in groups['gallery'] + groups['query']]
    if cache.get('manifest_sha256') != manifest_sha or cache.get('ids') != expected_ids:
        raise ValueError('Teacher image order/protocol mismatch')
    if cache['vectors'].shape != (len(expected_ids), 64) or not torch.isfinite(cache['vectors']).all():
        raise ValueError('Expected finite frozen64D teacher vectors')
    train_ids = {r['sample_id'] for r in groups['train']}
    query_ids = {r['sample_id'] for r in groups['query']}
    used = {r['sample_id'] for r in fit['train'] + fit['gallery']}
    if train_ids & query_ids or not used <= train_ids:
        raise ValueError('Non-TRAIN image in teacher fitting pool')
    # Only TRAIN rows survive this function; TEST cache entries never enter any loss.
    indices = {sid: i for i, sid in enumerate(expected_ids)}
    return {sid: F.normalize(cache['vectors'][indices[sid]].float().clone(), dim=0) for sid in sorted(used)}


def relational_loss(z, bank, teacher_query, teacher_bank, excluded):
    n = len(teacher_query)
    student = F.normalize(z[:n].float(), dim=-1) @ F.normalize(bank.detach().float(), dim=-1).T / .1
    teacher = teacher_query.detach().float() @ teacher_bank.detach().float().T / .1
    if excluded is not None:
        student = student.masked_fill(excluded, -1e4)
        teacher = teacher.masked_fill(excluded, -1e4)
    target = teacher.softmax(1)
    return F.kl_div(student.log_softmax(1), target, reduction='batchmean')


def router_acceptable(metrics):
    values = metrics['router']
    return all(m in values and min(values[m]['selection_share']) >= .05
               and values[m]['most_common_top2_fraction'] <= .8
               and values[m]['unique_top2_sets'] >= 3 for m in ('vision', 'language'))


def selection_score(metrics, refined):
    base = (metrics['test']['hit_at_1'], metrics['test']['map_at_10'])
    return (int(router_acceptable(metrics)), *base) if refined else base
