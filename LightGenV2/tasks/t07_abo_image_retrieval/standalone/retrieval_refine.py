"""Training-only repair objectives; no optical forward/inference changes."""
import hashlib
import torch
from torch.nn import functional as F
from .io import sha256

PROFILES = {
    'standard': dict(warmup=0, category_probability=0., positive_weight=0., teacher_weight=0.),
    'route_repair': dict(warmup=3, category_probability=.5, positive_weight=.1, teacher_weight=0.),
    'route_distill': dict(warmup=3, category_probability=.5, positive_weight=.1, teacher_weight=.2),
    'shape_views': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0.),
    'sku_regularized': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=.015, weight_decay=.03, phase_dropout=.03),
    'sku_mild_adamw': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=0., weight_decay=.01, phase_dropout=0., mild_augmentation=True, route_scale=.25, noise_probability=.1),
    'sku_mild_sam': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=.002, weight_decay=.01, phase_dropout=0., mild_augmentation=True, route_scale=.25, noise_probability=.1),
    'sku_external_relations': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=0., weight_decay=.01, phase_dropout=0., mild_augmentation=True, route_scale=.25, noise_probability=.1, external_teacher_weight=1., external_instance_weight=0.),
    'phase_only': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=0., weight_decay=.01, phase_dropout=0., mild_augmentation=True, route_scale=.25, noise_probability=.1, optical_only=True, phase_lr_multiplier=1., router_lr_multiplier=.1),
    'phase_only_hot': dict(warmup=0, category_probability=.5, positive_weight=.1, teacher_weight=0., sam_rho=0., weight_decay=.01, phase_dropout=0., mild_augmentation=True, route_scale=.25, noise_probability=.1, optical_only=True, phase_lr_multiplier=3., router_lr_multiplier=.1),
}

# An optional larger *optical-electronic* teacher, not a Qwen/attention teacher.
# Keep a matched unexpanded control. A teacher result is not a compact-student
# result; compression/distillation is a separate, subsequently verified stage.
PROFILES['sku_capacity_control'] = dict(PROFILES['sku_mild_adamw'], router_lr_multiplier=.1)
# Training-only: both already encoded views query the detached TRAIN gallery.
# No extra forward, inference parameters, optical geometry or TEST supervision.
PROFILES['sku_symmetric_bank'] = dict(PROFILES['sku_capacity_control'], symmetric_bank=True)
PROFILES['sku_two_view_joint'] = dict(PROFILES['sku_capacity_control'], symmetric_bank=True,
    ranking_loss='two_view_softplus', supcon_weight=0., positive_weight=0.)
PROFILES['sku_phase_head'] = dict(PROFILES['sku_capacity_control'],
    phase_head_only=True, readout_input_dropout=.1, head_lr_multiplier=3.)
PROFILES['sku_phase_head_top1'] = dict(PROFILES['sku_phase_head'],
    ranking_loss='top1_softplus', symmetric_bank=True,
    supcon_weight=0., positive_weight=0.)
PROFILES['sku_phase_head_viewblend'] = dict(PROFILES['sku_phase_head_top1'],
    same_sku_blend_probability=.3, same_sku_blend_range=(.05, .15))
PROFILES['sku_alpha_only'] = dict(PROFILES['sku_capacity_control'],
    alpha_only=True, alpha_lr_multiplier=100., noise_probability=0.,
    ranking_loss='top1_softplus', symmetric_bank=True,
    supcon_weight=0., positive_weight=0.)
# Isolate training regularizers: identical loss, optimizer, capacity and optics.
# Do not infer separate augmentation/dropout effects from the old combined SAM run.
PROFILES['sku_augmentation_only'] = dict(PROFILES['sku_capacity_control'], mild_augmentation=False)
PROFILES['sku_phase_dropout_only'] = dict(PROFILES['sku_capacity_control'], phase_dropout=.03)
PROFILES['sku_retrieval_only'] = dict(PROFILES['sku_capacity_control'],
    positive_weight=0., supcon_weight=0.)
PROFILES['sku_optical_pretrain'] = dict(PROFILES['sku_capacity_control'],
    external_optical_only=True, external_phase_lr_multiplier=5.)
PROFILES['sku_conv_teacher'] = dict(PROFILES['sku_capacity_control'],
    electronic_expansion=dict(kernels=dict(vision=7, language=7), mlp_width=768))
PROFILES['sku_spatial_readout'] = dict(PROFILES['sku_capacity_control'], head_expansion='spatial2x2_64')
PROFILES['sku_fullfield_language'] = dict(PROFILES['sku_capacity_control'],
    ccd_readout_modes=dict(vision='prefix_rows',language='fullfield_rows'))


def blend_train_pairs(images, rows, count, rng, probability=.3, weight_range=(.05, .15)):
    """Mild TRAIN-only same-SKU pixel blend; not a rendered physical new view.

    Returns per-query source identities for exclusion from the detached TRAIN
    gallery. Prepare once outside optimizer closures, including SAM replay.
    No input image/row is mutated; probability zero consumes no RNG.
    """
    import math
    from PIL import Image
    low, high = weight_range
    if (not math.isfinite(probability) or not 0 <= probability <= 1
            or not 0 < low <= high < .5 or count < 2
            or len(rows) != 2 * count or len(images) != len(rows)):
        raise ValueError('Invalid TRAIN pair blend configuration')
    for i, row in enumerate(rows):
        partner = rows[(i + count) % len(rows)]
        for r in (row, partner):
            if not (r['split'] == 'train' or
                    (r['split'] == 'gallery' and r.get('source_split') == 'train')):
                raise ValueError('Blend requires TRAIN-only source photos')
        if (row['product_id'] != partner['product_id']
                or row['sample_id'] == partner['sample_id']
                or row['image_path'] == partner['image_path']):
            raise ValueError('Blend needs two distinct photos of the same SKU')
        other = images[(i + count) % len(rows)]
        if images[i].size != other.size or images[i].mode != other.mode:
            raise ValueError('Blend image size/mode mismatch')
    output, sources, weights = [], [], []
    for i, (im, row) in enumerate(zip(images, rows)):
        j = (i + count) % len(rows)
        weight = rng.uniform(low, high) if probability and rng.random() < probability else 0.
        output.append(Image.blend(im, images[j], weight) if weight else im)
        sources.append((row['sample_id'], rows[j]['sample_id']) if weight else (row['sample_id'],))
        weights.append(weight)
    return output, sources, weights


def training_source_exclusion(sources, gallery_ids, device=None):
    """Exclude every image used to construct a TRAIN query, not only its label."""
    if not sources or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('Invalid TRAIN gallery/source identities')
    known = set(gallery_ids)
    if any(not s or len(set(s)) != len(s) or not set(s) <= known for s in sources):
        raise ValueError('TRAIN source missing from gallery or repeated')
    return torch.tensor([[sid in source for sid in gallery_ids] for source in sources],
                        dtype=torch.bool, device=device)


def prepare_capacity_payload(payload, profile, protocol, fresh=False):
    expansion = profile.get('electronic_expansion')
    head = profile.get('head_expansion')
    modes = profile.get('ccd_readout_modes')
    if modes:
        if (expansion or head or fresh or profile.get('optical_only')
                or protocol != 'abo200_enrolled_sku_hash8train4query_v1'):
            raise ValueError('CCD readout ablation requires isolated enrolled continuation')
        if (modes != dict(vision='prefix_rows',language='fullfield_rows')
                or payload['metadata'].get('ccd_readout_modes',dict(vision='prefix_rows',language='prefix_rows'))
                != dict(vision='prefix_rows',language='prefix_rows')
                or payload['metadata'].get('retrieval_head','linear64') != 'linear64'):
            raise ValueError('CCD ablation must begin with original prefix readout and linear64 head')
        from .generalization import apply_contract
        converted = apply_contract(payload,dict(input_preprocessing=payload['metadata']['input_preprocessing'],
                                                ccd_readout_modes=modes))
        if converted['state_dict'] is not payload['state_dict']:
            raise ValueError('CCD mode conversion must not change any weights')
        return converted, dict(expanded=False,extra_parameters=0,role='Electronic CCD pooling ablation, not a new optical geometry',
            source_readout_modes=dict(vision='prefix_rows',language='prefix_rows'),target_readout_modes=modes,
            function_preserving=False,
            initialization='Reuse all tensors but re-evaluate fullfield L readout; old-contract score is NOT fallback in this run')
    if head:
        if expansion or fresh or protocol != 'abo200_enrolled_sku_hash8train4query_v1' or profile.get('optical_only'):
            raise ValueError('Bounded spatial readout requires enrolled continuation, no simultaneous teacher expansion')
        if head != 'spatial2x2_64' or payload['metadata'].get('retrieval_head','linear64') != 'linear64':
            raise ValueError('Spatial ablation must begin from original linear64 head')
        from .generalization import expand_retrieval_head
        converted = expand_retrieval_head(payload,head)
        before, after = payload['state_dict'], converted['state_dict']
        if before.keys() != after.keys() or any(not torch.equal(old,after[name]) for name,old in before.items()
                                                if name!='readout.projection.weight'):
            raise ValueError('Readout conversion modified protected weights')
        extra = after['readout.projection.weight'].numel()-before['readout.projection.weight'].numel()
        if extra != 49152:
            raise ValueError('Unexpected spatial readout parameter increase')
        return converted, dict(expanded=True,extra_parameters=extra,role='Bounded final readout candidate; not teacher distillation',
            changed_tensor_shapes=['readout.projection.weight'],
            initialization='Append768 zero-weight fixed spatial features to original384 normalized global features; finite precision must be re-evaluated',
            optical_frontend_alpha_electronic_residuals_unchanged_at_conversion=True)
    if expansion is None:
        return payload, dict(expanded=False, extra_parameters=0)
    if fresh or protocol != 'abo200_enrolled_sku_hash8train4query_v1' or profile.get('optical_only'):
        raise ValueError('Capacity teacher requires current enrolled-SKU continuation, not fresh/phase-only training')
    from .generalization import expand_electronic_context, expand_electronic_mlp
    converted = expand_electronic_mlp(
        expand_electronic_context(payload, expansion['kernels']), expansion['mlp_width'])
    before, after = payload['state_dict'], converted['state_dict']
    if before.keys() != after.keys():
        raise ValueError('Capacity conversion must not add branches or change tensor identities')
    changed = []
    for name, old in before.items():
        allowed = any(name.startswith(f'{m}.blocks.{i}.')
                      for m in ('vision', 'language') for i in (0, 1)) and (
            name.endswith('token_depthwise.weight') or any(
                name.endswith('mlp.' + suffix) for suffix in ('0.weight', '0.bias', '3.weight')))
        if not allowed and not torch.equal(old, after[name]):
            raise ValueError(f'Capacity conversion changed protected tensor: {name}')
        if old.shape != after[name].shape:
            changed.append(name)
    extra = sum(x.numel() for x in after.values()) - sum(x.numel() for x in before.values())
    return converted, dict(expanded=True, extra_parameters=extra, changed_tensor_shapes=changed,
        source_kernels=payload['metadata'].get('electronic_context_kernels', dict(vision=3, language=5)),
        target_kernels=expansion['kernels'], source_mlp_width=payload['metadata'].get('electronic_mlp_width', 384),
        target_mlp_width=expansion['mlp_width'],
        role='Larger optical-electronic teacher candidate; not the final compressed student',
        initialization='Zero-padded depthwise kernels and duplicated/halved MLP neurons preserve eval function algebraically; finite-precision initialization must be re-evaluated',
        optical_frontend_alpha_head_unchanged_at_conversion=True)


def train_ranking_loss(logits, positive, excluded, kind='nll'):
    """TRAIN nearest-positive/negative surrogate, never a test-time reranker.

    Logits are cosine/.1. The .2 logit margin means .02 cosine margin.
    Top1 needs *one* correct SKU view before every wrong SKU, not every
    positive ahead of every negative. Self matches participate in neither set.
    """
    positive = positive & ~excluded
    negative = ~positive & ~excluded
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError('Require nonself TRAIN positives and different-SKU negatives')
    valid = logits.masked_fill(excluded, -torch.inf)
    pos = logits.masked_fill(~positive | excluded, -torch.inf)
    if kind == 'nll':
        return (valid.logsumexp(1) - pos.logsumexp(1)).mean()
    if kind == 'top1_softplus':
        neg = logits.masked_fill(~negative, -torch.inf)
        return F.softplus(neg.amax(1) - pos.amax(1) + .2).mean()
    if kind == 'top1_squared_hinge':
        # TRAIN-only active-margin refinement: already satisfied queries have
        # zero ranking gradient. Shared weight updates may still move them;
        # this is not a guarantee of preserving every previous prediction.
        neg = logits.masked_fill(~negative, -torch.inf)
        violation = F.relu(neg.amax(1) - pos.amax(1) + .2)
        return .5 * violation.square().mean()
    if kind == 'two_view_softplus':
        # TRAIN only: two distinct nonself photos of this SKU contribute.
        # Do not force all seven views together or change inference relevance.
        if (positive.sum(1) < 2).any():
            raise ValueError('Two-view objective requires two distinct nonself TRAIN positives')
        neg = logits.masked_fill(~negative, -torch.inf)
        best_two = pos.topk(2, dim=1).values
        return F.softplus(neg.amax(1) - best_two.mean(1) + .2).mean()
    if kind == 'hybrid_nll_top1':
        # Fixed equal mixture: improve nearest-SKU ordering without discarding
        # the original all-gallery multi-positive probability objective.
        nll = (valid.logsumexp(1) - pos.logsumexp(1)).mean()
        neg = logits.masked_fill(~negative, -torch.inf)
        top1 = F.softplus(neg.amax(1) - pos.amax(1) + .2).mean()
        return .5 * (nll + top1)
    raise ValueError('Unknown TRAIN ranking loss')


def optical_parameter(name):
    return '.optics.experts.' in name or name.endswith('optics.global_phase') or name.endswith('raw_router_phase')


def projection_parameter(name):
    return name in ('readout.projection.weight', 'readout.projection.bias')


def configure_phase_head_scope(model):
    if model.readout.kind != 'linear64':
        raise ValueError('Phase/head-only fitting requires original linear64')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(optical_parameter(name) or projection_parameter(name))
    return non_optical_digest(model, exclude_projection=True)


def alpha_parameter(name):
    return name in {f'{modality}.block{block}_optical_fusion_logit'
                    for modality in ('vision','language') for block in (1,2)}


def frozen_except_alpha_digest(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not alpha_parameter(name):
            digest.update(f'{name}|{p.dtype}|{tuple(p.shape)}'.encode())
            digest.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure_alpha_scope(model):
    selected = [(n,p) for n,p in model.named_parameters() if alpha_parameter(n)]
    if len(selected) != 4 or any(p.numel() != 1 for _,p in selected):
        raise ValueError('Alpha-only requires exactly four existing scalar fusion logits')
    for n,p in model.named_parameters():
        p.requires_grad_(alpha_parameter(n))
    return frozen_except_alpha_digest(model)


def attach_train_readout_dropout(model, probability):
    """A training hook only: checkpoint has no new layer or inference behavior."""
    if model.readout.kind != 'linear64' or not 0 <= probability < 1:
        raise ValueError('Invalid original-head training dropout')
    def hook(module, values):
        if module.training and probability:
            return (F.dropout(values[0], probability, training=True),)
        return None
    return model.readout.projection.register_forward_pre_hook(hook)


def set_parameter_scope(params, router_only=False, optical_only=False):
    if router_only and optical_only:
        raise ValueError('Cannot combine router-only and all-optical-only scopes')
    for name, p in params:
        p.requires_grad_(name.endswith('raw_router_phase') if router_only else optical_parameter(name) if optical_only else True)


def optical_curriculum_scope(profile, external, pretrain_epochs):
    """An external optical-only stage followed by normal joint target fitting."""
    staged = profile.get('external_optical_only', False)
    if staged and (pretrain_epochs <= 0 or profile.get('optical_only') or profile.get('warmup')
                   or profile.get('teacher_weight') or profile.get('external_teacher_weight')
                   or any(profile.get(k) for k in ('electronic_expansion', 'head_expansion', 'ccd_readout_modes'))):
        raise ValueError('Optical pretraining requires a separate external stage without teacher/capacity/readout changes')
    return bool(profile.get('optical_only') or (staged and external))


def learning_rate_multiplier(profile, name, external, refined):
    if alpha_parameter(name):
        return profile.get('alpha_lr_multiplier', 1.)
    if projection_parameter(name):
        return profile.get('head_lr_multiplier', 1.)
    if name.endswith('raw_router_phase'):
        return profile.get('router_lr_multiplier', 5 if refined else 1)
    if optical_parameter(name):
        return profile.get('external_phase_lr_multiplier', profile.get('phase_lr_multiplier', 1.)) if external else profile.get('phase_lr_multiplier', 1.)
    return 1.


@torch.no_grad()
def update_trainable_ema(params, ema, decay=.99):
    # Updating an unchanged float as decay*x+(1-decay)*x may round by an ULP.
    # Frozen electronic parameters must remain BITWISE identical in phase-only runs.
    for name, p in params:
        if p.requires_grad:
            ema[name].mul_(decay).add_(p, alpha=1-decay)


def non_optical_digest(model, exclude_projection=False):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not optical_parameter(name) and not (exclude_projection and projection_parameter(name)):
            digest.update(f'{name}|{p.dtype}|{tuple(p.shape)}'.encode())
            digest.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
