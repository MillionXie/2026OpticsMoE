"""CPU TRAIN-only readout fitting on the fixed optical backbone.

normalize(A normalize(Wx+b)) == normalize((AW)x+Ab) in exact arithmetic.
Cached BF16 descriptors are approximate: a result is NOT deployable/official
until the folded checkpoint is independently re-encoded on the original GPU.
The explicit relu128 option instead trains the existing supported compact
nonlinear head: +32896 inference parameters, never described as a folded linear.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from .io import sha256, source_commit, write_json
from .retrieval_screen import load_screen, rank_instances, split_routing_report
from .retrieval_refine import train_ranking_loss
from .generalization import backward_with_sam


def fit_step(loss_closure, optimizer, parameters, sam_rho=0.):
    """TRAIN-only update; SAM replays dropout and restores weights before Adam.

    Uses the task's existing SAM implementation. Nothing is added to the saved
    model, and detached TRAIN inputs/anchor references remain outside optimizer.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        if sam_rho:
            raise ValueError('LBFGS requires deterministic full TRAIN closure, without SAM')
        original = [p.detach().clone() for p in parameters]
        audit = dict(rho=0., loss_gap=0., perturbation_norm=0.,
                     optimizer='lbfgs', closure_calls=0)
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = loss_closure()
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('LBFGS requires a finite scalar TRAIN loss')
            loss.backward()
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError('LBFGS encountered invalid TRAIN gradients')
            # Do not clip: strong-Wolfe search needs the objective's true gradient.
            audit['gradient_norm'] = float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
            if not math.isfinite(audit['gradient_norm']):
                raise ValueError('LBFGS gradient norm is nonfinite')
            audit['closure_calls'] += 1
            return loss
        try:
            loss = optimizer.step(closure)
            if any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError('LBFGS produced nonfinite parameters')
        except BaseException:
            with torch.no_grad():
                for p, old in zip(parameters, original):
                    p.copy_(old)
            raise
        return loss.detach(), audit
    result, sam = backward_with_sam(lambda: dict(loss=loss_closure()), optimizer, sam_rho)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return result['loss'].detach(), dict(sam, gradient_norm=float(norm))


@torch.no_grad()
def projection_vectors(inputs, weight, bias, precision='cpu_fp32'):
    """Selection only: replay the unchanged original head, not a new inference head.

    CUDA batch4 matches the raw verification contract. Optical/electronic
    backbone inputs are frozen, but final raw-image verification is still required.
    """
    if precision == 'cpu_fp32':
        return F.normalize(F.linear(inputs, weight, bias), dim=-1)
    if precision != 'cuda_bf16':
        raise ValueError('Unknown selection precision')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA selection requires an explicitly available GPU')
    w, b = weight.detach().cuda(), bias.detach().cuda()
    values = []
    for start in range(0, len(inputs), 4):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vector = F.normalize(F.linear(inputs[start:start+4].cuda(), w, b), p=2, dim=-1)
        values.append(vector.float().cpu())
    return torch.cat(values)


def fold_metric(payload, matrix):
    if payload['metadata'].get('retrieval_head', 'linear64') != 'linear64':
        raise ValueError('Metric folding requires the existing linear64 head')
    if matrix.shape != (64, 64) or not torch.isfinite(matrix).all():
        raise ValueError('Expected finite 64x64 matrix')
    result = dict(payload, state_dict=dict(payload['state_dict']))
    for suffix in ('weight', 'bias'):
        name = 'readout.projection.' + suffix
        old = payload['state_dict'][name]
        result['state_dict'][name] = (matrix.double() @ old.double()).to(old.dtype)
    return result


def metric_loss(matrix, train_vectors, train_labels, indices, anchor):
    if train_vectors.requires_grad or train_vectors.shape[1] != 64:
        raise ValueError('Only detached 64D TRAIN vectors may be fitted')
    if not math.isfinite(anchor) or anchor < 0:
        raise ValueError('Invalid identity anchor')
    z = F.normalize(train_vectors @ matrix.T, dim=-1)
    logits = z[indices] @ z.T / .1
    excluded = indices[:, None].eq(torch.arange(len(z))[None])
    positive = train_labels[indices, None].eq(train_labels[None]) & ~excluded
    if not positive.any(1).all():
        raise ValueError('Missing nonself TRAIN positive')
    logits = logits.masked_fill(excluded, -torch.inf)
    nll = (logits.logsumexp(1) - logits.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()
    penalty = (matrix - torch.eye(64)).square().sum() / 64
    return nll + anchor * penalty


def validate_cache(cache, rows, manifest_sha):
    if (cache.get('manifest_sha256') != manifest_sha
            or cache.get('ids') != [r['sample_id'] for r in rows]
            or cache['vectors'].shape != (len(rows), 64)
            or not torch.isfinite(cache['vectors']).all()):
        raise ValueError('Cache identity/order/values do not match the protocol')
    if (cache['vectors'].float().norm(dim=1) < 1e-8).any():
        raise ValueError('Zero cached descriptor')


@torch.no_grad()
def consistent_teacher_targets(vectors, labels, min_margin=0.):
    """Only TRAIN-to-TRAIN relations; GT decides whether to trust each teacher row."""
    if (vectors.requires_grad or vectors.ndim != 2 or len(vectors) != len(labels)
            or labels.ndim != 1 or not torch.isfinite(vectors).all()
            or not math.isfinite(min_margin) or min_margin < 0
            or len(vectors) < 2 or (vectors.float().norm(dim=1) < 1e-8).any()):
        raise ValueError('Require detached finite TRAIN teacher vectors and valid labels/margin')
    z = F.normalize(vectors.float(), dim=-1)
    excluded = torch.eye(len(z), dtype=torch.bool)
    positive = labels[:,None].eq(labels[None]) & ~excluded
    negative = ~positive & ~excluded
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError('Each TRAIN row needs nonself same-SKU and different-SKU references')
    similarity = z @ z.T
    pos = similarity.masked_fill(~positive, -torch.inf).amax(1)
    neg = similarity.masked_fill(~negative, -torch.inf).amax(1)
    eligible = pos - neg > min_margin  # Reject ties and wrong-SKU teacher retrievals.
    # Finite masking avoids 0 * -inf in KL; self has exactly zero probability.
    targets = (similarity / .1).masked_fill(excluded, -1e4).softmax(1)
    return targets.detach(), eligible.detach()


def load_consistent_teacher(path, expected_sha, rows, manifest_sha, labels, min_margin=0.):
    if sha256(path) != expected_sha:
        raise ValueError('Teacher cache SHA mismatch')
    report_path = path.parent / 'final_report.json'
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'complete' or report.get('model_kind') != 'qwen64'
            or report.get('frozen') is not True or report.get('trainable_parameters') != 0
            or report.get('manifest_sha256') != manifest_sha):
        raise ValueError('Require verified frozen Qwen64 on exactly this protocol')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    validate_cache(cache, rows, manifest_sha)
    n = len(labels)
    if n != 1600 or len(rows) != 2400 or any(r['split'] != 'gallery' for r in rows[:n]) or any(r['split'] != 'query' for r in rows[n:]):
        raise ValueError('Teacher fitting requires fixed1600 TRAIN/gallery before800 QUERY')
    # Only TRAIN copies survive this loader. QUERY values/labels never enter a target.
    targets, eligible = consistent_teacher_targets(cache['vectors'][:n].float().detach().clone(), labels, min_margin)
    return targets, eligible, dict(cache_sha256=expected_sha, report_sha256=sha256(report_path),
        source_commit=report.get('source_commit'), training_rows=n, eligible_rows=int(eligible.sum()),
        min_correct_minus_wrong_margin=min_margin, temperature=.1, query_excluded=True,
        inference_teacher_required=False, gate='TRAIN best same-SKU cosine > best wrong-SKU cosine + min_margin; self excluded',
        loss='KL(teacher TRAIN-gallery probability || student clean TRAIN-gallery probability), mean over eligible sampled TRAIN rows only; no temperature-square multiplier')


def teacher_relation_loss(student, targets, eligible, indices):
    n = len(student)
    if (student.ndim != 2 or targets.shape != (n,n) or targets.requires_grad
            or eligible.shape != (n,) or eligible.dtype != torch.bool
            or indices.ndim != 1 or not torch.isfinite(student).all()
            or not torch.isfinite(targets).all() or (targets < 0).any()):
        raise ValueError('Invalid detached TRAIN relation target or student shape')
    selected = indices[eligible[indices]]
    if not len(selected):
        return student.sum() * 0.
    z = F.normalize(student.float(), dim=-1)
    logits = z[selected] @ z.T / .1
    excluded = selected[:,None].eq(torch.arange(n)[None])
    log_prob = logits.masked_fill(excluded, -1e4).log_softmax(1)
    return F.kl_div(log_prob, targets[selected], reduction='batchmean')


def replace_projection(payload, weight, bias):
    if payload['metadata'].get('retrieval_head', 'linear64') != 'linear64':
        raise ValueError('Require the original linear64 head')
    if weight.shape != (64, 384) or bias.shape != (64,) or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('Invalid original linear64 parameters')
    result = dict(payload, state_dict=dict(payload['state_dict']))
    for suffix, value in [('weight', weight), ('bias', bias)]:
        key = 'readout.projection.' + suffix
        if payload['state_dict'][key].shape != value.shape:
            raise ValueError('Existing projection shape mismatch')
        result['state_dict'][key] = value.detach().clone().to(payload['state_dict'][key].dtype)
    return result


@torch.no_grad()
def centered_projection_bias(weight, bias, train_inputs, strength):
    """Fold TRAIN pre-L2 centering into the existing bias; never an extra layer."""
    if (not math.isfinite(strength) or not 0 <= strength <= 1
            or train_inputs.requires_grad or train_inputs.ndim != 2
            or train_inputs.shape[1] != 384 or not len(train_inputs)
            or weight.shape != (64,384) or bias.shape != (64,)):
        raise ValueError('Invalid TRAIN projection centering inputs')
    if not all(torch.isfinite(t).all() for t in (weight,bias,train_inputs)):
        raise ValueError('Nonfinite TRAIN projection centering inputs')
    if strength == 0:
        return bias.detach().clone()
    mean = F.linear(train_inputs,weight,bias).mean(0)
    return bias.detach() - strength * mean


def bounded_diagonal_projection(raw_gain, reference):
    """TRAIN 64 degrees of freedom, folded into existing W/b for inference.

    Positive gains exp(.1*tanh(raw)) stay within exp(+/- .1); raw=0 exactly
    preserves the source parameters. No rotations, offsets or new head layers.
    """
    weight, bias = reference
    if (raw_gain.shape != (64,) or weight.shape != (64, 384) or bias.shape != (64,)
            or weight.requires_grad or bias.requires_grad
            or any(not torch.isfinite(t).all() for t in (raw_gain, weight, bias))):
        raise ValueError('Diagonal fit requires 64 finite gains and detached original W/b')
    gain = (.1 * raw_gain.tanh()).exp()
    return gain[:, None] * weight, gain * bias


def initialize_relu_projection(payload):
    """Reuse the supported compact head and its existing +/- initialization."""
    from .generalization import expand_retrieval_head
    from .model import RetrievalHead
    converted = expand_retrieval_head(payload, 'relu128')
    head = RetrievalHead('relu128').projection
    prefix = 'readout.projection.'
    head.load_state_dict({k[len(prefix):]: v for k, v in converted['state_dict'].items()
                          if k.startswith(prefix)}, strict=True)
    return converted, head


def replace_relu_projection(converted, head):
    if converted['metadata'].get('retrieval_head') != 'relu128':
        raise ValueError('Require the explicitly converted relu128 architecture')
    state = dict(converted['state_dict'])
    prefix = 'readout.projection.'
    expected = {k[len(prefix):] for k in state if k.startswith(prefix)}
    if expected != set(head.state_dict()):
        raise ValueError('ReLU head tensor identity mismatch')
    for name, value in head.state_dict().items():
        key = prefix + name
        if value.shape != state[key].shape or not torch.isfinite(value).all():
            raise ValueError('Invalid ReLU projection tensor')
        state[key] = value.detach().clone().to(state[key].dtype)
    return dict(converted, state_dict=state)


@torch.no_grad()
def module_projection_vectors(inputs, head, precision='cpu_fp32'):
    if precision == 'cpu_fp32':
        return F.normalize(head(inputs.float()), p=2, dim=-1)
    if precision != 'cuda_bf16' or not torch.cuda.is_available():
        raise ValueError('Require available CUDA for the declared head replay')
    module = copy.deepcopy(head).eval().cuda()
    values = []
    for start in range(0, len(inputs), 4):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = F.normalize(module(inputs[start:start+4].cuda()), p=2, dim=-1)
        values.append(z.float().cpu())
    return torch.cat(values)


def nonlinear_projection_loss(head, train_inputs, train_labels, indices, reference,
                              anchor, input_dropout=0., ranking_loss='nll'):
    if (train_inputs.requires_grad or train_inputs.ndim != 2 or train_inputs.shape[1] != 384
            or not math.isfinite(anchor) or anchor < 0
            or not math.isfinite(input_dropout) or not 0 <= input_dropout < 1):
        raise ValueError('Require detached TRAIN384 inputs and valid regularizers')
    parameters = list(head.parameters())
    if (len(parameters) != len(reference) or any(r.requires_grad or r.shape != p.shape
            for p, r in zip(parameters, reference))):
        raise ValueError('Nonlinear anchor reference must be frozen and shape matched')
    z = F.normalize(head(F.dropout(train_inputs, input_dropout, training=True)), dim=-1)
    q = (F.normalize(head(F.dropout(train_inputs[indices], input_dropout, training=True)), dim=-1)
         if input_dropout else z[indices])
    logits = q @ z.T / .1
    excluded = indices[:, None].eq(torch.arange(len(z))[None])
    positive = train_labels[indices, None].eq(train_labels[None]) & ~excluded
    loss = train_ranking_loss(logits, positive, excluded, ranking_loss)
    penalty = sum((p-r).square().sum() for p, r in zip(parameters, reference))
    scale = sum(r.square().sum() for r in reference).clamp_min(1e-12)
    return loss + anchor * penalty / scale


def projection_loss(weight, bias, train_inputs, train_labels, indices, reference, anchor, input_dropout=0., ranking_loss='nll'):
    """Only detached TRAIN inputs; all gallery rows receive parameter gradients."""
    if train_inputs.requires_grad or train_inputs.ndim != 2 or train_inputs.shape[1] != 384:
        raise ValueError('Only detached 384D TRAIN inputs may be fitted')
    if not math.isfinite(anchor) or anchor < 0:
        raise ValueError('Invalid projection anchor')
    if not math.isfinite(input_dropout) or not 0 <= input_dropout < 1:
        raise ValueError('Invalid train-only input dropout')
    if input_dropout:
        # Independent query/gallery masks, TRAIN only. No inference module added.
        z = F.normalize(F.linear(F.dropout(train_inputs, p=input_dropout, training=True), weight, bias), dim=-1)
        q = F.normalize(F.linear(F.dropout(train_inputs[indices], p=input_dropout, training=True), weight, bias), dim=-1)
    else:
        z = F.normalize(F.linear(train_inputs, weight, bias), dim=-1)
        q = z[indices]
    logits = q @ z.T / .1
    excluded = indices[:, None].eq(torch.arange(len(z))[None])
    positive = train_labels[indices, None].eq(train_labels[None]) & ~excluded
    if not positive.any(1).all():
        raise ValueError('Missing nonself TRAIN positive')
    nll = train_ranking_loss(logits, positive, excluded, ranking_loss)
    w0, b0 = reference
    if w0.requires_grad or b0.requires_grad:
        raise ValueError('Anchor reference must remain frozen')
    penalty = ((weight-w0).square().sum()+(bias-b0).square().sum()) / (w0.square().sum()+b0.square().sum()).clamp_min(1e-12)
    return nll + anchor * penalty


def validate_projection_cache(cache, checkpoint_sha, payload):
    value = cache.get('readout_inputs')
    if (cache.get('checkpoint_sha256') != checkpoint_sha or not isinstance(value, torch.Tensor)
            or value.shape != (len(cache['ids']), 384) or not torch.isfinite(value).all()):
        raise ValueError('Missing/mismatched original projection inputs')
    projected = F.normalize(F.linear(value.float(), payload['state_dict']['readout.projection.weight'].float(),
        payload['state_dict']['readout.projection.bias'].float()), dim=-1)
    disagreement = 1-F.cosine_similarity(projected, cache['vectors'].float(), dim=-1)
    if disagreement.max() > .001:
        raise ValueError('Projection inputs do not reproduce source descriptors within BF16 tolerance')


def validate_optimizer_recipe(args):
    kind = getattr(args, 'optimizer', 'adam')
    if kind not in ('adam', 'lbfgs'):
        raise ValueError('Unknown readout optimizer')
    if kind == 'lbfgs' and (args.fit_space != 'projection384' or args.batch_size != 1600
            or args.ranking_loss != 'nll' or args.input_dropout or args.sam_rho
            or getattr(args, 'train_center', 0.) or args.steps < 1):
        raise ValueError('LBFGS requires original projection, full1600 TRAIN, smooth NLL, no dropout/SAM/centering')
    return kind


def run(args):
    optimizer_kind = validate_optimizer_recipe(args)
    teacher_weight = getattr(args, 'teacher_weight', 0.)
    teacher_path = getattr(args, 'teacher_cache', None)
    teacher_sha = getattr(args, 'expected_teacher_sha256', None)
    teacher_margin = getattr(args, 'teacher_min_margin', 0.)
    if (not math.isfinite(teacher_weight) or not 0 <= teacher_weight <= 1
            or not math.isfinite(teacher_margin) or teacher_margin < 0
            or bool(teacher_weight) != bool(teacher_path) or bool(teacher_weight) != bool(teacher_sha)
            or (teacher_weight and (args.fit_space != 'projection384'
                                   or getattr(args, 'train_center', 0.) or optimizer_kind != 'adam'))):
        raise ValueError('Teacher relations require pinned cache, weight in(0,1], original projection384 Adam, no centering')
    diagonal = args.fit_space == 'diagonal64'
    nonlinear = args.fit_space == 'relu128'
    if (diagonal or nonlinear) and getattr(args, 'train_center', 0.):
        raise ValueError('Constrained/nonlinear fit cannot also calibrate the source bias')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol, groups = load_screen(args.manifest, args.data)
    if protocol['protocol'] != 'abo200_enrolled_sku_hash8train4query_v1':
        raise ValueError('Only the fixed enrolled ABO protocol is supported')
    rows = groups['gallery'] + groups['query']
    if (len(groups['gallery']), len(groups['query'])) != (1600, 800):
        raise ValueError('Require the original 1600/800 protocol')
    train_ids = {r['sample_id'] for r in groups['train']}
    if train_ids != {r['sample_id'] for r in groups['gallery']} or train_ids & {r['sample_id'] for r in groups['query']}:
        raise ValueError('TRAIN/gallery/QUERY separation mismatch')
    checkpoint = args.source_run / 'best.pt'
    if sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError('Source checkpoint SHA mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest_sha = sha256(args.manifest)
    if payload.get('manifest_sha256') != manifest_sha:
        raise ValueError('Checkpoint protocol mismatch')
    verification = args.verification_dir or args.source_run / 'verification'
    report = json.loads((verification / 'final_report.json').read_text())
    if (report.get('checkpoint_sha256') != args.expected_checkpoint_sha256
            or report.get('manifest_sha256') != manifest_sha or report.get('status') != 'complete'):
        raise ValueError('Require completed source verification for this checkpoint')
    caches = {}
    for kind in ('normal', 'remove_optical'):
        cache = torch.load(verification / f'{kind}_features.pt', map_location='cpu', weights_only=True)
        validate_cache(cache, rows, manifest_sha)
        caches[kind] = cache
    z = F.normalize(caches['normal']['vectors'].float(), dim=-1)
    train_z = z[:1600].detach().clone()  # ONLY this tensor enters optimization.
    names = sorted({r['product_id'] for r in groups['train']})
    if len(names) != 200:
        raise ValueError('Require 200 training SKUs')
    train_labels = torch.tensor([names.index(r['product_id']) for r in groups['gallery']])
    teacher_audit = None
    if teacher_weight:
        teacher_targets, teacher_eligible, teacher_audit = load_consistent_teacher(
            teacher_path, teacher_sha, rows, manifest_sha, train_labels, teacher_margin)
    baseline, _ = rank_instances(z, rows)
    if abs(baseline['hit_at_1'] - args.expected_hit) > 1e-9:
        raise ValueError('Cached baseline did not reproduce the expected score')
    matrix = torch.nn.Parameter(torch.eye(64))
    projection = args.fit_space in ('projection384', 'diagonal64', 'relu128')
    if args.selection_precision != 'cpu_fp32' and not projection:
        raise ValueError('CUDA head replay requires direct original projection fitting')
    if projection:
        for cache in caches.values():
            validate_projection_cache(cache, args.expected_checkpoint_sha256, payload)
        input_vectors = caches['normal']['readout_inputs'].float()
        train_inputs = input_vectors[:1600].detach().clone()
        weight = torch.nn.Parameter(payload['state_dict']['readout.projection.weight'].float().clone())
        bias = torch.nn.Parameter(payload['state_dict']['readout.projection.bias'].float().clone())
        reference = (weight.detach().clone(), bias.detach().clone())
        parameters = [weight, bias]
        if diagonal:
            raw_gain = torch.nn.Parameter(torch.zeros(64))
            parameters = [raw_gain]
            weight, bias = bounded_diagonal_projection(raw_gain, reference)
        if args.selection_precision == 'cuda_bf16':
            replay = projection_vectors(input_vectors, weight, bias, args.selection_precision)
            if not torch.equal(replay, caches['normal']['vectors']):
                raise ValueError('Initial CUDA head replay must be BITWISE equal to this source raw verification; do not silently change GPU/batch/precision')
        if nonlinear:
            converted, head = initialize_relu_projection(payload)
            parameters = list(head.parameters())
            head_reference = [p.detach().clone() for p in parameters]
    else:
        parameters = [matrix]
    center_strength = getattr(args, 'train_center', 0.)
    if center_strength:
        if not projection:
            raise ValueError('TRAIN centering requires original projection384')
        with torch.no_grad():
            # Only the already detached 1600 TRAIN inputs enter this statistic.
            # Source raw replay above occurs BEFORE calibration.
            bias.copy_(centered_projection_bias(weight, bias, train_inputs, center_strength))
    optimizer = (torch.optim.LBFGS(parameters, lr=args.lr, max_iter=1, history_size=10,
        line_search_fn='strong_wolfe') if optimizer_kind == 'lbfgs'
        else torch.optim.Adam(parameters, lr=args.lr))
    args.output.mkdir(parents=True)
    identity = dict(source_commit=source_commit(), command=sys.argv, pid=os.getpid(),
        source_checkpoint_sha256=sha256(checkpoint), manifest_sha256=manifest_sha,
        cache_sha256={k: sha256(verification / f'{k}_features.pt') for k in caches},
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        device='cpu', training_rows=1600, query_rows=800, test_selected=True,
        extra_inference_parameters=32896 if nonlinear else 0,
        inference=('Existing supported Linear384->128/ReLU/Linear128->64 then L2; explicitly larger readout, unchanged frozen optical/electronic backbone' if nonlinear else 'Same original Linear(384,64); 64 bounded positive gains folded rowwise into original W/b' if diagonal else 'Same original Linear(384,64), direct weight/bias fitting' if projection else 'Same linear64 layer, Wnew=A@W, bnew=A@b'),
        status='cached_candidate_only', raw_gpu_verification_required=True,
        optical_weights_changed=False, training_note='Only TRAIN gallery labels/vectors enter loss; QUERY used for periodic selection')
    identity['selection_precision'] = args.selection_precision
    identity['teacher_relations'] = dict(enabled=bool(teacher_weight), weight=teacher_weight, audit=teacher_audit)
    identity['fitted_parameter_count'] = sum(p.numel() for p in parameters)
    identity['nonlinear_conversion'] = dict(enabled=nonlinear,
        initialization='First layer [W;-W],[b;-b]; second layer [I,-I],0: algebraically source preserving, finite precision must be evaluated separately',
        source_architecture='linear64', target_architecture='relu128' if nonlinear else None,
        selection='Converted step0 is independently scored, not assigned the source 83%; original source retained outside this run')
    identity['diagonal_constraint'] = dict(enabled=diagonal, gain_bounds=[math.exp(-.1), math.exp(.1)],
        note='g=exp(.1*tanh(raw64)); W_new=g[:,None]*W_source, b_new=g*b_source; no new inference tensor')
    identity['optimizer_recipe'] = dict(kind=optimizer_kind,
        note='Full TRAIN smooth NLL; one quasi-Newton iteration per outer step, strong-Wolfe TRAIN-only line search, history10, constant LR, no gradient clipping; query never enters closure' if optimizer_kind == 'lbfgs' else 'Original minibatch Adam/SAM, cosine LR and gradient norm clip1')
    identity['train_center'] = dict(strength=center_strength, rows=1600,
        note='Existing bias only: b_new=b-strength*mean_TRAIN(Wx+b), pre L2. QUERY excluded. Zero steps means calibration only, not SGD training.')
    identity['selection_gpu'] = torch.cuda.get_device_name() if args.selection_precision == 'cuda_bf16' else None
    identity['selection_note'] = 'Original head CUDA autocast BF16, batch4, source raw replay bitwise checked; CPU FP32 TRAIN gradients; final raw verification required' if args.selection_precision == 'cuda_bf16' else 'Legacy CPU FP32 projection scoring; may differ from raw autocast inference'
    identity['ranking_objective'] = {
        'nll': 'Original all-gallery multi-positive NLL, temperature .1',
        'top1_softplus': 'softplus((nearest_wrong_cosine-nearest_correct_cosine+.02)/.1), mean over TRAIN queries, self excluded',
        'two_view_softplus': 'softplus((nearest_wrong_cosine-mean_two_nearest_correct_photo_cosines+.02)/.1), TRAIN only, two distinct nonself positive images; no inference reranking',
        'top1_squared_hinge': '.5*relu((nearest_wrong_cosine-nearest_correct_cosine+.02)/.1)^2, TRAIN nonself only; zero ranking gradient after margin is satisfied, shared parameter updates may still change predictions',
        'hybrid_nll_top1': 'Fixed .5 original multi-positive NLL + .5 nearest-SKU softplus, temperature .1, top1 cosine margin .02; TRAIN self excluded',
    }[args.ranking_loss]
    identity['sam'] = dict(rho=args.sam_rho,
        note='TRAIN-only existing SAM; same sampled indices and dropout in both passes; restore before Adam; no inference changes')
    write_json(args.output / 'execution.json', identity)
    status = dict(status='running', pid=os.getpid(), source_commit=identity['source_commit'])
    history, best_score = [], (-1., -1.)
    best_matrix = matrix.detach().clone()
    started = time.time()

    def save(name, step):
        candidate = (replace_relu_projection(converted, head) if nonlinear else
                     replace_projection(payload, weight, bias) if projection else fold_metric(payload, matrix.detach()))
        candidate.update(source_commit=identity['source_commit'], epoch=step, variant='metric',
            stage='readout_metric_fit', test_selected=True, manifest_sha256=manifest_sha,
            cached_candidate_only=True, source_checkpoint_sha256=identity['source_checkpoint_sha256'])
        torch.save(candidate, args.output / name)

    try:
        for step in range(args.steps + 1):
            if step:
                optimizer.param_groups[0]['lr'] = (args.lr if optimizer_kind == 'lbfgs'
                    else args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps))))
                idx = torch.arange(1600) if optimizer_kind == 'lbfgs' else torch.randperm(1600)[:args.batch_size]
                def loss_closure():
                    if teacher_weight:
                        supervised = projection_loss(weight, bias, train_inputs, train_labels, idx,
                            reference, args.anchor, args.input_dropout, args.ranking_loss)
                        # Clean TRAIN view for matching the frozen teacher cache;
                        # GT loss retains its configured independent dropout.
                        student = F.linear(train_inputs, weight, bias)
                        kd = teacher_relation_loss(student, teacher_targets, teacher_eligible, idx)
                        return supervised + teacher_weight * kd
                    if nonlinear:
                        return nonlinear_projection_loss(head, train_inputs, train_labels, idx,
                            head_reference, args.anchor, args.input_dropout, args.ranking_loss)
                    if diagonal:
                        w, b = bounded_diagonal_projection(raw_gain, reference)
                        return projection_loss(w, b, train_inputs, train_labels, idx, reference,
                            args.anchor, args.input_dropout, args.ranking_loss)
                    return (projection_loss(weight, bias, train_inputs, train_labels, idx, reference, args.anchor, args.input_dropout, args.ranking_loss)
                            if projection else metric_loss(matrix, train_z, train_labels, idx, args.anchor))
                loss, step_audit = fit_step(loss_closure, optimizer, parameters, args.sam_rho)
            if step % args.eval_every == 0 or step == args.steps:
                with torch.no_grad():
                    if diagonal:
                        weight, bias = bounded_diagonal_projection(raw_gain, reference)
                    vectors = (module_projection_vectors(input_vectors, head, args.selection_precision) if nonlinear else
                               projection_vectors(input_vectors, weight, bias, args.selection_precision)
                               if projection else F.normalize(z @ matrix.T, dim=-1))
                    metrics, _ = rank_instances(vectors, rows)
                    sim = vectors[:1600] @ vectors[:1600].T
                    sim.fill_diagonal_(-torch.inf)
                    train_hit = float((train_labels[sim.argmax(1)] == train_labels).float().mean())
                if nonlinear:
                    delta = float(torch.sqrt(sum((p.detach()-r).square().sum()
                                                  for p, r in zip(parameters, head_reference))))
                elif projection:
                    delta = float(torch.sqrt((weight.detach()-reference[0]).square().sum()
                                             +(bias.detach()-reference[1]).square().sum()))
                else:
                    delta = float((matrix.detach()-torch.eye(64)).norm())
                entry = dict(step=step, cached_test=metrics, train_hit_at_1=train_hit,
                    fit_space=args.fit_space,
                    parameter_delta_frobenius=delta,
                    loss=float(loss.detach()) if step else None)
                entry['optimizer_audit'] = step_audit if step else None
                if teacher_weight:
                    with torch.no_grad():
                        entry['clean_train_teacher_kl'] = float(teacher_relation_loss(
                            F.linear(train_inputs, weight, bias), teacher_targets, teacher_eligible,
                            torch.arange(1600)))
                if diagonal:
                    gain = (.1 * raw_gain.detach().tanh()).exp()
                    entry['diagonal_gain_range'] = [float(gain.min()), float(gain.max())]
                score = (metrics['hit_at_1'], metrics['map_at_10'])
                if score > best_score:
                    best_score, best_matrix = score, matrix.detach().clone()
                    save('best.pt', step)
                save('last.pt', step)
                history.append(entry)
                write_json(args.output / 'history.json', history)
                status.update(step=step)
                write_json(args.output / 'status.json', status)
                print(json.dumps(entry), flush=True)
        if nonlinear:
            best_payload = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
            prefix = 'readout.projection.'
            head.load_state_dict({k[len(prefix):]:v for k,v in best_payload['state_dict'].items() if k.startswith(prefix)}, strict=True)
            removal_vectors = module_projection_vectors(caches['remove_optical']['readout_inputs'].float(), head, args.selection_precision)
        elif projection:
            best_payload = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
            removal_vectors = projection_vectors(caches['remove_optical']['readout_inputs'].float(),
                best_payload['state_dict']['readout.projection.weight'].float(),
                best_payload['state_dict']['readout.projection.bias'].float(), args.selection_precision)
        else:
            removal_vectors = F.normalize(caches['remove_optical']['vectors'].float() @ best_matrix.T, dim=-1)
        removal, _ = rank_instances(removal_vectors, rows)
        routing = split_routing_report(caches['normal']['router_selected_mask'], rows)
        write_json(args.output / 'final_report.json', dict(identity, cached_best_hit=best_score[0],
            cached_remove_optical=removal, inherited_routing=routing,
            best_sha256=sha256(args.output / 'best.pt'), last_sha256=sha256(args.output / 'last.pt'),
            elapsed_seconds=time.time()-started))
        status.update(status='complete')
    except BaseException as exc:
        status.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        write_json(args.output / 'status.json', status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'manifest', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--verification-dir', type=Path, help='Completed source raw verification directory (default source-run/verification)')
    p.add_argument('--fit-space', choices=['metric64', 'projection384', 'diagonal64', 'relu128'], default='metric64',
        help='diagonal64 folds64 gains into original W/b; relu128 explicitly adds a supported compact ReLU head (+32896 inference parameters)')
    p.add_argument('--input-dropout', type=float, default=0., help='TRAIN-only independent query/gallery feature dropout for projection384/diagonal64/relu128; never used in evaluation')
    p.add_argument('--train-center', type=float, default=0., help='Fold TRAIN pre-L2 feature mean subtraction into existing bias, strength [0,1]; projection384 only. With positive strength, --steps 0 permits calibration without SGD.')
    p.add_argument('--ranking-loss', choices=['nll', 'top1_softplus', 'hybrid_nll_top1', 'two_view_softplus', 'top1_squared_hinge'], default='nll',
        help='TRAIN objective only; top1_softplus uses nearest positive/negative, cosine margin .02, temperature .1; hybrid equally mixes original NLL and top1. Alternatives require projection384/diagonal64')
    p.add_argument('--expected-hit', type=float, required=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--optimizer', choices=['adam','lbfgs'], default='adam',
        help='LBFGS requires projection384, batch1600, NLL, no dropout/SAM/centering; deterministic full-TRAIN closure, constant LR, no new inference layers.')
    p.add_argument('--anchor', type=float, default=1.)
    p.add_argument('--sam-rho', type=float, default=0., help='TRAIN-only SAM radius on existing fitted head; zero preserves ordinary Adam')
    p.add_argument('--selection-precision', choices=['cpu_fp32','cuda_bf16'], default='cpu_fp32',
        help='Selection-only original head replay; CUDA uses raw batch4 autocast and requires bitwise source reproduction. Training remains CPU FP32.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--teacher-cache', type=Path, help='Optional pinned frozen Qwen64 cache; only first1600 TRAIN/gallery rows enter relation targets')
    p.add_argument('--expected-teacher-sha256')
    p.add_argument('--teacher-weight', type=float, default=0., help='TRAIN-only GT-consistent relation KL weight; zero leaves original training unchanged')
    p.add_argument('--teacher-min-margin', type=float, default=0., help='Teacher same-SKU vs wrong-SKU TRAIN cosine margin must exceed this value; ties rejected')
    args = p.parse_args()
    if not math.isfinite(args.train_center) or not 0 <= args.train_center <= 1 or (args.train_center and args.fit_space != 'projection384'):
        p.error('TRAIN centering requires strength [0,1] and projection384')
    if args.steps < 0 or (args.steps == 0 and not args.train_center) or args.eval_every < 1 or not 2 <= args.batch_size <= 1600 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.anchor) or args.anchor < 0:
        p.error('Invalid training bounds')
    if not math.isfinite(args.input_dropout) or not 0 <= args.input_dropout < 1 or (args.input_dropout and args.fit_space not in ('projection384', 'diagonal64', 'relu128')):
        p.error('Input dropout must be in [0,1), supported only with projection384/diagonal64/relu128')
    if args.ranking_loss != 'nll' and args.fit_space not in ('projection384', 'diagonal64', 'relu128'):
        p.error('Alternative ranking loss requires projection384/diagonal64/relu128')
    if not math.isfinite(args.sam_rho) or args.sam_rho < 0:
        p.error('SAM radius must be finite and nonnegative')
    run(args)


if __name__ == '__main__':
    main()
