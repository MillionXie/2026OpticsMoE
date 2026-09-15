"""CPU TRAIN-only metric fitting, folded into the existing linear64 head.

normalize(A normalize(Wx+b)) == normalize((AW)x+Ab) in exact arithmetic.
Cached BF16 descriptors are approximate: a result is NOT deployable/official
until the folded checkpoint is independently re-encoded on the original GPU.
"""
import argparse
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
    baseline, _ = rank_instances(z, rows)
    if abs(baseline['hit_at_1'] - args.expected_hit) > 1e-9:
        raise ValueError('Cached baseline did not reproduce the expected score')
    matrix = torch.nn.Parameter(torch.eye(64))
    projection = args.fit_space == 'projection384'
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
        if args.selection_precision == 'cuda_bf16':
            replay = projection_vectors(input_vectors, weight, bias, args.selection_precision)
            if not torch.equal(replay, caches['normal']['vectors']):
                raise ValueError('Initial CUDA head replay must be BITWISE equal to this source raw verification; do not silently change GPU/batch/precision')
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
        extra_inference_parameters=0, inference=('Same original Linear(384,64), direct weight/bias fitting' if projection else 'Same linear64 layer, Wnew=A@W, bnew=A@b'),
        status='cached_candidate_only', raw_gpu_verification_required=True,
        optical_weights_changed=False, training_note='Only TRAIN gallery labels/vectors enter loss; QUERY used for periodic selection')
    identity['selection_precision'] = args.selection_precision
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
        candidate = replace_projection(payload, weight, bias) if projection else fold_metric(payload, matrix.detach())
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
                    return (projection_loss(weight, bias, train_inputs, train_labels, idx, reference, args.anchor, args.input_dropout, args.ranking_loss)
                            if projection else metric_loss(matrix, train_z, train_labels, idx, args.anchor))
                loss, step_audit = fit_step(loss_closure, optimizer, parameters, args.sam_rho)
            if step % args.eval_every == 0 or step == args.steps:
                with torch.no_grad():
                    vectors = (projection_vectors(input_vectors, weight, bias, args.selection_precision)
                               if projection else F.normalize(z @ matrix.T, dim=-1))
                    metrics, _ = rank_instances(vectors, rows)
                    sim = vectors[:1600] @ vectors[:1600].T
                    sim.fill_diagonal_(-torch.inf)
                    train_hit = float((train_labels[sim.argmax(1)] == train_labels).float().mean())
                entry = dict(step=step, cached_test=metrics, train_hit_at_1=train_hit,
                    fit_space=args.fit_space,
                    parameter_delta_frobenius=(float(torch.sqrt((weight.detach()-reference[0]).square().sum()+(bias.detach()-reference[1]).square().sum()))
                        if projection else float((matrix.detach()-torch.eye(64)).norm())),
                    loss=float(loss.detach()) if step else None)
                entry['optimizer_audit'] = step_audit if step else None
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
        if projection:
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
    p.add_argument('--fit-space', choices=['metric64', 'projection384'], default='metric64')
    p.add_argument('--input-dropout', type=float, default=0., help='TRAIN-only independent query/gallery feature dropout for projection384; never used in evaluation')
    p.add_argument('--train-center', type=float, default=0., help='Fold TRAIN pre-L2 feature mean subtraction into existing bias, strength [0,1]; projection384 only. With positive strength, --steps 0 permits calibration without SGD.')
    p.add_argument('--ranking-loss', choices=['nll', 'top1_softplus', 'hybrid_nll_top1', 'two_view_softplus', 'top1_squared_hinge'], default='nll',
        help='TRAIN objective only; top1_softplus uses nearest positive/negative, cosine margin .02, temperature .1; hybrid equally mixes original NLL and top1. Alternatives require projection384')
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
    args = p.parse_args()
    if not math.isfinite(args.train_center) or not 0 <= args.train_center <= 1 or (args.train_center and args.fit_space != 'projection384'):
        p.error('TRAIN centering requires strength [0,1] and projection384')
    if args.steps < 0 or (args.steps == 0 and not args.train_center) or args.eval_every < 1 or not 2 <= args.batch_size <= 1600 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.anchor) or args.anchor < 0:
        p.error('Invalid training bounds')
    if not math.isfinite(args.input_dropout) or not 0 <= args.input_dropout < 1 or (args.input_dropout and args.fit_space != 'projection384'):
        p.error('Input dropout must be in [0,1), supported only with projection384')
    if args.ranking_loss != 'nll' and args.fit_space != 'projection384':
        p.error('Alternative ranking loss requires projection384')
    if not math.isfinite(args.sam_rho) or args.sam_rho < 0:
        p.error('SAM radius must be finite and nonnegative')
    run(args)


if __name__ == '__main__':
    main()
