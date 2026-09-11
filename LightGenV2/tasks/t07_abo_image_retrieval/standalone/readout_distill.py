"""Fit the existing Linear384->64 to train-only frozen teacher descriptors.

No inference module, optical tensor, LayerNorm or preprocessing is changed.
Ridge strength is selected on held-out TRAIN products, never task test images.
The final fit uses all original+external training rows. This is a readout-only
diagnostic/optimization, not a claim that optical phases were updated again.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def ridge_fit(features, targets, strength):
    """Centered multi-output ridge; intercept unpenalized, scale dimensionless."""
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets) or len(features) < 2:
        raise ValueError('Aligned training matrices with at least two rows required')
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError('Positive finite ridge strength required')
    x, y = features.detach().cpu().double(), targets.detach().cpu().double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite training matrices')
    xm, ym = x.mean(0), y.mean(0)
    x, y = x-xm, y-ym
    covariance = x.T@x/len(x)
    scale = covariance.trace()/features.shape[1]
    if scale <= 0:
        raise ValueError('Constant features cannot fit a readout')
    penalty = float(strength*scale)
    weight = torch.linalg.solve(covariance+penalty*torch.eye(features.shape[1], dtype=torch.float64), x.T@y/len(x))
    bias = ym-xm@weight
    return weight.T.float(), bias.float(), penalty


def training_product_partition(samples, seed=42):
    """Deterministic 80/20 product split inside each TRAIN category only."""
    groups = {}
    for s in samples:
        if s.split != 'train':
            raise ValueError('Readout fitting accepts training samples only')
        groups.setdefault(s.category_id, set()).add(s.product_id)
    rng = random.Random(seed)
    held = set()
    for category in sorted(groups):
        products = sorted(groups[category])
        if len(products) < 2:
            raise ValueError('Need multiple training products per category')
        rng.shuffle(products)
        held.update(products[:max(1, len(products)//5)])
    mask = torch.tensor([s.product_id in held for s in samples], dtype=torch.bool)
    if mask.all() or not mask.any():
        raise ValueError('Empty fit or held-out training partition')
    return mask


def select_ridge(features, targets, held, strengths=(.001, .01, .1, 1.)):
    if held.shape != (len(features),) or held.dtype != torch.bool or held.all() or not held.any():
        raise ValueError('Nonempty boolean training partition required')
    records = []
    for strength in strengths:
        w, b, penalty = ridge_fit(features[~held], targets[~held], strength)
        predictions = F.linear(features[held].float(), w, b)
        cosine = F.cosine_similarity(predictions, targets[held].float(), dim=-1).mean().item()
        records.append(dict(strength=float(strength), penalty=penalty, heldout_train_cosine=cosine))
    if not records:
        raise ValueError('No ridge candidates')
    selected = max(records, key=lambda r: (r['heldout_train_cosine'], r['strength']))
    w, b, penalty = ridge_fit(features, targets, selected['strength'])
    return w, b, dict(candidates=records, selected_strength=selected['strength'], final_penalty=penalty,
                     selection='held-out training-product teacher cosine; no task test selection')


def replace_projection(payload, weight, bias):
    if payload['metadata'].get('retrieval_head', 'linear64') != 'linear64':
        raise ValueError('Only the existing linear64 readout is supported')
    if weight.shape != (64, 384) or bias.shape != (64,) or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('Invalid fitted readout')
    state = dict(payload['state_dict'])
    for key, value in [('readout.projection.weight', weight), ('readout.projection.bias', bias)]:
        if state[key].shape != value.shape:
            raise ValueError('Source readout shape mismatch')
        state[key] = value.to(state[key])
    # Do not inherit old accuracy, auxiliary heads or epoch as a new result.
    return dict(metadata=dict(payload['metadata']), state_dict=state, epoch=0,
                stage='train_only_teacher_ridge', selection_variant='closed_form_readout')


def run(args):
    from transformers import AutoProcessor
    from .broad_transfer import load_pool
    from .cli import encode, evaluate, preview
    from .data import _load_contract
    from .domain_data import combine_training
    from .io import verify_assets, sha256, write_json, source_commit
    from .model import OpticalRetrieval
    from .teacher_relations import load_teacher_cache

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    verify_assets(args.assets)
    source_sha = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload['metadata'].get('fusion_alpha_min', 0) <= .4:
        raise ValueError('Requires an alpha>0.4 source checkpoint')
    if payload['metadata'].get('retrieval_head', 'linear64') != 'linear64':
        raise ValueError('Requires the original linear64 readout')
    target, _ = _load_contract(args.target)
    external, report = load_pool(args.pool, args.abo, args.target)
    if not report.get('target_types_only'):
        raise ValueError('Requires target-mapped training pool')
    samples, target_count = combine_training(target, external)
    teacher, teacher_audit = load_teacher_cache(args.teacher_cache, samples, args.target, args.pool, 'cpu')
    targets = F.normalize(teacher[:, :64], dim=-1)
    held = training_product_partition(samples)
    args.output.mkdir(parents=True)
    execution = dict(status='running', source_commit=source_commit(), command=sys.argv,
                     source_checkpoint_sha256=source_sha, teacher=teacher_audit, teacher_at_inference=False,
                     torch=torch.__version__, python=sys.version, training_images=len(samples),
                     training_products=len({s.product_id for s in samples}), target_train_images=target_count,
                     heldout_train_product_ids=sorted({s.product_id for s, h in zip(samples, held) if h}),
                     fit_scope='training only; all teacher rows, no correctness filtering',
                     only_changed_parameters=['readout.projection.weight', 'readout.projection.bias'])
    write_json(args.output/'execution.json', execution)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = OpticalRetrieval(payload['metadata']).to(device).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    processor = AutoProcessor.from_pretrained(args.assets/'processor', local_files_only=True)
    values = []
    def capture(_module, arguments):
        values.append(arguments[0].detach().float().cpu())
    hook = model.readout.projection.register_forward_pre_hook(capture)
    try:
        encode(model, processor, samples, device, args.batch_size)
    finally:
        hook.remove()
    features = torch.cat(values)
    if features.shape != (len(samples), 384):
        raise ValueError('Pre-projection training feature shape mismatch')
    cache_file = args.output/'train_readout_features.pt'
    torch.save(dict(schema=1, ids=[s.sample_id for s in samples], features=features,
                    source_checkpoint_sha256=source_sha, teacher_cache_sha256=teacher_audit['cache_sha256']), cache_file)
    weight, bias, fit_report = select_ridge(features, targets, held)
    fitted = replace_projection(payload, weight, bias)
    assert all(fitted['state_dict'][k] is v for k, v in payload['state_dict'].items()
               if k not in execution['only_changed_parameters'])
    if sha256(args.checkpoint) != source_sha:
        raise ValueError('Source checkpoint changed during fitting')
    torch.save(fitted, args.output/'best.pt')
    model.load_state_dict(fitted['state_dict'], strict=True)
    # Only now access task test pixels, after the complete fit is frozen.
    train = [s for s in target if s.split == 'train']
    test = [s for s in target if s.split == 'test']
    metrics = evaluate(model, processor, train, test, device, args.batch_size, args.output, include_train_metrics=True)
    model.set_remove_optical(True)
    removed = evaluate(model, processor, train, test, device, args.batch_size)
    model.set_remove_optical(False)
    preview(model, args.output)
    final = dict(status='complete', source_commit=source_commit(), selected_epoch=0,
                 checkpoint_sha256=sha256(args.output/'best.pt'), source_checkpoint_sha256=source_sha,
                 readout_fit=fit_report, model_audit=model.audit(), metrics=metrics,
                 remove_optical_same_weights=removed,
                 optical_removal_hit1_drop_percentage_points=100*(metrics['hit_at_1']-removed['hit_at_1']),
                 optical_and_other_nonprojection_weights_unchanged=True, new_phase_training=False,
                 teacher_at_inference=False, train_feature_cache_sha256=sha256(cache_file),
                 gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                 batch_size=args.batch_size)
    write_json(args.output/'final_report.json', final)
    execution['status'] = 'complete'
    write_json(args.output/'execution.json', execution)
    print(json.dumps(final), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'checkpoint', 'target', 'abo', 'pool', 'teacher-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
