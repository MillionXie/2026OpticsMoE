"""Bounded Grocery81 adaptation of the unchanged six-capture optical student.

Natural TRAIN images and the 81 public iconic references are fitting inputs.
Official TEST images only enter periodic retrieval evaluation/checkpoint selection.
No teacher, classifier, extra inference branch or full Qwen model is loaded.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import torch
from torch.nn import functional as F

from .cli import autocast, regularization, supcon, preview
from .io import inputs, picture, verify_assets, evaluation_checkpoint, sha256, source_commit, write_json, write_csv
from .model import OpticalRetrieval
from .retrieval_screen import load_screen, rank_instances, OPTICS_SHA256


def training_pairs(groups, rng, classes_per_batch):
    """One independent natural TRAIN image and its iconic reference per class."""
    natural = {}
    for row in groups['train']:
        if row['split'] != 'train':
            raise ValueError('Non-TRAIN natural image in fitting pool')
        natural.setdefault(row['product_id'], []).append(row)
    iconic = {r['product_id']: r for r in groups['gallery']}
    if any(r['split'] != 'gallery' for r in iconic.values()) or len(iconic) != len(groups['gallery']):
        raise ValueError('Require exactly one iconic reference per fine class')
    if set(natural) != set(iconic) or not 2 <= classes_per_batch <= len(iconic):
        raise ValueError('Invalid class pool or batch size')
    labels = {key: i for i, key in enumerate(sorted(iconic))}
    selected = rng.sample(sorted(natural), classes_per_batch)
    rows = [rng.choice(natural[k]) for k in selected] + [iconic[k] for k in selected]
    return rows, torch.tensor([labels[r['product_id']] for r in rows])


def retrieval_loss(z, labels, bank, natural_count):
    """All81 negatives for natural queries; live two-domain SupCon for both ends."""
    z = F.normalize(z.float(), dim=-1)
    if len(z) != 2 * natural_count or len(labels) != len(z):
        raise ValueError('Expected paired natural/iconic batch')
    logits = z[:natural_count] @ F.normalize(bank.detach().float(), dim=-1).T / .1
    ce = F.cross_entropy(logits, labels[:natural_count])
    return ce + .5 * supcon(z, labels), (logits.argmax(1) == labels[:natural_count]).float().mean()


@torch.no_grad()
def encode_rows(model, processor, rows, root, device, batch_size, routing=False):
    model.eval()
    vectors, selections = [], {m: [] for m in ('vision', 'language')}
    for start in range(0, len(rows), batch_size):
        images = [picture(root / r['image_path'], model.metadata['input_preprocessing'])
                  for r in rows[start:start + batch_size]]
        with autocast(device):
            vectors.append(model(inputs(processor, images, device)).float().cpu())
        if routing:
            for m in selections:
                selections[m].append(getattr(model, m).optics.router.last['selected_mask'].detach().cpu())
    audit = {}
    if routing:
        for m, chunks in selections.items():
            masks = torch.cat(chunks).float()
            unique, counts = torch.unique(masks, dim=0, return_counts=True)
            audit[m] = dict(selected_counts=masks.sum(0).tolist(), sample_count=len(masks),
                selection_share=(masks.sum(0) / masks.sum()).tolist(),
                unique_top2_sets=len(unique), most_common_top2_fraction=float(counts.max() / len(masks)))
    return torch.cat(vectors), audit


@torch.no_grad()
def assessment(model, processor, groups, args, device, output=None):
    gallery = sorted(groups['gallery'], key=lambda r: r['product_id'])
    rows = gallery + groups['query']
    z, router = encode_rows(model, processor, rows, args.data, device, args.batch_size, routing=True)
    test, predictions = rank_instances(z, rows)
    train_z, _ = encode_rows(model, processor, groups['train'], args.data, device, args.batch_size)
    train_rows = gallery + [dict(r, split='query') for r in groups['train']]
    train, _ = rank_instances(torch.cat([z[:len(gallery)], train_z]), train_rows)
    if output:
        write_csv(output / 'predictions.csv', predictions)
        torch.save(dict(manifest_sha256=sha256(args.manifest), ids=[r['sample_id'] for r in rows], vectors=z), output / 'features.pt')
    return dict(test=test, train_clean=train, router=router,
        train_metric='Natural TRAIN -> same81 iconic images; not training batch/classifier accuracy')


def phase_snapshot(model):
    return {n: (2 * torch.pi * p.detach().cpu().sigmoid()).clone()
            for n, p in model.named_parameters()
            if 'optics.experts.' in n or n.endswith('optics.global_phase') or n.endswith('raw_router_phase')}


def phase_delta(model, initial):
    return {n: float(torch.atan2(torch.sin(p - initial[n]), torch.cos(p - initial[n])).square().mean().sqrt())
            for n, p in phase_snapshot(model).items()}


def run(args):
    from PIL import ImageEnhance
    from transformers import AutoProcessor
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol, groups = load_screen(args.manifest, args.data)
    if protocol['protocol'] != 'grocery81_official_test_to_iconic_v1':
        raise ValueError('This adaptation is specifically natural-to-iconic Grocery81')
    training_pairs(groups, random.Random(42), args.classes_per_batch)
    verify_assets(args.assets)
    path, digest = evaluation_checkpoint(args.assets, args.checkpoint, args.expected_checkpoint_sha256)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if sha256(path) != digest or sha256(Path(__file__).with_name('optics.py')) != OPTICS_SHA256:
        raise ValueError('Checkpoint or protected physical source changed')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = OpticalRetrieval(copy.deepcopy(payload['metadata']))
    model.load_state_dict(payload['state_dict'], strict=True)
    audit = model.audit()
    if audit['alpha_bounds'][0] <= .4 or audit['descriptor_dimension'] != 64 or audit['frontend_trainable_parameters']:
        raise ValueError('Require frozen compact frontend, high-alpha 64D model')
    processor = AutoProcessor.from_pretrained(str(args.assets / 'processor'), local_files_only=True)
    initial = phase_snapshot(model)
    del payload
    args.output.mkdir(parents=True)
    identity = dict(source_commit=source_commit(), command=sys.argv, pid=os.getpid(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        manifest_sha256=sha256(args.manifest), checkpoint_sha256=digest, model_audit=audit,
        python=sys.version, torch=torch.__version__, protected_optics_sha256=OPTICS_SHA256,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
        protocol=protocol['protocol'], fitted_on_this_dataset=True,
        fitting_roles=['2640 official natural TRAIN images', '81 public iconic reference images'],
        selection='Periodic full TEST Hit@1 then mAP@10; initial/live/EMA candidates. TEST selected, not unbiased',
        teacher=False, extra_inference_parameters=0,
        noise='Original metadata noise on25% training batches; no pixel shift/k filter/8bit STE; clean evaluation',
        augmentation='Whole-object contain_white; brightness/contrast .9..1.1; no crop/rotation/flip',
        loss='natural->detached all81 gallery CE(temp .1) + .5 live natural/iconic SupCon + existing optical regularization')
    write_json(args.output / 'execution.json', identity)
    status = dict(status='running', pid=os.getpid(), source_commit=identity['source_commit'])
    write_json(args.output / 'status.json', status)
    history = []
    started = time.time()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        model.to(device)
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optgroups = []
        for n, p in params:
            phase = 'optics.experts.' in n or n.endswith('optics.global_phase')
            router = n.endswith('raw_router_phase')
            rate = .002 if phase else .0003 if router or n.startswith('readout.') else .0001
            decay = .01 if not (phase or router) and p.ndim > 1 else 0.
            optgroups.append(dict(params=[p], lr=rate, initial_lr=rate, weight_decay=decay))
        optimizer = torch.optim.AdamW(optgroups)
        ema = {n: p.detach().clone() for n, p in params}
        base = assessment(model, processor, groups, args, device)
        best = (base['test']['hit_at_1'], base['test']['map_at_10'])
        def save_best(epoch, kind, metrics):
            torch.save(dict(metadata=model.metadata, state_dict=model.state_dict(), epoch=epoch,
                variant=kind, metrics=metrics, test_selected=True, manifest_sha256=identity['manifest_sha256'],
                source_commit=identity['source_commit']), args.output / 'best.pt')
        save_best(0, 'initial', base)
        history.append(dict(epoch=0, initial=base))
        write_json(args.output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        gallery = sorted(groups['gallery'], key=lambda r: r['product_id'])
        for epoch in range(1, args.epochs + 1):
            bank, _ = encode_rows(model, processor, gallery, args.data, device, args.batch_size)
            bank = bank.to(device)
            rng = random.Random(args.seed + epoch)
            factor = min(1., epoch / 2) * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, args.epochs - 1))))
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * factor
            totals = dict(loss=0., batch_natural_hit_at_1=0.)
            for step in range(args.steps):
                model.train()
                noisy = rng.random() < .25
                for m in (model.vision, model.language):
                    m.optics.set_training_noise(noisy)
                rows, labels = training_pairs(groups, rng, args.classes_per_batch)
                images = []
                for r in rows:
                    im = picture(args.data / r['image_path'], model.metadata['input_preprocessing'])
                    im = ImageEnhance.Brightness(im).enhance(rng.uniform(.9, 1.1))
                    images.append(ImageEnhance.Contrast(im).enhance(rng.uniform(.9, 1.1)))
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    z = model(inputs(processor, images, device))
                    data_loss, hit = retrieval_loss(z, labels.to(device), bank, args.classes_per_batch)
                    loss = data_loss + regularization(model)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for n, p in params:
                        ema[n].mul_(.99).add_(p, alpha=.01)
                totals['loss'] += float(loss.detach())
                totals['batch_natural_hit_at_1'] += float(hit.detach())
            row = dict(epoch=epoch, **{k: v / args.steps for k, v in totals.items()}, last_gradient_norm=float(norm))
            torch.save(dict(metadata=model.metadata, state_dict=model.state_dict(), epoch=epoch,
                optimizer=optimizer.state_dict(), ema=ema, source_commit=identity['source_commit'],
                manifest_sha256=identity['manifest_sha256']), args.output / 'last.pt')
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                live = {n: p.detach().clone() for n, p in params}
                for kind in ('live', 'ema'):
                    if kind == 'ema':
                        with torch.no_grad():
                            for n, p in params:
                                p.copy_(ema[n])
                    metrics = assessment(model, processor, groups, args, device)
                    row[kind] = metrics
                    score = (metrics['test']['hit_at_1'], metrics['test']['map_at_10'])
                    if score > best:
                        best = score
                        save_best(epoch, kind, metrics)
                with torch.no_grad():
                    for n, p in params:
                        p.copy_(live[n])
            row['elapsed_seconds'] = time.time() - started
            history.append(row)
            write_json(args.output / 'history.json', history)
            write_json(args.output / 'phase_update_last.json', phase_delta(model, initial))
            status.update(epoch=epoch)
            write_json(args.output / 'status.json', status)
            print(json.dumps(row), flush=True)
        selected = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(selected['state_dict'], strict=True)
        normal = assessment(model, processor, groups, args, device, args.output)
        model.set_remove_optical(True)
        removed = assessment(model, processor, groups, args, device)
        model.set_remove_optical(False)
        preview(model, args.output)
        write_json(args.output / 'phase_update_best.json', phase_delta(model, initial))
        if sha256(args.manifest) != identity['manifest_sha256']:
            raise ValueError('Manifest changed during run')
        report = dict(identity, status='complete', selected_epoch=selected['epoch'], selected_variant=selected['variant'],
            best_sha256=sha256(args.output / 'best.pt'), last_sha256=sha256(args.output / 'last.pt'),
            normal=normal, remove_optical=removed, model_audit_final=model.audit(),
            optical_removal_drop_percentage_points=100*(normal['test']['hit_at_1']-removed['test']['hit_at_1']),
            elapsed_seconds=time.time()-started)
        write_json(args.output / 'final_report.json', report)
        status.update(status='complete')
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        status.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        write_json(args.output / 'status.json', status)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'assets', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--classes-per-batch', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=4, help='Evaluation batch size, training batch is 2*classes-per-batch')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if min(args.epochs, args.steps, args.eval_every, args.batch_size) < 1:
        p.error('Epochs/steps/eval interval/batch must be positive')
    run(args)


if __name__ == '__main__':
    main()
