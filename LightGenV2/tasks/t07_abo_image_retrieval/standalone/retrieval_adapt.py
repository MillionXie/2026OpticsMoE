"""Bounded locked-protocol adaptation of the unchanged six-capture student.

Grocery: natural TRAIN images + public iconic references; shared fine classes.
COIL: only the60 TRAIN objects, separate query/reference views; heldout40 untouched.
TEST images only enter periodic retrieval evaluation/checkpoint selection.
Optional TRAIN-only cached teacher relations; no classifier, extra inference branch or full Qwen model is loaded.
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
from .retrieval_screen import load_screen, rank_instances, OPTICS_SHA256, GALLERY_ANGLES
from .retrieval_refine import (PROFILES, validate_continuation, route_objective,
    all_view_loss, load_train_teacher, relational_loss, selection_score, router_acceptable)
from .enrolled_regularization import load_external_pool, augment_whole_object, curriculum_epoch
from .generalization import backward_with_sam


def fitting_groups(protocol, groups, multi_view=False):
    if protocol['protocol'] in ('shape_hash8_categories_official_train_gallery_v1', 'abo200_enrolled_sku_hash8train4query_v1'):
        by_id = {}
        for r in groups['train']:
            by_id.setdefault(r['product_id'], []).append(r)
        natural, references = [], []
        for key in sorted(by_id):
            images = sorted(by_id[key], key=lambda r: r['sample_id'])
            if len(images) < 2:
                continue  # Training-pair eligibility only; TEST/full gallery unchanged.
            references.extend(dict(r, split='gallery', source_split='train') for r in (images if multi_view else images[:1]))
            natural.extend(images if multi_view else images[1:])
        if len(references) < 2:
            raise ValueError('Insufficient SHAPE multi-view TRAIN identities')
        return dict(train=natural, gallery=references,
            exclude_self=multi_view,
            note=('TRAIN-only all views as reference and query; different photos paired and identical query excluded from bank/clean TRAIN ranking. ' if multi_view else 'TRAIN-only first lexical image/SKU as reference, remaining as queries; ')
            + 'Singleton TRAIN SKUs not sampled. TEST retains ALL selected training images as gallery and ALL selected test queries; enrolled SKU, not unseen SKU')
    if protocol['protocol'] == 'grocery81_official_test_to_iconic_v1':
        return dict(train=groups['train'], gallery=groups['gallery'],
            note='Natural TRAIN -> same81 public iconic images; shared fine classes, NOT unseen SKU')
    if protocol['protocol'] != 'heldout40_objects_four_gallery_views_eight_queries_v1':
        raise ValueError('Unknown adaptation protocol')
    train_ids = {r['product_id'] for r in groups['train']}
    test_ids = {r['product_id'] for r in groups['gallery'] + groups['query']}
    if train_ids & test_ids or any(r['split'] != 'train' for r in groups['train']):
        raise ValueError('COIL heldout identity leaked into fitting')
    # Fitting references come ONLY from the original TRAIN pool. Do not use the
    # heldout test gallery even though it is public at retrieval evaluation.
    references = [dict(r, split='gallery', source_split='train') for r in groups['train']
                  if r['angle'] in GALLERY_ANGLES]
    natural = [r for r in groups['train'] if r['angle'] not in GALLERY_ANGLES]
    if len(train_ids) != 60 or len(references) != 240 or len(natural) != 4080:
        raise ValueError('COIL fitting must contain60 TRAIN objects,4 refs+68 queries each')
    return dict(train=natural, gallery=references,
        note='60 TRAIN identities:4080 nonreference views ->240 reference views; no self retrieval. TEST:40 other identities,320->160')


def training_pairs(groups, rng, classes_per_batch, category_probability=0.):
    """One TRAIN query and one reference per identity/relevance class."""
    natural = {}
    for row in groups['train']:
        if row['split'] != 'train':
            raise ValueError('Non-TRAIN natural image in fitting pool')
        natural.setdefault(row['product_id'], []).append(row)
    iconic = {}
    for r in groups['gallery']:
        if r['split'] != 'gallery':
            raise ValueError('Non-reference row in reference pool')
        iconic.setdefault(r['product_id'], []).append(r)
    if len({r['sample_id'] for r in groups['gallery']}) != len(groups['gallery']):
        raise ValueError('Duplicate reference image')
    if set(natural) != set(iconic) or not 2 <= classes_per_batch <= len(iconic):
        raise ValueError('Invalid class pool or batch size')
    labels = {key: i for i, key in enumerate(sorted(iconic))}
    pool = sorted(natural)
    if category_probability > 0 and rng.random() < category_probability:
        categories = {}
        for key in pool:
            category = natural[key][0].get('category_id')
            if category is not None:
                categories.setdefault(str(category), []).append(key)
        eligible = sorted(k for k, v in categories.items() if len(v) >= classes_per_batch)
        if eligible:
            pool = categories[rng.choice(eligible)]
    selected = rng.sample(pool, classes_per_batch)
    queries = [rng.choice(natural[k]) for k in selected]
    references = []
    for k, query in zip(selected, queries):
        candidates = [r for r in iconic[k] if r['sample_id'] != query['sample_id']]
        if not candidates:
            raise ValueError('Training pair requires independently identified photos')
        references.append(candidates[0] if len(candidates)==1 else rng.choice(candidates))
    rows = queries + references
    return rows, torch.tensor([labels[r['product_id']] for r in rows])


def retrieval_loss(z, labels, bank, natural_count, bank_labels=None, excluded=None):
    """Whole fitting-gallery NLL; multiple references of an object are positives."""
    z = F.normalize(z.float(), dim=-1)
    if len(z) != 2 * natural_count or len(labels) != len(z):
        raise ValueError('Expected paired natural/iconic batch')
    logits = z[:natural_count] @ F.normalize(bank.detach().float(), dim=-1).T / .1
    if bank_labels is None:
        bank_labels = torch.arange(len(bank), device=z.device)
    bank_labels = bank_labels.to(z.device)
    if len(bank_labels) != len(bank):
        raise ValueError('Bank identity length mismatch')
    if excluded is not None:
        if excluded.shape != logits.shape or excluded.dtype != torch.bool:
            raise ValueError('Invalid self exclusion mask')
        positive = labels[:natural_count, None].eq(bank_labels[None]) & ~excluded
        if not positive.any(1).all():
            raise ValueError('No nonself fitting positive')
        logits = logits.masked_fill(excluded, -torch.inf)
    if torch.equal(bank_labels, torch.arange(len(bank), device=z.device)):
        ce = F.cross_entropy(logits, labels[:natural_count])
    else:
        positive = labels[:natural_count, None].eq(bank_labels[None])
        if excluded is not None:
            positive = positive & ~excluded
        if not positive.any(1).all():
            raise ValueError('Query without fitting-gallery positive')
        ce = (logits.logsumexp(1) - logits.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()
    return ce + .5 * supcon(z, labels), (bank_labels[logits.argmax(1)] == labels[:natural_count]).float().mean()


@torch.no_grad()
def encode_rows(model, processor, rows, root, device, batch_size, routing=False):
    model.eval()
    # remove_optical skips the router entirely; its .last belongs to a previous
    # normal batch and must never be reported as a fresh routing measurement.
    routing_requested = routing
    routing = routing and not (model.vision.remove_optical or model.language.remove_optical)
    vectors, selections = [], {m: [] for m in ('vision', 'language')}
    for start in range(0, len(rows), batch_size):
        images = [picture(root / r['image_path'], model.metadata['input_preprocessing'])
                  for r in rows[start:start + batch_size]]
        with autocast(device):
            vectors.append(model(inputs(processor, images, device)).float().cpu())
        if routing:
            for m in selections:
                selections[m].append(getattr(model, m).optics.router.last['selected_mask'].detach().cpu())
    audit = {} if not routing_requested or routing else {'executed': False, 'reason': 'optical path removed; no fresh router observations'}
    if routing:
        for m, chunks in selections.items():
            masks = torch.cat(chunks).float()
            unique, counts = torch.unique(masks, dim=0, return_counts=True)
            audit[m] = dict(selected_counts=masks.sum(0).tolist(), sample_count=len(masks),
                selection_share=(masks.sum(0) / masks.sum()).tolist(),
                unique_top2_sets=len(unique), most_common_top2_fraction=float(counts.max() / len(masks)))
    return torch.cat(vectors), audit


@torch.no_grad()
def assessment(model, processor, groups, args, device, output=None, fit=None):
    # Same manifest order as frozen Qwen and retrieval_screen, including ties.
    # Training-bank sorting is separate and must not reorder the TEST gallery.
    gallery = groups['gallery']
    rows = gallery + groups['query']
    z, router = encode_rows(model, processor, rows, args.data, device, args.batch_size, routing=True)
    test, predictions = rank_instances(z, rows)
    fit = fit or dict(train=groups['train'], gallery=gallery, note='Natural TRAIN -> same81 iconic images')
    train_rows = sorted(fit['gallery'], key=lambda r: r['product_id']) + [dict(r, split='query') for r in fit['train']]
    train_z, _ = encode_rows(model, processor, train_rows, args.data, device, args.batch_size)
    train, _ = rank_instances(train_z, train_rows, exclude_self=fit.get('exclude_self', False))
    if output:
        write_csv(output / 'predictions.csv', predictions)
        torch.save(dict(manifest_sha256=sha256(args.manifest), ids=[r['sample_id'] for r in rows], vectors=z), output / 'features.pt')
    return dict(test=test, train_clean=train, router=router,
        train_metric=fit['note'] + '; not training batch/classifier accuracy')


def phase_snapshot(model):
    return {n: (2 * torch.pi * p.detach().cpu().sigmoid()).clone()
            for n, p in model.named_parameters()
            if 'optics.experts.' in n or n.endswith('optics.global_phase') or n.endswith('raw_router_phase')}


def phase_delta(model, initial):
    return {n: float(torch.atan2(torch.sin(p - initial[n]), torch.cos(p - initial[n])).square().mean().sqrt())
            for n, p in phase_snapshot(model).items()}


def load_initial_weights(model, payload, assets, fresh):
    """Fresh protocol must not inherit ANY ABO-trained parameter/buffer."""
    if not fresh:
        model.load_state_dict(payload['state_dict'], strict=True)
        return 'Warm start from pinned checkpoint'
    packaged = torch.load(assets / 'best.pt', map_location='cpu', weights_only=True)
    front = {n.removeprefix('frontend.'): t for n, t in packaged['state_dict'].items() if n.startswith('frontend.')}
    model.frontend.load_state_dict(front, strict=True)
    with torch.no_grad():
        for m in (model.vision, model.language):
            low, high = m.alpha_bounds
            if not low < .45 < high:
                raise ValueError('Fresh profile requires alpha=.45 inside bounds')
            raw = math.log((.45-low)/(high-.45))
            m.block1_optical_fusion_logit.fill_(raw)
            m.block2_optical_fusion_logit.fill_(raw)
    # Everything outside frontend remains constructor initialization (raw phase0 => pi).
    return 'All trainable parameters freshly initialized (alpha=.45, raw phases0 => pi); only verified packaged pretrained Qwen frontend tensors loaded; old ABO trained state ignored'


def run(args):
    from PIL import ImageEnhance
    from transformers import AutoProcessor
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol, groups = load_screen(args.manifest, args.data)
    if args.multi_view and protocol['protocol'] not in ('shape_hash8_categories_official_train_gallery_v1', 'abo200_enrolled_sku_hash8train4query_v1'):
        raise ValueError('Multi-view fitting is restricted to explicitly enrolled-SKU protocols')
    fit = fitting_groups(protocol, groups, args.multi_view)
    training_pairs(fit, random.Random(42), args.classes_per_batch)
    verify_assets(args.assets)
    path, digest = evaluation_checkpoint(args.assets, args.checkpoint, args.expected_checkpoint_sha256)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    validate_continuation(protocol['protocol'], payload, sha256(args.manifest), args.fresh_trainable)
    profile = dict(PROFILES[args.refine_profile])
    if args.router_warmup_epochs is not None:
        if args.refine_profile == 'standard':
            raise ValueError('Router warmup override requires a refinement profile')
        profile['warmup'] = args.router_warmup_epochs
    refined = args.refine_profile != 'standard'
    regularized = args.refine_profile == 'sku_regularized'
    external_fit, external_audit = None, None
    pretrain_epochs = getattr(args, 'external_pretrain_epochs', 0)
    if pretrain_epochs:
        if not regularized or not args.multi_view or protocol['protocol'] != 'abo200_enrolled_sku_hash8train4query_v1':
            raise ValueError('External curriculum requires enrolled ABO multi-view sku_regularized profile')
        external_fit, external_audit = load_external_pool(args.external_pool, args.external_root,
            protocol, groups, args.expected_external_sha256)
    elif getattr(args, 'external_pool', None) is not None:
        raise ValueError('External pool supplied but no external pretraining epochs')
    teacher = None
    if profile['teacher_weight']:
        if args.teacher_features is None:
            raise ValueError('Distillation requires an explicitly pinned frozen teacher cache')
        teacher = load_train_teacher(args.teacher_features, args.expected_teacher_sha256, sha256(args.manifest), groups, fit)
    elif args.teacher_features is not None:
        raise ValueError('Teacher cache supplied to a no-teacher profile')
    if sha256(path) != digest or sha256(Path(__file__).with_name('optics.py')) != OPTICS_SHA256:
        raise ValueError('Checkpoint or protected physical source changed')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = OpticalRetrieval(copy.deepcopy(payload['metadata']))
    initialization = load_initial_weights(model, payload, args.assets, args.fresh_trainable)
    if regularized:
        model.metadata['phase_dropout'] = dict(expert_global_probability=profile['phase_dropout'], router_probability=0., block_size=8)
        for m in (model.vision, model.language):
            m.optics.configure_phase_dropout(model.metadata['phase_dropout'])
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
        initialization=initialization,
        fitting_roles=fit['note'],
        selection=('Router eligibility first (min share5%,max pair80%,at least3 pairs per modality), then TEST Hit@1/mAP@10; ' if refined else '')+'Periodic full TEST; initial/live/EMA candidates. TEST selected, not unbiased',
        teacher=bool(teacher), teacher_sha256=args.expected_teacher_sha256,
        teacher_train_ids=sorted(teacher) if teacher else [], extra_inference_parameters=0,
        refinement=profile,
        external_curriculum=external_audit,
        curriculum_note='Continue verified current-protocol best -> external instance pretraining (last state, no TEST selection) -> target fine-tune. Initial best retained; not from-scratch pretraining' if external_fit else None,
        noise='Original metadata noise on25% joint-training batches; refined profiles keep router noise disabled. Warmup clean. No pixel shift/k filter/8bit STE; clean evaluation',
        augmentation='Whole-object .85..1 scale into white canvas, bounded placement, brightness/contrast .85..1.15,15% mild blur; no crop/flip' if regularized else 'Whole-object contain_white; brightness/contrast .9..1.1; no crop/rotation/flip',
        loss='query->detached entire FITTING gallery multi-positive NLL(temp .1) + .5 live query/reference SupCon + existing optical regularization')
    write_json(args.output / 'fitting_manifest.json', dict(parent_manifest_sha256=identity['manifest_sha256'],
        note=fit['note'], train=fit['train'], references=fit['gallery']))
    if external_fit:
        write_json(args.output / 'external_fitting_manifest.json', dict(audit=external_audit, train=external_fit['train']))
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
            rate = (.002 if phase else .0003 if router or n.startswith('readout.') else .0001) * args.lr_scale
            decay = profile.get('weight_decay', .01) if not (phase or router) and p.ndim > 1 else 0.
            optgroups.append(dict(params=[p], lr=rate, initial_lr=rate, weight_decay=decay, name=n))
        optimizer = torch.optim.AdamW(optgroups)
        ema = {n: p.detach().clone() for n, p in params}
        base = assessment(model, processor, groups, args, device, fit=fit)
        best = selection_score(base, refined)
        def save_best(epoch, kind, metrics):
            torch.save(dict(metadata=model.metadata, state_dict=model.state_dict(), epoch=epoch,
                target_epoch=max(0, epoch-pretrain_epochs),
                variant=kind, metrics=metrics, test_selected=True, manifest_sha256=identity['manifest_sha256'],
                source_commit=identity['source_commit']), args.output / 'best.pt')
        save_best(0, 'initial', base)
        history.append(dict(epoch=0, initial=base))
        write_json(args.output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        for epoch in range(1, args.epochs + pretrain_epochs + 1):
            external, phase_epoch, phase_epochs = curriculum_epoch(epoch, pretrain_epochs, args.epochs)
            if pretrain_epochs and not external and phase_epoch == 1:
                optimizer.state.clear()  # Do not carry external Adam moments into target fitting.
                with torch.no_grad():
                    for n, p in params:
                        ema[n].copy_(p)
            active_fit = external_fit if external else fit
            gallery = sorted(active_fit['gallery'], key=lambda r: r['product_id'])
            label_map = {k: i for i, k in enumerate(sorted({r['product_id'] for r in gallery}))}
            bank_labels = torch.tensor([label_map[r['product_id']] for r in gallery], device=device)
            gallery_ids = [r['sample_id'] for r in gallery]
            teacher_bank = torch.stack([teacher[sid] for sid in gallery_ids]).to(device) if teacher else None
            warming = phase_epoch <= profile['warmup'] and not external
            for name, p in params:
                p.requires_grad_(not warming or name.endswith('raw_router_phase'))
            bank, _ = encode_rows(model, processor, gallery, args.data, device, args.batch_size)
            bank = bank.to(device)
            rng = random.Random(args.seed + epoch)
            factor = min(1., phase_epoch / 2) * (.1 + .9 * .5 * (1 + math.cos(math.pi * (phase_epoch - 1) / max(1, phase_epochs - 1))))
            for g in optimizer.param_groups:
                g['lr'] = (.01 if g['name'].endswith('raw_router_phase') else 0.) if warming else g['initial_lr'] * factor * (5 if refined and g['name'].endswith('raw_router_phase') else 1)
            totals = dict(loss=0., batch_natural_hit_at_1=0., route_aux=0., teacher_loss=0., all_view_loss=0., sam_loss_gap=0.)
            for step in range(args.steps):
                model.train(not warming)
                noisy = rng.random() < .25 and not warming
                for m in (model.vision, model.language):
                    m.optics.set_training_noise(noisy)
                    if refined:
                        m.optics.router.noise_enabled = False
                rows, labels = training_pairs(active_fit, rng, args.classes_per_batch, profile['category_probability'])
                images = []
                for r in rows:
                    im = picture(args.data / r['image_path'], model.metadata['input_preprocessing'])
                    if regularized:
                        images.append(augment_whole_object(im, rng))
                        continue
                    im = ImageEnhance.Brightness(im).enhance(rng.uniform(.9, 1.1))
                    images.append(ImageEnhance.Contrast(im).enhance(rng.uniform(.9, 1.1)))
                batch_inputs = inputs(processor, images, device)
                def closure():
                  with autocast(device):
                    z = model(batch_inputs)
                    excluded = (torch.tensor([[r['sample_id'] == sid for sid in gallery_ids]
                        for r in rows[:args.classes_per_batch]], device=device) if args.multi_view else None)
                    data_loss, hit = retrieval_loss(z, labels.to(device), bank, args.classes_per_batch, bank_labels, excluded)
                    route = route_objective(model) if refined else z.new_zeros(())
                    all_views = all_view_loss(z, labels.to(device), bank, bank_labels, args.classes_per_batch, excluded) if profile['positive_weight'] else z.new_zeros(())
                    kd = z.new_zeros(())
                    if teacher and not warming:
                        tq = torch.stack([teacher[r['sample_id']] for r in rows[:args.classes_per_batch]]).to(device)
                        kd = relational_loss(z, bank, tq, teacher_bank, excluded)
                    taper = max(0., 1 - phase_epoch / max(1., phase_epochs*.8))
                    loss = route if warming else data_loss + regularization(model) + (.03+.17*taper)*route + profile['positive_weight']*all_views + profile['teacher_weight']*taper*kd
                  return dict(loss=loss, hit=hit.detach(), route=route.detach(), kd=kd.detach(), all_views=all_views.detach())
                result, sam = backward_with_sam(closure, optimizer, 0. if warming else profile.get('sam_rho', 0.))
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for n, p in params:
                        ema[n].mul_(.99).add_(p, alpha=.01)
                totals['loss'] += float(result['loss'].detach())
                totals['batch_natural_hit_at_1'] += float(result['hit'])
                totals['route_aux'] += float(result['route'])
                totals['teacher_loss'] += float(result['kd'])
                totals['all_view_loss'] += float(result['all_views'])
                totals['sam_loss_gap'] += sam['loss_gap']
            row = dict(epoch=epoch, phase='external_pretrain' if external else 'target_finetune', phase_epoch=phase_epoch, router_only_warmup=warming, **{k: v / args.steps for k, v in totals.items()}, last_gradient_norm=float(norm))
            torch.save(dict(metadata=model.metadata, state_dict=model.state_dict(), epoch=epoch,
                optimizer=optimizer.state_dict(), ema=ema, source_commit=identity['source_commit'],
                manifest_sha256=identity['manifest_sha256']), args.output / 'last.pt')
            if not external and (phase_epoch % args.eval_every == 0 or phase_epoch == args.epochs):
                live = {n: p.detach().clone() for n, p in params}
                for kind in ('live', 'ema'):
                    if kind == 'ema':
                        with torch.no_grad():
                            for n, p in params:
                                p.copy_(ema[n])
                    metrics = assessment(model, processor, groups, args, device, fit=fit)
                    row[kind] = metrics
                    score = selection_score(metrics, refined)
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
            status.update(epoch=epoch, phase=row['phase'], phase_epoch=phase_epoch)
            write_json(args.output / 'status.json', status)
            print(json.dumps(row), flush=True)
        for _, p in params:
            p.requires_grad_(True)
        selected = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(selected['state_dict'], strict=True)
        normal = assessment(model, processor, groups, args, device, args.output, fit=fit)
        model.set_remove_optical(True)
        removed = assessment(model, processor, groups, args, device, fit=fit)
        model.set_remove_optical(False)
        preview(model, args.output)
        write_json(args.output / 'phase_update_best.json', phase_delta(model, initial))
        if sha256(args.manifest) != identity['manifest_sha256']:
            raise ValueError('Manifest changed during run')
        report = dict(identity, status='complete', selected_epoch=selected['epoch'], selected_variant=selected['variant'],
            selected_target_epoch=selected.get('target_epoch', selected['epoch']),
            best_sha256=sha256(args.output / 'best.pt'), last_sha256=sha256(args.output / 'last.pt'),
            normal=normal, remove_optical=removed, model_audit_final=model.audit(),
            router_eligible=router_acceptable(normal),
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
    p.add_argument('--multi-view', action='store_true', help='All TRAIN views in bank, distinct-photo pairs, self excluded; enrolled protocols only')
    p.add_argument('--fresh-trainable', action='store_true', help='Reset all trainable weights, load packaged frozen frontend only')
    p.add_argument('--lr-scale', type=float, default=1.)
    p.add_argument('--refine-profile', choices=tuple(PROFILES), default='standard')
    p.add_argument('--teacher-features', type=Path)
    p.add_argument('--expected-teacher-sha256')
    p.add_argument('--router-warmup-epochs', type=int, help='Optional explicit curriculum override; default comes from profile')
    p.add_argument('--external-pool', type=Path)
    p.add_argument('--external-root', type=Path)
    p.add_argument('--expected-external-sha256')
    p.add_argument('--external-pretrain-epochs', type=int, default=0, help='Additional external epochs before --epochs target fine-tuning; same-SKU positives, no TEST selection in this phase')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if min(args.epochs, args.steps, args.eval_every, args.batch_size) < 1:
        p.error('Epochs/steps/eval interval/batch must be positive')
    if not 0 < args.lr_scale <= 10:
        p.error('lr-scale must be in (0,10]')
    if args.router_warmup_epochs is not None and args.router_warmup_epochs < 0:
        p.error('Warmup epochs cannot be negative')
    if args.external_pretrain_epochs < 0 or (args.external_pretrain_epochs and not all((args.external_pool, args.external_root, args.expected_external_sha256))):
        p.error('External pretraining requires nonnegative epochs and pool/root/SHA')
    run(args)


if __name__ == '__main__':
    main()
