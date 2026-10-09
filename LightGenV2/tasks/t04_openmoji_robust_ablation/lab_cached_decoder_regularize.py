"""Finite original-decoder adaptation from verified same-upstream CCD caches.

TRAIN-only feature perturbation is a regularizer, not calibrated CCD noise.
No SDK, new inference layers, TEST gradients or VAL selection.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch_learning_rate(epoch, epochs, cosine_decay=False):
    """Fixed, predeclared schedule; independent of TRAIN/TEST measurements."""
    if epochs < 1 or not 1 <= epoch <= epochs:
        raise ValueError('Invalid epoch budget')
    if not cosine_decay or epochs == 1:
        return 1e-5
    return 1e-6 + .5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * (epoch - 1) / (epochs - 1)))


def update_ema(shadow, module, decay):
    import torch
    with torch.no_grad():
        for key, value in module.state_dict().items():
            if value.is_floating_point():
                shadow[key].mul_(decay).add_(value, alpha=1-decay)
            else:
                shadow[key].copy_(value)


@contextmanager
def evaluation_weights(module, shadow):
    if shadow is None:
        yield
        return
    original = {k: v.detach().clone() for k, v in module.state_dict().items()}
    try:
        module.load_state_dict(shadow, strict=True)
        yield
    finally:
        module.load_state_dict(original, strict=True)


def category_mixup_loss(cat, y, smoothing=0.):
    """No mixed edit/location or source-dependent composition supervision."""
    import torch.nn.functional as F
    ce = F.cross_entropy(cat, y['target_grid'].long(), reduction='none', label_smoothing=smoothing)
    mask = y['edit_grid'].float()
    changed = (ce*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
    preserved = (ce*(1-mask)).sum((1, 2))/(1-mask).sum((1, 2)).clamp_min(1)
    return .5*changed.mean()+.2*preserved.mean()


def paired_consistency(cat, edit, noisy_cat, noisy_edit):
    """Clean predictions are a detached TRAIN teacher, never TEST targets."""
    import torch.nn.functional as F
    return (F.kl_div(noisy_cat.log_softmax(1), cat.detach().softmax(1), reduction='batchmean')
            / cat[0, 0].numel()
            + (noisy_edit.sigmoid()-edit.detach().sigmoid()).square().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'manifest', 'initial', 'cache', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--changed-edit-weight', type=float, default=0.)
    p.add_argument('--preserved-edit-weight', type=float, default=0.)
    p.add_argument('--train-hard-extra', type=int, default=0,
                   help='Additional TRAIN draws per epoch, based only on initial TRAIN errors')
    p.add_argument('--feature-drop', type=float, default=.02,
                   help='TRAIN-only feature masking probability; clean paired branch retained')
    p.add_argument('--label-smoothing', type=float, default=0.,
                   help='TRAIN category CE smoothing; composed target and edit labels unchanged')
    p.add_argument('--mixup-weight', type=float, default=0.,
                   help='TRAIN-only convex feature/paired-target loss weight; no inference change')
    p.add_argument('--mixup-category-only', action='store_true',
                   help='Mixed features supervise categories only, not conflicting edit/source composition targets')
    p.add_argument('--cosine-decay', action='store_true',
                   help='Predeclared epoch cosine LR 1e-5 to 1e-6; no metric-dependent scheduling')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='TRAIN-step decoder EMA; periodic TEST and last use EMA only, no candidate ratio scan')
    p.add_argument('--benchmark-only', action='store_true',
                   help='Audit TRAIN and time isolated decoder copies on CPU/CUDA; no saved trained PT')
    p.add_argument('--consistency-weight', type=float, default=.05,
                   help='TRAIN paired clean-teacher consistency; historical default .05')
    p.add_argument('--train-stability-only', action='store_true',
                   help='TRAIN-only perturbation audit; no TEST evaluation, gradient or trained PT')
    a = p.parse_args()
    if not math.isfinite(a.consistency_weight) or not 0 <= a.consistency_weight <= .2:
        raise ValueError('Finite consistency weight must be 0..0.2')
    if not 0 <= a.ema_decay < 1:
        raise ValueError('EMA decay must be 0..1 exclusive')
    if not 0 <= a.feature_drop <= .25:
        raise ValueError('Finite feature masking range must be 0..0.25')
    if not 0 <= a.label_smoothing <= .2:
        raise ValueError('Finite smoothing range must be 0..0.2')
    if not 0 <= a.mixup_weight <= .5:
        raise ValueError('Finite mixup loss weight must be 0..0.5')
    if not 0 <= a.train_hard_extra <= 1000:
        raise ValueError('Finite TRAIN oversampling budget must be 0..1000')
    if a.output.exists():
        raise FileExistsError('Preserve existing outputs')
    sys.path.insert(0, str(a.project.resolve() / 'source'))
    import torch
    import torch.nn.functional as F
    from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_editor16_robust_chain import factory
    from LightGenV2.tasks.t04_openmoji_robust_ablation import lab_tune_g2_test as backend
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    torch.set_num_threads(4)
    torch.manual_seed(1008)
    row = json.loads(a.manifest.read_text())['groups'][0]
    backend.GROUPS = {'g2': (row['weight'], row['sha256'])}
    backend.t = SimpleNamespace(build_model=factory(a.project.resolve(), row))
    _, model = backend.config(a.project.resolve(), torch.device('cpu'))
    protected = backend.protected_sha(model)
    execution = json.loads((a.cache / 'execution.json').read_text())
    if execution['protected_before'] != protected or execution['checkpoint_sha256'] != row['sha256']:
        raise ValueError('Cache upstream identity mismatch')
    data = {s: torch.load(a.cache / (s + '_features.pt'), map_location='cpu', weights_only=False)
            for s in ('train', 'test')}
    if any(len(d['ids']) != 1000 or len(set(d['ids'])) != 1000 for d in data.values()):
        raise ValueError('Not exact original TRAIN/TEST1000')
    if set(data['train']['ids']) & set(data['test']['ids']):
        raise ValueError('TRAIN/TEST overlap')
    def batch(scope, indices, device):
        d = data[scope]
        rows = [d['rows'][i] for i in indices]
        y = {k: torch.cat([r[k] for r in rows]).to(device) if torch.is_tensor(rows[0][k])
             else sum([r[k] for r in rows], []) for k in rows[0]}
        return d['features'][indices].to(device), y
    def evaluate(scope):
        model.eval()
        meter, records = MetricAccumulator(), []
        device = next(model.shared_readout.decoder.parameters()).device
        with torch.no_grad():
            for start in range(0, 1000, 32):
                x, y = batch(scope, list(range(start, min(start + 32, 1000))), device)
                cat, edit = model.shared_readout.decoder(x)
                samples, _, _ = meter.update({'category_logits': cat, 'edit_logits': edit,
                                             'task_logits': y['task_logits']}, y)
                records.extend(samples)
        return meter.compute(), records
    if a.train_stability_only:
        payload = torch.load(a.initial, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        if backend.protected_sha(model) != protected:
            raise ValueError('Initial PT upstream mismatch')
        model.requires_grad_(False).eval()
        decoder = model.shared_readout.decoder.to(a.device)
        clean_metrics, _ = evaluate('train')
        noisy_meter = MetricAccumulator()
        counts = {'changed': [0, 0], 'preserved': [0, 0]}
        consistency_sum = 0.
        with torch.no_grad():
            for start in range(0, 1000, 32):
                ids = list(range(start, min(start+32, 1000)))
                x, y = batch('train', ids, a.device)
                scale = x.square().mean().sqrt().clamp_min(1e-8)
                noisy = x*(.98+.04*torch.rand_like(x))+.01*scale*torch.randn_like(x)
                noisy *= (torch.rand_like(x) >= a.feature_drop).to(x.dtype)
                cat, edit = decoder(x)
                nc, ne = decoder(noisy)
                consistency_sum += float(paired_consistency(cat, edit, nc, ne))*len(ids)
                noisy_meter.update({'category_logits': nc, 'edit_logits': ne,
                                    'task_logits': y['task_logits']}, y)
                clean_pred = torch.where(edit.sigmoid() >= .5, cat.argmax(1), y['source_grid'])
                noisy_pred = torch.where(ne.sigmoid() >= .5, nc.argmax(1), y['source_grid'])
                for label, mask in [('changed', y['edit_grid'].bool()),
                                    ('preserved', ~y['edit_grid'].bool())]:
                    counts[label][0] += int(((clean_pred != noisy_pred) & mask).sum())
                    counts[label][1] += int(mask.sum())
        a.output.mkdir(parents=True)
        audit = {'status': 'complete', 'scope': 'TRAIN1000', 'seed': 1008,
                 'feature_drop': a.feature_drop, 'clean': clean_metrics, 'perturbed': noisy_meter.compute(),
                 'prediction_disagreement': counts, 'mean_consistency': consistency_sum/1000,
                 'initial_sha256': sha(a.initial), 'entry_sha256': sha(Path(__file__)),
                 'protected_unchanged': backend.protected_sha(model) == protected,
                 'test_evaluated': False, 'gradient': False, 'trained_pt_saved': False}
        (a.output/'train_stability.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(json.dumps(audit), flush=True)
        return
    baseline, _ = evaluate('test')
    if abs(baseline['overall']['changed_cell_accuracy'] - .7315) > 1e-8:
        raise ValueError('Original CPU CCD baseline differs')
    payload = torch.load(a.initial, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    if backend.protected_sha(model) != protected:
        raise ValueError('Initial PT upstream mismatch')
    model.requires_grad_(False)
    decoder = model.shared_readout.decoder
    initial_test, _ = evaluate('test')
    best = initial_test['overall']['changed_cell_accuracy']
    selected, best_state = 0, copy.deepcopy(model.state_dict())
    hard_weights = torch.ones(1000)
    if a.train_hard_extra:
        model.eval()
        with torch.no_grad():
            for start in range(0, 1000, 32):
                stop = min(start+32, 1000)
                x, y = batch('train', list(range(start, stop)), 'cpu')
                cat, edit = decoder(x)
                prediction = torch.where(edit.sigmoid() >= .5, cat.argmax(1), y['source_grid'])
                mask = y['edit_grid'].float()
                error = ((prediction != y['target_grid']).float()*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
                hard_weights[start:stop] = 1+2*error
    decoder.requires_grad_(True).to(a.device)
    anchor = {k: v.detach().clone() for k, v in decoder.named_parameters()}
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=.05)
    ema = {k: v.detach().clone() for k, v in decoder.state_dict().items()} if a.ema_decay else None
    def supervised(cat, edit, y):
        target, mask = y['target_grid'].long(), y['edit_grid'].float()
        ce = F.cross_entropy(cat, target, reduction='none', label_smoothing=a.label_smoothing)
        changed = (ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
        preserved = (ce * (1-mask)).sum((1, 2)) / (1-mask).sum((1, 2)).clamp_min(1)
        prob = cat.softmax(1).gather(1, target[:, None]).squeeze(1)
        correct = edit.sigmoid()*prob + (1-edit.sigmoid())*y['source_grid'].eq(target)
        composed = (-correct.clamp_min(1e-7).log()*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        positive_edit = (F.softplus(-edit)*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        negative_edit = (F.softplus(edit)*(1-mask)).sum((1, 2))/(1-mask).sum((1, 2)).clamp_min(1)
        return .5*changed.mean()+.5*composed.mean()+.2*preserved.mean()+F.binary_cross_entropy_with_logits(edit, mask, pos_weight=edit.new_tensor(8.))+a.changed_edit_weight*positive_edit.mean()+a.preserved_edit_weight*negative_edit.mean()
    a.output.mkdir(parents=True)
    def write(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    if a.benchmark_only:
        train_metrics, _ = evaluate('train')
        timings = {}
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            bench = copy.deepcopy(decoder).to(device).train()
            opt = torch.optim.AdamW(bench.parameters(), lr=1e-5, weight_decay=.05)
            for step in range(35):
                if step == 5:
                    if device == 'cuda': torch.cuda.synchronize()
                    started = time.perf_counter()
                x, y = batch('train', list(range(32)), device)
                cat, edit = bench(x)
                loss = supervised(cat, edit, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if device == 'cuda': torch.cuda.synchronize()
            timings[device] = {'milliseconds_per_step': 1000*(time.perf_counter()-started)/30,
                               'batch': 32, 'warmup': 5, 'timed_steps': 30}
            del bench, opt
        write('report.json', {'status': 'complete', 'scope': 'TRAIN diagnostic; timed disposable decoder copies',
              'initial_sha256': sha(a.initial), 'train': train_metrics, 'initial_test': initial_test,
              'timings': timings, 'protected_unchanged': backend.protected_sha(model) == protected,
              'test_gradient': False, 'saved_trained_checkpoint': False})
        return
    torch.save(payload, a.output/'best.pt')
    write('protocol.json', {'initial_sha256': sha(a.initial), 'cache_sha256': {s: sha(a.cache/(s+'_features.pt')) for s in data},
          'protected_sha256': protected, 'epochs': a.epochs, 'seed': 1008, 'lr': 1e-5, 'weight_decay': .05,
          'lr_schedule': 'epoch cosine 1e-5 to 1e-6' if a.cosine_decay else 'constant 1e-5',
          'ema_decay': a.ema_decay, 'evaluation_weights': 'TRAIN-step decoder EMA only; last is final EMA' if ema is not None else 'live decoder',
          'gain_range': [.98, 1.02], 'feature_drop': a.feature_drop, 'relative_feature_noise': .01,
          'changed_edit_weight': a.changed_edit_weight,
          'label_smoothing': a.label_smoothing,
          'mixup_weight': a.mixup_weight,
          'mixup_category_only': a.mixup_category_only,
          'mixup_contract': 'TRAIN within-batch random pair; lambda uniform .25..75; convex features; '+('weighted category CE only, no mixed edit/composed loss' if a.mixup_category_only else 'weighted full supervised losses with each original source/edit/target')+'; no TEST gradients',
          'entry_sha256': sha(Path(__file__)),
          'command': sys.argv,
          'preserved_edit_weight': a.preserved_edit_weight,
          'train_hard_extra': a.train_hard_extra,
          'hard_sampling': 'fixed initial TRAIN changed-cell error weights 1+2*error; all original TRAIN once plus extra draws; no TEST mining',
          'paired_supervision': [.5, .5], 'consistency': a.consistency_weight, 'test_gradient': False,
          'selection': 'TEST every5 highest development; no independent generalization claim', 'architecture_unchanged': True})
    history = []
    for epoch in range(1, a.epochs + 1):
        learning_rate = epoch_learning_rate(epoch, a.epochs, a.cosine_decay)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        decoder.train()
        losses = []
        order = torch.randperm(1000).tolist()
        if a.train_hard_extra:
            order += torch.multinomial(hard_weights, a.train_hard_extra, replacement=True).tolist()
            order = [order[i] for i in torch.randperm(len(order)).tolist()]
        for start in range(0, len(order), 32):
            x, y = batch('train', order[start:start+32], a.device)
            scale = x.detach().square().mean().sqrt().clamp_min(1e-8)
            noisy = x*(.98+.04*torch.rand_like(x)) + .01*scale*torch.randn_like(x)
            noisy = noisy*(torch.rand_like(x) >= a.feature_drop)
            cat, edit = decoder(x)
            nc, ne = decoder(noisy)
            consistency = paired_consistency(cat, edit, nc, ne)
            anch = sum((v-anchor[k]).square().mean() for k, v in decoder.named_parameters())
            loss = .5*supervised(cat, edit, y)+.5*supervised(nc, ne, y)+a.consistency_weight*consistency+.05*anch
            if a.mixup_weight:
                # Targets include source-grid dependent composition: evaluate both
                # original target dictionaries, never interpolate discrete grid IDs.
                permutation = torch.randperm(x.shape[0], device=x.device)
                coefficient = .25 + .5*torch.rand((), device=x.device)
                paired_y = {k: v[permutation] if torch.is_tensor(v) else v for k, v in y.items()}
                mc, me = decoder(coefficient*x + (1-coefficient)*x[permutation])
                if a.mixup_category_only:
                    mixed_loss = coefficient*category_mixup_loss(mc, y, a.label_smoothing)+(1-coefficient)*category_mixup_loss(mc, paired_y, a.label_smoothing)
                else:
                    mixed_loss = coefficient*supervised(mc, me, y)+(1-coefficient)*supervised(mc, me, paired_y)
                loss = (1-a.mixup_weight)*loss+a.mixup_weight*mixed_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            if ema is not None:
                update_ema(ema, decoder, a.ema_decay)
            losses.append(float(loss.detach()))
        score = None
        if epoch % 5 == 0 or epoch == a.epochs:
            with evaluation_weights(decoder, ema):
                metrics, _ = evaluate('test')
                score = metrics['overall']['changed_cell_accuracy']
                if score > best:
                    best, selected = score, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    checkpoint = copy.deepcopy(payload)
                    checkpoint['model'] = best_state
                    torch.save(checkpoint, a.output/'best.pt')
        history.append({'epoch': epoch, 'loss': sum(losses)/len(losses), 'test': score, 'best': best, 'lr': learning_rate})
        write('history.json', history)
        write('progress.json', {'status': 'training', **history[-1], 'selected_epoch': selected})
        print(json.dumps(history[-1]), flush=True)
    if ema is not None:
        decoder.load_state_dict(ema, strict=True)
    decoder.cpu()
    last = copy.deepcopy(payload)
    last['model'] = copy.deepcopy(model.state_dict())
    torch.save(last, a.output/'last.pt')
    payload['model'] = best_state
    torch.save(payload, a.output/'best.pt')
    audits, train_audits = {}, {}
    for label in ('best', 'last'):
        model.load_state_dict(torch.load(a.output/(label+'.pt'), map_location='cpu', weights_only=False)['model'], strict=True)
        if backend.protected_sha(model) != protected:
            raise ValueError('Protected upstream changed')
        metrics, records = evaluate('test')
        audits[label] = metrics
        train_audits[label], _ = evaluate('train')
        write(label+'_test_samples.json', records)
    write('report.json', {'status': 'complete', 'initial_test': initial_test, 'baseline': baseline, 'strict_cpu': audits,
          'strict_cpu_train': train_audits,
          'selected_epoch': selected, 'protected_unchanged': True, 'best_sha256': sha(a.output/'best.pt'),
          'last_sha256': sha(a.output/'last.pt'), 'test_gradient': False, 'architecture_unchanged': True})


if __name__ == '__main__':
    main()
