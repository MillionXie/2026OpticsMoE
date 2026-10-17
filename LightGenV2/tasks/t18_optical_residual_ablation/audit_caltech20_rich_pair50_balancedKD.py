"""Independent saved objective/restoration/EMA/plan/prediction audit,0forward."""
import argparse, hashlib, math, os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_pair50_balancedKD as m
import audit_caltech20_rich_pair50_alpha as independent

def audit(base, out, phase):
    assert base == m.OUT and (not out.exists()) and (os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    state = m.s.read(base / 'status.json')
    assert not state['active'] and (not state['pending']) and (not state['failures'])
    controls = m.admit()
    cfg = m.config()
    meta = m.s.read(base / 'metadata.json')
    assert meta['config'] == cfg and meta['sources'] == m.sources() and (meta['frozen_controls'] == controls)
    m.engine.verify_sources(meta['sources'], meta['git_commit'])
    manifest = m.base.dataset.manifest()
    checks = {}
    torch.set_num_threads(4)
    if phase == 'fit':
        for key, (depth, rho) in m.CASES.items():
            folder = base / key
            ex = m.s.read(folder.with_name(key + '.fit_exit.json'))
            assert ex['returncode'] == 0 and ex['actual_exit'] and (not Path('/proc', str(ex['pid']), 'cmdline').exists())
            metadata = m.s.read(folder / 'metadata.json')
            assert metadata['sources'] == meta['sources'] and metadata['config'] == cfg and (metadata['frozen_controls'] == controls)
            parent = m.BASE / key
            old = torch.load(parent / 'last.pt', map_location='cpu', weights_only=False)
            incumbent = torch.load(parent / 'best.pt', map_location='cpu', weights_only=False)
            last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
            best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            h = m.s.read(folder / 'history.json')
            f = m.s.read(folder / 'fit_result.json')
            assert len(h) == f['extra_epochs'] == 20 and f['actual_joint_epochs'] == 70 and (f['steps'] == last['steps'] == 13790) and (f['stop_reason'] == 'strategy_review_after20')
            assert metadata['parent_last_sha256'] == m.s.sha(parent / 'last.pt') and metadata['parent_best_sha256'] == m.s.sha(parent / 'best.pt') and (metadata['parent_history_sha256'] == m.s.sha(parent / 'history.json'))
            assert metadata['parent_raw_digest'] == m.s.digest(old['model']) and metadata['parent_ema_digest'] == m.s.digest(old['ema']) and (metadata['restored_optimizer_moment_digest'] == m.moments(old['optimizer']))
            for field, file in [('best_sha256', 'best.pt'), ('last_sha256', 'last.pt'), ('history_sha256', 'history.json')]:
                assert f[field] == m.s.sha(folder / file)
            assert best['sources'] == last['sources'] == meta['sources'] and best['config'] == last['config'] == cfg and (last['epoch'] == 70) and (last['scheduler']['last_epoch'] == 20)
            assert all((float(q['step']) == 13790 for q in last['optimizer']['state'].values())) and [q['weight_decay'] for q in last['optimizer']['param_groups']] == [0, cfg['electronic_weight_decay']]
            m.engine.seed(17)
            model = m.base.RegularizedMoE(depth, rho)
            assert m.s.digest(model.state_dict()) == old['initial_digest'] == metadata['initial_digest']
            model.load_state_dict(last['model'], strict=True)
            model.load_state_dict(best['model'], strict=True)
            assert model.contract() == best['contract']
            counts = np.asarray(manifest['supports']['train'], np.float64)
            expected_weights = counts.sum() / (20 * counts)
            expected_weights *= counts.sum() / np.dot(counts, expected_weights)
            assert abs(np.dot(counts, expected_weights) / counts.sum() - 1) < 1e-12 and np.allclose(metadata['class_weights'], expected_weights, atol=1e-07, rtol=1e-07)
            gate = m.s.read(folder / 'numeric_gate.json')
            assert gate['passed'] and gate['parent_raw_digest'] == m.s.digest(old['model']) and np.allclose(gate['class_weights'], expected_weights, atol=1e-07, rtol=1e-07) and (len(gate['checks']) == 12)
            for check in gate['checks']:
                norms = check['gradient_norms']
                assert set(norms) == set(dict(model.named_parameters())) and all((np.isfinite(q) and q > 0 for q in norms.values())) and (check['gradient_max_relative_l2'] <= 0.02) and (check['gradient_min_cosine'] >= 0.999) and (check['probability_maxabs'] <= 2e-06) and (check['capture_maxabs'] <= 2e-06)
            for i, row in enumerate(h, 1):
                assert row['epoch'] == 50 + i and row['tune_epoch'] == i and (row['adam_steps'] == 9850 + 197 * i)
                assert tuple((row[k] for k in ('order_sha256', 'affine_sha256', 'mix_sha256'))) == independent.plans(50 + i, cfg, 3141) and tuple((row[k] for k in ('flip_sha256', 'mask_sha256'))) == independent.mask_plans(50 + i, cfg, 3141)
                lr = [cfg['minimum_tune_lr'] + (start - cfg['minimum_tune_lr']) * (1 + math.cos(math.pi * (i - 1) / 20)) / 2 for start in (cfg['phase_lr'], cfg['electronic_lr'])]
                assert np.allclose(lr, row['lr'], atol=1e-12, rtol=0)
                for role in ('train', 'val'):
                    independent.history_metric(row[role], manifest['supports'][role], .7)
            candidates = [dict(tune_epoch=0, epoch=incumbent['epoch'], train=incumbent['metrics']['train'], val=incumbent['metrics']['val'], ema_digest=m.s.digest(incumbent['model']))] + h
            selected = min(candidates, key=lambda q: q['val']['balanced_nll'])
            assert best['tune_epoch'] == f['selected_tune'] == selected['tune_epoch'] and best['epoch'] == f['selected_epoch'] == selected['epoch'] and (m.s.digest(best['model']) == selected['ema_digest'])
            assert m.s.digest(last['model']) == h[-1]['raw_digest'] and m.s.digest(last['ema']) == h[-1]['ema_digest']
            assert set(last['previous_final_ema']) == set(last['model']) == set(last['ema'])
            for name, previous in last['previous_final_ema'].items():
                expected = previous.numpy().astype(np.float64) * cfg['ema_decay'] + last['model'][name].numpy().astype(np.float64) * (1 - cfg['ema_decay'])
                assert np.allclose(expected, last['ema'][name].numpy(), atol=2e-07, rtol=1e-06)
            assert f['method_trace_sha256'] == m.s.sha(folder / 'method_trace.npz') and f['final_optimizer_boundary_sha256'] == m.s.sha(folder / 'final_optimizer_boundary.pt')
            with np.load(folder / 'method_trace.npz', allow_pickle=False) as z:
                trace = {k: z[k] for k in z.files}
            assert len(trace['probabilities']) == 20 * 3141 and len(trace['offsets']) == 20 * 197 + 1 and (trace['offsets'][0] == 0) and (trace['offsets'][-1] == 20 * 3141)
            teacher_state = torch.load(m.teacher.OUT / 'selected_head.pt', map_location='cpu', weights_only=False)
            with np.load(m.teacher.OUT / 'train_features.npz', allow_pickle=False) as z:
                teacher_x = z['features']
                teacher_labels = z['labels']
                teacher_ids = z['ids']
            assert teacher_ids.tolist() == manifest['split_ids']['train']
            teacher_x = ((teacher_x - teacher_state['mean']) / teacher_state['std']).astype(np.float32).astype(np.float64)
            logits = teacher_x @ teacher_state['model']['weight'].numpy().astype(np.float64).T + teacher_state['model']['bias'].numpy().astype(np.float64)
            natural = np.exp(logits - logits.max(1, keepdims=True))
            natural /= natural.sum(1, keepdims=True)
            valid = ((logits.argmax(1) == teacher_labels) & (natural.max(1) >= cfg['teacher_correct_confidence_gate'])).astype(np.float32)
            qq = np.exp(logits / cfg['temperature'] - (logits / cfg['temperature']).max(1, keepdims=True))
            qq = (qq / qq.sum(1, keepdims=True)).astype(np.float32)
            batch_number = 0
            ce_error = 0.0
            kd_error = 0.0
            for tune, row in enumerate(h, 1):
                epoch = 50 + tune
                rg = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, 20261013]))
                order = rg.permutation(3141)
                target_hashes = []
                epoch_losses = []
                for batch, off in enumerate(range(0, 3141, 16)):
                    idx = order[off:off + 16]
                    rg = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, batch, 20261013, 4]))
                    lam = float(rg.beta(cfg['mixup_alpha'], cfg['mixup_alpha']))
                    lam = max(lam, 1 - lam)
                    perm = rg.permutation(len(idx))
                    q = qq[idx]
                    mass = valid[idx]
                    weight = lam * mass + (1 - lam) * mass[perm]
                    target = (lam * mass[:, None] * q + (1 - lam) * mass[perm, None] * q[perm]) / weight[:, None].clip(1e-12)
                    target_hashes.append(m.s.digest(np.r_[target.ravel(), weight].astype('<f4')))
                    ys = np.eye(20, dtype=np.float32)[teacher_labels[idx]] * (1 - cfg['label_smoothing']) + cfg['label_smoothing'] / 20
                    ys = lam * ys + (1 - lam) * ys[perm]
                    lo, hi = trace['offsets'][batch_number:batch_number + 2]
                    assert hi - lo == len(idx)
                    p = trace['probabilities'][lo:hi].astype(np.float64)
                    q = trace['teacher_soft'][lo:hi].astype(np.float64)
                    y = trace['labels_soft'][lo:hi].astype(np.float64)
                    w = trace['teacher_mass'][lo:hi].astype(np.float64)
                    assert np.allclose(q, target, atol=2e-07, rtol=0) and np.allclose(y, ys, atol=2e-07, rtol=0) and np.allclose(w, weight, atol=2e-07, rtol=0) and np.allclose(p.sum(1), 1, atol=2e-06, rtol=0)
                    ce = -(y * expected_weights * np.log(p.clip(1e-12))).sum(1).mean() - cfg['capture_weight'] * np.log(trace['capture'][lo:hi].astype(np.float64).clip(1e-12)).mean() + cfg['phase_smooth_weight'] * trace['phaseTV'][batch_number]
                    sp = np.log(p.clip(1e-12)) / cfg['temperature']
                    logs = sp - sp.max(1, keepdims=True) - np.log(np.exp(sp - sp.max(1, keepdims=True)).sum(1, keepdims=True))
                    kd = cfg['temperature'] ** 2 * (w * (q * (np.log(q.clip(1e-12)) - logs)).sum(1)).mean()
                    ec = abs(ce - trace['loss'][batch_number])
                    ek = abs(kd - trace['KD_loss'][batch_number])
                    ce_error = max(ce_error, ec)
                    kd_error = max(kd_error, ek)
                    assert max(ec, ek) < 0.0002, (ec, ek)
                    scales = np.minimum(cfg['KD_scale_cap'], cfg['KD_gradient_fraction'] * trace['CE_norm'][batch_number] / (trace['KD_norm'][batch_number] + 1e-12))
                    assert np.allclose(scales, trace['group_scales'][batch_number], atol=1e-12, rtol=0)
                    assert np.isfinite(trace['gradient_cosine'][batch_number]).all()
                    epoch_losses.append(trace['loss'][batch_number] + np.mean(scales) * trace['KD_loss'][batch_number])
                    batch_number += 1
                assert row['teacher_sha256'] == hashlib.sha256(''.join(target_hashes).encode()).hexdigest() and abs(np.mean(epoch_losses) - row['loss']) < 1e-09
            boundary = torch.load(folder / 'final_optimizer_boundary.pt', map_location='cpu', weights_only=False)
            names = [list(q) for q in ([n for n in boundary['raw_before'] if n == 'raw_router' or n.startswith('phase_sets.')], [n for n in boundary['raw_before'] if n != 'raw_router' and (not n.startswith('phase_sets.'))])]
            combined = {}
            expected_scales = []
            for group, keys in enumerate(names):
                gc = {n: boundary['CE_gradients'][n].numpy().astype(np.float64) for n in keys}
                gk = {n: boundary['KD_gradients'][n].numpy().astype(np.float64) for n in keys}
                cn = np.sqrt(sum((np.square(v).sum() for v in gc.values())))
                kn = np.sqrt(sum((np.square(v).sum() for v in gk.values())))
                eta = min(cfg['KD_scale_cap'], cfg['KD_gradient_fraction'] * cn / (kn + 1e-12))
                expected_scales.append(eta)
                assert np.allclose([cn, kn], [boundary['group_info']['CE_norm'][group], boundary['group_info']['KD_norm'][group]], rtol=1e-10, atol=1e-12)
                for n in keys:
                    combined[n] = gc[n] + eta * gk[n]
            assert np.allclose(expected_scales, boundary['group_info']['scales'], atol=1e-12, rtol=0)
            norm = np.sqrt(sum((np.square(g).sum() for g in combined.values())))
            assert np.allclose(norm, boundary['clip_norm'], rtol=2e-06, atol=1e-07)
            factor = min(1.0, 1.0 / (norm + 1e-06))
            optimizer = boundary['optimizer_before']
            max_raw_error = 0.0
            max_moment_error = 0.0
            for group, keys in zip(optimizer['param_groups'], names):
                assert len(group['params']) == len(keys)
                for pid, name in zip(group['params'], keys):
                    old_state = optimizer['state'][pid]
                    next_state = last['optimizer']['state'][pid]
                    step = int(float(old_state['step'])) + 1
                    assert step == 13790 == int(float(next_state['step']))
                    gradient = combined[name] * factor
                    b1, b2 = group['betas']
                    mm = old_state['exp_avg'].numpy().astype(np.float64) * b1 + (1 - b1) * gradient
                    vv = old_state['exp_avg_sq'].numpy().astype(np.float64) * b2 + (1 - b2) * gradient ** 2
                    raw = boundary['raw_before'][name].numpy().astype(np.float64) * (1 - group['lr'] * group['weight_decay']) - group['lr'] / (1 - b1 ** step) * mm / (np.sqrt(vv) / np.sqrt(1 - b2 ** step) + group['eps'])
                    er = float(np.abs(raw - last['model'][name].numpy()).max())
                    em = max(float(np.abs(mm - next_state['exp_avg'].numpy()).max()), float(np.abs(vv - next_state['exp_avg_sq'].numpy()).max()))
                    max_raw_error = max(max_raw_error, er)
                    max_moment_error = max(max_moment_error, em)
                    assert er < 2e-06 and em < 2e-07, (name, er, em)
            checks[key + '_KD'] = dict(independent_CE_regularizer_maxabs=ce_error, independent_KL_maxabs=kd_error, batches=batch_number, teacher_valid_TRAIN_samples=int(valid.sum()), final_Adam_raw_maxabs=max_raw_error, final_Adam_moments_maxabs=max_moment_error, model_forward_replay=False, method_trace_sha256=f['method_trace_sha256'])
            assert f['fit_seconds'] <= cfg['maximum_seconds'] + max((q['epoch_seconds'] for q in h)) + 50
            for role in ('train', 'val'):
                actual, receipt = independent.verify_predictions(folder / ('selected_' + role + '.npz'), best['model'], role, manifest)
                independent.agree(actual, f['metrics'][role])
                independent.agree(actual, best['metrics'][role])
                independent.agree(actual, selected[role])
                checks[key + '_' + role] = receipt
            del old, incumbent, last, best, model
    else:
        science = m.s.read(base / 'audit_completed.json')
        assert science['actual_audit_returncode'] == 0 and science['audit_sha256'] == m.s.sha(base / 'science/audit.json')
        for key in m.CASES:
            folder = base / 'once_tests' / key
            ex = m.s.read(folder.with_name(key + '.test_exit.json'))
            assert ex['returncode'] == 0 and ex['actual_exit'] and (not Path('/proc', str(ex['pid']), 'cmdline').exists())
            best = torch.load(base / key / 'best.pt', map_location='cpu', weights_only=False)
            started = m.s.read(folder / 'started.json')
            result = m.s.read(folder / 'result.json')
            assert started['sources'] == meta['sources'] and started['selected_sha256'] == m.s.sha(base / key / 'best.pt') and (started['model_digest'] == m.s.digest(best['model'])) and (result['selected_epoch'] == started['selected_epoch'] == best['epoch']) and (result['test_used_for_selection'] is False)
            if result['test_reused']:
                assert result['test_evaluations'] == started['test_evaluations'] == 0 and m.s.read(m.BASE / 'once_tests' / key / 'started.json')['model_digest'] == started['model_digest'] and (started['reused_result_sha256'] == m.s.sha(m.BASE / 'once_tests' / key / 'result.json')) and (m.s.sha(folder / 'predictions.npz') == m.s.sha(m.BASE / 'once_tests' / key / 'predictions.npz'))
            else:
                assert result['test_evaluations'] == started['test_evaluations'] == 1
            actual, checks[key] = independent.verify_predictions(folder / 'predictions.npz', best['model'], 'test', manifest)
            independent.agree(actual, result['metrics'])
    out.mkdir()
    m.s.save(out / 'audit.json', dict(passed=True, phase=phase, sources=meta['sources'], training_commit=meta['git_commit'], checks=checks, models_for_shape_seed=len(m.CASES) if phase == 'fit' else 0, model_forward_calls=0, TEST_images_read=0, frozen_controls=controls), exclusive=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['fit', 'test'], default='fit')
    a = p.parse_args()
    try:
        audit(a.base, a.out, a.phase)
    except BaseException as e:
        mark = a.base / (a.phase + '_audit_failure.json')
        if not mark.exists():
            m.s.save(mark, dict(error=repr(e), automatic_retry=False), exclusive=True)
        raise
