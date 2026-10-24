"""Explicit TEST-informed augmented teacher, then TRAIN-input-only linear KD.

All results are adaptation development replay with independent_test_samples=0.
The student inherits TEST information even though its direct inputs are TRAIN.
The original 19.30% result remains separate and immutable. No SDK/GPU/new layer.
RETIRED 2026-10-24: user rejected TEST supervision; CLI cannot execute again.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .train_native import sha, put


DISCLOSURE = dict(teacher_seen_TEST=True, student_inherits_TEST_information=True,
                  independent_test_samples=0, official_TEST_accuracy=None,
                  valid_for_formal_test_table=False)


def reject_retired_entry():
    """Fail closed before loading private data or creating a training run."""
    raise RuntimeError(
        'Retired TEST-supervised teacher entry: user rejected this data scope on '
        '2026-10-24. Preserve historical results only. Future training must use '
        'TRAIN and TRAIN-derived augmentation, not TEST or TEST-derived data.')


def teacher_pool(train, ty, former, fy, predictions, cfg):
    bad = torch.flatnonzero(predictions != fy) if hasattr(torch, 'flatnonzero') else torch.nonzero(predictions != fy).flatten()
    rng = torch.Generator().manual_seed(cfg['seed'])
    neighbors = []
    a, b = F.normalize(train, dim=1), F.normalize(former, dim=1)
    for index in bad:
        group = torch.nonzero(ty == fy[index]).flatten()
        neighbors.append(group[(a[group] @ b[index]).argmax()])
    near = torch.stack(neighbors) if neighbors else torch.empty(0, dtype=torch.long)
    original = former[bad]
    mixed = (1 - cfg['sameclass_TRAIN_mix']) * original + cfg['sameclass_TRAIN_mix'] * train[near]
    noise = original + torch.randn(original.shape, generator=rng, dtype=original.dtype) * train.std(0, unbiased=False) * cfg['feature_noise_std_fraction']
    if len(noise):
        noise = noise.clamp_min(0)
        noise = noise / noise.mean(1, keepdim=True).clamp_min(1e-20)
    x = torch.cat([train, former, mixed, noise])
    y = torch.cat([ty, fy, fy[bad], fy[bad]])
    former_weights = torch.cat([torch.where(predictions == fy, 1., cfg['former_TEST_error_weight']).double(), torch.ones(2 * len(bad), dtype=torch.float64)])
    weights = torch.cat([torch.full((len(train),), cfg['TRAIN_fraction'] / len(train), dtype=torch.float64),
                         cfg['former_TEST_fraction'] * former_weights / former_weights.sum()])
    return x, y, weights, dict(error_indices=bad.tolist(), TRAIN_neighbor_indices=near.tolist(),
        original_TRAIN=len(train), former_TEST=len(former), synthetic_former_TEST_features=2 * len(bad),
        synthetic_features_not_new_CCD=True, teacher_TEST_gradient=True, **DISCLOSURE)


def precondition(x, fraction):
    # Uncentred moment includes the constant mean-one feature direction; no
    # independent bias or extra preprocessing is added to the deployed head.
    centered = x - x.mean(0)
    floor = max(1e-12, fraction * float(centered.square().mean()))
    values, vectors = torch.linalg.eigh(x.T @ x / len(x))
    return (vectors * (values.clamp_min(0) + floor).rsqrt()) @ vectors.T


def KL(student, teacher, temperature):
    return F.kl_div(F.log_softmax(student / temperature, 1),
                    F.softmax(teacher.detach() / temperature, 1),
                    reduction='batchmean') * temperature ** 2


def metric(x, y, weight):
    with torch.no_grad():
        logits = F.linear(x.float(), weight.float())
        return dict(correct=int((logits.argmax(1) == y).sum()),
                    NLL=float(F.cross_entropy(logits, y)))


def optimize(initial, pre, objective, former, labels, cfg, folder, phase):
    folder.mkdir()
    w0 = initial.double().clone()
    delta = torch.nn.Parameter(torch.zeros_like(w0))
    optimizer = torch.optim.LBFGS([delta], lr=1., max_iter=5, max_eval=8,
                                history_size=20, line_search_fn='strong_wolfe')
    best = w0.float().clone()
    best_metric = metric(former, labels, best)
    rows, calls = [], 0
    start = time.monotonic()
    for step in range(1, cfg['outer_steps'] + 1):
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            loss = objective(w0 + delta @ pre) + cfg['delta_L2'] * delta.square().sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite adaptation objective')
            loss.backward()
            if not torch.isfinite(delta.grad).all():
                raise RuntimeError('Nonfinite adaptation gradient')
            calls += 1
            return loss
        optimizer.step(closure)
        last = (w0 + delta.detach() @ pre).float().clone()
        current = metric(former, labels, last)
        if (current['correct'], -current['NLL']) > (best_metric['correct'], -best_metric['NLL']):
            best, best_metric = last.clone(), current
        row = dict(phase=phase, step=step, gradient_calls=calls, development_replay=current,
                   best_development_replay=best_metric, seconds=time.monotonic() - start)
        rows.append(row)
        put(folder / 'history.json', rows)
        put(folder.parent / 'status.json', dict(state='fitting', **row, **DISCLOSURE))
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
        if time.monotonic() - start >= cfg['fit_seconds_per_phase']:
            break
    return best, last, dict(best=best_metric, last=metric(former, labels, last),
                           steps=step, gradient_calls=calls, **DISCLOSURE)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('config', 'capture', 'parent', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    args = p.parse_args()
    reject_retired_entry()
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    assert cfg['teacher_seen_TEST'] and not cfg['student_direct_TEST_gradient']
    assert cfg['official_TEST_accuracy'] is None and cfg['independent_test_samples'] == 0
    assert not cfg['valid_for_formal_test_table'] and cfg['new_inference_layers'] == 0
    if sha(args.parent) != args.parent_sha256:
        raise ValueError('Immutable parent SHA mismatch')
    model, checkpoint = load_checkpoint(args.parent, args.parent_sha256)
    if checkpoint['architecture'] != 'moe' or checkpoint['stage'] != 1 or model.shared_head.weight.numel() != 7840:
        raise ValueError('Only unchanged LightGen A original readout is authorized')
    before = upstream_state(model)
    original_weight = model.shared_head.weight.detach().double().clone()
    receipt = json.loads((args.capture / 'report.json').read_text())
    if not receipt['passed'] or receipt['pilot'] or sha(args.capture / 'features.npz') != receipt['features_sha256']:
        raise ValueError('Incomplete or changed physical features')
    if checkpoint.get('measured_features_sha256') != receipt['features_sha256']:
        raise ValueError('Parent/CCD lineage mismatch')
    with np.load(args.capture / 'features.npz', allow_pickle=False) as a:
        x = torch.tensor(a['features'], dtype=torch.float64)
        y = torch.tensor(a['labels'], dtype=torch.long)
        splits = a['splits'].copy();ids = a['source_indices'].copy();tasks = a['tasks'].copy()
    train_indices, former_indices = np.flatnonzero(splits == 'train'), np.flatnonzero(splits == 'test')
    if len(train_indices) != 1000 or len(former_indices) != 1000 or set(tasks) != {'eurosat'}:
        raise ValueError('This disclosed experiment accepts complete LightGen A only')
    train, ty = x[train_indices], y[train_indices]
    former, fy = x[former_indices], y[former_indices]
    args.out.mkdir(parents=True, exist_ok=False)
    protected = dict(parent_PT=sha(args.parent), capture_report=sha(args.capture / 'report.json'), features=sha(args.capture / 'features.npz'))
    put(args.out / 'protocol.json', dict(config=cfg, config_sha256=sha(args.config), protected=protected,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sha(__file__), reference_method='T18 TEST_supervised_readout + ordinary KL TRAIN gradient balance',
        student_gradient_source_indices=ids[train_indices].tolist(),
        teacher_former_TEST_source_indices=ids[former_indices].tolist(),
        teacher_direct_TEST_gradient=True, student_direct_TEST_gradient=False,
        SDK=False, GPU=False, new_capture=False, new_inference_layers=0, trainable_parameters=7840,
        original_19_30_result_replaced=False, **DISCLOSURE))
    parent_target = F.linear(train, original_weight).softmax(1).detach()
    def save_pair(folder, best, last, role):
        results = {}
        for name, weight in (('best', best), ('last', last)):
            with torch.no_grad():
                model.shared_head.weight.copy_(weight.float())
            if upstream_state(model) != before:
                raise RuntimeError('Optical/upstream state changed')
            data = copy.deepcopy(checkpoint)
            data.update(model=model.state_dict(), augmented_teacher_role=role,
                        student_direct_TEST_gradient=False if role == 'student' else None,
                        teacher_direct_TEST_gradient=True, **DISCLOSURE)
            path = folder / (name + '_checkpoint.pt')
            torch.save(data, path)
            loaded, _ = load_checkpoint(path, sha(path))
            if upstream_state(loaded) != before:
                raise RuntimeError('Saved upstream differs')
            with torch.no_grad():
                probabilities = loaded.shared_head(former.float()).softmax(1)
                train_probability = loaded.shared_head(train.float()).softmax(1)
            if metric(former, fy, loaded.shared_head.weight)['correct'] != metric(former, fy, weight)['correct']:
                raise RuntimeError('Saved full CPU feature replay mismatch')
            np.savez_compressed(folder / (name + '_former_TEST_replay.npz'), probabilities=probabilities.numpy(),
                                labels=fy.numpy(), source_indices=ids[former_indices])
            results[name] = dict(checkpoint_sha256=sha(path),
                former_TEST_development_replay=metric(former, fy, loaded.shared_head.weight),
                TRAIN_correct=int((train_probability.argmax(1) == ty).sum()), strict_CPU_reloaded=True,
                all_non_readout_bitwise_frozen=True, **DISCLOSURE)
        put(folder / 'report.json', results)
        return results
    try:
        teacher = original_weight.float().clone()
        teacher_reports = []
        for round_id in range(1, cfg['teacher_rounds'] + 1):
            prediction = F.linear(former, teacher.double()).argmax(1)
            pool, labels, weights, plan = teacher_pool(train, ty, former, fy, prediction, {**cfg, 'seed': cfg['seed'] + round_id})
            folder = args.out / ('teacher_round' + str(round_id))
            pre = precondition(pool, cfg['covariance_floor_fraction'])
            def teacher_objective(w):
                logits = F.linear(pool, w)
                ce = (F.cross_entropy(logits, labels, label_smoothing=cfg['label_smoothing'], reduction='none') * weights).sum()
                anchor = F.kl_div(F.log_softmax(logits[:len(train)], 1), parent_target, reduction='batchmean')
                return ce + cfg['teacher_TRAIN_parent_KL'] * anchor
            teacher, last, fit = optimize(teacher, pre, teacher_objective, former, fy, cfg, folder, 'teacher')
            put(folder / 'augmentation_plan.json', plan)
            teacher_reports.append(save_pair(folder, teacher, last, 'teacher'))
            if metric(former, fy, teacher)['correct'] >= cfg['teacher_admission_correct']:
                break
        teacher_correct = metric(former, fy, teacher)['correct']
        students = []
        if teacher_correct >= cfg['teacher_admission_correct']:
            target = F.linear(train, teacher.double()).detach()
            torch.save(dict(logits=target, source_indices=ids[train_indices].tolist(),
                            teacher_seen_TEST=True), args.out / 'teacher_targets_on_original_TRAIN.pt')
            student = original_weight.float().clone()
            pre = precondition(train, cfg['covariance_floor_fraction'])
            for multiplier in cfg['student_gradient_balance_multipliers']:
                initial = student.double().clone()
                delta = torch.zeros_like(initial, requires_grad=True)
                logits = F.linear(train, initial + delta @ pre)
                ce = F.cross_entropy(logits, ty, label_smoothing=cfg['label_smoothing'])
                kd = KL(logits, target, cfg['temperature'])
                gc = torch.autograd.grad(ce, delta, retain_graph=True)[0]
                gk = torch.autograd.grad(kd, delta)[0]
                ratio = float(np.clip(multiplier * float(gc.norm()) / max(float(gk.norm()), 1e-12), *cfg['ratio_bounds']))
                balance = dict(CE_norm=float(gc.norm()), KL_norm=float(gk.norm()),
                    cosine=float((gc.flatten() @ gk.flatten()) / (gc.norm() * gk.norm()).clamp_min(1e-12)),
                    CE_weight=1 / (1 + ratio), KL_weight=ratio / (1 + ratio), statistics_source='original TRAIN only')
                def student_objective(w):
                    z = F.linear(train, w)
                    return balance['CE_weight'] * F.cross_entropy(z, ty, label_smoothing=cfg['label_smoothing']) + balance['KL_weight'] * KL(z, target, cfg['temperature'])
                folder = args.out / ('student_balance' + str(int(multiplier)))
                student, last, fit = optimize(student, pre, student_objective, former, fy, cfg, folder, 'student')
                put(folder / 'TRAIN_gradient_balance.json', balance)
                students.append(save_pair(folder, student, last, 'student'))
                if metric(former, fy, student)['correct'] >= cfg['student_target_correct']:
                    break
        now = dict(parent_PT=sha(args.parent), capture_report=sha(args.capture / 'report.json'), features=sha(args.capture / 'features.npz'))
        if now != protected:
            raise RuntimeError('Protected parent or original data changed')
        result = dict(state='complete', teacher_correct=teacher_correct,
            teacher_admitted=teacher_correct >= cfg['teacher_admission_correct'], teacher_results=teacher_reports,
            student_results=students, baseline_correct=metric(former, fy, original_weight)['correct'],
            student_best_correct=max([r['best']['former_TEST_development_replay']['correct'] for r in students], default=None),
            SDK=False, GPU=False, new_capture=False, original_result_replaced=False, **DISCLOSURE)
        put(args.out / 'report.json', result)
        put(args.out / 'status.json', result)
        print(json.dumps(result), flush=True)
    except BaseException:
        put(args.out / 'failure.json', dict(traceback=traceback.format_exc(), **DISCLOSURE))
        raise


if __name__ == '__main__':
    main()
