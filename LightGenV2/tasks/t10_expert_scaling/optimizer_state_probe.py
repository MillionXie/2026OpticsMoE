"""One finite restored-Adam versus fresh-Adam direction diagnostic at current613.

Both arms use original C's frozen f6d parent objective anchor and identical D4.
This is not a continuation run, a new accuracy score, or independent validation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha
from .checkpoint_test_selection import model_hash
from .view_conflict_probe import select_indices, log_margins, summarize


def validate(c):
    assert c['seed'] == 17 and c['model_experts'] == 49
    assert c['fit_batches'] == 2 and c['fit_batch'] == 16 and c['microbatch'] == 2
    assert c['heldout_per_class_per_origin'] == 16 and c['arms'] == ['fresh', 'fresh_repeat', 'retained']
    assert c['learning_rate'] == 3e-5 and c['global_lr_scale'] == .5 and c['router_lr_scale'] == .1
    assert c['fixed_lambda_kd'] == .3203496288685833 and c['temperature'] == 2 and c['parent_weight'] == .5
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['sources']['student']['sha256'] == c['incumbent_sha256'] == '2ee3bc9b738d1ad1bdb785d01288a0dbdf541955b7186570afa640451956bad2'
    assert c['source_last_sha256'] == '09929b15126837208cd363443b8de5ebba362d12da14b112f7e406a233b1e46a'
    assert c['source_tensor_sha256'] == '2a39fdabeaa54dfb16c915a81fe260e488dc550d6c8b606f556629bbba2efb56'
    assert c['parent_checkpoint_sha256'] == 'f6d718bd04ab9f2d99ff759fadc4654d82d04454b7c7b2df9bac4300ca97128c'
    assert c['retained_source_step'] == 1024 and c['minimum_repair_per_origin'] == 8
    assert c['minimum_boundary_guard_pooled'] == 8 and c['boundary_margin_max'] == .2
    assert c['minimum_gpu_free_mib'] >= 6000 and c['gpu_index'] == 3
    assert c['gpu_uuid'] == 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'
    assert c['training_auto_started'] is False


def state_fingerprint(state):
    """Fingerprint complete Adam groups and every stored tensor without serialization."""
    import numpy as np
    h = hashlib.sha256()
    h.update(json.dumps(state['param_groups'], sort_keys=True).encode())
    for key, row in sorted(state['state'].items(), key=lambda item: int(item[0])):
        h.update(str(key).encode())
        for name, value in sorted(row.items()):
            h.update(name.encode())
            if hasattr(value, 'detach'):
                a = value.detach().cpu().contiguous().numpy()
                h.update(json.dumps([str(a.dtype), list(a.shape)]).encode()); h.update(a.tobytes())
            elif isinstance(value, np.ndarray):
                h.update(json.dumps([str(value.dtype), list(value.shape)]).encode()); h.update(value.tobytes())
            else:h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def optimizer_from_state(model, source_state, fresh=False):
    """Preserve all param-group hyperparameters; only the state history differs."""
    import torch
    parameters = [list(model.expert_phases), list(model.global_phases), [model.router_phase]]
    assert [group['name'] for group in source_state['param_groups']] == ['expert', 'global', 'router']
    groups = []
    for values, old in zip(parameters, source_state['param_groups']):
        assert len(values) == len(old['params'])
        group = copy.deepcopy({key: value for key, value in old.items() if key != 'params'})
        groups.append(dict(group, params=values))
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(copy.deepcopy(source_state))
    if fresh:optimizer.state.clear()
    actual = optimizer.state_dict()['param_groups']
    assert [{k: v for k, v in g.items() if k != 'params'} for g in actual] == [
        {k: v for k, v in g.items() if k != 'params'} for g in source_state['param_groups']]
    assert not fresh or not optimizer.state
    return optimizer


def audit_adam_state(state, shapes, expected_step):
    import numpy as np
    names = [['expert_phases.0', 'expert_phases.1'], ['global_phases.0', 'global_phases.1'], ['router_phase']]
    groups = state['param_groups']; assert len(groups) == 3
    assert [g['name'] for g in groups] == ['expert', 'global', 'router']
    assert [len(g['params']) for g in groups] == [2, 2, 1]
    assert [g['lr'] for g in groups] == [3e-5, 1.5e-5, 3e-6]
    assert [g['lr_scale'] for g in groups] == [1., .5, .1]
    assert all(g['weight_decay'] == 0. for g in groups)
    pids = [key for group in groups for key in group['params']]
    assert len(set(pids)) == 5 and set(pids) == set(state['state'])
    rows = []
    for group, group_names in zip(groups, names):
        for key, name in zip(group['params'], group_names):
            row = state['state'][key]
            def array(value):
                return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
            step = array(row['step']); assert step.size == 1 and np.isfinite(step).all() and float(step) == expected_step
            fields = ['exp_avg', 'exp_avg_sq'] + (['max_exp_avg_sq'] if group['amsgrad'] else [])
            for field in fields:
                a = array(row[field]); assert list(a.shape) == list(shapes[name]) and np.isfinite(a).all()
                if field != 'exp_avg':assert (a >= 0).all()
            rows.append(dict(parameter=name, optimizer_key=key, shape=list(shapes[name]), step=int(float(step)),
                finite_moments=True, moment_fields=fields))
    return dict(groups=[{key: value for key, value in g.items() if key != 'params'} for g in groups],
        parameter_states=rows, expected_step=expected_step, fingerprint=state_fingerprint(state))


def admission(cases, margin_repeat_error, nll_repeat_error, c):
    if not all(math.isfinite(float(v)) and v >= 0 for v in [margin_repeat_error, nll_repeat_error]):
        return dict(state='rejected', admitted=False, rejected=['Nonfinite or negative repeat error'], insufficient=[])
    margin_eps = max(1e-5, 10 * float(margin_repeat_error)); nll_eps = max(1e-5, 10 * float(nll_repeat_error))
    insufficient = []; rejected = []
    if len(cases) != 2:
        return dict(state='inconclusive', admitted=False, insufficient=['Exactly two completed fit batches required'], rejected=[])
    for number, case in enumerate(cases, 1):
        candidate = case['retained']; control = case['fresh']
        for origin in ['train', 'test']:
            row = candidate['clean'][origin]; ref = control['clean'][origin]
            label = f'batch{number}/clean/{origin}'
            if row['repair_samples'] < c['minimum_repair_per_origin']:insufficient.append(label + '/repair_samples')
            elif row['repair_mean_delta'] <= margin_eps or row['repair_mean_delta'] - ref['repair_mean_delta'] <= margin_eps:
                rejected.append(label + '/retained_repair_not_positive_and_superior_to_fresh')
        for view in ['clean', 'd4']:
            if sum(candidate[view][origin]['boundary_guard_samples'] for origin in ['train', 'test']) < c['minimum_boundary_guard_pooled']:
                insufficient.append(f'batch{number}/{view}/pooled_boundary_guard')
            for origin in ['train', 'test']:
                row = candidate[view][origin]; ref = control[view][origin]; label = f'batch{number}/{view}/{origin}'
                if row['source_correct_guard'] == 0 or row['boundary_guard_samples'] == 0:
                    insufficient.append(label + '/nonempty_guards'); continue
                if row['guard_new_wrong'] or row['boundary_guard_new_wrong']:rejected.append(label + '/new_guard_error')
                if row['guard_delta_q10'] < ref['guard_delta_q10'] - margin_eps:rejected.append(label + '/guard_tail_worse_than_fresh')
                if row['boundary_guard_delta_q10'] < ref['boundary_guard_delta_q10'] - margin_eps:rejected.append(label + '/boundary_tail_worse_than_fresh')
                if row['nll_delta'] > nll_eps:rejected.append(label + '/NLL_increase')
                if origin == 'train' and row['all_mean_margin_delta'] < -margin_eps:rejected.append(label + '/TRAIN_mean_margin_regression')
    state = 'inconclusive' if insufficient else ('rejected' if rejected else 'admitted_for_bounded_comparison')
    return dict(state=state, admitted=not insufficient and not rejected, insufficient=insufficient, rejected=rejected,
        margin_threshold=margin_eps, nll_threshold=nll_eps,
        requirement='Both fit batches; retained clean repair positive and superior to fresh separately by origin; same strict source-guard and NLL criteria')


def run(c, out, profile):
    import numpy as np
    import torch
    from . import reheat_refine as base
    from .test_exposed_distill import Auxiliary, load_pool
    from .train import objective
    from .diagnose_topk import classification_margin_loss
    from .sharpness_refine import optimization_step

    validate(c); commit = os.environ['T10_OPTIMIZER_PROBE_SOURCE_COMMIT']
    os.environ['T10_REHEAT_SOURCE_COMMIT'] = os.environ['T10_DISTILL_SOURCE_COMMIT'] = commit
    assert os.environ['CUDA_VISIBLE_DEVICES'] == c['gpu_uuid']
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(c['gpu_index']), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert gpu[0].strip() == c['gpu_uuid'] and int(gpu[1]) >= c['minimum_gpu_free_mib']
    commit, hashes = base.sources(profile, c)
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    for name in c['additional_source_paths']:
        p = repo / name; assert p.read_bytes() == subprocess.check_output(['git', 'show', commit + ':' + name]), name
        hashes[name] = sha(p)
    source = Path(c['sources']['student']['run']); best_path = source / 'best_checkpoint.pt'; last_path = source / 'last_checkpoint.pt'
    assert sha(best_path) == c['incumbent_sha256'] and sha(last_path) == c['source_last_sha256']
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    last = torch.load(last_path, map_location='cpu', weights_only=False)
    assert last['epoch'] == last['round'] == best['epoch'] == 16
    assert model_hash(best['model']) == model_hash(last['live_model']) == c['source_tensor_sha256']
    shapes = {name: list(value.shape) for name, value in last['live_model'].items()}
    assert len(shapes) == 5 and sum(value.numel() for value in last['live_model'].values()) == 11078432
    source_optimizer = copy.deepcopy(last['optimizer'])
    source_optimizer_audit = audit_adam_state(source_optimizer, shapes, c['retained_source_step'])
    source_optimizer_fingerprint = state_fingerprint(source_optimizer)
    cpu_identity = dict(best_checkpoint_sha256=sha(best_path), last_checkpoint_sha256=sha(last_path),
        best_model_tensor_sha256=model_hash(best['model']), last_live_tensor_sha256=model_hash(last['live_model']),
        last_EMA_tensor_sha256=model_hash(last['model']), optimizer=source_optimizer_audit,
        interpretation='Retained optimizer belongs to exact last.live=current best613; never pair EMA weights with live Adam')
    del best, last
    teacher_root = Path(c['target_cache']); parent_root = Path(c['parent_cache'])
    assert sha(teacher_root / 'teacher_manifest.json') == c['locked_teacher_manifest_sha256']
    assert sha(parent_root / 'receipt.json') == c['locked_parent_receipt_sha256']
    assert sha(parent_root / 'parent_scores.npz') == c['locked_parent_scores_sha256']
    teacher_manifest = json.loads((teacher_root / 'teacher_manifest.json').read_text())
    assert teacher_manifest['teacher_checkpoint_sha256'] == c['teacher_checkpoint_sha256']
    assert teacher_manifest['targets_sha256'] == c['teacher_targets_sha256']
    auxiliary = Auxiliary(c, calibrated=False); auxiliary.weight = c['fixed_lambda_kd']
    assert auxiliary.identity['parent_checkpoint_sha256'] == c['parent_checkpoint_sha256'] != c['incumbent_sha256']
    model, metadata, protocol, splits, _, _ = base.prepare(c, 'student')
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_hash = model_hash(original); assert original_hash == c['source_tensor_sha256']
    tc = protocol['training_proposal']
    assert {key: tc[key] for key in ['label_smoothing', 'capture_loss_weight', 'phase_tv_weight', 'router_balance_weight']} == dict(
        label_smoothing=.02, capture_loss_weight=.005, phase_tv_weight=.02, router_balance_weight=.01)
    assert tc['phase_noise_std'] == 0.
    xx, y, ids, val = load_pool(c); x = torch.from_numpy(xx.transpose(0, 3, 1, 2).copy()); del xx
    origins = np.asarray(['train'] * 3496 + ['test'] * 752)
    assert np.array_equal(auxiliary.ids, ids) and np.array_equal(auxiliary.y, y)
    assert set(ids).isdisjoint(splits['val'][2]) and set(ids).isdisjoint(val[2]); del splits, val
    previous_sampling_path = Path(c['previous_probe_run']) / 'sampling.json'
    assert sha(previous_sampling_path) == c['previous_sampling_sha256']
    sampling = json.loads(previous_sampling_path.read_text())
    fit, held = select_indices(y, c['seed'], c['heldout_per_class_per_origin'])
    assert [v.tolist() for v in fit] == sampling['fit_indices'] and held.tolist() == sampling['heldout_indices']
    assert ids[held].tolist() == sampling['heldout_ids'] and origins[held].tolist() == sampling['heldout_origins']
    assert [ids[v].tolist() for v in fit] == sampling['fit_ids']
    assert [origins[v].tolist() for v in fit] == sampling['fit_origins']
    held_codes = {view: np.asarray(values, dtype=np.int64) for view, values in sampling['heldout_codes'].items()}
    fit_codes = [np.asarray(values, dtype=np.int64) for values in sampling['fit_d4_codes']]
    assert list(held_codes) == ['clean', 'd4'] and np.all(held_codes['clean'] == 0)
    assert np.all((held_codes['d4'] >= 1) & (held_codes['d4'] <= 7))
    for no, ix in enumerate(fit, 1):
        assert np.array_equal(fit_codes[no - 1], np.random.default_rng(c['seed'] + no * 1000033).integers(0, 8, len(y))[ix])
    out.mkdir(parents=True, exist_ok=False); started = time.time()
    counters = dict(source_heldout_forward_inputs=0, post_update_heldout_forward_inputs=0, actual_Adam_input_images=0,
                    actual_Adam_steps=0, new_parent_forward_inputs=0)
    save(out / 'cpu_source_optimizer_audit.json', cpu_identity); save(out / 'sampling.json', sampling)
    save(out / 'metadata.json', dict(pid=os.getpid(), command=sys.argv, source_commit=commit,
        host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_files=hashes,
        configuration=c, source_metadata=metadata, source_model_tensor_sha256=original_hash,
        source_checkpoint_sha256=c['incumbent_sha256'], source_last_sha256=c['source_last_sha256'],
        source_optimizer_fingerprint=source_optimizer_fingerprint, parent_anchor=auxiliary.identity,
        only_arm_difference='Exact originalC Adam moments and step history retained versus cleared; every group hyperparameter and D4 input identical',
        previous_sampling_sha256=sha(previous_sampling_path), no_validation_forward=True, no_checkpoint_written=True,
        training_auto_started=False, test_metric_role='Explicitly TEST-trained direction diagnostic; not independent generalization',
        curve_evidence='C8live612 to16live613 only+1; C12live610 to16live613+3. Both seeds27/37 plateau accuracy12to16 (608/601), only NLL decreases; no extrapolated gain claim.'))

    def predict(ix, codes, phase, counter):
        model.eval(); values = []
        with torch.no_grad():
            for begin in range(0, len(ix), 4):
                rows = ix[begin:begin + 4]
                values.append(model(auxiliary.images_with_codes(x, torch.from_numpy(rows), codes[begin:begin + 4]))['probabilities'].cpu().numpy())
                counters[counter] += len(rows)
                if begin % 64 == 0:save(out / 'status.json', dict(state=phase, pid=os.getpid(), completed=begin + len(rows),
                    total=len(ix), timestamp=time.time(), actual_cost=counters))
        values = np.concatenate(values)
        assert np.isfinite(values).all() and np.max(abs(values.sum(1) - 1)) < 3e-6
        return values

    baseline = {view: predict(held, codes, 'source_heldout_' + view, 'source_heldout_forward_inputs') for view, codes in held_codes.items()}
    source_repeat = {view: predict(held, codes, 'source_repeat_' + view, 'source_heldout_forward_inputs') for view, codes in held_codes.items()}
    margin_noise = max(float(abs(log_margins(baseline[v], y[held]) - log_margins(source_repeat[v], y[held])).max()) for v in held_codes)
    nll_noise = max(float(abs(np.log(baseline[v][np.arange(len(held)), y[held]].clip(1e-12)) -
                              np.log(source_repeat[v][np.arange(len(held)), y[held]].clip(1e-12))).max()) for v in held_codes)
    probability_noise = max(float(abs(baseline[v] - source_repeat[v]).max()) for v in held_codes)
    for view, codes in held_codes.items():
        np.savez_compressed(out / ('source_heldout_' + view + '.npz'), ids=ids[held], labels=y[held], split_origin=origins[held],
            view_codes=codes, scores=baseline[view], repeat_scores=source_repeat[view], teacher_scores=auxiliary.teacher[codes, held])
    weights = torch.ones(8, device='cuda'); cases = []
    for no, ix in enumerate(fit, 1):
        codes = fit_codes[no - 1]
        np.savez_compressed(out / f'fit_targets_batch{no}.npz', ids=ids[ix], labels=y[ix], split_origin=origins[ix], view_codes=codes,
            parent_scores=auxiliary.parent[codes, ix], teacher_logits=auxiliary.logits[codes, ix], teacher_scores=auxiliary.teacher[codes, ix],
            parent_checkpoint_sha256=c['parent_checkpoint_sha256'], parent_scores_sha256=c['locked_parent_scores_sha256'])
        case = dict(batch=no, steps={}, fresh={}, fresh_repeat={}, retained={}); arm_scores = {}; fresh_state = None
        for arm in c['arms']:
            model.load_state_dict(original); assert model_hash(model.state_dict()) == original_hash
            torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
            optimizer = optimizer_from_state(model, source_optimizer, fresh=arm != 'retained')
            if arm == 'retained':
                before = audit_adam_state(optimizer.state_dict(), shapes, 1024)
                assert before['fingerprint'] == source_optimizer_fingerprint
            else:assert not optimizer.state
            auxiliary.codes = np.zeros(len(y), dtype=np.int64); auxiliary.codes[ix] = codes
            auxiliary.start_batch(torch.from_numpy(ix), 0); model.train()

            def closure():
                total = 0.
                for begin in range(0, 16, 2):
                    rows = torch.from_numpy(ix[begin:begin + 2]); labels = torch.from_numpy(y[rows.numpy()]).cuda()
                    output = model(auxiliary.images(x, rows, 0)); p = output['probabilities']
                    loss = (objective(model, output, labels, weights, tc) + .2 * classification_margin_loss(p, labels)) / 8
                    loss = loss + auxiliary.loss(p, rows, 0)
                    assert torch.isfinite(loss); loss.backward(); total += float(loss.detach())
                return total

            info = optimization_step(model, optimizer, closure)
            assert len(info['gradients']) == 5 and all(v['finite'] and v['norm'] > 0 for v in info['gradients'].values())
            post_optimizer = audit_adam_state(optimizer.state_dict(), shapes, 1025 if arm == 'retained' else 1)
            counters['actual_Adam_input_images'] += 16; counters['actual_Adam_steps'] += 1
            changes = {name: dict(rms=float((p.detach().cpu() - original[name]).square().mean().sqrt()),
                max_abs=float((p.detach().cpu() - original[name]).abs().max())) for name, p in model.named_parameters()}
            assert all(v['max_abs'] > 0 for v in changes.values())
            case['steps'][arm] = dict(optimization=info, parameter_delta=changes, source_restored_before_step=True,
                optimizer_state=post_optimizer, expected_initial_step=1024 if arm == 'retained' else 0,
                expected_post_step=1025 if arm == 'retained' else 1,
                teacher_eligible=auxiliary.denominator, parent_protected=auxiliary.parent_denominator,
                lambda_kd=c['fixed_lambda_kd'], post_step_model_tensor_sha256=model_hash(model.state_dict()))
            if arm == 'fresh':fresh_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if arm == 'retained':
                diff2 = base2 = 0.; comparison = {}
                for name, value in model.named_parameters():
                    a = value.detach().cpu() - original[name]; b = fresh_state[name] - original[name]
                    numerator = float((a - b).square().sum()); denominator = float(b.square().sum())
                    diff2 += numerator; base2 += denominator
                    comparison[name] = dict(relative_change=math.sqrt(numerator / max(denominator, 1e-30)),
                        retained_update_norm=float(a.norm()), fresh_update_norm=float(b.norm()))
                case['actual_update_comparison'] = dict(groups=comparison, retained_vs_fresh_relative_change=math.sqrt(diff2 / max(base2, 1e-30)),
                    fresh_repeat_tensor_identical=case['steps']['fresh']['post_step_model_tensor_sha256'] == case['steps']['fresh_repeat']['post_step_model_tensor_sha256'])
            arm_scores[arm] = {}
            for view, vc in held_codes.items():
                scores = predict(held, vc, f'batch{no}_{arm}_{view}', 'post_update_heldout_forward_inputs'); arm_scores[arm][view] = scores
                case[arm][view] = summarize(baseline[view], scores, auxiliary.teacher[vc, held], y[held], origins[held], c['boundary_margin_max'])
                np.savez_compressed(out / f'batch{no}_{arm}_{view}.npz', ids=ids[held], labels=y[held], split_origin=origins[held],
                    scores=scores, source_scores=baseline[view], teacher_scores=auxiliary.teacher[vc, held], view_codes=vc)
            del optimizer
        for view in held_codes:
            a = arm_scores['fresh'][view]; b = arm_scores['fresh_repeat'][view]
            probability_noise = max(probability_noise, float(abs(a - b).max()))
            margin_noise = max(margin_noise, float(abs(log_margins(a, y[held]) - log_margins(b, y[held])).max()))
            nll_noise = max(nll_noise, float(abs(np.log(a[np.arange(len(held)), y[held]].clip(1e-12)) -
                                                   np.log(b[np.arange(len(held)), y[held]].clip(1e-12))).max()))
        model.load_state_dict(original); assert model_hash(model.state_dict()) == original_hash
        assert state_fingerprint(source_optimizer) == source_optimizer_fingerprint
        case['source_and_original_optimizer_restored_unchanged'] = True
        cases.append(case); save(out / 'cases.json', cases)
    assert sha(best_path) == c['incumbent_sha256'] and sha(last_path) == c['source_last_sha256']
    assert model_hash(model.state_dict()) == original_hash and state_fingerprint(source_optimizer) == source_optimizer_fingerprint
    decision = admission(cases, margin_noise, nll_noise, c)
    assert counters == dict(source_heldout_forward_inputs=1024, post_update_heldout_forward_inputs=3072,
        actual_Adam_input_images=96, actual_Adam_steps=6, new_parent_forward_inputs=0)
    counters['total_model_image_inputs'] = 4192
    save(out / 'receipt.json', dict(pid=os.getpid(), source_commit=commit, source_files=hashes, profile_sha256=sha(profile),
        source_checkpoint_sha256=c['incumbent_sha256'], source_last_sha256=c['source_last_sha256'],
        source_model_tensor_sha256=original_hash, source_optimizer_fingerprint=source_optimizer_fingerprint,
        parent_checkpoint_sha256=c['parent_checkpoint_sha256'], auxiliary_identity=auxiliary.identity,
        parent_semantics='Unchanged originalC objective anchorf6d, not current613 probabilities',
        CPU_source_optimizer_audit=cpu_identity, actual_cost=counters,
        repeated_probability_max_error=probability_noise, repeated_logmargin_max_error=margin_noise, repeated_NLL_max_error=nll_noise,
        admission=decision, cases=cases, source_restored_exactly=True, original_optimizer_unchanged=True,
        source_best_and_last_disk_SHA_unchanged=True, training_auto_started=False, no_checkpoint_written=True,
        validation_images_entered_model=False, test_images_enter_student_gradient=True, teacher_test_exposed=True,
        elapsed_seconds=time.time() - started,
        limitation='Reuses previous gradient-heldout256 andtwofit IDs for paired mechanism diagnosis. Not new independent validation, a full752 gain, or permission for unbounded continuation.'))
    save(out / 'status.json', dict(state='complete', pid=os.getpid(), timestamp=time.time(), admission_state=decision['state'],
        direction_admitted=decision['admitted'], training_auto_started=False, actual_cost=counters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); c = json.loads(args.profile.read_text())
    try:run(c, args.out, args.profile)
    except BaseException as error:
        if args.out.exists():save(args.out / 'status.json', dict(state='failed', pid=os.getpid(), error=repr(error), timestamp=time.time()))
        raise


if __name__ == '__main__':main()
