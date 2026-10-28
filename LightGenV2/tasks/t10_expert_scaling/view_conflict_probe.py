"""Six source-restored Adam steps testing clean versus D4 objective directions.

This is an explicitly TEST-exposed diagnostic, not training or an independent
generalization measurement. The two arms share current613 as initialization AND
parent anchor; this is not a single-variable comparison with historical C.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha
from .checkpoint_test_selection import model_hash


def validate(c):
    assert c['seed'] == 17 and c['model_experts'] == 49
    assert c['fit_batches'] == 2 and c['fit_batch'] == 16 and c['microbatch'] == 2
    assert c['heldout_per_class_per_origin'] == 16
    assert c['learning_rate'] == 3e-5 and c['global_lr_scale'] == .5 and c['router_lr_scale'] == .1
    assert c['fixed_lambda_kd'] == .3203496288685833
    assert c['temperature'] == 2 and c['parent_weight'] == .5
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['sources']['student']['sha256'] == c['incumbent_sha256'] == '2ee3bc9b738d1ad1bdb785d01288a0dbdf541955b7186570afa640451956bad2'
    assert c['source_tensor_sha256'] == '2a39fdabeaa54dfb16c915a81fe260e488dc550d6c8b606f556629bbba2efb56'
    assert c['minimum_repair_per_origin'] == 8 and c['minimum_boundary_guard_pooled'] == 8
    assert c['boundary_margin_max'] == .2 and c['minimum_gpu_free_mib'] >= 6000
    assert c['gpu_index'] == 3 and c['gpu_uuid'] == 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'
    assert c['training_auto_started'] is False and c['arms'] == ['d4', 'd4_repeat', 'clean']


def select_indices(labels, seed=17, heldout_per_cell=16):
    """Uniform class/origin sampling, independent of source/teacher predictions."""
    import numpy as np
    y = np.asarray(labels)
    assert y.shape == (4248,) and heldout_per_cell == 16
    rng = np.random.default_rng(seed + 810019)
    rows = np.arange(len(y)); held = []; batches = [[], []]
    for k in range(8):
        for origin, mask in enumerate([rows < 3496, rows >= 3496]):
            choices = rng.permutation(np.flatnonzero((y == k) & mask))
            assert len(choices) == (437 if origin == 0 else 94)
            held.extend(choices[:heldout_per_cell].tolist())
            for batch in range(2):batches[batch].append(int(choices[heldout_per_cell + batch]))
    held = np.asarray(held, dtype=np.int64); rng.shuffle(held)
    batches = [np.asarray(v, dtype=np.int64) for v in batches]
    for batch in batches:rng.shuffle(batch)
    assert len(held) == 256 and len(set(held)) == 256
    assert len(set(held.tolist() + batches[0].tolist() + batches[1].tolist())) == 288
    return batches, held


def log_margins(scores, labels):
    import numpy as np
    p = np.asarray(scores)
    assert p.shape == (len(labels), 8) and np.isfinite(p).all()
    values = np.log(p.clip(1e-12)); true = values[np.arange(len(labels)), labels].copy()
    values[np.arange(len(labels)), labels] = -np.inf
    return true - values.max(1)


def summarize(source, candidate, teacher, labels, origins, boundary=.2):
    """All groups use fixed source/teacher predictions, never candidate selection."""
    import numpy as np
    before = log_margins(source, labels); after = log_margins(candidate, labels)
    delta = after - before; old = source.argmax(1) == labels; now = candidate.argmax(1) == labels
    repair = (teacher.argmax(1) == labels) & ~old
    result = {}
    for name in ['train', 'test']:
        cell = origins == name; r = cell & repair; guard = cell & old
        edge = guard & (before > 0) & (before <= boundary)
        def mean(mask):return float(delta[mask].mean()) if mask.any() else None
        def q10(mask):return float(np.quantile(delta[mask], .1)) if mask.any() else None
        nll = -np.log(candidate[np.arange(len(labels)), labels].clip(1e-12))
        old_nll = -np.log(source[np.arange(len(labels)), labels].clip(1e-12))
        result[name] = dict(samples=int(cell.sum()), repair_samples=int(r.sum()),
            repair_mean_delta=mean(r), repair_corrected=int((r & now).sum()),
            source_correct_guard=int(guard.sum()), guard_new_wrong=int((guard & ~now).sum()),
            guard_delta_mean=mean(guard), guard_delta_q10=q10(guard),
            boundary_guard_samples=int(edge.sum()), boundary_guard_new_wrong=int((edge & ~now).sum()),
            boundary_guard_delta_mean=mean(edge), boundary_guard_delta_q10=q10(edge),
            all_mean_margin_delta=mean(cell), nll_delta=float((nll[cell] - old_nll[cell]).mean()),
            source_correct=int((cell & old).sum()), candidate_correct=int((cell & now).sum()))
    return result


def admission(cases, margin_repeat_error, nll_repeat_error, c):
    """A direction gate, not evidence of a trained model or guaranteed transfer."""
    if not all(math.isfinite(float(v)) and v >= 0 for v in [margin_repeat_error, nll_repeat_error]):
        return dict(state='rejected', admitted=False, reasons=['Nonfinite or negative repeat error'])
    margin_eps = max(1e-5, 10 * float(margin_repeat_error))
    nll_eps = max(1e-5, 10 * float(nll_repeat_error))
    insufficient = []; rejected = []
    if len(cases) != 2:
        return dict(state='inconclusive', admitted=False, reasons=['Exactly two completed fit batches required'],
                    margin_threshold=margin_eps, nll_threshold=nll_eps)
    for number, case in enumerate(cases, 1):
        clean = case['clean']; control = case['d4']
        for origin in ['train', 'test']:
            row = clean['clean'][origin]; ref = control['clean'][origin]
            label = f'batch{number}/clean/{origin}'
            if row['repair_samples'] < c['minimum_repair_per_origin']:
                insufficient.append(label + '/repair_samples')
            elif (row['repair_mean_delta'] <= margin_eps or
                  row['repair_mean_delta'] - ref['repair_mean_delta'] <= margin_eps):
                rejected.append(label + '/repair_not_positive_and_better_than_D4')
        for view in ['clean', 'd4']:
            pooled = sum(clean[view][origin]['boundary_guard_samples'] for origin in ['train', 'test'])
            if pooled < c['minimum_boundary_guard_pooled']:
                insufficient.append(f'batch{number}/{view}/pooled_boundary_guard')
            for origin in ['train', 'test']:
                row = clean[view][origin]; ref = control[view][origin]
                label = f'batch{number}/{view}/{origin}'
                if row['source_correct_guard'] == 0 or row['boundary_guard_samples'] == 0:
                    insufficient.append(label + '/nonempty_guards')
                    continue
                if row['guard_new_wrong'] or row['boundary_guard_new_wrong']:
                    rejected.append(label + '/new_guard_error')
                if row['guard_delta_q10'] < ref['guard_delta_q10'] - margin_eps:
                    rejected.append(label + '/guard_tail_worse_than_D4')
                if row['boundary_guard_delta_q10'] < ref['boundary_guard_delta_q10'] - margin_eps:
                    rejected.append(label + '/boundary_tail_worse_than_D4')
                if row['nll_delta'] > nll_eps:
                    rejected.append(label + '/NLL_increase')
                if origin == 'train' and row['all_mean_margin_delta'] < -margin_eps:
                    rejected.append(label + '/TRAIN_mean_margin_regression')
    state = 'inconclusive' if insufficient else ('rejected' if rejected else 'admitted_for_bounded_comparison')
    return dict(state=state, admitted=not insufficient and not rejected,
                insufficient=insufficient, rejected=rejected, margin_threshold=margin_eps, nll_threshold=nll_eps,
                requirement='Both fit batches; clean repair positive and superior to D4 separately by origin; fixed-view guards preserved')


def run(c, out, profile):
    import numpy as np
    import torch
    from . import reheat_refine as base
    from .cached_distill import Auxiliary as ImageAuxiliary, conditional_kl_part
    from .test_exposed_distill import load_pool
    from .cnn_distill import temperature_kl_part
    from .train import objective
    from .diagnose_topk import classification_margin_loss
    from .sharpness_refine import optimization_step

    validate(c)
    commit = os.environ['T10_VIEW_PROBE_SOURCE_COMMIT']
    os.environ['T10_REHEAT_SOURCE_COMMIT'] = os.environ['T10_DISTILL_SOURCE_COMMIT'] = commit
    assert os.environ['CUDA_VISIBLE_DEVICES'] == c['gpu_uuid']
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(c['gpu_index']),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert gpu[0].strip() == c['gpu_uuid'] and int(gpu[1]) >= c['minimum_gpu_free_mib']
    commit, hashes = base.sources(profile, c)
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    for name in c['additional_source_paths']:
        p = repo / name
        assert p.read_bytes() == subprocess.check_output(['git', 'show', commit + ':' + name]), name
        hashes[name] = sha(p)
    source = Path(c['sources']['student']['run']); checkpoint_path = source / 'best_checkpoint.pt'
    assert sha(checkpoint_path) == c['incumbent_sha256']
    teacher_root = Path(c['target_cache']); manifest_path = teacher_root / 'teacher_manifest.json'
    assert sha(manifest_path) == c['locked_teacher_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_commit'] == c['teacher_cache_source_commit']
    assert manifest['data_sha256'] == c['data_sha256']
    assert manifest['teacher_checkpoint_sha256'] == c['teacher_checkpoint_sha256']
    assert manifest['targets_sha256'] == c['teacher_targets_sha256'] == sha(teacher_root / 'teacher_targets.npz')

    model, metadata, protocol, splits, _, _ = base.prepare(c, 'student')
    tc = protocol['training_proposal']
    assert {name: tc[name] for name in ['label_smoothing', 'capture_loss_weight', 'phase_tv_weight', 'router_balance_weight']} == dict(
        label_smoothing=.02, capture_loss_weight=.005, phase_tv_weight=.02, router_balance_weight=.01)
    assert tc['phase_noise_std'] == 0.
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_hash = model_hash(original); assert original_hash == c['source_tensor_sha256']
    assert len(list(model.parameters())) == 5 and sum(p.numel() for p in model.parameters()) == 11078432
    xx, y, ids, val = load_pool(c)
    x = torch.from_numpy(xx.transpose(0, 3, 1, 2).copy()); del xx
    origins = np.asarray(['train'] * 3496 + ['test'] * 752)
    assert set(ids).isdisjoint(splits['val'][2]) and set(ids).isdisjoint(val[2])
    del splits, val
    with np.load(teacher_root / 'teacher_targets.npz', allow_pickle=False) as z:
        assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], y)
        assert np.array_equal(z['split_origin'], origins) and z['view_codes'].tolist() == list(range(8))
        logits = z['teacher_logits'].copy(); teacher = z['teacher_scores'].copy()
    assert logits.shape == teacher.shape == (8, 4248, 8) and np.isfinite(logits).all()
    soft = np.exp(logits - logits.max(-1, keepdims=True)); soft /= soft.sum(-1, keepdims=True)
    assert np.max(abs(soft - teacher)) < 1e-6
    fit, held = select_indices(y, c['seed'], c['heldout_per_class_per_origin'])
    held_codes = {'clean': np.zeros(len(held), dtype=np.int64),
                  'd4': np.random.default_rng(c['seed'] + 680011).integers(1, 8, len(held))}
    fit_codes = [np.random.default_rng(c['seed'] + no * 1000033).integers(0, 8, len(y))[ix]
                 for no, ix in enumerate(fit, 1)]
    out.mkdir(parents=True, exist_ok=False); started = time.time()
    counters = dict(frozen_source_parent_forward_inputs=0, source_heldout_forward_inputs=0,
                    post_update_heldout_forward_inputs=0, actual_Adam_input_images=0, actual_Adam_steps=0)
    save(out / 'metadata.json', dict(pid=os.getpid(), command=sys.argv, source_commit=commit,
        host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_files=hashes, configuration=c, source_metadata=metadata, source_model_tensor_sha256=original_hash,
        source_checkpoint_sha256=sha(checkpoint_path), initialization='current613 model field, fresh Adam before every actual step',
        parent_anchor='Current613 frozen probabilities of each fit ID at the matching view; not historical f6d parent',
        teacher_manifest_sha256=sha(manifest_path), teacher_targets_sha256=manifest['targets_sha256'],
        teacher_checkpoint_sha256=manifest['teacher_checkpoint_sha256'],
        test_metric_role='TEST images/labels enter diagnostic gradients; development fit, not independent generalization',
        no_validation_forward=True, no_checkpoint_written=True, training_auto_started=False,
        gpu_sharing_authorized=True, initial_gpu_free_mib=int(gpu[1])))
    save(out / 'sampling.json', dict(fit_indices=[v.tolist() for v in fit], fit_ids=[ids[v].tolist() for v in fit],
        fit_origins=[origins[v].tolist() for v in fit], fit_d4_codes=[v.tolist() for v in fit_codes],
        heldout_indices=held.tolist(), heldout_ids=ids[held].tolist(), heldout_origins=origins[held].tolist(),
        heldout_codes={name: value.tolist() for name, value in held_codes.items()},
        selection='Uniform fixed class/origin RNG; source/teacher predictions do not select IDs; 288 unique underlying images'))

    def images(ix, codes):
        return ImageAuxiliary.images_with_codes(None, x, torch.from_numpy(ix), codes)

    def predict(ix, codes, phase, counter):
        model.eval(); values = []
        with torch.no_grad():
            for begin in range(0, len(ix), 4):
                rows = ix[begin:begin + 4]
                values.append(model(images(rows, codes[begin:begin + 4]))['probabilities'].cpu().numpy())
                counters[counter] += len(rows)
                if begin % 64 == 0:
                    save(out / 'status.json', dict(state=phase, pid=os.getpid(), completed=begin + len(rows),
                        total=len(ix), timestamp=time.time(), actual_cost=counters))
        result = np.concatenate(values)
        assert np.isfinite(result).all() and np.max(abs(result.sum(1) - 1)) < 3e-6
        return result

    baseline = {view: predict(held, codes, 'source_heldout_' + view, 'source_heldout_forward_inputs')
                for view, codes in held_codes.items()}
    baseline_repeat = {view: predict(held, codes, 'source_repeat_' + view, 'source_heldout_forward_inputs')
                       for view, codes in held_codes.items()}
    source_margin_noise = max(float(abs(log_margins(baseline[v], y[held]) -
                                       log_margins(baseline_repeat[v], y[held])).max()) for v in held_codes)
    source_nll_noise = max(float(abs(-np.log(baseline[v][np.arange(len(held)), y[held]].clip(1e-12)) +
                                     np.log(baseline_repeat[v][np.arange(len(held)), y[held]].clip(1e-12))).max())
                           for v in held_codes)
    probability_noise = max(float(abs(baseline[v] - baseline_repeat[v]).max()) for v in held_codes)
    for view, scores in baseline.items():
        np.savez_compressed(out / ('source_heldout_' + view + '.npz'), ids=ids[held], labels=y[held],
                            split_origin=origins[held], scores=scores, repeat_scores=baseline_repeat[view],
                            teacher_scores=teacher[held_codes[view], held], view_codes=held_codes[view])
    parents = []
    for no, ix in enumerate(fit):
        values = {}
        for view, codes in [('clean', np.zeros(16, dtype=np.int64)), ('d4', fit_codes[no])]:
            values[view] = predict(ix, codes, f'parent_batch{no + 1}_{view}', 'frozen_source_parent_forward_inputs')
            np.savez_compressed(out / f'parent_batch{no + 1}_{view}.npz', ids=ids[ix], labels=y[ix],
                split_origin=origins[ix], view_codes=codes, scores=values[view],
                source_checkpoint_sha256=c['incumbent_sha256'], source_tensor_sha256=original_hash)
        parents.append(values)
    assert counters['frozen_source_parent_forward_inputs'] == 64
    weights = torch.ones(8, device='cuda'); cases = []; margin_noise = source_margin_noise; nll_noise = source_nll_noise

    def new_optimizer():
        return torch.optim.AdamW([
            {'params': list(model.expert_phases), 'lr': c['learning_rate'], 'name': 'expert'},
            {'params': list(model.global_phases), 'lr': c['learning_rate'] * c['global_lr_scale'], 'name': 'global'},
            {'params': [model.router_phase], 'lr': c['learning_rate'] * c['router_lr_scale'], 'name': 'router'}], weight_decay=0.)

    for no, ix in enumerate(fit, 1):
        case = dict(batch=no, steps={}, delta={}, d4={}, d4_repeat={}, clean={}); arm_scores = {}
        control_state = None
        for arm in c['arms']:
            model.load_state_dict(original); assert model_hash(model.state_dict()) == original_hash
            torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
            optimizer = new_optimizer(); assert not optimizer.state
            view = 'clean' if arm == 'clean' else 'd4'
            codes = np.zeros(16, dtype=np.int64) if view == 'clean' else fit_codes[no - 1]
            target = logits[codes, ix]; parent = parents[no - 1][view]
            eligible = target.argmax(1) == y[ix]; protected = (parent.argmax(1) == y[ix]) & ~eligible
            model.train()

            def closure():
                total = 0.
                for begin in range(0, 16, 2):
                    rows = ix[begin:begin + 2]; vc = codes[begin:begin + 2]
                    labels = torch.from_numpy(y[rows]).cuda(); output = model(images(rows, vc)); p = output['probabilities']
                    loss = (objective(model, output, labels, weights, protocol['training_proposal']) +
                            .2 * classification_margin_loss(p, labels)) / 8
                    loss = loss + c['fixed_lambda_kd'] * temperature_kl_part(p,
                        torch.from_numpy(target[begin:begin + 2]).cuda(),
                        torch.from_numpy(eligible[begin:begin + 2]).cuda(), int(eligible.sum()))
                    loss = loss + c['parent_weight'] * conditional_kl_part(p,
                        torch.from_numpy(parent[begin:begin + 2]).cuda(),
                        torch.from_numpy(protected[begin:begin + 2]).cuda(), int(protected.sum()))
                    assert torch.isfinite(loss); loss.backward(); total += float(loss.detach())
                return total

            info = optimization_step(model, optimizer, closure)
            assert len(info['gradients']) == 5 and all(v['finite'] and v['norm'] > 0 for v in info['gradients'].values())
            assert len(optimizer.state) == 5 and all(int(v['step']) == 1 for v in optimizer.state.values())
            counters['actual_Adam_input_images'] += 16; counters['actual_Adam_steps'] += 1
            changes = {name: dict(max_abs=float((value.detach().cpu() - original[name]).abs().max()),
                rms=float((value.detach().cpu() - original[name]).square().mean().sqrt()))
                for name, value in model.named_parameters()}
            assert all(value['max_abs'] > 0 for value in changes.values())
            case['steps'][arm] = dict(optimization=info, learning_rates={g['name']: g['lr'] for g in optimizer.param_groups},
                teacher_eligible=int(eligible.sum()), parent_protected=int(protected.sum()),
                lambda_kd=c['fixed_lambda_kd'], source_restored_before_step=True, optimizer_fresh=True,
                parameter_delta=changes, post_step_model_tensor_sha256=model_hash(model.state_dict()))
            if arm == 'd4':
                control_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            elif arm == 'clean':
                comparison = {}; squared_difference = 0.; squared_control = 0.
                for name, value in model.named_parameters():
                    clean_delta = value.detach().cpu() - original[name]
                    d4_delta = control_state[name] - original[name]
                    numerator = float((clean_delta - d4_delta).square().sum())
                    denominator = float(d4_delta.square().sum())
                    squared_difference += numerator; squared_control += denominator
                    comparison[name] = dict(relative_change=math.sqrt(numerator / max(denominator, 1e-30)),
                        clean_update_norm=float(clean_delta.norm()), d4_update_norm=float(d4_delta.norm()))
                case['actual_update_comparison'] = dict(groups=comparison,
                    clean_vs_D4_relative_change=math.sqrt(squared_difference / max(squared_control, 1e-30)),
                    D4_repeat_tensor_identical=case['steps']['d4']['post_step_model_tensor_sha256'] == case['steps']['d4_repeat']['post_step_model_tensor_sha256'])
            arm_scores[arm] = {}
            for held_view, vc in held_codes.items():
                scores = predict(held, vc, f'batch{no}_{arm}_{held_view}', 'post_update_heldout_forward_inputs')
                arm_scores[arm][held_view] = scores
                case[arm][held_view] = summarize(baseline[held_view], scores, teacher[vc, held],
                    y[held], origins[held], c['boundary_margin_max'])
                np.savez_compressed(out / f'batch{no}_{arm}_{held_view}.npz', ids=ids[held], labels=y[held],
                    split_origin=origins[held], scores=scores, source_scores=baseline[held_view],
                    teacher_scores=teacher[vc, held], view_codes=vc)
            del optimizer
        for view in held_codes:
            a = arm_scores['d4'][view]; b = arm_scores['d4_repeat'][view]
            probability_noise = max(probability_noise, float(abs(a - b).max()))
            margin_noise = max(margin_noise, float(abs(log_margins(a, y[held]) - log_margins(b, y[held])).max()))
            nll_noise = max(nll_noise, float(abs(np.log(a[np.arange(len(held)), y[held]].clip(1e-12)) -
                                                  np.log(b[np.arange(len(held)), y[held]].clip(1e-12))).max()))
        case['source_restored_after_batch'] = True; model.load_state_dict(original)
        assert model_hash(model.state_dict()) == original_hash
        cases.append(case); save(out / 'cases.json', cases)
        save(out / 'status.json', dict(state='probing', pid=os.getpid(), completed=no, total=2,
            timestamp=time.time(), actual_cost=counters))
    model.load_state_dict(original); assert model_hash(model.state_dict()) == original_hash
    assert sha(checkpoint_path) == c['incumbent_sha256']
    decision = admission(cases, margin_noise, nll_noise, c)
    assert counters == dict(frozen_source_parent_forward_inputs=64, source_heldout_forward_inputs=1024,
        post_update_heldout_forward_inputs=3072, actual_Adam_input_images=96, actual_Adam_steps=6)
    counters['total_model_image_inputs'] = sum(value for name, value in counters.items() if name != 'actual_Adam_steps')
    assert counters['total_model_image_inputs'] == 4256
    receipt = dict(source_commit=commit, profile_sha256=sha(profile), source_files=hashes,
        source_checkpoint_sha256=c['incumbent_sha256'], source_model_tensor_sha256=original_hash,
        parent_checkpoint_sha256=c['incumbent_sha256'], parent_model_tensor_sha256=original_hash,
        parent_fit_prediction_files=[f'parent_batch{batch}_{view}.npz' for batch in [1, 2] for view in ['clean', 'd4']],
        teacher_manifest_sha256=sha(manifest_path), teacher_targets_sha256=manifest['targets_sha256'],
        source_restored_exactly=True, source_disk_checkpoint_unchanged=True,
        fresh_optimizer_each_of_six_steps=True, no_checkpoint_written=True, actual_cost=counters,
        repeated_probability_max_error=probability_noise, repeated_logmargin_max_error=margin_noise,
        repeated_NLL_max_error=nll_noise, admission=decision, cases=cases,
        training_auto_started=False, validation_images_entered_model=False,
        teacher_test_exposed=True, test_images_enter_student_gradient=True,
        comparison_scope='Current613/freshAdam/current613 parent shared between arms; only fit view differs. Not historical-C matched continuation.',
        limitation='Direction diagnostic on fixed heldout subset; passing does not establish full752 accuracy gain or independent generalization.',
        elapsed_seconds=time.time() - started, pid=os.getpid())
    save(out / 'receipt.json', receipt)
    save(out / 'status.json', dict(state='complete', pid=os.getpid(), timestamp=time.time(),
        admission_state=decision['state'], direction_admitted=decision['admitted'], training_auto_started=False,
        actual_cost=counters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); c = json.loads(args.profile.read_text())
    try:run(c, args.out, args.profile)
    except BaseException as error:
        if args.out.exists():save(args.out / 'status.json', dict(state='failed', pid=os.getpid(), error=repr(error), timestamp=time.time()))
        raise


if __name__ == '__main__':main()
