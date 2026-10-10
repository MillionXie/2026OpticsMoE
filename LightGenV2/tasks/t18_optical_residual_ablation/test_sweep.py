"""Explicitly authorized test-selected DEVELOPMENT sweep, not held-out evaluation.

Read only retained Mango L6 rho=.3 best/last states. No training or ensembling.
Audit published source identities and reuse existing same-state test predictions.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISIONS = ['e6b125e9b', '65917810b', '7cc822e81', 'b149b3e88',
             'a438155fd', 'fad46834a', 'c06ba28b8']
RUNS = ['mango_variety_s17_20261010_3gpu',
        'mango_variety_s17_e100_20261010_3gpu',
        'mango_variety_s17_e100_lr3_20261010_3gpu',
        'mango_rho03_continue50_20261010_3gpu',
        'mango_rho03_requested_gpu1_gpu4_20261010',
        'mango_rho03_L6_continue50_round2_20261010',
        'mango_rho03_L6_aug_ema_20261010',
        'mango_rho03_L6_capture_smooth_20261010',
        'mango_rho03_L6_lowlr_ls_20261010']
ALLOWED_GPUS = {'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
                'GPU-1b963983-7909-af6e-0528-f0f0661ab549'}
DATA_SHA = 'd9369d96501efec883fcb97a09c51bff99f5ab12da49f14cb593038447e33290'


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(x.dtype), list(x.shape)],
                            separators=(',', ':')).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def check_predictions(path, metrics, manifest):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ids = manifest['split_ids']['test']
    assert [q['sample_id'] for q in rows] == ids
    labels = [int(q['label_true']) for q in rows]
    cm = [[0] * 8 for _ in range(8)]
    for q, y in zip(rows, labels):
        scores = [float(q[f'score{k}']) for k in range(8)]
        pred = max(range(8), key=lambda k: scores[k])
        assert pred == int(q['label_pred'])
        cm[y][pred] += 1
    assert cm == metrics['confusion_matrix']
    assert [sum(row) for row in cm] == manifest['supports']['test']
    assert metrics['n'] == len(rows) == 418
    assert abs(sum(cm[k][k] for k in range(8)) / len(rows) -
               metrics['accuracy']) < 1e-12
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--test-selection-authorized', action='store_true')
    a = p.parse_args()
    assert a.test_selection_authorized, 'Requires explicit user test-selection authorization'
    import run as t
    a.out.mkdir(parents=True, exist_ok=False)
    t.torch.set_num_threads(4)
    assert t.r.sha(a.data) == DATA_SHA
    manifest = t.r.read(a.data.parent / 'data_manifest.json')
    assert manifest['dataset'] == 'MangoLeafVarietyBD_raw_v2'
    assert manifest['cache_sha256'] == DATA_SHA
    src = t.source_identity()
    archived = {hashlib.sha256(subprocess.check_output([
        'git', 'show', rev + ':LightGenV2/tasks/t18_optical_residual_ablation/continue_residual.py'
    ])).hexdigest(): rev for rev in REVISIONS}
    # Published changes here only add scheduling/epoch/LR options, not inference.
    # Keep complete original identities; never relabel old checkpoints as current.
    originals = []
    for rev in ['bfd4718f0', '32a9d32fb', '261ac4fd0', '58147df49']:
        task = {name: hashlib.sha256(subprocess.check_output([
            'git', 'show', rev + ':LightGenV2/tasks/t18_optical_residual_ablation/' + name
        ])).hexdigest() for name in src['task']}
        assert task['model.py'] == src['task']['model.py']
        assert task['prepare_mango.py'] == src['task']['prepare_mango.py']
        originals.append((dict(historical=src['historical'], task=task), rev))
    states = []
    for run in RUNS:
        for path in sorted((HERE / 'runs/simulation' / run).rglob('*_checkpoint.pt')):
            if path.name not in ['best_checkpoint.pt', 'last_checkpoint.pt']:
                continue
            ck = t.torch.load(path, map_location='cpu', weights_only=False)
            cfg = ck['config']
            if ck['depth'] != 6 or cfg['residual_rho'] != .3:
                continue
            assert ck['arch'] == 'moe' and ck['seed'] == 17
            assert cfg['dataset'] == manifest['dataset'] and cfg['classes'] == manifest['classes']
            assert cfg['residual_scope'] == 'expert_and_global_only'
            assert cfg['expert_vectorize'] is False
            base = t.config(.3)
            for key in ['detector', 'encoding', 'phase_init_raw_uniform', 'residual_formula']:
                assert cfg[key] == base[key], (path, key)
            matches = [rev for identity, rev in originals if ck['sources'] == identity]
            if matches:
                revision = matches[0]
            else:
                assert ck['sources']['parent'] == src, path
                revision = archived[ck['sources']['continuation']]
            metadata = next((q / 'metadata.json' for q in path.parents
                             if (q / 'metadata.json').exists()), None)
            assert metadata is not None
            md = t.r.read(metadata)
            assert md['data_sha256'] == DATA_SHA and md['sources'] == ck['sources']
            assert md['config'] == cfg
            keys = ['model'] if path.name == 'best_checkpoint.pt' else ['model', 'ema']
            for key in keys:
                states.append(dict(checkpoint=str(path), checkpoint_sha256=t.r.sha(path),
                    state_key=key, state_sha256=state_sha(ck[key]), epoch=ck['epoch'],
                    kind='best_ema' if path.name.startswith('best') else 'last_' + key,
                    profile=cfg.get('training_profile', 'base'), training_source_revision=revision,
                    config=cfg, training_sources=ck['sources']))
            del ck
    assert len(states) == 36, f'Expected 12 retained arms x 3 states, found {len(states)}'
    audit = dict(scope='TEST-SELECTED DEVELOPMENT METRIC, NOT INDEPENDENT GENERALIZATION',
        user_authorization='2026-10-10: directly select PT on test and find current maximum',
        selection='maximum test accuracy; ties minimum test balanced NLL then stable path/key',
        training=False, ensembling=False, no_residual_baseline_unchanged=True,
        retained_runs=RUNS, data_sha256=DATA_SHA, states=states,
        source_identity=src, evaluator_sha256=t.r.sha(Path(__file__)),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=sys.argv, pid=os.getpid(), time=t.r.now())
    t.r.save(a.out / 'manifest.json', audit)
    if a.audit_only:
        print(json.dumps(dict(audit_passed=True, states=len(states),
            unique_states=len({x['state_sha256'] for x in states}))))
        return
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in ALLOWED_GPUS
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    assert not any(gpu in line for line in procs.splitlines()), 'Requested physical GPU occupied'
    audit['gpu_uuid'] = gpu
    t.r.save(a.out / 'manifest.json', audit)
    known = {}
    by_file = {x['checkpoint_sha256']: x for x in states if x['kind'] == 'best_ema'}
    for run in RUNS:
        for path in sorted((HERE / 'runs/simulation' / run).rglob('metrics.json')):
            m = t.r.read(path)
            sha = m.get('checkpoint_sha256')
            if sha in by_file and 'test' in m:
                preds = path.parent / 'test_predictions.csv'
                check_predictions(preds, m['test'], manifest)
                state = by_file[sha]['state_sha256']
                assert state not in known, 'Duplicate historical test of same state'
                known[state] = dict(test=m['test'], predictions=str(preds),
                                    reused_from=str(path))
    data = t.k.load_data(a.data, 'test')
    results = []
    for i, item in enumerate(states):
        identity = item['state_sha256']
        if identity in known:
            record = dict(item, **known[identity], inference='reused_same_state')
        else:
            assert t.r.sha(Path(item['checkpoint'])) == item['checkpoint_sha256']
            ck = t.torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
            assert state_sha(ck[item['state_key']]) == identity
            model = t.build('moe', 6, ck['config'])
            model.load_state_dict(ck[item['state_key']], strict=True)
            assert sum(x.numel() for x in model.parameters()) == 1329544
            tm, rows = t.b.evaluate(model, data, 'moe', ck['config']['batch_size'])
            preds = a.out / f'state_{i:02d}_test_predictions.csv'
            t.r.csvwrite(preds, rows)
            check_predictions(preds, tm, manifest)
            known[identity] = dict(test=tm, predictions=str(preds), reused_from=None)
            record = dict(item, **known[identity], inference='new_test_once')
            del model, ck
            gc.collect()
            t.torch.cuda.empty_cache()
        results.append(record)
        t.r.save(a.out / 'progress.json', dict(completed=len(results), total=len(states)))
        print(f"{i+1}/{len(states)} {item['kind']} test={record['test']['accuracy']:.6f}", flush=True)
    ordered = sorted(results, key=lambda x: (-x['test']['accuracy'],
        x['test']['balanced_nll'], x['checkpoint'], x['state_key']))
    winner = ordered[0]
    t.r.save(a.out / 'results.json', dict(scope=audit['scope'], manifest_sha256=t.r.sha(a.out/'manifest.json'),
        winner=winner, results=results, states=len(states), unique_states=len(known),
        new_inferences=sum(x['inference']=='new_test_once' for x in results),
        gpu_uuid=gpu, time=t.r.now()))
    # The selected artifact retains genuine training sources/config and explicit state provenance.
    ck = t.torch.load(winner['checkpoint'], map_location='cpu', weights_only=False)
    export = {k: ck[k] for k in ['epoch', 'arch', 'depth', 'seed', 'sources', 'config']}
    export.update(model=ck[winner['state_key']], test_selection=dict(scope=audit['scope'],
        source_checkpoint=winner['checkpoint'], source_checkpoint_sha256=winner['checkpoint_sha256'],
        source_state_key=winner['state_key'], state_sha256=winner['state_sha256'],
        metrics=winner['test'], manifest_sha256=t.r.sha(a.out/'manifest.json')))
    t.r.save_torch(a.out/'best_checkpoint.pt', export)
    t.r.save(a.out/'selection.json', dict(winner=winner, selected_artifact=str(a.out/'best_checkpoint.pt'),
        selected_artifact_sha256=t.r.sha(a.out/'best_checkpoint.pt'), scope=audit['scope']))
    t.r.save(a.out/'status.json', dict(state='complete', training=False, time=t.r.now()))
    print(json.dumps(dict(accuracy=winner['test']['accuracy'], checkpoint=winner['checkpoint'],
                         state_key=winner['state_key'], epoch=winner['epoch'])), flush=True)


if __name__ == '__main__':
    main()
