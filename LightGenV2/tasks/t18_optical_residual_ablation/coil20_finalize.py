"""Seal all validation choices before eight once-only selected-EMA test evaluations."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPUS = ['GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
    'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b', 'GPU-1b963983-7909-af6e-0528-f0f0661ab549']
KEYS = [f'L{depth}_{rho}' for depth in [2,4,8,16] for rho in ['rho0','rho03']]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream: json.dump(value, stream, indent=2)
    else:
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def verify_arm(arm):
    result = read(arm/'fit_result.json'); history = read(arm/'history.json')
    metadata = read(arm/'metadata.json')
    assert result['test_evaluations'] == 0 and 'test' not in result['metrics']
    assert not result['test_used_for_selection'] and result['selection_split'] == 'val'
    assert all('test' not in row for row in history)
    best = min(history, key=lambda row: row['val']['balanced_nll'])
    assert best['epoch'] == result['selected_epoch'] and best['val'] == result['metrics']['val']
    assert sha(arm/'history.json') == result['history_sha256']
    assert sha(arm/'best_checkpoint.pt') == result['checkpoint_sha256']
    assert result['config'] == read(arm/'config.json') and result['sources'] == metadata['sources']
    for path, digest in result['sources'].items():
        blob = subprocess.check_output(['git', 'show', metadata['git_commit']+':'+path], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest
    return result


def seal(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert read(args.base/'status.json')['state'] == 'fit_complete_awaiting_validation_review'
    assert read(args.adjust/'status.json')['state'] == 'fit_complete_awaiting_validation_review'
    assert read(args.base/'current_processes.json')['active'] == []
    assert read(args.adjust/'current_processes.json')['active'] == []
    manifest = read(args.data.parent/'data_manifest.json')
    assert sha(args.data) == manifest['cache_sha256']
    arms = read(args.base/'fit_manifest.json')['arms']
    assert set(arms) == set(KEYS)
    # Audit every candidate from saved PT/CSV/history, with CUDA unavailable. No model inference.
    audits, choices = {}, {}
    for depth in [2,4,8,16]:
        options = {'base': [Path(arms[f'L{depth}_{name}']) for name in ['rho0','rho03']]}
        if depth == 16: options['low_variance'] = [args.adjust/name for name in ['rho0','rho03']]
        candidate_info = {}
        for candidate, pair in options.items():
            results = [verify_arm(arm) for arm in pair]
            expected_epochs = 10 if candidate == 'low_variance' else 30
            assert all(r['epochs_completed'] == expected_epochs and r['data_sha256'] == manifest['cache_sha256'] for r in results)
            assert all(r['depth'] == depth for r in results)
            assert [r['rho'] for r in results] == [0.,.3]
            if candidate == 'low_variance':
                original = verify_arm(options['base'][0])['config']
                changed = results[0]['config']
                allowed = {'phase_init_raw_uniform','maximum_epochs','minimum_epochs','maximum_seconds',
                    'requested_epochs','adjustment_policy'}
                assert all(original[k] == changed[k] for k in set(original)&set(changed)-allowed)
                assert original['phase_init_raw_uniform']==1. and changed['phase_init_raw_uniform']==.1
            report = args.out/'candidate_audits'/f'L{depth}_{candidate}'
            cmd = [sys.executable, str(HERE/'audit_medicinal_pilot.py'), '--run', str(args.base),
                '--depth', str(depth), '--arm0', str(pair[0]), '--arm03', str(pair[1]),
                '--data', str(args.data), '--out', str(report)]
            with (args.out/f'audit_L{depth}_{candidate}.log').open('w') as log:
                subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT, check=True)
            assert read(report/'pilot_audit.json')['cpu_checks_passed']
            audits[f'L{depth}_{candidate}'] = sha(report/'pilot_audit.json')
            candidate_info[candidate] = dict(arms=[str(p.resolve()) for p in pair],
                epochs=expected_epochs, mean_best_val_balanced_nll=sum(r['metrics']['val']['balanced_nll'] for r in results)/2,
                fit_result_sha256=[sha(p/'fit_result.json') for p in pair],
                best_checkpoint_sha256=[sha(p/'best_checkpoint.pt') for p in pair])
        chosen = 'base'
        if depth == 16 and candidate_info['low_variance']['mean_best_val_balanced_nll'] < candidate_info['base']['mean_best_val_balanced_nll']:
            chosen = 'low_variance'
        choices[str(depth)] = dict(selected_candidate=chosen, candidates=candidate_info,
            rule='Mean of paired best EMA val balanced NLL; exact tie retains base; no test')
    assert not (args.out/'global_selection_lock.json').exists(), 'Never replace a sealed choice'
    entries = {}
    for depth in [2,4,8,16]:
        decision = choices[str(depth)]
        pair_lock = args.out/f'pair_lock_L{depth}.json'
        save(pair_lock, decision, exclusive=True)
        for key, arm in zip([f'L{depth}_rho0', f'L{depth}_rho03'], decision['candidates'][decision['selected_candidate']]['arms']):
            path = Path(arm); result = verify_arm(path)
            entry = dict(arm=arm, depth=depth, rho=result['rho'], candidate=decision['selected_candidate'],
                selected_epoch=result['selected_epoch'], selected_state_kind='ema', selection_split='val',
                test_evaluations_before_lock=0, test_used_for_selection=False,
                pair_lock_sha256=sha(pair_lock), fit_result_sha256=sha(path/'fit_result.json'),
                checkpoint_sha256=sha(path/'best_checkpoint.pt'), history_sha256=sha(path/'history.json'),
                config_sha256=sha(path/'config.json'), metadata_sha256=sha(path/'metadata.json'),
                train_predictions_sha256=sha(path/'train_predictions.csv'), val_predictions_sha256=sha(path/'val_predictions.csv'))
            save(args.out/key/'selection_lock.json', entry, exclusive=True)
            entries[key] = dict(entry, selection_lock_sha256=sha(args.out/key/'selection_lock.json'))
    save(args.out/'global_selection_lock.json', dict(entries=entries, choices=choices, candidate_audits=audits,
        data_sha256=sha(args.data), manifest_sha256=sha(args.data.parent/'data_manifest.json'),
        lock_time_unix=time.time(), source_sha256=sha(Path(__file__)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        scope='Validation-selected exploratory known-object withheld-view evaluation; not unseen-instance generalization'), exclusive=True)
    save(args.out/'status.json', dict(state='selection_locked_test_not_started', test_evaluations=0))


def verify_global(out, data):
    global_path = out/'global_selection_lock.json'
    lock = read(global_path)
    assert set(lock['entries']) == set(KEYS), 'All eight choices required before test'
    assert lock['source_sha256'] == sha(Path(__file__)), 'Changed sealed finalizer source'
    assert sha(data) == lock['data_sha256'] and sha(data.parent/'data_manifest.json') == lock['manifest_sha256']
    for key, entry in lock['entries'].items():
        assert entry['selected_state_kind'] == 'ema' and entry['selection_split'] == 'val'
        assert entry['test_evaluations_before_lock'] == 0 and not entry['test_used_for_selection']
        local = out/key/'selection_lock.json'
        assert sha(local) == entry['selection_lock_sha256']
        assert read(local) == {k:v for k,v in entry.items() if k != 'selection_lock_sha256'}
        assert sha(out/f"pair_lock_L{entry['depth']}.json") == entry['pair_lock_sha256']
        choice = read(out/f"pair_lock_L{entry['depth']}.json")
        assert choice == lock['choices'][str(entry['depth'])]
        assert entry['candidate'] == choice['selected_candidate']
        selected_pair = choice['candidates'][choice['selected_candidate']]['arms']
        assert entry['arm'] == selected_pair[0 if entry['rho']==0. else 1]
        arm = Path(entry['arm'])
        for field, name in [('fit_result_sha256','fit_result.json'),('checkpoint_sha256','best_checkpoint.pt'),
                ('history_sha256','history.json'),('config_sha256','config.json'),('metadata_sha256','metadata.json'),
                ('train_predictions_sha256','train_predictions.csv'),('val_predictions_sha256','val_predictions.csv')]:
            assert sha(arm/name) == entry[field], f'Changed locked artifact {key}/{name}'
    return lock


def claim_test(out, key, global_digest):
    assert key in KEYS
    save(out/key/'test_started.json', dict(global_selection_lock_sha256=global_digest,
        pid=os.getpid(), time_unix=time.time(), policy='Exclusive marker; failure forbids automatic re-evaluation'), exclusive=True)


def worker(args):
    lock = verify_global(args.out, args.data)
    entry = lock['entries'][args.key]; arm = Path(entry['arm'])
    result = verify_arm(arm)
    import medicinal_pilot as pilot
    execution, o, torch = pilot.execution, pilot.o, pilot.torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.gpu and args.gpu in GPUS
    metadata = read(arm/'metadata.json')
    # Completed controller versions differ; evaluate() must be structurally identical.
    for path, digest in result['sources'].items():
        if path.endswith('/medicinal_pilot.py'):
            blob = subprocess.check_output(['git','show',metadata['git_commit']+':'+path], cwd=ROOT)
            def evaluation_ast(source):
                return ast.dump(next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='evaluate'), include_attributes=False)
            assert evaluation_ast(blob) == evaluation_ast((ROOT/path).read_bytes())
        else: assert sha(ROOT/path) == digest, f'Changed inference dependency {path}'
    o.f.gpu_budget.ALLOWED.update(GPUS)
    policy = o.f.gpu_budget.configure(torch, args.gpu, shared=True, memory_gib=3)
    torch.set_num_threads(4)
    profile = read(HERE/('coil20_L16_low_variance_profile.json' if entry['candidate']=='low_variance' else 'coil20_pilot_profile.json'))
    checkpoint = torch.load(arm/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['selected_state_kind']=='ema' and checkpoint['selection_split']=='val'
    assert checkpoint['epoch']==entry['selected_epoch'] and checkpoint['config']==result['config']
    assert checkpoint['sources']==result['sources'] and checkpoint['data_sha256']==lock['data_sha256']
    model = execution.build(entry['depth'], profile, result['config'], batched=result['config']['expert_batched'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    # This exclusive receipt is written before the first materialization of test arrays.
    claim_test(args.out, args.key, sha(args.out/'global_selection_lock.json'))
    manifest = read(args.data.parent/'data_manifest.json')
    with pilot.np.load(args.data, allow_pickle=False) as cache:
        images, labels, ids = [cache['test_'+name].copy() for name in ['images','labels','ids']]
    assert images.dtype==pilot.np.uint8 and images.shape==(360,150,150,3)
    assert ids.tolist()==manifest['split_ids']['test']
    assert pilot.np.bincount(labels,minlength=20).tolist()==manifest['supports']['test']
    data = (torch.from_numpy(images.transpose(0,3,1,2).copy()),torch.from_numpy(labels).long(),ids)
    metric, rows = pilot.evaluate(model, data, result['config']['evaluation_batch_size'])
    o.r.csvwrite(args.out/args.key/'test_predictions.csv', rows)
    save(args.out/args.key/'result.json', dict(depth=entry['depth'], rho=entry['rho'], candidate=entry['candidate'],
        selected_epoch=entry['selected_epoch'], selected_state_kind='ema', selection_split='val',
        metrics=dict(result['metrics'],test=metric), parameters=result['parameters'],
        test_evaluations=1, test_used_for_selection=False, global_selection_lock_sha256=sha(args.out/'global_selection_lock.json'),
        checkpoint_sha256=entry['checkpoint_sha256'], data_sha256=lock['data_sha256'],
        test_predictions_sha256=sha(args.out/args.key/'test_predictions.csv'),
        inference_source_sha256=sha(Path(__file__)), gpu_policy=policy, scope=lock['scope']))


def suite(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    verify_global(args.out,args.data)
    assert not any((args.out/key/'test_started.json').exists() for key in KEYS), 'No automatic test retries'
    pending, active, completed = list(reversed(KEYS)), {}, []
    try:
        while pending or active:
            for gpu, (child, log, key) in list(active.items()):
                code=child.poll()
                if code is None: continue
                log.close(); assert code==0, f'{key} test failed; do not retry'
                assert read(args.out/key/'result.json')['test_evaluations']==1
                completed.append(key); del active[gpu]
            for gpu in GPUS:
                if gpu in active or not pending: continue
                key=pending.pop(0); log=(args.out/key/'test.log').open('x')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','worker','--out',str(args.out),
                    '--data',str(args.data),'--key',key,'--gpu',gpu]
                child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(child,log,key)
            save(args.out/'current_processes.json',dict(active=[dict(key=k,gpu_uuid=g,pid=c.pid) for g,(c,l,k) in active.items()]))
            save(args.out/'status.json',dict(state='selected_test_running',completed=completed))
            if active: time.sleep(3)
        save(args.out/'status.json',dict(state='tests_complete_awaiting_cpu_audit',completed=completed))
    except BaseException as error:
        for child,log,key in active.values():
            if child.poll() is None: child.terminate()
        for child,log,key in active.values():
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
            log.close()
        save(args.out/'status.json',dict(state='test_failed_no_automatic_retry',error=repr(error)))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['lock','suite','worker'],required=True)
    parser.add_argument('--base',type=Path);parser.add_argument('--adjust',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--key',choices=KEYS);parser.add_argument('--gpu',choices=GPUS)
    args=parser.parse_args()
    if args.phase=='lock':
        args.out.mkdir(parents=True,exist_ok=False);seal(args)
    elif args.phase=='suite': suite(args)
    else: worker(args)


if __name__=='__main__': main()
