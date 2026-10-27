"""Select among preserved SAM/control best and last PTs by authorized development TEST.

The active training code is never modified. This CPU waiter imports Torch only
after the training supervisor and its workers have exited and released GPU1.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .expert_tail_schedule import save, sha


def choose(rows):
    if not rows or any(not math.isfinite(r['test']['macro_nll']) for r in rows):
        raise ValueError('Missing or nonfinite development scores')
    return max(rows, key=lambda r: (r['test_correct'], -r['test']['macro_nll']))


def ready(status, release, alive):
    if status.get('state') == 'failed':
        raise RuntimeError('Training failed; do not silently select incomplete PTs')
    return (status.get('state') == 'complete' and not status.get('running')
            and release.get('children_reaped') is True
            and not any(alive(p) for p in [status['scheduler_pid'], *release['child_pids']]))


def model_hash(state):
    """Canonical tensor identity, independent of serialization/optimizer state."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def selected_payload(checkpoint, candidate):
    field = candidate.get('state_field', 'model')
    if field not in ['model', 'live_model'] or field not in checkpoint:
        raise ValueError('Requested retained state is unavailable')
    return dict(model=checkpoint[field], epoch=checkpoint['epoch'],
                round=checkpoint.get('round'), metadata=checkpoint['metadata'], power_fraction=1.,
                development_selection=dict(source_checkpoint=candidate['checkpoint'],
                    source_checkpoint_sha256=candidate['checkpoint_sha256'], state_field=field,
                    exported_weights_only_not_optimizer_resume=True))


def sources(profile, c):
    commit = os.environ['T10_TEST_SELECT_SOURCE_COMMIT']
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    paths = [Path(__file__), profile, *[Path(__file__).with_name(n) for n in ['model.py', 'train.py', 'plan.py']],
             Path(__file__).parent/'configs/study.json', repo/'LightGenV2/common/optical_contract.py']
    hashes = {}
    for p in paths:
        name = p.resolve().relative_to(repo).as_posix()
        if p.read_bytes() != subprocess.check_output(['git', 'show', commit+':'+name]):
            raise ValueError('Selector source differs from published object: '+name)
        hashes[name] = sha(p)
    plan = json.loads((Path(c['training_root'])/'plan.json').read_text())
    assert plan['source_commit'] == c['training_source_commit']
    for name, value in plan['source_files'].items():
        assert sha(repo/name) == value, 'Training source changed: '+name
    return commit, hashes


def run(profile, out):
    c = json.loads(profile.read_text())
    assert c['seed'] == 17 and c['recipes'] == ['hard_label', 'sam']
    assert c['checkpoint_names'] == ['best_checkpoint.pt', 'last_checkpoint.pt']
    fields = c.get('state_fields', ['model'])
    assert fields in [['model'], ['model', 'live_model']]
    assert c['target_correct'] == 616 and c['gpu_index'] == 1
    commit, hashes = sources(profile, c)
    root = Path(c['training_root'])
    out.mkdir(parents=True, exist_ok=False)
    save(out/'metadata.json', dict(pid=os.getpid(), source_commit=commit, source_files=hashes,
         configuration=c, host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         role='User explicitly requested direct TEST PT selection; development only'))
    started = time.time()
    try:
        while True:
            status = json.loads((root/'status.json').read_text())
            rp = root/'release_check.json'
            release = json.loads(rp.read_text()) if rp.exists() else {}
            if ready(status, release, lambda pid: Path('/proc',str(pid)).exists()):
                break
            if time.time()-started > c['wait_timeout_seconds']:
                raise TimeoutError('Bounded wait expired; training was not interrupted')
            save(out/'status.json', dict(state='waiting_for_training_release',pid=os.getpid(),
                 training_state=status['state'], timestamp=time.time(), no_gpu_inference=True))
            time.sleep(c['poll_seconds'])
        # Recheck byte identities after waiting; no active training PT can race us.
        sources(profile, c)
        previous = Path(c['previous_selection_run']) if c.get('previous_selection_run') else None
        known = []
        if previous:
            old_status=json.loads((previous/'status.json').read_text())
            assert old_status['state']=='complete' and not Path('/proc',str(old_status['pid'])).exists()
            known=json.loads((previous/'summary.json').read_text())['candidates']
        assert os.environ['CUDA_VISIBLE_DEVICES'] == c['gpu_uuid']
        assert subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip() == c['gpu_uuid']
        import numpy as np
        import torch
        from .model import ScalingOptics
        from .train import evaluate, data
        from .diagnose_topk import save_route_audit
        torch.set_num_threads(4)
        ref = Path(c['reference_run'])
        m = json.loads((ref/'metadata.json').read_text())
        cache = Path(m['arguments']['data'])
        mp = cache.parent/'data_manifest.json'
        manifest = json.loads(mp.read_text())
        assert sha(cache) == m['data_sha256'] == manifest['cache_sha256'] == c['data_sha256']
        assert sha(ref/'best_checkpoint.pt') == c['reference_sha256']
        with np.load(cache, allow_pickle=False) as z:
            x, y, ids = z['test_images'], z['test_labels'].astype(np.int64), z['test_ids'].astype(str)
        assert len(ids) == len(set(ids)) == 752 and ids.tolist() == manifest['split_ids']['test']
        assert set(ids).isdisjoint(manifest['split_ids']['train']) and set(ids).isdisjoint(manifest['split_ids']['val'])
        split = (torch.from_numpy(x.transpose(0,3,1,2).copy()), torch.from_numpy(y), ids)
        candidates = [dict(name='original_reference', directory=str(ref), checkpoint=str(ref/'best_checkpoint.pt'),state_field='model')]
        for recipe in c['recipes']:
            d = root/(recipe+'_s17')
            meta = json.loads((d/'metadata.json').read_text())
            assert meta['git_commit'] == c['training_source_commit'] and meta['data_sha256'] == c['data_sha256']
            assert meta['arguments']['seed'] == 17 and meta['arguments']['experts'] == meta['arguments']['top_k'] == 49
            assert len(json.loads((d/'history.json').read_text())) == 24
            for n in c['checkpoint_names']:
                for field in fields:
                    candidates.append(dict(name=recipe+'_'+n.replace('_checkpoint.pt','')+('_live' if field=='live_model' else ''),
                                           directory=str(d),checkpoint=str(d/n),state_field=field))
        for row in candidates:
            row['checkpoint_sha256'] = sha(Path(row['checkpoint']))
        save(out/'all_checkpoint_lock.json', dict(candidates=candidates, source_commit=commit,
             selection='Direct development TEST correct count, then TEST macro-NLL; VAL not used',
             candidate_scope='Original reference plus two recipes retained best/last states '+str(fields)+'; no unretained intermediate epochs',
             timestamp=time.time(), data_sha256=c['data_sha256'], manifest_sha256=sha(mp)))
        rows, identities = [], {}
        old_predictions = None
        for candidate in candidates:
            name = candidate['name']; d = out/name; d.mkdir()
            save(out/'status.json', dict(state='testing', pid=os.getpid(), current=name,
                 completed=[r['name'] for r in rows], timestamp=time.time()))
            checkpoint = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
            assert checkpoint.get('power_fraction',1.) == 1. and checkpoint.get('routing_temperature',.05) == .05
            state = checkpoint[candidate['state_field']]; identity = model_hash(state)
            counter = checkpoint.get('round',checkpoint['epoch'])
            duplicate = identities.get(identity)
            original = Path(candidate['directory'])
            old_path = original/'test_result.json'
            reuse = None
            cached = next((x for x in known if x['model_tensor_sha256']==identity),None)
            if duplicate:
                reuse = dict(from_candidate=duplicate['name'], identical_model_tensors=True)
                source_predictions = out/duplicate['name']/'test_predictions.npz'
                metrics = duplicate['test']
            elif cached:
                assert cached['data_sha256']==c['data_sha256']
                source_predictions=previous/cached['name']/'test_predictions.npz'
                assert sha(source_predictions)==cached['predictions_sha256']
                metrics=cached['test'];reuse=dict(from_previous_selection=str(previous),from_candidate=cached['name'],identical_model_tensors=True)
            elif candidate['state_field']=='model' and Path(candidate['checkpoint']).name == 'best_checkpoint.pt' and old_path.exists():
                old = json.loads(old_path.read_text())
                assert old['checkpoint_sha256'] == candidate['checkpoint_sha256'] and old['data_sha256'] == c['data_sha256']
                for n, value in old.get('evaluation_dependencies', {}).items():
                    assert sha(Path(__file__).with_name(n)) == value
                for n, value in old.get('source_files', {}).items():
                    assert sha(Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())/n) == value
                source_predictions = original/'test_predictions.npz'
                assert sha(source_predictions) == old['predictions_sha256']
                metrics = old['test']; reuse = dict(from_result=str(old_path), verified_checkpoint_data_dependencies=True)
            if reuse:
                shutil.copy2(source_predictions, d/'test_predictions.npz')
                save(d/'test_reuse_receipt.json',dict(**reuse,gpu_forward_executed=False,model_tensor_sha256=identity))
                with np.load(d/'test_predictions.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],y)
                    score_array = z['scores'].copy()
            else:
                meta = json.loads((original/'metadata.json').read_text()); a = meta['arguments']
                model = ScalingOptics(meta['protocol'],49,49,'moe_oeo',8,a['layers'],a['padding']).cuda()
                assert sum(p.numel() for p in model.parameters()) == 11078432
                model.load_state_dict(state)
                metrics, scores = evaluate(model,split,c['eval_batch']);score_array=scores.numpy()
                np.savez_compressed(d/'test_predictions.npz',ids=ids,labels=y,scores=score_array,predictions=score_array.argmax(1))
                del model,scores;torch.cuda.empty_cache()
            assert np.isfinite(score_array).all() and score_array.shape == (752,8)
            pred = score_array.argmax(1); correct = int((pred==y).sum())
            cm = np.bincount(y*8+pred,minlength=64).reshape(8,8)
            assert cm.tolist() == metrics['confusion_matrix'] and round(metrics['accuracy']*752) == correct
            if old_predictions is None:old_predictions=pred.copy()
            row = dict(**candidate,model_tensor_sha256=identity,counter=counter,test=metrics,test_correct=correct,
                 predictions_sha256=sha(d/'test_predictions.npz'),data_sha256=c['data_sha256'],power_fraction=1.,
                 role=c['role'],source_commit=commit,reused=bool(reuse),paired=dict(
                 corrected=int(((old_predictions!=y)&(pred==y)).sum()),new_wrong=int(((old_predictions==y)&(pred!=y)).sum())))
            save(d/'test_result.json',row);rows.append(row);identities[identity]=row
            print(json.dumps(dict(candidate=name,test_correct=correct,paired=row['paired'],reused=bool(reuse))),flush=True)
            del checkpoint,state
        winner = choose(rows)
        assert sha(Path(winner['checkpoint'])) == winner['checkpoint_sha256']
        if winner['state_field']=='model':
            shutil.copy2(winner['checkpoint'],out/'best_checkpoint.pt')
        else:
            selected=torch.load(winner['checkpoint'],map_location='cpu',weights_only=False)
            torch.save(selected_payload(selected,winner),out/'best_checkpoint.pt');del selected
        shutil.copy2(out/winner['name']/'test_predictions.npz',out/'selected_test_predictions.npz')
        # Final same-weight TRAIN/VAL and routes accompany the TEST-selected PT.
        metadata=json.loads((Path(winner['directory'])/'metadata.json').read_text());a=metadata['arguments']
        checkpoint=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        model=ScalingOptics(metadata['protocol'],49,49,'moe_oeo',8,a['layers'],a['padding']).cuda();model.load_state_dict(checkpoint['model'])
        splits,_=data(cache);final={}
        for label in ['train','val']:
            final[label],scores=evaluate(model,splits[label],c['eval_batch'])
            np.savez_compressed(out/('selected_'+label+'_predictions.npz'),ids=splits[label][2],labels=splits[label][1].numpy(),scores=scores.numpy())
            save_route_audit(model,splits[label],out/('selected_'+label+'_routes.npz'))
        save(out/'summary.json',dict(candidates=rows,selected=winner,**final,selection='TEST correct, TEST NLL tie-break; VAL not used',
             pilot_target_met=winner['test_correct']>=616,target_correct=616,three_seed_mean_target_established=False,
             checkpoint_sha256=sha(out/'best_checkpoint.pt'),role=c['role'],source_commit=commit))
        del model;torch.cuda.empty_cache()
        save(out/'status.json',dict(state='complete',pid=os.getpid(),selected=winner['name'],test_correct=winner['test_correct'],pilot_target_met=winner['test_correct']>=616))
    except BaseException as error:
        save(out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)));raise
    finally:
        save(out/'release_check.json',dict(pid=os.getpid(),gpu_uuid=c['gpu_uuid'],timestamp=time.time(),
             note='Independently verify actual process exit and GPU allocation; no training process was signaled'))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.profile,a.out)
