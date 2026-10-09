"""Sequential single-GPU D2NN depth follow-up using the unchanged historical model."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import kather2016_experiment as k
from kather2016_experiment import b, m, r, torch, np

PROFILE = Path(__file__).with_name('kather_depth_followup_20261009.json')
SPEC = r.read(PROFILE)
BASE_CONFIG = k.config
BASE_SOURCES = k.sources


def config(candidate):
    cfg = BASE_CONFIG('base')
    cfg.update(SPEC['candidates'][candidate])
    return cfg


def sources():
    src = BASE_SOURCES()
    for path in (Path(__file__), PROFILE):
        src[path.relative_to(b.TASK).as_posix()] = r.sha(path)
    return src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-contract', action='store_true')
    parser.add_argument('--candidate', choices=list(SPEC['candidates']))
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--select-from', type=Path, nargs='+')
    args = parser.parse_args()
    assert not args.candidate or args.validation_only
    # There is no worker pool: a single process trains every job in dependency order.
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    for candidate in SPEC['candidates']:
        cfg = config(candidate)
        base = BASE_CONFIG('base')
        assert cfg['detector'] == base['detector']
        assert cfg['encoding'] == base['encoding']
        assert cfg['phase_init_raw_uniform'] == base['phase_init_raw_uniform']
        assert cfg['capture_weight'] == base['capture_weight']
    if args.check_contract:
        print(json.dumps(dict(passed=True,configurations={c:config(c) for c in SPEC['candidates']})))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    src = sources()
    r.save(args.out/'metadata.json', dict(command=sys.argv, pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=m.environment(), sources=src, specification=SPEC,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], data_sha256=r.sha(args.data),
        test_previously_observed=True, test_read=False, time=r.now()))
    entries = []
    try:
        if args.select_from:
            for folder in args.select_from:
                meta = r.read(folder/'metadata.json')
                assert meta['sources']==src and meta['data_sha256']==r.sha(args.data)
                assert r.read(folder/'status.json')['state']=='validation_complete'
                entries.extend(r.read(folder/'validation_results.json'))
            assert sorted((e['candidate'],e['result']['depth'],e['result']['seed'])
                for e in entries)==sorted((c,d,17) for c in SPEC['candidates'] for d in SPEC['depths'])
            means={c:float(np.mean([e['result']['metrics']['val']['balanced_nll']
                for e in entries if e['candidate']==c])) for c in SPEC['candidates']}
            chosen=min(means,key=means.get)
            r.save(args.out/'candidate_selection.json',dict(chosen=chosen,
                mean_validation_balanced_nll=means,criterion=SPEC['selection'],entries=entries))
            selected=[e for e in entries if e['candidate']==chosen]
            r.save(args.out/'selection_lock.json',dict(entries=selected,config=config(chosen),
                candidate=chosen,sources=src,data_sha256=r.sha(args.data),time=r.now(),
                test_previously_observed=True,selection=SPEC['selection']))
            k.sources=sources
            k.evaluate(args)
            r.save(args.out/'status.json',dict(state='complete',time=r.now(),candidate=chosen,
                test_scope='exploratory_single_seed_followup',gpu_released_on_exit=True))
            return
        data = k.load_data(args.data,'train')
        val = k.load_data(args.data,'val')

        def train(candidate, depth, seed):
            out = args.out/'jobs'/f'{candidate}_d2nn_wide_L{depth}_seed{seed}'
            out.mkdir(parents=True,exist_ok=False)
            cfg = config(candidate)
            r.save(out/'metadata.json',dict(config=cfg, sources=src, depth=depth,
                seed=seed, arch='d2nn_wide', time=r.now(), test_read=False))
            r.save(args.out/'status.json',dict(state='training',candidate=candidate,
                depth=depth,seed=seed,time=r.now()))
            result = b.train('d2nn_wide',depth,seed,data,val,cfg,out,src)
            r.save(out/'result.json',result)
            entry = dict(result=result,folder=str((out/result['name']).resolve()),
                candidate=candidate,reused=False)
            entries.append(entry)
            return entry

        for candidate in ([args.candidate] if args.candidate else SPEC['candidates']):
            for depth in SPEC['depths']:
                train(candidate,depth,17)
        r.save(args.out/'validation_results.json',entries)
        if args.validation_only:
            r.save(args.out/'status.json',dict(state='validation_complete',time=r.now(),
                test_read=False,gpu_released_on_exit=True))
            return
        means = {c:float(np.mean([e['result']['metrics']['val']['balanced_nll']
            for e in entries if e['candidate']==c])) for c in SPEC['candidates']}
        chosen = min(means,key=means.get)
        r.save(args.out/'candidate_selection.json',dict(chosen=chosen,
            mean_validation_balanced_nll=means,criterion=SPEC['selection'],entries=entries))
        selected = [e for e in entries if e['candidate']==chosen]
        for seed in SPEC['seeds'][1:]:
            for depth in SPEC['depths']:
                selected.append(train(chosen,depth,seed))
        assert len(selected)==len(SPEC['depths'])*len(SPEC['seeds'])
        r.save(args.out/'selection_lock.json',dict(entries=selected,config=config(chosen),
            candidate=chosen,sources=src,data_sha256=r.sha(args.data),time=r.now(),
            test_previously_observed=True,selection=SPEC['selection']))
        del data,val
        torch.cuda.empty_cache()
        k.sources = sources
        k.evaluate(args)
        r.save(args.out/'status.json',dict(state='complete',time=r.now(),
            candidate=chosen,test_scope='exploratory_followup',gpu_released_on_exit=True))
    except Exception:
        r.save(args.out/'status.json',dict(state='failed',time=r.now(),
            traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
