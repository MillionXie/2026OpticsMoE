"""Bounded 2/4/12/16 paired regularized fits; four allowed GPUs, no test."""
import argparse
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
        'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',
        'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',
        'GPU-1b963983-7909-af6e-0528-f0f0661ab549']
DEPTHS = [2, 4, 12, 16]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)
def idle(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    return gpu not in [r.strip() for r in rows.splitlines()]


def probe(a):
    import medicinal_pilot as pilot
    o, torch = pilot.o, pilot.torch
    profile, manifest = read(a.profile), read(a.data.parent/'data_manifest.json')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu and a.gpu in GPUS
    o.f.gpu_budget.ALLOWED.update(GPUS)
    policy = o.f.gpu_budget.configure(torch, a.gpu, shared=False, memory_gib=3)
    torch.set_num_threads(4)
    # Fit-data loader accepts only train/val; no test arrays are opened.
    data = pilot.load_fit_data(a.data, 'train', manifest)
    weights = (len(data[1])/(20*torch.bincount(data[1], minlength=20).float())).cuda()
    a.out.mkdir(parents=True, exist_ok=False)
    results = []
    for rho in (0., .3):
        cfg = o.config(profile, manifest, rho, 'base')
        for micro in (1, 2, 4):
            started = time.monotonic()
            try:
                row = pilot.numerical_gate(data, weights, profile, cfg, True, micro, a.depth)
            except (AssertionError, torch.cuda.OutOfMemoryError) as error:
                row = dict(passed=False, depth=a.depth, rho=rho, expert_batched=True, microbatch=micro, error=repr(error))
                torch.cuda.empty_cache()
            row.update(rho=rho, seconds=time.monotonic()-started)
            results.append(row)
            save(a.out/'probe.json', dict(results=results, gpu_policy=policy, test_evaluations=0,
                scope='Fresh actual MixUp soft-target train-only objective against serial micro1; failed modes retained, thresholds unchanged'))
            print(json.dumps(row), flush=True)
    assert all(any(r['passed'] and r['rho'] == rho for r in results) for rho in (0., .3))


def suite(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    profile, manifest = read(a.profile), read(a.data.parent/'data_manifest.json')
    assert profile['depths'] == DEPTHS and len(manifest['classes']) == 20
    assert sha(a.data) == manifest['cache_sha256']
    assert manifest['preparation_source_sha256'] == sha(HERE/'prepare_caltech_selected.py')
    assert all(idle(g) for g in GPUS), 'No other project is killed or displaced'
    a.out.mkdir(parents=True, exist_ok=False)
    metadata = dict(pid=os.getpid(), command=sys.argv, profile_sha256=sha(a.profile), data_sha256=sha(a.data),
        manifest_sha256=sha(a.data.parent/'data_manifest.json'), plan_sha256=sha(HERE/'caltech_selected_plan.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        gpu_uuids=GPUS, maximum_epochs=100, maximum_seconds_per_arm=36000, test_evaluations=0)
    save(a.out/'metadata.json', metadata)
    active = {}
    for depth, gpu in zip([12,16,2,4], GPUS):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--phase','probe', '--depth',str(depth), '--gpu',gpu,
            '--data',str(a.data),'--profile',str(a.profile),'--out',str(a.out/f'probe_L{depth}')]
        with (a.out/f'probe_L{depth}.log').open('x') as log:
            proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[depth] = proc
    save(a.out/'current_processes.json', dict(active=[dict(depth=d,pid=p.pid,phase='probe',gpu_uuid=GPUS[[12,16,2,4].index(d)]) for d,p in active.items()], pending=[]))
    save(a.out/'status.json',dict(state='numerical_preflight',test_evaluations=0))
    codes = {d:p.wait() for d,p in active.items()}
    if any(codes.values()):
        save(a.out/'status.json',dict(state='preflight_failed_no_training',exit_codes=codes,test_evaluations=0))
        raise RuntimeError(codes)
    results = [r for d in DEPTHS for r in read(a.out/f'probe_L{d}/probe.json')['results']]
    modes = {d:max(m for m in (1,2,4) if all(any(r['passed'] and r['depth']==d and r['rho']==rho and r['microbatch']==m for r in results) for rho in (0.,.3))) for d in DEPTHS}
    save(a.out/'execution_probe.json',dict(results=results, common_pair_modes=modes, test_evaluations=0))
    import medicinal_pilot as pilot
    sources = pilot.o.source_identity(a.profile)
    for name in ('fast_optical.py','medicinal_pilot.py','prepare_caltech_selected.py','prepare_caltech50.py','probe_fast_optical.py','caltech_selected_queue.py','caltech_selected_plan.json'):
        sources[str((HERE/name).relative_to(ROOT)).replace('\\','/')] = sha(HERE/name)
    def verify():
        for path, digest in sources.items():
            assert sha(ROOT/path)==digest
            assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)).hexdigest()==digest
    verify()
    metadata.update(sources=sources, modes=modes); save(a.out/'metadata.json',metadata)
    jobs = [dict(depth=d,rho=r,name=f'L{d}_'+('rho0' if r==0 else 'rho03'),micro=modes[d]) for d in (12,16,4,2) for r in (0.,.3)]
    pending, active, completed, failures, arms = jobs[:], {}, [], [], {}
    started = time.time()
    while active or (pending and not failures):
        for gpu,(proc,job) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            try:
                assert code==0, code
                result=read(a.out/job['name']/'fit_result.json')
                assert result['epochs_completed']==100 and result['stop_reason']=='maximum_epochs' and result['test_evaluations']==0
                arms[job['name']]=str((a.out/job['name']).resolve())
                completed.append(dict(job,exit_code=code))
            except Exception as error: failures.append(dict(job,exit_code=code,error=repr(error)))
            del active[gpu]
        if not failures:
            for gpu in GPUS:
                if gpu in active or not pending or not idle(gpu): continue
                verify(); job=pending.pop(0)
                cmd=[sys.executable,str(HERE/'medicinal_pilot.py'),'--out',str(a.out/job['name']),'--data',str(a.data),
                    '--profile',str(a.profile),'--probe',str(a.out/'execution_probe.json'),'--gpu',gpu,'--depth',str(job['depth']),
                    '--rho',str(job['rho']),'--micro',str(job['micro']),'--batched','1','--epochs','100','--seconds','36000','--full-budget']
                with (a.out/(job['name']+'.log')).open('x') as log:
                    proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(pid=proc.pid,gpu_uuid=gpu,command=cmd); active[gpu]=(proc,job)
        save(a.out/'current_processes.json',dict(active=[j for p,j in active.values()],pending=pending))
        save(a.out/'status.json',dict(state='fit_failure_draining' if failures else 'fit_running', completed=completed,failures=failures,
            pending=len(pending),active=len(active),seconds=time.time()-started,test_evaluations=0))
        if active or (pending and not failures): time.sleep(5)
    save(a.out/'fit_manifest.json',dict(arms=arms,data_sha256=sha(a.data),test_evaluations=0))
    save(a.out/'status.json',dict(state='fit_failed_no_automatic_retry' if failures else 'fit_complete_awaiting_validation_review',
        completed=completed,failures=failures,active=0,pending=pending,seconds=time.time()-started,test_evaluations=0))
    if failures: raise RuntimeError(failures)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['probe','suite'],required=True)
    for key in ('out','data','profile'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu');p.add_argument('--depth',type=int,choices=DEPTHS)
    a=p.parse_args(); probe(a) if a.phase=='probe' else suite(a)
