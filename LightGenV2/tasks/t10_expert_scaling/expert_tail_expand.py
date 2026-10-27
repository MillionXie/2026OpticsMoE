"""Adopt the live GPU3 queue without restarting trainers; add authorized GPU1."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha


def task_key(job):
    return tuple(int(job[x]) for x in ('experts', 'top_k', 'seed'))


def validate_partition(snapshot, jobs):
    expected = {task_key(j) for j in jobs}
    groups = snapshot['pending']+snapshot['complete']+list(snapshot['running'].values())
    keys = [task_key(j) for j in groups]
    if len(expected) != len(jobs) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Handoff must partition the configured unique tasks exactly once')
    by_key = {task_key(j):j for j in jobs}
    for j in groups:
        original = by_key[task_key(j)]
        for field in ('source', 'out', 'best_sha256'):
            if j[field] != original[field]:
                raise ValueError(f'Handoff source/output identity changed: {field}')
        for field in ('source_epochs', 'last_sha256'):
            if j.get(field) != original.get(field):
                raise ValueError(f'Handoff continuation identity changed: {field}')


def process_matches(pid, module, out):
    try:
        raw = Path('/proc', str(pid), 'cmdline').read_bytes()
    except FileNotFoundError:
        return False
    args = raw.decode().split(chr(0))
    return module in args and str(out) in args


def alive(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z'
    except FileNotFoundError:
        return False


def launch_command(job, cfg):
    return [sys.executable, '-u', '-m', 'LightGenV2.tasks.t10_expert_scaling.diagnose_topk',
            '--source', job['source'], '--out', job['out'], '--epochs', str(cfg['extra_epochs']),
            '--lr', str(cfg['lr']), '--batch', str(cfg['microbatch']),
            '--margin-weight', str(cfg['margin_weight']), '--capture-weight', str(cfg['capture_weight']),
            '--selection-metric', 'accuracy', '--route-audit', '--defer-test']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((a.out/'plan.json').read_text())
    cfg = plan['profile']
    snapshot = json.loads(a.snapshot.read_text())
    jobs = [dict(j, out=str(a.out/f"N{j['experts']}_k{j['top_k']}_s{j['seed']}")) for j in cfg['jobs']]
    validate_partition(snapshot, jobs)
    parent = int(plan['pid'])
    if not process_matches(parent, 'LightGenV2.tasks.t10_expert_scaling.expert_tail_schedule', a.out):
        raise ValueError('Original scheduler identity changed')
    pending, complete = list(snapshot['pending']), list(snapshot['complete'])
    running, pids = {}, []
    adopted = False
    uuids = {gpu:subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip() for gpu in (1,3)}
    if uuids[1] != 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d':
        raise ValueError('GPU1 identity changed')
    if uuids[3] != 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd':
        raise ValueError('GPU3 identity changed')
    source_commit = plan['published_source_commit']
    controller_commit = os.environ['T10_CONTROLLER_COMMIT']
    root = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    own = Path(__file__).resolve()
    if subprocess.check_output(['git','show',controller_commit+':'+own.relative_to(root).as_posix()]) != own.read_bytes():
        raise ValueError('Controller differs from published Git object')

    def env_for(gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=uuids[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   T10_SOURCE_COMMIT=source_commit, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        return env

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for pid, j in snapshot['running'].items():
            pid = int(pid)
            if not process_matches(pid, 'LightGenV2.tasks.t10_expert_scaling.diagnose_topk', j['out']):
                if (Path(j['out'])/'result.json').exists():
                    complete.append(j)
                    continue
                raise ValueError(f'Inherited trainer not identifiable: {pid}')
            running[pid] = dict(proc=None, log=None, job=j, gpu=3)
            pids.append(pid)
        # The caller freezes the old scheduler. Killing only this verified parent
        # bypasses its finally block; the original trainer sessions stay alive.
        state = Path('/proc',str(parent),'stat').read_text().rsplit(')',1)[1].split()[0]
        if state not in ('T','t'):
            raise ValueError('Old scheduler must be frozen before transfer')
        os.kill(parent, signal.SIGKILL)
        adopted = True
        save(a.out/'handoff_ack.json',dict(pid=os.getpid(),previous_scheduler=parent,adopted_pids=pids,
             controller_commit=controller_commit,training_source_commit=source_commit,gpus=uuids,
             snapshot_sha256=sha(a.snapshot),timestamp=time.time()))
        while pending or running:
            for gpu in (1,3):
                while pending and sum(v['gpu']==gpu for v in running.values()) < 2:
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                                                      '--format=csv,noheader,nounits'],text=True).strip())
                    if free < cfg['min_free_mib_to_start']:
                        break
                    job=pending.pop(0)
                    if Path(job['out']).exists():
                        raise ValueError('Refusing duplicate output: '+job['out'])
                    log=(a.out/(Path(job['out']).name+'.log')).open('x')
                    proc=subprocess.Popen(launch_command(job,cfg),env=env_for(gpu),stdout=log,
                                          stderr=subprocess.STDOUT,start_new_session=True)
                    pids.append(proc.pid)
                    running[proc.pid]=dict(proc=proc,log=log,job=job,gpu=gpu)
            for pid, item in list(running.items()):
                proc=item['proc']
                if (proc is None and alive(pid)) or (proc is not None and proc.poll() is None):
                    continue
                if item['log']:item['log'].close()
                if ((proc is not None and proc.returncode) or
                    not (Path(item['job']['out'])/'result.json').exists()):
                    raise RuntimeError(f'Trainer failed: {pid}, {item["job"]}')
                complete.append(item['job']);del running[pid]
            save(a.out/'status.json',dict(state='training',scheduler_pid=os.getpid(),pending=pending,
                 complete=complete,running={str(pid):v['job'] for pid,v in running.items()},
                 running_gpu={str(pid):v['gpu'] for pid,v in running.items()},gpus=uuids))
            if pending or running:time.sleep(10)
        locks=[]
        for job in jobs:
            p=Path(job['out']);r=json.loads((p/'result.json').read_text());digest=sha(p/'best_checkpoint.pt')
            if digest != r['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            locks.append(dict(run_dir=str(p.resolve()),checkpoint_sha256=digest))
        lock=a.out/'all_models_test_lock.json'
        save(lock,dict(role='development_followup_after_observed_test',selection='validation_accuracy_including_original_best',runs=locks))
        rows=[]
        for job in jobs:
            save(a.out/'status.json',dict(state='testing',scheduler_pid=os.getpid(),current=job,complete=complete,gpus=uuids))
            p=Path(job['out']);m=json.loads((p/'metadata.json').read_text())
            proc=subprocess.Popen([sys.executable,'-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test',
                 '--run',str(p),'--data',m['arguments']['data'],'--selection-lock',str(lock),
                 '--batch',str(cfg['microbatch'])],env=env_for(1),start_new_session=True)
            pids.append(proc.pid);running[proc.pid]=dict(proc=proc,log=None,job=job,gpu=1)
            if proc.wait():raise RuntimeError('Test failed: '+job['out'])
            running.clear()
            r=json.loads((p/'result.json').read_text());t=json.loads((p/'test_result.json').read_text())
            rows.append(dict(**job,best_epoch=r['best_epoch'],train=r['train'],val=r['val'],test=t['test'],checkpoint_sha256=r['checkpoint_sha256']))
        groups=[]
        for n,k in cfg['pairs']:
            group=[r for r in rows if r['experts']==n and r['top_k']==k]
            groups.append(dict(experts=n,top_k=k,n_seeds=len(group),**{
                split:dict(mean=statistics.mean(r[split]['accuracy'] for r in group),
                           sample_sd=statistics.stdev(r[split]['accuracy'] for r in group))
                for split in ('train','val','test')}))
        save(a.out/'summary.json',dict(role='development_followup_unequal_to_original_training_budget',rows=rows,groups=groups))
        save(a.out/'status.json',dict(state='complete',complete=complete,pending=[],running={},scheduler_pid=os.getpid()))
    except BaseException as error:
        if adopted:
            save(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),complete=complete,pending=pending))
        else:
            os.kill(parent,signal.SIGCONT)
        raise
    finally:
        if adopted:
            for pid,item in running.items():
                if process_matches(pid,'LightGenV2.tasks.t10_expert_scaling.diagnose_topk',item['job']['out']) or (
                        item['proc'] is not None and item['proc'].poll() is None):
                    os.killpg(pid,signal.SIGTERM)
                    deadline=time.time()+20
                    while alive(pid) and time.time()<deadline:time.sleep(.1)
                    if alive(pid):os.killpg(pid,signal.SIGKILL)
                if item['log']:item['log'].close()
            save(a.out/'release_check.json',dict(own_children_exited=all(not alive(pid) for pid in pids),
                 child_pids=pids,timestamp=time.time(),gpus=uuids))


if __name__=='__main__':
    main()
