"""Bounded Kather top-k tail follow-up, only the explicitly authorized GPU3."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_jobs(cfg, root):
    return [dict(j, out=str(root/f"N{j['experts']}_k{j['top_k']}_s{j['seed']}"))
            for j in cfg['jobs']]


def audit_source(job):
    p = Path(job['source'])
    m = json.loads((p/'metadata.json').read_text())
    r = json.loads((p/'result.json').read_text())
    for field in ('experts', 'top_k', 'seed'):
        if int(m['arguments'][field]) != job[field]:
            raise ValueError(f'Source {field} mismatch: {p}')
    expected_epochs = int(job.get('source_epochs', 60))
    if int(m['arguments']['epochs']) != expected_epochs or int(m['arguments']['layers']) != 4:
        raise ValueError(f'Expected {expected_epochs}-epoch, four-layer source: {p}')
    actual = sha(p/'best_checkpoint.pt')
    if actual != job['best_sha256'] or actual != r['checkpoint_sha256']:
        raise ValueError(f'Source checkpoint mismatch: {p}')
    if m.get('training_profile') or not (p/'last_checkpoint.pt').exists():
        raise ValueError(f'Expected legacy source with optimizer: {p}')
    last_sha = sha(p/'last_checkpoint.pt')
    if job.get('last_sha256') and last_sha != job['last_sha256']:
        raise ValueError(f'Source last-checkpoint mismatch: {p}')
    if expected_epochs != 60:
        if not job.get('last_sha256'):
            raise ValueError('Chained continuation requires a locked last-checkpoint SHA')
        history = json.loads((p/'history.json').read_text())
        if not history or int(history[-1]['epoch']) != expected_epochs:
            raise ValueError(f'Source history does not reach {expected_epochs}: {p}')
    return dict(source=str(p.resolve()), best_sha256=actual,
                last_sha256=last_sha, data_sha256=m['data_sha256'], source_epochs=expected_epochs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.profile.read_text())
    if cfg['gpu'] != 3 or cfg['slots'] not in (1, 2):
        raise ValueError('Only GPU3, at most two concurrent children')
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(cfg, a.out)
    audits = [audit_source(j) for j in jobs]
    save(a.out/'plan.json', dict(profile=cfg, audits=audits, profile_sha256=sha(a.profile),
         command=sys.argv, pid=os.getpid(), commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         published_source_commit=os.environ.get('T10_SOURCE_COMMIT'),
         source_identity='Published T10 files checked against Git blobs; host HEAD may be older',
         source_files={str(p):sha(p) for p in [Path(__file__), Path(__file__).with_name('diagnose_topk.py')]}))
    uuid = subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if cfg.get('gpu_uuid') and uuid != cfg['gpu_uuid']:
        raise ValueError('GPU3 UUID differs from the authorized device')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    pending, complete, running, pids = list(jobs), [], {}, []

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or running:
            while pending and len(running) < cfg['slots']:
                free = int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free',
                                                   '--format=csv,noheader,nounits'],text=True).strip())
                if free < cfg['min_free_mib_to_start']:
                    break
                j = pending.pop(0)
                cmd = [sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.diagnose_topk',
                       '--source',j['source'],'--out',j['out'],'--epochs',str(cfg['extra_epochs']),
                       '--lr',str(cfg['lr']),'--batch',str(cfg['microbatch']),
                       '--margin-weight',str(cfg['margin_weight']),'--capture-weight',str(cfg['capture_weight']),
                       '--selection-metric','accuracy','--route-audit','--defer-test']
                log = (a.out/(Path(j['out']).name+'.log')).open('w')
                proc = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[proc.pid] = (proc,log,j); pids.append(proc.pid)
            for pid,(proc,log,j) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close();del running[pid]
                if proc.returncode:
                    raise RuntimeError(f'Child failed {proc.returncode}: {j}')
                complete.append(j)
            save(a.out/'status.json',dict(state='training',pending=pending,complete=complete,
                 running={str(pid):j for pid,(_,_,j) in running.items()},gpu_uuid=uuid))
            if pending or running:
                time.sleep(10)
        locks = []
        for j in jobs:
            p=Path(j['out']);r=json.loads((p/'result.json').read_text());h=sha(p/'best_checkpoint.pt')
            if h != r['checkpoint_sha256']:
                raise ValueError('Selected model changed')
            locks.append(dict(run_dir=str(p.resolve()),checkpoint_sha256=h))
        lock=a.out/'all_models_test_lock.json'
        save(lock,dict(role='development_followup_after_observed_test',selection='validation_accuracy_including_original_best',runs=locks))
        rows=[]
        for j in jobs:
            save(a.out/'status.json',dict(state='testing',current=j,complete=complete))
            p=Path(j['out']);m=json.loads((p/'metadata.json').read_text())
            proc=subprocess.Popen([sys.executable,'-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test',
                 '--run',str(p),'--data',m['arguments']['data'],'--selection-lock',str(lock),'--batch',str(cfg['microbatch'])],
                 env=env,start_new_session=True)
            pids.append(proc.pid);running[proc.pid]=(proc,None,j)
            if proc.wait():raise RuntimeError(f'Test failed: {j}')
            running.clear()
            r=json.loads((p/'result.json').read_text());t=json.loads((p/'test_result.json').read_text())
            rows.append(dict(**j,best_epoch=r['best_epoch'],train=r['train'],val=r['val'],test=t['test'],checkpoint_sha256=r['checkpoint_sha256']))
        groups=[]
        for n,k in cfg['pairs']:
            group=[r for r in rows if r['experts']==n and r['top_k']==k]
            groups.append(dict(experts=n,top_k=k,n_seeds=len(group),**{
                split:dict(mean=statistics.mean(r[split]['accuracy'] for r in group),
                           sample_sd=statistics.stdev(r[split]['accuracy'] for r in group) if len(group)>1 else None)
                for split in ('train','val','test')}))
        save(a.out/'summary.json',dict(role='development_followup_unequal_to_original_training_budget',rows=rows,groups=groups))
        save(a.out/'status.json',dict(state='complete',complete=complete,pending=[],running={}))
    except BaseException as error:
        save(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),complete=complete,pending=pending))
        raise
    finally:
        for proc,log,j in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if log is not None:log.close()
        save(a.out/'release_check.json',dict(own_children_exited=True,child_pids=pids,timestamp=time.time()))


if __name__ == '__main__':
    main()
