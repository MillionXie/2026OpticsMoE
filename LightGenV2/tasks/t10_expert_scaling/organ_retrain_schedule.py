"""Run one fixed Organ profile for k=1..4, seed17, at most three idle GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .train import save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs='+', default=[1, 3, 2, 0, 4, 5])
    ap.add_argument('--slots', type=int, default=2)
    ap.add_argument('--data', type=Path, default=Path('/DATA/DATA1/guest3/demo_reproduction_data/organcmnist_fixed/organcmnist_fixed_split.npz'))
    a = ap.parse_args()
    if 6 in a.gpus or a.slots < 1:
        raise ValueError('GPU6 excluded; slots must be positive')
    cfg = json.loads(a.profile.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    pending = list(cfg['top_k'])
    running, completed, failed, pids = {}, [], [], []

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(a.out/'identity.json', dict(pid=os.getpid(), command=sys.argv, profile=cfg,
         profile_sha256=sha(a.profile), commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    try:
        while pending or running:
            for gpu in a.gpus:
                if gpu == 6:
                    continue
                owned = [g for (g, _) in running if g == gpu]
                if not owned:
                    if len({g for g, _ in running}) >= 3 or not pending:
                        continue
                    used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                                       '--format=csv,noheader,nounits'],text=True).strip())
                    if used > 200:
                        continue
                for slot in range(a.slots):
                    if (gpu,slot) in running or not pending:
                        continue
                    k = pending.pop(0)
                    run = a.out/f'k{k}_s{cfg["seed"]}'
                    uuid = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid',
                                                     '--format=csv,noheader'],text=True).strip()
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
                    cmd = [sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.train',
                           '--data',str(a.data),'--out',str(run),'--arch','moe_oeo','--experts','4',
                           '--top-k',str(k),'--layers',str(cfg['layers']),'--epochs',str(cfg['epochs']),
                           '--lr',str(cfg['learning_rate']),'--microbatch',str(cfg['microbatch']),
                           '--seed',str(cfg['seed']),'--profile',str(a.profile)]
                    log = (a.out/f'k{k}.log').open('w')
                    proc = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    pids.append(proc.pid)
                    running[(gpu,slot)] = (proc,log,dict(k=k,run=str(run),pid=proc.pid,gpu_uuid=uuid))
            for key,(proc,log,job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[key]
                if proc.returncode:
                    failed.append(dict(**job,exit_code=proc.returncode))
                    raise RuntimeError(f'Training failed: {job}')
                completed.append(job)
            save(a.out/'status.json',dict(state='training',pending=pending,complete=completed,
                 running={str(g):j for g,(_,_,j) in running.items()},failed=failed))
            if pending or running:
                time.sleep(10)
        # Lock all four models before accessing any new TEST result.
        locked = []
        for job in sorted(completed,key=lambda j:j['k']):
            result = json.loads((Path(job['run'])/'result.json').read_text())
            locked.append(dict(run_dir=job['run'],checkpoint_sha256=result['checkpoint_sha256']))
        lock = a.out/'test_selection_lock.json'
        save(lock,dict(selection='validation_accuracy_tiebreak_macro_nll',role='development_followup',runs=locked))
        # Reuse an idle physical GPU, one evaluation process at a time.
        while True:
            for gpu in a.gpus:
                used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                                  '--format=csv,noheader,nounits'],text=True).strip())
                if used <= 200:
                    break
            else:
                save(a.out/'status.json',dict(state='waiting_for_idle_test_gpu',complete=completed))
                time.sleep(10)
                continue
            if used <= 200:
                break
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID')
        for job in sorted(completed,key=lambda j:j['k']):
            save(a.out/'status.json',dict(state='testing',current=job,complete=completed))
            proc=subprocess.Popen([sys.executable,'-m','LightGenV2.tasks.t10_expert_scaling.evaluate_test',
                 '--run',job['run'],'--data',str(a.data),'--selection-lock',str(lock),'--batch',str(cfg['microbatch']),
                 '--test-source','/DATA/DATA1/guest3/demo_reproduction_data/organcmnist/organcmnist.npz'],env=env,start_new_session=True)
            pids.append(proc.pid)
            running[('test',job['k'])]=(proc,None,job)
            if proc.wait():
                raise RuntimeError(f'TEST failed: {job}')
            running.clear()
        rows=[]
        for job in sorted(completed,key=lambda j:j['k']):
            r=json.loads((Path(job['run'])/'result.json').read_text())
            t=json.loads((Path(job['run'])/'test_result.json').read_text())
            rows.append(dict(k=job['k'],seed=cfg['seed'],best_epoch=r['best_epoch'],train=r['train'],
                             val=r['val'],test=t['test'],checkpoint_sha256=r['checkpoint_sha256']))
        save(a.out/'summary.json',dict(profile=cfg,rows=rows,role='development_followup'))
        save(a.out/'status.json',dict(state='complete',complete=completed,failed=[]))
    except BaseException as error:
        save(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),complete=completed,failed=failed))
        raise
    finally:
        for proc,log,_ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if log is not None:log.close()
        save(a.out/'release_check.json',dict(own_children_exited=True,child_pids=pids,timestamp=time.time()))


if __name__ == '__main__':
    main()
