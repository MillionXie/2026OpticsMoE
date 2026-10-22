"""Run the four paired continuation jobs on at most two idle GPUs; exclude GPU6."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .train import save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sources', type=Path, default=Path('/DATA/DATA1/guest3/t10_fixed478_3datasets_20260921_runs'))
    ap.add_argument('--gpu-candidates', type=int, nargs='+', default=[1, 0, 2, 3, 4, 5])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [dict(dataset=ds, k=k) for ds in ('kather2016', 'organcmnist') for k in (3, 4)]
    running, complete, failed = {}, [], []
    children = []
    interrupted = False

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(a.out/'identity.json', dict(pid=os.getpid(), command=sys.argv,
         commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         protocol='single_seed17_paired_k3_k4_extra30_lr2e-4_to2e-5', max_gpus=2))
    try:
        while jobs or running:
            for gpu in a.gpu_candidates:
                if gpu == 6 or gpu in running or len(running) >= 2 or not jobs:
                    continue
                used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                if used > 200:
                    continue
                job = jobs.pop(0)
                name = f"{job['dataset']}_k{job['k']}_s17_extra30"
                source = a.sources/f"{job['dataset']}__moe_oeo_N4_k{job['k']}_L4_lr0.002_s17"
                uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
                cmd = [sys.executable, '-u', '-m', 'LightGenV2.tasks.t10_expert_scaling.diagnose_topk',
                       '--source', str(source), '--out', str(a.out/name)]
                if job['dataset'] == 'organcmnist':
                    cmd += ['--test-source', '/DATA/DATA1/guest3/demo_reproduction_data/organcmnist/organcmnist.npz']
                log = (a.out/f'{name}.log').open('w')
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(proc.pid)
                running[gpu] = (proc, log, dict(**job, pid=proc.pid, gpu_uuid=uuid))
            for gpu, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[gpu]
                if proc.returncode:
                    failed.append(dict(**job, exit_code=proc.returncode))
                    raise RuntimeError(f'Failed job: {job}')
                complete.append(job)
            save(a.out/'status.json', dict(state='running' if jobs or running else 'complete',
                 complete=complete, pending=jobs, running={str(g):j for g, (_, _, j) in running.items()}, failed=failed))
            if jobs or running:
                time.sleep(10)
    except BaseException as error:
        interrupted = True
        save(a.out/'status.json', dict(state='interrupted_or_failed', error=repr(error),
             complete=complete, pending=jobs, failed=failed))
        raise
    finally:
        for proc, log, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        save(a.out/'release_check.json', dict(own_children_exited=True, child_pids=children,
             interrupted=interrupted, failed=failed, timestamp=time.time()))


if __name__ == '__main__':
    main()
