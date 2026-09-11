"""Run three bounded controls serially on one explicitly chosen GPU.

No torch import in this supervisor: each child owns and releases its CUDA
context. Stop on failure; never kill another user's process or overwrite a run.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='An explicitly inspected idle GPU UUID')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--profiles', nargs='+', choices=['preserve_adam','preserve_sam','preserve_fullfield_sam'],
                        default=['preserve_sam','preserve_adam','preserve_fullfield_sam'])
    args = parser.parse_args()
    if min(args.epochs,args.steps)<1:parser.error('Positive epochs and steps required')
    if len(set(args.profiles))!=len(args.profiles):parser.error('Duplicate profiles')
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running',pid=os.getpid(),gpu=args.gpu,planned=args.profiles,completed=[],
                  started_unix=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    child = None
    def save():
        (args.output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    def interrupt(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for profile in args.profiles:
            # Other users may claim a formerly idle GPU while the queue waits.
            state=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used,util=[int(v.strip()) for v in state.strip().split(',')]
            if used>1024 or util>15:
                raise RuntimeError(f'GPU is no longer idle ({state.strip()}); stop queue, do not evict other work')
            run=args.output/profile
            run.mkdir()
            command=[sys.executable,'-u','-m','LightGenV2.tasks.t07_abo_image_retrieval.standalone.broad_transfer',
                     '--mode','adapt','--profile',profile,'--assets',str(args.assets),'--checkpoint',str(args.checkpoint),
                     '--target',str(args.target),'--output',str(run/'artifacts'),'--adapt-epochs',str(args.epochs),
                     '--steps',str(args.steps),'--batch-size','4']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (run/'console.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                status.update(active_profile=profile,child_pid=child.pid,command=command);save()
                while child.poll() is None:
                    status['heartbeat_unix']=time.time();save();time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f'{profile} failed with exit code {child.returncode}; inspect {run}/console.log')
            report=json.loads((run/'artifacts/final_report.json').read_text(encoding='utf-8'))
            status['completed'].append(dict(profile=profile,hit1=report['metrics']['hit_at_1'],
                removed_hit1=report['remove_optical_same_weights']['hit_at_1'],epoch=report['selected_epoch'],
                alpha=report['model_audit']['alpha']))
            child=None
            status.update(active_profile=None,child_pid=None);save()
        status.update(status='complete',finished_unix=time.time())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        status.update(status='failed_or_interrupted',error=str(exc),child_pid=None,finished_unix=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
