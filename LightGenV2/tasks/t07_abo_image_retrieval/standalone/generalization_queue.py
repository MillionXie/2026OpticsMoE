"""Run bounded controls serially on one explicitly chosen GPU.

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


def dependency_state(path, expected_gpu):
    """Read-only dependency check. A partial heartbeat write is retried.

    A stale running dependency never starts a competing GPU job. No PID is
    killed or treated as sufficient evidence that its entire queue completed.
    """
    try:
        status = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return 'waiting_for_dependency'
    if status.get('gpu') != expected_gpu:
        raise RuntimeError('Dependency queue GPU does not match the selected GPU')
    if status.get('status') == 'complete':
        return 'ready'
    if status.get('status') in ('running', 'waiting_for_dependency'):
        return 'waiting_for_dependency'
    raise RuntimeError(f'Dependency did not complete successfully: {status.get("status")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='An explicitly inspected idle GPU UUID')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--abo', type=Path)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--profiles', nargs='+', choices=['preserve_adam','preserve_sam','preserve_fullfield_sam','preserve_fullfield_both_sam','regularized_control','regularized_phase05','domain_mixed','domain_curriculum','domain_target_control'],
                        default=['preserve_sam','preserve_adam','preserve_fullfield_sam'])
    parser.add_argument('--after-queue', type=Path, help='Existing status.json; wait without a CUDA context until this queue completes')
    args = parser.parse_args()
    if min(args.epochs,args.steps)<1:parser.error('Positive epochs and steps required')
    if len(set(args.profiles))!=len(args.profiles):parser.error('Duplicate profiles')
    if any(p.startswith('domain_') for p in args.profiles) and (args.abo is None or args.pool is None):parser.error('Domain profiles require --abo and --pool')
    if args.after_queue is not None and not args.after_queue.is_file():parser.error('--after-queue must be an existing status.json')
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
        if args.after_queue is not None:
            status['dependency_status_file']=str(args.after_queue.resolve())
            while dependency_state(args.after_queue,args.gpu) != 'ready':
                status.update(status='waiting_for_dependency',heartbeat_unix=time.time());save();time.sleep(30)
            status.update(status='running',dependency_complete=True);save()
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
            if profile.startswith('domain_'):command+=['--abo',str(args.abo),'--pool',str(args.pool)]
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
            released_pid=child.pid
            live_gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
            if any(line.split(',')[0].strip()==str(released_pid) for line in live_gpu_processes.splitlines()):
                raise RuntimeError(f'Exited child {released_pid} is still listed by nvidia-smi; do not start another job')
            status['completed'][-1].update(child_pid=released_pid,gpu_context_released=True)
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
