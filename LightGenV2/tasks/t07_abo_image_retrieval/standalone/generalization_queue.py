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

# Keep the supervisor torch-free: profile names only, no model module import.
PINNED_TEACHER_PROFILES = ('domain_distill_teacher_continue', 'domain_distill_teacher_continue_sam', 'domain_distill_teacher_continue_softgt', 'domain_distill_teacher_continue_fp32gallery')


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
    parser.add_argument('--teacher-cache', type=Path, help='Existing or dependency-produced train-only cache')
    parser.add_argument('--teacher-alignment', type=Path, help='Pinned train-only teacher basis for teacher_continue')
    parser.add_argument('--teacher-model', type=Path, help='Pinned local Qwen snapshot, only for build_teacher_cache')
    parser.add_argument('--reuse-teacher-cache',type=Path,help='Optional pinned older training cache for builder only')
    parser.add_argument('--reuse-teacher-cache-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--profiles', nargs='+', choices=['preserve_adam','preserve_sam','preserve_fullfield_sam','preserve_fullfield_both_sam','regularized_control','regularized_phase05','domain_mixed','domain_curriculum','domain_target_control','domain_refine_control','domain_refine_wide','domain_refine_views','domain_refine_pool500_mix13','domain_refine_context7','domain_refine_balanced','build_teacher_cache','domain_distill_light','domain_distill_strong','domain_distill_stronger','domain_distill_resumeaux','domain_distill_resumeaux_full','domain_distill_sharpteacher','domain_distill_teacher_agreement','domain_distill_aligned_feature','domain_distill_feature_mlp','domain_distill_teacher_first','domain_distill_bounded_aspect','domain_distill_readout_ridge','domain_distill_position_jitter','domain_distill_refit250','domain_distill_refit500',*PINNED_TEACHER_PROFILES],
                        default=['preserve_sam','preserve_adam','preserve_fullfield_sam'])
    parser.add_argument('--after-queue', type=Path, help='Existing status.json; wait without a CUDA context until this queue completes')
    args = parser.parse_args()
    if min(args.epochs,args.steps)<1:parser.error('Positive epochs and steps required')
    if len(set(args.profiles))!=len(args.profiles):parser.error('Duplicate profiles')
    if (args.reuse_teacher_cache is None)!=(args.reuse_teacher_cache_sha256 is None):parser.error('Supply reuse cache and SHA together')
    if args.reuse_teacher_cache is not None and 'build_teacher_cache' not in args.profiles:parser.error('Cache reuse requires build_teacher_cache')
    if any(p in PINNED_TEACHER_PROFILES for p in args.profiles) != (args.teacher_alignment is not None):
        parser.error('--teacher-alignment is required only for a teacher_continue profile')
    if any(p.startswith('domain_') for p in args.profiles) and (args.abo is None or args.pool is None):parser.error('Domain profiles require --abo and --pool')
    if 'build_teacher_cache' in args.profiles:
        if args.teacher_model is None or args.abo is None or args.pool is None:parser.error('Teacher builder requires --teacher-model, --abo and --pool')
        if args.profiles[0]!='build_teacher_cache':parser.error('Teacher cache build must precede training')
    if any(p.startswith('domain_distill_') for p in args.profiles) and args.teacher_cache is None and 'build_teacher_cache' not in args.profiles:
        parser.error('Distillation requires --teacher-cache or a preceding build_teacher_cache job')
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
            if profile=='build_teacher_cache':
                command=[sys.executable,'-u','-m','LightGenV2.tasks.t07_abo_image_retrieval.standalone.teacher_relations',
                    '--target',str(args.target),'--abo',str(args.abo),'--pool',str(args.pool),'--model',str(args.teacher_model),
                    '--output',str(run/'artifacts')]
                if args.reuse_teacher_cache is not None:
                    command+=['--reuse-cache',str(args.reuse_teacher_cache),'--reuse-cache-sha256',args.reuse_teacher_cache_sha256]
            elif profile.startswith('domain_distill_'):
                command+=['--teacher-cache',str(args.teacher_cache)]
                if profile in PINNED_TEACHER_PROFILES:
                    command+=['--teacher-alignment',str(args.teacher_alignment)]
            if profile=='domain_distill_readout_ridge':
                command=[sys.executable,'-u','-m','LightGenV2.tasks.t07_abo_image_retrieval.standalone.readout_distill',
                    '--assets',str(args.assets),'--checkpoint',str(args.checkpoint),'--target',str(args.target),
                    '--abo',str(args.abo),'--pool',str(args.pool),'--teacher-cache',str(args.teacher_cache),
                    '--batch-size','4','--output',str(run/'artifacts')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (run/'console.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                status.update(active_profile=profile,child_pid=child.pid,command=command);save()
                while child.poll() is None:
                    status['heartbeat_unix']=time.time();save();time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f'{profile} failed with exit code {child.returncode}; inspect {run}/console.log')
            report=json.loads((run/'artifacts/final_report.json').read_text(encoding='utf-8'))
            if profile=='build_teacher_cache':
                args.teacher_cache=run/'artifacts/cache.pt'
                status['completed'].append(dict(profile=profile,cache=str(args.teacher_cache),cache_sha256=report['cache_sha256'],
                    training_images=report['training_images'],teacher_trainable_parameters=report['teacher_trainable_parameters']))
            else:
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
