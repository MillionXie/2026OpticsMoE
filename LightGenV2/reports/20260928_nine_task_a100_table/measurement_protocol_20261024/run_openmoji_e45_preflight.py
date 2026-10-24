"""Finite CPU frontend preflight, only after the pinned title query queue ends.

No heavy CPU work runs concurrently with any A100 measurement. This controller
does not create MPS, run CUDA, train, or alter the other task's original queue.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from prepare_openmoji_e45 import TASK,GPU,sha,write


def predecessor_action(state,retained=None):
    if state.get('status')=='complete':
        expected=['ours_repeat01','ours_repeat02','ours_repeat03']
        if retained is not None:
            if (retained.get('status')!='stopped' or retained.get('completed')!=['ours_repeat01']
                    or retained.get('repeats')!=[1,2,3] or retained.get('gallery')!=state.get('gallery')
                    or state.get('repeats')!=[2,3]):
                raise ValueError('Retain only qualified round1 and explicitly supplement rounds2/3 against the same gallery')
            expected=['ours_repeat02','ours_repeat03']
        if state.get('completed')!=expected or not state.get('mps_released'):
            raise ValueError('Title predecessor did not retain its complete canonical triple/release')
        return 'admit'
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('Pinned predecessor stopped; diagnose and re-enroll explicitly, never bypass it')
    return 'wait'


def main():
    import fcntl  # Linux queue only; pure admission contracts also run on Windows.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--retained-title-run',type=Path,
        help='Explicit immutable stopped title run containing only qualified round1; predecessor must supply exactly2/3')
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    root=args.output.resolve();previous=args.predecessor.resolve()
    t08=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    retained_root=args.retained_title_run.resolve() if args.retained_title_run else None
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation') or previous.parent!=t08
            or retained_root is not None and (retained_root.parent!=t08 or retained_root==previous)
            or not 0<args.deadline-time.time()<=7200):
        raise ValueError('New finite immutable task queue and explicitly pinned predecessor required')
    root.mkdir(parents=True)
    state=dict(status='waiting_previous_queue',pid=os.getpid(),no_gpu=True,command=sys.argv,predecessor=str(previous),
        retained_title_run=str(retained_root) if retained_root else None,
        deadline=args.deadline,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    frozen={p.name:sha(p) for p in (Path(__file__),HERE/'prepare_openmoji_e45.py',HERE/'protocol_identity.py',HERE/'protocol_amendment_openmoji_e45.json')}
    state['frozen_preflight_sources']=frozen;child=None
    def stable():
        if any(sha(HERE/name)!=digest for name,digest in frozen.items()):raise ValueError('Frozen CPU preflight source changed')
    def save():write(root/'suite_status.json',state)
    save()
    try:
        retained=None;retained_artifacts={}
        if retained_root is not None:
            retained=json.loads((retained_root/'suite_status.json').read_text())
            release=json.loads((retained_root/'post_external_interruption_release_check01.json').read_text())
            if (retained.get('status')!='stopped' or retained.get('completed')!=['ours_repeat01']
                    or not release.get('own_mps_released') or release.get('existing_owned_pids')):
                raise ValueError('Retained title round or independent post-interruption release evidence incomplete')
            retained_artifacts={str(p.relative_to(REPO)):sha(p) for p in
                (retained_root/'suite_status.json',retained_root/'post_external_interruption_release_check01.json')}
            state['retained_artifacts_sha256']=retained_artifacts;save()
        while True:
            stable()
            if any(sha(REPO/name)!=digest for name,digest in retained_artifacts.items()):
                raise ValueError('Immutable retained predecessor evidence changed')
            if time.time()>args.deadline-300:raise TimeoutError('No finite CPU preflight budget after predecessor')
            if predecessor_action(json.loads((previous/'suite_status.json').read_text()),retained)=='admit':break
            time.sleep(30)
        # Recompute every retained/supplemented raw round, not merely its summary.
        # This is CPU data audit only: no Torch/model execution or CUDA.
        from audit_abo_title_student import audit
        sources=[(previous,1),(previous,2),(previous,3)] if retained_root is None else [
            (retained_root,1),(previous,2),(previous,3)]
        audits=[dict(run=str(run),repeat=repeat,audit=audit(run/f'ours_repeat{repeat:02d}',
            REPO/'data/abo_easy100_dataset_20260906')) for run,repeat in sources]
        if any(a['audit']['status']!='passed' or a['audit']['repeat']!=a['repeat'] for a in audits):
            raise ValueError('Canonical title raw triple did not pass independent CPU replay')
        write(root/'canonical_title_predecessor_audit.json',audits)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        engine.GPU=GPU
        path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                if time.time()>args.deadline-300:raise TimeoutError('CPU lock admission budget exhausted')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable()
                if time.time()>args.deadline-300:raise TimeoutError('Foreign occupancy exceeded CPU admission budget')
                records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            folder=root/'original_live_input_CPU_contract'
            command=[sys.executable,'-u',str(HERE/'prepare_openmoji_e45.py'),'--output',str(folder)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='cpu_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable()
                    if time.time()>args.deadline:raise TimeoutError('CPU worker finite deadline')
                    if engine.base.guard_records(GPU):raise RuntimeError('Foreign GPU appeared during CPU contract')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Original real-input CPU contract exit'+str(child.returncode))
            contract=json.loads((folder/'contract.json').read_text())
            if contract['status']!='passed' or contract['no_gpu'] is not True or contract['real_test_examples']!=2:
                raise ValueError('CPU contract incomplete')
            state.update(status='complete',finished_unix=time.time(),contract_sha256=sha(folder/'contract.json'));save()
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=15)
        save()


if __name__=='__main__':main()
