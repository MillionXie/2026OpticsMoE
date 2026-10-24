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


def predecessor_action(state):
    if state.get('status')=='complete':
        if state.get('completed')!=['ours_repeat01','ours_repeat02','ours_repeat03'] or not state.get('mps_released'):
            raise ValueError('Title predecessor did not retain its complete canonical triple/release')
        return 'admit'
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('Pinned predecessor stopped; diagnose and re-enroll explicitly, never bypass it')
    return 'wait'


def main():
    import fcntl  # Linux queue only; pure admission contracts also run on Windows.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    root=args.output.resolve();previous=args.predecessor.resolve()
    t08=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or previous.parent!=t08 or not 0<args.deadline-time.time()<=7200:
        raise ValueError('New finite immutable task queue and explicitly pinned predecessor required')
    root.mkdir(parents=True)
    state=dict(status='waiting_previous_queue',pid=os.getpid(),no_gpu=True,command=sys.argv,predecessor=str(previous),
        deadline=args.deadline,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    frozen={p.name:sha(p) for p in (Path(__file__),HERE/'prepare_openmoji_e45.py',HERE/'protocol_identity.py',HERE/'protocol_amendment_openmoji_e45.json')}
    state['frozen_preflight_sources']=frozen;child=None
    def stable():
        if any(sha(HERE/name)!=digest for name,digest in frozen.items()):raise ValueError('Frozen CPU preflight source changed')
    def save():write(root/'suite_status.json',state)
    save()
    try:
        while True:
            stable()
            if time.time()>args.deadline-300:raise TimeoutError('No finite CPU preflight budget after predecessor')
            if predecessor_action(json.loads((previous/'suite_status.json').read_text()))=='admit':break
            time.sleep(30)
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
