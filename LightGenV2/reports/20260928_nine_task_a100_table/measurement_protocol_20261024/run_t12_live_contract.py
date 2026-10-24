"""Only missing real T12 CPU input/stage contract, after image-query timing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from t12_live_frontend import TASK,ASSETS,sha,write
from measure_abo_text_qwen import GPU
PREVIOUS=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_image_student_input_inclusive_20261025_01'


def predecessor_ready(state):
    if state.get('status') in ('failed','stopped'):raise RuntimeError('Stopped predecessor must be diagnosed and explicit missing rounds rebound')
    return state.get('status')=='complete' and state.get('mps_released') is True and state.get('completed')==[
        'offline_title_bank','ours_repeat01','ours_repeat02','ours_repeat03']


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:raise ValueError('New finite immutable CPU queue required')
    frozen={p.relative_to(REPO).as_posix():sha(p) for p in (Path(__file__),HERE/'t12_live_frontend.py',HERE/'stage_meter.py')}
    root.mkdir(parents=True);child=None
    state=dict(status='waiting_previous_queue',pid=os.getpid(),predecessor=str(PREVIOUS),deadline=args.deadline,sources=frozen,command=sys.argv)
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-180:raise TimeoutError('Finite CPU admission budget')
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Active T12 contract sources changed')
    save()
    try:
        while not predecessor_ready(json.loads((PREVIOUS/'suite_status.json').read_text())):stable();time.sleep(15)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        engine.GPU=GPU
        path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable();records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign=records)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if records else 10)
            folder=root/'original_live_CPU_contract'
            command=[str(ASSETS/'venv/bin/python'),'-u',str(HERE/'t12_live_frontend.py'),'--output',str(folder)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='real_live_CPU_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable()
                    if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 process during CPU contract')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Original live-input T12 CPU contract failed')
            proof=json.loads((folder/'contract.json').read_text())
            if proof['status']!='passed' or not proof['no_gpu']:raise ValueError('Real CPU proof missing')
            state.update(status='complete',contract_sha256=sha(folder/'contract.json'),finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        save()


if __name__=='__main__':main()
