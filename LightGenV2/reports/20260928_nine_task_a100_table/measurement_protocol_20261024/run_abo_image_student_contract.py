"""Only the missing .8058 real-input CPU contract, after the active GPU queue."""
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
from abo_image_student_graph import TASK,BODY,sha,write

PREDECESSOR=REPO/'LightGenV2/tasks/t04_semantic_interaction/runs/simulation/a100_qwen_input_inclusive_20261025_01'
MODEL=Path('/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda')
DATA=REPO/'data/abo_easy100_dataset_20260906'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def predecessor_ready(state):
    if state.get('status') in ('failed','stopped'):raise RuntimeError('Inspect stopped predecessor before replacement')
    if state.get('status')!='complete':return False
    if state.get('completed')!=['qwen_repeat01','qwen_repeat02','qwen_repeat03'] or not state.get('mps_released'):
        raise ValueError('Predecessor must retain all three qualified rounds and release MPS')
    return True


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:
        raise ValueError('New immutable finite task contract required')
    frozen={p.relative_to(REPO).as_posix():sha(p) for p in (Path(__file__),HERE/'abo_image_student_graph.py',
        HERE/'abo_student_graph.py',HERE/'stage_meter.py')}
    root.mkdir(parents=True);child=None
    state=dict(status='waiting_previous_queue',pid=os.getpid(),deadline=args.deadline,sources=frozen,
        predecessor=str(PREDECESSOR),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-120:raise TimeoutError('Finite missing-input contract budget')
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Frozen contract source changed')
    save()
    try:
        while True:
            stable()
            if predecessor_ready(json.loads((PREDECESSOR/'suite_status.json').read_text())):break
            time.sleep(10)
        for i in (1,2,3):
            if json.loads((PREDECESSOR/f'qwen_repeat{i:02d}.cpu_audit.json').read_text())['status']!='passed':
                raise ValueError('Predecessor raw audit missing; never rerun qualified measurements')
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        engine.GPU=GPU
        lockpath=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockpath.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable();records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            folder=root/'original_image_title_CPU_contract'
            command=[sys.executable,'-u',str(HERE/'abo_image_student_graph.py'),'--output',str(folder),
                '--model',str(MODEL),'--data',str(DATA),'--checkpoint',str(BODY)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='real_image_title_CPU_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable()
                    if engine.base.guard_records(GPU):raise RuntimeError('Non-owned A100 appeared during CPU contract')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Actual .8058 image/title CPU contract failed')
            proof=json.loads((folder/'contract.json').read_text())
            if proof['status']!='passed' or not proof['no_gpu'] or not all(c['bitwise_equal'] for c in proof['checks']):
                raise ValueError('Missing original numerical contract')
            state.update(status='complete',finished_unix=time.time(),contract_sha256=sha(folder/'contract.json'))
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        save()


if __name__=='__main__':main()
