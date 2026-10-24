"""Finite CPU-only real-title contract after the existing image-Qwen queue.

No model inference, CPU-heavy preflight or CUDA initialization until the pinned
predecessor is complete and the physical A100 has passed the idle guards. This
does not launch a formal200x3 measurement, create MPS, or repeat Qwen inference.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
TASK=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PREDECESSOR=TASK/'runs/simulation/a100_image_input_inclusive_20261024_03'
MODEL=Path('/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda')
BODY=TASK/'runs/simulation/optical_text_to_image_64_10cm_compact_e0p5_seed42_20260925/best_checkpoint.pt'


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def source_hashes():
    # These are declarations, not imports of Torch or a model.
    paths=[Path(__file__),HERE/'abo_student_graph.py',HERE/'stage_meter.py',
        HERE/'measure_ours.py',HERE/'measure_salicon.py',HERE/'measure_abo_text_qwen.py',
        HERE/'abo_query_contract.py',TASK/'reverse_runtime.py',TASK/'text_to_image.py',
        TASK/'configs/text_to_image_10cm_adopted_eval.yaml',
        REPO/'LightGenV2/tasks/t01_object_retrieval/modeling.py',
        REPO/'LightGenV2/tasks/t01_object_retrieval/settings.py',
        REPO/'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/features.py',
        REPO/'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/modeling.py']
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def predecessor_action(state):
    if state.get('status')=='complete':
        if state.get('completed')!=['qwen_repeat03']:
            raise ValueError('Predecessor complete without its one required new round')
        return 'admit'
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('Predecessor stopped; inspect its evidence, never bypass or repeat it')
    return 'wait'


def check_result(data):
    checks=data.get('checks',[])
    if (data.get('status')!='passed' or data.get('no_gpu') is not True
            or data.get('mode')!='real_title_original_full_student_CPU_bitwise'
            or [c.get('title_index') for c in checks]!=[0,99]
            or not all(c.get('bitwise_equal') is True for c in checks)
            or data.get('identity',{}).get('body_sha256')!='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'):
        raise ValueError('Actual original .8600 two-title CPU contract is incomplete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to((TASK/'runs/simulation').resolve()):
        raise ValueError('Fresh immutable T08 queue required')
    if not 0<args.deadline-time.time()<=7200:raise ValueError('Finite budget of at most two hours required')
    root.mkdir(parents=True)
    frozen=source_hashes()
    state=dict(status='waiting_previous_queue',pid=os.getpid(),command=sys.argv,
        predecessor=str(PREDECESSOR),deadline=args.deadline,sources=frozen,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        no_gpu=True,not_a_formal_timing_or_energy_run=True)
    write(root/'suite_status.json',state)
    child=None
    try:
        while True:
            if time.time()>args.deadline-300:raise TimeoutError('No budget for CPU contract')
            previous=json.loads((PREDECESSOR/'suite_status.json').read_text())
            if predecessor_action(previous)=='admit':break
            time.sleep(30)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        engine.GPU=GPU
        lockfile=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockfile.open('a') as lock:
            while True:
                if time.time()>args.deadline-300:raise TimeoutError('CPU admission budget exhausted')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                if time.time()>args.deadline-300:raise TimeoutError('Foreign occupancy exceeded CPU admission budget')
                records=engine.base.guard_records(GPU)
                if records:
                    idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    validate_idle_memory(GPU);idle+=1
                    state.update(status='idle_guard',idle_observations=idle,foreign_processes=[])
                write(root/'suite_status.json',state);time.sleep(10 if not records else 30)
            if source_hashes()!=frozen:raise RuntimeError('Frozen CPU contract sources changed')
            output=root/'original_title_cpu_contract'
            command=[sys.executable,'-u',str(HERE/'abo_student_graph.py'),'--model',str(MODEL),
                '--data',str(args.data),'--checkpoint',str(BODY),'--output',str(output)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='cpu_contract',child_pid=child.pid);write(root/'suite_status.json',state)
                while child.poll() is None:
                    if time.time()>args.deadline:raise TimeoutError('CPU contract deadline exceeded')
                    # No CPU-heavy preflight concurrent with another GPU user.
                    if engine.base.guard_records(GPU):raise RuntimeError('Foreign compute appeared during CPU contract')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Original CPU contract exit '+str(child.returncode))
            check_result(json.loads((output/'contract.json').read_text()))
            if source_hashes()!=frozen:raise RuntimeError('Frozen sources changed during contract')
            state.update(status='complete',finished_unix=time.time(),contract=str(output/'contract.json'))
    except BaseException as error:
        state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        write(root/'suite_status.json',state)


if __name__=='__main__':main()
