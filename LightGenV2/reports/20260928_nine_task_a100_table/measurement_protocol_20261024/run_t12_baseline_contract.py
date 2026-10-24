"""Finite missing Qwen28 CPU proof after the explicit original T12 CPU proof.

No MPS or CUDA, no model loads before predecessor and shared-lock admission.
This is not a formal200x3 queue or a full-TEST accuracy evaluation.
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
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from t12_live_frontend import TASK,ASSETS,BODY_SHA,sha,write
from measure_abo_text_qwen import GPU


def predecessor_ready(state,proof=None,digest=None):
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('Stopped original T12 CPU predecessor: diagnose and explicitly rebind missing work')
    if state.get('status')!='complete':return False
    if (proof is None or state.get('contract_sha256')!=digest or proof.get('status')!='passed'
            or proof.get('no_gpu') is not True or proof.get('body_sha256')!=BODY_SHA
            or [r.get('test_index') for r in proof.get('checks',[])]!=[200,2303]
            or not all(r.get('bitwise_equal') for r in proof.get('checks',[]))):
        raise ValueError('Actual original two-input CPU predecessor proof required')
    return True


def predecessor_file_ready(root):
    try:state=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    if state.get('status')!='complete':return predecessor_ready(state)
    path=root/'original_live_CPU_contract/contract.json'
    return predecessor_ready(state,json.loads(path.read_text()),sha(path))


def frozen_sources():
    paths=[Path(__file__),HERE/'t12_baseline_live.py',HERE/'t12_live_frontend.py',
        REPO/'maintenance/storage/T12_HISTORICAL_SEVEN_MODEL_IDENTITY_20261004.json']
    paths.extend(TASK/name for name in ('half_qwen.py','electronic_turbo_infer.py','product_repair_model.py',
        'feature_cache.py','export_matched_summary.py','product_unified_edit_data_v2.py'))
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    root=args.output.resolve();previous=args.previous.resolve()
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation')
            or not previous.is_relative_to(TASK/'runs/simulation') or root==previous
            or not 0<args.deadline-time.time()<=7200):raise ValueError('New finite explicit canonical CPU queue required')
    frozen=frozen_sources();root.mkdir(parents=True);child=None
    state=dict(status='waiting_previous_CPU_contract',pid=os.getpid(),predecessor=str(previous),
        deadline=args.deadline,sources=frozen,command=sys.argv,no_gpu=True,
        not_a_formal_timing_or_full_test_evaluation=True)
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-180:raise TimeoutError('Finite Qwen28 CPU admission/runtime budget')
        if frozen_sources()!=frozen:raise RuntimeError('Active CPU original sources changed')
    save()
    try:
        while not predecessor_file_ready(previous):stable();time.sleep(15)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<12:
                stable();records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign=records)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if records else 10)
            folder=root/'original_live_CPU_contract'
            command=[str(ASSETS/'venv/bin/python'),'-u',str(HERE/'t12_baseline_live.py'),'--output',str(folder)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,env=env)
                state.update(status='real_Qwen28_live_CPU_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable();foreign=engine.base.guard_records(GPU)
                    if foreign:
                        write(root/'foreign_CPU_interruption.json',dict(unix=time.time(),foreign=foreign,own_worker=child.pid))
                        raise RuntimeError('Foreign A100 activity during CPU contract; stop only own worker')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Original Qwen28 real-input CPU contract failed')
            proof=json.loads((folder/'contract.json').read_text())
            if (proof.get('status')!='passed' or proof.get('no_gpu') is not True
                    or [r.get('test_index') for r in proof.get('checks',[])]!=[200,2303]
                    or not all(r.get('bitwise_equal') for r in proof.get('checks',[]))):
                raise ValueError('Complete real Qwen28 CPU proof missing')
            state.update(status='complete',contract_sha256=sha(folder/'contract.json'),finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        save()


if __name__=='__main__':main()
