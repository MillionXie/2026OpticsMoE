"""Finite CPU-only original LGVQ proof after the actual T12 Qwen CPU proof.

Never loads models while waiting, never starts MPS/CUDA or a formal GPU cost
round. Shares the lab admission lock, and stops only its own CPU worker.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from lgvq_live_frontend import TASK,sources,sha,write
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
BASELINE_SHA='599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52'
PYTHON='/home/guest3/miniconda3/envs/xml/bin/python'


def predecessor_ready(state,proof=None,digest=None):
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('CPU predecessor stopped; preserve evidence and rebind only missing work')
    if state.get('status')!='complete':return False
    if (not proof or proof.get('status')!='passed' or proof.get('no_gpu') is not True
            or state.get('contract_sha256')!=digest
            or proof.get('baseline_sha256')!=BASELINE_SHA
            or [r.get('test_index') for r in proof.get('checks',[])]!=[200,2303]
            or not all(r.get('bitwise_equal') for r in proof.get('checks',[]))):
        raise ValueError('Actual original Qwen28 CPU proof required')
    return True


def predecessor_file_ready(root):
    try: state=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    if state.get('status')!='complete':return predecessor_ready(state)
    path=root/'original_live_CPU_contract/contract.json'
    return predecessor_ready(state,json.loads(path.read_text()),sha(path))


def frozen_sources():
    return dict(sources(),**{Path(__file__).relative_to(REPO).as_posix():sha(__file__)})


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve();previous=args.previous.resolve()
    t12=REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation'
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation')
            or not previous.is_relative_to(t12) or not 0<args.deadline-time.time()<=14400):
        raise ValueError('New bounded canonical CPU successor required')
    frozen=frozen_sources();root.mkdir(parents=True);child=None
    state=dict(status='waiting_previous_CPU_contract',pid=os.getpid(),predecessor=str(previous),
        deadline=args.deadline,sources=frozen,command=sys.argv,no_gpu=True,
        not_a_formal_timing_or_full_TEST_evaluation=True)
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-300:raise TimeoutError('Finite LGVQ CPU admission/runtime budget')
        if frozen_sources()!=frozen:raise RuntimeError('Active original LGVQ CPU sources changed')
    save()
    try:
        while not predecessor_file_ready(previous):stable();time.sleep(15)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        path=TASK/'runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<12:
                stable();foreign=engine.base.guard_records(GPU)
                if foreign:idle=0;state.update(status='waiting_foreign_process',foreign=foreign)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if foreign else 10)
            folder=root/'original_live_CPU_contract'
            command=[PYTHON,'-u',str(HERE/'lgvq_live_frontend.py'),'--output',str(folder)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for name in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_LOG_DIRECTORY'):
                env.pop(name,None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='real_LGVQ_live_CPU_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable();foreign=engine.base.guard_records(GPU)
                    if foreign:
                        write(root/'foreign_CPU_interruption.json',dict(unix=time.time(),foreign=foreign,own_worker=child.pid))
                        raise RuntimeError('Foreign A100 activity; stop only own CPU worker')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Original LGVQ real-input CPU proof failed')
            proof=json.loads((folder/'contract.json').read_text())
            if (proof.get('status')!='passed' or proof.get('no_gpu') is not True or len(proof.get('checks',[]))!=4
                    or not all(c.get('bitwise_equal') and c.get('full_original_output_bitwise_equal')
                               for c in proof['checks'])):
                raise ValueError('Complete real LGVQ CPU proof required')
            state.update(status='complete',contract_sha256=sha(folder/'contract.json'),finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        save()


if __name__=='__main__':main()
