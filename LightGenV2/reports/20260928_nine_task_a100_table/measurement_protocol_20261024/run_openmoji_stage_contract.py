"""Finite exclusive CPU stage check; no model/GPU inference until admission."""
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
from openmoji_e45_graph import FRONTEND_CONTRACT,FRONTEND_CONTRACT_SHA,TASK,GPU,sha,write


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:
        raise ValueError('New immutable finite task stage contract required')
    proof=json.loads(FRONTEND_CONTRACT.read_text())
    if sha(FRONTEND_CONTRACT)!=FRONTEND_CONTRACT_SHA or proof['status']!='passed' or not proof['no_gpu']:
        raise ValueError('Exact previously-passed live frontend contract required')
    frozen=dict(proof['sources'])
    for file in (Path(__file__),HERE/'openmoji_e45_graph.py',HERE/'stage_meter.py'):
        frozen[file.relative_to(REPO).as_posix()]=sha(file)
    root.mkdir(parents=True)
    state=dict(status='exclusive_admission',pid=os.getpid(),deadline=args.deadline,no_gpu=True,
        command=sys.argv,sources=frozen,frontend_contract_sha256=FRONTEND_CONTRACT_SHA,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def stable():
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Frozen stage contract source changed')
    def save():write(root/'suite_status.json',state)
    save();child=None
    try:
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        engine.GPU=GPU
        lockpath=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockpath.open('a') as lock:
            while True:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('No CPU contract lock budget')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('Foreign occupancy exceeded stage admission budget')
                records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            folder=root/'original_instrumented_live_CPU_contract'
            command=[sys.executable,'-u',str(HERE/'openmoji_e45_graph.py'),'--output',str(folder)]
            write(root/'cpu_contract.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            with (root/'cpu_contract.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='cpu_stage_contract',child_pid=child.pid);save()
                while child.poll() is None:
                    stable()
                    if time.time()>args.deadline:raise TimeoutError('CPU stage worker deadline')
                    if engine.base.guard_records(GPU):raise RuntimeError('Foreign GPU appeared during CPU stage contract')
                    time.sleep(3)
                if child.returncode:raise RuntimeError('Original/instrumented CPU stage worker exit'+str(child.returncode))
            result=json.loads((folder/'contract.json').read_text())
            if (result['status']!='passed' or result['no_gpu'] is not True or result['real_test_examples']!=2
                    or [r['test_index'] for r in result['checks']]!=[200,999]
                    or not all(r['bitwise_equal'] for r in result['checks'])):
                raise ValueError('Original stage contract not fully passed')
            state.update(status='complete',finished_unix=time.time(),contract_sha256=sha(folder/'contract.json'))
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        save()


if __name__=='__main__':main()
