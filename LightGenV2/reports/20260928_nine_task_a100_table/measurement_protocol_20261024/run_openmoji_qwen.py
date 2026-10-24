"""Finite original native baseline MPS100/fresh200x3; reuse passed CPU proof."""
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
from measure_openmoji_qwen import TASK,GPU,sha,write,validate_contract


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--repeats',type=int,nargs='+',choices=(1,2,3),default=[1,2,3]);args=parser.parse_args()
    root=args.output.resolve();contract=args.contract.resolve()
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not contract.is_relative_to(TASK/'runs/simulation')
            or not 0<args.deadline-time.time()<=7200 or args.repeats!=sorted(set(args.repeats))):
        raise ValueError('New immutable finite unique round plan required')
    proof=validate_contract(contract);parent=json.loads((contract.parent.parent/'suite_status.json').read_text())
    if parent['status']!='complete' or parent['contract_sha256']!=sha(contract):raise ValueError('Passed original native CPU queue required')
    frozen=dict(proof['sources'])
    for name in ('run_openmoji_qwen.py','measure_openmoji_qwen.py','audit_openmoji_qwen.py','openmoji_grid_audit.py','run_temporal_remainder.py'):
        frozen[(HERE/name).relative_to(REPO).as_posix()]=sha(HERE/name)
    root.mkdir(parents=True)
    state=dict(status='exclusive_admission',pid=os.getpid(),deadline=args.deadline,command=sys.argv,
        sources=frozen,cpu_contract=str(contract),cpu_contract_sha256=sha(contract),repeats=args.repeats,completed=[],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def stable():
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Frozen native measurement source changed')
    def save():write(root/'suite_status.json',state)
    save();child=None;active=False
    try:
        from audit_openmoji_qwen import audit
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
        path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('Exclusive task lock budget')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('Admission finite budget exhausted')
                records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            for repeat in args.repeats:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('No budget for another fresh round')
                name=f'qwen_repeat{repeat:02d}';folder=root/name
                cmd=[sys.executable,'-u',str(HERE/'measure_openmoji_qwen.py'),'--output',str(folder),
                    '--contract',str(contract),'--repeat',str(repeat)]
                write(root/(name+'.command.json'),cmd);observations=[]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:
                        end=min(args.deadline,time.time()+1800)
                        while True:
                            stable()
                            if time.time()>end:raise TimeoutError('Fresh native worker deadline')
                            try:code=child.wait(timeout=1);break
                            except subprocess.TimeoutExpired:pass
                            if not (folder/'admission_bound.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                                    raise RuntimeError('Non-owned A100 process in original model loading')
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_openmoji_qwen.py');child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if code:raise RuntimeError(f'{name} exit{code}; preserve raw and supplement missing rounds only')
                result=audit(folder);write(root/(name+'.cpu_audit.json'),result)
                state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
