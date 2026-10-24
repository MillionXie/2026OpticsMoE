"""Finite T08 original-title CPU contract then three fresh MPS100 query runs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import TASK,GPU,source_map,sha,write
import run_temporal_remainder as loading

PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024'
MODEL='/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:
        raise ValueError('New immutable task run and finite two-hour plan required')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    from audit_abo_text_qwen import audit
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    frozen=source_map()
    for path in (Path(__file__),HERE/'audit_abo_text_qwen.py',HERE/'run_temporal_remainder.py'):
        frozen[str(path.relative_to(REPO))]=sha(path)
    data=REPO/'data/abo_easy100_dataset_20260906'
    common=['--model',MODEL,'--data',str(data),'--bank',str(TASK/'runs/simulation/text_to_image_frozen_20260920/embeddings.pt')]
    def stable():
        if any(sha(REPO/name)!=digest for name,digest in frozen.items()):raise ValueError('Bound runtime changed')
    lock_path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 process; do not mix or stop it')
        validate_idle_memory(GPU)
        root.mkdir(parents=True);active=False
        state=dict(status='admission',pid=os.getpid(),completed=[],sources=frozen,command=sys.argv,
            deadline=args.deadline,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        def save():write(root/'suite_status.json',state)
        save()
        def run(name,command,env,output,cpu_only=False):
            stable()
            if time.time()>args.deadline-120:raise TimeoutError('No time for another fresh worker')
            write(root/(name+'.command.json'),command)
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',phase=name,child_pid=child.pid);save()
                observations=[];end=min(args.deadline,time.time()+1800)
                try:
                    while True:
                        if time.time()>=end:raise TimeoutError('Finite private worker deadline')
                        try:code=child.wait(timeout=1);break
                        except subprocess.TimeoutExpired:pass
                        if cpu_only:
                            records=engine.base.guard_records(GPU)
                            observations.append(dict(unix=time.time(),foreign_gpu_processes=records))
                            if records:raise RuntimeError('A100 acquired by another process during CPU admission')
                        elif not (output/'admission_bound.json').exists():
                            snapshot=loading.process_snapshot(GPU,child.pid,PIPE);observations.append(snapshot)
                            if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                                raise RuntimeError('Foreign A100 process during loading')
                except BaseException:
                    if child.poll() is None:
                        # Reuse the verified PID/start-command-specific stop;
                        # it cannot target another user's process or process group.
                        loading.terminate_own_loading_worker({'worker_pid':child.pid},output,'measure_abo_text_qwen.py')
                        child.wait(timeout=20)
                    raise
                finally:write(root/(name+'.loading_process_audit.json'),observations)
            if code:raise RuntimeError(f'{name} exited{code}; keep raw, no automatic retry')
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            contract=root/'original_title_cpu_contract'
            run('cpu_contract',[sys.executable,'-u',str(HERE/'measure_abo_text_qwen.py'),*common,
                '--cpu-contract','--output',str(contract)],env,contract,cpu_only=True)
            if json.loads((contract/'contract.json').read_text())['status']!='passed':raise ValueError('CPU contract not passed')
            for _ in range(3):
                stable()
                if engine.base.guard_records(GPU):raise RuntimeError('A100 no longer idle')
                validate_idle_memory(GPU);time.sleep(10)
            engine.daemon(True);active=True
            env.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100')
            for repeat in (1,2,3):
                name=f'qwen_repeat{repeat:02d}';folder=root/name
                run(name,[sys.executable,'-u',str(HERE/'measure_abo_text_qwen.py'),*common,'--repeat',str(repeat),
                    '--contract',str(contract/'contract.json'),'--output',str(folder)],env,folder)
                result=audit(folder,data);write(root/(name+'.cpu_audit.json'),result)
                state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:
            if active:
                try:engine.daemon(False);state['mps_released']=True
                except BaseException as error:state['release_error']=repr(error)
            save()


if __name__=='__main__':main()
