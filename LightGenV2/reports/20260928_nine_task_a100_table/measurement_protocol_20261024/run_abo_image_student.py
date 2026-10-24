"""Original .8058 offline100 bank and isolated image-query200x3, only missing work."""
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
from measure_abo_image_student import TASK,GPU,CPU,cpu_contract,sha,write
from run_abo_image_student_contract import MODEL,DATA,BODY


def trial_plan(repeats,bank):
    if not repeats or repeats!=sorted(set(repeats)) or any(v not in (1,2,3) for v in repeats):raise ValueError('Unique missing repeats only')
    return ([] if bank else [('bank','offline_title_bank',None)])+[('query',f'ours_repeat{i:02d}',i) for i in repeats]


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--bank',type=Path,help='Reuse an already qualified separate bank; never regenerate it')
    parser.add_argument('--repeats',type=int,nargs='+',default=[1,2,3])
    args=parser.parse_args();root=args.output.resolve();plan=trial_plan(args.repeats,args.bank)
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:
        raise ValueError('Fresh finite immutable task queue required')
    proof=cpu_contract();frozen=dict(proof['sources'])
    for name in ('run_abo_image_student.py','measure_abo_image_student.py','audit_abo_image_student.py','run_temporal_remainder.py'):
        frozen[(HERE/name).relative_to(REPO).as_posix()]=sha(HERE/name)
    root.mkdir(parents=True);bank=args.bank or root/'offline_title_bank'
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,command=sys.argv,repeats=args.repeats,
        bank_root=str(bank),cpu_contract_sha256=sha(CPU),completed=[],sources=frozen,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-120:raise TimeoutError('Finite missing workload budget')
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Active original sources changed')
    save();active=False;child=None
    try:
        from audit_abo_image_student import audit,audit_bank
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
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
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            if args.bank:audit_bank(bank,DATA)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            for mode,name,repeat in plan:
                stable();folder=root/name
                command=[sys.executable,'-u',str(HERE/'measure_abo_image_student.py'),'--mode',mode,'--output',str(folder),
                    '--model',str(MODEL),'--data',str(DATA),'--checkpoint',str(BODY)]
                if mode=='query':command+=['--bank',str(bank),'--repeat',str(repeat)]
                write(root/(name+'.command.json'),command);observations=[]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            stable()
                            if not (folder/'admission_bound.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(root/(name+'.foreign_interruption.json'),snapshot);raise RuntimeError('Foreign A100 process during original bank/query loading')
                            time.sleep(1)
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_abo_image_student.py');child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if child.returncode:raise RuntimeError(name+' failed; preserve raw and diagnose before a new immutable run')
                result=audit_bank(folder,DATA) if mode=='bank' else audit(folder,DATA)
                write(root/(name+'.cpu_audit.json'),result);state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
