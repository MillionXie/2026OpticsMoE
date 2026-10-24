"""Finite isolated fresh-process .8600 title-query200x3; never rebuild gallery."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_title_student import TASK,GPU,GALLERY,validate_gallery
from measure_abo_text_qwen import sha,write
from run_abo_student_contract import source_hashes,MODEL,BODY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--gallery',type=Path,default=GALLERY)
    p.add_argument('--repeats',type=int,nargs='+',choices=(1,2,3),default=[1,2,3])
    args=p.parse_args();root=args.output.resolve();bank=args.gallery.resolve()
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not bank.is_relative_to(TASK/'runs/simulation')
            or not 0<args.deadline-time.time()<=7200 or args.repeats!=sorted(set(args.repeats))):
        raise ValueError('New immutable finite unique trial plan required')
    frozen=source_hashes()
    for file in (Path(__file__),HERE/'measure_abo_title_student.py',HERE/'audit_abo_title_student.py',HERE/'run_temporal_remainder.py'):
        frozen[str(file.relative_to(REPO))]=sha(file)
    root.mkdir(parents=True)
    state=dict(status='waiting_gallery',pid=os.getpid(),gallery=str(bank),sources=frozen,repeats=args.repeats,completed=[],
        deadline=args.deadline,command=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def save():write(root/'suite_status.json',state)
    def stable():
        if any(sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Frozen title measurement source changed')
    save();active=False;child=None
    try:
        while True:
            stable()
            if time.time()>args.deadline-300:raise TimeoutError('No budget after gallery preparation')
            previous=json.loads((bank/'controller_status.json').read_text())
            if previous['status']=='complete':break
            if previous['status'] in ('stopped','failed'):raise RuntimeError('Gallery stopped: preserve raw and diagnose, do not substitute a cache')
            time.sleep(20)
        from audit_abo_text_qwen import original_csv_contract
        data=REPO/'data/abo_easy100_dataset_20260906';titles,_,test=original_csv_contract(data)
        validate_gallery(bank,titles,test)
        from audit_abo_title_student import audit
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
        lockpath=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockpath.open('a') as lock:
            while True:
                if time.time()>args.deadline-300:raise TimeoutError('Exclusive admission budget')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            idle=0
            while idle<3:
                stable()
                if time.time()>args.deadline-300:raise TimeoutError('Foreign occupancy exceeded finite admission budget')
                records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign_processes=[])
                save();time.sleep(30 if records else 10)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            for repeat in args.repeats:
                stable()
                if time.time()>args.deadline-180:raise TimeoutError('No budget for another fresh trial')
                name=f'ours_repeat{repeat:02d}';folder=root/name
                command=[sys.executable,'-u',str(HERE/'measure_abo_title_student.py'),'--model',str(MODEL),'--checkpoint',str(BODY),
                    '--data',str(data),'--gallery',str(bank),'--repeat',str(repeat),'--output',str(folder)]
                write(root/(name+'.command.json'),command)
                observations=[]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:
                        end=min(args.deadline,time.time()+1800)
                        while True:
                            stable()
                            if time.time()>end:raise TimeoutError('Fresh title worker deadline')
                            try:code=child.wait(timeout=1);break
                            except subprocess.TimeoutExpired:pass
                            if not (folder/'admission_bound.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                                    raise RuntimeError('Foreign A100 process in title model loading')
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_abo_title_student.py');child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if code:raise RuntimeError(f'{name} exit{code}; diagnose before a new immutable attempt')
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
