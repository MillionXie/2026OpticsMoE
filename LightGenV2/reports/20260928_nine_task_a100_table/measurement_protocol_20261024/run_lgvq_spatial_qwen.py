"""Finite three-repeat spatial baseline, strictly after the existing LSP suite."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from measure_lgvq_spatial_qwen import PIN,GPU,validate_reference,source_map,TASK
import run_temporal_remainder as loading

PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024'
MODEL='/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203'
PT='/DATA/DATA1/guest3/2026OpticsMoE_a100_formal/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/qwen3vl_spatial_quality_tokens_4f_r448_dataset_once_a100_20260907/checkpoints/frames4/best_checkpoint.pt'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--after-status',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation'):raise ValueError('New original-task run required')
    if not 0<args.deadline-time.time()<=21600:raise ValueError('Finite six-hour plan')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    from audit_lgvq_spatial_qwen import audit
    manifest=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    validate_reference(entry.sha);entry.input_rows(manifest)
    if entry.sha(PT)!=PIN:raise ValueError('Wrong original .6908 spatial head')
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    frozen=source_map()
    for path in (Path(__file__),HERE/'audit_lgvq_spatial_qwen.py',HERE/'run_temporal_remainder.py'):
        frozen[str(path.relative_to(REPO))]=entry.sha(path)
    root.mkdir(parents=True);active=False
    state=dict(status='waiting_previous_queue',pid=os.getpid(),completed=[],sources=frozen,command=sys.argv,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),deadline=args.deadline)
    def save():write(root/'suite_status.json',state)
    def stable():
        if any(entry.sha(REPO/name)!=digest for name,digest in frozen.items()):raise ValueError('Frozen source changed')
    save()
    try:
        while True:
            stable()
            if time.time()>args.deadline-300:raise TimeoutError('No time after prior suite')
            previous=entry.read(args.after_status)
            if previous['status']=='complete':break
            if previous['status']=='stopped':raise RuntimeError('Prior suite stopped; preserve all evidence')
            time.sleep(30)
        with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(3):
                if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 process')
                validate_idle_memory(GPU);time.sleep(10)
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for repeat in (1,2,3):
                stable()
                if time.time()>args.deadline-120:raise TimeoutError('No time for next worker')
                name=f'qwen_repeat{repeat:02d}';folder=root/name
                command=[sys.executable,'-u',str(HERE/'measure_lgvq_spatial_qwen.py'),'--model',MODEL,'--checkpoint',PT,
                    '--manifest',str(manifest),'--repeat',str(repeat),'--output',str(folder)]
                write(root/(name+'.command.json'),command)
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid,
                        worker_deadline=min(args.deadline,time.time()+1800));save()
                    observations=[]
                    try:
                        while True:
                            if time.time()>=state['worker_deadline']:
                                raise TimeoutError('Finite worker deadline')
                            try:code=child.wait(timeout=1);break
                            except subprocess.TimeoutExpired:pass
                            if not (folder/'identity.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                                    raise RuntimeError('Foreign compute during model loading')
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_lgvq_spatial_qwen.py')
                            child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if code:raise RuntimeError(name+f' exit{code}; preserve raw, no retry')
                result=audit(folder,manifest);write(root/(name+'.cpu_audit.json'),result)
                state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
