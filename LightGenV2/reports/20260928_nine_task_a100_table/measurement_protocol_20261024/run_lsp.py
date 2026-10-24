"""Finite LSP preflight and six original-model trials, after the active queue.

CPU equivalence is scheduled only after previous timing work finishes. Never
compete for the A100 or add heavy CPU load to another timed E2E window.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import run_temporal_remainder as loading_guard

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
TASK=REPO/'LightGenV2/tasks/t02_keypoint_detection'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024'
MODEL='/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'
LEGACY=REPO/'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after-status',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation'):raise ValueError('New task run required')
    if not 0<args.deadline-time.time()<=21600:raise ValueError('Finite at-most-six-hour plan')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    from LightGenV2.common.baseline_measurement import sha256_file
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    variants={
        'ours':(TASK/'configs/moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml',TASK/'runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt'),
        'qwen':(LEGACY/'configs/lsp_pose_opt2.yaml',LEGACY/'runs/lsp_pose_optical_moe16_opt2/checkpoints/teacher_best_train_loss.pt')}
    root.mkdir(parents=True);state=dict(status='waiting_previous_queue',pid=os.getpid(),completed=[],deadline=args.deadline,
        command=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    frozen={str(path.relative_to(REPO)):sha256_file(path) for path in [Path(__file__),HERE/'measure_lsp.py',HERE/'measure_salicon.py',HERE/'measure_ours.py',HERE/'stage_meter.py',HERE/'run_temporal_remainder.py']}
    state['sources']=frozen
    def save():write(root/'suite_status.json',state)
    def stable():
        if any(sha256_file(REPO/name)!=value for name,value in frozen.items()):raise ValueError('Bound suite source changed')
    def worker(name,command,env):
        stable();write(root/(name+'.command.json'),command)
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',phase=name,child_pid=child.pid);save()
            observations=[];worker_deadline=min(time.time()+1800,args.deadline)
            try:
                while True:
                    if time.time()>=worker_deadline:raise TimeoutError('Original finite worker deadline')
                    try:code=child.wait(timeout=1);break
                    except subprocess.TimeoutExpired:pass
                    if name!='cpu_contract' and not (root/name/'identity.json').exists():
                        snapshot=loading_guard.process_snapshot(GPU,child.pid,PIPE);observations.append(snapshot)
                        if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                            write(root/(name+'.foreign_loading_interruption.json'),snapshot)
                            loading_guard.terminate_own_loading_worker(snapshot,root/name,'measure_lsp.py')
                            child.wait(timeout=20)
                            raise RuntimeError('Foreign GPU process during loading; no retry')
            except BaseException:
                if child.poll() is None:
                    loading_guard.terminate_own_loading_worker({'worker_pid':child.pid},root/name,'measure_lsp.py')
                    child.wait(timeout=15)
                raise
            finally:write(root/(name+'.loading_process_audit.json'),observations)
        if code:raise RuntimeError(name+f' failed exit{code}; do not retry or overwrite')
    def command(method,output,repeat=1):
        config,checkpoint=variants[method]
        return [sys.executable,'-u',str(HERE/'measure_lsp.py'),'--system',method,'--config',str(config),
            '--checkpoint',str(checkpoint),'--data-root',str(REPO/'data/lsp_pose'),'--model',MODEL,
            '--output',str(output),'--repeat',str(repeat),'--gpu-uuid',GPU]
    active=False;save()
    try:
        while True:
            if time.time()>args.deadline-300:raise TimeoutError('Deadline waiting for original queue')
            stable()
            previous=json.loads(args.after_status.read_text())
            if previous['status']=='complete':break
            if previous['status']=='stopped':raise RuntimeError('Previous queue stopped; preserve and request diagnosis')
            time.sleep(30)
        lockfile=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockfile.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 compute; do not interrupt')
            validate_idle_memory(GPU)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            cpu=root/'cpu_contract';worker('cpu_contract',command('ours',cpu)+['--inspect-only'],env)
            if json.loads((cpu/'cpu_contract.json').read_text())['status']!='passed':raise ValueError('CPU equivalence failed')
            if json.loads((cpu/'identity.json').read_text())['script_sha256']!=sha256_file(HERE/'measure_lsp.py'):raise ValueError('Stale CPU preflight')
            for _ in range(3):
                if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 compute during admission')
                validate_idle_memory(GPU);time.sleep(10)
            engine.daemon(True);active=True
            env.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100')
            for method in ('ours','qwen'):
                for repeat in (1,2,3):
                    if time.time()>args.deadline-120:raise TimeoutError('Insufficient time for next original worker')
                    name=f'{method}_repeat{repeat:02d}';folder=root/name
                    worker(name,command(method,folder,repeat),env)
                    result=json.loads((folder/'summary.json').read_text())
                    if result['status']!='complete' or result['valid_samples']!=200:raise ValueError('Incomplete200 worker')
                    state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
