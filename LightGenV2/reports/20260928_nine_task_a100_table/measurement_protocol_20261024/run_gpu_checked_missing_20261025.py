"""Missing Qwen/LGVQ only, native-GPU checks then original fresh 200x3 costs.

Human 2026-10-25: remove expensive CPU inference as a global timing gate.
The already-qualified T12 Ours triple is an independent predecessor, not rerun.
All jobs share the original exclusive lock, full loading/foreign/private-MPS
guards, twelve idle checks, 250W and original bounded single-point deadlines.
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
from gpu_numerical_equivalence_20261025 import GPU,sha,write,source_inventory,t12_proof,lgvq_proof
from finite_queue_budget import valid_window,point_deadline


def previous_finished(root):
    try:s=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    if s['status'] not in ('complete','stopped','failed') or Path('/proc',str(s['pid'])).exists():return False
    if s['status']=='complete' and s.get('mps_released') is not True:raise ValueError('Previous private MPS not released')
    return True


def missing_plan(lgvq_only=False):
    prefix=[] if lgvq_only else [('t12_validation','t12',True,None)] + [
        (f'qwen_repeat{i:02d}','t12',False,i) for i in (1,2,3)] + [('lgvq_validation','lgvq',True,None)]
    return prefix+[(f'{target}_repeat{i:02d}','lgvq',False,i)
                   for target in ('temporal','spatial') for i in (1,2,3)]


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('t12-output','lgvq-output','previous-cost'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--queue-seconds',type=int,default=21600)
    p.add_argument('--lgvq-only',action='store_true')
    p.add_argument('--retained-lgvq-proof',type=Path)
    p.add_argument('--retained-lgvq-proof-sha256')
    args=p.parse_args();t12=args.t12_output.resolve();lgvq=args.lgvq_output.resolve()
    if (t12.exists() or lgvq.exists() or not t12.is_relative_to(REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation')
            or not lgvq.is_relative_to(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation')
            or not valid_window(args.deadline,args.queue_seconds)):raise ValueError('New explicit bounded missing-only roots required')
    retained=None
    if args.lgvq_only:
        if (args.retained_lgvq_proof is None or args.retained_lgvq_proof_sha256 is None
                or not args.retained_lgvq_proof.resolve().is_relative_to(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation')
                or sha(args.retained_lgvq_proof)!=args.retained_lgvq_proof_sha256):raise ValueError('Exact retained original GPU proof required')
        lgvq_proof(args.retained_lgvq_proof);retained=args.retained_lgvq_proof.resolve()
    elif args.retained_lgvq_proof is not None or args.retained_lgvq_proof_sha256 is not None:
        raise ValueError('Do not silently accept an unused retained proof')
    t12.mkdir(parents=True);lgvq.mkdir(parents=True);frozen=source_inventory();child=None;active=False
    for name in ('measure_t12_cost.py','audit_t12_cost.py','measure_lgvq_cost.py','audit_lgvq_cost.py',
                 't12_baseline_live.py','t12_live_frontend.py','lgvq_live_frontend.py','stage_meter.py',
                 'gpu_checked_cost_entry_20261025.py','gpu_numerical_equivalence_20261025.py','run_temporal_remainder.py'):
        file=HERE/name;frozen[file.relative_to(REPO).as_posix()]=sha(file)
    state=dict(status='waiting_previous_Ours_cost_release',pid=os.getpid(),deadline=args.deadline,queue_seconds=args.queue_seconds,
        previous_cost=str(args.previous_cost.resolve()),t12_root=str(t12),lgvq_root=str(lgvq),completed=[],sources=frozen,
        command=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        CPU_inference_not_a_global_timing_gate=True,formal_cost_protocol_unchanged=True)
    if retained is not None:state.update(retained_lgvq_proof=str(retained),retained_lgvq_proof_sha256=sha(retained))
    def save():
        write(t12/'suite_status.json',state);write(lgvq/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-120:raise TimeoutError('Finite missing-only GPU queue budget')
        for name,digest in frozen.items():
            if sha(REPO/name)!=digest:raise ValueError('Active frozen source changed: '+name)
        if retained is not None and sha(retained)!=args.retained_lgvq_proof_sha256:raise ValueError('Retained original GPU proof changed')
    save()
    try:
        while not previous_finished(args.previous_cost):stable();time.sleep(5)
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=t12
        with (REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            idle=0
            while idle<12:
                stable();foreign=engine.base.guard_records(GPU)
                if foreign:idle=0;state.update(status='waiting_foreign_process',foreign=foreign)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if foreign else 10)
            control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,
                timeout=5,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=engine.PIPE))
            write(t12/'private_control_before.json',dict(returncode=control.returncode,stdout=control.stdout,stderr=control.stderr))
            if control.returncode==0:raise ValueError('Do not reuse a prior private MPS daemon')
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            t12python='/DATA/DATA1/guest3/t12_assets/venv/bin/python';xml='/home/guest3/miniconda3/envs/xml/bin/python'
            tproof=t12/'native_GPU_contract';lproof=retained.parent if retained is not None else lgvq/'native_GPU_contract'
            inputs=REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation/a100_shared_reference_files_20261025_01'
            plan=[(name,task,(tproof if task=='t12' else lproof) if validation else (t12 if task=='t12' else lgvq)/name,validation,repeat)
                  for name,task,validation,repeat in missing_plan(args.lgvq_only)]
            for name,task,folder,validation,repeat in plan:
                stable();python=t12python if task=='t12' else xml;deadline=point_deadline(args.deadline)
                if validation:
                    cmd=[python,'-u',str(HERE/'gpu_numerical_equivalence_20261025.py'),'--task',task,'--output',str(folder),'--deadline',str(deadline)]
                else:
                    cmd=[python,'-u',str(HERE/'gpu_checked_cost_entry_20261025.py'),'--task',task,'--output',str(folder),
                        '--repeat',str(repeat),'--deadline',str(deadline),'--cpu-proof',str((tproof if task=='t12' else lproof)/'contract.json')]
                    if task=='t12':cmd+=['--system','qwen','--inputs',str(inputs)]
                    else:cmd+=['--target',name.split('_')[0]]
                write(folder.parent/(name+'.command.json'),cmd);observations=[]
                with (folder.parent/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='GPU_validation' if validation else 'running',phase=name,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            stable()
                            if not (folder/'admission_bound.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(folder.parent/(name+'.foreign_interruption.json'),snapshot)
                                    raise RuntimeError('Foreign A100 during original loading/validation')
                            time.sleep(1)
                    except BaseException:
                        if child.poll() is None:loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'gpu_numerical_equivalence_20261025.py' if validation else 'gpu_checked_cost_entry_20261025.py');child.wait(timeout=20)
                        raise
                    finally:write(folder.parent/(name+'.loading_process_audit.json'),observations)
                if child.returncode:raise RuntimeError(name+' failed; preserve evidence, no blind retry')
                if validation:
                    if task=='t12':t12_proof(folder/'contract.json','qwen')
                    else:lgvq_proof(folder/'contract.json')
                else:
                    auditcmd=[python,str(HERE/'gpu_checked_cost_entry_20261025.py'),'--task',task,'--audit','--folder',str(folder),
                              '--audit-output',str(folder.parent/(name+'.cpu_audit.json'))]
                    checked=subprocess.run(auditcmd,cwd=REPO,capture_output=True,text=True,timeout=600)
                    if checked.returncode:
                        write(folder.parent/(name+'.audit_error.json'),dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
                        raise RuntimeError(name+' independent raw audit failed')
                state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as e:state.update(status='stopped',error=repr(e));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as e:state['release_error']=repr(e)
        save()


if __name__=='__main__':main()
