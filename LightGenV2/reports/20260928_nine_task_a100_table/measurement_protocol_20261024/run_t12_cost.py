"""Only missing original T12 cost points, after actual two-system CPU proofs.

Uses the same lab lock as all CPU/GPU queues. A waiting LGVQ CPU job may run
first; no heavy CPU proof and GPU cost ever overlap on this server.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_t12_cost import TASK,GPU,MAX_QUEUE_SECONDS,proof_contract,sha,write
from finite_queue_budget import valid_window,point_deadline
from t12_reference_files import load_bundle


def point_plan(names):
    if not names or len(set(names))!=len(names):raise ValueError('Unique missing points required')
    result=[]
    for name in names:
        match=re.fullmatch(r'(ours|qwen)_repeat(0[123])',name)
        if not match:raise ValueError('Only original two systems/three fresh repeats')
        result.append((name,match[1],int(match[2])))
    return result


def predecessor_ready(state,proof,digest):
    if state.get('status') in ('failed','stopped'):
        raise RuntimeError('Original CPU predecessor stopped; preserve and diagnose/rebind missing work')
    if state.get('status')!='complete':return False
    if state.get('contract_sha256')!=digest or proof.get('status')!='passed' or proof.get('no_gpu') is not True:
        raise ValueError('Actual CPU proof/parent SHA required')
    return True


def predecessor_file_ready(root,system):
    try:state=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    if state.get('status')!='complete':return predecessor_ready(state,{},None)
    path=root/'original_live_CPU_contract/contract.json'
    return predecessor_ready(state,proof_contract(path,system),sha(path))


def input_proof(root,manifest):
    """Read only an earlier independent receipt, never warm PNGs at admission."""
    path=root/'server_independent_input_audit01.json'
    value=json.loads(path.read_text(encoding='utf8'))
    if (value.get('status')!='passed' or value.get('samples')!=200
            or value.get('input_manifest_sha256')!=sha(manifest)
            or value.get('no_model_or_GPU_forward') is not True):
        raise ValueError('Actual shared input preparation/audit required')
    return path,sha(path)


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','inputs','ours-cpu','qwen-cpu'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--queue-seconds',type=int,default=MAX_QUEUE_SECONDS)
    parser.add_argument('--points',nargs='+',default=[f'{s}_repeat{i:02d}' for s in ('ours','qwen') for i in (1,2,3)])
    args=parser.parse_args();root=args.output.resolve();plan=point_plan(args.points)
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not valid_window(args.deadline,args.queue_seconds):
        raise ValueError('New bounded immutable canonical cost queue required')
    for path in (args.ours_cpu,args.qwen_cpu,args.inputs):
        if not path.resolve().is_relative_to(TASK/'runs/simulation'):raise ValueError('Original canonical artifacts required')
    load_bundle(args.inputs);manifest=args.inputs/'input_manifest.json'
    input_receipt,input_digest=input_proof(args.inputs,manifest)
    frozen={}
    for path in (args.ours_cpu,args.qwen_cpu):
        frozen.update(json.loads((path/'suite_status.json').read_text())['sources'])
    for name in ('run_t12_cost.py','measure_t12_cost.py','audit_t12_cost.py','t12_reference_files.py',
                 't12_baseline_live.py','t12_live_frontend.py','run_temporal_remainder.py','finite_queue_budget.py'):
        p=HERE/name;frozen[p.relative_to(REPO).as_posix()]=sha(p)
    root.mkdir(parents=True);active=False;child=None
    state=dict(status='waiting_original_CPU_proofs',pid=os.getpid(),deadline=args.deadline,queue_seconds=args.queue_seconds,sources=frozen,
        command=sys.argv,points=args.points,completed=[],ours_CPU=str(args.ours_cpu.resolve()),qwen_CPU=str(args.qwen_cpu.resolve()),
        input_root=str(args.inputs.resolve()),input_manifest_sha256=sha(manifest),
        earlier_shared_input_audit_sha256=input_digest,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-120:raise TimeoutError('Finite original T12 missing cost budget')
        if any(sha(REPO/name)!=digest for name,digest in frozen.items()):raise ValueError('Active frozen T12 cost source changed')
        if sha(manifest)!=state['input_manifest_sha256']:raise ValueError('Original input bundle changed')
        if sha(input_receipt)!=input_digest:raise ValueError('Independent shared input audit changed')
    save()
    try:
        while not (predecessor_file_ready(args.ours_cpu,'ours') and predecessor_file_ready(args.qwen_cpu,'qwen')):
            stable();time.sleep(15)
        from audit_t12_cost import audit
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
        lockpath=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
        with lockpath.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:state.update(status='waiting_shared_lock');save();time.sleep(10)
            idle=0
            while idle<12:
                stable();records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign=records)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if records else 10)
            # Pixel/SHA audit is an earlier independent preparation stage.
            # Do not pre-read the first200 files immediately before timing.
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for name,system,repeat in plan:
                stable();folder=root/name
                proofroot=args.ours_cpu if system=='ours' else args.qwen_cpu
                proof=proofroot/'original_live_CPU_contract/contract.json'
                command=['/DATA/DATA1/guest3/t12_assets/venv/bin/python','-u',str(HERE/'measure_t12_cost.py'),
                    '--system',system,'--repeat',str(repeat),'--output',str(folder),'--inputs',str(args.inputs.resolve()),
                    '--cpu-proof',str(proof.resolve()),'--deadline',str(point_deadline(args.deadline))]
                write(root/(name+'.command.json'),command);observations=[]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            stable()
                            if not (folder/'admission_bound.json').exists():
                                observed=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(observed)
                                if observed['foreign_gpu_processes'] or observed['foreign_mps_clients']:
                                    write(root/(name+'.foreign_interruption.json'),observed)
                                    raise RuntimeError('Foreign A100 process during original T12 loading')
                            time.sleep(1)
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_t12_cost.py');child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if child.returncode:raise RuntimeError(name+' failed; preserve raw and diagnose before immutable missing-only continuation')
                write(root/(name+'.cpu_audit.json'),audit(folder));state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
