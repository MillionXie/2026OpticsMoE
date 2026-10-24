"""Bounded missing-only original LGVQ costs after actual CPU proof/T12 release."""
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
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from measure_lgvq_cost import TASK,GPU,MAX_QUEUE_SECONDS,live,proof_contract,sha,write


def point_plan(names):
    if not names or len(set(names))!=len(names):raise ValueError('Unique missing original points only')
    result=[]
    for name in names:
        match=re.fullmatch(r'(temporal|spatial)_repeat(0[123])',name)
        if not match:raise ValueError('Only original temporal/spatial three repeats')
        result.append((name,match[1],int(match[2])))
    return result


def actual_cpu_ready(root):
    try:state=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    if state['status'] in ('failed','stopped'):raise RuntimeError('Original CPU predecessor stopped; preserve/rebind missing work')
    if state['status']!='complete':return False
    path=root/'original_live_CPU_contract/contract.json';proof_contract(path)
    if sha(path)!=state.get('contract_sha256'):raise ValueError('Actual CPU proof/parent SHA differs')
    return True


def previous_cost_finished(state,alive):
    if state.get('status') not in ('complete','stopped','failed') or alive:return False
    # A failed prior cost does not invalidate an independent already-passed CPU
    # proof. Actual empty GPU/private-MPS admission is checked again below.
    return True


def cost_file_finished(root):
    try:state=json.loads((root/'suite_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    return previous_cost_finished(state,Path('/proc',str(state['pid'])).exists())


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','cpu-root','previous-cost'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--points',nargs='+',default=[f'{t}_repeat{i:02d}' for t in ('temporal','spatial') for i in (1,2,3)])
    args=p.parse_args();root=args.output.resolve();plan=point_plan(args.points)
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation')
            or not 0<args.deadline-time.time()<=MAX_QUEUE_SECONDS
            or not args.cpu_root.resolve().is_relative_to(TASK/'runs/simulation')
            or not args.previous_cost.resolve().is_relative_to(REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation')):
        raise ValueError('Fresh finite explicit canonical missing-cost queue required')
    frozen=live.sources()
    for name in ('run_lgvq_cost.py','measure_lgvq_cost.py','audit_lgvq_cost.py','run_lgvq_live_contract.py',
                 'measure_t12_cost.py','audit_t12_cost.py','run_temporal_remainder.py'):
        file=HERE/name;frozen[file.relative_to(REPO).as_posix()]=sha(file)
    root.mkdir(parents=True);active=False;child=None
    state=dict(status='waiting_actual_LGVQ_CPU_proof',pid=os.getpid(),deadline=args.deadline,sources=frozen,
        CPU_predecessor=str(args.cpu_root.resolve()),cost_predecessor=str(args.previous_cost.resolve()),
        points=args.points,completed=[],command=sys.argv,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def save():write(root/'suite_status.json',state)
    def stable():
        if time.time()>args.deadline-120:raise TimeoutError('Finite original LGVQ missing-cost budget')
        if any(sha(REPO/p)!=h for p,h in frozen.items()):raise ValueError('Active frozen original LGVQ cost source changed')
    save()
    try:
        while not actual_cpu_ready(args.cpu_root):stable();time.sleep(15)
        state.update(status='waiting_previous_T12_cost_release');save()
        while not cost_file_finished(args.previous_cost):stable();time.sleep(15)
        from audit_lgvq_cost import audit
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
        import run_temporal_remainder as loading
        engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=root
        path=TASK/'runs/simulation/.temporal_telemetry_a100.lock'
        with path.open('a') as lock:
            while True:
                stable()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:state.update(status='waiting_shared_lock');save();time.sleep(10)
            idle=0
            while idle<12:
                stable();foreign=engine.base.guard_records(GPU)
                if foreign:idle=0;state.update(status='waiting_foreign_process',foreign=foreign)
                else:validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle,foreign=[])
                save();time.sleep(30 if foreign else 10)
            control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',capture_output=True,text=True,
                timeout=5,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=engine.PIPE))
            write(root/'private_control_before.json',dict(returncode=control.returncode,stdout=control.stdout,stderr=control.stderr))
            if control.returncode==0:raise ValueError('Previous private MPS daemon not yet released; do not reuse it')
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for name,target,repeat in plan:
                stable();folder=root/name
                command=[sys.executable,'-u',str(HERE/'measure_lgvq_cost.py'),'--target',target,'--repeat',str(repeat),
                    '--output',str(folder),'--cpu-proof',str(args.cpu_root.resolve()/'original_live_CPU_contract/contract.json'),
                    '--deadline',str(args.deadline)]
                write(root/(name+'.command.json'),command);observations=[]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            stable()
                            if not (folder/'admission_bound.json').exists():
                                snapshot=loading.process_snapshot(GPU,child.pid,engine.PIPE);observations.append(snapshot)
                                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                                    write(root/(name+'.foreign_interruption.json'),snapshot)
                                    raise RuntimeError('Foreign A100 during original LGVQ loading; stop only own worker')
                            time.sleep(1)
                    except BaseException:
                        if child.poll() is None:
                            loading.terminate_own_loading_worker({'worker_pid':child.pid},folder,'measure_lgvq_cost.py');child.wait(timeout=20)
                        raise
                    finally:write(root/(name+'.loading_process_audit.json'),observations)
                if child.returncode:raise RuntimeError(name+' failed; preserve/diagnose then immutable missing-only continuation')
                write(root/(name+'.cpu_audit.json'),audit(folder));state['completed'].append(name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


if __name__=='__main__':main()
