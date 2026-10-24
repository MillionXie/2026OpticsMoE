"""Reuse qualified first two rounds; wait for exclusivity and measure only third.

Original bank and CPU contract remain immutable referenced data, not copied
projects or additional gallery/query warmup. Never stop another user's process.
"""
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
from measure_abo_image_qwen import TASK,GPU,source_map,sha,write
from run_abo_text_qwen import MODEL,PIPE
from audit_abo_image_qwen import audit
import run_temporal_remainder as loading


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();root=args.output.resolve()
    if root.exists() or not root.is_relative_to(TASK/'runs/simulation') or not 300<args.deadline-time.time()<=7200:
        raise ValueError('New immutable continuation and finite budget required')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    engine.GPU,engine.PIPE,engine.ROOT=GPU,PIPE,root
    old=TASK/'runs/simulation/a100_image_input_inclusive_20261024_02';data=REPO/'data/abo_easy100_dataset_20260906'
    old_state=json.loads((old/'suite_status.json').read_text())
    if (old_state['status']!='stopped' or old_state['completed']!=['offline_title_bank','qwen_repeat01','qwen_repeat02']
            or not (old/'qwen_repeat03.foreign_loading_interruption.json').is_file()
            or json.loads((old/'qwen_repeat03/status.json').read_text())['samples']!=0):
        raise ValueError('Not the verified external-loading-only third-round interruption')
    frozen=source_map()
    for path in (Path(__file__),HERE/'audit_abo_image_qwen.py',HERE/'run_temporal_remainder.py'):
        frozen[str(path.relative_to(REPO))]=sha(path)
    def stable():
        if any(sha(REPO/name)!=digest for name,digest in frozen.items()):raise ValueError('Frozen source changed')
    with (REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(parents=True);active=False;child=None
        state=dict(status='auditing_reused_rounds',pid=os.getpid(),completed=[],sources=frozen,command=sys.argv,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),deadline=args.deadline,
            original_bank=str(old/'offline_title_bank'),original_CPU_contract=str(old/'original_image_cpu_contract/contract.json'))
        observations=[]
        def save():write(root/'suite_status.json',state)
        save()
        try:
            reused=[dict(path=str(old/f'qwen_repeat{i:02d}'),audit=audit(old/f'qwen_repeat{i:02d}',data)) for i in (1,2)]
            write(root/'reused_qualified_rounds.json',reused);state['reused_rounds']=[r['path'] for r in reused]
            idle=0
            while idle<3:
                stable()
                if time.time()>args.deadline-300:raise TimeoutError('No new point after finite exclusive-admission deadline')
                records=engine.base.guard_records(GPU)
                if records:idle=0;state.update(status='waiting_foreign_process',foreign_processes=records)
                else:
                    try:validate_idle_memory(GPU)
                    except RuntimeError:idle=0;state.update(status='waiting_memory_release')
                    else:idle+=1;state.update(status='confirming_exclusive_idle',idle_checks=idle,foreign_processes=[])
                observations.append(dict(unix=time.time(),processes=records,status=state['status']));save()
                time.sleep(10 if not records else 30)
            engine.daemon(True);active=True
            folder=root/'qwen_repeat03'
            command=[sys.executable,'-u',str(HERE/'measure_abo_image_qwen.py'),'--model',MODEL,'--data',str(data),
                '--bank',str(old/'offline_title_bank'),'--contract',str(old/'original_image_cpu_contract/contract.json'),
                '--repeat','3','--output',str(folder)]
            write(root/'qwen_repeat03.command.json',command)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            with (root/'qwen_repeat03.log').open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',phase='qwen_repeat03',child_pid=child.pid);save()
                end=min(args.deadline,time.time()+1800)
                while True:
                    stable()
                    if time.time()>end:raise TimeoutError('Finite worker deadline')
                    try:code=child.wait(timeout=1);break
                    except subprocess.TimeoutExpired:pass
                    if not (folder/'admission_bound.json').exists():
                        snapshot=loading.process_snapshot(GPU,child.pid,PIPE);observations.append(snapshot)
                        if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                            write(root/'qwen_repeat03.foreign_loading_interruption.json',snapshot)
                            raise RuntimeError('Foreign A100 process during loading; never mix')
            if code:raise RuntimeError(f'Missing third round exited{code}; diagnose and preserve before retry')
            result=audit(folder,data);write(root/'qwen_repeat03.cpu_audit.json',result)
            state.update(status='complete',completed=['qwen_repeat03'],finished_unix=time.time())
        except BaseException as error:
            state.update(status='stopped',error=repr(error))
            if child is not None and child.poll() is None:
                loading.terminate_own_loading_worker({'worker_pid':child.pid},root/'qwen_repeat03','measure_abo_image_qwen.py')
                child.wait(timeout=20)
            raise
        finally:
            write(root/'admission_and_loading_process_audit.json',observations)
            if active:
                try:engine.daemon(False);state['mps_released']=True
                except BaseException as error:state['release_error']=repr(error)
            save()


if __name__=='__main__':main()
