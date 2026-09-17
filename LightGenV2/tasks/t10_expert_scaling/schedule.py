"""Bounded three-GPU queue. Every child exits between jobs; failures stop queue."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .train import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--gpus',type=int,nargs='+',required=True)
    ap.add_argument('--mode',choices=['calibration','scan'],default='calibration')
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();assert 1<=len(a.gpus)<=3 and len(set(a.gpus))==len(a.gpus)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir()
    arms=['moe_oeo','d2nn_total_parameter','d2nn_same_aperture']
    if a.mode=='calibration':
        jobs=[dict(arch=arm,experts=9,top_k=9,layers=depth,lr=lr) for depth in [4,6] for lr in [.001,.002] for arm in arms]
    else:
        if a.calibration is None:raise ValueError('--calibration is required')
        save(a.out/'status.json',dict(state='waiting_for_calibration',gpu_processes_started=0))
        while True:
            state=json.loads((a.calibration/'status.json').read_text())
            if state['state']=='complete':break
            if state['state']!='training':raise RuntimeError(f'Calibration stopped: {state}')
            time.sleep(30)
        # Wait until the parent has joined every child and recorded release.
        while not (a.calibration/'release_check.json').exists():time.sleep(5)
        records=[]
        for folder in a.calibration.iterdir():
            if not (folder/'result.json').exists():continue
            result=json.loads((folder/'result.json').read_text())
            meta=json.loads((folder/'metadata.json').read_text())
            records.append(dict(folder=str(folder),result=result,args=meta['arguments']))
        if len(records)!=12:raise ValueError('Expected all 12 completed calibration results')
        selected={}
        for depth in [4,6]:
            selected[depth]={arm:min([r for r in records if r['args']['arch']==arm and r['args']['layers']==depth],
                key=lambda r:r['result']['val']['macro_nll']) for arm in arms}
        means={d:sum(r['result']['val']['macro_nll'] for r in by_arm.values())/len(arms) for d,by_arm in selected.items()}
        depth=min(means,key=lambda d:(means[d],d))
        lr_by_arm={arm:r['args']['lr'] for arm,r in selected[depth].items()}
        save(a.out/'selection_lock.json',dict(common_layers=depth,lr_by_arm=lr_by_arm,
            validation_mean_macro_nll=means,selected=selected[depth],test_read=False))
        # Roughly geometric spacing, with quarter/half/dense operating points.
        grid={4:[1,2,4],9:[1,3,5,9],16:[1,4,8,16],25:[1,6,12,25],36:[1,9,18,36],49:[1,12,24,49]}
        jobs=[]
        for n in [4,16,25,36,49,9]:
            for arm in arms:
                for k in (grid[n] if arm=='moe_oeo' else [n]):
                    if n==9 and (arm!='moe_oeo' or k==9):continue
                    jobs.append(dict(arch=arm,experts=n,top_k=k,layers=depth,lr=lr_by_arm[arm]))
        save(a.out/'scan_design.json',dict(top_k_grid=grid,reused_calibration=selected[depth],
             missing_gpu_ablation_jobs=len(jobs),expert_side=224,logical_pitch_um=17,phase_device_pitch_um=8,gap=30))
    for j in jobs:j['name']=f"{j['arch']}_N{j['experts']}_k{j['top_k']}_L{j['layers']}_lr{j['lr']}_s17"
    save(a.out/'jobs.json',jobs);save(a.out/'identity.json',dict(pid=os.getpid(),gpus=a.gpus,command=sys.argv,
         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),max_gpus=3))
    if a.dry_run:
        save(a.out/'status.json',dict(state='dry_run_complete',jobs=len(jobs),gpu_processes_started=0))
        return
    running={};finished=[];failed=[]
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while jobs or running:
            for gpu in a.gpus:
                if gpu in running or not jobs:continue
                # Do not start on a GPU newly occupied by another process.
                memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if memory>200:continue
                j=jobs.pop(0);cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.train',
                    '--data',str(a.data),'--out',str(a.out/j['name']),'--arch',j['arch'],
                    '--experts',str(j['experts']),'--top-k',str(j['top_k']),'--layers',str(j['layers']),
                    '--lr',str(j['lr']),'--microbatch','2','--epochs','60','--seed','17']
                uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid;env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
                log=(a.out/'logs'/(j['name']+'.log')).open('w')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(proc,log,j);save(a.out/f'gpu{gpu}.json',dict(pid=proc.pid,gpu_uuid=uuid,job=j,command=cmd))
            for gpu,(proc,log,j) in list(running.items()):
                if proc.poll() is None:continue
                log.close();del running[gpu]
                if proc.returncode!=0:failed.append(dict(job=j,exit_code=proc.returncode));raise RuntimeError(f'Job failed: {j}')
                finished.append(j)
            save(a.out/'status.json',dict(state='training',completed=len(finished),remaining=len(jobs),
                 running={str(g):dict(pid=p.pid,job=j) for g,(p,l,j) in running.items()},failed=failed))
            if jobs or running:time.sleep(10)
        save(a.out/'status.json',dict(state='complete',completed=len(finished),running={},failed=failed))
    except BaseException as error:
        save(a.out/'status.json',dict(state='failed_or_interrupted',completed=len(finished),
             error=str(error),failed=failed))
        raise
    finally:
        for proc,log,j in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
        save(a.out/'release_check.json',dict(own_children_exited=True,remaining_gpu_processes=snapshot,failed=failed))


if __name__=='__main__':main()
