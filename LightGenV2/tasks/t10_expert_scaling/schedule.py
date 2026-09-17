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
    ap.add_argument('--mode',choices=['calibration','pilot'],default='calibration')
    a=ap.parse_args();assert 1<=len(a.gpus)<=3 and len(set(a.gpus))==len(a.gpus)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir()
    arms=['moe_oeo','d2nn_total_parameter','d2nn_same_aperture']
    if a.mode=='calibration':
        jobs=[dict(arch=arm,experts=9,top_k=9,layers=depth,lr=lr) for depth in [4,6] for lr in [.001,.002] for arm in arms]
    else:
        raise ValueError('Pilot requires a locked calibration selection; use calibration first')
    for j in jobs:j['name']=f"{j['arch']}_N{j['experts']}_k{j['top_k']}_L{j['layers']}_lr{j['lr']}_s17"
    save(a.out/'jobs.json',jobs);save(a.out/'identity.json',dict(pid=os.getpid(),gpus=a.gpus,command=sys.argv,
         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),max_gpus=3))
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
