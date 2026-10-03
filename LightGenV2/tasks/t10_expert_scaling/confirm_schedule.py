"""Bounded confirmatory queue for the corrected dynamic-router profile."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
from .train import save

K = {4: 3, 16: 16, 25: 12, 49: 24, 100: 50}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True); p.add_argument('--seeds',type=int,nargs='+',default=[17,27,37])
    p.add_argument('--experts',type=int,nargs='+',default=[4,16,25,49,100]); p.add_argument('--epochs',type=int,default=60)
    a=p.parse_args(); assert 1<=len(a.gpus)<=4 and len(set(a.gpus))==len(a.gpus)
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'logs').mkdir(exist_ok=True)
    jobs=[]
    for seed in a.seeds:
        for n in a.experts:
            k=K[n];
            for arch,top in [('moe_oeo',k),('d2nn_expert_global',n)]:
                jobs.append(dict(arch=arch,experts=n,top_k=top,layers=4,lr=.002,seed=seed,
                    name=f'{arch}_N{n}_k{top}_L4_lr0.002_s{seed}'))
    save(a.out/'jobs.json',jobs); save(a.out/'identity.json',dict(command=os.sys.argv,gpus=a.gpus,seeds=a.seeds,experts=a.experts,profile='dynamic_router_full_aperture_v1'))
    pending=list(jobs); running={}; done=[]; failed=[]
    try:
        while pending or running:
            for gpu in a.gpus:
                if gpu in running or not pending: continue
                j=pending.pop(0); folder=a.out/j['name']
                if (folder/'result.json').exists(): done.append(j); continue
                uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID')
                log=(a.out/'logs'/(j['name']+'.log')).open('w')
                cmd=[os.sys.executable,'-u','-m','LightGenV2.tasks.t10_expert_scaling.train','--data',str(a.data),'--out',str(folder),'--arch',j['arch'],'--experts',str(j['experts']),'--top-k',str(j['top_k']),'--layers','4','--lr','.002','--microbatch','2','--epochs',str(a.epochs),'--seed',str(j['seed'])]
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(proc,log,j); save(a.out/f'gpu{gpu}.json',dict(pid=proc.pid,gpu_uuid=uuid,job=j,command=cmd))
            for gpu,(proc,log,j) in list(running.items()):
                if proc.poll() is None: continue
                log.close(); del running[gpu]
                if proc.returncode: failed.append(dict(job=j,exit_code=proc.returncode)); raise RuntimeError(j)
                done.append(j)
            save(a.out/'status.json',dict(state='training',completed=len(done),remaining=len(pending),running={str(g):j for g,(p,l,j) in running.items()},failed=failed))
            if pending or running: time.sleep(10)
        save(a.out/'status.json',dict(state='complete',completed=len(done),remaining=0,running={},failed=failed))
    finally:
        for proc,log,j in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(30)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            log.close()
        save(a.out/'release_check.json',dict(own_children_exited=True,remaining_gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True),failed=failed))

if __name__=='__main__': main()
