"""Launch exactly three independent runs on verified idle physical GPU UUIDs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpus', nargs=3, required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--run-prefix', required=True)
    p.add_argument('--profile',choices=['four_top2','sixteen_dense'],default='four_top2')
    args=p.parse_args()
    if len(set(args.gpus))!=3: raise ValueError('three distinct GPU UUIDs required')
    root=Path(__file__).resolve().parents[3]
    task=Path(__file__).resolve().parent
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used',
                                 '--format=csv,noheader,nounits'],text=True)
    free={row.split(',')[0].strip():int(row.split(',')[1]) for row in raw.splitlines()}
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                     '--format=csv,noheader'],text=True)
    for uuid in args.gpus:
        if uuid not in free or free[uuid]>100 or uuid in running:
            raise RuntimeError(f'GPU not idle: {uuid}')
    launches=[]
    for variant,uuid in zip(('optical','electronic','d2nn'),args.gpus):
        out=task/'runs'/'simulation'/f'{args.run_prefix}_{variant}'
        if out.exists(): raise FileExistsError(out)
    for variant,uuid in zip(('optical','electronic','d2nn'),args.gpus):
        run=f'{args.run_prefix}_{variant}'
        out=task/'runs'/'simulation'/run
        out.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','LightGenV2.tasks.t17_router_classification.train',
                 '--architecture',variant,'--profile',args.profile,'--data',args.data,'--manifest',args.manifest,
                 '--out',str(out)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
        env['OMP_NUM_THREADS']='4'
        with (out.parent/f'{run}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
        launches.append(dict(variant=variant,profile=args.profile,pid=process.pid,gpu_uuid=uuid,out=str(out)))
    receipt=task/'runs'/'simulation'/f'{args.run_prefix}_launch.json'
    receipt.write_text(json.dumps(launches,indent=2)+'\n')
    print(json.dumps(launches,indent=2))

if __name__=='__main__': main()
