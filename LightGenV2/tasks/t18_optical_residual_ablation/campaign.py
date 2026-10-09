"""Two-GPU fixed paired campaign; releases training GPUs before evaluations."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',nargs=2,required=True)
    a=p.parse_args()
    assert len(set(a.gpus))==2 and all(g.startswith('GPU-') for g in a.gpus)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True).splitlines()
    assert not set(a.gpus).intersection(x.strip() for x in occupied)
    a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'metadata.json',dict(command=sys.argv,pid=os.getpid(),gpus=a.gpus,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    run=Path(__file__).with_name('run.py')
    processes=[]
    try:
        for rho,gpu in zip([0.,.3],a.gpus):
            folder=a.out/f'rho{rho}'
            cmd=[sys.executable,'-u',str(run),'--phase','train','--rho',str(rho),
                '--data',str(a.data),'--out',str(folder)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
            log=(a.out/f'rho{rho}.log').open('w')
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((proc,log))
            save(a.out/f'process_rho{rho}.json',dict(pid=proc.pid,command=cmd,gpu_uuid=gpu))
        save(a.out/'status.json',dict(state='training',test_read=False))
        for proc,log in processes:
            assert proc.wait()==0,proc.pid
            log.close()
        results=[json.loads((a.out/f'rho{rho}'/'result.json').read_text()) for rho in [0.,.3]]
        assert results[0]['orders']==results[1]['orders']
        assert results[0]['transforms']==results[1]['transforms']
        assert results[0]['parameters']==results[1]['parameters']==1329544
        save(a.out/'selection_lock.json',dict(results=results,paired_orders=True,
            paired_transforms=True,criterion='each arm minimum validation balanced NLL'))
        save(a.out/'status.json',dict(state='locked_evaluation'))
        for rho,result in zip([0.,.3],results):
            folder=a.out/f'rho{rho}'
            cmd=[sys.executable,'-u',str(run),'--phase','evaluate','--rho',str(rho),
                '--data',str(a.data),'--out',str(a.out/f'evaluation_rho{rho}'),
                '--checkpoint',str(folder/result['name']/'best_checkpoint.pt')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus[0])
            subprocess.run(cmd,env=env,check=True)
        metrics={str(rho):json.loads((a.out/f'evaluation_rho{rho}'/'metrics.json').read_text())
            for rho in [0.,.3]}
        save(a.out/'results.json',metrics)
        save(a.out/'status.json',dict(state='complete',test_evaluations_per_arm=1,
            gpu_released_on_exit=True,time=time.time()))
    except BaseException as error:
        for proc,log in processes:
            if proc.poll() is None:
                proc.terminate();proc.wait()
            log.close()
        save(a.out/'status.json',dict(state='failed',error=repr(error),time=time.time()))
        raise


if __name__=='__main__':
    main()
