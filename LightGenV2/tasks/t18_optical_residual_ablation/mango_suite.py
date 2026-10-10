"""Prepare a new licensed dataset on CPU then run paired depths on at most two GPUs."""
import argparse
import json
import subprocess
import sys
import os
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpus',nargs='+',required=True)
    p.add_argument('--wait-preparation-pid',type=int)
    p.add_argument('--prepared',action='store_true')
    p.add_argument('--epochs',type=int,choices=[30,100],default=30)
    p.add_argument('--profile',choices=['base','lr3'],default='base')
    p.add_argument('--validation-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert len(a.gpus) in (2,3) and len(set(a.gpus))==len(a.gpus)
    here=Path(__file__).resolve().parent
    def state(value): (a.out/'status.json').write_text(json.dumps(value,indent=2))
    (a.out/'metadata.json').write_text(json.dumps(dict(command=sys.argv,pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    try:
        state(dict(state='preparing_data',gpu_processes=0))
        if a.prepared:
            assert (a.data_root/'data_manifest.json').exists()
            assert (a.data_root/'mango_variety_fixed_split.npz').exists()
        elif a.wait_preparation_pid:
            deadline=time.monotonic()+7200
            while not (a.data_root/'data_manifest.json').exists():
                assert time.monotonic()<deadline,'Preparation timed out'
                assert Path(f'/proc/{a.wait_preparation_pid}').exists(),'Existing preparation exited without manifest'
                time.sleep(10)
        else:
            subprocess.run([sys.executable,'-u',str(here/'prepare_mango.py'),'--out',str(a.data_root)],check=True)
        data=a.data_root/'mango_variety_fixed_split.npz'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus[0])
        subprocess.run([sys.executable,str(here/'run.py'),'--phase','smoke','--rho','.3',
            '--depth','2','--dataset','mango_variety','--data',str(data),'--out',str(a.out/'smoke'),'--epochs',str(a.epochs),'--profile',a.profile],env=env,check=True)
        if len(a.gpus)==3:
            workers=[]
            try:
                for depth,gpu in zip([2,4,6],a.gpus):
                    log=(a.out/f'L{depth}.log').open('w')
                    cmd=[sys.executable,'-u',str(here/'campaign.py'),'--dataset','mango_variety',
                        '--depth',str(depth),'--data',str(data),'--out',str(a.out/f'L{depth}'),
                        '--gpus',gpu,'--epochs',str(a.epochs),'--profile',a.profile]
                    if a.validation_only:cmd.append('--validation-only')
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    workers.append((proc,log))
                state(dict(state='parallel_depth_training',depths=[2,4,6],gpu_limit=3))
                for proc,log in workers:
                    assert proc.wait()==0,proc.pid
                    log.close()
            except BaseException:
                for proc,log in workers:
                    if proc.poll() is None:
                        # Campaign may have a training child; terminate its children too.
                        subprocess.run(['pkill','-TERM','-P',str(proc.pid)],check=False)
                        proc.terminate();proc.wait()
                    log.close()
                raise
        for depth in ([] if len(a.gpus)==3 else [2,4,6]):
            state(dict(state='paired_training_and_evaluation',depth=depth))
            cmd=[sys.executable,'-u',str(here/'campaign.py'),'--dataset','mango_variety',
                '--depth',str(depth),'--data',str(data),'--out',str(a.out/f'L{depth}'),
                '--epochs',str(a.epochs),'--profile',a.profile,'--gpus',*a.gpus]
            if a.validation_only:cmd.append('--validation-only')
            subprocess.run(cmd,check=True)
        filename='validation_results.json' if a.validation_only else 'results.json'
        results={str(d):json.loads((a.out/f'L{d}'/filename).read_text()) for d in [2,4,6]}
        (a.out/filename).write_text(json.dumps(results,indent=2))
        state(dict(state='validation_complete' if a.validation_only else 'complete',gpu_released_on_exit=True))
    except BaseException as error:
        state(dict(state='failed',error=repr(error)));raise


if __name__=='__main__':main()
