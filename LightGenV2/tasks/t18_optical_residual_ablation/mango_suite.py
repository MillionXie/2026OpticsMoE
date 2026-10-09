"""Prepare a new licensed dataset on CPU then run paired depths on at most two GPUs."""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpus',nargs=2,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    def state(value): (a.out/'status.json').write_text(json.dumps(value,indent=2))
    (a.out/'metadata.json').write_text(json.dumps(dict(command=sys.argv,pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    try:
        state(dict(state='preparing_data',gpu_processes=0))
        subprocess.run([sys.executable,'-u',str(here/'prepare_mango.py'),'--out',str(a.data_root)],check=True)
        data=a.data_root/'mango_variety_fixed_split.npz'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus[0])
        subprocess.run([sys.executable,str(here/'run.py'),'--phase','smoke','--rho','.3',
            '--depth','2','--dataset','mango_variety','--data',str(data),'--out',str(a.out/'smoke')],env=env,check=True)
        for depth in [2,4,6]:
            state(dict(state='paired_training_and_evaluation',depth=depth))
            subprocess.run([sys.executable,'-u',str(here/'campaign.py'),'--dataset','mango_variety',
                '--depth',str(depth),'--data',str(data),'--out',str(a.out/f'L{depth}'),
                '--gpus',*a.gpus],check=True)
        results={str(d):json.loads((a.out/f'L{d}'/'results.json').read_text()) for d in [2,4,6]}
        (a.out/'results.json').write_text(json.dumps(results,indent=2))
        state(dict(state='complete',gpu_released_on_exit=True))
    except BaseException as error:
        state(dict(state='failed',error=repr(error)));raise


if __name__=='__main__':main()
