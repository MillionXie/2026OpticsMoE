"""Three independent residual-only continuations; no automatic test access."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--gpus',nargs=3,required=True)
    a=p.parse_args()
    assert len(set(a.gpus))==3 and all(g.startswith('GPU-') for g in a.gpus)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True).splitlines()
    assert not set(a.gpus).intersection(x.strip() for x in occupied)
    a.out.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(a.out/name).write_text(json.dumps(value,indent=2),encoding='utf8')
    save('metadata.json',dict(command=sys.argv,pid=os.getpid(),gpus=a.gpus,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    jobs=[]
    try:
        for depth,gpu in zip([2,4,6],a.gpus):
            checkpoint=a.parent/f'L{depth}'/'rho0.3'/f'moe_L{depth}_seed17'/'best_checkpoint.pt'
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('continue_residual.py')),
                 '--phase','train','--depth',str(depth),'--checkpoint',str(checkpoint),
                 '--data',str(a.data),'--out',str(a.out/f'L{depth}')]
            log=(a.out/f'L{depth}.log').open('w')
            proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                                  stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log))
            save(f'process_L{depth}.json',dict(pid=proc.pid,gpu_uuid=gpu,command=cmd))
        save('status.json',dict(state='residual_only_training',test_read=False,gpu_limit=3))
        for proc,log in jobs:
            assert proc.wait()==0,proc.pid
            log.close()
        results={str(d):json.loads((a.out/f'L{d}'/'result.json').read_text()) for d in [2,4,6]}
        save('validation_results.json',results)
        save('status.json',dict(state='validation_complete',test_read=False,gpu_released=True))
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
        save('status.json',dict(state='failed',error=repr(error)))
        raise


if __name__=='__main__':
    main()
