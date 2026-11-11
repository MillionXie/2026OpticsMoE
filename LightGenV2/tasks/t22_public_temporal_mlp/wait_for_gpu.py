"""Bounded internal job queue. Never kills/shares occupied GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def idle_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'],text=True)
    occupied={row.split(',')[0].strip() for row in processes.splitlines() if ',' in row}
    eligible=[]
    for line in output.splitlines():
        uuid,name,memory,util=[x.strip() for x in line.split(',')]
        if 'RTX 4090' in name and float(memory)<=256 and float(util)<=5 and uuid not in occupied:
            eligible.append(uuid)
    return eligible


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dispatch-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/trial_relu_s163.json')
    p.add_argument('--max-wait-seconds',type=int,default=1800)
    p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--initialize',type=Path)
    a=p.parse_args();a.dispatch_dir.mkdir(parents=True,exist_ok=False)
    status=a.dispatch_dir/'status.json';started=time.time();candidate=None;count=0
    while time.time()-started<a.max_wait_seconds:
        choices=idle_gpus()
        if candidate in choices:count+=1
        else:candidate=choices[0] if choices else None;count=1 if candidate else 0
        record={'state':'waiting_for_idle_RTX4090','pid':os.getpid(),'eligible':choices,
                'candidate':candidate,'consecutive_checks':count,'elapsed_seconds':time.time()-started}
        status.write_text(json.dumps(record,indent=2)+'\n')
        if count>=3:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=candidate;env['PYTHONDONTWRITEBYTECODE']='1'
            command=[sys.executable,str(Path(__file__).with_name('run.py')),'--paths',str(a.paths),
                     '--config',str(a.config),'--out',str(a.out),'--device','cuda']
            if a.initialize:command.extend(['--initialize',str(a.initialize)])
            with (a.dispatch_dir/'training.log').open('w',buffering=1) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                status.write_text(json.dumps({'state':'training','pid':os.getpid(),'training_pid':child.pid,
                    'gpu_uuid':candidate,'command':command},indent=2)+'\n')
                returncode=child.wait()
            status.write_text(json.dumps({'state':'completed' if returncode==0 else 'failed',
                'returncode':returncode,'gpu_uuid':candidate,'training_pid':child.pid,
                'elapsed_seconds':time.time()-started},indent=2)+'\n')
            return returncode
        time.sleep(a.poll_seconds)
    status.write_text(json.dumps({'state':'not_started_no_idle_gpu','pid':os.getpid(),
                                 'elapsed_seconds':time.time()-started},indent=2)+'\n')
    return 2


if __name__=='__main__':raise SystemExit(main())
