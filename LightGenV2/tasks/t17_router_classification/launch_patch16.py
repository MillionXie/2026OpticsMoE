"""Run optical/electronic first, then ordinary D2NN; at most two idle cards."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .prepare_patch16 import write

def idle_cards():
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={line.split(',')[0].strip():int(line.split(',')[1]) for line in usage.splitlines()}
    busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
    return {uuid for uuid,used in memory.items() if used<100 and uuid not in busy}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',nargs=2,required=True);a=p.parse_args()
    if len(set(a.gpus))!=2:raise ValueError('two distinct UUIDs required')
    if not set(a.gpus).issubset(idle_cards()):raise RuntimeError('requested cards are not idle')
    a.out.mkdir(parents=True,exist_ok=False)
    queue=['optical','electronic','d2nn'];running={};receipts=[];done=[]
    start=time.monotonic()
    while queue or running:
        for uuid,(proc,item) in list(running.items()):
            if proc.poll() is not None:
                item['exit_code']=proc.returncode
                done.append(item);del running[uuid]
                if proc.returncode:
                    write(a.out/'suite_status.json',dict(state='failed',completed=done,running=[i for p,i in running.values()],pending=queue))
                    raise RuntimeError(f'child failed: {item["architecture"]}')
        available=idle_cards()-set(running)
        for uuid in a.gpus:
            if not queue:break
            if uuid not in available:continue
            architecture=queue.pop(0);out=a.out/architecture
            cmd=[sys.executable,'-u','-m','LightGenV2.tasks.t17_router_classification.train_patch16',
                '--data-root',str(a.data_root),'--out',str(out),'--architecture',architecture,
                '--recipe','standard' if architecture=='d2nn' else 'enhanced',
                '--epochs','120','--min-epochs','60','--patience','15','--train-eval-every','5','--batch','32','--workers','4']
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid;env['OMP_NUM_THREADS']='4'
            with (a.out/f'{architecture}.log').open('x') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(architecture=architecture,pid=proc.pid,gpu_uuid=uuid,out=str(out),command=cmd)
            running[uuid]=(proc,item);receipts.append(item);write(a.out/'launch_receipt.json',receipts)
        write(a.out/'suite_status.json',dict(state='running' if running else 'waiting_idle',completed=done,
            running=[i for p,i in running.values()],pending=queue,seconds=time.monotonic()-start,
            max_project_gpus=2,test_evaluations=0))
        if queue or running:time.sleep(20)
    write(a.out/'suite_status.json',dict(state='completed_validation_only',completed=done,test_evaluations=0,max_project_gpus=2))

if __name__=='__main__':main()
