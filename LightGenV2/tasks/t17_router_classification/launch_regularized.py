"""Launch a published candidate config only on distinct idle physical cards."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',required=True)
    p.add_argument('--run-prefix',required=True)
    args=p.parse_args()
    task=Path(__file__).resolve().parent;root=task.parents[2]
    candidates=json.loads(args.config.read_text())['candidates']
    if not 1<=len(candidates)<=3 or len(args.gpus)!=len(candidates) or len(set(args.gpus))!=len(args.gpus):
        raise ValueError('one to three candidates on distinct GPUs')
    if len({x['name'] for x in candidates})!=len(candidates):raise ValueError('duplicate run name')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'],text=True)
    memory={r.split(',')[0].strip():int(r.split(',')[1]) for r in usage.splitlines()}
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                     '--format=csv,noheader'],text=True)
    receipt=task/'runs'/'simulation'/f'{args.run_prefix}_launch.json'
    if receipt.exists():raise FileExistsError(receipt)
    for item,uuid in zip(candidates,args.gpus):
        if uuid not in memory or memory[uuid]>100 or uuid in running:raise RuntimeError(f'GPU not idle: {uuid}')
        if (task/'runs'/'simulation'/f'{args.run_prefix}_{item["name"]}').exists():raise FileExistsError(item['name'])
        parent=task/'runs'/'simulation'/item['init_run']
        cfg=json.loads((parent/'config.json').read_text())
        if cfg['architecture']!=item['architecture'] or cfg['profile']!='four_top2':raise ValueError('parent contract')
        if not (parent/'best_checkpoint.pt').is_file():raise FileNotFoundError(parent)
    launches=[]
    for item,uuid in zip(candidates,args.gpus):
        run=f'{args.run_prefix}_{item["name"]}'
        out=task/'runs'/'simulation'/run;out.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','LightGenV2.tasks.t17_router_classification.train_regularized',
                 '--init-run',str(task/'runs'/'simulation'/item['init_run']),'--out',str(out)]
        for key in ('epochs','phase_lr','electronic_lr','head_decay','label_smoothing','ema_decay'):
            command.extend(['--'+key.replace('_','-'),str(item[key])])
        if 'class_weight_power' in item:
            command.extend(['--class-weight-power',str(item['class_weight_power'])])
        if 'augmentation_probability' in item:
            command.extend(['--augmentation-probability',str(item['augmentation_probability'])])
        if 'router_balance_weight' in item:
            command.extend(['--router-balance-weight',str(item['router_balance_weight'])])
        if 'batch' in item:command.extend(['--batch',str(item['batch'])])
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid;env['OMP_NUM_THREADS']='4'
        with (out.parent/f'{run}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append(dict(name=item['name'],architecture=item['architecture'],pid=process.pid,
            gpu_uuid=uuid,out=str(out),command=command))
    receipt.write_text(json.dumps(launches,indent=2)+'\n');print(json.dumps(launches,indent=2))

if __name__=='__main__':main()
