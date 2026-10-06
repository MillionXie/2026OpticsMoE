"""Serial paired unseen-class runs on one verified RTX; retain timestamped GPU audit."""
import argparse
import json
import os
import subprocess
import sys
import time
from .run_spatial_suite import ROOT, TASK, inventory, processes
from .launch_rtx import validate_device


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--class-set', required=True, choices=['a','b'])
    p.add_argument('--prefix', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    name = validate_device(args.gpu_uuid, inventory())
    root = TASK/'runs'/('smoke' if args.smoke else 'simulation')
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root/f'{args.prefix}_{args.class_set}_execution.json'
    if audit_path.exists(): raise FileExistsError(audit_path)
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    state = {'git_sha':sha, 'gpu_uuid':args.gpu_uuid, 'device':name, 'records':[], 'complete':False}
    for seed_index, seed in enumerate([101] if args.smoke else [101,202,303]):
        for shots in ([5] if args.smoke else [5,20]):
            methods = ['direct','qwen_vision_lora']
            if (seed_index + (args.class_set == 'b')) % 2: methods.reverse()
            for method in methods:
                while processes().get(args.gpu_uuid):
                    print('WAIT assigned RTX occupied', args.gpu_uuid, flush=True); time.sleep(5)
                run_id = f'{args.prefix}_{args.class_set}_{method}_{shots}shot_s{seed}'
                run = root/run_id
                if run.exists(): raise FileExistsError(run)
                command = [sys.executable,'-u','-m','TransferFromElectricity.tasks.t01_object_retrieval.run_unseen',
                           '--method',method,'--class-set',args.class_set,'--shots',str(shots),'--support-seed',str(seed),
                           '--gpu-uuid',args.gpu_uuid,'--run-dir',str(run)]
                if args.smoke: command += ['--smoke']
                with (root/(run_id+'.log')).open('w') as log:
                    proc = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONHASHSEED':str(seed)})
                    row = {'run_id':run_id,'pid':proc.pid,'command':command,'gpu_uuid':args.gpu_uuid,
                           'started':time.time(),'status':'running','own_gpu_seen':False,'foreign_pids':[],'samples':[]}
                    state['records'].append(row)
                    print('START',run_id,proc.pid,flush=True)
                    while proc.poll() is None:
                        observed = processes().get(args.gpu_uuid,set())
                        row['own_gpu_seen'] |= proc.pid in observed
                        row['foreign_pids'] = sorted(set(row['foreign_pids']) | (observed-{proc.pid}))
                        row['samples'].append({'time':time.time(),'pids':sorted(observed)})
                        audit_path.write_text(json.dumps(state,indent=2))
                        time.sleep(5)
                    row.update(finished=time.time(),returncode=proc.returncode,
                               status='complete' if proc.returncode==0 else 'failed')
                    audit_path.write_text(json.dumps(state,indent=2))
                    if run.exists(): (run/'gpu_execution.json').write_text(json.dumps(row,indent=2))
                    if proc.returncode: raise RuntimeError(f'Run failed: {run_id}')
                    if not row['own_gpu_seen']: raise RuntimeError('Training PID never observed on assigned GPU')
                    print('DONE',run_id,flush=True)
    state['complete'] = True
    audit_path.write_text(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
