"""Run serial method queues on idle RTX GPUs; keep each dataset on one physical GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .launch_rtx import validate_device

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
METHODS=['direct','clip_vision_lora','qwen_vision_pooled_lora','qwen_vision_lora','qwen_vision_global_lora']


def inventory():
    return list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines()))


def processes():
    result={}
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines())
    for pid,uuid in rows:result.setdefault(uuid.strip(),set()).add(int(pid.strip()))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--gpu-uuids',nargs='+',required=True)
    parser.add_argument('--datasets',nargs='+',default=['caltech','cifar','imagenette'],choices=['caltech','cifar','imagenette'])
    parser.add_argument('--methods',nargs='+',default=METHODS,choices=METHODS)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if len(args.gpu_uuids)>len(args.datasets) or len(set(args.gpu_uuids))!=len(args.gpu_uuids):
        raise ValueError('Use distinct GPU UUIDs, no more than the number of datasets')
    devices=inventory();occupied=processes()
    for uuid in args.gpu_uuids:
        validate_device(uuid,devices)
        if occupied.get(uuid):raise RuntimeError(f'GPU already occupied: {uuid}: {occupied[uuid]}')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    kind='smoke' if args.smoke else 'simulation'
    directory=TASK/'runs'/kind;directory.mkdir(parents=True,exist_ok=True)
    audit_path=directory/f'{args.prefix}_{"_".join(args.datasets)}_execution.json'
    if audit_path.exists():raise FileExistsError(audit_path)
    state={'git_sha':sha,'config_family':'spatial_v4','smoke':args.smoke,'records':[],
           'policy':'One serial queue per physical GPU, process inventory sampled every five seconds','complete':False}
    pending={uuid:[] for uuid in args.gpu_uuids}
    for index,dataset in enumerate(args.datasets):
        uuid=args.gpu_uuids[index%len(args.gpu_uuids)]
        # Counterbalance method order across datasets without changing method initialization.
        offset=index%len(args.methods);methods=args.methods[offset:]+args.methods[:offset]
        pending[uuid].extend((dataset,method) for method in methods)
    active={};failed=False
    def save():audit_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    while active or any(pending.values()):
        observed=processes()
        for uuid in args.gpu_uuids:
            if uuid not in active and pending[uuid] and not failed:
                if observed.get(uuid):raise RuntimeError(f'GPU became occupied before next run: {uuid}')
                dataset,method=pending[uuid].pop(0)
                run_id=f'{args.prefix}_{dataset}_{method}_s42';run=directory/run_id
                if run.exists():raise FileExistsError(run)
                command=[sys.executable,'-u','-m','TransferFromElectricity.tasks.t01_object_retrieval.launch_rtx',
                    '--gpu-uuid',uuid,'--method',method,'--config',f'TransferFromElectricity/tasks/t01_object_retrieval/configs/spatial_v4/{dataset}.yaml',
                    '--steps-per-epoch','120','--run-dir',str(run)]
                if args.smoke:command+=['--smoke','--validation-only']
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONHASHSEED':'42'}
                log=(directory/(run_id+'.log')).open('w')
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                record={'run_id':run_id,'dataset':dataset,'method':method,'gpu_uuid':uuid,'pid':process.pid,'command':command,
                        'started':time.time(),'own_gpu_seen':False,'foreign_pids':[],'samples':0,'status':'running'}
                state['records'].append(record);active[uuid]=(process,log,record,run);save()
                print('START',run_id,uuid,process.pid,flush=True)
        for uuid,(process,log,record,run) in list(active.items()):
            observed_pids=observed.get(uuid,set());record['samples']+=1
            record['own_gpu_seen'] |= process.pid in observed_pids
            record['foreign_pids']=sorted(set(record['foreign_pids']) | (observed_pids-{process.pid}))
            history=run/'history.json'
            if history.exists():
                try:epoch=len(json.loads(history.read_text()))
                except json.JSONDecodeError:epoch=record.get('last_epoch',0)
                if epoch!=record.get('last_epoch',0):
                    record['last_epoch']=epoch;print(record['run_id'],'epoch',epoch,flush=True)
            code=process.poll()
            if code is not None:
                record.update(returncode=code,finished=time.time(),status='complete' if code==0 else 'failed')
                log.close();del active[uuid]
                if code!=0:
                    failed=True
                    for queue in pending.values():queue.clear()
                (run/'gpu_execution.json').write_text(json.dumps({'git_sha':sha,**record},indent=2))
                print(record['status'].upper(),record['run_id'],'foreign',record['foreign_pids'],flush=True)
        save()
        if active or any(pending.values()):time.sleep(5)
    state['complete']=not failed;save()
    if failed:raise RuntimeError('A training run failed; preserved logs and stopped pending launches')
    print('SUITE COMPLETE',audit_path,flush=True)


if __name__=='__main__':main()
