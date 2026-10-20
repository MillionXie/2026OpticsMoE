"""Independent full558 FP32 accuracy references, after the strict cost suite.

No energy/timing result is inferred from this dataset evaluation. No training,
batch search, resource-control retry or modification of any completed cost point.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from . import lab_busy_resource_20261020 as lab

COST=lab.ROOT/'recovery_1102'
ROOT=COST/'accuracy_fp32'
PLAN=[('clip',1),('clip',10),('yolo',1),('yolo',3),('qwen',1),('qwen',3),('glm',1)]


def supported(model,batch):
    method='mps' if batch==1 else 'temporal'
    path=COST/f'{method}_{model}_fp32_100'/'status.json'
    return path.exists() and json.loads(path.read_text())['status']=='complete'


def worker(args):
    import torch
    import numpy as np
    from . import quality_token_common as core
    if args.output.exists() or not supported(args.model,args.batch):raise RuntimeError('Complete same-batch cost reference and fresh accuracy output required')
    args.output.mkdir();args.gpu_uuid=lab.frozen.A100_UUID;args.power_limit_w=250
    args.precision='fp32';args.model_path,args.checkpoint,_,args.reproduction_receipt=lab.locations(args.model)
    bound=lab.sources();own_sha=lab.frozen.sha256(Path(__file__))
    class Guard(lab.shared.Guard):
        def check(self):
            if lab.sources()!=bound or lab.frozen.sha256(Path(__file__))!=own_sha:raise RuntimeError('Accuracy runtime changed')
            super().check()
    guard=Guard(args);identity=dict(model=args.model,batch=args.batch,precision='fp32',
        gpu_uuid=args.gpu_uuid,pid=os.getpid(),sources={**bound,str(Path(__file__).relative_to(lab.REPO)):own_sha},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=lab.REPO,text=True).strip())
    try:
        guard.check();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        power=float(subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if power!=250:raise RuntimeError('Original250W required')
        model,rows,prepare,transfer,compute,first,assets=lab.fp32.runtime(args,guard,identity)
        if len(rows)!=558:raise RuntimeError('Original full558 only')
        method='mps' if args.batch==1 else 'temporal'
        reference=COST/f'{method}_{args.model}_fp32_100'/'identity.json'
        cost_identity=json.loads(reference.read_text())
        if cost_identity['checkpoint_sha256']!=identity['checkpoint_sha256'] or cost_identity['assets']!=assets:
            raise RuntimeError('Cost/accuracy checkpoint or backbone identity differs')
        if cost_identity['floating_parameter_dtypes']!=identity['floating_parameter_dtypes']:
            raise RuntimeError('Cost/accuracy precision differs')
        identity.update(cost_identity_source=str(reference.relative_to(lab.REPO)),cost_identity_sha256=lab.frozen.sha256(reference))
        lab.shared.old.write(args.output/'identity.json',identity)
        predictions=[];targets=[]
        with (args.output/'predictions.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['sample_index','sample_id','target_mos','prediction','frame_positions'])
            for start in range(0,558,args.batch):
                guard.fast();selected=rows[start:start+args.batch];cpu,positions=prepare(selected)
                result,_=compute(transfer(cpu));torch.cuda.synchronize();values=result.detach().cpu().tolist()
                if len(values)!=len(selected) or not all(math.isfinite(float(v)) for v in values):raise RuntimeError('Invalid native GPU MOS')
                for index,(row,value,frames) in enumerate(zip(selected,values,positions),start):
                    target=float(row['temporal']);predictions.append(float(value));targets.append(target)
                    writer.writerow([index,row['sample_id'],target,value,json.dumps(frames)])
                stream.flush();lab.shared.old.write(args.output/'progress.json',dict(samples=len(predictions),total=558))
                if len(predictions)%25<args.batch:print(f'{args.model} FP32 B{args.batch}: {len(predictions)}/558',flush=True)
        if lab.previous.costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Official assets changed')
        guard.check();report=dict(status='complete',samples=558,model=args.model,batch=args.batch,precision='fp32',
            checkpoint_sha256=identity['checkpoint_sha256'],assets=assets,metrics=core.metrics(np.array(targets),np.array(predictions)),
            frozen_no_training=True,accuracy_independent_of_cost=True,not_fresh_accuracy_at_each_resource_cap=True,
            selection='original TEST-selected frozen head; development estimate, noVAL')
        lab.shared.old.write(args.output/'report.json',report)
        lab.shared.old.write(args.output/'status.json',dict(status='complete',finished=time.time()))
    except BaseException as error:
        lab.shared.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:lab.shared.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    if ROOT.exists():raise RuntimeError('Preserve prior accuracy suite')
    ROOT.mkdir();state=dict(status='waiting_for_cost_suite',completed=[],deadline=args.deadline,pid=os.getpid())
    def save():lab.shared.old.write(ROOT/'suite_status.json',state)
    save()
    while True:
        if time.time()>=args.deadline-180:state['status']='waiting_expired';save();return
        path=COST/'suite_status.json';cost=json.loads(path.read_text()) if path.exists() else {}
        if cost.get('status') in ('complete','stopped'):break
        time.sleep(10)
    with (lab.TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stable=0
        while stable<3:
            if time.time()>=args.deadline-180:state['status']='waiting_expired';save();return
            stable=stable+1 if not lab.shared.guard_records(lab.frozen.A100_UUID) else 0
            if stable<3:time.sleep(10)
        try:
            for model,batch in PLAN:
                if not supported(model,batch):state.setdefault('missing_cost_reference',[]).append([model,batch]);continue
                if time.time()>=args.deadline-180:raise TimeoutError('Bounded accuracy budget')
                name=f'{model}_batch{batch:02d}';_,_,python,_=lab.locations(model)
                cmd=[str(python),'-u','-m',__package__+'.lab_strict_fp32_accuracy_20261020',
                     '--model',model,'--batch',str(batch),'--deadline',str(args.deadline),'--manifest',str(args.manifest)]
                lab.shared.old.write(ROOT/(name+'.command.json'),cmd)
                with (ROOT/(name+'.log')).open('x') as log:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=lab.frozen.A100_UUID,OMP_NUM_THREADS='8')
                    env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
                    child=subprocess.Popen(cmd,cwd=lab.REPO,env=env,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:code=child.wait(timeout=min(1800,max(1,args.deadline-time.time())))
                    except BaseException:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
                if code:raise RuntimeError(f'{name} failed; preserve without retry')
                state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',action='store_true')
    parser.add_argument('--model',choices=('clip','yolo','qwen','glm'));parser.add_argument('--batch',type=int)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    if not 0<args.deadline-time.time()<=14400:raise ValueError('Finite four-hour admission only')
    if args.suite:return suite(args)
    if (args.model,args.batch) not in PLAN or os.environ.get('CUDA_VISIBLE_DEVICES')!=lab.frozen.A100_UUID:raise ValueError('Fixed original-platform accuracy reference only')
    args.output=ROOT/f'{args.model}_batch{args.batch:02d}';worker(args)


if __name__=='__main__':main()
