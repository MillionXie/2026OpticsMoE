"""One B3 equal-chunk execution trial, at most15min, finalized B1 untouched."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from . import measure_residual_sync as previous
from . import measure_qwen_host_metadata as metadata
from . import unified_resident_utilization as uniform
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior
from . import measure_frozen_batch_costs as costs

TASK,REPO=prior.TASK,prior.REPO


def sources():
    return {**previous.sources(),**{p.relative_to(REPO).as_posix():frozen.sha256(p) for p in [Path(__file__),TASK/'batched_visual_qwen_execution.py']}}


def exact_pair(control,candidate,inputs,old,new,model,head,dtypes=False):
    import torch
    from .batched_visual_qwen_execution import normalized_learned_work
    a,oa=previous.capture(control,inputs,old,model,head,'qwen',dtypes)
    b,ob=previous.capture(candidate,inputs,new,model,head,'qwen',dtypes)
    wa,wb=normalized_learned_work(oa),normalized_learned_work(ob)
    if dtypes and wa!=wb:raise RuntimeError('Learned operator work/dtype changed beyond independent SDPA grouping')
    result={}
    for key in a:
        if not torch.equal(a[key],b[key]) or not torch.isfinite(a[key]).all():raise RuntimeError('Nonidentical '+key)
        digest=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        result[key]=dict(equal=True,max_abs=0,native_sha256=digest(a[key]),candidate_sha256=digest(b[key]),shape=list(a[key].shape),dtype=str(a[key].dtype))
    return dict(outputs=result,normalized_learned_work_equal=(wa==wb) if dtypes else None,actual_control_signatures=oa,actual_candidate_signatures=ob),a['mos'].tolist(),b['mos'].tolist()


def worker(args):
    frozen.preflight_files(args);reproduction.cuda_guard()
    import torch,numpy as np
    from . import quality_token_common as core
    from . import audit_utilization_execution as execution
    from .residual_sync_execution import IndexedHostQwen
    from .batched_visual_qwen_execution import BatchedVisualQwen
    args.output.mkdir();bound=sources();assets=costs.model_hashes(args.model)
    frozen.write_json(args.output/'identity.json',dict(pid=os.getpid(),source_sha256=bound,asset_sha256=assets,command=sys.argv,checkpoint_sha256=frozen.sha256(args.checkpoint),manifest_sha256=frozen.sha256(args.manifest),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),trial_batch=3,maximum_seconds=900,original_measurement_loop=True))
    state=dict(status='running',batch=3,full558_exact=False);timer=None;handles=[]
    def save():frozen.write_json(args.output/'report.json',state)
    try:
        with torch.inference_mode():
            model,head,rows,prepare,transfer,original,timer,handles=frozen.load_runtime(args)
            if len(rows)!=558 or any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen558 required')
            native=execution.make_gpu_forward(args,model,head,original)
            def build(cpu,inputs):
                old=IndexedHostQwen(model.model,cpu,inputs);new=BatchedVisualQwen(model.model,cpu,inputs)
                fn=lambda a:execution.make_gpu_forward(args,SimpleNamespace(model=a,lm_head=model.lm_head),head,original)
                return old,new,fn(old),fn(new)
            cpu,pos=prepare(rows[:3]);inputs=transfer(cpu)
            reference=torch.tensor(original(inputs)[0]);timer.close()
            if not torch.equal(reference,native(inputs).detach().cpu()):raise RuntimeError('Original MOS differs')
            old,new,control,candidate=build(cpu,inputs)
            with metadata.validation_guard(args.output/'before_process_audit.json'):
                eq,_,_=exact_pair(control,candidate,inputs,old,new,model,head,True)
                frozen.write_json(args.output/'before_equality.json',eq)
            repeats={'control':[],'candidate':[]}
            for repeat in [1,2,3]:
                for name,fn in [('control',control),('candidate',candidate)]:
                    if sources()!=bound:raise RuntimeError('Source changed')
                    frozen.write_json(args.output/'status.json',dict(status='measuring',repeat=repeat,mode=name,updated_unix=time.time()))
                    repeats[name].append(uniform.measure_repeat(args,fn,inputs,args.output/f'{name}_repeat{repeat:02d}',uniform.config()))
            with metadata.validation_guard(args.output/'after_process_audit.json'):
                eq,_,_=exact_pair(control,candidate,inputs,old,new,model,head)
                frozen.write_json(args.output/'after_equality.json',eq)
            state.update(control_mean=metadata.pooled(repeats['control']),candidate_mean=metadata.pooled(repeats['candidate']),repeats=repeats);save()
            del cpu,inputs,old,new,control,candidate
            predictions=[];evidence=[]
            with (args.output/'full558_predictions.csv').open('x',newline='') as stream,metadata.validation_guard(args.output/'full558_process_audit.json') as guard:
                writer=csv.DictWriter(stream,fieldnames=['sample_index','sample_id','target_mos','native_mos','candidate_mos','equal','selected_frame_positions']);writer.writeheader()
                for start in range(0,558,3):
                    guard.check()
                    if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Bounded trial/source')
                    selected=rows[start:start+3];cpu,pos=prepare(selected);inputs=transfer(cpu)
                    _,new,_,candidate=build(cpu,inputs)
                    eq,a,b=exact_pair(native,candidate,inputs,model.model,new,model,head)
                    evidence.append(dict(start=start,equality=eq));predictions.extend(a)
                    for i,(row,x,y,p) in enumerate(zip(selected,a,b,pos)):writer.writerow(dict(sample_index=start+i,sample_id=row['sample_id'],target_mos=row['temporal'],native_mos=x,candidate_mos=y,equal=x==y,selected_frame_positions=json.dumps(p)))
                    stream.flush();frozen.write_json(args.output/'status.json',dict(status='full558_validation',completed_videos=len(predictions),updated_unix=time.time()))
                    del cpu,inputs,new,candidate
            if sources()!=bound or costs.model_hashes(args.model)!=assets:raise RuntimeError('Source/assets changed')
            metrics=core.metrics(np.asarray([float(x['temporal']) for x in rows]),np.asarray(predictions))
            frozen.write_json(args.output/'full558_equality.json',dict(videos=558,batches=evidence,metrics=metrics))
            state.update(status='complete',full558_exact=True,metrics=metrics,completed_unix=time.time(),no_formal_table_replacement=True);save()
    except BaseException as error:
        state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise
    finally:
        if timer is not None:timer.close()
        for h in handles:h.remove()


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--baseline',default='qwen');p.add_argument('--batch',type=int,default=3)
    for name in ['manifest','output','model','checkpoint']:p.add_argument('--'+name,type=Path,required=name in ['manifest','output'])
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();args.output=args.output.resolve()
    if args.baseline!='qwen' or args.batch!=3 or os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise RuntimeError('Only original A100/B3/manifest')
    if not 0<args.deadline-time.time()<=900:raise ValueError('Single trial at most15min')
    if args.phase=='worker':worker(args);return
    reproduction.require_new_output(args.output);args.output.mkdir()
    state=dict(status='waiting_for_idle',pid=os.getpid(),deadline=args.deadline,source_sha256=sources(),no_B1_execution=True)
    def save():frozen.write_json(args.output/'suite_status.json',state)
    save()
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            costs.wait_for_idle(args,dict(maximum_initial_idle_wait_seconds=120));reproduction.cuda_guard()
            cmd=prior.child_command('qwen',3,args.manifest,args.output/'qwen')
            cmd[cmd.index(__package__+'.evaluate_temporal_baseline_batches')]=__package__+'.measure_batched_visual';cmd.remove('--telemetry');cmd+=['--phase','worker','--deadline',str(args.deadline)]
            frozen.write_json(args.output/'worker.command.json',cmd);state.update(status='running');save()
            with (args.output/'worker.log').open('x') as f:subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=args.deadline-time.time())
            state.update(status='complete',completed_unix=time.time());save()
        except BaseException as error:state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise


if __name__=='__main__':main()
