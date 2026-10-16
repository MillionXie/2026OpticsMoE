"""Paired equivalent execution candidates; unchanged original uniform loop."""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from . import measure_qwen_host_metadata as prior_metadata
from . import evaluate_temporal_baseline_batches as frozen
from . import run_temporal_telemetry_suite as prior
from . import reproduce_temporal_baseline as reproduction
from . import unified_resident_utilization as uniform
from . import measure_frozen_batch_costs as costs

TASK,REPO=prior.TASK,prior.REPO
CONFIG=TASK/'configs/baselines/residual_sync_optimization_20261016.json'


def config():
    c=json.loads(CONFIG.read_text())
    if c['cases']!={'qwen':[2,3,4,6],'clip':[4,8,10]} or not c['no_B1_execution_or_replacement']:raise ValueError('Bounded no-B1 trial')
    if (c['warmup_s'],c['measurement_s'],c['queue_depth'],c['repeats'])!=(5,30,8,3):raise ValueError('Original measurement method')
    return c


def sources():
    return {**prior_metadata.sources(),**{p.relative_to(REPO).as_posix():frozen.sha256(p) for p in [Path(__file__),CONFIG,TASK/'residual_sync_execution.py']}}


def capture(fn,inputs,core,model,head,baseline,check_dtypes=False):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    evidence={};operators=[];batch=inputs['input_ids'].shape[0] if baseline=='qwen' else inputs.shape[0]//4
    class AuditMath(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            value=func(*args,**(kwargs or {}));name=str(func)
            if check_dtypes and any(x in name for x in ['aten.mm.','aten.addmm.','aten.bmm.','aten.convolution.','layer_norm','scaled_dot_product']):
                operators.append(dict(op=name,tensors=[dict(shape=list(x.shape),dtype=str(x.dtype)) for x in args if torch.is_tensor(x)]))
            if baseline=='clip' and name=='aten.mm.default' and torch.is_tensor(value) and tuple(value.shape)==(batch,5):
                evidence['quality_logits']=value.detach().cpu().clone()
            return value
    def output_hook(key):
        def hook(module,args,out):
            out=out.last_hidden_state if hasattr(out,'last_hidden_state') else out
            evidence[key]=out.detach().cpu().clone()
        return hook
    if baseline=='qwen':
        hooks=[core.register_forward_hook(output_hook('hidden')),model.lm_head.register_forward_hook(output_hook('full_vocab_logits')),head.register_forward_hook(output_hook('quality_logits'))]
    else:
        hooks=[model.visual.transformer.register_forward_hook(output_hook('hidden')),model.visual.register_forward_hook(output_hook('image_features'))]
    try:
        with AuditMath():evidence['mos']=fn(inputs).detach().cpu().clone()
    finally:
        for h in hooks:h.remove()
    required={'hidden','quality_logits','mos','full_vocab_logits' if baseline=='qwen' else 'image_features'}
    if set(evidence)!=required:raise RuntimeError('Incomplete exact-output evidence')
    return evidence,operators


def strict_pair(control,candidate,inputs,control_core,candidate_core,model,head,baseline,dtypes=False):
    import torch,hashlib
    a,ops_a=capture(control,inputs,control_core,model,head,baseline,dtypes)
    b,ops_b=capture(candidate,inputs,candidate_core,model,head,baseline,dtypes)
    if dtypes and ops_a!=ops_b:raise RuntimeError('Real learned operator signature/dtype changed')
    result={}
    for key in a:
        if not torch.equal(a[key],b[key]) or not torch.isfinite(a[key]).all():raise RuntimeError('Nonidentical '+key)
        digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        result[key]=dict(equal=True,max_abs=0,shape=list(a[key].shape),dtype=str(a[key].dtype),native_sha256=digest(a[key]),candidate_sha256=digest(b[key]))
    return dict(outputs=result,actual_learned_operators_equal=(ops_a==ops_b) if dtypes else None,actual_operator_signatures=ops_a),a['mos'].tolist(),b['mos'].tolist()


def worker(args):
    frozen.preflight_files(args);reproduction.cuda_guard()
    import torch,numpy as np
    from . import quality_token_common as core
    from . import audit_utilization_execution as execution
    from .qwen_host_metadata_execution import HostMetadataQwenModel
    from .residual_sync_execution import IndexedHostQwen,clip_explicit_fp16_forward
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w
    if abs(gpu_power_limit_w()-250)>.1:raise RuntimeError('Original250W required')
    args.output.mkdir();bound=sources();assets=costs.model_hashes(args.model)
    frozen.write_json(args.output/'identity.json',dict(pid=os.getpid(),command=sys.argv,protocol=config(),source_sha256=bound,asset_sha256=assets,checkpoint_sha256=frozen.sha256(args.checkpoint),manifest_sha256=frozen.sha256(args.manifest),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    state=dict(status='running',baseline=args.baseline,points=[],selected_batch=None,full558_exact=False)
    def save():frozen.write_json(args.output/'report.json',state)
    timer=None;handles=[]
    try:
        with torch.inference_mode():
            model,head,rows,prepare,transfer,original,timer,handles=frozen.load_runtime(args)
            if len(rows)!=558 or any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen558 required')
            native=execution.make_gpu_forward(args,model,head,original)
            scores=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['level_scores'].float().to('cuda:0')
            closed=False
            def build(cpu,inputs):
                if args.baseline=='qwen':
                    old=HostMetadataQwenModel(model.model,cpu,inputs);new=IndexedHostQwen(model.model,cpu,inputs)
                    make=lambda m:execution.make_gpu_forward(args,SimpleNamespace(model=m,lm_head=model.lm_head),head,original)
                    return make(old),make(new),old,new
                return native,clip_explicit_fp16_forward(model,head,scores),None,None
            for batch in config()['cases'][args.baseline]:
                if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Source/deadline')
                folder=args.output/f'batch{batch:02d}';folder.mkdir()
                cpu,positions=prepare(rows[:batch]);inputs=transfer(cpu)
                if not closed:
                    ref=torch.tensor(original(inputs)[0]);timer.close();closed=True
                    if not torch.equal(ref,native(inputs).detach().cpu()):raise RuntimeError('Original parity')
                control,candidate,old,new=build(cpu,inputs)
                with prior_metadata.validation_guard(folder/'before_process_audit.json'):
                    eq,_,_=strict_pair(control,candidate,inputs,old,new,model,head,args.baseline,True)
                    frozen.write_json(folder/'before_equality.json',eq)
                repeats={'control':[],'candidate':[]}
                for repeat in range(1,4):
                    for mode,fn in [('control',control),('candidate',candidate)]:
                        if sources()!=bound:raise RuntimeError('Source changed')
                        frozen.write_json(args.output/'status.json',dict(status='measuring',batch=batch,mode=mode,repeat=repeat,updated_unix=time.time()))
                        repeats[mode].append(uniform.measure_repeat(args,fn,inputs,folder/f'{mode}_repeat{repeat:02d}',uniform.config()))
                with prior_metadata.validation_guard(folder/'after_process_audit.json'):
                    eq,_,_=strict_pair(control,candidate,inputs,old,new,model,head,args.baseline)
                    frozen.write_json(folder/'after_equality.json',eq)
                point=dict(batch=batch,control_mean=prior_metadata.pooled(repeats['control']),candidate_mean=prior_metadata.pooled(repeats['candidate']),repeats=repeats,strict_equality=True,frame_positions=positions)
                frozen.write_json(folder/'report.json',point);state['points'].append(point);state['selected_batch']=batch;save()
                print(json.dumps(dict(batch=batch,control=point['control_mean'],candidate=point['candidate_mean'])),flush=True)
                del inputs,cpu,control,candidate,old,new
                if point['candidate_mean']>=98:break
            batch=state['selected_batch'];prediction=[];batch_evidence=[]
            with (args.output/'full558_predictions.csv').open('x',newline='') as stream,prior_metadata.validation_guard(args.output/'full558_process_audit.json') as guard:
                writer=csv.DictWriter(stream,fieldnames=['sample_index','sample_id','target_mos','native_mos','candidate_mos','equal','selected_frame_positions']);writer.writeheader()
                for start in range(0,558,batch):
                    guard.check()
                    if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Validation budget/source')
                    selected=rows[start:start+batch];cpu,pos=prepare(selected);inputs=transfer(cpu)
                    _,candidate,_,new=build(cpu,inputs)
                    eq,a,b=strict_pair(native,candidate,inputs,model.model if args.baseline=='qwen' else None,new,model,head,args.baseline)
                    batch_evidence.append(dict(start=start,videos=len(selected),equality=eq));prediction.extend(a)
                    for i,(row,x,y,p) in enumerate(zip(selected,a,b,pos)):
                        writer.writerow(dict(sample_index=start+i,sample_id=row['sample_id'],target_mos=row['temporal'],native_mos=x,candidate_mos=y,equal=x==y,selected_frame_positions=json.dumps(p)))
                    stream.flush();frozen.write_json(args.output/'status.json',dict(status='full558_validation',batch=batch,completed_test_videos=len(prediction),updated_unix=time.time()))
                    del cpu,inputs,candidate,new
            if sources()!=bound or costs.model_hashes(args.model)!=assets:raise RuntimeError('Bound files changed')
            performance=core.metrics(np.asarray([float(x['temporal']) for x in rows]),np.asarray(prediction))
            frozen.write_json(args.output/'full558_equality.json',dict(videos=558,batches=batch_evidence,metrics=performance))
            state.update(status='complete',full558_exact=True,metrics=performance,no_formal_table_replacement=True,completed_unix=time.time());save()
    except BaseException as error:
        state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise
    finally:
        if timer is not None:timer.close()
        for h in handles:h.remove()


def suite(args):
    import fcntl
    reproduction.require_new_output(args.output);args.output.mkdir()
    bound=sources();state=dict(status='waiting_for_idle',pid=os.getpid(),deadline=args.deadline,completed=[],source_sha256=bound,protocol=config())
    def save():frozen.write_json(args.output/'suite_status.json',state)
    save()
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            costs.wait_for_idle(args,config())
            for model in config()['cases']:
                reproduction.cuda_guard()
                if sources()!=bound:raise RuntimeError('Source changed')
                state.update(status='running',model=model,updated_unix=time.time());save()
                cmd=prior.child_command(model,config()['cases'][model][0],args.manifest,args.output/model)
                cmd[cmd.index(__package__+'.evaluate_temporal_baseline_batches')]=__package__+'.measure_residual_sync';cmd.remove('--telemetry');cmd+=['--phase','worker','--deadline',str(args.deadline)]
                frozen.write_json(args.output/(model+'.command.json'),cmd)
                with (args.output/(model+'.log')).open('x') as stream:
                    subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=min(config()['maximum_worker_seconds'],args.deadline-time.time()))
                report=json.loads((args.output/model/'report.json').read_text())
                if report['status']!='complete' or not report['full558_exact']:raise RuntimeError('Incomplete worker')
                state['completed'].append(model);save()
            state.update(status='complete',completed_unix=time.time());save()
        except BaseException as error:
            state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--baseline',choices=['qwen','clip']);p.add_argument('--batch',type=int,default=2)
    for name in ['manifest','output','model','checkpoint','reproduction-receipt']:p.add_argument('--'+name,type=Path,required=name in ['manifest','output'])
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();args.output=args.output.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise RuntimeError('Original A100/manifest')
    if not 0<args.deadline-time.time()<=config()['maximum_suite_seconds']:raise ValueError('Bounded1h')
    if args.phase=='suite':suite(args)
    else:
        if args.batch not in config()['cases'][args.baseline] or frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS[args.baseline]:raise RuntimeError('OriginalPT/batch')
        worker(args)


if __name__=='__main__':main()
