"""Bounded paired execution experiment; original measurement loop is reused."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior
from . import unified_resident_utilization as uniform
from . import measure_frozen_batch_costs as costs

TASK, REPO = prior.TASK, prior.REPO
CONFIG = TASK/'configs/baselines/qwen_host_metadata_20261016.json'


def config():
    c=json.loads(CONFIG.read_text())
    if c['batches'] != [2,3,4,6,8] or not c['no_B1_execution_or_replacement']:
        raise ValueError('Finite small batches; no B1 experiment')
    if (c['repeats'],c['warmup_seconds'],c['measurement_seconds'],c['queue_depth'])!=(3,5,30,8):
        raise ValueError('Original utilization method required')
    if c['maximum_suite_seconds']!=3600 or c['maximum_initial_idle_wait_seconds']!=1800:
        raise ValueError('Finite budget required')
    return c


def sources():
    paths=[Path(__file__),CONFIG,TASK/'qwen_host_metadata_execution.py']
    return {**costs.sources(),**{p.relative_to(REPO).as_posix():frozen.sha256(p) for p in paths}}


@contextlib.contextmanager
def validation_guard(path):
    from .measure_resident_utilization import ForeignGuard
    guard=ForeignGuard();guard.start()
    try:
        guard.check();yield guard;guard.check()
    finally:
        guard.stop();frozen.write_json(path,guard.records)


def capture(compute, inputs, model, head):
    """Untimed full-hidden/logit evidence. Hooks are removed BEFORE measurement."""
    import torch
    result={}
    def hidden_hook(module,args,out):result['hidden']=out.last_hidden_state.detach().cpu().clone()
    def logits_hook(module,args,out):result['quality_logits']=out.detach().cpu().clone()
    def vocab_hook(module,args,out):result['full_vocab_logits']=out.detach().cpu().clone()
    hooks=[model.model.language_model.register_forward_hook(hidden_hook),head.register_forward_hook(logits_hook),model.lm_head.register_forward_hook(vocab_hook)]
    try:
        result['mos']=compute(inputs).detach().cpu().clone()
        torch.cuda.synchronize()
    finally:
        for h in hooks:h.remove()
    if set(result)!={'hidden','quality_logits','full_vocab_logits','mos'}:
        raise RuntimeError('Incomplete output-equality evidence')
    return result


def strict_pair(native, candidate, inputs, model, head):
    import torch
    a,b=capture(native,inputs,model,head),capture(candidate,inputs,model,head)
    evidence={}
    for key in a:
        if not torch.equal(a[key],b[key]) or not torch.isfinite(a[key]).all():
            raise RuntimeError('Output is not exactly equal: '+key)
        digest=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        evidence[key]=dict(shape=list(a[key].shape),dtype=str(a[key].dtype),native_sha256=digest(a[key]),candidate_sha256=digest(b[key]),equal=True,max_abs=0)
    return evidence,a['mos'].tolist(),b['mos'].tolist()


def pooled(repeats):
    n=sum(x['measurement']['samples'] for x in repeats)
    return sum(x['measurement']['mean_gpu_utilization_percent']*x['measurement']['samples'] for x in repeats)/n


def worker(args):
    frozen.preflight_files(args);reproduction.cuda_guard()
    import torch
    import numpy as np
    from . import quality_token_common as core
    from . import audit_utilization_execution as execution
    from .qwen_host_metadata_execution import HostMetadataQwenModel,validate_metadata_unchanged
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w
    from .diagnose_batch_numerics import input_records
    if abs(gpu_power_limit_w()-250)>.1:raise RuntimeError('Original250W required')
    args.output.mkdir()
    bound=sources();assets=costs.model_hashes(args.model)
    identity=dict(protocol=config(),source_sha256=bound,asset_sha256=assets,checkpoint_sha256=frozen.sha256(args.checkpoint),manifest_sha256=frozen.sha256(args.manifest),pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=sys.argv,torch=torch.__version__,original_measurement_loop_reused=True)
    frozen.write_json(args.output/'identity.json',identity)
    timer=None;handles=[]
    state=dict(status='running',points=[],selected_batch=None,full558_exact=False)
    def save():frozen.write_json(args.output/'report.json',state)
    try:
        with torch.inference_mode():
            model,head,_,prepare,transfer,original,timer,handles=frozen.load_runtime(args)
            if any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen only')
            native=execution.make_gpu_forward(args,model,head,original)
            rows=[x for x in core.read_manifest(args.manifest) if x['split']=='test']
            if len(rows)!=558:raise RuntimeError('Original558 TEST required')
            native_checked=False
            for batch in config()['batches']:
                if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Source/deadline guard')
                frozen.process_audit()
                folder=args.output/f'batch{batch:02d}';folder.mkdir()
                frozen.write_json(args.output/'status.json',dict(status='preparing',batch=batch,updated_unix=time.time()))
                cpu,positions=prepare(rows[:batch]);records=input_records(cpu,batch);inputs=transfer(cpu)
                adapter=HostMetadataQwenModel(model.model,cpu,inputs)
                candidate=execution.make_gpu_forward(args,SimpleNamespace(model=adapter,lm_head=model.lm_head),head,original)
                with validation_guard(folder/'before_process_audit.json'):
                    if not native_checked:
                        ref=torch.tensor(original(inputs)[0]);timer.close();native_checked=True
                        if not torch.equal(native(inputs).detach().cpu(),ref):raise RuntimeError('Original runtime MOS differs')
                    evidence,_,_=strict_pair(native,candidate,inputs,model,head)
                    frozen.write_json(folder/'before_equality.json',evidence)
                repeats={'native':[],'host_metadata':[]}
                for repeat in range(1,4):
                    for mode,compute in [('native',native),('host_metadata',candidate)]:
                        if sources()!=bound:raise RuntimeError('Source changed')
                        frozen.write_json(args.output/'status.json',dict(status='measuring',batch=batch,mode=mode,repeat=repeat,updated_unix=time.time()))
                        target=folder/f'{mode}_repeat{repeat:02d}'
                        repeats[mode].append(uniform.measure_repeat(args,compute,inputs,target,uniform.config()))
                with validation_guard(folder/'after_process_audit.json'):
                    evidence,_,_=strict_pair(native,candidate,inputs,model,head)
                    frozen.write_json(folder/'after_equality.json',evidence)
                if not validate_metadata_unchanged(adapter,cpu):raise RuntimeError('CPU metadata mutated')
                point=dict(batch=batch,repeats=repeats,pooled_native_percent=pooled(repeats['native']),pooled_candidate_percent=pooled(repeats['host_metadata']),frame_positions=positions,sample_ids=[x['sample_id'] for x in rows[:batch]],input_records=records,strict_equality=True)
                frozen.write_json(folder/'report.json',point);state['points'].append(point);save()
                print(json.dumps({k:v for k,v in point.items() if k.startswith('pooled') or k=='batch'}),flush=True)
                state['selected_batch']=batch
                del adapter,candidate,inputs,cpu
                if point['pooled_candidate_percent']>=98:break
            batch=state['selected_batch'];save()
            target=args.output/'full558_predictions.csv';batch_evidence=[];native_values=[];candidate_values=[]
            with target.open('x',newline='') as stream,validation_guard(args.output/'full558_process_audit.json') as guard:
                writer=csv.DictWriter(stream,fieldnames=['sample_index','sample_id','target_mos','native_mos','candidate_mos','equal','selected_frame_positions']);writer.writeheader()
                for start in range(0,558,batch):
                    if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Validation deadline/source')
                    guard.check();selected=rows[start:start+batch]
                    cpu,positions=prepare(selected);inputs=transfer(cpu)
                    adapter=HostMetadataQwenModel(model.model,cpu,inputs)
                    candidate=execution.make_gpu_forward(args,SimpleNamespace(model=adapter,lm_head=model.lm_head),head,original)
                    eq,va,vb=strict_pair(native,candidate,inputs,model,head)
                    if not validate_metadata_unchanged(adapter,cpu):raise RuntimeError('Metadata mutated')
                    batch_evidence.append(dict(start=start,videos=len(selected),equality=eq))
                    native_values.extend(va);candidate_values.extend(vb)
                    for i,(row,a,b,pos) in enumerate(zip(selected,va,vb,positions)):
                        writer.writerow(dict(sample_index=start+i,sample_id=row['sample_id'],target_mos=row['temporal'],native_mos=a,candidate_mos=b,equal=a==b,selected_frame_positions=json.dumps(pos)))
                    stream.flush();frozen.write_json(args.output/'status.json',dict(status='full558_validation',batch=batch,completed_test_videos=len(native_values),updated_unix=time.time()))
                    del adapter,candidate,inputs,cpu
            if sources()!=bound or costs.model_hashes(args.model)!=assets:raise RuntimeError('Bound source/assets changed')
            values=np.asarray([float(r['temporal']) for r in rows])
            performance=core.metrics(values,np.asarray(native_values));other=core.metrics(values,np.asarray(candidate_values))
            if native_values!=candidate_values or performance!=other:raise RuntimeError('Full558 exact equality failed')
            frozen.write_json(args.output/'full558_equality.json',dict(batch=batch,complete_videos=558,metrics_native=performance,metrics_candidate=other,batches=batch_evidence))
            state.update(status='complete',full558_exact=True,full558_metrics=performance,completed_unix=time.time(),qwen_b1_unchanged=True,no_formal_table_replacement=True);save()
    except BaseException as error:
        state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise
    finally:
        if timer is not None:timer.close()
        for h in handles:h.remove()


def suite(args):
    import fcntl
    reproduction.require_new_output(args.output);args.output.mkdir()
    state=dict(status='waiting_for_idle',pid=os.getpid(),deadline=args.deadline,source_sha256=sources(),protocol=config())
    def save():frozen.write_json(args.output/'suite_status.json',state)
    save()
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            costs.wait_for_idle(args,config());reproduction.cuda_guard()
            state.update(status='running',admitted_unix=time.time());save()
            cmd=prior.child_command('qwen',2,args.manifest,args.output/'paired_execution')
            cmd[cmd.index(__package__+'.evaluate_temporal_baseline_batches')]=__package__+'.measure_qwen_host_metadata'
            cmd.remove('--telemetry');cmd+=['--phase','worker','--deadline',str(args.deadline)]
            frozen.write_json(args.output/'worker.command.json',cmd)
            with (args.output/'worker.log').open('x') as log:
                subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.deadline-time.time())
            result=json.loads((args.output/'paired_execution/report.json').read_text())
            if result['status']!='complete' or not result['full558_exact']:raise RuntimeError('No fully validated result')
            state.update(status='complete',selected_batch=result['selected_batch'],completed_unix=time.time());save()
        except BaseException as error:
            state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--baseline',default='qwen');p.add_argument('--batch',type=int,default=2)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=prior.QWEN_MODEL);p.add_argument('--checkpoint',type=Path,default=prior.QWEN_PT)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();args.output=args.output.resolve()
    if args.baseline!='qwen' or args.batch==1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID:raise ValueError('Original A100 and no B1')
    if not 0<args.deadline-time.time()<=config()['maximum_suite_seconds']:raise ValueError('One-hour absolute budget')
    if frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise ValueError('Original manifest required')
    if args.phase=='suite':suite(args)
    else:worker(args)


if __name__=='__main__':main()
