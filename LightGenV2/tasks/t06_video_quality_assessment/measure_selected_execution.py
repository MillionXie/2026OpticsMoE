"""Predeclared sparse figure points: frozen cost and independent resident util.

Unchanged native workers for five models. Qwen B3/B4 use the separately
equivalence-gated execution adapter. Historical Qwen B1 is never executed.
No monkey-patching, learned-result caching, Graph, model training or batch scan.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from . import measure_frozen_batch_costs as costs
from . import unified_resident_utilization as uniform
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior

TASK, REPO = prior.TASK, prior.REPO
CONFIG = TASK/'configs/baselines/selected_execution_20261016.json'


def config():
    c=json.loads(CONFIG.read_text())
    expected={'qwen':[3,4],'glm':[1,2],'small':[1,3,4],'clip':[1,9,10],'yolo':[1,2,3],'deepseek':[1,6,8]}
    if c['batches']!=expected or (c['test_videos'],c['cost_videos'],c['equivalent_videos'],c['host_proxy_w'])!=(558,200,16,338.2):
        raise ValueError('Only the predeclared sparse frozen points')
    if c['maximum_suite_seconds']!=10800 or not c['no_training_or_graph_or_precision_change']:
        raise ValueError('Original finite scientific contract')
    return c


def sources():
    from . import measure_batched_visual as grouped
    return {**grouped.sources(),**{p.relative_to(REPO).as_posix():frozen.sha256(p) for p in [Path(__file__),CONFIG]}}


def require_point(args):
    if args.baseline not in config()['batches'] or args.batch not in config()['batches'][args.baseline]:
        raise ValueError('Unapproved point, including protected Qwen B1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:
        raise ValueError('Original UUID/manifest only')


def qwen_runtime(args):
    import torch
    from . import audit_utilization_execution as execution
    from .batched_visual_qwen_execution import BatchedVisualQwen
    before=costs.model_hashes(args.model)
    model,head,rows,prepare,transfer,original,timer,handles=frozen.load_runtime(args)
    timer.close()
    if frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS['qwen'] or len(rows)!=558:
        raise RuntimeError('Exact frozen Qwen558 required')
    if any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen parameters only')
    native=execution.make_gpu_forward(args,model,head,original)
    def build(cpu,inputs):
        adapter=BatchedVisualQwen(model.model,cpu,inputs)
        fn=execution.make_gpu_forward(args,SimpleNamespace(model=adapter,lm_head=model.lm_head),head,original)
        return adapter,fn
    def finish():
        if costs.model_hashes(args.model)!=before:raise RuntimeError('Official assets changed')
        for h in handles:h.remove()
    return model,head,rows,prepare,transfer,native,build,finish,before


def qwen_cost(args):
    """Cold dataset-once cost; no equivalence/warmup forwards before first cost."""
    import torch,numpy as np
    from . import quality_token_common as core
    from . import batch_telemetry as telemetry
    from .measure_resident_utilization import ForeignGuard
    from LightGenV2.common import baseline_measurement as measurement
    frozen.preflight_files(args);reproduction.cuda_guard();args.output.mkdir()
    bound=sources();guard=ForeignGuard();guard.start();sampler=None;finish=None
    try:
        if abs(measurement.gpu_power_limit_w(frozen.A100_UUID)-250)>.1:raise RuntimeError('250W required')
        with torch.inference_mode():
            model,head,rows,prepare,transfer,native,build,finish,assets=qwen_runtime(args)
            identity=dict(baseline='qwen',batch_size_videos=args.batch,pid=os.getpid(),source_sha256=bound,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                manifest_sha256=frozen.sha256(args.manifest),checkpoint_sha256=frozen.sha256(args.checkpoint),
                model_file_sha256_before=assets,command=sys.argv,execution=config()['qwen_execution'],
                timing_boundary='first original-equivalent vision block input to GPU MOS synchronized host wall',
                explicit_warmup_forwards=0,first_test_batch_included=True,equivalence_audit_after_cost_in_fresh_process=True)
            frozen.write_json(args.output/'identity.json',identity)
            sampler=telemetry.TelemetrySampler(frozen.A100_UUID);sampler.start();sampler.set_phase('idle');time.sleep(2);sampler.set_phase(None)
            predictions=[];targets=[];timings=[]
            with (args.output/'predictions.csv').open('x',newline='') as ps,(args.output/'batch_timing.csv').open('x',newline='') as ts:
                pw=csv.writer(ps);tw=csv.writer(ts)
                pw.writerow(['sample_index','sample_id','target_mos','prediction','batch_index','selected_frame_positions'])
                tw.writerow(['batch_index','videos','in_cost_subset','synchronized_wall_ms','cuda_ms','preprocessing_audit_ms','h2d_audit_ms'])
                for index,(start,end) in enumerate(costs.plan(args.batch)):
                    guard.check()
                    if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Finite deadline/source')
                    tick=time.perf_counter();cpu,positions=prepare(rows[start:end]);prep=1000*(time.perf_counter()-tick)
                    tick=time.perf_counter();inputs=transfer(cpu);torch.cuda.synchronize();h2d=1000*(time.perf_counter()-tick)
                    adapter,compute=build(cpu,inputs)
                    # Hooks live on the equivalent first block wrapper, not a skipped
                    # original __call__. BoundaryTimer records the SAME host/event order.
                    timer=core.BoundaryTimer(SimpleNamespace(model=adapter))
                    in_cost=end<=200;sampler.set_phase(f'active:{index}' if in_cost else None)
                    try:
                        torch.cuda.synchronize();timer.reset();output=compute(inputs);timing=timer.finish()
                        values=output.detach().cpu().tolist()
                    finally:timer.close();sampler.set_phase(None)
                    wall=timing['vision_first_block_to_score_host_ms'];cuda=timing['vision_first_block_to_score_cuda_ms']
                    if len(values)!=end-start or not all(math.isfinite(v) for v in values) or wall<=0 or cuda<=0:raise RuntimeError('Invalid timing/predictions')
                    timings.append([index,end-start,in_cost,wall,cuda,prep,h2d]);tw.writerow(timings[-1])
                    for offset,value in enumerate(values):
                        row=rows[start+offset];predictions.append(value);targets.append(float(row['temporal']))
                        pw.writerow([start+offset,row['sample_id'],row['temporal'],value,index,json.dumps(positions[offset])])
                    ps.flush();ts.flush();guard.check()
                    frozen.write_json(args.output/'status.json',dict(status='evaluating',completed_test_videos=end,pid=os.getpid(),updated_unix=time.time()))
                    del adapter,compute,inputs,cpu,output
            samples=sampler.stop();sampler=None;telemetry.save_samples(args.output/'telemetry.csv',samples)
            power=measurement.power_report(samples,[r[3] for r in timings if r[2]],power_limit_w=250)
            measured=telemetry.workload_cost([r for r in timings if r[2]],power['active_mean_w'])
            finish();finish=None;guard.check()
            if sources()!=bound or frozen.sha256(args.checkpoint)!=identity['checkpoint_sha256']:raise RuntimeError('Source/PT changed')
            report=dict(status='complete',**identity,test_videos=558,cost_test_videos=200,
                performance=core.metrics(np.asarray(targets),np.asarray(predictions)),cost=measured,power=power,
                telemetry=telemetry.telemetry_report(samples),test_batch_plan=costs.plan(args.batch),
                excluded_from_final_cost=['decode','crop/resize','processor','H2D','audit','model loading'],
                energy_formula='(338.2W host proxy + same-run active A100 W) * T16/1000',
                provisional_until_separate_full558_strict_validation=True)
            frozen.write_json(args.output/'report.json',report)
    finally:
        if sampler is not None:telemetry.save_samples(args.output/'telemetry.csv',sampler.stop())
        guard.stop();frozen.write_json(args.output/'process_audit.json',guard.records)
        if finish is not None:finish()


def qwen_util(args):
    import torch,numpy as np
    from . import quality_token_common as core
    from . import measure_qwen_host_metadata as metadata
    from .measure_batched_visual import exact_pair
    frozen.preflight_files(args);reproduction.cuda_guard();args.output.mkdir()
    bound=sources();state=dict(status='running',batch=args.batch,execution=config()['qwen_execution']);finish=None
    def save():frozen.write_json(args.output/'report.json',state)
    try:
        with torch.inference_mode():
            model,head,rows,prepare,transfer,native,build,finish,assets=qwen_runtime(args)
            identity=dict(pid=os.getpid(),source_sha256=bound,checkpoint_sha256=frozen.sha256(args.checkpoint),
                manifest_sha256=frozen.sha256(args.manifest),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                official_asset_sha256=assets,protocol=uniform.config(),command=sys.argv)
            frozen.write_json(args.output/'identity.json',identity)
            cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu);adapter,compute=build(cpu,inputs)
            with metadata.validation_guard(args.output/'before_process_audit.json'):
                eq,_,_=exact_pair(native,compute,inputs,model.model,adapter,model,head,True)
                frozen.write_json(args.output/'before_equality.json',eq)
            repeats=[]
            for repeat in [1,2,3]:
                if sources()!=bound:raise RuntimeError('Source changed')
                frozen.write_json(args.output/'status.json',dict(status='measuring',repeat=repeat,updated_unix=time.time()))
                repeats.append(uniform.measure_repeat(args,compute,inputs,args.output/f'repeat{repeat:02d}',uniform.config()))
            with metadata.validation_guard(args.output/'after_process_audit.json'):
                eq,_,_=exact_pair(native,compute,inputs,model.model,adapter,model,head)
                frozen.write_json(args.output/'after_equality.json',eq)
            del cpu,inputs,adapter,compute
            cost_rows=list(csv.DictReader((args.output.parent/'cost/predictions.csv').open()))
            evidence=[];predictions=[]
            with (args.output/'full558_predictions.csv').open('x',newline='') as stream,metadata.validation_guard(args.output/'full558_process_audit.json') as guard:
                writer=csv.DictWriter(stream,fieldnames=['sample_index','sample_id','target_mos','native_mos','candidate_mos','cost_mos','equal','selected_frame_positions']);writer.writeheader()
                for start,end in costs.plan(args.batch):
                    guard.check()
                    if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Deadline/source')
                    selected=rows[start:end];cpu,positions=prepare(selected);inputs=transfer(cpu);adapter,compute=build(cpu,inputs)
                    eq,a,b=exact_pair(native,compute,inputs,model.model,adapter,model,head)
                    evidence.append(dict(start=start,equality=eq));predictions.extend(b)
                    for offset,(row,x,y,p) in enumerate(zip(selected,a,b,positions)):
                        old=cost_rows[start+offset]
                        if old['sample_id']!=row['sample_id'] or float(old['prediction'])!=y:raise RuntimeError('Cold cost prediction differs from strict validation')
                        writer.writerow(dict(sample_index=start+offset,sample_id=row['sample_id'],target_mos=row['temporal'],native_mos=x,candidate_mos=y,cost_mos=old['prediction'],equal=x==y,selected_frame_positions=json.dumps(p)))
                    stream.flush();frozen.write_json(args.output/'status.json',dict(status='full558_validation',completed_test_videos=end,updated_unix=time.time()))
                    del cpu,inputs,adapter,compute
            finish();finish=None
            if sources()!=bound:raise RuntimeError('Source changed')
            metrics=core.metrics(np.asarray([float(r['temporal']) for r in rows]),np.asarray(predictions))
            frozen.write_json(args.output/'full558_equality.json',dict(videos=558,batches=evidence,metrics=metrics))
            state.update(status='complete',full558_exact=True,cold_cost_predictions_equal=True,mean_utilization_percent=uniform.pooled_mean(repeats),repeats=repeats,metrics=metrics);save()
    except BaseException as error:state.update(status='failed',error=repr(error));save();raise
    finally:
        if finish is not None:finish()


def worker(args):
    require_point(args);bound=sources()
    if args.baseline=='qwen':
        (qwen_cost if args.phase=='cost' else qwen_util)(args)
    elif args.phase=='cost':costs.worker(args,costs.config())
    else:
        c=uniform.config();c['batches']={args.baseline:[args.batch]}
        uniform.worker(args,c)
    if sources()!=bound:raise RuntimeError('Driver source changed')
    frozen.write_json(args.output/'selected_driver_receipt.json',dict(status='complete',source_sha256=bound,protocol=config(),phase=args.phase,baseline=args.baseline,batch=args.batch,checkpoint_sha256=frozen.sha256(args.checkpoint)))


def suite(args):
    import fcntl,signal
    c=config();reproduction.require_new_output(args.output)
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);costs.wait_for_idle(args,c);args.output.mkdir()
        state=dict(status='running',pid=os.getpid(),source_sha256=sources(),protocol=c,completed=[],deadline=args.deadline,started_unix=time.time())
        def save():frozen.write_json(args.output/'suite_status.json',state)
        save()
        try:
            for baseline,batches in c['batches'].items():
                for batch in batches:
                    point=f'{baseline}_batch{batch:02d}';folder=args.output/point;folder.mkdir()
                    for phase in ['cost','util']:
                        reproduction.cuda_guard()
                        if time.time()>=args.deadline or sources()!=state['source_sha256']:raise RuntimeError('Finite suite deadline/source')
                        command=costs.child_command(baseline,batch,args,folder/phase)
                        command[command.index(__package__+'.measure_frozen_batch_costs')]=__package__+'.measure_selected_execution'
                        command[command.index('--phase')+1]=phase
                        state.update(phase=point+'/'+phase,phase_started_unix=time.time());save()
                        frozen.write_json(folder/(phase+'.command.json'),command)
                        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                        if baseline=='small':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
                        with (folder/(phase+'.log')).open('x') as stream:
                            p=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                            try:code=p.wait(timeout=min(c['maximum_stage_seconds'],args.deadline-time.time()))
                            except subprocess.TimeoutExpired:
                                os.killpg(p.pid,signal.SIGTERM)
                                try:p.wait(timeout=10)
                                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                                raise
                        if code:raise RuntimeError(f'{point}/{phase} exited{code}; no automatic retry')
                        result=json.loads((folder/phase/'report.json').read_text())
                        if result['status']!='complete':raise RuntimeError('Incomplete point')
                    state['completed'].append(point);save()
            reproduction.cuda_guard();state.update(status='complete',completed_unix=time.time());save()
        except BaseException as error:state.update(status='failed',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','cost','util'],required=True)
    p.add_argument('--baseline');p.add_argument('--batch',type=int)
    for name in ['manifest','output','model','checkpoint','reproduction-receipt']:p.add_argument('--'+name,type=Path,required=name in ['manifest','output'])
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();args.output=args.output.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise RuntimeError('Only original A100/split')
    if not 0<args.deadline-time.time()<=config()['maximum_suite_seconds']:raise ValueError('At most3h, no extension')
    try:(suite if args.phase=='suite' else worker)(args)
    except BaseException as error:
        if args.output.exists():frozen.write_json(args.output/'failure.json',dict(error=repr(error),pid=os.getpid(),no_automatic_retry=True))
        raise


if __name__=='__main__':main()
