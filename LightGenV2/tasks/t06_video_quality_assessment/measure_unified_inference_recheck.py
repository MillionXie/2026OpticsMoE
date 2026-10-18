"""Unified dataset-once inference cost; inter-call waits are audit-only.

The historical B1 clock starts at the first native block and stops when MOS
is ready on GPU. Resident utilization and its calibrated duty are independent
evidence, not this run's power or throughput. No Qwen B1 or rental execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

from . import measure_frozen_batch_costs as costs
from . import measure_selected_execution as selected
from . import measure_busy50_service as paced
from . import batch_telemetry as telemetry
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import unified_resident_utilization as uniform
from .measure_resident_utilization import ForeignGuard

TASK, REPO = costs.TASK, costs.REPO
CONFIG = TASK/'configs/baselines/unified_inference_recheck_20261018.json'
OLD = TASK/'runs/simulation/lgvq_resident_busy50_a100_20261017'
CONTROL = {'qwen': OLD/'attempt02_20261018/qwen_batch03',
           'glm': OLD/'attempt02_20261018/glm_batch02',
           'clip': OLD/'attempt03_remaining_20261018/clip_batch10',
           'yolo': OLD/'attempt03_remaining_20261018/yolo_batch03'}
NATIVE_UTIL = {
    1: TASK/'runs/simulation/lgvq_selected_execution_a100_20261016/clip_batch01/util/batch01/report.json',
    10: TASK/'runs/simulation/lgvq_yolo_clip_sequential_a100_20261017/clip/batch10/report.json'}


def config():
    c=json.loads(CONFIG.read_text())
    if (c['paced_batches'], c['clip_native_batches'], c['clip_native_repeats'],
        c['test_videos'], c['cost_videos'], c['explicit_warmup_forwards']) != (
        {'qwen':3,'glm':2,'clip':10,'yolo':3},[1,10],3,558,200,0):
        raise ValueError('Only the human-approved existing points, zero warmup')
    return c


def points(c):
    # CLIP power verification first. Every complete repeat is retained, not selected.
    return ([('clip',b,'native',r) for b in c['clip_native_batches'] for r in range(1,4)]
            +[(m,b,'paced',1) for m,b in c['paced_batches'].items()])


def require_point(args,c):
    if (args.baseline,args.batch,args.kind,args.repeat) not in points(c):
        raise ValueError('Point not authorized; Qwen B1 is protected')


def source_identity():
    return {**selected.sources(), **{p.relative_to(REPO).as_posix():frozen.sha256(p)
                                    for p in (Path(__file__),CONFIG,Path(costs.__file__),Path(paced.__file__))}}


def control_evidence(args):
    if args.kind=='native':
        file=NATIVE_UTIL[args.batch];report=json.loads(file.read_text())
        if report['status']!='complete' or len(report['repeats'])!=3:
            raise ValueError('Independent native utilization must be complete')
        return dict(duty=1.0, independent_util_percent=report['mean_utilization_percent'],
                    utilization_source=file.relative_to(REPO).as_posix(),
                    utilization_source_sha256=frozen.sha256(file), utilization_not_from_cost_run=True)
    folder=CONTROL[args.baseline]
    report=json.loads((folder/'report.json').read_text());identity=json.loads((folder/'identity.json').read_text())
    if (report['status']!='complete' or not report['target_reached']
        or identity['checkpoint_sha256']!=costs.CHECKPOINTS[args.baseline]
        or identity['manifest_sha256']!=frozen.MANIFEST_SHA
        or identity['batch']!=args.batch or identity['gpu_uuid']!=frozen.A100_UUID):
        raise ValueError('Prior calibrated point is incomplete or has changed identity')
    return dict(duty=report['frozen_duty'],independent_util_percent=report['mean_gpu_busy_percent'],
                utilization_source=(folder/'report.json').relative_to(REPO).as_posix(),
                utilization_source_sha256=frozen.sha256(folder/'report.json'),
                identity_sha256=frozen.sha256(folder/'identity.json'),
                utilization_not_from_cost_run=True)


def excluded_wait_seconds(wall_ms,duty):
    return paced.imposed_wait(wall_ms/1000,duty)


def load_runtime(args):
    """Use the published native or equality-audited graph; no graph patches."""
    import torch
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import FirstBlockTimer
    if args.baseline=='qwen':
        model,head,rows,prepare,transfer,native,build,finish,assets=selected.qwen_runtime(args)
        def run(cpu,inputs):
            adapter,compute=build(cpu,inputs)
            timer=core.BoundaryTimer(SimpleNamespace(model=adapter))
            try:
                torch.cuda.synchronize();timer.reset();output=compute(inputs);timing=timer.finish()
                return output.detach().cpu().tolist(),timing['vision_first_block_to_score_host_ms'],timing['vision_first_block_to_score_cuda_ms']
            finally:timer.close()
        return model,head,rows,prepare,transfer,run,finish
    if args.baseline=='glm':
        model,head,prepare,transfer,compute,reference,finish=uniform.special_runtime(args)
        rows=[r for r in core.read_manifest(args.manifest) if r['split']=='test']
        timer=FirstBlockTimer(model.model.visual.blocks[0])
        def run(cpu,inputs):
            torch.cuda.synchronize();timer.reset();output=compute(inputs);timing=timer.finish()
            return output.detach().cpu().tolist(),timing['host_ms'],timing['cuda_ms']
        def close():
            timer.close();finish()
        return model,head,rows,prepare,transfer,run,close
    before=costs.model_hashes(args.model)
    model,head,rows,prepare,transfer,native,timer,handles=frozen.load_runtime(args)
    def close():
        timer.close()
        for handle in handles:handle.remove()
        if costs.model_hashes(args.model)!=before:raise RuntimeError('Frozen official assets changed')
    return model,head,rows,prepare,transfer,lambda cpu,inputs:native(inputs),close


def worker(args,c):
    require_point(args,c);reproduction.require_new_output(args.output);reproduction.cuda_guard()
    if args.baseline!='glm':frozen.preflight_files(args)
    import torch
    import numpy as np
    from . import quality_token_common as core
    from LightGenV2.common import baseline_measurement as measurement
    if abs(measurement.gpu_power_limit_w(frozen.A100_UUID)-250)>.1:raise RuntimeError('250W must remain unchanged')
    args.output.mkdir();bound=source_identity();control=control_evidence(args);finish=None
    sampler=None;guard=ForeignGuard();guard.start();samples=[]
    identity=dict(protocol=c,baseline=args.baseline,batch=args.batch,kind=args.kind,repeat=args.repeat,
        pid=os.getpid(),gpu_uuid=frozen.A100_UUID,command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=bound,manifest_sha256=frozen.sha256(args.manifest),
        checkpoint_sha256=costs.CHECKPOINTS[args.baseline],independent_utilization=control)
    if identity['manifest_sha256']!=frozen.MANIFEST_SHA:
        raise ValueError('Formal assets changed')
    frozen.write_json(args.output/'identity.json',identity)
    def status(value,**extra):frozen.write_json(args.output/'status.json',dict(status=value,updated_unix=time.time(),**extra))
    status('loading',pid=os.getpid())
    try:
        with torch.inference_mode():
            model,head,rows,prepare,transfer,run,finish=load_runtime(args)
            if frozen.sha256(args.checkpoint)!=identity['checkpoint_sha256']:raise RuntimeError('Exact frozen head differs')
            identity.update(checkpoint=str(args.checkpoint),model=str(args.model),actual_checkpoint_verified=True)
            frozen.write_json(args.output/'identity.json',identity)
            if len(rows)!=558 or len({r['sample_id'] for r in rows})!=558:raise RuntimeError('Full558 identity')
            if any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen weights required')
            sampler=telemetry.TelemetrySampler(frozen.A100_UUID,raw_path=args.output/'telemetry_stdout.jsonl')
            sampler.start();sampler.set_phase('idle');time.sleep(2);sampler.set_phase(None)
            timings=[];predictions=[];targets=[];waits=[]
            with (args.output/'predictions.csv').open('x',newline='') as ps,(args.output/'batch_timing.csv').open('x',newline='') as ts:
                pw,tw=csv.writer(ps),csv.writer(ts)
                pw.writerow(['sample_index','sample_id','target_mos','prediction','batch_index','selected_frame_positions'])
                tw.writerow(['batch_index','videos','in_cost_subset','synchronized_wall_ms','cuda_ms','preprocessing_audit_ms','h2d_audit_ms'])
                for index,(start,end) in enumerate(costs.plan(args.batch)):
                    guard.check();sampler.check()
                    if time.time()>=args.deadline:raise TimeoutError('Finite measurement deadline')
                    tick=time.perf_counter();cpu,positions=prepare(rows[start:end]);prep=1000*(time.perf_counter()-tick)
                    tick=time.perf_counter();inputs=transfer(cpu);torch.cuda.synchronize();h2d=1000*(time.perf_counter()-tick)
                    in_cost=end<=200;sampler.set_phase(f'active:{index}' if in_cost else None)
                    try:values,wall,cuda=run(cpu,inputs)
                    finally:sampler.set_phase(None)
                    if len(values)!=end-start or wall<=0 or cuda<=0 or not all(math.isfinite(v) for v in values):raise RuntimeError('Invalid timed output')
                    timings.append([index,end-start,in_cost,wall,cuda,prep,h2d]);tw.writerow(timings[-1])
                    for offset,value in enumerate(values):
                        row=rows[start+offset];predictions.append(value);targets.append(float(row['temporal']))
                        pw.writerow([start+offset,row['sample_id'],row['temporal'],value,index,json.dumps(positions[offset])])
                    ps.flush();ts.flush()
                    requested=excluded_wait_seconds(wall,control['duty'])
                    sampler.set_phase(f'excluded_pacing:{index}');tick=time.monotonic();until=tick+requested
                    while time.monotonic()<until:
                        guard.check();sampler.check()
                        if time.time()>=args.deadline:raise TimeoutError('Finite pacing deadline')
                        time.sleep(min(.1,max(0,until-time.monotonic())))
                    waits.append(dict(batch_index=index,start_monotonic_s=tick,end_monotonic_s=time.monotonic(),requested_s=requested,in_cost_subset=in_cost))
                    sampler.set_phase(None);del inputs,cpu
                    status('evaluating',pid=os.getpid(),completed_test_videos=end)
                    if index==0 or end%50<args.batch or end==558:print(f'{args.baseline} {args.kind} B{args.batch} repeat{args.repeat} {end}/558 wall={wall:.4f}ms',flush=True)
            samples=sampler.stop();sampler=None;telemetry.save_samples(args.output/'telemetry.csv',samples)
            cost_rows=[r for r in timings if r[2]]
            power=measurement.power_report(samples,[r[3] for r in cost_rows],power_limit_w=250)
            result=telemetry.workload_cost(cost_rows,power['active_mean_w'])
            finish();finish=None;guard.check()
            if source_identity()!=bound or frozen.sha256(args.checkpoint)!=identity['checkpoint_sha256']:raise RuntimeError('Bound source or weight changed')
            report=dict(status='complete',**identity,test_videos=558,cost_test_videos=200,
                cost=result,power=power,performance=core.metrics(np.asarray(targets),np.asarray(predictions)),
                explicit_warmup_forwards=0,first_test_batch_included=True,
                timing_boundary='first native or equality-audited equivalent block input to GPU MOS synchronized host wall',
                excluded_from_final_cost=['decode','crop/resize','processor','H2D','model loading','inter-inference pacing waits'],
                excluded_pacing_total_s=sum(w['end_monotonic_s']-w['start_monotonic_s'] for w in waits),
                cost_run_active_utilization_audit_only=telemetry.telemetry_report(samples),
                displayed_utilization_source_is_independent=True,
                power_uncertainty_note='50ms driver power windows; sparse active samples are reported, not substituted with resident power',
                no_training_or_test_selection=True)
            frozen.write_json(args.output/'report.json',report);status('complete',pid=os.getpid(),completed_test_videos=558)
    except BaseException as error:
        status('failed',pid=os.getpid(),error=repr(error));raise
    finally:
        if sampler is not None:
            sampler.stop(raise_on_error=False);telemetry.save_samples(args.output/'telemetry.csv',sampler.samples)
        guard.stop();frozen.write_json(args.output/'process_audit.json',guard.records)
        frozen.write_json(args.output/'excluded_pacing_waits.json',locals().get('waits',[]))
        if finish is not None:finish()


def child_command(args,model,batch,kind,repeat,output):
    child=costs.child_command(model,batch,args,output)
    i=child.index(__package__+'.measure_frozen_batch_costs')
    child[i]=__package__+'.measure_unified_inference_recheck'
    return child+['--kind',kind,'--repeat',str(repeat)]


def suite(args,c):
    import fcntl
    reproduction.require_new_output(args.output)
    if args.deadline<=time.time() or args.deadline>time.time()+c['maximum_suite_seconds']+30:raise ValueError('Bounded90-minute window required')
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);costs.wait_for_idle(args,c);args.output.mkdir()
        bound=source_identity();state=dict(status='running',pid=os.getpid(),source_sha256=bound,protocol=c,deadline=args.deadline,completed=[])
        def save():frozen.write_json(args.output/'suite_status.json',state)
        save()
        try:
            for model,batch,kind,repeat in points(c):
                reproduction.cuda_guard()
                if time.time()>=args.deadline or source_identity()!=bound:raise RuntimeError('Deadline or source changed')
                name=f'{model}_{kind}_b{batch:02d}_r{repeat:02d}'
                state.update(phase=name,phase_started_unix=time.time());save()
                cmd=child_command(args,model,batch,kind,repeat,args.output/name)
                frozen.write_json(args.output/(name+'.command.json'),cmd)
                with (args.output/(name+'.log')).open('x') as log:
                    subprocess.run(cmd,cwd=REPO,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(c['maximum_point_seconds'],args.deadline-time.time()))
                report=json.loads((args.output/name/'report.json').read_text())
                if report['status']!='complete' or report['cost_test_videos']!=200:raise RuntimeError('Incomplete point')
                state['completed'].append(name);save()
            reproduction.cuda_guard();state.update(status='complete',finished_unix=time.time())
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    for key in ('manifest','output','model','checkpoint','reproduction-receipt'):p.add_argument('--'+key,type=Path,required=key in ('manifest','output'))
    p.add_argument('--baseline',choices=['clip','qwen','glm','yolo']);p.add_argument('--batch',type=int)
    p.add_argument('--kind',choices=['native','paced']);p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    for key in ('manifest','output','model','checkpoint','reproduction_receipt'):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve())
    if frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise ValueError('Formal manifest required')
    c=config()
    if args.phase=='worker':worker(args,c)
    else:suite(args,c)


if __name__=='__main__':main()
