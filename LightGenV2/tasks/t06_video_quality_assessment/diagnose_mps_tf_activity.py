"""Finite MPS/B1 conditional kernel-busy diagnosis; never official cost data.

CUPTI kernel intervals are clipped to the unchanged native TF-entry/output
boundaries. NVML samples remain unfiltered audit data, not an exact short-window
counter. Profiling overhead is disclosed. No warming, pacing, synthetic work,
Graph, training or changes to the protected costs.
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
import sys
import time
from . import measure_mps_dataset_once as original

TASK=original.TASK
REPO=original.REPO
ROOT=TASK/'runs/simulation/lgvq_mps_tf_activity_a100_20261021'
CONFIG=TASK/'configs/baselines/mps_tf_activity_20261021.json'
PIPE='/DATA/DATA1/guest3/.cache/t06_mps_tf_activity_20261021'


def configuration():
    c=json.loads(CONFIG.read_text())
    assert c['models']==['clip','qwen','yolo','glm']
    assert c['caps']==[100,50,25,10] and c['batch']==1
    assert c['videos']==12 and c['maximum_suite_s']==2700
    assert c['no_formal_cost_overwrite'] and c['no_explicit_warmup']
    return c


def sources():
    return {**original.sources(),**{str(p.relative_to(REPO)):original.frozen.sha256(p)
        for p in (Path(__file__),CONFIG)}}


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if not math.isfinite(start+end) or end<start:raise ValueError('Invalid interval')
        if end==start:continue
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def duration(intervals):return sum(b-a for a,b in union(intervals))


def clipped_duration(kernels,start,end):
    return duration((max(start,a),min(end,b)) for a,b in kernels if max(start,a)<min(end,b))


def summarize_trace(trace,calls,anchors):
    events=trace['traceEvents']
    kernels=union((float(e['ts']),float(e['ts'])+float(e['dur'])) for e in events
        if e.get('ph')=='X' and e.get('cat')=='kernel' and e.get('dur',0)>0)
    if not kernels:raise RuntimeError('No CUPTI CUDA kernels: cannot report conditional utilization')
    low,high=-math.inf,math.inf
    for anchor in anchors:
        matches=[e for e in events if e.get('ph')=='X' and e.get('name')==anchor['name']]
        if len(matches)!=1:raise RuntimeError('Missing or duplicate clock anchor')
        event=matches[0];ts=float(event['ts']);end=ts+float(event['dur'])
        low=max(low,end-anchor['after_us']);high=min(high,ts-anchor['before_us'])
    if not low<=high or not math.isfinite(low+high):raise RuntimeError('Clock correlation has no consistent bound')
    offset=(low+high)/2;error=(high-low)/2
    per_video=[]
    for call in calls:
        start=call['first_block_start']*1e6+offset;end=call['gpu_output']*1e6+offset
        elapsed=end-start
        if elapsed<=0:raise RuntimeError('Nonpositive TF/output interval')
        busy=clipped_duration(kernels,start,end)
        per_video.append(dict(sample_index=call['sample_index'],sample_id=call['sample_id'],
            tf_to_output_us=elapsed,kernel_busy_us=busy,
            conditional_kernel_busy_percent=100*busy/elapsed))
    elapsed=sum(x['tf_to_output_us'] for x in per_video);busy=sum(x['kernel_busy_us'] for x in per_video)
    uncertainty=2*error*len(calls)
    lower=100*max(0,busy-uncertainty)/elapsed;upper=100*min(elapsed,busy+uncertainty)/elapsed
    return dict(metric='CUPTI profiled compute-kernel busy fraction conditional on native first-block-to-GPU-output intervals',
        conditional_kernel_busy_percent=100*busy/elapsed,lower_percent=lower,upper_percent=upper,
        kernel_union_us=busy,tf_to_output_us=elapsed,clock_offset_us=offset,clock_error_bound_us=error,
        target50_reached=45<=lower and upper<=55,target100_reached=95<=lower and upper<=100,
        per_video=per_video,nvml_utilization_equivalent=False,sm_occupancy_equivalent=False,
        profiled_diagnostic_not_formal_cost=True,all_cold_and_later_samples_retained=True)


def worker(args):
    import torch
    c=configuration();engine=original.engine;frozen=original.frozen
    engine.GPU=frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT
    if args.output.exists():raise RuntimeError('Preserve prior worker evidence')
    if os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap):raise RuntimeError('Cap must precede CUDA initialization')
    args.output.mkdir();bound=sources();calls=[];anchors=[];sampler=None;finish=None
    class Guard(engine.MPSGuard):
        def check(self):
            if sources()!=bound:raise RuntimeError('Diagnostic source changed')
            super().check()
    guard=Guard(args)
    identity=dict(model=args.model,cap=args.cap,batch=1,pid=os.getpid(),command=sys.argv,sources=bound,
        checkpoint_sha256=frozen.sha256(args.checkpoint),manifest_sha256=frozen.sha256(args.manifest),
        gpu_uuid=frozen.A100_UUID,protocol=c,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def status(value,**extra):frozen.write_json(args.output/'status.json',dict(status=value,time=time.time(),**extra))
    def anchor():
        name='T06_clock_'+str(len(anchors));before=time.perf_counter()*1e6
        with torch.profiler.record_function(name):pass
        anchors.append(dict(name=name,before_us=before,after_us=time.perf_counter()*1e6))
    try:
        guard.check();torch.set_num_threads(8)
        if identity['manifest_sha256']!=frozen.MANIFEST_SHA:raise RuntimeError('Original split required')
        limit=float(subprocess.check_output(['nvidia-smi','--id='+frozen.A100_UUID,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if abs(limit-250)>.1:raise RuntimeError('Unchanged250W required')
        status('loading')
        with torch.inference_mode():
            rows,prepare,transfer,forward,start,finish=original.runtime(args,guard,identity)
            visible=torch.cuda.get_device_properties(0).multi_processor_count
            if not 1<=visible<=108 or (args.cap==100 and visible!=108) or (args.cap==50 and visible!=54):raise RuntimeError('MPS cap not effective')
            identity.update(visible_multiprocessors=visible,power_limit_w=limit,profiler_overhead_present=True,
                official_costs_unchanged=True,explicit_warmup_forwards=0)
            frozen.write_json(args.output/'identity.json',identity)
            sampler=original.ProcessTelemetrySampler(frozen.A100_UUID,args.output/'telemetry_stdout.jsonl')
            sampler.start();sampler.set_phase('profiled_tf_activity')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for i,row in enumerate(rows[:c['videos']]):
                    guard.fast();sampler.check()
                    cpu,positions=prepare([row]);inputs=transfer(cpu);torch.cuda.synchronize();anchor()
                    values,wall,cuda=forward(inputs);s=start();e=s+wall/1000;anchor()
                    if len(values)!=1 or not math.isfinite(values[0]):raise RuntimeError('Invalid output')
                    calls.append(dict(sample_index=i,sample_id=row['sample_id'],first_block_start=s,gpu_output=e,
                        prediction=values[0],target_mos=row['temporal'],legacy_host_ms=wall,cuda_ms=cuda,
                        frame_positions=positions[0]))
                    del inputs,cpu
                    status('profiling',samples=i+1,total=c['videos'])
                    print(f'{args.model} cap{args.cap} {i+1}/{c["videos"]} profiled_wall={wall:.3f}ms',flush=True)
            target=args.output/'cuda_trace.json';prof.export_chrome_trace(str(target))
            if target.stat().st_size>256*1024**2:raise RuntimeError('Trace exceeds finite storage budget')
            result=summarize_trace(json.loads(target.read_text()),calls,anchors)
            samples=sampler.stop();sampler=None;original.telemetry.save_samples(args.output/'telemetry.csv',samples)
            finish();finish=None;guard.check()
            report=dict(status='complete',identity=identity,measurement=result,unfiltered_nvml_audit=original.telemetry.telemetry_report(samples,prefix=''))
            frozen.write_json(args.output/'report.json',report);status('complete',samples=len(calls))
            print(json.dumps(dict(model=args.model,cap=args.cap,sm=visible,measurement=result)),flush=True)
    except BaseException as error:status('failed',error=repr(error),samples=len(calls));raise
    finally:
        if sampler is not None:original.telemetry.save_samples(args.output/'telemetry.csv',sampler.stop(raise_on_error=False))
        if finish is not None:finish()
        frozen.write_json(args.output/'calls.json',calls);frozen.write_json(args.output/'clock_anchors.json',anchors)
        guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    c=configuration();engine=original.engine;engine.GPU=original.frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ROOT.exists():raise RuntimeError('Do not overwrite any diagnostic')
        ROOT.mkdir();state=dict(status='admission',pid=os.getpid(),completed=[],protocol=c,deadline=args.deadline);active=False
        def save():original.frozen.write_json(ROOT/'suite_status.json',state)
        save()
        try:
            for _ in range(3):
                if original.lab.shared.guard_records(engine.GPU):raise RuntimeError('Foreign A100 process; no waiting or restart')
                original.validate_idle_memory(engine.GPU)
                if time.time()>=args.deadline:raise TimeoutError('Diagnostic deadline')
                time.sleep(10)
            engine.daemon(True);active=True
            for model in c['models']:
                mp,pt,python,receipt=original.lab.locations(model);found50=found100=False
                for cap in c['caps']:
                    if cap not in (100,50) and found50 and found100:break
                    if args.deadline-time.time()<120:raise TimeoutError('No time for another diagnostic worker')
                    name=f'{model}_cap{cap}';output=ROOT/name
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=engine.GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,
                        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache')
                    cmd=[python,'-u','-m',__package__+'.diagnose_mps_tf_activity','--phase','worker','--model',model,
                        '--cap',str(cap),'--model-path',str(mp),'--checkpoint',str(pt),'--manifest',str(args.manifest),
                        '--output',str(output),'--deadline',str(args.deadline)]
                    if receipt:cmd+=['--reproduction-receipt',str(receipt)]
                    with (ROOT/(name+'.log')).open('x') as log:
                        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running',phase=name,child_pid=child.pid);save()
                        try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                        except BaseException:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                    if code:raise RuntimeError(f'{name} exit{code}; preserve evidence, no automatic retry')
                    report=json.loads((output/'report.json').read_text());found50|=report['measurement']['target50_reached'];found100|=report['measurement']['target100_reached']
                    state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:
            if active:
                try:engine.daemon(False);state['mps_released']=True
                except BaseException as error:state['release_error']=repr(error)
            save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('suite','worker'),required=True)
    p.add_argument('--model',choices=('clip','qwen','yolo','glm'));p.add_argument('--cap',type=int,choices=(100,50,25,10))
    p.add_argument('--model-path',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    if not 0<a.deadline-time.time()<=configuration()['maximum_suite_s']:raise ValueError('Finite2700s budget required')
    if a.phase=='suite':suite(a)
    else:worker(a)


if __name__=='__main__':main()
