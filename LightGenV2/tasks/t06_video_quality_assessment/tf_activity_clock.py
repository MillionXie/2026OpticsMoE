"""Clock correlation for CUDA-only diagnostic traces, without CPU op recording.

Kineto JSON timestamps are microseconds relative to baseTimeNanoseconds. Empty
CPU-only traces before/after the GPU trace check the Unix/perf-clock conversion.
The error allowance is empirical, not a calibrated hardware clock guarantee.
"""
import json
import math
import time


def annotation_bounds(trace,anchors):
    low,high=-math.inf,math.inf
    for a in anchors:
        events=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('name')==a['name']]
        if len(events)!=1:raise RuntimeError('Clock annotation missing or ambiguous')
        e=events[0];low=max(low,e['ts']+e['dur']-a['after_us']);high=min(high,e['ts']-a['before_us'])
    if not low<=high or not math.isfinite(low+high):raise RuntimeError('Inconsistent annotation clock')
    return low,high


def epoch_bounds(trace,anchors,allowance_us=0):
    base=trace['baseTimeNanoseconds'];low,high=-math.inf,math.inf
    for a in anchors:
        stamp=(a['wall_ns']-base)/1000
        low=max(low,stamp-a['after_us']-allowance_us)
        high=min(high,stamp-a['before_us']+allowance_us)
    if not low<=high or not math.isfinite(low+high):raise RuntimeError('Inconsistent Unix/perf clock')
    return low,high


def calibration_error(trace,anchors):
    a,b=annotation_bounds(trace,anchors);c,d=epoch_bounds(trace,anchors)
    return max(abs(a-d),abs(b-c))


def snapshot():
    before=time.perf_counter()*1e6;wall=time.time_ns();after=time.perf_counter()*1e6
    return dict(before_us=before,wall_ns=wall,after_us=after)


def empty_cpu_calibration(folder,name):
    import torch
    anchors=[]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for i in range(16):
            label='clock_'+str(i);before=time.perf_counter()*1e6;wall=time.time_ns()
            with torch.profiler.record_function(label):pass
            anchors.append(dict(name=label,before_us=before,wall_ns=wall,after_us=time.perf_counter()*1e6))
    target=folder/(name+'.json');prof.export_chrome_trace(str(target));trace=json.loads(target.read_text())
    error=calibration_error(trace,anchors)
    if error>100:raise RuntimeError('Clock calibration exceeds100us; preserve diagnostic')
    with (folder/(name+'_anchors.json')).open('x') as stream:json.dump(anchors,stream,indent=2)
    return error
