"""Per-forward first-block accounting, with explicit internal pacing waits.

This is a NEW timing protocol, not a relabelling of complete_window results.
Every forward is synchronized at first-block entry and after GPU MOS output.
Pre-block work is executed normally but excluded from cost; never cached away.
The full wall-window NVML busy rate is reported separately from cost intervals.
"""
from __future__ import annotations
from bisect import bisect_right
import csv
import math
from pathlib import Path
import statistics
import time
from . import rental_precision_recheck as base

PROTOCOL='per_forward_first_block_plus_internal_wait_v1'


def interval_energy(intervals, samples, host_w=338.2):
    """Integrate held board-power readings over exactly the charged intervals.

    Polling is not instantaneous metering. Preserve raw readings, enforce a
    maximum 250-ms sample age, and never substitute resident/idle power values.
    """
    if not intervals or not samples:raise ValueError('Intervals and telemetry required')
    stamps=[s.host_monotonic_s for s in samples]
    if any(b<=a for a,b in zip(stamps,stamps[1:])):raise ValueError('Telemetry timestamps must increase')
    elapsed=energy=0.;used=set();previous_end=-math.inf;max_age=0.
    for start,end in intervals:
        if not math.isfinite(start+end) or not previous_end<=start<end:raise ValueError('Ordered nonoverlapping positive intervals required')
        previous_end=end;index=bisect_right(stamps,start)-1
        if index<0:raise ValueError('No observed power before interval')
        tick=start
        while tick<end:
            boundary=min(end,stamps[index+1] if index+1<len(stamps) else end)
            age=boundary-stamps[index]
            if age>.25:raise ValueError('Power observation older than250ms')
            watts=samples[index].watts
            if not math.isfinite(watts) or watts<=0:raise ValueError('Invalid observed board power')
            energy+=(host_w+watts)*(boundary-tick);elapsed+=boundary-tick
            max_age=max(max_age,age);used.add(index);tick=boundary;index+=1
    return dict(elapsed_s=elapsed,energy_j=energy,power_samples_used=len(used),
                max_power_sample_age_s=max_age,host_proxy_w=host_w,
                power_integration='zero-order hold on charged intervals; raw50ms NVML observations')


def measure(args,compute,inputs,first,folder,guard,duty):
    """Three complete trials; pace calibration only, never select a best trial."""
    import torch
    folder.mkdir();sampler=base.telemetry.TelemetrySampler(args.gpu_uuid,raw_path=folder/'telemetry_stdout.jsonl')
    clocks=base.prior.ClockSampler(args.gpu_uuid);armed={};calls=[];waits=[];repeats=[];updates=[]
    def first_input(*_):
        if armed.get('enabled') and armed.get('start') is None:
            # Drain GPU frontend kernels BEFORE starting the charged TF interval.
            torch.cuda.synchronize();armed['start']=time.monotonic()
    hook=first.register_forward_pre_hook(first_input)
    def group(rep=None):
        guard.fast();sampler.check();start=time.monotonic();duration=0.
        for index in range(8):
            armed.update(enabled=True,start=None);before=time.monotonic()
            out,_=compute(inputs);torch.cuda.synchronize();end=time.monotonic();armed['enabled']=False
            tf=armed['start']
            if tf is None or not before<=tf<end:raise RuntimeError('Missing or invalid first-block boundary')
            duration+=end-tf
            if rep is not None:calls.append(dict(repeat=rep,call_index=len(calls),call_start=before,
                first_block_start=tf,gpu_output=end,excluded_preblock_s=tf-before,compute_s=end-tf))
        return out,start,time.monotonic(),duration
    def idle(seconds):
        start=time.monotonic();until=start+seconds
        while time.monotonic()<until:
            guard.fast();sampler.check();time.sleep(min(.05,max(0,until-time.monotonic())))
        return start,time.monotonic()
    def warm(seconds,d):
        start=time.monotonic();durations=[]
        while time.monotonic()-start<seconds:
            _,a,b,c=group();durations.append(c)
            if d<1:idle((b-a)*(1/d-1))
        return durations
    sampler.start();clocks.start()
    try:
        times=warm(5,1)
        # Synchronization has changed, so old queue8 duties are NOT reused.
        if duty<1:
            sampler.set_phase('calibration');start=last=time.monotonic();duty=.5
            while time.monotonic()-start<60:
                _,a,b,_=group();idle((b-a)*(1/duty-1));now=time.monotonic()
                if now-last>=5:
                    values=[s.utilization_percent for s in sampler.samples if last<=s.host_monotonic_s<now]
                    if len(values)<10:raise RuntimeError('Insufficient calibration telemetry')
                    observed=statistics.mean(values);new=min(1,max(.05,duty*min(1.25,max(.8,50/max(observed,1)))))
                    updates.append(dict(start=last,end=now,observed=observed,old_duty=duty,new_duty=new))
                    duty=new;last=now
        count=max(2,math.ceil(35*duty/statistics.mean(times)))
        if count>100000:raise RuntimeError('Bounded complete-trial count exceeded')
        for rep in (1,2,3):
            sampler.set_phase('warmup'+str(rep));warm(5,duty)
            sampler.set_phase('measured'+str(rep));first_call=len(calls);first_wait=len(waits)
            wall_start=time.monotonic()
            for i in range(count):
                out,a,b,_=group(rep)
                if i+1<count and duty<1:
                    request=(b-a)*(1/duty-1)*count/(count-1);wa,wb=idle(request)
                    waits.append(dict(repeat=rep,group=i,start=wa,end=wb,requested_s=request))
            wall_end=time.monotonic();sampler.set_phase('outside_window')
            selected_calls=calls[first_call:];selected_waits=waits[first_wait:]
            # Only actual first-TF->output and actual INTERNAL control waits count.
            intervals=sorted([(c['first_block_start'],c['gpu_output']) for c in selected_calls]+
                             [(w['start'],w['end']) for w in selected_waits])
            energy=interval_energy(intervals,list(sampler.samples))
            if energy['elapsed_s']<30:raise RuntimeError('Charged complete trial shorter than30s')
            rows=[s for s in sampler.samples if wall_start<=s.host_monotonic_s<wall_end]
            info=base.telemetry.telemetry_report(rows,prefix='');videos=count*8*args.batch
            output=out.detach().cpu().tolist()
            if not output or not all(math.isfinite(float(v)) for v in output):raise RuntimeError('Nonfinite GPU MOS output')
            if repeats and any(abs(float(a)-float(b))>2e-4 for a,b in zip(output,repeats[0]['output'])):
                raise RuntimeError('Repeated frozen-input outputs changed')
            result=dict(repeat=rep,timing_protocol=PROTOCOL,wall_start=wall_start,wall_end=wall_end,
                wall_elapsed_s=wall_end-wall_start,groups=count,forward_calls=count*8,video_evaluations=videos,
                unique_videos=args.batch,compute_s=sum(c['compute_s'] for c in selected_calls),
                internal_wait_s=sum(w['end']-w['start'] for w in selected_waits),
                excluded_preblock_s=sum(c['excluded_preblock_s'] for c in selected_calls),
                **energy,time16_ms=energy['elapsed_s']*16000/videos,energy16_j=energy['energy_j']*16/videos,
                full_wall_telemetry=info,output=output)
            repeats.append(result);base.old.write(folder/f'repeat{rep:02d}.json',result)
            print(f'{args.model} {args.precision} {args.method} repeat{rep}: busy={info["mean_gpu_utilization_percent"]:.3f}% T16={result["time16_ms"]:.6f}ms',flush=True)
        videos=sum(r['video_evaluations'] for r in repeats)
        busy=sum(r['full_wall_telemetry']['samples']*r['full_wall_telemetry']['mean_gpu_utilization_percent'] for r in repeats)/sum(r['full_wall_telemetry']['samples'] for r in repeats)
        report=dict(status='complete',timing_protocol=PROTOCOL,model=args.model,precision=args.precision,
            method=args.method,batch=args.batch,nominal_resource_cap=args.cap if args.method=='mps' else None,
            frozen_duty=duty,repeats=repeats,time16_ms=sum(r['elapsed_s'] for r in repeats)*16000/videos,
            energy16_j=sum(r['energy_j'] for r in repeats)*16/videos,mean_gpu_busy_percent=busy,
            target50_reached=abs(busy-50)<=5 if args.method=='temporal' and duty<1 else None,
            unrestricted_target98_reached=busy>=98 if duty==1 else None,
            every_preblock_interval_excluded=True,internal_wait_included=True,terminal_wait_excluded=True,
            individually_synchronized_forwards=True,cost_intervals_are_not_contiguous_wall_window=True,
            utilization_basis='all samples in complete wall trials, not masked or peak-selected',
            accuracy_not_evaluated_in_repeated_cost_window=True,throughput_normalized_not_single_batch_latency=True,
            host_proxy_w=338.2,no_window_selection=True)
        base.old.write(folder/'report.json',report);return report
    finally:
        hook.remove();samples=sampler.stop(raise_on_error=False);base.telemetry.save_samples(folder/'telemetry.csv',samples)
        clocks.stop(folder/'clock_telemetry.csv')
        for name,records in (('calls.csv',calls),('waits.csv',waits)):
            with (folder/name).open('x',newline='') as stream:
                if records:
                    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        base.old.write(folder/'controller_updates.json',updates);sampler.check()
