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


def validated_duty(duty,observe,record):
    """Bounded PRE-measurement rehearsal; never retry/select formal trials.

    observe must retain its full raw window independently. Aim for +/-2 during
    calibration, while the unchanged formal acceptance remains +/-5.
    """
    for attempt in range(1,4):
        observed=observe(duty,attempt)
        if not math.isfinite(observed) or not 0<observed<=100:
            raise RuntimeError('Invalid calibration-validation busy rate')
        passed=abs(observed-50)<=2
        new=duty if passed else min(.999,max(.05,duty*min(1.15,max(.85,50/observed))))
        record(dict(stage='full_window_validation_v1',attempt=attempt,observed=observed,
                    old_duty=duty,new_duty=new,passed=passed))
        if passed:return duty
        duty=new
    raise RuntimeError('Three calibration rehearsals miss50+/-2; no formal paced trials started')


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


def multi_gpu_energy(intervals, streams):
    records=[interval_energy(intervals,rows,338.2 if index==0 else 0.) for index,rows in enumerate(streams)]
    if not records:raise ValueError('At least one observed GPU required')
    return dict(elapsed_s=records[0]['elapsed_s'],energy_j=sum(r['energy_j'] for r in records),
                host_proxy_w=338.2,per_gpu_energy=records,
                power_integration='sum observed used-GPU interval energies; host proxy counted ONCE')


def measure(args,compute,inputs,first,folder,guard,duty):
    """Three complete trials; pace calibration only, never select a best trial."""
    import torch
    folder.mkdir();gpus=tuple(getattr(args,'gpu_uuids',(args.gpu_uuid,)))
    samplers=[base.telemetry.TelemetrySampler(gpu,raw_path=folder/('telemetry_stdout.jsonl' if i==0 else f'telemetry_gpu{i:02d}_stdout.jsonl')) for i,gpu in enumerate(gpus)]
    clock_samplers=[base.prior.ClockSampler(gpu) for gpu in gpus]
    armed={};calls=[];waits=[];repeats=[];updates=[];calibration_calls=[];calibration_waits=[]
    def check():
        guard.fast()
        for sampler in samplers:sampler.check()
    def phase(value):
        for sampler in samplers:sampler.set_phase(value)
    def synchronize():
        if len(gpus)==1:torch.cuda.synchronize()
        else:
            for device in range(len(gpus)):torch.cuda.synchronize(device)
    def first_input(*_):
        if armed.get('enabled') and armed.get('start') is None:
            # Drain GPU frontend kernels BEFORE starting the charged TF interval.
            synchronize();armed['start']=time.monotonic()
    hook=first.register_forward_pre_hook(first_input)
    def group(rep=None,audit=None):
        check();start=time.monotonic();duration=0.
        for index in range(8):
            armed.update(enabled=True,start=None);before=time.monotonic()
            out,_=compute(inputs);synchronize();end=time.monotonic();armed['enabled']=False
            tf=armed['start']
            if tf is None or not before<=tf<end:raise RuntimeError('Missing or invalid first-block boundary')
            duration+=end-tf
            records=calls if rep is not None else audit
            if records is not None:records.append(dict(repeat=rep,call_index=len(records),call_start=before,
                first_block_start=tf,gpu_output=end,excluded_preblock_s=tf-before,compute_s=end-tf))
        return out,start,time.monotonic(),duration
    def idle(seconds):
        start=time.monotonic();until=start+seconds
        while time.monotonic()<until:
            check();time.sleep(min(.05,max(0,until-time.monotonic())))
        return start,time.monotonic()
    def warm(seconds,d):
        start=time.monotonic();durations=[]
        while time.monotonic()-start<seconds:
            _,a,b,c=group();durations.append(c)
            if d<1:idle((b-a)*(1/d-1))
        return durations
    active_samplers=[];active_clocks=[]
    try:
        for sampler in samplers:sampler.start();active_samplers.append(sampler)
        for clocks in clock_samplers:clocks.start();active_clocks.append(clocks)
        times=warm(5,1)
        # Synchronization has changed, so old queue8 duties are NOT reused.
        if duty<1:
            phase('calibration');start=last=time.monotonic();duty=.5
            while time.monotonic()-start<60:
                _,a,b,_=group();idle((b-a)*(1/duty-1));now=time.monotonic()
                if now-last>=5:
                    streams=[[s.utilization_percent for s in sampler.samples if last<=s.host_monotonic_s<now] for sampler in samplers]
                    if any(len(values)<10 for values in streams):raise RuntimeError('Insufficient calibration telemetry')
                    per_gpu=[statistics.mean(values) for values in streams]
                    observed=max(per_gpu);new=min(1,max(.05,duty*min(1.25,max(.8,50/max(observed,1)))))
                    updates.append(dict(start=last,end=now,observed=observed,per_gpu_busy=per_gpu,old_duty=duty,new_duty=new))
                    duty=new;last=now
            if getattr(args,'validate_pacing',False):
                def observe(candidate,attempt):
                    phase('calibration_validation_warmup'+str(attempt));warm(5,candidate)
                    phase('calibration_validation'+str(attempt))
                    count=max(2,math.ceil(35*candidate/statistics.mean(times)))
                    if count>100000:raise RuntimeError('Bounded rehearsal count exceeded')
                    wall_start=time.monotonic();begin=len(calibration_calls)
                    for i in range(count):
                        _,a,b,_=group(audit=calibration_calls)
                        if i+1<count:
                            request=(b-a)*(1/candidate-1)*count/(count-1)
                            wa,wb=idle(request)
                            calibration_waits.append(dict(attempt=attempt,group=i,start=wa,end=wb,requested_s=request))
                    wall_end=time.monotonic()
                    selected=calibration_calls[begin:]
                    phase('outside_calibration_validation')
                    streams=[[s.utilization_percent for s in sampler.samples if wall_start<=s.host_monotonic_s<wall_end] for sampler in samplers]
                    if any(len(values)<100 for values in streams):raise RuntimeError('Insufficient full rehearsal telemetry')
                    observed=max(statistics.mean(values) for values in streams)
                    base.old.write(folder/f'calibration_validation{attempt:02d}.json',dict(
                        wall_start=wall_start,wall_end=wall_end,observed=observed,duty=candidate,
                        groups=count,forward_calls=len(selected),not_formal_measurement=True))
                    return observed
                def record(update):
                    updates.append(update);base.old.write(folder/'controller_updates.json',updates)
                duty=validated_duty(duty,observe,record)
        count=max(2,math.ceil(35*duty/statistics.mean(times)))
        if count>100000:raise RuntimeError('Bounded complete-trial count exceeded')
        for rep in (1,2,3):
            phase('warmup'+str(rep));warm(5,duty)
            phase('measured'+str(rep));first_call=len(calls);first_wait=len(waits)
            wall_start=time.monotonic()
            for i in range(count):
                out,a,b,_=group(rep)
                if i+1<count and duty<1:
                    request=(b-a)*(1/duty-1)*count/(count-1);wa,wb=idle(request)
                    waits.append(dict(repeat=rep,group=i,start=wa,end=wb,requested_s=request))
            wall_end=time.monotonic();phase('outside_window')
            selected_calls=calls[first_call:];selected_waits=waits[first_wait:]
            # Only actual first-TF->output and actual INTERNAL control waits count.
            intervals=sorted([(c['first_block_start'],c['gpu_output']) for c in selected_calls]+
                             [(w['start'],w['end']) for w in selected_waits])
            energy=multi_gpu_energy(intervals,[list(sampler.samples) for sampler in samplers])
            if energy['elapsed_s']<30:raise RuntimeError('Charged complete trial shorter than30s')
            infos=[base.telemetry.telemetry_report([s for s in sampler.samples if wall_start<=s.host_monotonic_s<wall_end],prefix='') for sampler in samplers]
            info=infos[0];videos=count*8*args.batch
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
                full_wall_telemetry=info,per_gpu_full_wall_telemetry=dict(zip(gpus,infos)),output=output)
            repeats.append(result);base.old.write(folder/f'repeat{rep:02d}.json',result)
            print(f'{args.model} {args.precision} {args.method} repeat{rep}: busy={info["mean_gpu_utilization_percent"]:.3f}% T16={result["time16_ms"]:.6f}ms',flush=True)
        videos=sum(r['video_evaluations'] for r in repeats)
        per_gpu_busy={gpu:sum(r['per_gpu_full_wall_telemetry'][gpu]['samples']*r['per_gpu_full_wall_telemetry'][gpu]['mean_gpu_utilization_percent'] for r in repeats)/sum(r['per_gpu_full_wall_telemetry'][gpu]['samples'] for r in repeats) for gpu in gpus}
        busy=max(per_gpu_busy.values())
        report=dict(status='complete',timing_protocol=PROTOCOL,model=args.model,precision=args.precision,
            method=args.method,batch=args.batch,nominal_resource_cap=args.cap if args.method=='mps' else None,
            frozen_duty=duty,repeats=repeats,time16_ms=sum(r['elapsed_s'] for r in repeats)*16000/videos,
            pacing_calibration='full_window_validation_v1' if duty<1 and getattr(args,'validate_pacing',False) else 'short_feedback60s_v1',
            energy16_j=sum(r['energy_j'] for r in repeats)*16/videos,mean_gpu_busy_percent=busy,
            per_gpu_busy_percent=per_gpu_busy,allocation_average_busy_percent=statistics.mean(per_gpu_busy.values()),
            busy_control_basis='single GPU' if len(gpus)==1 else 'maximum of per-GPU pooled busy rates; NOT allocation average',
            target50_reached=abs(busy-50)<=5 if args.method=='temporal' and duty<1 else None,
            unrestricted_target98_reached=busy>=98 if duty==1 else None,
            every_preblock_interval_excluded=True,internal_wait_included=True,terminal_wait_excluded=True,
            individually_synchronized_forwards=True,cost_intervals_are_not_contiguous_wall_window=True,
            utilization_basis='all samples in complete wall trials, not masked or peak-selected',
            accuracy_not_evaluated_in_repeated_cost_window=True,throughput_normalized_not_single_batch_latency=True,
            host_proxy_w=338.2,no_window_selection=True)
        base.old.write(folder/'report.json',report);return report
    finally:
        hook.remove();errors=[]
        for i,sampler in enumerate(active_samplers):
            try:
                samples=sampler.stop(raise_on_error=False)
                base.telemetry.save_samples(folder/('telemetry.csv' if i==0 else f'telemetry_gpu{i:02d}.csv'),samples)
                sampler.check()
            except BaseException as error:errors.append(repr(error))
        for i,clocks in enumerate(active_clocks):
            try:clocks.stop(folder/('clock_telemetry.csv' if i==0 else f'clock_telemetry_gpu{i:02d}.csv'))
            except BaseException as error:errors.append(repr(error))
        for name,records in (('calls.csv',calls),('waits.csv',waits),('calibration_calls.csv',calibration_calls),('calibration_waits.csv',calibration_waits)):
            with (folder/name).open('x',newline='') as stream:
                if records:
                    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        base.old.write(folder/'controller_updates.json',updates)
        if errors:raise RuntimeError('Telemetry cleanup/audit failed: '+repr(errors))
