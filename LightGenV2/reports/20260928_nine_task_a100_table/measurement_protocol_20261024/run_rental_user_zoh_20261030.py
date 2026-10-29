"""Human-authorized held-observation energy, actual gaps reported, five pending points.

2026-10-30 human explicitly requested finishing and using the prior integration
for observations older than250ms. This is not strict250ms qualification. No raw
sample is inserted, discarded, retimed or replaced. Native model/math unchanged.
"""
from bisect import bisect_right
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import run_rental_energy_remeasure_20261029 as entry
import rental_power_observer_20261029 as power

ENTRY=Path(__file__).resolve();ROOT=entry.ROOT.with_name('rental_energy_userZOH_20261030_04')
PIPE='/data/tmp/rental_energy_userZOH_20261030_04'
PROTOCOL='rental80_user_authorized_ZOH_actual_gaps_512_20261030'
VERSION='rental400_power_only_actual_API_return_no250ms_rejection_userZOH_v3'
NATIVE_CONFIGURE=entry.configure;NATIVE_CONFIG=entry.config
PLAN=[('deepseek',50,2),('deepseek',50,3),('deepseek',100,1),('deepseek',100,2),('deepseek',100,3)]

def interval_energy(intervals,samples,host_w=338.2):
    if not intervals or not samples:raise ValueError('Actual intervals and samples required')
    stamps=[s.host_monotonic_s for s in samples]
    if any(not math.isfinite(x) for x in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):raise ValueError('Actual monotonic sample timestamps required')
    elapsed=energy=gap_time=max_age=0.;used=set();long=set();previous=-math.inf
    for start,end in intervals:
        if not math.isfinite(start+end) or not previous<=start<end:raise ValueError('Ordered positive intervals required')
        previous=end;i=bisect_right(stamps,start)-1
        if i<0 or stamps[-1]<end:raise ValueError('Whole window must be bracketed by actual observations')
        tick=start
        while tick<end:
            boundary=min(end,stamps[i+1]);dt=boundary-tick;age=boundary-stamps[i];watts=samples[i].watts
            if not math.isfinite(watts) or not 0<watts<=1000:raise ValueError('Invalid actual board power')
            energy+=(host_w+watts)*dt;elapsed+=dt;max_age=max(max_age,age);used.add(i)
            if age>.25:gap_time+=dt;long.add(i)
            tick=boundary;i+=1
    return dict(elapsed_s=elapsed,energy_j=energy,power_samples_used=len(used),max_power_sample_age_s=max_age,
                host_proxy_w=host_w,power_integration='zero-order hold of actual previous reading; user-authorized gap-aware estimate',
                strict250ms_qualified=max_age<=.25,gap_over250ms_count=len(long),gap_over250ms_support_s=gap_time,
                gap_over250ms_support_fraction=gap_time/elapsed,no_synthetic_or_retimed_samples=True)

def observed_energy(intervals,raw):
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    return interval_energy(intervals,[Sample(float(x['host_time_unix']),float(x['host_monotonic_s']),x['phase'],float(x['watts']),None,None) for x in raw],0.)['energy_j']

def acquire(nvml,handle,phase,clock=time.monotonic,unix=time.time):
    start=clock();mw=nvml.nvmlDeviceGetPowerUsage(handle);end=clock();w=float(mw)/1000.
    if not start<=end or not math.isfinite(w) or not 0<w<=1000:raise ValueError('Invalid actual power observation')
    sample=power.direct.Sample(unix(),end,phase,w,None,None)
    return sample,dict(query_start_monotonic_s=start,power_API_return_monotonic_s=end,query_end_monotonic_s=end,
        power_milliwatts=mw,phase=phase,valid_numeric_row=True,utilization_percent=None,memory_used_mib=None,
        timestamp_source='actual API return; never backfilled',API_bracket_s=end-start,API_bracket_over250ms=end-start>.25)

class Sampler(power.PowerSampler):
    def start(self):
        super().start();self.identity.update(version=VERSION,API_bracket_maximum_s=None,
            qualification='user-authorized ZOH estimate; original250ms qualification reported separately',
            user_zoh_source_sha256=power.direct.hashlib.sha256(ENTRY.read_bytes()).hexdigest())

def observer_command(gpu,path):
    if gpu!=entry.GPU:raise ValueError('Exact original GPU required')
    return [sys.executable,'-u',str(ENTRY),'--observer','--gpu',gpu,'--raw',str(path)]

def config():return dict(NATIVE_CONFIG(),protocol_id=PROTOCOL,power_maximum_support_age_s=None,
    observer=VERSION,authorization='Human2026-10-30 explicitly requested prior held-power integration beyond250ms and finishing all remaining points',
    energy_qualification='user_authorized_ZOH_estimate_not_strict250ms',sampling_gaps_must_be_reported=True)

def audit(folder,manifest):
    result=entry.QUALIFIED_AUDIT(folder,manifest);r=entry.base.read(Path(folder)/'report.json')
    storage=entry.timing.storage_receipt(entry.timing.STORAGE_RECEIPT.read_text())
    entry.timing.validate_storage_input_hashes(storage,entry.base.input_rows(manifest),r['input_files'])
    obs=entry.base.read(Path(folder)/'telemetry_observer_identity.json')
    if obs['version']!=VERSION or obs['gpu_uuid']!=entry.GPU or obs['power_limit_milliwatts']!=400000:raise ValueError('Original GPU and declared sampler required')
    return dict(result,energy_qualified=True,qualification='user_authorized_ZOH_estimate_not_strict250ms',
                strict250ms_qualified=r['cost']['power_integration']['strict250ms_qualified'],original512_bytes_verified=True)

def configure():
    NATIVE_CONFIGURE();entry.SOURCES|={ENTRY.relative_to(entry.REPO).as_posix(),Path(entry.__file__).relative_to(entry.REPO).as_posix()}
    from LightGenV2.tasks.t06_video_quality_assessment import strict_tf_window_20261020 as windows
    from LightGenV2.tasks.t06_video_quality_assessment.reports import strict_tf_resource_audit_20261020 as independent
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    windows.interval_energy=interval_energy;independent.observed_energy=observed_energy;observer.observer_command=observer_command
    entry.base.config=config;entry.base.plan=lambda *a,**kw:PLAN;entry.base.audit_point=audit
    for fn in (entry.base.worker,entry.base.suite,entry.base.summarize,entry.QUALIFIED_AUDIT):
        fn.__globals__.update(vars(entry.base));fn.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=PROTOCOL)

if __name__=='__main__':
    if '--observer' in sys.argv:
        power.rental.configure_receiver();power.direct.acquire=acquire;power.direct.Sampler=Sampler
        power.direct.VERSION=VERSION;power.direct.main()
    else:
        entry.ROOT=ROOT;entry.PIPE=PIPE;entry.ENTRY=ENTRY;entry.PROTOCOL=PROTOCOL;entry.configure=configure;entry.main()
