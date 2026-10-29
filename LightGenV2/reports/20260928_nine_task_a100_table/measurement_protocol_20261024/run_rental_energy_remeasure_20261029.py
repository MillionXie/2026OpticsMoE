"""New human-authorized full Kimi/DS time+energy remeasurement, never a resume.

Native full models, trained heads, first512 TEST/B1/four-frame preparation,
MPS54/108 and all input-byte/metadata gates are reused. Power-only NVML avoids
blocking utilization/memory diagnostics; their missing values stay missing.
Same actual 250ms support bound and whole file-to-CPU-score energy integral.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import run_rental_fullscope_20261029 as native
import run_rental_timing_only_20261029 as timing
import rental_power_observer_20261029 as power

ENTRY=Path(__file__).resolve();base=native.base;REPO=native.REPO
ROOT=base.TASK/'runs/simulation/rental_energy_remeasure_20261029_01'
PIPE='/data/tmp/rental_energy_remeasure_20261029_01';GPU=native.GPU
PROTOCOL='rental80_REMEASURE_FULL_TIME_ENERGY_powerOnly250ms_512x3_20261029'
ORIGINAL_AUDIT=base.audit_point
SOURCES=None;QUALIFIED_AUDIT=None

def sources():return {n:base.sha(REPO/n) for n in SOURCES}
def config():
    return dict(native.config(),protocol_id=PROTOCOL,
        authorization='2026-10-29 human explicitly requested both LGVQ full models remeasured with energy and LightGen on same rental',
        energy_qualified_requires_full_window_audit=True,
        observer=power.VERSION,power_maximum_support_age_s=.25,
        telemetry_missing_fields=['utilization_percent','memory_used_mib'],
        resource_guard='unchanged original independent MPS/process guard',
        input_storage='same original512 bytes on verified local overlay',
        host_proxy_is_not_measured_rental_host_power=True)

def audit_point(folder,manifest):
    folder=Path(folder);result=QUALIFIED_AUDIT(folder,manifest)
    report=base.read(folder/'report.json')
    storage=timing.storage_receipt(timing.STORAGE_RECEIPT.read_text())
    timing.validate_storage_input_hashes(storage,base.input_rows(manifest),report['input_files'])
    with (folder/'telemetry.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows or any(r['utilization_percent'] or r['memory_used_mib'] for r in rows):
        raise ValueError('Power-only diagnostics must remain explicitly missing')
    obs=base.read(folder/'telemetry_observer_identity.json')
    if obs['version']!=power.VERSION or obs['gpu_uuid']!=GPU or obs['power_limit_milliwatts']!=400000:
        raise ValueError('Exact power-only sampler and original400W required')
    if report['cost']['power_integration']['max_power_sample_age_s']>.25:
        raise ValueError('Original250ms support gate failed')
    return dict(result,energy_qualified=True,power_only=True,
                utilization_and_memory_not_measured=True,original512_bytes_verified=True)

def configure():
    global SOURCES,QUALIFIED_AUDIT
    native.configure()
    paths=[ENTRY,Path(power.__file__),Path(timing.__file__)]
    SOURCES=set(native.sources())|{p.relative_to(REPO).as_posix() for p in paths}
    base.CANONICAL=ROOT;base.source_map=sources;base.config=config
    QUALIFIED_AUDIT=native.adapter.adapt(ORIGINAL_AUDIT,power_only=True)
    base.audit_point=audit_point
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    observer.observer_command=power.observer_command
    for fn in (base.worker,base.suite,base.summarize,QUALIFIED_AUDIT):
        fn.__globals__.update(vars(base));fn.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=PROTOCOL)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('suite','worker','audit'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suite-root',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--deadline',type=float)
    p.add_argument('--profile',choices=('rental',),default='rental')
    p.add_argument('--gpu-uuid',choices=(GPU,),default=GPU);p.add_argument('--pipe',default=PIPE)
    p.add_argument('--model',choices=('kimi','deepseek'));p.add_argument('--cap',type=int,choices=(50,100))
    p.add_argument('--repeat',type=int,choices=(1,2,3));p.add_argument('--model-path',type=Path);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.suite_root=(a.suite_root or a.output).resolve()
    if (a.suite_root!=ROOT or a.pipe!=PIPE or (a.phase=='suite' and a.output!=ROOT)
            or (a.phase!='suite' and a.output.parent!=ROOT)):raise ValueError('One new authorized immutable remeasurement root')
    if a.phase!='audit' and (a.deadline is None or not 0<a.deadline-time.time()<=9000):
        raise ValueError('New bounded remeasurement, maximum150 minutes, never renew')
    a.models=['kimi','deepseek'];a.caps=[50,100];a.repeats=[1,2,3];configure()
    if a.phase=='audit':print(json.dumps(audit_point(a.output,a.manifest)));return
    native.identity.verify(os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources(),REPO)
    if a.phase=='suite':
        timing.storage_receipt(timing.STORAGE_RECEIPT.read_text());native.prepare_pipe_parent(PIPE)
        if ROOT.exists():raise FileExistsError('No automatic retry of this remeasurement')
    (base.suite if a.phase=='suite' else base.worker)(a)

if __name__=='__main__':main()
