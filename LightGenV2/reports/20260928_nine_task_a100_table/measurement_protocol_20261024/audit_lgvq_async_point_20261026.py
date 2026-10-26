"""Independent local data replay of one backed LG async leaf; no model/CUDA.

Verify the data-only receipt, original observer numeric rows, complete bytewise
scores/reference, event/host clocks, witnessed DAG, same-window board diagnostic
and fixed-component-union proxy. Never rewrite original raw/report/reference.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from lgvq_async_raw_adapter_20261026 import adapt
from lgvq_async_worker_20261026 import GPU, REFERENCE_RELATIVE, REFERENCE_SHA256, REPO, validate_predictions

CHECKPOINT_SHA256='5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c'
OBSERVER_RELATIVE='LightGenV2/tasks/t06_video_quality_assessment/e2e_telemetry_observer.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(expected,actual,label='report'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or not set(expected)<=set(actual):raise ValueError('Missing keys: '+label)
        for key,value in expected.items():compare(value,actual[key],label+'.'+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):raise ValueError('List differs: '+label)
        for i,(a,b) in enumerate(zip(expected,actual)):compare(a,b,label+f'[{i}]')
    elif isinstance(expected,bool) or expected is None:
        if actual is not expected:raise ValueError('Flag differs: '+label)
    elif isinstance(expected,(int,float)):
        if (isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(expected) or not math.isfinite(actual)
                or not math.isclose(expected,actual,rel_tol=1e-10,abs_tol=1e-6)):
            raise ValueError('Numeric replay differs: '+label)
    elif expected!=actual:raise ValueError('Value differs: '+label)


def verify_receipt(folder,receipt):
    records=receipt['verified_files'];seen=set();total=0
    for row in records:
        member=Path(row['path'])
        if member.is_absolute() or member.drive or '..' in member.parts or row['path'] in seen:
            raise ValueError('Invalid or duplicate original receipt member')
        seen.add(row['path']);path=folder/member
        if (path.is_symlink() or not path.resolve().is_relative_to(folder.resolve())
                or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):
            raise ValueError('Backed original SHA/size differs: '+row['path'])
        total+=row['bytes']
    required={'status.json','identity.json','async_identity.json','external_preparation.json','input_files.json',
              'telemetry.csv','telemetry_stdout.jsonl','telemetry_observer_identity.json','outputs.npz',
              'async_event_records.json','async_raw_timing.json','async_causal_trace.json','cost_report.json','process_audit.json'}
    if not required<=seen:raise ValueError('Incomplete original LG data receipt')
    return dict(files=len(records),bytes=total)


def verify_git_sources(identity):
    commit=identity['published_commit']
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact published source identity required')
    checked=set()
    for path,digest in identity['sources'].items():
        if Path(path).name.startswith('lgvq_async_') or Path(path).name=='run_lgvq_async_20261026.py':
            data=subprocess.check_output(['git','show',commit+':'+path],cwd=REPO,timeout=10)
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Original published source blob differs')
            checked.add(Path(path).name)
    required={'lgvq_async_worker_20261026.py','lgvq_async_capture_20261026.py','lgvq_async_replay_20261026.py',
              'lgvq_async_trace_contract_20261026.py','lgvq_async_component_energy_20261026.py',
              'lgvq_async_observed_dag_20261026.py','lgvq_async_raw_adapter_20261026.py','run_lgvq_async_20261026.py'}
    if not required<=checked:raise ValueError('Original candidate/guard source closure is incomplete')
    return len(checked)


def audit(folder,output,reference=None,*,verify_git=True):
    folder=Path(folder).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing audit')
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8-sig'))
    receipt=verify_receipt(folder,read('backup_receipt_complete01.json'))
    identity=read('identity.json');async_identity=read('async_identity.json')
    if (read('status.json')['status']!='complete' or identity['GPU']!=GPU or identity['cap']!=100
            or identity['video_calls']!=2048 or async_identity['gpu_uuid']!=GPU
            or async_identity['actual_visible_multiprocessors']!=108 or async_identity['power_limit_w']!=250
            or async_identity['checkpoint_sha256']!=CHECKPOINT_SHA256
            or async_identity['reference_sha256']!=REFERENCE_SHA256):
        raise ValueError('Original complete point/platform/checkpoint/reference identity differs')
    if async_identity['sources']!={k:v for k,v in identity['sources'].items() if k in async_identity['sources']}:
        raise ValueError('Outer guard and actual worker source closures disagree')
    for name in ('guard_rejection_snapshot.json','async_partial_host_records.json'):
        if (folder/name).exists():raise ValueError('A failed/rejected leaf cannot be relabelled complete')
    snapshots=read('process_audit.json')
    if not snapshots:raise ValueError('Original exclusive guard evidence missing')
    for row in snapshots:
        platform=row['platform']
        if (row.get('human_authorized_nonexclusive') is not False or row['foreign_gpu_processes']
                or row['foreign_mps_clients'] or row['worker_pid']!=identity['pid']
                or platform['uuid']!=GPU or platform['name']!='NVIDIA A100-PCIE-40GB'
                or platform['power_limit_w']!=250 or platform['requested_cap']!=100):
            raise ValueError('Original exclusive GPU guard evidence differs')
    if not any(identity['pid'] in row['clients'] for row in snapshots):
        raise ValueError('No original bound private MPS client evidence')
    original_source_count=verify_git_sources(identity) if verify_git else 0
    reference=Path(reference or REPO/REFERENCE_RELATIVE)
    if sha(reference)!=REFERENCE_SHA256:raise ValueError('Sealed local reference SHA differs')
    with np.load(folder/'outputs.npz',allow_pickle=False) as f:array=f['prediction']
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction']
    validate_predictions(array,expected)
    files=read('input_files.json');preparation=read('external_preparation.json')
    if (len(files)!=512 or len({r['sample_id'] for r in files})!=512
            or len(preparation['rows'])!=32 or [r['group'] for r in preparation['rows']]!=list(range(32))
            or any(len(r['CPU_preparation'])!=16 for r in preparation['rows'])
            or preparation['neural_features_cached'] is not False or preparation['raw_inputs_GPU_resident'] is not True
            or preparation['excluded_from_GPU_window'] is not True):
        raise ValueError('Original 512-video / 32-field input preparation differs')
    if any(not isinstance(r.get('bytes'),int) or r['bytes']<=0 or len(r.get('sha256',''))!=64
           or any(c not in '0123456789abcdef' for c in r['sha256']) for r in files):
        raise ValueError('Original input file size/SHA evidence missing')
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample,parse_telemetry_line
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    with (folder/'telemetry.csv').open(encoding='utf-8-sig',newline='') as stream:
        samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),
                        float(r['utilization_percent']),float(r['memory_used_mib'])) for r in csv.DictReader(stream)]
    raw_rows=[json.loads(line) for line in (folder/'telemetry_stdout.jsonl').read_text().splitlines()]
    if len(raw_rows)!=len(samples):raise ValueError('Power samples/rows dropped or inserted')
    previous_raw=-math.inf
    for raw,sample in zip(raw_rows,samples):
        if raw.get('valid_numeric_row') is not True:raise ValueError('Original invalid NVML row cannot be discarded')
        values=parse_telemetry_line(raw['raw_line'])
        if values!=(sample.watts,sample.utilization_percent,sample.memory_used_mib):
            raise ValueError('Original NVML numeric row differs from saved CSV')
        if not math.isfinite(sample.host_monotonic_s+sample.host_time_unix):raise ValueError('Invalid receipt clock')
        tick=raw['host_monotonic_s']
        if not math.isfinite(tick) or not previous_raw<tick<=sample.host_monotonic_s:
            raise ValueError('Original observer receipt clocks retimed or out of order')
        previous_raw=tick
    observer=read('telemetry_observer_identity.json')
    if (observer['gpu_uuid']!=GPU or observer['cuda_context_created'] is not False
            or observer['version']!='independent_process_memory_raw_v1'
            or observer['raw_file_IO_in_acquisition_reader'] is not False or observer['raw_file_written_after_stop'] is not True
            or '--loop-ms=50' not in observer['query'] or '--id='+GPU not in observer['query']
            or observer['observer_source_sha256']!=identity['sources'][OBSERVER_RELATIVE]):
        raise ValueError('Original independent 50-ms power receiver identity differs')
    timing=read('async_raw_timing.json');events=read('async_event_records.json')
    compare(dict(records=timing['records'],host_start=timing['host_start'],host_end=timing['host_end'],
                 measured_software_wall_ms=timing['actual_async_software_wall_ms'],
                 origin_host_clock_bracket=timing['origin_host_clock_bracket']),events,'original_events')
    before,after=timing['perf_monotonic_alignment_bracket']
    if not before<=after:raise ValueError('Original clock alignment bracket differs')
    offset=(before+after)/2-timing['host_start']
    energy=interval_energy([(timing['host_start']+offset,timing['host_end']+offset)],samples,host_w=0.)
    compare(energy,timing['raw_board_energy'],'original_board_only_FFT_diagnostic')
    trace,replayed=adapt(timing)
    compare(trace,read('async_causal_trace.json'),'causal_trace')
    report=read('cost_report.json');compare(replayed,report)
    for key,name in [('raw_timing_sha256','async_raw_timing.json'),('event_records_sha256','async_event_records.json'),
                     ('outputs_sha256','outputs.npz'),('telemetry_sha256','telemetry.csv')]:
        if report[key]!=sha(folder/name):raise ValueError('Original report file hash differs: '+name)
    result=dict(status='passed',video_calls=2048,fields=128,unique_videos=512,cycles=4,
        all_2048_scores_bitwise_equal_sealed_reference=True,four_cycles_bitwise_equal=True,
        every_original_DAG_node_start_reproduced=True,actual_async_software_wall_ms=replayed['actual_async_software_wall_ms'],
        projected_time16_ms=replayed['time16_ms'],projected_energy16_j=replayed['energy16_j'],
        projected_common_tokens_per_s=replayed['common_tokens_per_s'],projected_native_tokens_per_s=replayed['native_tokens_per_s'],
        actual_board_FFT_diagnostic_j=energy['energy_j'],max_power_sample_age_s=energy['max_power_sample_age_s'],
        controller_and_board_not_double_counted=True,physical_hardware_time_or_energy_measured=False,
        kernel_occupancy_concurrency_proven=False,original_published_source_blobs_verified=original_source_count,
        native_sources_and_assets_frozen_by_worker_not_model_reexecuted_locally=True,backup_receipt=receipt,
        raw_sha256={name:sha(folder/name) for name in ('async_raw_timing.json','async_event_records.json',
            'async_causal_trace.json','outputs.npz','telemetry.csv','cost_report.json')})
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.output or args.folder/'local_independent_async_audit01.json',args.reference),ensure_ascii=False))
