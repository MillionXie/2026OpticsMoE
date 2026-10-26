"""Independent CPU-only same-card software A/B audit; never a formal A100 audit.

Both original roots, the original A100 numerical rejection, and rejected CUDA
clock evidence remain unchanged. This audit reports only original continuous
CPU wall, same-card exact scores and same-window board-only FFT diagnostics.
It never repairs a CUDA clock or emits an optical/A100 projection.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from audit_lgvq_async_point_20261026 import compare
from lgvq_async_worker_20261026 import validate_predictions,REFERENCE_RELATIVE,REFERENCE_SHA256
from run_lgvq_development_20261027 import GPU,SAME_CARD_SHA256
from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample,parse_telemetry_line
from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def receipt(root,name):
    value=read(root/name);seen=set()
    for row in value['verified_files']:
        member=Path(row['path']);p=root/member
        if (member.is_absolute() or member.drive or '..' in member.parts or row['path'] in seen
                or p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise ValueError('Original backup SHA/size/member differs')
        seen.add(row['path'])
    return dict(files=len(seen),bytes=sum(r['bytes'] for r in value['verified_files']),receipt_sha256=sha(root/name))


def scores(folder,expected):
    with np.load(folder/'outputs.npz',allow_pickle=False) as f:array=f['prediction']
    validate_predictions(array,expected)
    return array


def evidence(folder):
    identity=read(folder/'identity.json');actual=read(folder/'async_identity.json')
    if (identity['profile']!='RTX4090_DEVELOPMENT_ONLY' or identity['gpu']!=GPU
            or identity['expected_sms']!=128 or identity['observed_cap_w']!=450
            or identity['qualified_for_final_hybrid_table'] is not False
            or actual['gpu_uuid']!=GPU or actual['actual_visible_multiprocessors']!=128
            or actual['power_limit_w']!=450 or actual['video_calls']!=2048
            or actual['checkpoint_sha256']!='5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c'):
        raise ValueError('Development platform/asset identity differs')
    for name,digest in identity['sources'].items():
        data=subprocess.check_output(['git','show',identity['published_commit']+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Published source blob differs')
    if any(identity['sources'].get(k)!=v for k,v in actual['sources'].items()):
        raise ValueError('Actual worker source closure differs')
    audits=read(folder/'process_audit.json')
    if not audits or any(r['foreign_gpu_processes'] or r['foreign_mps_clients']
            or r['worker_pid']!=identity['pid'] or r['platform']['uuid']!=GPU
            or r['platform']['name']!='NVIDIA GeForce RTX 4090' or r['platform']['power_limit_w']!=450 for r in audits):
        raise ValueError('Exclusive process audit differs')
    if not any(identity['pid'] in r['clients'] for r in audits):raise ValueError('No bound MPS observation')
    files=read(folder/'input_files.json');preparation=read(folder/'external_preparation.json')
    if (len(files)!=512 or len({f['sample_id'] for f in files})!=512
            or len(preparation['rows'])!=32 or any(len(r['CPU_preparation'])!=16 for r in preparation['rows'])
            or preparation['neural_features_cached'] is not False):raise ValueError('Original input workload differs')
    events=read(folder/'async_event_records.json');rows=events['records']
    if len(rows)!=128 or [r['field'] for r in rows]!=list(range(128)):raise ValueError('Full ordered fields required')
    if [r['mode'] for r in rows[:5]]!=['eager_real']*3+['capture_real']*2:raise ValueError('Original first fields changed')
    for r in rows:
        kinds=[e['kind'] for e in r['events'] if e['stream']=='body']
        if kinds.count('optical_simulation')!=6 or kinds.count('detector_ROI')!=6 or kinds.count('electronic_residual')!=4:
            raise ValueError('Original six-pass math scope changed')
    if not math.isclose((events['host_end']-events['host_start'])*1000,events['measured_software_wall_ms'],abs_tol=1e-8):
        raise ValueError('Original continuous software wall differs')
    if rows[-1]['final_synchronization_observation']['end']!=events['host_end']:raise ValueError('Actual final score-ready return missing')
    with (folder/'telemetry.csv').open(encoding='utf-8-sig',newline='') as stream:
        samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),
            float(r['utilization_percent']),float(r['memory_used_mib'])) for r in csv.DictReader(stream)]
    raw_rows=[json.loads(s) for s in (folder/'telemetry_stdout.jsonl').read_text().splitlines()]
    if len(raw_rows)!=len(samples):raise ValueError('NVML rows dropped/inserted')
    previous=-math.inf
    for raw,sample in zip(raw_rows,samples):
        if (raw.get('valid_numeric_row') is not True or parse_telemetry_line(raw['raw_line'])!=
                (sample.watts,sample.utilization_percent,sample.memory_used_mib)
                or not previous<raw['host_monotonic_s']<=sample.host_monotonic_s):raise ValueError('Original NVML rows/clocks differ')
        previous=raw['host_monotonic_s']
    observer=read(folder/'telemetry_observer_identity.json')
    if (observer['gpu_uuid']!=GPU or observer['cuda_context_created'] is not False
            or observer['raw_file_IO_in_acquisition_reader'] is not False
            or observer['raw_file_written_after_stop'] is not True or '--loop-ms=50' not in observer['query']):
        raise ValueError('Original power observer differs')
    lo,hi=events['perf_monotonic_alignment_bracket'];offset=(lo+hi)/2-events['host_start']
    energy=interval_energy([(events['host_start']+offset,events['host_end']+offset)],samples,host_w=0.)
    if energy['max_power_sample_age_s']>.25:raise ValueError('Original power age gate failed')
    rawfile=folder/'async_raw_timing.json'
    if rawfile.exists():
        raw=read(rawfile);compare(energy,raw['raw_board_energy'])
        compare(dict(records=raw['records'],host_start=raw['host_start'],host_end=raw['host_end']),events)
        if raw['reference_sha256']!=SAME_CARD_SHA256 or raw['all_predictions_bitwise_equal'] is not True:
            raise ValueError('Same-card bitwise reference differs')
    seconds=events['measured_software_wall_ms']/1000
    return dict(folder=str(folder.resolve()),software_ms16=events['measured_software_wall_ms']/128,
        measured_software_wall_ms=events['measured_software_wall_ms'],board_FFT_diagnostic=energy,
        board_FFT_diagnostic_mean_w=energy['energy_j']/seconds,
        full_2048_scores_verified=True,four_cycles_verified=True,exclusive_observations=len(audits),
        raw_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()})


def audit(baseline,candidate,output):
    baseline=Path(baseline);candidate=Path(candidate);output=Path(output)
    if output.exists():raise FileExistsError('Preserve independent development audit')
    receipts=[receipt(baseline.parent,'backup_receipt_stopped01.json'),receipt(candidate.parent,'backup_receipt_complete01.json')]
    state=read(candidate.parent/'suite_status.json')
    if state['status']!='complete_development_only' or state['mps_released'] is not True:raise ValueError('Candidate not complete/released')
    if read(candidate/'status.json')['status']!='complete_development_only':raise ValueError('Candidate not complete')
    if read(baseline/'status.json')['status']!='failed':raise ValueError('Do not relabel original baseline rejection')
    if sha(baseline/'outputs.npz')!=SAME_CARD_SHA256:raise ValueError('Exact baseline reference changed')
    with np.load(baseline/'outputs.npz',allow_pickle=False) as f:reference=f['prediction']
    scores(baseline,reference);scores(candidate,reference)
    if read(baseline/'input_files.json')!=read(candidate/'input_files.json'):raise ValueError('Input video SHA/size/order differs')
    a=evidence(baseline);b=evidence(candidate);report=read(candidate/'cost_report.json')
    if report['qualified_for_final_hybrid_table'] is not False or report['development_only'] is not True:
        raise ValueError('Different-hardware result cannot qualify formal table')
    method=read(candidate/'identity.json')['performance_change']
    pool=read(candidate/'async_event_records.json')['records']
    for row in pool[3:5]:
        graphs=row['graph_construction']
        if len(graphs)!=38:raise ValueError('All original capture segments required')
        if method=='same_slot_same_stream_graph_allocator_pool':
            if any(g['independent_default_pool'] is not False
                    or g['development_pool_scope']!='same_real_slot_same_stream_only' for g in graphs):
                raise ValueError('Actual slot/stream pool evidence incomplete')
            if graphs[-1]['development_stream_pool_count']!=2:raise ValueError('Exactly separate frontend/body pools required')
        elif method in ('CPU_intraop_threads_1','constant_prompt_TEXT_render_once_inside_window','Conv3d_channels_last_3d_storage_only'):
            identity=read(candidate/'identity.json')
            threads=1 if method=='CPU_intraop_threads_1' else 8
            if identity['CPU_intraop_threads']!=threads or any(g['independent_default_pool'] is not True for g in graphs):
                raise ValueError('Isolated CPU-thread candidate required')
        else:raise ValueError('Unknown development candidate')
    if read(candidate/'async_raw_timing.json')['event_reuse']['enabled'] is not False:raise ValueError('One-change graph pool candidate required')
    if method=='constant_prompt_TEXT_render_once_inside_window':
        info=read(candidate/'prompt_render_identity.json');first=pool[0]['tokenizer'];a0,a1=info['first_render_interval']
        if (info['requests']!=128 or info['renders']!=1 or not first['start']<=a0<=a1<=first['end']
                or info['token_tensors_cached'] or info['features_cached'] or info['scores_cached']):
            raise ValueError('Only one timed TEXT render; all original tokenizer work retained')
    if method=='Conv3d_channels_last_3d_storage_only':
        info=read(candidate/'conv_layout_identity.json')
        if (info['weight_dtype']!='torch.bfloat16' or info['bias_dtype']!='torch.bfloat16'
                or info['original_convolution_and_bias_retained'] is not True
                or info['weight_values_or_precision_changed'] is not False
                or info['input_layout_conversion_inside_actual_forward'] is not True or info['extra_forward_calls']!=0):
            raise ValueError('Original biased BF16 Conv3d layout-only identity required')
    if report.get('causal_DAG_accepted') is False:
        if report.get('projection_computed') is not False or report.get('clock_values_modified') is not False:
            raise ValueError('Rejected clock cannot produce a repaired projection')
    result=dict(status='passed_SOFTWARE_DEVELOPMENT_ONLY',qualified_for_final_hybrid_table=False,
        auditor_sha256=sha(__file__),candidate_method=method,
        baseline=a,candidate=b,backup_receipts=receipts,
        software_gain_ms16=a['software_ms16']-b['software_ms16'],
        software_reduction_percent=(1-b['software_ms16']/a['software_ms16'])*100,
        single_round_comparison_not_statistical_significance=True,
        A100_speedup_demonstrated=False,A100_reference_unchanged=True,
        optical_projection_available=report.get('projection_computed') is not False,
        no_clock_repair=True,original_failed_roots_unchanged=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();r=audit(args.baseline,args.candidate,args.output)
    print(json.dumps({k:r[k] for k in ['status','software_gain_ms16','software_reduction_percent','optical_projection_available']},ensure_ascii=True))
